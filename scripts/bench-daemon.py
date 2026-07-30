#!/usr/bin/env python3
"""Leak-safe local benchmark for one-shot and daemon-routed NSys queries.

The report deliberately omits the input path and command output. "Cold"
means a newly started daemon with no resident session; it does not mean a
cold operating-system page cache. Daemon lifecycle time is excluded from
query measurements. Interval workloads use distinct windows for cache misses
and report exact repeats separately. Supplying a second trace also measures
rebuild after deterministic max-sessions=1 eviction.

Interval states require an already prepared, fresh gpu-work-events sidecar.
The harness first records one distinct scan miss as reuse evidence, then
observes source-memory accounting around the next varying miss so a fallback
execution cannot be mislabeled as resident-index construction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


DEFAULT_SAMPLES = 5
INTERVAL_WINDOW_PARTS = 4
TIMELINE_BUCKETS_PER_WINDOW = 8
GAP_THRESHOLD_PARTS = 20
EMPTY_METRIC = ""
INTERVAL_SIDECAR_KEY = "sidecar|gpu-work-events"


class BenchmarkFailure(RuntimeError):
    pass


def run(
    executable: str,
    arguments: Sequence[str],
    *,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        [executable, *arguments],
        check=False,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            "a benchmark command failed; rerun the workload manually for diagnostics"
        )
    return completed


def run_json(executable: str, arguments: Sequence[str]) -> dict:
    completed = run(executable, arguments, capture_stdout=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkFailure(
            "a benchmark setup command did not return its JSON contract"
        ) from error
    if not isinstance(payload, dict):
        raise BenchmarkFailure("a benchmark setup command returned an invalid JSON contract")
    return payload


def measure(executable: str, arguments: Sequence[str]) -> int:
    started = time.monotonic_ns()
    run(executable, arguments)
    return time.monotonic_ns() - started


def daemon_state(executable: str) -> str:
    payload = run_json(executable, ["daemon", "status"])
    try:
        state = payload["data"]["rows"][0]["state"]
    except (KeyError, IndexError, TypeError) as error:
        raise BenchmarkFailure("daemon status returned an invalid lifecycle contract") from error
    if not isinstance(state, str):
        raise BenchmarkFailure("daemon status returned an invalid state")
    return state


def discover_drill_row(executable: str, trace: Path) -> str:
    payload = run_json(
        executable,
        [
            "search",
            str(trace),
            "--type",
            "all",
            "--all-devices",
            "--limit",
            "1",
            "--daemon",
            "off",
        ],
    )
    try:
        row_id = payload["data"]["rows"][0]["row_id"]
    except (KeyError, IndexError, TypeError) as error:
        raise BenchmarkFailure(
            "the input has no searchable row for the drill benchmark"
        ) from error
    if not isinstance(row_id, str) or not row_id:
        raise BenchmarkFailure("the discovered drill row id is invalid")
    return row_id


def percentile(sorted_values: Sequence[int], fraction: float) -> int:
    index = max(0, math.ceil(len(sorted_values) * fraction) - 1)
    return sorted_values[index]


def report(
    workload: str,
    state: str,
    samples: Sequence[int],
    metrics: dict[str, int] | None = None,
) -> None:
    ordered = sorted(samples)
    scale = 1_000_000
    metrics = metrics or {}
    values = [
        workload,
        state,
        str(len(samples)),
        f"{ordered[0] / scale:.3f}",
        f"{statistics.median(ordered) / scale:.3f}",
        f"{percentile(ordered, 0.95) / scale:.3f}",
        f"{ordered[-1] / scale:.3f}",
        str(metrics.get("source_memory_bytes", EMPTY_METRIC)),
        str(metrics.get("resident_memory_bytes", EMPTY_METRIC)),
        str(metrics.get("exact_response_entries", EMPTY_METRIC)),
        str(metrics.get("cache_hits", EMPTY_METRIC)),
        str(metrics.get("cache_misses", EMPTY_METRIC)),
    ]
    print(",".join(values))


def benchmark_workload(
    executable: str,
    arguments: Sequence[str],
    samples: int,
) -> tuple[list[int], list[int], list[int]]:
    one_shot = [
        measure(executable, [*arguments, "--daemon", "off"]) for _ in range(samples)
    ]

    daemon_cold = []
    for _ in range(samples):
        run(executable, ["daemon", "start"])
        try:
            daemon_cold.append(
                measure(executable, [*arguments, "--daemon", "required"])
            )
        finally:
            run(executable, ["daemon", "stop"])

    run(executable, ["daemon", "start"])
    try:
        run(executable, [*arguments, "--daemon", "required"])
        daemon_warm = [
            measure(executable, [*arguments, "--daemon", "required"])
            for _ in range(samples)
        ]
    finally:
        run(executable, ["daemon", "stop"])

    return one_shot, daemon_cold, daemon_warm


def trace_span(executable: str, trace: Path) -> tuple[int, int]:
    payload = run_json(
        executable,
        ["summary", str(trace), "--daemon", "off"],
    )
    try:
        origin_ns = payload["trace_span"]["origin_ns"]
        span_ns = payload["trace_span"]["span_ns"]
    except (KeyError, TypeError) as error:
        raise BenchmarkFailure(
            "the benchmark input does not expose an NSys primary trace span"
        ) from error
    if not isinstance(origin_ns, int) or not isinstance(span_ns, int) or span_ns <= 0:
        raise BenchmarkFailure("the benchmark input has an invalid primary trace span")
    return origin_ns, span_ns


def require_fresh_interval_sidecar(executable: str, trace: Path) -> None:
    payload = run_json(
        executable,
        ["prep", "--status", str(trace), "--daemon", "off"],
    )
    try:
        rows = payload["data"]["rows"]
        sidecar = next(row for row in rows if row.get("key") == INTERVAL_SIDECAR_KEY)
        ready = (
            sidecar["present"] is True
            and sidecar["fingerprint_match"] is True
            and sidecar["format_version_on_disk"]
            == sidecar["format_version_expected"]
        )
    except (KeyError, StopIteration, TypeError) as error:
        raise BenchmarkFailure(
            "prep --status did not expose gpu-work-events readiness"
        ) from error
    if not ready:
        raise BenchmarkFailure(
            "interval benchmarks require a fresh gpu-work-events sidecar; "
            "run `veloq prep TRACE` first"
        )


def interval_workloads(
    trace: Path,
    origin_ns: int,
    span_ns: int,
    samples: int,
) -> dict[str, list[list[str]]]:
    # One distinct miss establishes reuse interest, the next constructs the
    # index, and `samples` further windows measure warm cache misses.
    variant_count = samples + 2
    window_ns = max(1, span_ns // INTERVAL_WINDOW_PARTS)
    offset_range = span_ns - window_ns
    if offset_range < variant_count - 1:
        raise BenchmarkFailure(
            "the benchmark trace span is too short to generate distinct cache-miss windows"
        )

    windows = []
    denominator = variant_count - 1
    for index in range(variant_count):
        start_ns = origin_ns + (offset_range * index // denominator)
        end_ns = start_ns + window_ns
        windows.append((f"@{start_ns}ns", f"@{end_ns}ns"))

    timeline_interval_ns = max(1, window_ns // TIMELINE_BUCKETS_PER_WINDOW)
    gap_threshold_ns = max(1, window_ns // GAP_THRESHOLD_PARTS)
    workloads: dict[str, list[list[str]]] = {
        "timeline-cache-miss": [],
        "concurrency-cache-miss": [],
        "gaps-cache-miss": [],
    }
    for start, end in windows:
        common = [
            str(trace),
            "--from",
            start,
            "--to",
            end,
            "--all-devices",
        ]
        workloads["timeline-cache-miss"].append(
            [
                "timeline",
                *common,
                "--type",
                "all",
                "--interval",
                f"{timeline_interval_ns}ns",
            ]
        )
        workloads["concurrency-cache-miss"].append(["concurrency", *common])
        workloads["gaps-cache-miss"].append(
            [
                "gaps",
                *common,
                "--min-duration",
                f"{gap_threshold_ns}ns",
            ]
        )
    return workloads


def daemon_metrics(executable: str) -> dict[str, int]:
    payload = run_json(executable, ["daemon", "status"])
    try:
        usage = payload["data"]["rows"][0]["usage"]
        sessions = payload["data"]["auxiliary"]["sessions"]
        session = sessions[0]
        source_memory = (
            session["resident_memory_estimate_bytes"]
            - session["exact_response_bytes_estimate"]
        )
        metrics = {
            "source_memory_bytes": source_memory,
            "resident_memory_bytes": usage["resident_memory_estimate_bytes"],
            "exact_response_entries": usage["exact_response_entries"],
            "cache_hits": usage["cache_hits"],
            "cache_misses": usage["cache_misses"],
        }
    except (KeyError, IndexError, TypeError) as error:
        raise BenchmarkFailure(
            "daemon status did not expose interval benchmark accounting"
        ) from error
    if any(not isinstance(value, int) or value < 0 for value in metrics.values()):
        raise BenchmarkFailure("daemon status returned invalid benchmark accounting")
    return metrics


def require_resident_interval_index(
    before: dict[str, int],
    after: dict[str, int],
) -> None:
    if after["source_memory_bytes"] <= before["source_memory_bytes"]:
        raise BenchmarkFailure(
            "the interval query did not register accounted resident state; "
            "refusing to label fallback execution as resident-index construction"
        )


def benchmark_interval_workload(
    executable: str,
    trace: Path,
    variants: Sequence[Sequence[str]],
    samples: int,
    eviction_trace: Path | None,
) -> dict[str, tuple[list[int], dict[str, int] | None]]:
    interest_variant = variants[0]
    construction_variant = variants[1]
    measured_variants = variants[2 : samples + 2]
    one_shot = [
        measure(executable, [*arguments, "--daemon", "off"])
        for arguments in measured_variants
    ]

    construction = []
    construction_metrics = None
    for arguments in measured_variants:
        run(executable, ["daemon", "start"])
        try:
            run(executable, ["summary", str(trace), "--daemon", "required"])
            run(executable, [*interest_variant, "--daemon", "required"])
            before = daemon_metrics(executable)
            construction.append(
                measure(executable, [*arguments, "--daemon", "required"])
            )
            construction_metrics = daemon_metrics(executable)
            require_resident_interval_index(before, construction_metrics)
        finally:
            run(executable, ["daemon", "stop"])

    run(executable, ["daemon", "start"])
    try:
        run(executable, ["summary", str(trace), "--daemon", "required"])
        run(executable, [*interest_variant, "--daemon", "required"])
        before = daemon_metrics(executable)
        measure(executable, [*construction_variant, "--daemon", "required"])
        after_construction = daemon_metrics(executable)
        require_resident_interval_index(before, after_construction)
        warm_miss = [
            measure(executable, [*arguments, "--daemon", "required"])
            for arguments in measured_variants
        ]
        warm_metrics = daemon_metrics(executable)
        exact_hit = [
            measure(executable, [*measured_variants[-1], "--daemon", "required"])
            for _ in range(samples)
        ]
        exact_metrics = daemon_metrics(executable)
    finally:
        run(executable, ["daemon", "stop"])

    results = {
        "one-shot": (one_shot, None),
        "daemon-construct": (construction, construction_metrics),
        "daemon-warm-miss": (warm_miss, warm_metrics),
        "daemon-exact-hit": (exact_hit, exact_metrics),
    }
    if eviction_trace is None:
        return results

    eviction_rebuild = []
    eviction_metrics = None
    run(executable, ["daemon", "start", "--max-sessions", "1"])
    try:
        run(executable, [*interest_variant, "--daemon", "required"])
        run(executable, [*construction_variant, "--daemon", "required"])
        for arguments in measured_variants:
            run(
                executable,
                ["summary", str(eviction_trace), "--daemon", "required"],
            )
            run(executable, [*interest_variant, "--daemon", "required"])
            eviction_rebuild.append(
                measure(executable, [*arguments, "--daemon", "required"])
            )
        eviction_metrics = daemon_metrics(executable)
    finally:
        run(executable, ["daemon", "stop"])
    results["daemon-eviction-rebuild"] = (eviction_rebuild, eviction_metrics)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-shot, resident construction, varying-argument cache "
            "misses, exact hits, and optional eviction rebuild latency without "
            "retaining trace identity or command output."
        )
    )
    parser.add_argument("trace", type=Path, help="local NSys report or parquetdir")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"samples per workload/state (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--veloq",
        default="veloq",
        help="VeloQ executable to benchmark (default: veloq from PATH)",
    )
    parser.add_argument(
        "--eviction-trace",
        type=Path,
        help=(
            "second local NSys input used to force max-sessions=1 eviction and "
            "measure interval-view rebuilds"
        ),
    )
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    if options.samples < 1:
        raise BenchmarkFailure("--samples must be at least 1")
    if not options.trace.exists():
        raise BenchmarkFailure("the benchmark input does not exist")
    if options.eviction_trace is not None:
        if not options.eviction_trace.exists():
            raise BenchmarkFailure("the eviction benchmark input does not exist")
        if options.eviction_trace.resolve() == options.trace.resolve():
            raise BenchmarkFailure("--eviction-trace must identify a different input")
    if daemon_state(options.veloq) != "stopped":
        raise BenchmarkFailure(
            "a daemon is already present; stop it before running this isolated benchmark"
        )

    row_id = discover_drill_row(options.veloq, options.trace)
    require_fresh_interval_sidecar(options.veloq, options.trace)
    workloads = {
        "drill": ["inspect", str(options.trace), row_id],
        "scan": [
            "stats",
            str(options.trace),
            "--type",
            "all",
            "--all-devices",
        ],
    }
    origin_ns, span_ns = trace_span(options.veloq, options.trace)
    interval_variants = interval_workloads(
        options.trace,
        origin_ns,
        span_ns,
        options.samples,
    )

    print(
        "workload,state,samples,min_ms,median_ms,p95_ms,max_ms,"
        "source_memory_bytes,resident_memory_bytes,exact_response_entries,"
        "cache_hits,cache_misses"
    )
    try:
        for workload, arguments in workloads.items():
            measurements = benchmark_workload(
                options.veloq, arguments, options.samples
            )
            for state, samples in zip(
                ("one-shot", "daemon-cold", "daemon-warm-exact"),
                measurements,
                strict=True,
            ):
                report(workload, state, samples)
        for workload, variants in interval_variants.items():
            measurements = benchmark_interval_workload(
                options.veloq,
                options.trace,
                variants,
                options.samples,
                options.eviction_trace,
            )
            for state, (samples, metrics) in measurements.items():
                report(workload, state, samples, metrics)
    finally:
        if daemon_state(options.veloq) != "stopped":
            run(options.veloq, ["daemon", "stop"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkFailure as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
