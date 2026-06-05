#!/usr/bin/env python3
"""veloq NCU export helper — ncu_report-native sidecar.

Drives NVIDIA's official `ncu_report` Python API (shipped in the NCU
install under `extras/python/`) to emit veloq's ncu_report-native NCU
sidecar as JSON on stdout. Uses ONLY the public API — no vendored
protos, no NVIDIA files redistributed.

Usage:
    python3 ncu_export.py <report.ncu-rep>          # JSON sidecar -> stdout
    python3 ncu_export.py <report.ncu-rep> --probe   # capability probe only

The Rust ingest path is authoritative: it sets PYTHONPATH to the located
`ncu_report` module directory before invoking this. We also self-locate
as a fallback so the helper runs standalone for fixture regeneration; the
fallback mirrors the Rust-side cross-platform discovery so the
two paths cannot diverge.

Tested against Nsight Compute 2026.1.1 (ncu_report API). Other versions
are expected to work: the helper resolves all version-specific enum
semantics (metric type/subtype/rollup, stall reasons) from the *live*
ncu_report enum by name rather than hard-coding integer codes,
so an enum renumber cannot corrupt output. A
structural API change (a renamed/relocated enum container) collapses the
reverse maps to empty, which is reported as a `classification: "degraded"`
marker plus a stderr warning rather than emitting wrong names.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# --- Locate + import ncu_report (cross-platform discovery) ---------
# Mirrors the Rust ingest path's discovery (crates/.../native/cache.rs):
# VELOQ_NCU_REPORT_DIR override, then per-platform NCU install roots. The
# Linux pattern follows mit-han-lab/ncu-report-skill and Enigmatisms/tachyon.
def _locate_ncu_report() -> str | None:
    override = os.environ.get("VELOQ_NCU_REPORT_DIR")
    if override and (Path(override) / "ncu_report.py").is_file():
        return override
    if sys.platform == "darwin":
        roots = ["/Applications"]
        globs = [
            "NVIDIA Nsight Compute*.app/Contents/MacOS/python",
            "NVIDIA Nsight Compute*/extras/python",
        ]
    elif sys.platform == "win32":
        roots = [
            os.environ.get("ProgramW6432", r"C:\Program Files"),
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        globs = ["NVIDIA Corporation/Nsight Compute */extras/python"]
    else:
        roots = ["/usr/local", "/opt/nvidia", "/opt/cuda"]
        globs = [
            "cuda-*/nsight-compute-*/extras/python",
            "nsight-compute-*/extras/python",
            "nsight-compute/*/extras/python",
            "nsight-compute/extras/python",
        ]
    for root in roots:
        p = Path(root)
        if not p.is_dir():
            continue
        for g in globs:
            for sub in sorted(p.glob(g), reverse=True):  # newest-ish first
                if (sub / "ncu_report.py").is_file():
                    return str(sub)
    return None


try:
    import ncu_report  # noqa: F401
except ImportError:
    _found = _locate_ncu_report()
    if _found:
        sys.path.insert(0, _found)
    try:
        import ncu_report  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "error: ncu_report Python module not importable. Set PYTHONPATH to "
            "<ncu-install>/extras/python, or install Nsight Compute.\n"
        )
        sys.exit(3)


# --- Value typing (ncu_report ValueKind) -------------------------------------
_KIND = {
    ncu_report.IMetric.ValueKind_UINT64: "uint64",
    ncu_report.IMetric.ValueKind_UINT32: "uint32",
    ncu_report.IMetric.ValueKind_DOUBLE: "double",
    ncu_report.IMetric.ValueKind_FLOAT: "float",
    ncu_report.IMetric.ValueKind_STRING: "string",
}


def _kind_name(k: int) -> str:
    return _KIND.get(k, "unknown")


# --- Enum name resolution --------------------
# Resolve metric_type / metric_subtype / rollup_operation to stable *name*
# strings from the LIVE ncu_report enum, so the Rust additivity classifier
# matches names (version-stable) instead of integer codes (version-specific).
# Same dir()-scan idiom as `_STALL_NAMES` below. The enum constants live on
# `ncu_report.IMetric` (where `ValueKind_*` lives); we also scan the module
# top-level as a fallback container.
def _build_name_map(prefixes: tuple[str, ...]) -> dict[int, str]:
    """value(int) -> lowercase short name, scanned from the live enum.

    An empty result means no attribute matched any prefix — the signature
    of a renamed/relocated enum container (a structural API change), which
    the caller surfaces as a `degraded` marker rather than guessing."""
    out: dict[int, str] = {}
    for container in (getattr(ncu_report, "IMetric", None), ncu_report):
        if container is None:
            continue
        for n in dir(container):
            for pre in prefixes:
                if n.startswith(pre):
                    try:
                        v = getattr(container, n)
                    except Exception:
                        continue
                    if isinstance(v, int):
                        out[v] = n[len(pre):].lower()
    return out


_METRIC_TYPE_NAMES = _build_name_map(("MetricType_",))
_METRIC_SUBTYPE_NAMES = _build_name_map(("MetricSubtype_",))
_ROLLUP_NAMES = _build_name_map(("RollupOperation_", "Rollup_"))
# Healthy iff we resolved the two maps the classifier actually gates on
# (metric_type and rollup). Subtype is advisory (only pct/ratio/per_second
# matter, and those route through metric_type if unresolved).
_ENUM_MAPS_OK = bool(_METRIC_TYPE_NAMES) and bool(_ROLLUP_NAMES)
if not _ENUM_MAPS_OK:
    sys.stderr.write(
        "warning: ncu_report metric_type/rollup enum reverse-map is empty "
        "(enum container renamed/relocated?). Emitting a degraded sidecar; "
        "additivity classification falls back to the name-suffix rule.\n"
    )


def _enum_name(name_map: dict[int, str], code):
    """(name, code) for an enum value, or (None, None) when the API
    returned no value. Unresolved codes -> 'unknown' (Rust maps that to the
    enum's Unknown arm -> name-suffix fallback), never silently wrong."""
    if code is None:
        return None, None
    c = int(code)
    return name_map.get(c, "unknown"), c


def _agg_value(m):
    """Aggregate metric value, typed by kind."""
    k = m.kind()
    if k == ncu_report.IMetric.ValueKind_UINT64:
        return m.as_uint64()
    if k in (ncu_report.IMetric.ValueKind_DOUBLE, ncu_report.IMetric.ValueKind_FLOAT):
        return m.as_double()
    if k == ncu_report.IMetric.ValueKind_STRING:
        return m.as_string()
    try:
        return m.as_double()
    except Exception:
        return None


def _instance_value(m, i: int):
    k = m.kind()
    if k == ncu_report.IMetric.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (ncu_report.IMetric.ValueKind_DOUBLE, ncu_report.IMetric.ValueKind_FLOAT):
        return m.as_double(i)
    if k == ncu_report.IMetric.ValueKind_STRING:
        return m.as_string(i)
    try:
        return m.as_double(i)
    except Exception:
        return None


# --- Placement classifier -----------------
# attributed:        source_info(cid) resolves to a (file, line)
# out_of_cubin:      sass_by_pc(cid) == "" (cid is not a PC in this cubin)
# in_cubin_no_source: a real PC (opcode or mid-instruction "N/A") with no line
def _placement(action, cid: int) -> str:
    if action.source_info(cid) is not None:
        return "attributed"
    if action.sass_by_pc(cid) == "":
        return "out_of_cubin"
    return "in_cubin_no_source"


def _source_ref(action, cid: int):
    si = action.source_info(cid)
    if si is None:
        return None
    return {"file": si.file_name(), "line": si.line()}


# --- Per-launch extraction ---------------------------------------------------
def _metric_entry(action, name: str, base):
    m = action.metric_by_name(name)
    if not m.has_value():
        return None
    mt_name, mt_code = _enum_name(_METRIC_TYPE_NAMES, m.metric_type())
    st_name, st_code = _enum_name(_METRIC_SUBTYPE_NAMES, m.metric_subtype())
    ro_name, ro_code = _enum_name(_ROLLUP_NAMES, m.rollup_operation())
    entry = {
        "name": name,
        "label": m.description() or None,
        "unit": m.unit() or None,
        "value": _agg_value(m),
        "value_type": _kind_name(m.kind()),
        # Enum *names* (version-stable), with the raw integer kept as
        # provenance so a renumber is detectable.
        "metric_type": mt_name if mt_name is not None else "unknown",
        "metric_type_code": mt_code,
        "metric_subtype": st_name,
        "metric_subtype_code": st_code,
        "rollup": ro_name,
        "rollup_code": ro_code,
    }
    # Per-PC instances only when the metric carries correlation IDs.
    if m.has_correlation_ids() and m.num_instances() > 0:
        cids = m.correlation_ids()
        insts = []
        for i in range(m.num_instances()):
            cid = cids.as_uint64(i)
            rel = (cid - base) if (base is not None and cid >= base) else None
            insts.append(
                {
                    "correlation_id": cid,
                    "rel_address": rel,
                    "value": _instance_value(m, i),
                    "placement": _placement(action, cid),
                }
            )
        insts.sort(key=lambda d: d["correlation_id"])
        entry["instances"] = insts
    return entry


def _cubin_load_base(action, metric_names):
    """min(correlation_id) over *in-cubin* instances (amended): an
    instance is in-cubin iff `sass_by_pc(cid) != ''`
    (a real opcode or a mid-instruction byte), regardless of whether
    `source_info` resolved a DWARF line. This anchors on the cubin's
    lowest sampled instruction — the true load base — rather than the
    lowest *source-attributed* PC, which is higher when the cubin's
    prologue carries no line info (e.g. a kernel built without
    `-lineinfo`, whose PCs are all `in_cubin_no_source`). On reports
    whose in-cubin instances are all attributed the value is identical
    to the old attributed-only derivation."""
    best = None
    for name in metric_names:
        m = action.metric_by_name(name)
        if not (m.has_correlation_ids() and m.num_instances() > 0):
            continue
        cids = m.correlation_ids()
        for i in range(m.num_instances()):
            cid = cids.as_uint64(i)
            if action.sass_by_pc(cid) != "":  # in cubin (real PC or mid-insn byte)
                best = cid if best is None else min(best, cid)
    return best


def _disasm(action, base):
    """Full SASS listing: 16-byte stride from base until sass_by_pc == ''."""
    if base is None:
        return None
    insns = []
    addr = base
    # Safety bound: kernels over 1 MiB of SASS are not expected; the loop
    # terminates on the first empty sass_by_pc past the kernel end.
    while addr - base <= (1 << 20):
        text = action.sass_by_pc(addr)
        if text == "":
            break
        if text != "N/A":
            si = action.source_info(addr)
            opcode, _, operands = text.strip().partition(" ")
            insns.append(
                {
                    "address": addr - base,
                    "opcode": opcode,
                    "operands": operands.strip(),
                    "source": (None if si is None else {"file": si.file_name(), "line": si.line()}),
                }
            )
        addr += 16
    return insns


# StallReason enum code -> lowercase name (e.g. 7 -> "long_scoreboard").
_STALL_NAMES = {
    getattr(ncu_report, n): n[len("StallReason_"):].lower()
    for n in dir(ncu_report)
    if n.startswith("StallReason_")
}


def _warp_stalls(action, base):
    """Aggregate `timed_warp_samples()` (the raw periodic warp-state
    stream, ~10^5 samples/launch) into a compact per-`(rel_address,
    stall_reason)` histogram. Per-sample timestamps are
    dropped. PCs are classified with the same placement as
    metric instances; `source_info` is resolved once per distinct PC
    (tens), not per sample. Returns `None` when no warp samples were
    captured (keeps non-warp sidecars byte-identical)."""
    import collections

    samples = action.timed_warp_samples()
    if not samples:
        return None

    per_pc = collections.defaultdict(collections.Counter)  # pc -> {reason: count}
    per_reason = collections.Counter()
    not_issued = 0
    for s in samples:
        pc = s["pc"]
        reason = _STALL_NAMES.get(int(s["stall_reason"]), "unknown")
        per_pc[pc][reason] += 1
        per_reason[reason] += 1
        if s["not_issued"]:
            not_issued += 1

    pcs = []
    out_of_cubin = 0
    for pc, reasons in per_pc.items():
        if action.sass_by_pc(pc) == "":  # out of this cubin
            out_of_cubin += sum(reasons.values())
            continue
        rel = (pc - base) if (base is not None and pc >= base) else None
        pcs.append(
            {
                "rel_address": rel,
                "source": _source_ref(action, pc),  # None => in_cubin_no_source
                "reasons": dict(sorted(reasons.items())),
            }
        )
    pcs.sort(key=lambda d: (d["rel_address"] is None, d["rel_address"] or 0))
    return {
        "total_samples": len(samples),
        "not_issued_samples": not_issued,
        "out_of_cubin_samples": out_of_cubin,
        "per_reason_totals": dict(sorted(per_reason.items())),
        "pcs": pcs,
    }


def _dim(action, prefix):
    """Read a 3-tuple launch dimension from launch__{prefix}_dim_{x,y,z}."""
    out = []
    for axis in "xyz":
        name = f"launch__{prefix}_dim_{axis}"
        if name in action.metric_names():
            m = action.metric_by_name(name)
            out.append(m.as_uint64() if m.kind() == ncu_report.IMetric.ValueKind_UINT64 else int(m.value()))
        else:
            out.append(0)
    return out


def _launch(action):
    names = sorted(action.metric_names())
    base = _cubin_load_base(action, names)
    metrics = []
    for name in names:
        e = _metric_entry(action, name, base)
        if e is not None:
            metrics.append(e)
    rules = sorted(
        action.rule_results_as_dicts(),
        key=lambda r: (r.get("section_identifier", ""), r.get("rule_identifier", "")),
    )
    stream = action.metric_by_name("launch__stream_id") if "launch__stream_id" in names else None
    launch = {
        "kernel_demangled": action.name(action.NameBase_DEMANGLED),
        "kernel_mangled": action.name(action.NameBase_MANGLED),
        "kernel_function": action.name(action.NameBase_FUNCTION),
        "grid_size": _dim(action, "grid"),
        "block_size": _dim(action, "block"),
        "stream_id": (None if stream is None else stream.as_uint64()),
        "cubin_load_base": base,
        "metrics": metrics,
        "rules": rules,
        "disasm": _disasm(action, base),
    }
    # Optional per-launch warp-stall histogram; omitted when
    # no warp-state sampling was captured.
    warp = _warp_stalls(action, base)
    if warp is not None:
        launch["warp_stalls"] = warp
    return launch


def _workload(action):
    """Lightweight non-KERNEL workload (RANGE/GRAPH): name + recovered
    context/device/stream + metric/rule counts. The list verbs surface
    only headline columns. CMDLIST (OptiX command
    lists) is not ingested — out of veloq's CUDA scope."""
    names = set(action.metric_names())

    def u64(n):
        if n not in names:
            return None
        m = action.metric_by_name(n)
        return (
            m.as_uint64()
            if m.kind() == ncu_report.IMetric.ValueKind_UINT64
            else int(m.value())
        )

    return {
        "name": action.name(action.NameBase_FUNCTION),
        "context_id": u64("launch__context_id"),
        "device_id": u64("launch__device_id"),
        "stream_id": u64("launch__stream_id"),
        "metric_count": len(names),
        "rule_count": len(action.rule_results_as_dicts()),
    }


def build_sidecar(path: str) -> dict:
    report = ncu_report.load_report(path)
    launches, ranges, graphs = [], [], []
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            wt = action.workload_type()
            if wt == action.WorkloadType_KERNEL:
                launches.append(_launch(action))
            elif wt == action.WorkloadType_RANGE:
                ranges.append(_workload(action))
            elif wt == action.WorkloadType_GRAPH:
                graphs.append(_workload(action))
            # WorkloadType_CMDLIST (OptiX command lists) is intentionally
            # skipped — out of veloq's CUDA scope.
    out = {
        "schema": "ncu-native-v1",
        "ncu_version": report.get_version(),
        "session": {"versions": [{"provider": "Nsight Compute", "version": report.get_version()}]},
        "launches": launches,
    }
    # Visible signal that enum-name resolution degraded to the suffix
    # fallback; absent on a healthy sidecar so the
    # common case stays byte-identical.
    if not _ENUM_MAPS_OK:
        out["classification"] = "degraded"
    # Emit non-KERNEL arrays only when present, so a KERNEL-only report's
    # sidecar omits the `ranges` / `graphs` keys.
    if ranges:
        out["ranges"] = ranges
    if graphs:
        out["graphs"] = graphs
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="veloq NCU ncu_report-native export helper")
    ap.add_argument("report", nargs="?", help="path to a .ncu-rep file")
    ap.add_argument("--probe", action="store_true", help="capability probe: print ncu_report version and exit 0")
    args = ap.parse_args()

    if args.probe:
        sys.stdout.write(json.dumps({"ncu_report": "ok"}) + "\n")
        return 0
    if not args.report:
        sys.stderr.write("error: <report> is required (or pass --probe)\n")
        return 2

    sidecar = build_sidecar(args.report)
    # Deterministic: sorted keys so re-export is byte-identical.
    json.dump(sidecar, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
