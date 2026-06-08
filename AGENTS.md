# VeloQ — Contributor Guidelines

This file is for agents/contributors **developing VeloQ itself**
(adding verbs, profile sources, fixes, refactors). User-facing
agents that use `veloq` as a black-box CLI to analyze profiles
should read the skills under `.agents/skills/`:

- `.agents/skills/nsys-profile-analysis/` — Nsight Systems timelines
- `.agents/skills/ncu-profile-analysis/` — Nsight Compute kernel reports
- `.agents/skills/pytorch-profile-analysis/` — PyTorch/Kineto traces

VeloQ (velo-query) is a profile-query CLI family. Pure CLI in /
JSON contract out by default, CSV/table projections for row-shaped
views, no GUI, no MCP server in v1. Today it covers Nsight Systems
(timeline traces), Nsight Compute (kernel reports), and experimental
PyTorch/Kineto Chrome traces through a single binary with a shared
envelope and pluggable `ProfileSource` trait. The PyTorch/Kineto source
covers the Perfetto-style Chrome trace shape used by PyTorch profiler.

## Wire-format invariants (do not break casually)

These constrain how every new verb/source must emit data. The
user-facing contract description (with examples) lives in each
skill's `SKILL.md`; this section is the maintainer-side rule set.

The JSON envelope and the per-source `version`s are VeloQ's public
contract; the crate's `0.x` Cargo version is independent of the wire
version (breaking shape changes bump `ENVELOPE_VERSION`/`source.version`
plus a CHANGELOG entry — see invariant 1; additive fields keep the
version).

1. **Envelope shape**: `veloq_core::Envelope<T>` is the only success
   payload VeloQ writes on stdout, and `veloq_core::EnvelopeError`
   is the only error payload. Both carry `schema` / `source` /
   `command` / `trace?` / `trace_span?` / `data | error`. Error
   details always carry `message` and `chain`; typed diagnostics may
   additionally populate `code` and `hint`. Bump `ENVELOPE_VERSION`
   only on a breaking shape change; additive fields keep the same
   version.
2. **Canonical list contract**: every list-shaped response uses
   `data: { count, total_matched, rows: Vec<Row>, auxiliary? }`.
   Each `Row` carries a `pub key: String` composed from the row's
   identifying axes — see the per-verb format below. Non-primary
   data (per-mode common blocks, bucket histograms, …) goes under
   `auxiliary`. New verbs MUST conform — don't add a parallel list
   field with a different name. `wire_format_smoke::every_primary_rows_item_carries_key`
   structurally enforces the `key` presence across every Response
   type.

   Per-verb `key` formats (the substrate for `INDEX(.rows; .key)`
   cross-trace joins):

   | Verb                        | Row key format                                                                                                                                                                                                |
   | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `stats`                     | `kind\|<name?>\|dev:<n?>\|stream:<n?>\|ctx:<n?>\|graph:<n?>\|graph_node:<n?>\|style:<push_pop\|start_end\|unknown?>\|nvtx:<rowid-or-none?>\|nvtx-path:<path-or-none?>\|grid:<x>x<y>x<z>?\|block:<x>x<y>x<z>?` |
   | `search`                    | `<row_id>` (e.g. `kernel:1234`)                                                                                                                                                                               |
   | `inspect`                   | `<row_id>` (matches the requested row_id; `NotFound` same)                                                                                                                                                    |
   | `timeline`                  | `bucket\|<start_ns>..<end_ns>`                                                                                                                                                                                |
   | `concurrency`               | `concurrency\|dev:<n>` (per device; nested `streams[]` carry `stream_id`, no key)                                                                                                                             |
   | `slices` instance           | `slice\|<name>\|@<cpu_start_ns>`                                                                                                                                                                              |
   | `slices` aggregate          | `scope\|<name>` / `scope\|path:<path>` per `--group-by`                                                                                                                                                       |
   | `gaps` device               | `gap\|dev:<n>\|@<start_ns>` (default; cross-stream sweep)                                                                                                                                                     |
   | `gaps` stream               | `gap\|dev:<n>\|stream:<n>\|@<start_ns>` (`--scope stream`)                                                                                                                                                    |
   | `gaps` trace                | `gap\|@<start_ns>` (`--scope trace`; multi-GPU)                                                                                                                                                               |
   | ↳ aux.streams               | `stream\|dev:<n>\|stream:<n>` (scope-independent summary)                                                                                                                                                     |
   | `correlate`                 | `<seed_row_id>` per result; embedded events use `<row_id>`                                                                                                                                                    |
   | `graph-replays`             | `graph-replay\|<synthetic_id>` where `synthetic_id` is packed `(device, context, correlationId)`                                                                                                              |
   | `hardware`                  | `host\|<hostname>`                                                                                                                                                                                            |
   | `summary`                   | `table\|<table_name>`                                                                                                                                                                                         |
   | `metrics` gpu               | `counter\|type:<type_id>\|metric:<metric_id>`                                                                                                                                                                 |
   | `metrics` nic               | `nic_counter\|nic:<id>\|port:<id>\|metric:<idx>`                                                                                                                                                              |
   | `metrics` cpu-sampling      | bare `<symbol>` / `<module-basename>` / `<tid>` / `<cpu>` per `--group-by`                                                                                                                                    |
   | `metrics` cpu-sched         | `tid:<id>` / `cpu:<id>` / `state:<name>` per `--group-by`                                                                                                                                                     |
   | `ncu summary`               | `totals` (single-row summary)                                                                                                                                                                                 |
   | `ncu launches`              | `launch:<idx>`                                                                                                                                                                                                |
   | `ncu inspect`               | `launch:<idx>`                                                                                                                                                                                                |
   | `ncu metrics` (long)        | `launch:<idx>\|counter:<name>`                                                                                                                                                                                |
   | `ncu metrics` (wide)        | `launch:<idx>`                                                                                                                                                                                                |
   | `ncu disasm`                | `kernel\|<function_name>`                                                                                                                                                                                     |
   | `ncu source-metrics` line   | `launch:<idx>\|line:<file>:<line>`                                                                                                                                                                            |
   | `ncu source-metrics` sass   | `launch:<idx>\|sass:0x<addr>`                                                                                                                                                                                 |
   | `ncu source-metrics` file   | `launch:<idx>\|file:<file>`                                                                                                                                                                                   |
   | `ncu warp-stalls` line      | `launch:<idx>\|line:<file>:<line>`                                                                                                                                                                            |
   | `ncu warp-stalls` sass      | `launch:<idx>\|sass:0x<addr>`                                                                                                                                                                                 |
   | `ncu warp-stalls` reason    | `launch:<idx>\|reason:<reason>`                                                                                                                                                                               |
   | `ncu ranges/graphs/sources` | `<entity>:<idx>` (e.g. `range:0`)                                                                                                                                                                             |
   | `pytorch summary`           | `trace\|<trace_file_path>`                                                                                                                                                                                    |
   | `pytorch search`            | `<row_id>` (e.g. `kernel:91`, `cpu_op:42`)                                                                                                                                                                    |
   | `pytorch inspect`           | `<row_id>` (matches the requested row_id; `found=false` same)                                                                                                                                                 |
   | `pytorch stats`             | `stats\|<axis>:<value>\|...`                                                                                                                                                                                  |
   | `pytorch correlate`         | `<seed_row_id>` per result; embedded events use `<row_id>`                                                                                                                                                    |
   | `pytorch timeline`          | `bucket\|<start_ns>..<end_ns>`                                                                                                                                                                                |
   | `pytorch slices` instance   | `slice\|<name>\|@<start_ns>`                                                                                                                                                                                  |
   | `pytorch slices` aggregate  | `scope\|<name-or-step>`                                                                                                                                                                                       |
   | `pytorch collectives`       | `collective\|<kind>\|rank:<n-or-none>\|step:<n-or-none>\|ordinal:<n>`                                                                                                                                          |
   | `pytorch prep`              | `sidecar\|<sidecar_name>`                                                                                                                                                                                     |

   Two traces of the same workload produce matching keys at
   matching axes — modulo `trace_span.origin_ns` if the recipe
   needs wall-clock normalization first.

   NCU launch row ids are `launch:<idx>`. `ncu inspect` is
   partial-batch friendly: out-of-range, malformed, and unsupported-kind
   row ids return success rows tagged `type: "not_found"` with `key` and
   `row_id` equal to the requested id. Other NCU drill verbs may reject
   invalid launch row ids with handled diagnostic errors.

   PyTorch rank-scoped commands are `search`, `stats`, `timeline`,
   `slices`, and `collectives`. On multi-rank traces they must require
   either `--rank <n>` or `--all-ranks`; `inspect` and `correlate` operate
   on explicit row ids and are not rank-scope gated.

3. **Per-source version (`SourceRef.version`)**: bumps independently
   from `ENVELOPE_VERSION` on any breaking shape change to that
   source's payloads. Today NSys is `v1` (`stats --group-by nvtx-path`
   rows carry the NVTX domain dimension — domain-qualified key plus
   resolved `domain_id`/`domain_pid`/`domain_name`); NCU is `v1` — the
   `ncu_report`-native wire (`inspect` drops the
   section catalog and cpu/python stacks, `summary.auxiliary.session`
   keeps only the NCU version), with the wire reporting each `ncu inspect` metric's `metric_type` /
   `metric_subtype` / `rollup` as the `ncu_report` enum _name_
   (`"counter"` rather than `1`), the raw integer kept alongside as
   `*_code`; PyTorch is `v0` — experimental, but documented fields,
   schema-target inventories, row ids/keys, command ids, and output-mode
   semantics are still covered by the source-version compatibility
   boundary.
4. **`RowId` is round-trippable**:
   `<kind>:<sqlite-compatible-rowid>` on the wire
   (`veloq_nsys_query::RowId`). Bit-packing stays inside the
   correlation index; the wire string stays human-readable.

   **`EventRef` (search/correlate rows) is a `#[serde(tag = "type")]`
   tagged enum.** Every row in
   `search.rows[]` and `correlate.events[]` carries a top-level
   `type` discriminator (`"kernel"`, `"memcpy"`, `"sync"`, …) plus
   the shared base fields (`key`, `row_id`, `name`, `start_ns`,
   `duration_ns`, optional `device_id` / `stream_id` / `global_tid`
   / `depth` / `nvtx_context`). Four kinds add per-kind headline
   columns so agents can reach grid/block/bytes/etc. without a
   follow-up `inspect` hop:

   | `type`   | Extra fields                                                                                                                                        |
   | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `kernel` | `grid: [i64;3]?`, `block: [i64;3]?`, `registers_per_thread?`, `static_shared_memory?`, `dynamic_shared_memory?`, `demangled_name?`, `mangled_name?` |
   | `memcpy` | `bytes?`, `copy_kind?`, `copy_kind_name?` (resolved label)                                                                                          |
   | `memset` | `bytes?`, `value?`                                                                                                                                  |
   | `nvtx`   | `event_type?`, `domain_id?`                                                                                                                         |

   `sync` / `runtime` / `osrt` / `graph` / `graph_node` /
   `graph_event` / `cuda_event` / `overhead` carry only the base.
   All extras are absent (not serialised) when missing from the
   trace's schema — agents reading them with jq should use the
   `// null` fallback or `select(has("grid"))` style guards.

5. **Parameterized SQL only.** Never `format!()` user input into a
   query.
6. **Type conversion at the boundary.** Read DuckDB columns as
   their native Arrow type (`Int64Array`) and convert once. Never
   force `UBIGINT`.
7. **One subcommand = one DuckDB connection = one process.** CLI
   calls stay stateless — no background threads, no channel-based
   handles. The Parquet + metadata sidecars on disk make repeat
   calls fast enough.

## Workspace layout

```
veloq/
└── crates/
    ├── veloq-core/             # Envelope, SourceRef, ProfileSource trait,
    │                             OutputFormat, sort + time helpers
    ├── veloq-data/             # Source-neutral file/parquet cache helpers
    ├── veloq/                  # The `veloq` binary — thin registry+dispatch
    │                             shell; meta verbs (`info`, `sources`, `clean`)
    ├── nsys/
    │   ├── veloq-nsys-data/    # Trace open + Parquet cache + CorrelationIndex
    │   ├── veloq-nsys-query/   # One module per NSys verb
    │   │                         (+ EventKind, RowId, KindFilter,
    │   │                          nvtx_attribution, nvtx_reverse, kind_sql)
    │   └── veloq-nsys/         # Nsys clap surface + dispatch + CSV/table
    │                             views; impls `NsysSource: ProfileSource`
    ├── ncu/
    │   └── veloq-ncu/          # `.ncu-rep` via NVIDIA's `ncu_report`
    │                             API → native sidecar + SASS/PTX
    │                             correlation; impls `NcuSource: ProfileSource`
    └── pytorch/
        ├── veloq-pytorch-data/ # Kineto Chrome trace JSON/GZ ingest,
        │                         sidecars, nesting/correlation/collectives
        ├── veloq-pytorch-query/# PyTorch verbs and response payloads
        └── veloq-pytorch/      # Pytorch clap surface + dispatch + CSV/table
                                  views; impls `PytorchSource: ProfileSource`
```

Each profile source lives under its own subdirectory
(`crates/<source>/`) so the workspace glob picks them up
(`crates/nsys/*`, `crates/ncu/*`, `crates/pytorch/*`). Future
non-Chrome trace sources slot in alongside without restructuring.

VeloQ is fully self-contained — no compile-time deps beyond
crates.io. `veloq` is the only non-library member.

## Shipped commands (status roadmap)

What's shipped vs not. Verb purposes and flag detail live in
`veloq <verb> --help` (projected from the same `JsonSchema` derive
as the response, so it can't drift) and in the per-source skill
files. Don't restate either here — record only the checkbox.

NSys verbs (registered in `crates/nsys/veloq-nsys/src/cli.rs`,
hoisted to the top level as the default source):

- [x] `summary` / `stats` / `search` / `inspect` / `correlate`
- [x] `gaps` / `slices` / `timeline` / `concurrency` / `graph-replays` / `hardware` / `metrics`
- [x] `prep` / `correlation-stats` / `nsys ncu-command`
- [x] `schema <target>`

NCU verbs (registered in `crates/ncu/veloq-ncu/src/cli.rs`,
namespaced under `ncu`):

- [x] `summary` / `launches` / `inspect` / `metrics` / `disasm`
- [x] `ranges` / `graphs` / `sources`
- [x] `source-metrics` / `warp-stalls`
- [x] `schema <target>`

PyTorch verbs (registered in `crates/pytorch/veloq-pytorch/src/cli.rs`,
namespaced under `pytorch`; experimental source version `v0`):

- [x] `summary` / `search` / `inspect` / `stats` / `correlate`
- [x] `timeline` / `slices` / `collectives`
- [x] `prep` / `schema <target>`

Meta verbs (root, owned by the binary):

- [x] `info <trace>` / `sources` / `clean <trace>` / `recipes` / `self-update`

Not shipped yet:

- [ ] `veloq compare a.nsys-rep b.nsys-rep` — cross-trace diff
- [ ] Additional profile sources. `ProfileSource` is the only
      contract a new source has to implement.

## Code conventions

- **Error-message style** (structured errors, parse diagnostics):
  one short sentence stating the offender +
  the why, optionally followed by one short suggestion. Examples:

  ```text
  --limit must be at least 1 (limit=0 would suppress total_matched too); use `--limit 1` for one row + totals
  slices requires `NVTX_EVENTS`, which is not present in this trace
  internal: stats only aggregates GPU kinds; got `runtime`
  ```

  Lowercase after a flag/identifier, no trailing period on
  single-clause messages, `internal:` prefix for invariant
  violations the user shouldn't reasonably trigger.

- **No local information leakage**: committed docs, governance
  artifacts, examples, tests, and benchmark notes must not include
  private/local machine details: absolute home paths, usernames,
  hostnames, unreviewed trace names, local worktree names, local
  artifact directories such as `.omc/`, or raw benchmark outputs tied
  to private traces. Use repo-relative paths, synthetic names, and
  portable summaries instead. Record local-only evidence in the
  working conversation or private scratch files, not in committed
  artifacts.

- **Generated artifact layout**: all veloq-generated products for a
  report live under one `<trace>.veloq/` artifact root. The
  `veloq clean <trace>` command removes that root only. It does not
  remove the input trace or a direct `_pqtdir/` input. `.nsys-rep`
  inputs export to
  `<trace>.veloq/parquetdir/` using the ctime ordering. If a caller passes that generated `parquetdir/` child
  back to VeloQ, resolve it as an alias for the owning `.nsys-rep`
  so sidecars stay under the same artifact root. Derived VeloQ caches
  invalidate on the source file mtime/size for `.nsys-rep`, or child
  parquet fingerprints for direct `_pqtdir/` inputs.
  - NSys:
    - `<trace>.veloq/parquetdir/<TABLE>.parquet` — nsys's own
      per-table parquet output (`nsys export -t parquetdir`). VeloQ
      reuses this directly as its parquet cache; no separate
      `veloq-parquet/`.
    - `<trace>.veloq/correlation.bin` — `CorrelationIndex`;
      `(device, context, correlationId)` index
    - `<trace>.veloq/meta.bin` — `TraceMetaCache`; schema version,
      capabilities, hardware, per-table counts, NVTX nesting
      (`HashMap<i64, NvtxEntry { depth, iter_index }>`). Built
      by `summary` / `prep`.
    - `<trace>.veloq/nvtx-parent.parquet` — `RuntimeNvtxParent`;
      runtime-row → enclosing NVTX chains for grouped stats paths.
    - `<trace>.veloq/nvtx-tree.parquet` — `NvtxTree`; flattened
      NVTX range tree for stack-at-time and path aggregate queries.
  - NCU:
    - `<report>.veloq/ncu-native.json.gz` — `native::cache::build_or_load`;
      gzipped JSON sidecar from the `ncu_report` ingest. The sole NCU
      ingest path, reused by every NCU verb. Freshness is keyed on a
      sha256 content-hash of the `.ncu-rep` (checkout-stable).
    - `<report>.veloq/disasm/<sha>.correlated.json` — per-cubin
      SASS/PTX/source-line index from nvdisasm + cuobjdump.
  - PyTorch:
    - `<input>.veloq/pytorch/meta.bin` — bincode cache for the typed
      single-trace model; freshness is keyed on trace-file path, mtime,
      and size.
    - `<input>.veloq/pytorch/events.parquet` — typed event rows.
    - `<input>.veloq/pytorch/args.parquet` — event arg key/value rows.
    - `<input>.veloq/pytorch/flows.parquet` — resolved flow edges.
    - `<input>.veloq/pytorch/links.parquet` — nesting, step, external,
      correlation, and flow links.
    - `<input>.veloq/pytorch/collectives.parquet` — grouped collective
      rows.

- **NSys version support**: only `v3_standard` (NSys schema 3.x)
  ships today. Pre-3.x traces fail at `Trace::open` with a clear
  error rather than being papered over; if a real legacy trace
  shows up, add a new adapter rather than reintroducing a generic
  fallback.

- **Domain knowledge** (load-bearing for SQL implementers):
  - `globalTid` bit layout:
    `[bits 48-63: HW/Host ID] [bits 24-47: PID (24b)] [bits 16-23: Source Domain ID (8b)] [bits 0-15: TID (16b)]`.
    Extraction: `(id >> 24) & 0xFFFFFF` for PID, `id & 0xFFFF` for
    TID — TID is 16 bits, not 24. The middle 8 bits are the
    source-domain id (`0x00` for OSRT tracer, `0x3B` for CUDA
    driver), and joining `PROCESSES.globalPid` to
    `ThreadNames.globalTid` across domains requires the `>> 24`
    PID-only mask (otherwise the source-domain byte adds a
    constant offset to the wrong-extracted "pid"). Use
    `veloq_nsys_query::decode_global_tid`.
  - `NVTX_EVENTS` is optional — always probe first.
  - `correlationId` is **not** globally unique: it resets per
    `(device, context)`. SQL that walks runtime → kernel/memcpy/
    memset must bridge through `TARGET_INFO_CUDA_CONTEXT_INFO`
    (`device, context → process_id`) and match the runtime's
    native_pid (high bits of `globalTid`). See
    `nvtx_attribution.rs` (forward) and `nvtx_reverse.rs`
    (reverse) for the canonical CTEs.
  - Synthetic Correlation ID: `device(8b) | context(16b) | raw_corr(40b)`.

## Authoring a new `ProfileSource`

A new backend lives in `crates/<source>/veloq-<source>/`
and implements [`veloq_core::ProfileSource`]. Five concrete obligations:

1. **Identity.** `kind()` returns a lowercase ASCII slug (becomes the
   CLI namespace `veloq <kind> <verb>` and lands in
   `envelope.source.kind`). `version()` returns a `&'static str`
   like `"v0"` / `"v1"` and bumps independently from the envelope
   schema version — bump on any breaking shape change to the
   source's responses.

2. **Trace detection.** `detect(&Path)` is a side-effect-free
   heuristic — file extension or magic-byte sniff, **no `open()` calls**.
   Used by `veloq info <trace>` to pick a source without the user
   naming one. Two sources MUST NOT both return `true` for the
   same path; tie-break is undefined.

3. **CLI tree.** `cli()` returns a `clap::Command`. Conventional shape:

   ```rust
   fn cli(&self) -> Command {
       let parent = Command::new(Self::KIND)
           .about("...")
           .subcommand_required(true)
           .arg_required_else_help(true);
       Cmd::augment_subcommands(parent)
   }
   ```

   The binary grafts this subtree under `veloq <kind> …`. If your
   source is registered as the configured default (today: NSys),
   its verbs are also hoisted to the top level.

4. **The `run -> SourceRunResult<i32>` tri-state.** Three outcomes, each
   with a precise stdout contract:

   | Return   | Meaning                                    | What's on stdout                                                                              |
   | -------- | ------------------------------------------ | --------------------------------------------------------------------------------------------- |
   | `Ok(0)`  | Verb succeeded                             | One pretty-JSON success [`Envelope`]                                                          |
   | `Ok(1)`  | Verb failed; source already wrote envelope | One pretty-JSON [`EnvelopeError`] with `source`/`command`/`trace` set                         |
   | `Err(_)` | Top-level/unhandled failure                | Nothing; the binary's `main` writes a CLI-level error envelope (no source/verb/trace context) |

   Splitting `Ok(1)` from `Err` lets the source keep `verb` and
   `trace` on the envelope — drop them and the agent loses
   dispatch context. In practice: user-facing failures should be
   caught at the source boundary, written as an `EnvelopeError`, and
   returned as `Ok(1)`; only top-level/unhandled failures should
   bubble as `Err(_)`. Prefer
   [`veloq_core::write_diagnostic_error_envelope`] for the handled
   write — it centralises the JSON-on-stdout envelope and the
   format-dependent stderr mirror so all sources stay consistent.

5. **stdout / stderr split.** stdout is reserved for the JSON
   envelope (success or error). In JSON mode, handled errors keep
   stderr quiet so agents do not have to dedupe a human mirror.
   In CSV/table mode, stderr carries the human mirror
   (`veloq: <message>`) plus any progress logs (`log::info!`-routed
   lines like Parquet build progress). Agents read stdout; humans
   read stderr. CSV/table outputs replace the JSON envelope on stdout.

Registration:

```rust
// crates/veloq/src/main.rs
let sources: Vec<Box<dyn ProfileSource>> = vec![
    Box::new(NsysSource),
    Box::new(NcuSource),
    Box::new(MyNewSource),   // ← add here
];
```

The dispatcher walks the registry by `kind()`. Adding a source is
one line plus the source crate.

## Pre-commit checklist

- [ ] Routine/default validation:
      `govctl verify GUARD-GOVCTL-CHECK`,
      `govctl verify GUARD-FMT`,
      `govctl verify GUARD-WORKSPACE-CHECK`, and
      `govctl verify GUARD-SOURCE-REGISTRY-CONTRACT`.
- [ ] Source or wire-contract changes → run the matching contract
      guard(s): `GUARD-NSYS-WIRE-CONTRACT`,
      `GUARD-NCU-WIRE-CONTRACT`,
      `GUARD-PYTORCH-WIRE-CONTRACT`,
      `GUARD-ROW-WIRE-CONTRACT`,
      `GUARD-CLI-IO-CONTRACT`, or
      `GUARD-ARTIFACT-CACHE-CONTRACT`.
- [ ] Release/full-CI validation:
      `govctl verify GUARD-WORKSPACE-CHECK`,
      `govctl verify GUARD-FULL-CI-CLIPPY`, and
      `govctl verify GUARD-FULL-CI-TEST`.
      The release commands are the full workspace check, full
      all-targets clippy, and full workspace test suite.
- [ ] No `unwrap()` / `expect()` / `[i]` indexing in lib **or**
      tests — the workspace's `clippy::unwrap_used` / `expect_used`
      / `indexing_slicing` denies apply to every target, and the
      release clippy guard runs `--all-targets` to actually enforce
      them on integration tests too. Use `ok_or_else` + `?` instead.
- [ ] New subcommand → updated this file's roadmap + README
      example + matching `.agents/skills/*` profile-analysis skill
      (the skill is the user-facing contract description; this
      file is the maintainer-side invariant).
