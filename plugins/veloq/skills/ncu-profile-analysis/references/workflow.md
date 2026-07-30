# NCU Kernel-Analysis Workflow

Assume `R` is an `.ncu-rep` or `.ncu-repz` path. Use JSON for agent analysis and
CSV/table only for human review or spreadsheets. Verb matrix and
envelope shape are in [`../SKILL.md`](../SKILL.md).

## Report inventory and rules

Inventory what the report actually contains before interpreting
metrics:

```bash
# Totals overview
veloq ncu summary R | jq '.data.rows[0]'

# Launch headlines
veloq ncu launches R \
  | jq '.data.rows[] | {key, kernel: .kernel_demangled,
                         context: .context_id, stream: .stream_id,
                         device: .device_id,
                         grid: .grid_size, block: .block_size}'

# Metric inventory for one launch
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           metric_count: (.metrics | length),
           metric_prefixes: ([.metrics[]?.name // empty
                              | split("__")[0]] | unique)}'
```

If the section or metric family needed for a hypothesis is missing,
stop and recapture/export with native `ncu`; do not treat an empty
filter as proof that the bottleneck is absent.

Inspect rule findings for one launch:

```bash
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           rules: [.rules[]? | {rule: .display_name,
                                section: .section_identifier,
                                speedup: .speedup,
                                messages: .messages}]}'
```

Or across all launches in one batch (good when looking for the loudest
rule across the report):

```bash
mapfile -t IDS < <(veloq ncu launches R --limit 1000 | jq -r '.data.rows[].key')
veloq ncu inspect R $(printf -- '--row-id %s ' "${IDS[@]}") \
  | jq '.data.rows[]
        | {kernel: .kernel_demangled,
           rules: [.rules[]? | {name: .display_name, speedup: .speedup}]}'
```

Rules are a triage signal, not a final diagnosis. Use the rule's
`section_identifier` and focus metrics to select the next metric
family.

## Human review: CSV / table totals

For spreadsheet or terminal review, `ncu summary --format
csv|table R` renders a totals + session projection (the JSON form
emits the full payload). It is a slim overview only — there are no
per-metric console pages.

For agent cross-launch sweeps prefer `ncu metrics --counter
'<glob>'` over the raw page — canonical `data.rows[]` composes
with the rest of the verb matrix. Branch on `data.format` before
reading row fields: `"long"` rows carry `launch_row_id`,
`counter_name`, `value`, and `unit`; `"per_launch"` rows carry
`row_id` plus a `counters` map.

## SOL quick classification

Use Speed of Light metrics as the first routing decision when they
exist in the report:

```bash
# One launch:
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           sol: [.metrics[]?
             | select((.name // "") | test("throughput|pct_of_peak|gpu__time_duration"))
             | {name, label, unit, value}]}'

# Cross-launch sweep:
veloq ncu metrics R --counter '*throughput*' \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
```

Then choose the next drill:

| SOL pattern                  | Next evidence                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| Compute throughput dominates | Instruction mix, pipe utilization, math mode, generated SASS/PTX                     |
| Memory throughput dominates  | DRAM/L2/L1 bytes, sectors, hit rates, replay/conflict/coalescing metrics             |
| Both are low                 | Launch/occupancy, eligible warps, stall reasons, synchronization, source correlation |
| SOL metrics absent           | Recapture/export with Speed of Light or another targeted native NCU section          |

## Bottleneck-specific drills

### Occupancy / launch configuration

Decision tree:

1. Confirm the report has Launch/Occupancy-style sections or
   `launch__*` metrics.
2. Check whether block size/grid size create too few waves or too
   little parallelism.
3. If occupancy limit metrics point to registers/shared memory/warps,
   verify the matching resource value.
4. If occupancy is low but issue/throughput metrics are already high,
   do not assume increasing occupancy will help.

Evidence to collect:

- Block size, grid size, waves per SM.
- Registers per thread and shared memory per block.
- Occupancy limit metrics: blocks, registers, shared memory, warps.

Useful queries:

```bash
# One launch — shape + occupancy metrics inline
veloq ncu inspect R --row-id launch:0 \
  | jq '.data.rows[0]
        | {kernel: .kernel_demangled,
           launch: {block: .block_size, grid: .grid_size},
           metrics: [.metrics[]?
             | select((.name // "") | test("launch__|occupancy|register|shared_mem"))
             | {name, label, unit, value}]}'

# Cross-launch occupancy sweep
veloq ncu metrics R --counter 'launch__*' \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
```

If occupancy is limited by registers/shared memory, VeloQ can expose
the evidence but cannot choose the fix. Turn to source changes,
launch bounds, PTXAS resource reports, or compiler flags.

### Memory throughput / coalescing

Decision tree:

1. Confirm memory sections or `dram__` / `lts__` / `l1tex__` metrics
   exist.
2. If percentage-of-peak throughput is high, treat the kernel as
   likely bandwidth-bound and look for ways to reduce bytes or improve
   locality.
3. If throughput is low but sectors/requests/conflicts/replay are high,
   suspect access inefficiency rather than raw bandwidth.
4. If both throughput and traffic are low, pivot back to scheduler,
   dependency, or launch/occupancy evidence.

Evidence to collect:

- DRAM/L1/L2 bytes and throughput.
- Sector/request counts.
- Hit rates and replay/conflict metrics.
- Percentage-of-peak metrics from Speed of Light or memory sections.

Useful queries. `ncu metrics --counter` takes one glob, not a
comma list — run each broad pattern separately and merge in jq
when you need a single output:

```bash
veloq ncu metrics R --counter 'dram__*'   \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter 'lts__*'    \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter 'l1tex__*'  \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
```

High memory throughput near peak means optimization is likely about
reducing bytes or improving locality. Low throughput with many
sectors/requests or conflict/replay metrics points to access pattern
or coalescing work — at that point switch to `ncu disasm --row-id
launch:<idx>` for source/SASS correlation.

### Scheduler / warp stalls

Decision tree:

1. Confirm scheduler/warp-state metrics exist.
2. If eligible warps are low, identify the dominant stall family.
3. For memory dependency stalls, pivot to memory workload metrics.
4. For barrier/sync stalls, inspect source synchronization and
   divergence.
5. For dispatch/pipe stalls, pivot to instruction mix and pipe
   utilization.

Evidence to collect:

- Active/eligible warps.
- Issue active / no eligible / stall reason metrics.
- Warp state or scheduler section rule messages.

Useful queries (single glob per call):

```bash
veloq ncu metrics R --counter '*warp*'     \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter '*stall*'    \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter '*eligible*' \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter '*issue*'    \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
```

If stalls point to memory dependencies, pivot to memory metrics. If
they point to barriers or synchronization, inspect source structure
and thread/block-level synchronization (use `ncu disasm` if a
source-line index is available).

### Instruction mix / pipeline pressure

Decision tree:

1. Confirm instruction or Speed of Light compute metrics exist.
2. If a pipe is near peak, look for algorithmic or instruction-count
   reductions rather than launch tuning.
3. If instruction count is unexpectedly high, inspect generated SASS/PTX
   and source transformations.
4. If compute throughput is low while memory stalls dominate, pivot to
   memory/scheduler analysis.

Evidence to collect:

- Instruction counts by operation class.
- Pipe utilization / percentage-of-peak metrics.
- Speed of Light compute throughput metrics.

Useful queries (single glob per call):

```bash
veloq ncu metrics R --counter 'sm__inst_*'   \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter '*pipe*'       \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
veloq ncu metrics R --counter '*throughput*' \
  | jq '.data.rows[] | {launch: .launch_row_id, counter: .counter_name, value, unit}'
```

Use this to decide whether to inspect generated SASS/PTX, math mode,
vectorization, instruction count, or algorithmic work.

### Source/SASS correlation

Only run this when a rule or metric points to source-local behavior.
The dedicated verb resolves one launch's cubin and returns the
correlated rows directly (cached per cubin under
`<file>.veloq/disasm/<sha>.correlated.json`):

```bash
veloq ncu disasm R --row-id launch:0 \
  | jq '.data.rows[0]
        | {function_name,
           instruction_count: (.instructions | length),
           ptx_lines: (.ptx_lines | length),
           indexed_lines: (.source_index | length)}'
```

`ncu disasm --row-id launch:<idx>` is the only disasm entry point;
it scopes correlation to one launch's cubin and avoids re-emitting
the entire report payload.

Use native NCU GUI when you need visual cross-highlighting or NCU's
full source page. VeloQ exposes machine-readable source/disassembly
evidence; it is not a GUI replacement.

## Multi-kernel reports

Always keep workload identity in the analysis:

- `launch:<idx>` keys are the stable cross-trace handle for
  launch-centered jq joins and the workload id inside the report.
- `kernel_demangled` is the human name.
- `context_id`, `stream_id`, `device_id`, `grid_size`, and
  `block_size` distinguish otherwise similar launches.

For human review:

```bash
veloq ncu summary --format table R
veloq ncu launches R | jq '.data.rows[]'
```

For diff-style cross-report agent work, use the `key` field on every
verb's `data.rows[]` as the join column — the same recipe works for
`launches`, `metrics`, `disasm`, `sources`, etc. across two reports.

If the question is "which kernel matters most in the application",
switch to NSys first. NCU reports explain a profiled kernel; they do
not prove that kernel dominates end-to-end runtime.
