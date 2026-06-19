<!--
  AUTO-GENERATED — do not edit by hand.
  Source of truth: `#[derive(JsonSchema)]` on the structs in
  `crates/veloq-nsys-query/src/inspect.rs`. Regenerate with:
  cargo run --release -p veloq-nsys-query --example gen_inspect_shapes -- write
  CI runs `cargo test -p veloq-nsys-query --test inspect_shapes_freshness`
  which asserts on-disk content == projected output.
-->

# `inspect` — per-EventKind sub-shapes

`veloq inspect <TRACE> <ROW_ID> [<ROW_ID>...]` returns `{ count, total_matched, rows: EventDetails[] }`. Row_ids are positional (one or more) in `<kind>:<rowid>` form, e.g. `kernel:1234`. Each row's `type` tag selects which sub-shape applies. The block below is projected from the Rust structs so it stays in sync with the actual wire format — same source as `veloq schema inspect`.

```
{ type: "kernel"|"memcpy"|"memset"|"runtime"|"osrt"|"nvtx"|"sync"|"graph"|"graph_node"|"graph_event"|"cuda_event"|"overhead"|"cpu_sample"|"not_found", …per-variant fields:
    "kernel" → KernelDetails
    "memcpy" → MemcpyDetails
    "memset" → MemsetDetails
    "runtime" → RuntimeDetails
    "osrt" → OsrtDetails
    "nvtx" → NvtxDetails
    "sync" → SyncDetails
    "graph" → GraphDetails
    "graph_node" → GraphNodeDetails
    "graph_event" → GraphEventDetails
    "cuda_event" → CudaEventDetails
    "overhead" → OverheadDetails
    "cpu_sample" → CpuSampleDetails
    "not_found" → { key: string, row_id: RowId, type: "not_found" }
  }

KernelDetails:
  { block: int[], context_id: int, correlation_id?: int, demangled_name?: string, device_id: int, duration_ns: int, dynamic_shared_memory?: int, end_ns: int, global_pid?: int, graph_id?: int, graph_node_id?: int, grid: int[], key: string, nvtx_context?: NvtxContext|null, registers_per_thread?: int, row_id: RowId, short_name?: string, start_ns: int, static_shared_memory?: int, stream_id: int }

NvtxContext:
  { depth: int, iter_index?: int, name: string, range_id: RowId }

RowId:
  string

MemcpyDetails:
  { bytes: int, context_id: int, copy_kind: int, copy_kind_name: string, correlation_id?: int, device_id: int, duration_ns: int, end_ns: int, graph_node_id?: int, key: string, nvtx_context?: NvtxContext|null, row_id: RowId, start_ns: int, stream_id: int }

MemsetDetails:
  { bytes: int, context_id: int, correlation_id?: int, device_id: int, duration_ns: int, end_ns: int, graph_node_id?: int, key: string, nvtx_context?: NvtxContext|null, row_id: RowId, start_ns: int, stream_id: int, value?: int }

RuntimeDetails:
  { correlation_id?: int, duration_ns: int, end_ns: int, global_tid: int, key: string, name: string, nvtx_context?: NvtxContext|null, row_id: RowId, start_ns: int }

OsrtDetails:
  { duration_ns: int, end_ns: int, global_tid: int, key: string, name: string, row_id: RowId, start_ns: int }

NvtxDetails:
  { depth?: int, domain_id: int, duration_ns?: int, end_ns?: int, event_type: int, global_tid: int, key: string, name: string, parent_name?: string, parent_row_id?: RowId|null, path?: string, row_id: RowId, start_ns: int }

SyncDetails:
  { context_id: int, correlation_id?: int, device_id: int, duration_ns: int, end_ns: int, event_sync_id?: int, key: string, nvtx_context?: NvtxContext|null, row_id: RowId, start_ns: int, stream_id: int, sync_type: int, sync_type_name: string }

GraphDetails:
  { context_id: int, correlation_id?: int, device_id: int, duration_ns: int, end_ns: int, graph_exec_id: int, graph_id: int, key: string, row_id: RowId, start_ns: int, stream_id: int }

GraphNodeDetails:
  { duration_ns: int, end_ns?: int, global_tid?: int, graph_exec_id?: int, graph_id?: int, graph_node_id: int, key: string, original_graph_node_id?: int, row_id: RowId, start_ns: int }

GraphEventDetails:
  { duration_ns: int, end_ns: int, event_class: int, event_class_name: string, global_tid?: int, graph_exec_id?: int, graph_id: int, key: string, original_graph_id?: int, row_id: RowId, start_ns: int }

CudaEventDetails:
  { context_id: int, correlation_id?: int, device_id: int, event_id: int, event_sync_id?: int, key: string, row_id: RowId, start_ns: int, stream_id: int }

OverheadDetails:
  { correlation_id?: int, duration_ns: int, end_ns: int, global_tid?: int, key: string, overhead_type: int, overhead_type_name: string, row_id: RowId, start_ns: int }

CpuSampleDetails:
  { callchain: CallchainFrame[], cpu: int, cpu_cycles: int, global_tid: int, key: string, pid: int, row_id: RowId, start_ns: int, thread_state: int, thread_state_name?: string, tid: int }

CallchainFrame:
  { depth: int, ip: string, kernel_mode: bool, module?: string, symbol?: string, unresolved: bool }
```
