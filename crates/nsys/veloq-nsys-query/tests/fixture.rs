//! Tiny synthetic NSys parquetdir traces for integration tests.
//!
//! veloq reads NSys traces as parquetdir exports.
//! Test fixtures are built by populating an in-memory DuckDB
//! (`CREATE TABLE` + `INSERT` for whichever subset of NSys tables the
//! test needs) and then COPYing each table to `<tempdir>/<TABLE>.parquet`.
//! The fixture's path points at a `_pqtdir/` directory directly
//! consumable by `Trace::open`.
//!
//! The schema mirrors the subset of real Nsight Systems exports veloq
//! cares about: enough columns on each table to exercise name
//! resolution, time-window clipping, and per-kind aggregation paths.
//! Tests opt in to whichever tables they need; absent tables exercise
//! veloq's table-presence guards.
//!
//! Builders return `Result<Fixture>` so the workspace's no-panic
//! policy applies to test setup too — failures become test failures
//! via `?`, not panics that abort the runner.

#![allow(
    dead_code,
    reason = "shared integration-test fixture: each test binary uses a different subset, so per-helper dead-code analysis flags items in every test except the one that consumes them"
)]

use anyhow::{Context, Result};
use duckdb::{Connection, params};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Handle returned by every fixture builder. Owns the tempdir so the
/// parquetdir outlives the test's `Trace::open(...)` call.
pub struct Fixture {
    pub path: PathBuf,
    _dir: TempDir,
}

/// COPY every user-created table from the in-memory DuckDB connection
/// to `<tempdir>/test_pqtdir/<TABLE>.parquet` and wrap the result in a
/// `Fixture`. Builders call this as their final step.
fn finalize_to_pqtdir(conn: &Connection, dir: TempDir) -> Result<Fixture> {
    let pqtdir = dir.path().join("test_pqtdir");
    fs::create_dir_all(&pqtdir)
        .with_context(|| format!("create parquetdir {}", pqtdir.display()))?;

    // Enumerate user tables in the default schema.
    let mut stmt = conn
        .prepare(
            "SELECT table_name FROM information_schema.tables \
             WHERE table_schema = 'main' AND table_type = 'BASE TABLE'",
        )
        .context("prepare information_schema scan")?;
    let mut rows = stmt.query([])?;
    let mut tables = Vec::new();
    while let Some(r) = rows.next()? {
        let t: String = r.get(0)?;
        tables.push(t);
    }
    for table in &tables {
        let out = pqtdir.join(format!("{table}.parquet"));
        let out_lit = out.to_string_lossy().replace('\'', "''");
        let sql = format!(r#"COPY (SELECT * FROM "{table}") TO '{out_lit}' (FORMAT PARQUET)"#);
        conn.execute(&sql, [])
            .with_context(|| format!("copy {table} to parquet"))?;
    }

    Ok(Fixture {
        path: pqtdir,
        _dir: dir,
    })
}

impl Fixture {
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// Canonical NSys-export schema, table by table. Each entry is
/// `(table_name, "CREATE TABLE <name> (...);")`. The column lists are
/// the union of every column any veloq query path reads against this
/// table, taken from the NSys schema adapter
/// ([[veloq_nsys_data::adapter::v3_standard]]) plus the documented
/// production columns we've observed.
///
/// Why a single canonical set rather than per-fixture CREATEs:
/// repeated "fixture missed a column" bugs kept surfacing
/// (e.g. CUPTI_ACTIVITY_KIND_KERNEL's `graphId` / `graphNodeId`
/// columns, which only some fixtures had). Centralising the schema
/// here means a new column gets added in exactly one place.
///
/// Listed in StringIds-first, TARGET_INFO-second, CUPTI-third,
/// auxiliary-last order. The ordering is stylistic; it makes a
/// `CREATE TABLE` print look like a real NSys schema dump.
///
/// Use [`setup_canonical_schema`] / [`setup_canonical_schema_minus`]
/// rather than splicing this constant directly.
pub const CANONICAL_TABLES: &[(&str, &str)] = &[
    (
        "StringIds",
        "CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);",
    ),
    (
        "TARGET_INFO_CUDA_CONTEXT_INFO",
        "CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (\
            deviceId BIGINT, contextId BIGINT, processId BIGINT\
         );",
    ),
    (
        "TARGET_INFO_GPU_METRICS",
        "CREATE TABLE TARGET_INFO_GPU_METRICS (\
            typeId BIGINT, sourceId BIGINT, typeName TEXT, \
            metricId BIGINT, metricName TEXT\
         );",
    ),
    (
        "TARGET_INFO_NETWORK_METRICS",
        "CREATE TABLE TARGET_INFO_NETWORK_METRICS (\
            metricsListId BIGINT NOT NULL, metricsIdx BIGINT NOT NULL, \
            name TEXT NOT NULL, description TEXT NOT NULL, unit TEXT NOT NULL\
         );",
    ),
    (
        "TARGET_INFO_NIC_INFO",
        "CREATE TABLE TARGET_INFO_NIC_INFO (\
            GUID BIGINT NOT NULL, stateName TEXT NOT NULL, \
            nicId BIGINT NOT NULL, name TEXT NOT NULL, \
            deviceId BIGINT NOT NULL, vendorId BIGINT NOT NULL, \
            linkLayer BIGINT NOT NULL\
         );",
    ),
    (
        "NVTX_EVENTS",
        "CREATE TABLE NVTX_EVENTS (\
            start BIGINT, \"end\" BIGINT, globalTid BIGINT, \
            textId BIGINT, text TEXT, domainId BIGINT, eventType BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (\
            start BIGINT, \"end\" BIGINT, \
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (\
            start BIGINT, \"end\" BIGINT, \
            deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
            shortName BIGINT, demangledName BIGINT, mangledName BIGINT, \
            gridX BIGINT, gridY BIGINT, gridZ BIGINT, \
            blockX BIGINT, blockY BIGINT, blockZ BIGINT, \
            correlationId BIGINT, \
            registersPerThread BIGINT, \
            staticSharedMemory BIGINT, \
            dynamicSharedMemory BIGINT, \
            globalPid BIGINT, \
            graphId BIGINT, \
            graphNodeId BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (\
            start BIGINT, \"end\" BIGINT, \
            deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
            bytes BIGINT, copyKind BIGINT, correlationId BIGINT, \
            graphNodeId BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (\
            start BIGINT, \"end\" BIGINT, \
            deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
            bytes BIGINT, value BIGINT, correlationId BIGINT, \
            graphNodeId BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (\
            start BIGINT, \"end\" BIGINT, \
            deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
            syncType BIGINT, correlationId BIGINT, \
            eventId BIGINT, eventSyncId BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_OVERHEAD",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_OVERHEAD (\
            start BIGINT, \"end\" BIGINT, \
            eventClass BIGINT, globalTid BIGINT, \
            correlationId BIGINT, nameId BIGINT, overheadType BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (\
            start BIGINT, \"end\" BIGINT, \
            deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
            correlationId BIGINT, globalPid BIGINT, \
            graphId BIGINT, graphExecId BIGINT\
         );",
    ),
    (
        "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
        "CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT (\
            timestamp BIGINT, \
            deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
            correlationId BIGINT, globalPid BIGINT, \
            eventId BIGINT, eventSyncId BIGINT\
         );",
    ),
    (
        "CUDA_GRAPH_EVENTS",
        "CREATE TABLE CUDA_GRAPH_EVENTS (\
            start BIGINT, \"end\" BIGINT, \
            eventClass BIGINT, globalTid BIGINT, nameId BIGINT, \
            graphId BIGINT, originalGraphId BIGINT, graphExecId BIGINT\
         );",
    ),
    (
        "CUDA_GRAPH_NODE_EVENTS",
        "CREATE TABLE CUDA_GRAPH_NODE_EVENTS (\
            start BIGINT, \"end\" BIGINT, \
            eventClass BIGINT, globalTid BIGINT, nameId BIGINT, \
            graphNodeId BIGINT NOT NULL, originalGraphNodeId BIGINT\
         );",
    ),
    (
        "OSRT_API",
        "CREATE TABLE OSRT_API (\
            start BIGINT, \"end\" BIGINT, \
            globalTid BIGINT, nameId BIGINT\
         );",
    ),
    (
        "COMPOSITE_EVENTS",
        "CREATE TABLE COMPOSITE_EVENTS (\
            id BIGINT PRIMARY KEY, start BIGINT, cpu BIGINT, \
            threadState BIGINT, globalTid BIGINT, cpuCycles BIGINT\
         );",
    ),
    (
        "SAMPLING_CALLCHAINS",
        "CREATE TABLE SAMPLING_CALLCHAINS (\
            id BIGINT, symbol BIGINT, module BIGINT, \
            kernelMode BIGINT, thumbCode BIGINT, unresolved BIGINT, \
            specialEntry BIGINT, originalIP BIGINT, \
            unwindMethod BIGINT, stackDepth BIGINT\
         );",
    ),
    (
        "SCHED_EVENTS",
        "CREATE TABLE SCHED_EVENTS (\
            start BIGINT NOT NULL, cpu BIGINT NOT NULL, \
            isSchedIn BIGINT NOT NULL, globalTid BIGINT, \
            threadState BIGINT, threadBlock BIGINT\
         );",
    ),
    (
        "ENUM_SAMPLING_THREAD_STATE",
        "CREATE TABLE ENUM_SAMPLING_THREAD_STATE (\
            id BIGINT PRIMARY KEY, name TEXT, label TEXT\
         );",
    ),
    (
        "GPU_METRICS",
        "CREATE TABLE GPU_METRICS (\
            rawTimestamp BIGINT, timestamp BIGINT, \
            typeId BIGINT, metricId BIGINT, value BIGINT\
         );",
    ),
    (
        "NET_NIC_METRIC",
        "CREATE TABLE NET_NIC_METRIC (\
            start BIGINT NOT NULL, \"end\" BIGINT NOT NULL, \
            globalId BIGINT NOT NULL, portId BIGINT NOT NULL, \
            metricsListId BIGINT NOT NULL, metricsIdx BIGINT NOT NULL, \
            value BIGINT NOT NULL\
         );",
    ),
    (
        "NIC_ID_MAP",
        "CREATE TABLE NIC_ID_MAP (\
            nicId BIGINT NOT NULL, globalId BIGINT NOT NULL\
         );",
    ),
];

/// Create every NSys table in [`CANONICAL_TABLES`]. Fixtures call this
/// once and then INSERT only the rows the scenario needs. No "column
/// missing on this fixture" bugs by construction.
pub fn setup_canonical_schema(conn: &Connection) -> Result<()> {
    setup_canonical_schema_minus(conn, &[])
}

/// Like [`setup_canonical_schema`], but skip every table whose name
/// appears in `exclude`. For negative-path tests that probe the
/// absence of a specific table (e.g. an "NVTX_EVENTS missing"
/// regression guard). Unknown names in `exclude` are silently
/// ignored — callers can safely pass over-broad lists.
pub fn setup_canonical_schema_minus(conn: &Connection, exclude: &[&str]) -> Result<()> {
    for (name, sql) in CANONICAL_TABLES {
        if exclude.iter().any(|e| e == name) {
            continue;
        }
        conn.execute_batch(sql)
            .with_context(|| format!("create canonical table {name}"))?;
    }
    Ok(())
}

/// Legacy `CUPTI_ACTIVITY_KIND_KERNEL` schema **without** the
/// `mangledName` column — represents older NSys exports. Pinned at
/// 20 cols so the `mangled_axis_falls_back_to_demangled_when_column_absent`
/// test has a real schema to exercise the column-absence fallback
/// path against.
///
/// New full-schema fixtures: use [`setup_canonical_schema`], which
/// includes `mangledName`. This constant is intentionally narrower —
/// don't add columns to it without checking what the fallback tests
/// actually pin.
pub const KERNEL_TABLE_SQL: &str = "\
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ( \
        start BIGINT, \"end\" BIGINT, \
        deviceId BIGINT, contextId BIGINT, streamId BIGINT, \
        shortName BIGINT, demangledName BIGINT, \
        gridX BIGINT, gridY BIGINT, gridZ BIGINT, \
        blockX BIGINT, blockY BIGINT, blockZ BIGINT, \
        correlationId BIGINT, \
        registersPerThread BIGINT, \
        staticSharedMemory BIGINT, \
        dynamicSharedMemory BIGINT, \
        globalPid BIGINT, \
        graphId BIGINT, \
        graphNodeId BIGINT \
    );";

/// Insert one anchor kernel row — the common case where a fixture
/// just needs *some* kernel so `Trace::open` finds a primary origin
/// for span-relative math. Defaults: deviceId=0, contextId=0,
/// streamId=7, kernel name id 1, grid/block 1×1×1/128×1×1,
/// correlationId 1.
///
/// Builders that need to vary `correlationId`, `deviceId`, etc.
/// keep their own open-coded `INSERT` — the helper is for the
/// "I just need a row at the trace origin" case.
pub fn insert_anchor_kernel(conn: &Connection, start_ns: i64, end_ns: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            start_ns, end_ns, 0i32, 0i64, 7i64, 1i64, 1i64, 1i64, 1i64, 1i64, 128i64, 1i64, 1i64,
            1i64, 32i64, 0i64, 0i64, 0i64,
        ],
    )?;
    Ok(())
}

/// Smallest useful fixture: 4 kernels (2 distinct shortNames), 2
/// memcpys, 1 memset, all on (device=0, context=0, stream=7). Time
/// origin is 100ms so a relative `--time-range 0-100ms` window is
/// empty and `0-200ms` includes everything.
pub fn minimal_gpu() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    // `minimal_gpu` stays *intentionally sparse*: only StringIds +
    // CUDA context ownership + KERNEL + MEMCPY + MEMSET. Five
    // negative-path tests (e.g.
    // `errors_on_missing_nvtx_events_table`,
    // `mangled_axis_falls_back_to_demangled_when_column_absent`,
    // `missing_*_errors_with_capture_hint`) rely on this fixture
    // lacking NVTX_EVENTS / GPU_METRICS / COMPOSITE_EVENTS /
    // NET_NIC_METRIC / a mangledName column. Switching to the
    // canonical helper would silently turn each of those into a
    // false negative.
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT, contextId BIGINT, processId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            bytes BIGINT, copyKind BIGINT, correlationId BIGINT,
            graphNodeId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            bytes BIGINT, value BIGINT, correlationId BIGINT,
            graphNodeId BIGINT
        );
        "#,
    )
    .context("create minimal_gpu schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for minimal_gpu")?;

    // StringIds: 1 = "fast_kernel", 2 = "slow_kernel"
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "fast_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![2i64, "slow_kernel"],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO \
         (deviceId, contextId, processId) VALUES (?, ?, ?)",
        params![0i32, 0i64, 12345i64],
    )?;

    // 2 fast_kernel (1ms each), 2 slow_kernel (10ms each) on stream 7.
    // start_ns ranges 100ms..150ms.
    let kernels: &[(i64, i64, i64)] = &[
        (100_000_000, 101_000_000, 1), // fast
        (102_000_000, 103_000_000, 1), // fast
        (110_000_000, 120_000_000, 2), // slow
        (130_000_000, 140_000_000, 2), // slow
    ];
    for (i, (s, e, name_id)) in kernels.iter().enumerate() {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s,
                *e,
                0i32,
                0i64,
                7i64,
                *name_id,
                *name_id,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                (i as i64) + 1,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }

    // 2 memcpys (H2D + D2H), 4 KiB each, 0.5ms each.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY \
         (start, \"end\", deviceId, contextId, streamId, bytes, copyKind, correlationId) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            104_000_000i64,
            104_500_000i64,
            0i32,
            0i64,
            7i64,
            4096i64,
            1i64,
            100i64
        ],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY \
         (start, \"end\", deviceId, contextId, streamId, bytes, copyKind, correlationId) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            141_000_000i64,
            141_500_000i64,
            0i32,
            0i64,
            7i64,
            4096i64,
            2i64,
            101i64
        ],
    )?;

    // 1 memset, 0.2ms, 1 KiB.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET \
         (start, \"end\", deviceId, contextId, streamId, bytes, value, correlationId) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            125_000_000i64,
            125_200_000i64,
            0i32,
            0i64,
            7i64,
            1024i64,
            0i64,
            102i64
        ],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

/// Concurrency/overlap fixture (device 0). Mirrors the worked example so the smoke test can assert exact numbers:
/// - stream 7: kernel `[0,60ms)` + kernel `[50,100ms)` — same-stream
///   PDL overlap of 10ms.
/// - stream 8: kernel `[30,90ms)` + memcpy `[80,120ms)` — compute/copy
///   overlap of 10ms.
///
/// Device: sum 210ms, union 120ms, overlap 90ms, max_concurrency 3;
/// compute_union 100ms, copy_union 40ms, compute_copy_overlap 20ms.
pub fn concurrency_overlap() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO \
         (deviceId, contextId, processId) VALUES (?, ?, ?)",
        params![0i32, 0i64, 12345i64],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "k"],
    )?;

    // (start_ns, end_ns, stream_id)
    let kernels: &[(i64, i64, i64)] = &[
        (0, 60_000_000, 7),           // K1, stream 7
        (50_000_000, 100_000_000, 7), // K2, stream 7 (PDL: overlaps K1)
        (30_000_000, 90_000_000, 8),  // K3, stream 8
    ];
    for (i, (s, e, stream)) in kernels.iter().enumerate() {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s,
                *e,
                0i32,
                0i64,
                *stream,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                (i as i64) + 1,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }

    // memcpy M1 on stream 8, [80ms,120ms) — overlaps K3 (compute/copy).
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY \
         (start, \"end\", deviceId, contextId, streamId, bytes, copyKind, correlationId) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            80_000_000i64,
            120_000_000i64,
            0i32,
            0i64,
            8i64,
            4096i64,
            2i64,
            200i64
        ],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

/// Two-device fixture for concurrency scope filtering. Device 0 and
/// device 1 both have work, with different durations so tests can
/// assert the selected device's measures directly.
pub fn concurrency_two_devices() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO \
         (deviceId, contextId, processId) VALUES (?, ?, ?), (?, ?, ?)",
        params![0i32, 0i64, 12345i64, 1i32, 0i64, 12345i64],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "k"],
    )?;

    for (device, start_ns, end_ns, stream_id, correlation_id) in &[
        (0i32, 0i64, 10_000_000i64, 7i64, 1i64),
        (1i32, 20_000_000i64, 50_000_000i64, 9i64, 2i64),
    ] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *start_ns,
                *end_ns,
                *device,
                0i64,
                *stream_id,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                *correlation_id,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Minimal NVTX-attribution fixture: two NVTX ranges, each with a
/// runtime launch and a kernel attributed via correlationId. Range A
/// runs 100..200ms, range B runs 300..400ms — well-separated so a
/// time window can cleanly isolate either, or straddle the gap
/// between them. Used by slices/stats `--nvtx` integration tests.
pub fn nvtx_attribution() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let pid: i64 = 12345;
    let global_tid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "step_kernel"],
    )?;

    // Two NVTX ranges. Range A: 100..200ms (text="step_a").
    //                  Range B: 300..400ms (text="step_b").
    for (s, e, name) in &[
        (100_000_000i64, 200_000_000i64, "step_a"),
        (300_000_000i64, 400_000_000i64, "step_b"),
    ] {
        conn.execute(
            "INSERT INTO NVTX_EVENTS \
             (start, \"end\", globalTid, textId, text, domainId, eventType) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, global_tid, None::<i64>, *name, 0i64, 60i64],
        )?;
    }

    // One runtime launch + kernel inside each range, distinct correlationIds.
    for (corr, runtime_start, kernel_start, kernel_end) in &[
        (1001i64, 120_000_000i64, 130_000_000i64, 140_000_000i64),
        (1002i64, 320_000_000i64, 330_000_000i64, 340_000_000i64),
    ] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                *runtime_start,
                *runtime_start + 1_000_000i64,
                global_tid,
                *corr,
                None::<i64>
            ],
        )?;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *kernel_start,
                *kernel_end,
                0i32,
                1i64,
                7i64,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                *corr,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// One NVTX range [100..200ms] enclosing two kernels on **different
/// streams** (stream 7: 10ms; stream 8: 20ms), both device 0. Exercises
/// `slices --stream` scoping of GPU attribution.
pub fn nvtx_attribution_multistream() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let pid: i64 = 12345;
    let global_tid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO NVTX_EVENTS \
         (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            200_000_000i64,
            global_tid,
            None::<i64>,
            "step_a",
            0i64,
            60i64
        ],
    )?;

    // Two launches inside the range, distinct correlationIds + streams.
    for (corr, runtime_start, kernel_start, kernel_end, stream) in &[
        (
            2001i64,
            120_000_000i64,
            130_000_000i64,
            140_000_000i64,
            7i64,
        ),
        (
            2002i64,
            150_000_000i64,
            160_000_000i64,
            180_000_000i64,
            8i64,
        ),
    ] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                *runtime_start,
                *runtime_start + 1_000_000i64,
                global_tid,
                *corr,
                None::<i64>
            ],
        )?;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *kernel_start,
                *kernel_end,
                0i32,
                1i64,
                *stream,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                *corr,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// NVTX-nesting fixture: 4 ranges + 1 instant marker on a single
/// `(globalTid, domainId)` so the nesting stack actually exercises
/// depth assignment. Layout (in ns):
///
/// ```text
///   outer  [0 .. 100ms]                  depth 0   rowid=1
///   inner  [25ms .. 75ms]   inside outer depth 1   rowid=2
///   leaf   [40ms .. 60ms]   inside inner depth 2   rowid=3
///   marker @ 50ms          (instant)     depth 2   rowid=4 (end IS NULL)
///   sibling[110ms .. 130ms] after outer  depth 0   rowid=5
/// ```
///
/// Used by `search` and the standalone nesting integration test.
pub fn nvtx_nested() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let gtid: i64 = 12345i64 << 24;
    // All on the default domain (id=0). Insert order is intentionally
    // out-of-time so the test exercises the stack scan's start-sort
    // pass; if we relied on insertion order, the bug would lurk.
    let rows: &[(i64, Option<i64>, &str)] = &[
        (40_000_000, Some(60_000_000), "leaf"),
        (0, Some(100_000_000), "outer"),
        (110_000_000, Some(130_000_000), "sibling"),
        (50_000_000, None, "marker"),
        (25_000_000, Some(75_000_000), "inner"),
    ];
    for (s, e, name) in rows {
        conn.execute(
            "INSERT INTO NVTX_EVENTS \
             (start, \"end\", globalTid, textId, text, domainId, eventType) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, gtid, None::<i64>, *name, 0i64, 60i64],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Sync-aware fixture: 2 kernels + 3 `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`
/// events of different `syncType` values (1=cudaEventSynchronize,
/// 3=cudaStreamSynchronize, 4=cudaDeviceSynchronize). One of the syncs
/// shares a `correlationId` with one of the kernels so `correlate` can
/// walk sync ↔ kernel. Also seeds `TARGET_INFO_CUDA_CONTEXT_INFO` so
/// runtime-style correlation lookups can resolve too.
///
/// Time origin is 100ms (matches minimal_gpu) so relative `--time-range`
/// flags compose the same way across fixtures.
pub fn with_sync() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let pid: i64 = 12345;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "step_kernel"],
    )?;

    // Two kernels. One shares correlationId=900 with the
    // cudaStreamSynchronize below so correlate can walk between them.
    let kernels: &[(i64, i64, i64)] = &[
        (100_000_000, 110_000_000, 900), // 10ms — paired with sync below
        (120_000_000, 130_000_000, 901),
    ];
    for (s, e, corr) in kernels {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s, *e, 0i32, 1i64, 7i64, 1i64, 1i64, 1i64, 1i64, 1i64, 128i64, 1i64, 1i64, *corr,
                32i64, 0i64, 0i64, 0i64,
            ],
        )?;
    }

    // 3 sync events:
    //   row 1: cudaEventSynchronize       (1ms)  on stream 0, eventSyncId=777
    //   row 2: cudaStreamSynchronize      (2ms)  on stream 7, correlationId=900 (pairs with kernel)
    //   row 3: cudaDeviceSynchronize      (5ms)  on stream 0
    type SyncRow = (i64, i64, i64, i64, Option<i64>, Option<i64>);
    let syncs: &[SyncRow] = &[
        // (start, end, stream, sync_type, correlationId, eventSyncId)
        (111_000_000, 112_000_000, 0, 1, None, Some(777)),
        (115_000_000, 117_000_000, 7, 3, Some(900), None),
        (140_000_000, 145_000_000, 0, 4, None, None),
    ];
    for (s, e, stream, sync_type, corr, esync) in syncs {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION \
             (start, \"end\", deviceId, contextId, streamId, syncType, correlationId, \
              eventId, eventSyncId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s,
                *e,
                0i32,
                1i64,
                *stream,
                *sync_type,
                *corr,
                Option::<i64>::None,
                *esync,
            ],
        )?;
    }

    // 2 cudaEventRecord placements. The first pairs with sync row 1
    // above via eventSyncId=777 — agents follow that chain to learn
    // "the cudaEventSynchronize at t=111ms was waiting on the event
    // recorded at t=105ms on stream 7."
    let cuda_events: &[(i64, i64, i64, i64, Option<i64>)] = &[
        // (timestamp, stream, eventId, eventSyncId, correlationId)
        (105_000_000, 7, 42, 777, Some(910)),
        (130_500_000, 7, 43, 778, Some(911)),
    ];
    for (ts, stream, eid, esync, corr) in cuda_events {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT \
             (timestamp, deviceId, contextId, streamId, \
              correlationId, globalPid, eventId, eventSyncId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![*ts, 0i32, 1i64, *stream, *corr, 0i64, *eid, *esync],
        )?;
    }

    // 3 overhead spans:
    //   row 1: overheadType=4 (cupti_instrumentation) — 100us, NULL correlation
    //   row 2: overheadType=8 (command_buffer_full)   — 50us,  NULL correlation
    //   row 3: overheadType=2 (driver_compiler)       — 200us, correlationId=900
    //          (paired with the first kernel for the
    //          correlate-overhead-with-real-correlation regression).
    let overheads: &[(i64, i64, i64, Option<i64>)] = &[
        (90_000_000, 90_100_000, 4, None),
        (150_000_000, 150_050_000, 8, None),
        (100_000_000, 100_200_000, 2, Some(900)),
    ];
    for (s, e, ot, corr) in overheads {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_OVERHEAD \
             (start, \"end\", eventClass, globalTid, correlationId, nameId, overheadType) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, 0i64, pid << 24, *corr, 0i64, *ot],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// CUDA-graph fixture targeting the `--cuda-graph-trace=graph` mode —
/// graph_trace rows are the *only* per-launch record for the captured
/// work (kernels-inside-graphs do not appear in
/// `CUPTI_ACTIVITY_KIND_KERNEL`). Mirrors the shape of a
/// `--cuda-graph-trace=graph` capture.
///
/// Layout:
///   - 3 `cudaGraphLaunch_v10000` runtime calls (correlationId 7100/7200/7300)
///   - 3 graph_trace rows, all `graphId=42`, `graphExecId=43`, each
///     sharing correlationId with its launch on stream 23 device 0.
///   - 1 unrelated kernel outside the graph windows (so stats sees both).
///   - 1 NVTX range `frame` covering the 3 launch windows so `--nvtx`
///     paths can be exercised; no graph rows are attributable to it
///     (graph_trace is excluded from NVTX attribution by design).
pub fn with_graph_trace() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let pid: i64 = 12345;
    let global_tid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "eager_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![2i64, "cudaGraphLaunch_v10000"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![3i64, "Graph Creation"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![4i64, "GraphExec Creation"],
    )?;

    // Three CUDA_GRAPH_EVENTS rows: 2 Graph Creations (eventClass=95)
    // at far-negative timestamps (app init) + 1 GraphExec Creation
    // (eventClass=94) for the captured graph that gets launched below.
    let graph_events: &[(i64, i64, i64, Option<i64>)] = &[
        // (eventClass, graphId, nameId, graphExecId)
        (95, 41, 3, None),     // template that's never instantiated
        (95, 42, 3, None),     // template that becomes the launched graph
        (94, 42, 4, Some(43)), // exec instantiation for graphId=42
    ];
    let init_ts: i64 = -1_000_000_000;
    for (cls, gid, name_id, gexec) in graph_events {
        conn.execute(
            "INSERT INTO CUDA_GRAPH_EVENTS \
             (start, \"end\", eventClass, globalTid, nameId, \
              graphId, originalGraphId, graphExecId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                init_ts,
                init_ts,
                *cls,
                global_tid,
                *name_id,
                *gid,
                Option::<i64>::None,
                *gexec,
            ],
        )?;
    }

    // One eager kernel outside any graph window so stats sees a kernel
    // row alongside the graph rows.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            200_000_000i64,
            201_000_000i64,
            0i32,
            1i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            8000i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    // Three (cudaGraphLaunch, graph_trace) pairs sharing correlationId.
    let launches: &[(i64, i64, i64)] = &[
        (100_000_000, 110_000_000, 7100),
        (120_000_000, 130_000_000, 7200),
        (140_000_000, 150_000_000, 7300),
    ];
    for (s, e, corr) in launches {
        // The host-side runtime API call (very brief — graph launches
        // are tens of microseconds on the host).
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![*s - 50_000, *s, global_tid, *corr, 2i64],
        )?;
        // The GPU-side graph execution row.
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE \
             (start, \"end\", deviceId, contextId, streamId, \
              correlationId, globalPid, graphId, graphExecId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, 0i32, 1i64, 23i64, *corr, 0i64, 42i64, 43i64],
        )?;
    }

    // Single NVTX range spanning the three launch windows. Lets callers
    // exercise `--nvtx 'frame'`; graph_trace rows are NVTX-opaque, so an
    // attributed scope sees the eager kernel only.
    conn.execute(
        "INSERT INTO NVTX_EVENTS \
         (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            90_000_000i64,
            210_000_000i64,
            global_tid,
            None::<i64>,
            "frame",
            0i64,
            60i64,
        ],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

/// Graph-trace fixture where two devices reuse the same raw
/// `correlationId`. Replay analysis must keep them distinct by the
/// full `(process, device, context, correlationId)` key.
pub fn graph_trace_reused_correlation_two_devices() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "cudaGraphLaunch_v10000"],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?), (?, ?, ?)",
        params![0i32, 10i64, 111i64, 1i32, 20i64, 222i64],
    )?;

    let corr = 9000i64;
    for (device, context, pid, start) in [
        (0i32, 10i64, 111i64, 100_000_000i64),
        (1i32, 20i64, 222i64, 120_000_000i64),
    ] {
        let global_tid = pid << 24;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![start - 50_000, start, global_tid, corr, 1i64],
        )?;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE \
             (start, \"end\", deviceId, contextId, streamId, \
              correlationId, globalPid, graphId, graphExecId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                start,
                start + 10_000_000,
                device,
                context,
                7i64,
                corr,
                0i64,
                42i64 + device as i64,
                100i64 + device as i64,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// CUDA-Graph-style fixture: 1 runtime event (`cudaGraphLaunch`)
/// drives `n_kernels` kernels that all share the same `correlationId`.
/// Used to regress the batched-rowid hydration in `correlate.rs`:
/// many kernels sharing one `correlationId` must hydrate in a single
/// batched prepare rather than one prepare per rowid.
pub fn cuda_graph(n_kernels: usize) -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    // pid=12345 packed into the upper bits of globalTid (skip the
    // 8-bit source-domain byte): globalTid = pid << 24.
    let pid: i64 = 12345;
    let global_tid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;

    // StringIds: 1 = "graph_kernel", 2 = "cudaGraphLaunch"
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "graph_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![2i64, "cudaGraphLaunch"],
    )?;

    let corr_id: i64 = 9000;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) \
         VALUES (?, ?, ?, ?, ?)",
        params![100_000_000i64, 100_100_000i64, global_tid, corr_id, 2i64],
    )?;

    // n_kernels kernels, all (device=0, context=1) and all sharing
    // the same correlationId — the CUDA-graph pattern.
    for i in 0..n_kernels {
        let start = 101_000_000i64 + (i as i64) * 10_000;
        let end = start + 5_000;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                start, end, 0i32, 1i64, 7i64, 1i64, 1i64, 1i64, 1i64, 1i64, 128i64, 1i64, 1i64,
                corr_id, 32i64, 0i64, 0i64, 0i64,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// GPU-metrics fixture. Two counters (`metricId=0` = `"SMs Active
/// [Throughput %]"`, `metricId=1` = `"PCIe Read Requests to BAR1
/// [Requests]"`) sampled at 1ms cadence across [100ms, 110ms]. Plus
/// the minimal CUPTI kernel table so `read_origins` resolves a primary
/// span anchored at 100ms — matches `minimal_gpu`'s origin so relative
/// `--from`/`--to` flags compose consistently across fixtures.
///
/// Counter `0` (Throughput %) values cycle 0..100 across the 10 samples
/// per counter so summary `min=0, max=90, mean=45` are easy to assert.
/// Counter `1` (Requests) values are constant `4` so `--bucket` mode's
/// `sum` aggregator produces predictable totals.
pub fn with_gpu_metrics() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE GPU_METRICS (
            rawTimestamp BIGINT,
            timestamp BIGINT,
            typeId BIGINT,
            metricId BIGINT,
            value BIGINT
        );
        CREATE TABLE TARGET_INFO_GPU_METRICS (
            typeId BIGINT,
            sourceId BIGINT,
            typeName TEXT,
            metricId BIGINT,
            metricName TEXT
        );
        "#,
    )
    .context("create with_gpu_metrics schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for with_gpu_metrics")?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    // One kernel to anchor the primary origin at 100ms so relative
    // time-window flags compose the same way as `minimal_gpu`.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    let type_id: i64 = 281479271677952; // mirrors nsys's GB202 encoding
    let counters: &[(i64, &str)] = &[
        (0, "SMs Active [Throughput %]"),
        (1, "PCIe Read Requests to BAR1 [Requests]"),
    ];
    for (metric_id, metric_name) in counters {
        conn.execute(
            "INSERT INTO TARGET_INFO_GPU_METRICS \
             (typeId, sourceId, typeName, metricId, metricName) \
             VALUES (?, ?, ?, ?, ?)",
            params![type_id, type_id, "", *metric_id, *metric_name],
        )?;
    }

    // 10 samples per counter, 1ms apart, starting at 100ms.
    // Counter 0 values: 0, 10, 20, …, 90  (min=0, max=90, mean=45)
    // Counter 1 values: 4 across the board (sum over 10 samples = 40).
    for i in 0..10i64 {
        let ts = 100_000_000i64 + i * 1_000_000;
        conn.execute(
            "INSERT INTO GPU_METRICS (rawTimestamp, timestamp, typeId, metricId, value) \
             VALUES (?, ?, ?, ?, ?)",
            params![ts, ts, type_id, 0i64, i * 10],
        )?;
        conn.execute(
            "INSERT INTO GPU_METRICS (rawTimestamp, timestamp, typeId, metricId, value) \
             VALUES (?, ?, ?, ?, ?)",
            params![ts, ts, type_id, 1i64, 4i64],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Multi-device GPU-metrics fixture. Two distinct `typeId`s — one
/// for each GPU in a hypothetical `--gpu-metrics-devices=all`
/// capture — both reporting the same `metricId=0` ("SMs Active").
/// Values are chosen so that any cross-device collapse is visible:
///
///   typeId A (device 0): 5 samples all = 10  → bucket mean 10
///   typeId B (device 1): 5 samples all = 90  → bucket mean 90
///
/// If `query_gpu_buckets` ever drops `typeId` from its GROUP BY
/// again, the two GPUs would silently merge into a single row with
/// mean = (10+90)/2 = 50 and samples = 10. The smoke test pins both
/// rows survive (value 10 vs 90, samples 5 each).
///
/// Samples land at 100ms, 101ms, …, 104ms — same anchor as
/// `with_gpu_metrics` so a 5ms bucket from origin starts at 100ms
/// and captures all 10 samples.
pub fn with_gpu_metrics_multi_device() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE GPU_METRICS (
            rawTimestamp BIGINT,
            timestamp BIGINT,
            typeId BIGINT,
            metricId BIGINT,
            value BIGINT
        );
        CREATE TABLE TARGET_INFO_GPU_METRICS (
            typeId BIGINT,
            sourceId BIGINT,
            typeName TEXT,
            metricId BIGINT,
            metricName TEXT
        );
        "#,
    )
    .context("create with_gpu_metrics_multi_device schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for with_gpu_metrics_multi_device")?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    // Two GPUs, one shared metricId. Distinct typeIds, both pointing
    // at the same metric name in the dictionary — that's how nsys
    // emits multi-device captures.
    let type_a: i64 = 281479271677952;
    let type_b: i64 = 281479271677953;
    for ty in &[type_a, type_b] {
        conn.execute(
            "INSERT INTO TARGET_INFO_GPU_METRICS \
             (typeId, sourceId, typeName, metricId, metricName) \
             VALUES (?, ?, ?, ?, ?)",
            params![*ty, *ty, "", 0i64, "SMs Active [Throughput %]"],
        )?;
    }

    // 5 samples per device, 1ms apart.
    // Device A constant 10, device B constant 90 — any cross-device
    // collapse averages to 50 and is visible in the test.
    for i in 0..5i64 {
        let ts = 100_000_000i64 + i * 1_000_000;
        conn.execute(
            "INSERT INTO GPU_METRICS (rawTimestamp, timestamp, typeId, metricId, value) \
             VALUES (?, ?, ?, ?, ?)",
            params![ts, ts, type_a, 0i64, 10i64],
        )?;
        conn.execute(
            "INSERT INTO GPU_METRICS (rawTimestamp, timestamp, typeId, metricId, value) \
             VALUES (?, ?, ?, ?, ?)",
            params![ts, ts, type_b, 0i64, 90i64],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// NIC-metrics fixture. Mirrors the NSys 2026.2 low-frequency export
/// shape captured with `nsys profile --nic-metrics=lf`:
///
/// - `NET_NIC_METRIC` holds interval samples keyed by
///   `(globalId, portId, metricsListId, metricsIdx)`.
/// - `TARGET_INFO_NETWORK_METRICS` maps `(metricsListId, metricsIdx)`
///   to counter name / description / unit.
/// - `NIC_ID_MAP` and `TARGET_INFO_NIC_INFO` map `globalId` back to a
///   human NIC id/name/GUID.
///
/// Two counters are sampled at 1ms cadence across [100ms, 110ms].
/// Values are deterministic so summary and bucket assertions can be
/// exact: bytes-sent is 0,10,...,90 and send-waits is constant 4.
pub fn with_nic_metrics() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE NET_NIC_METRIC (
            start BIGINT NOT NULL,
            "end" BIGINT NOT NULL,
            globalId BIGINT NOT NULL,
            portId BIGINT NOT NULL,
            metricsListId BIGINT NOT NULL,
            metricsIdx BIGINT NOT NULL,
            value BIGINT NOT NULL
        );
        CREATE TABLE TARGET_INFO_NETWORK_METRICS (
            metricsListId BIGINT NOT NULL,
            metricsIdx BIGINT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            unit TEXT NOT NULL
        );
        CREATE TABLE NIC_ID_MAP (
            nicId BIGINT NOT NULL,
            globalId BIGINT NOT NULL
        );
        CREATE TABLE TARGET_INFO_NIC_INFO (
            GUID BIGINT NOT NULL,
            stateName TEXT NOT NULL,
            nicId BIGINT NOT NULL,
            name TEXT NOT NULL,
            deviceId BIGINT NOT NULL,
            vendorId BIGINT NOT NULL,
            linkLayer BIGINT NOT NULL
        );
        "#,
    )
    .context("create with_nic_metrics schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for with_nic_metrics")?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    let nic_id: i64 = 0;
    let global_id: i64 = 281_474_976_710_656;
    let guid: i64 = -2_261_524_981_146_488_322;
    conn.execute(
        "INSERT INTO NIC_ID_MAP (nicId, globalId) VALUES (?, ?)",
        params![nic_id, global_id],
    )?;
    conn.execute(
        "INSERT INTO TARGET_INFO_NIC_INFO \
         (GUID, stateName, nicId, name, deviceId, vendorId, linkLayer) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            guid,
            "Local (CLI)",
            nic_id,
            "mlx5_0",
            4129i64,
            5555i64,
            2i64
        ],
    )?;

    let counters: &[(i64, &str, &str, &str)] = &[
        (
            6,
            "IB: Bytes sent",
            "Amount of InfiniBand bytes sent through the NIC port",
            "bytes/ms",
        ),
        (
            10,
            "IB: Send waits",
            "The number of ticks during which the NIC port had data to transmit but no data was sent",
            "ticks/ms",
        ),
    ];
    for (idx, name, description, unit) in counters {
        conn.execute(
            "INSERT INTO TARGET_INFO_NETWORK_METRICS \
             (metricsListId, metricsIdx, name, description, unit) \
             VALUES (?, ?, ?, ?, ?)",
            params![0i64, *idx, *name, *description, *unit],
        )?;
    }

    for i in 0..10i64 {
        let start = 100_000_000i64 + i * 1_000_000;
        let end = start + 1_000_000;
        conn.execute(
            "INSERT INTO NET_NIC_METRIC \
             (start, \"end\", globalId, portId, metricsListId, metricsIdx, value) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![start, end, global_id, 0i64, 0i64, 6i64, i * 10],
        )?;
        conn.execute(
            "INSERT INTO NET_NIC_METRIC \
             (start, \"end\", globalId, portId, metricsListId, metricsIdx, value) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![start, end, global_id, 0i64, 0i64, 10i64, 4i64],
        )?;
    }

    // Real low-frequency NIC exports can contain a zero-valued
    // bootstrap row with a huge negative start and a positive end.
    // The query layer filters invalid intervals so this row must not
    // perturb counts, spans, or bucket anchors.
    conn.execute(
        "INSERT INTO NET_NIC_METRIC \
         (start, \"end\", globalId, portId, metricsListId, metricsIdx, value) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            -1_000_000_000i64,
            50_000_000i64,
            global_id,
            0i64,
            0i64,
            6i64,
            9999i64
        ],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

/// CPU-sampling fixture. Seeds `COMPOSITE_EVENTS` +
/// `SAMPLING_CALLCHAINS` + `ENUM_SAMPLING_THREAD_STATE` +
/// `StringIds` with deterministic values that exercise every
/// cpu-sampling path:
///
/// - 4 samples on (cpu=10, globalTid=A): two leaf in `func_a`,
///   one leaf in `func_b`, one leaf unresolved in `[kernel.kallsyms]`.
/// - 2 samples on (cpu=11, globalTid=B): both leaf in `func_a` —
///   useful for `--group-by symbol` totals across CPUs.
///
/// All samples carry a 3-frame stack with the deepest frame marked
/// `[Max depth]` so `truncated_stack_share = 1.0`. The leaf frame
/// for the unresolved sample sets `unresolved=1` and `kernelMode=1`;
/// every other leaf is `unresolved=0`, `kernelMode=0`.
///
/// Samples are timestamped at 100ms..150ms in 10ms steps so a
/// relative `--from 0 --to 25ms` window captures the first 3 of 6.
pub fn with_cpu_sampling() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE COMPOSITE_EVENTS (
            id BIGINT PRIMARY KEY,
            start BIGINT,
            cpu BIGINT,
            threadState BIGINT,
            globalTid BIGINT,
            cpuCycles BIGINT
        );
        CREATE TABLE SAMPLING_CALLCHAINS (
            id BIGINT,
            symbol BIGINT,
            module BIGINT,
            kernelMode BIGINT,
            thumbCode BIGINT,
            unresolved BIGINT,
            specialEntry BIGINT,
            originalIP BIGINT,
            unwindMethod BIGINT,
            stackDepth BIGINT
        );
        CREATE TABLE ENUM_SAMPLING_THREAD_STATE (
            id BIGINT PRIMARY KEY,
            name TEXT,
            label TEXT
        );
        "#,
    )
    .context("create with_cpu_sampling schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for with_cpu_sampling")?;

    // Anchor primary origin at 100ms via one CUPTI kernel row.
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            160_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    // String dictionary entries for symbols + modules + the truncation
    // sentinel. Keep ids stable so SQL can be reasoned about.
    let strings: &[(i64, &str)] = &[
        (10, "func_a"),
        (11, "func_b"),
        (12, "<unresolved_addr_string>"),
        (13, "[Max depth]"),
        (20, "/usr/lib/libapp.so"),
        (21, "[kernel.kallsyms]"),
    ];
    for (id, val) in strings {
        conn.execute(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            params![*id, *val],
        )?;
    }
    let states: &[(i64, &str)] = &[(0, "Unknown"), (1, "Running"), (2, "Interruptible")];
    for (id, name) in states {
        conn.execute(
            "INSERT INTO ENUM_SAMPLING_THREAD_STATE (id, name, label) VALUES (?, ?, ?)",
            params![*id, *name, *name],
        )?;
    }

    let pid_a: i64 = 1234;
    let tid_a: i64 = 56;
    // NSys's globalTid layout is HW/Host (16b) | PID (24b) | Source
    // Domain (8b) | TID (16b). For OSRT samples the source domain is
    // 0x00, so `(pid << 24) | tid` simplifies to the same value as
    // the full pack. Keep tid values < 2^16 to stay faithful to the
    // real schema.
    let gtid_a: i64 = (pid_a << 24) | tid_a;
    let pid_b: i64 = 1234;
    let tid_b: i64 = 78;
    let gtid_b: i64 = (pid_b << 24) | tid_b;

    type Sample = (i64, i64, i64, i64);
    let samples: &[Sample] = &[
        (1, 100_000_000, 10, gtid_a), // leaf func_a (resolved, user)
        (2, 110_000_000, 10, gtid_a), // leaf func_a
        (3, 120_000_000, 10, gtid_a), // leaf func_b
        (4, 130_000_000, 10, gtid_a), // leaf <unresolved>@[kernel.kallsyms]
        (5, 140_000_000, 11, gtid_b), // leaf func_a (different cpu+tid)
        (6, 150_000_000, 11, gtid_b), // leaf func_a
    ];
    for (sid, ts, cpu, gtid) in samples {
        conn.execute(
            "INSERT INTO COMPOSITE_EVENTS (id, start, cpu, threadState, globalTid, cpuCycles) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![*sid, *ts, *cpu, 1i64, *gtid, 1i64],
        )?;
    }

    type Frame = (i64, i64, Option<i64>, Option<i64>, i64, i64, i64);
    // (sample_id, depth, symbol_id, module_id, kernel_mode, unresolved, originalIP)
    let frames: &[Frame] = &[
        // sample 1 — leaf func_a in libapp
        (1, 0, Some(10), Some(20), 0, 0, 0x4000_0000),
        (1, 1, Some(10), Some(20), 0, 0, 0x4000_1000),
        (1, 2, Some(13), None, 0, 0, 0),
        // sample 2 — leaf func_a
        (2, 0, Some(10), Some(20), 0, 0, 0x4000_0000),
        (2, 1, Some(10), Some(20), 0, 0, 0x4000_1000),
        (2, 2, Some(13), None, 0, 0, 0),
        // sample 3 — leaf func_b
        (3, 0, Some(11), Some(20), 0, 0, 0x4000_2000),
        (3, 1, Some(10), Some(20), 0, 0, 0x4000_1000),
        (3, 2, Some(13), None, 0, 0, 0),
        // sample 4 — leaf unresolved kernel
        (4, 0, Some(12), Some(21), 1, 1, 0x7FFF_FFFF_8000_0000_i64),
        (4, 1, Some(10), Some(20), 0, 0, 0x4000_1000),
        (4, 2, Some(13), None, 0, 0, 0),
        // samples 5, 6 — leaf func_a
        (5, 0, Some(10), Some(20), 0, 0, 0x4000_0000),
        (5, 1, Some(10), Some(20), 0, 0, 0x4000_1000),
        (5, 2, Some(13), None, 0, 0, 0),
        (6, 0, Some(10), Some(20), 0, 0, 0x4000_0000),
        (6, 1, Some(10), Some(20), 0, 0, 0x4000_1000),
        (6, 2, Some(13), None, 0, 0, 0),
    ];
    for (sid, depth, sym, modu, kmode, unres, ip) in frames {
        conn.execute(
            "INSERT INTO SAMPLING_CALLCHAINS \
             (id, symbol, module, kernelMode, thumbCode, unresolved, \
              specialEntry, originalIP, unwindMethod, stackDepth) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *sid,
                *sym,
                *modu,
                *kmode,
                0i64,
                *unres,
                Option::<i64>::None,
                *ip,
                8i64,
                *depth,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// CPU-sampling fixture *without* the optional `SAMPLING_CALLCHAINS`
/// table. Mirrors traces captured with `--cpuctxsw=true --samplefreq=N`
/// but no `--backtrace` — `COMPOSITE_EVENTS` is present but the stack
/// walk is not. One sample row keyed by `id=1` so callers can target
/// `inspect cpu_sample:1`.
pub fn with_cpu_sampling_no_callchains() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE COMPOSITE_EVENTS (
            id BIGINT PRIMARY KEY,
            start BIGINT,
            cpu BIGINT,
            threadState BIGINT,
            globalTid BIGINT,
            cpuCycles BIGINT
        );
        CREATE TABLE ENUM_SAMPLING_THREAD_STATE (
            id BIGINT PRIMARY KEY,
            name TEXT,
            label TEXT
        );
        "#,
    )
    .context("create with_cpu_sampling_no_callchains schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for with_cpu_sampling_no_callchains")?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            160_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;
    conn.execute(
        "INSERT INTO ENUM_SAMPLING_THREAD_STATE (id, name, label) VALUES (?, ?, ?)",
        params![1i64, "Running", "Running"],
    )?;
    let pid: i64 = 1234;
    let tid: i64 = 56;
    let gtid: i64 = (pid << 24) | tid;
    conn.execute(
        "INSERT INTO COMPOSITE_EVENTS (id, start, cpu, threadState, globalTid, cpuCycles) \
         VALUES (?, ?, ?, ?, ?, ?)",
        params![1i64, 100_000_000i64, 10i64, 1i64, gtid, 1i64],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

/// OSRT-API fixture. Seeds `OSRT_API` with a single `pthread_mutex_lock`
/// row plus an anchor kernel so trace-span math has a primary origin.
/// Minimal — exists so `inspect osrt:1` has something to resolve.
pub fn with_osrt() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![10i64, "pthread_mutex_lock"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            160_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    // Single OSRT call on the host thread: pthread_mutex_lock, 250us.
    // globalTid layout: (pid << 24) | tid, with pid=1234, tid=56.
    let pid: i64 = 1234;
    let tid: i64 = 56;
    let gtid: i64 = (pid << 24) | tid;
    conn.execute(
        "INSERT INTO OSRT_API (start, \"end\", globalTid, nameId) \
         VALUES (?, ?, ?, ?)",
        params![120_000_000i64, 120_250_000i64, gtid, 10i64],
    )?;

    finalize_to_pqtdir(&conn, dir)
}

/// CPU-sched fixture. Seeds `SCHED_EVENTS` +
/// `ENUM_SAMPLING_THREAD_STATE` + an anchor kernel so trace-span
/// math has a primary origin.
///
/// Layout (3 paired quanta across 2 cpus / 2 threads):
///
/// ```text
/// Thread A (gtid_a) on cpu 10:
///   in  @ 200ms  (state=Running)
///   out @ 210ms  (state=Interruptible)   <- 10ms quantum
///   in  @ 220ms  (state=Running)
///   out @ 230ms  (state=Unknown)         <- 10ms quantum
/// Thread B (gtid_b) on cpu 11:
///   in  @ 205ms  (state=Running)
///   out @ 225ms  (state=Interruptible)   <- 20ms quantum
/// ```
///
/// Aggregates the tests rely on:
///
/// - tid axis: A → on_cpu=20ms, ctx=2; B → on_cpu=20ms, ctx=1.
/// - cpu axis: cpu 10 → on_cpu=20ms (1 distinct tid); cpu 11 →
///   on_cpu=20ms (1 distinct tid).
/// - state axis: Interruptible → on_cpu=30ms (10+20), off_cpu=10ms
///   (A's out→in at 210→220), ctx_switches=2. Unknown →
///   on_cpu=10ms (A's second quantum), ctx_switches=1.
/// - per_cpu_max_gap_ns: 20ms (cpu 11 has events 20ms apart).
/// - unresolved_state_share: 1/6 (only A's last sched-out is Unknown).
pub fn with_cpu_sched() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE SCHED_EVENTS (
            start BIGINT NOT NULL,
            cpu BIGINT NOT NULL,
            isSchedIn BIGINT NOT NULL,
            globalTid BIGINT,
            threadState BIGINT,
            threadBlock BIGINT
        );
        CREATE TABLE ENUM_SAMPLING_THREAD_STATE (
            id BIGINT PRIMARY KEY,
            name TEXT,
            label TEXT
        );
        "#,
    )
    .context("create with_cpu_sched schema")?;
    conn.execute_batch(KERNEL_TABLE_SQL)
        .context("create kernel table for with_cpu_sched")?;

    // Primary-span anchor: one kernel row keeps `trace_span_ns`
    // bounded so the cpu-sched response carries meaningful coverage
    // numbers without leaning on the SCHED span itself.
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "anchor_kernel"],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            200_000_000i64,
            230_000_000i64,
            0i32,
            0i64,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            1i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    let states: &[(i64, &str)] = &[(0, "Unknown"), (1, "Running"), (2, "Interruptible")];
    for (id, name) in states {
        conn.execute(
            "INSERT INTO ENUM_SAMPLING_THREAD_STATE (id, name, label) VALUES (?, ?, ?)",
            params![*id, *name, *name],
        )?;
    }

    let pid_a: i64 = 1234;
    let tid_a: i64 = 56;
    let gtid_a: i64 = (pid_a << 24) | tid_a;
    let pid_b: i64 = 1234;
    let tid_b: i64 = 78;
    let gtid_b: i64 = (pid_b << 24) | tid_b;

    // (start_ns, cpu, isSchedIn, globalTid, threadState)
    type Sched = (i64, i64, i64, i64, i64);
    let events: &[Sched] = &[
        (200_000_000, 10, 1, gtid_a, 1), // A sched-in (Running)
        (205_000_000, 11, 1, gtid_b, 1), // B sched-in
        (210_000_000, 10, 0, gtid_a, 2), // A sched-out → Interruptible
        (220_000_000, 10, 1, gtid_a, 1), // A sched-in again
        (225_000_000, 11, 0, gtid_b, 2), // B sched-out → Interruptible
        (230_000_000, 10, 0, gtid_a, 0), // A sched-out → Unknown
    ];
    for (start, cpu, is_in, gtid, state) in events {
        conn.execute(
            "INSERT INTO SCHED_EVENTS \
             (start, cpu, isSchedIn, globalTid, threadState, threadBlock) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![*start, *cpu, *is_in, *gtid, *state, Option::<i64>::None],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// `--cuda-graph-trace=node` fixture. Captures kernels-inside-graphs
/// directly in `CUPTI_ACTIVITY_KIND_KERNEL` (with `graphId` /
/// `graphNodeId` populated) plus per-node metadata in
/// `CUDA_GRAPH_NODE_EVENTS`. There is NO
/// `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` table — node-mode replaces it.
///
/// Layout:
///   - 3 captured-graph launches: 2 nodes (1001, 1002) per replay
///     → 6 kernel rows with graphId=42.
///   - 1 eager kernel with graphId/graphNodeId NULL.
///   - 2 NODE_EVENTS rows (one per distinct graphNodeId).
///   - 3 `cudaGraphLaunch_v10000` runtime rows, all covered by one
///     NVTX range `frame`, so launch-scoped replay filtering can be
///     tested without a graph-trace table.
pub fn with_graph_nodes() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    // Node-mode captures (`--cuda-graph-trace=node`) replace GRAPH_TRACE
    // with NODE_EVENTS — the two are mutually exclusive in real NSys
    // exports, and the `capability_bit_set_when_node_events_present`
    // test pins that `has_graph_trace` stays false here.
    setup_canonical_schema_minus(&conn, &["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"])?;

    let pid: i64 = 12345;
    let global_tid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "graph_inner_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![2i64, "eager_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![3i64, "Graph Node Creation"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![4i64, "cudaGraphLaunch_v10000"],
    )?;

    let launches: &[(i64, i64)] = &[(99_950_000, 7100), (199_950_000, 7200), (299_950_000, 7300)];
    for (s, corr) in launches {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![*s, *s + 50_000i64, global_tid, *corr, 4i64],
        )?;
    }
    conn.execute(
        "INSERT INTO NVTX_EVENTS \
         (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            90_000_000i64,
            320_000_000i64,
            global_tid,
            None::<i64>,
            "frame",
            0i64,
            60i64,
        ],
    )?;

    // Two distinct nodes (1001, 1002). Each replayed 3 times → 6
    // kernel rows total. All share graphId=42 (one captured graph).
    // Node 1001 takes 5ms per replay, node 1002 takes 10ms.
    let graph_kernels: &[(i64, i64, i64, i64)] = &[
        (100_000_000, 105_000_000, 1001, 7100),
        (105_500_000, 115_500_000, 1002, 7100),
        (200_000_000, 205_000_000, 1001, 7200),
        (205_500_000, 215_500_000, 1002, 7200),
        (300_000_000, 305_000_000, 1001, 7300),
        (305_500_000, 315_500_000, 1002, 7300),
    ];
    for (s, e, node, corr) in graph_kernels {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid, \
              graphId, graphNodeId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s, *e, 0i32, 1i64, 7i64, 1i64, 1i64, 1i64, 1i64, 1i64, 128i64, 1i64, 1i64, *corr,
                32i64, 0i64, 0i64, 0i64, 42i64, *node,
            ],
        )?;
    }

    // One eager kernel (no graphId / graphNodeId) so `--group-by graph`
    // sees both populated and NULL rows.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, \
          shortName, demangledName, gridX, gridY, gridZ, \
          blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            400_000_000i64,
            402_000_000i64,
            0i32,
            1i64,
            7i64,
            2i64,
            2i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            9999i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;

    // NODE_EVENTS — one row per distinct graphNodeId. eventClass=77 is
    // "Graph Node Creation" per NSys; nameId points to StringIds[3].
    for node in [1001i64, 1002i64] {
        conn.execute(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS \
             (start, \"end\", eventClass, globalTid, nameId, graphNodeId, originalGraphNodeId) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                10_000_000i64,
                10_000_000i64,
                77i64,
                global_tid,
                3i64,
                node,
                Option::<i64>::None
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Two rank processes expose one private CUDA device each, so both
/// reuse the exact local identity `(device=0, context=1, stream=7,
/// correlationId=42)`. Their graph replays are 635ms apart. This is the
/// regression shape that used to collapse into one long replay when
/// process identity was omitted from the graph/correlation key.
pub fn process_private_cuda_identity_collision() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema_minus(&conn, &["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"])?;

    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?), (?, ?)",
        params![
            1i64,
            "rank_private_graph_kernel",
            2i64,
            "cudaGraphLaunch_v10000"
        ],
    )?;

    let corr = 42i64;
    for (pid, replay_start, range_name) in [
        (1001i64, 100_000_000i64, "rank0_step"),
        (2002i64, 735_000_000i64, "rank1_step"),
    ] {
        let global_tid = pid << 24;
        insert_cuda_context(&conn, 0, 1, pid)?;
        insert_nvtx_range(
            &conn,
            replay_start - 20_000_000,
            replay_start + 30_000_000,
            global_tid,
            "shared_step",
        )?;
        insert_nvtx_range(
            &conn,
            replay_start - 10_000_000,
            replay_start + 20_000_000,
            global_tid,
            range_name,
        )?;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                replay_start - 1_000_000,
                replay_start - 500_000,
                global_tid,
                corr,
                2i64
            ],
        )?;
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid, \
              graphId, graphNodeId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                replay_start,
                replay_start + 10_000_000,
                0i32,
                1i64,
                7i64,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                corr,
                32i64,
                0i64,
                0i64,
                0i64,
                99i64,
                100i64,
            ],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Host-API fixture used by the runtime/osrt tests. Builds a
/// trace with:
/// - 3 runtime API calls (`cudaMalloc`, `cudaMalloc_v3020`, `cudaFree`)
/// - 1 OSRT call (`read`)
///
/// No NVTX, no kernels — useful for exercising the runtime/osrt
/// surfaces (stats first-class admission, `--collapse-versioned`,
/// inspect-without-nvtx-context, null-location policy errors)
/// without unrelated rows polluting the totals.
pub fn host_api() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    let gtid: i64 = 1234 << 24;
    for (id, value) in [
        (1, "cudaMalloc"),
        (2, "cudaMalloc_v3020"),
        (3, "cudaFree"),
        (4, "read"),
    ] {
        conn.execute(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            params![id as i64, value],
        )?;
    }
    for (start, dur, name_id) in [(0i64, 10i64, 1i64), (100, 20, 2), (200, 30, 3)] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![start, start + dur, gtid, 1000_i64, name_id],
        )?;
    }
    conn.execute(
        "INSERT INTO OSRT_API (start, \"end\", globalTid, nameId) VALUES (?, ?, ?, ?)",
        params![300_i64, 350_i64, gtid, 4_i64],
    )?;
    finalize_to_pqtdir(&conn, dir)
}

/// All-five-kinds-with-NVTX fixture for `--group-by nvtx-parent`.
/// Each attributable kind has:
/// - one row inside `step_a` (an outer NVTX range)
/// - one row inside `step_b` (a sibling outer range)
/// - one row outside every range (sentinel bucket)
///
/// Layout (single globalTid, pid=12345):
///
/// ```text
///   NVTX "step_a"   [100ms..200ms]    domain=0  eventType=60
///   NVTX "step_b"   [300ms..400ms]    domain=0  eventType=60
///   runtime  corr=11  [110ms..111ms]   → inside step_a
///   runtime  corr=21  [310ms..311ms]   → inside step_b
///   runtime  corr=31  [500ms..501ms]   → outside (sentinel)
///   kernel   corr=11  [120ms..130ms]   → via corr → step_a
///   kernel   corr=21  [320ms..330ms]   → via corr → step_b
///   kernel   corr=31  [510ms..520ms]   → via corr → sentinel
///   memcpy   corr=11  [140ms..145ms]
///   memcpy   corr=21  [340ms..345ms]
///   memcpy   corr=31  [530ms..535ms]
///   memset   corr=11  [150ms..152ms]
///   memset   corr=21  [350ms..352ms]
///   memset   corr=31  [540ms..542ms]
///   sync     corr=11  [160ms..170ms]
///   sync     corr=21  [360ms..370ms]
///   sync     corr=31  [550ms..560ms]
/// ```
///
/// Each kind has exactly 3 rows split as 1-1-1 across step_a / step_b /
/// sentinel. Parity tests sum bucket totals and assert they equal the
/// trace-wide sum for that kind.
pub fn nvtx_parent_attribution() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![0i32, 1i64, pid],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "the_kernel"],
    )?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![2i64, "cudaMalloc"],
    )?;
    // NVTX ranges
    for (s, e, name) in &[
        (100_000_000i64, 200_000_000i64, "step_a"),
        (300_000_000i64, 400_000_000i64, "step_b"),
    ] {
        conn.execute(
            "INSERT INTO NVTX_EVENTS \
             (start, \"end\", globalTid, textId, text, domainId, eventType) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, gtid, None::<i64>, *name, 0i64, 60i64],
        )?;
    }

    // Runtimes (one per correlation bucket).
    let runtimes: &[(i64, i64, i64)] = &[
        (110_000_000, 111_000_000, 11),
        (310_000_000, 311_000_000, 21),
        (500_000_000, 501_000_000, 31),
    ];
    for (s, e, corr) in runtimes {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
             (start, \"end\", globalTid, correlationId, nameId) \
             VALUES (?, ?, ?, ?, ?)",
            params![*s, *e, gtid, *corr, 2i64],
        )?;
    }
    // Kernels — one per correlation.
    let kernels: &[(i64, i64, i64)] = &[
        (120_000_000, 130_000_000, 11),
        (320_000_000, 330_000_000, 21),
        (510_000_000, 520_000_000, 31),
    ];
    for (s, e, corr) in kernels {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s, *e, 0i32, 1i64, 7i64, 1i64, 1i64, 1i64, 1i64, 1i64, 128i64, 1i64, 1i64, *corr,
                32i64, 0i64, 0i64, 0i64,
            ],
        )?;
    }
    // Memcpys
    let memcpys: &[(i64, i64, i64)] = &[
        (140_000_000, 145_000_000, 11),
        (340_000_000, 345_000_000, 21),
        (530_000_000, 535_000_000, 31),
    ];
    for (s, e, corr) in memcpys {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY \
             (start, \"end\", deviceId, contextId, streamId, bytes, copyKind, correlationId, graphNodeId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, 0i32, 1i64, 7i64, 4096i64, 1i64, *corr, Option::<i64>::None],
        )?;
    }
    // Memsets
    let memsets: &[(i64, i64, i64)] = &[
        (150_000_000, 152_000_000, 11),
        (350_000_000, 352_000_000, 21),
        (540_000_000, 542_000_000, 31),
    ];
    for (s, e, corr) in memsets {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET \
             (start, \"end\", deviceId, contextId, streamId, bytes, value, correlationId, graphNodeId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, 0i32, 1i64, 7i64, 1024i64, 0i64, *corr, Option::<i64>::None],
        )?;
    }
    // Syncs
    let syncs: &[(i64, i64, i64)] = &[
        (160_000_000, 170_000_000, 11),
        (360_000_000, 370_000_000, 21),
        (550_000_000, 560_000_000, 31),
    ];
    for (s, e, corr) in syncs {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION \
             (start, \"end\", deviceId, contextId, streamId, syncType, correlationId, eventId, eventSyncId) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, 0i32, 1i64, 7i64, 3i64, *corr, Option::<i64>::None, Option::<i64>::None],
        )?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Kernels with a `mangledName` column populated — exercises the
/// `--group-by mangled` axis. Two distinct mangled
/// symbols share one shortName ("MyKernel") and one demangled
/// signature ("void MyKernel<int>(int*)") so:
/// - `--group-by short` collapses everything to one row.
/// - `--group-by demangled` collapses to one row (same demangled).
/// - `--group-by mangled` splits into two rows.
///
/// The third kernel uses a distinct shortName/demangled/mangled triple
/// so all three axes have at least two groups, isolating the
/// short-vs-demangled-vs-mangled cardinality test.
pub fn kernel_with_mangled_names() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;

    let strings: &[(i64, &str)] = &[
        (1, "MyKernel"),
        (2, "void MyKernel<int>(int*)"),
        (3, "_Z8MyKernelIiEvPi"),
        (4, "_Z8MyKernelIiEvPiS_"),
        (5, "OtherKernel"),
        (6, "void OtherKernel(float*)"),
        (7, "_Z11OtherKernelPf"),
    ];
    for (id, val) in strings {
        conn.execute(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            params![*id, *val],
        )?;
    }

    // Three kernels: rows 1+2 share shortName=1 and demangled=2 but
    // differ on mangled (3 vs 4); row 3 is fully distinct.
    let rows: &[(i64, i64, i64, i64, i64)] = &[
        (0, 1_000_000, 1, 2, 3),
        (1_000_000, 2_000_000, 1, 2, 4),
        (2_000_000, 3_500_000, 5, 6, 7),
    ];
    for (s, e, sh, dem, mng) in rows {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, mangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s, *e, 0i32, 1i64, 7i64, *sh, *dem, *mng, 1i64, 1i64, 1i64, 128i64, 1i64, 1i64,
                1i64, 32i64, 0i64, 0i64, 0i64,
            ],
        )?;
    }
    finalize_to_pqtdir(&conn, dir)
}

/// Kernels with multiple launch configurations — exercises the
/// `--group-by grid_block` axis. Four kernels:
/// - rows 1+2 share shortName=1, demangled=1, grid=(2,1,1), block=(128,1,1) (fold)
/// - row 3 shares shortName=1, demangled=1 but launches grid=(4,1,1), block=(128,1,1)
/// - row 4 shares the same name but with grid=(2,1,1), block=(256,1,1)
///
/// So:
/// - `--group-by short` → 1 row (one shortName).
/// - `--group-by short,grid_block` → 3 rows (3 distinct launch configs).
/// - `--group-by grid_block` alone → 3 rows (no name axis groups everything by shape).
pub fn kernels_with_launch_configs() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "the_kernel"],
    )?;

    // (start, end, gridX, blockX) — gridY/Z and blockY/Z stay 1.
    let kernels: &[(i64, i64, i64, i64)] = &[
        (0, 1_000_000, 2, 128),
        (1_000_000, 2_000_000, 2, 128),
        (2_000_000, 3_500_000, 4, 128),
        (3_500_000, 5_000_000, 2, 256),
    ];
    for (s, e, gx, bx) in kernels {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, \
              shortName, demangledName, gridX, gridY, gridZ, \
              blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                *s, *e, 0i32, 1i64, 7i64, 1i64, 1i64, *gx, 1i64, 1i64, *bx, 1i64, 1i64, 1i64,
                32i64, 0i64, 0i64, 0i64,
            ],
        )?;
    }
    finalize_to_pqtdir(&conn, dir)
}

/// NVTX styles fixture: same-`text` ranges with different `eventType`
/// values so `stats --type nvtx` can verify the
/// PushPop↔StartEnd split.
///
/// Layout (single globalTid, default domain):
///
/// ```text
///   "iteration"  [   0..100ms]  eventType=59  (PushPop)
///   "iteration"  [200..300ms]   eventType=60  (StartEnd)
///   "iteration"  [400..500ms]   eventType=70  (PushPop-extended)  → push_pop bucket
///   "iteration"  [600..700ms]   eventType=71  (StartEnd-extended) → start_end bucket
///   "weird"      [800..900ms]   eventType=99  (unknown/future int)
/// ```
///
/// Under default `--group-by short`, the three buckets are
/// `push_pop` (rows 1+3), `start_end` (rows 2+4), and `unknown`
/// (row 5). All five rows share the same name in two of the buckets,
/// so without the style axis they'd silently fold into one — the
/// regression this fixture pins.
pub fn nvtx_styles() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;

    setup_canonical_schema(&conn)?;
    // Anchor kernel so Trace::open finds a primary origin.
    insert_anchor_kernel(&conn, 0, 1)?;

    let gtid: i64 = 12345i64 << 24;
    let rows: &[(i64, i64, &str, i64)] = &[
        (0, 100_000_000, "iteration", 59),
        (200_000_000, 300_000_000, "iteration", 60),
        (400_000_000, 500_000_000, "iteration", 70),
        (600_000_000, 700_000_000, "iteration", 71),
        (800_000_000, 900_000_000, "weird", 99),
    ];
    for (s, e, name, et) in rows {
        conn.execute(
            "INSERT INTO NVTX_EVENTS \
             (start, \"end\", globalTid, textId, text, domainId, eventType) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![*s, *e, gtid, None::<i64>, *name, 0i64, *et],
        )?;
    }
    finalize_to_pqtdir(&conn, dir)
}

// ===========================================================================
// NVTX schema helpers
//
// Many NVTX tests build small synthetic traces sharing the same
// StringIds + TARGET_INFO_CUDA_CONTEXT_INFO + NVTX_EVENTS +
// CUPTI_ACTIVITY_KIND_RUNTIME (+ optionally KERNEL) preamble.
// Centralizing the schema here keeps it from drifting per test (e.g. a
// column added to KERNEL would otherwise have to be patched into every
// fixture that hard-coded the column list).
// ===========================================================================

/// **Deprecated** — call [`setup_canonical_schema`] instead. Kept for
/// call sites that follow it with `KERNEL_TABLE_SQL`. Creates the
/// 4-table set (StringIds + TARGET_INFO_CUDA_CONTEXT_INFO +
/// NVTX_EVENTS + RUNTIME) but not the kernel table; the canonical
/// helper creates the kernel table itself.
pub fn create_nvtx_runtime_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT, contextId BIGINT, processId BIGINT
        );
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, globalTid BIGINT,
            textId BIGINT, text TEXT, domainId BIGINT, eventType BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT,
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT
        );
        "#,
    )
    .context("create NVTX/runtime schema")
}

/// Insert one CUDA context mapping into
/// `TARGET_INFO_CUDA_CONTEXT_INFO`. Tests typically need exactly one
/// row to bridge (device, context) → process.
pub fn insert_cuda_context(
    conn: &Connection,
    device_id: i32,
    context_id: i64,
    process_id: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
         VALUES (?, ?, ?)",
        params![device_id, context_id, process_id],
    )?;
    Ok(())
}

/// Insert one NVTX range. `name` is stored inline (`text` column);
/// `domain_id` defaults to 0 in callers that don't care.
pub fn insert_nvtx_range(
    conn: &Connection,
    start_ns: i64,
    end_ns: i64,
    global_tid: i64,
    name: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO NVTX_EVENTS \
         (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![start_ns, end_ns, global_tid, None::<i64>, name, 0i64, 60i64],
    )?;
    Ok(())
}

/// Insert one runtime API row. `correlation_id = None` for runtime
/// calls without GPU work (e.g. cudaGetDeviceCount).
pub fn insert_runtime(
    conn: &Connection,
    start_ns: i64,
    end_ns: i64,
    global_tid: i64,
    correlation_id: Option<i64>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) \
         VALUES (?, ?, ?, ?, ?)",
        params![start_ns, end_ns, global_tid, correlation_id, None::<i64>],
    )?;
    Ok(())
}

/// Insert one kernel row on `(device_id, context_id, stream_id=7)`.
/// All grid/block columns default to small constants.
pub fn insert_kernel(
    conn: &Connection,
    start_ns: i64,
    end_ns: i64,
    device_id: i32,
    context_id: i64,
    correlation_id: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            start_ns,
            end_ns,
            device_id,
            context_id,
            7i64,
            1i64,
            1i64,
            1i64,
            1i64,
            1i64,
            128i64,
            1i64,
            1i64,
            correlation_id,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;
    Ok(())
}

/// Build a trace with `n` kernels, each its own correlationId and
/// runtime row, all launched inside one outer NVTX "step" range on
/// one host thread. Useful for tests that need many attributable
/// kernels (e.g. cross-`SIDECAR_BUILD_THRESHOLD` batched lookups).
pub fn many_kernels_in_nvtx(n: usize) -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    insert_cuda_context(&conn, 0, 1, pid)?;
    insert_nvtx_range(&conn, 0, (n as i64) * 100_000_000, gtid, "step")?;
    for i in 0..n {
        let base = (i as i64) * 100_000_000;
        let corr = 1000 + i as i64;
        insert_runtime(
            &conn,
            base + 10_000_000,
            base + 20_000_000,
            gtid,
            Some(corr),
        )?;
        insert_kernel(&conn, base + 30_000_000, base + 40_000_000, 0, 1, corr)?;
    }
    finalize_to_pqtdir(&conn, dir)
}

/// Build a trace with NVTX + RUNTIME but **no**
/// `TARGET_INFO_CUDA_CONTEXT_INFO` and **no** GPU activity tables.
/// The single runtime row carries `correlationId = NULL`, exercising
/// the nullable-correlation invariant.
pub fn runtime_only_with_null_correlation() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    // Drop TARGET_INFO_CUDA_CONTEXT_INFO so the verb hits the
    // "no GPU bridge → runtime-only attribution" code path. Other
    // CUPTI activity tables exist but are empty — harmless because
    // the test only inserts a runtime row with NULL correlationId,
    // and the GPU JOIN paths gate on `correlation_id IS NOT NULL`.
    setup_canonical_schema_minus(&conn, &["TARGET_INFO_CUDA_CONTEXT_INFO"])?;
    let gtid: i64 = 12345i64 << 24;
    insert_nvtx_range(&conn, 0, 1_000_000_000, gtid, "step")?;
    insert_runtime(&conn, 100_000_000, 200_000_000, gtid, None)?;
    finalize_to_pqtdir(&conn, dir)
}

/// Build a trace with nested NVTX — outer `training_step` (0..1s)
/// containing inner `fwd_pass` (100ms..900ms) — and a single kernel
/// launched inside the inner range with a matching runtime row.
pub fn nested_nvtx_with_kernel() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    insert_cuda_context(&conn, 0, 1, pid)?;
    insert_nvtx_range(&conn, 0, 1_000_000_000, gtid, "training_step")?;
    insert_nvtx_range(&conn, 100_000_000, 900_000_000, gtid, "fwd_pass")?;
    insert_runtime(&conn, 200_000_000, 210_000_000, gtid, Some(7777))?;
    insert_kernel(&conn, 220_000_000, 230_000_000, 0, 1, 7777)?;
    finalize_to_pqtdir(&conn, dir)
}

/// Two NVTX branches reuse the same leaf name (`work`) under
/// different parents. Used by tests to prove full-path
/// grouping does not collapse same-name ranges.
pub fn same_leaf_nested_nvtx_paths() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1i64, "path_kernel"],
    )?;
    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    insert_cuda_context(&conn, 0, 1, pid)?;

    let branches: &[(i64, &str, i64)] = &[(0, "phase_a", 1000), (200_000_000, "phase_b", 2000)];
    for &(base, phase, corr) in branches {
        insert_nvtx_range(&conn, base, base + 100_000_000, gtid, phase)?;
        insert_nvtx_range(&conn, base + 10_000_000, base + 90_000_000, gtid, "work")?;
        insert_runtime(
            &conn,
            base + 20_000_000,
            base + 30_000_000,
            gtid,
            Some(corr),
        )?;
        insert_kernel(&conn, base + 40_000_000, base + 50_000_000, 0, 1, corr)?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// Insert one `NvtxDomainCreate` row (eventType 75) registering a
/// domain name for `(pid_of(global_tid), domain_id)`. Used so
/// [`same_leaf_same_parent_distinct_domains`] can assert the resolved
/// `domain_name`.
pub fn insert_nvtx_domain_create(
    conn: &Connection,
    global_tid: i64,
    domain_id: i64,
    name: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO NVTX_EVENTS \
         (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            None::<i64>,
            None::<i64>,
            global_tid,
            None::<i64>,
            name,
            domain_id,
            75i64
        ],
    )?;
    Ok(())
}

/// Insert one NVTX range in an explicit domain. Mirrors
/// [`insert_nvtx_range`] but lets the caller pick `domain_id` so a
/// fixture can place same-name/same-parent ranges in distinct
/// `(pid, domainId)` domains.
pub fn insert_nvtx_range_in_domain(
    conn: &Connection,
    start_ns: i64,
    end_ns: i64,
    global_tid: i64,
    domain_id: i64,
    name: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO NVTX_EVENTS \
         (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            start_ns,
            end_ns,
            global_tid,
            None::<i64>,
            name,
            domain_id,
            60i64
        ],
    )?;
    Ok(())
}

/// Two NVTX branches reuse BOTH the same leaf name (`work`) AND the
/// same parent chain (`phase/work`), but live in DISTINCT
/// `(pid, domainId)` domains: domain 1 ("alpha", registered) and
/// domain 2 ("beta", registered), each in its own process. Used to
/// prove the domain-qualified key keeps them as two distinct rows.
pub fn same_leaf_same_parent_distinct_domains() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;

    // Two processes so the domains are genuinely distinct identities
    // (same domainId in two pids = two domains). Each
    // process runs the identical `phase/work` parent chain.
    let branches: &[(i64, i64, &str, i64, i64)] = &[
        // (pid, domain_id, domain_name, context_id, corr)
        (12345, 1, "alpha", 1, 1000),
        (67890, 2, "beta", 2, 2000),
    ];
    for &(pid, domain_id, domain_name, ctx, corr) in branches {
        let gtid: i64 = pid << 24;
        insert_cuda_context(&conn, 0, ctx, pid)?;
        insert_nvtx_domain_create(&conn, gtid, domain_id, domain_name)?;
        insert_nvtx_range_in_domain(&conn, 0, 100_000_000, gtid, domain_id, "phase")?;
        insert_nvtx_range_in_domain(&conn, 10_000_000, 90_000_000, gtid, domain_id, "work")?;
        insert_runtime(&conn, 20_000_000, 30_000_000, gtid, Some(corr))?;
        insert_kernel(&conn, 40_000_000, 50_000_000, 0, ctx, corr)?;
    }

    finalize_to_pqtdir(&conn, dir)
}

/// One kernel enclosed in a registered domain range plus one kernel
/// outside any NVTX range. `--group-by nvtx-path` must yield a domain
/// row for the enclosed kernel and a no-NVTX sentinel row that carries
/// NO domain identity.
pub fn nvtx_path_enclosed_and_sentinel() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;

    let pid: i64 = 4242;
    let gtid: i64 = pid << 24;
    insert_cuda_context(&conn, 0, 1, pid)?;
    insert_nvtx_domain_create(&conn, gtid, 1, "alpha")?;
    insert_nvtx_range_in_domain(&conn, 0, 100_000_000, gtid, 1, "work")?;
    // Enclosed kernel — correlation 1000 fires inside [0, 100ms).
    insert_runtime(&conn, 20_000_000, 30_000_000, gtid, Some(1000))?;
    insert_kernel(&conn, 40_000_000, 50_000_000, 0, 1, 1000)?;
    // Un-enclosed kernel — correlation 2000 fires after the range ends.
    insert_runtime(&conn, 200_000_000, 210_000_000, gtid, Some(2000))?;
    insert_kernel(&conn, 220_000_000, 230_000_000, 0, 1, 2000)?;

    finalize_to_pqtdir(&conn, dir)
}

/// Build a trace where two NVTX ranges share a start timestamp but
/// have different ends (outer 0..1s, inner 0..60ms). A runtime row +
/// kernel sit inside the tighter inner range. Exercises the
/// `(start ASC, end DESC, rowid ASC)` tie-break.
pub fn same_start_nested_nvtx() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    setup_canonical_schema(&conn)?;
    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    insert_cuda_context(&conn, 0, 1, pid)?;
    // Both start at 0; outer ends at 1s, inner ends at 60ms.
    insert_nvtx_range(&conn, 0, 1_000_000_000, gtid, "outer")?;
    insert_nvtx_range(&conn, 0, 60_000_000, gtid, "inner")?;
    insert_runtime(&conn, 20_000_000, 30_000_000, gtid, Some(7777))?;
    insert_kernel(&conn, 40_000_000, 50_000_000, 0, 1, 7777)?;
    finalize_to_pqtdir(&conn, dir)
}

/// Build a graph-only trace: NVTX + RUNTIME + GRAPH_TRACE, no
/// kernel/memcpy/memset/sync. Used to verify forward-attribution
/// verbs (timeline, search, stats) implicitly narrow non-attributable
/// kinds when `--nvtx` is set.
pub fn graph_only_with_nvtx() -> Result<Fixture> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
    // Drop kernel/memcpy/memset/sync so the trace is genuinely
    // "graph-only" — the test exercises the implicit-narrowing path
    // that fires when `KindFilter::All` resolves against a trace
    // that has no attributable GPU kinds present. Keeping these
    // tables (even empty) would route through the empty-result path
    // instead, masking the narrowing behaviour.
    setup_canonical_schema_minus(
        &conn,
        &[
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        ],
    )?;
    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    insert_cuda_context(&conn, 0, 1, pid)?;
    insert_nvtx_range(&conn, 0, 1_000_000_000, gtid, "step")?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE \
         (start, \"end\", deviceId, contextId, streamId, graphId, correlationId) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            100_000_000i64,
            110_000_000i64,
            0i32,
            1i64,
            7i64,
            1i64,
            7777i64
        ],
    )?;
    finalize_to_pqtdir(&conn, dir)
}
