//! `stats --group-by nvtx-parent`:
//! - Each event attributes to its innermost enclosing NVTX range, with
//!   a visible `__no_nvtx__` sentinel for events outside every range
//! - Five per-kind parity tests assert bucket sums = trace-wide sums
//!   (kernel / memcpy / memset / sync / runtime)
//! - Composition with name + device axes preserves attribution
//! - --type nvtx + --group-by nvtx-parent rejects (self-attribute)
//! - graph / graph_node + nvtx-parent rejects (different attribution
//!   model)
//! - Missing NVTX prereq tables error early with a redirect

mod fixture;

use anyhow::{Result, anyhow, bail};
use veloq_nsys_query::stats::{GroupBy, NameAxis, StatsRequest};
use veloq_nsys_query::{EventKind, KindFilter};

fn parent_axis() -> GroupBy {
    GroupBy {
        nvtx_parent: true,
        ..Default::default()
    }
}

// ---------- per-kind parity (5 tests) --------------------------------------

fn parity_for_kind(kind: EventKind) -> Result<()> {
    let trace = fixture::nvtx_parent_attribution()?;

    let baseline = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![kind]),
            ..Default::default()
        },
    )?;

    let parent = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![kind]),
            group_by: parent_axis(),
            ..Default::default()
        },
    )?;

    assert_eq!(
        parent.total_events, baseline.total_events,
        "{kind:?}: nvtx-parent must not drop events"
    );
    assert_eq!(
        parent.total_duration_ns, baseline.total_duration_ns,
        "{kind:?}: bucket sums must equal trace-wide total"
    );
    // Every row must surface the three nvtx_parent_* fields.
    for row in &parent.rows {
        assert!(
            row.nvtx_parent_key.is_some(),
            "{kind:?}: row missing nvtx_parent_key"
        );
        assert!(
            row.nvtx_parent_name.is_some(),
            "{kind:?}: row missing nvtx_parent_name"
        );
    }
    Ok(())
}

#[test]
fn parity_kernel() -> Result<()> {
    parity_for_kind(EventKind::Kernel)
}
#[test]
fn parity_memcpy() -> Result<()> {
    parity_for_kind(EventKind::Memcpy)
}
#[test]
fn parity_memset() -> Result<()> {
    parity_for_kind(EventKind::Memset)
}
#[test]
fn parity_sync() -> Result<()> {
    parity_for_kind(EventKind::Sync)
}
#[test]
fn parity_runtime() -> Result<()> {
    parity_for_kind(EventKind::Runtime)
}

// ---------- sentinel + bucket layout ----------------------------------------

#[test]
fn kernel_splits_three_ways_with_visible_sentinel() -> Result<()> {
    // The fixture has 3 kernels: 1 inside step_a, 1 inside step_b,
    // 1 outside every range. --group-by nvtx-parent must produce
    // 3 rows with the names {step_a, step_b, __no_nvtx__}.
    let trace = fixture::nvtx_parent_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: parent_axis(),
            ..Default::default()
        },
    )?;
    assert_eq!(r.total_matched, 3);

    let names: std::collections::HashSet<_> = r
        .rows
        .iter()
        .filter_map(|row| row.nvtx_parent_name.clone())
        .collect();
    assert!(names.contains("step_a"), "got {names:?}");
    assert!(names.contains("step_b"), "got {names:?}");
    assert!(names.contains("__no_nvtx__"), "got {names:?}");

    // Sentinel row carries nvtx_parent_key=nvtx:none, depth=None.
    let sentinel = r
        .rows
        .iter()
        .find(|row| row.nvtx_parent_name.as_deref() == Some("__no_nvtx__"))
        .ok_or_else(|| anyhow!("missing sentinel row"))?;
    assert_eq!(sentinel.nvtx_parent_key.as_deref(), Some("nvtx:none"));
    assert!(sentinel.nvtx_parent_depth.is_none());

    // Real-range rows carry nvtx:<rowid> key + depth=0 (outer ranges).
    let real = r
        .rows
        .iter()
        .find(|row| row.nvtx_parent_name.as_deref() == Some("step_a"))
        .ok_or_else(|| anyhow!("missing step_a row"))?;
    assert!(
        real.nvtx_parent_key
            .as_deref()
            .is_some_and(|k| k.starts_with("nvtx:") && k != "nvtx:none"),
        "got key {:?}",
        real.nvtx_parent_key
    );
    assert_eq!(real.nvtx_parent_depth, Some(0));
    Ok(())
}

// ---------- composition: name + nvtx-parent + device -----------------------

#[test]
fn composes_with_name_axis() -> Result<()> {
    let trace = fixture::nvtx_parent_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                name: NameAxis::ShortName,
                nvtx_parent: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    // All three kernels share shortName=the_kernel, so cardinality
    // comes from the nvtx-parent axis alone.
    assert_eq!(r.total_matched, 3);
    for row in &r.rows {
        assert_eq!(row.name.as_deref(), Some("the_kernel"));
    }
    Ok(())
}

#[test]
fn composes_with_device_axis() -> Result<()> {
    let trace = fixture::nvtx_parent_attribution()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                nvtx_parent: true,
                device: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    // Single device in the fixture → 3 nvtx_parent buckets × 1 device.
    assert_eq!(r.total_matched, 3);
    for row in &r.rows {
        assert_eq!(row.device_id, Some(0));
    }
    Ok(())
}

// ---------- conflict rejects ------------------------------------------------

#[test]
fn rejects_nvtx_parent_plus_type_nvtx() -> Result<()> {
    let trace = fixture::nvtx_parent_attribution()?;
    let outcome = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Nvtx]),
            group_by: parent_axis(),
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => return Err(anyhow!("expected reject for nvtx+nvtx-parent, got Ok")),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("self-attribute") && msg.contains("nvtx-parent"),
        "got: {msg}"
    );
    Ok(())
}

#[test]
fn rejects_nvtx_parent_plus_graph() -> Result<()> {
    let trace = fixture::nvtx_parent_attribution()?;
    let outcome = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                nvtx_parent: true,
                graph: true,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => return Err(anyhow!("expected reject for nvtx-parent+graph, got Ok")),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("mutually exclusive") && msg.contains("graph"),
        "got: {msg}"
    );
    Ok(())
}

#[test]
fn rejects_nvtx_parent_plus_graph_node() -> Result<()> {
    let trace = fixture::nvtx_parent_attribution()?;
    let outcome = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                nvtx_parent: true,
                graph_node: true,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => {
            return Err(anyhow!(
                "expected reject for nvtx-parent+graph_node, got Ok"
            ));
        }
        Err(e) => e,
    };
    assert!(format!("{err:#}").contains("mutually exclusive"));
    Ok(())
}

// ---------- missing prereq ---------------------------------------------------

#[test]
fn runtime_fanout_does_not_inflate_count_or_duration() -> Result<()> {
    // P2 review guard: when the dev/ctx map fans out (two contexts
    // in the same process share a correlationId), the sidecar
    // emits multiple rows per `rt_rowid`. The runtime-side join in
    // nvtx_parent.rs joins on `rt_rowid` only, so without dedupe
    // the runtime event would be counted once per fan-out copy
    // and its duration multiplied. This test forces the fan-out
    // and asserts count = 1 / duration = the actual interval.
    use duckdb::{Connection, params};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&path)?;
    let conn = Connection::open_in_memory()?;
    fixture::setup_canonical_schema(&conn)?;
    let pid: i64 = 12345;
    let gtid: i64 = pid << 24;
    // Two contexts in the same process — both will claim
    // correlationId=42 below, forcing dev/ctx fan-out.
    for ctx_id in [1i64, 2i64] {
        conn.execute(
            "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO (deviceId, contextId, processId) \
             VALUES (?, ?, ?)",
            params![0i32, ctx_id, pid],
        )?;
    }
    conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            0i64,
            1_000_000_000i64,
            gtid,
            None::<i64>,
            "step",
            0i64,
            60i64
        ],
    )?;
    // One runtime row inside the NVTX range with correlationId=42,
    // duration 10 ms.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) VALUES (?, ?, ?, ?, ?)",
        params![100_000_000i64, 110_000_000i64, gtid, 42i64, None::<i64>],
    )?;
    // Two kernels, both with correlationId=42 but on different
    // (deviceId, contextId) pairs. dev_ctx_map will see two
    // candidates for (native_pid=pid, corr=42) → fan-out fires.
    for ctx_id in [1i64, 2i64] {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
             (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
              gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
              registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                120_000_000i64,
                130_000_000i64,
                0i32,
                ctx_id,
                7i64,
                1i64,
                1i64,
                1i64,
                1i64,
                1i64,
                128i64,
                1i64,
                1i64,
                42i64,
                32i64,
                0i64,
                0i64,
                0i64,
            ],
        )?;
    }
    {
        let mut stmt = conn.prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'")?;
        let mut rows = stmt.query([])?;
        let mut tables = Vec::new();
        while let Some(r) = rows.next()? {
            tables.push(r.get::<_, String>(0)?);
        }
        for t in &tables {
            let out = path.join(format!("{t}.parquet"));
            let out_lit = out.to_string_lossy().replace('\x27', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{t}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
    }

    let resp = veloq_nsys_query::stats::run(
        &path,
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            group_by: parent_axis(),
            ..Default::default()
        },
    )?;
    // Exactly one runtime event, attributed to "step". count must
    // be 1 (not 2 from fanout); duration must be 10 ms (not 20).
    let step_row = resp
        .rows
        .iter()
        .find(|r| r.nvtx_parent_name.as_deref() == Some("step"))
        .ok_or_else(|| anyhow!("no row for 'step'; got {:?}", resp.rows))?;
    assert_eq!(
        step_row.count, 1,
        "runtime event must not be duplicated by fan-out"
    );
    assert_eq!(
        step_row.total_ns, 10_000_000,
        "runtime duration must not be inflated by fan-out"
    );
    let _keep = dir;
    Ok(())
}

#[test]
fn default_kinds_all_on_runtime_only_trace_does_not_require_target_info() -> Result<()> {
    // `KindFilter::All` is the default. On a trace that has
    // NVTX_EVENTS + CUPTI_ACTIVITY_KIND_RUNTIME but *no* GPU
    // activity tables, the resolved kind set narrows to runtime
    // alone at SQL time — so the preflight must not demand
    // `TARGET_INFO_CUDA_CONTEXT_INFO` either. P2 review finding.
    use duckdb::{Connection, params};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&path)?;
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, globalTid BIGINT,
            textId BIGINT, text TEXT, domainId BIGINT, eventType BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT,
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT
        );
        "#,
    )?;
    let gtid: i64 = 12345i64 << 24;
    conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            0i64,
            1_000_000_000i64,
            gtid,
            None::<i64>,
            "step",
            0i64,
            60i64
        ],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) VALUES (?, ?, ?, ?, ?)",
        params![100_000_000i64, 200_000_000i64, gtid, 7777i64, None::<i64>],
    )?;
    {
        let mut stmt = conn.prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'")?;
        let mut rows = stmt.query([])?;
        let mut tables = Vec::new();
        while let Some(r) = rows.next()? {
            tables.push(r.get::<_, String>(0)?);
        }
        for t in &tables {
            let out = path.join(format!("{t}.parquet"));
            let out_lit = out.to_string_lossy().replace('\x27', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{t}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
    }

    // Note: KindFilter::All is `..Default::default()` on StatsRequest.

    let resp = veloq_nsys_query::stats::run(
        &path,
        StatsRequest {
            group_by: parent_axis(),
            ..Default::default()
        },
    )?;
    assert!(
        !resp.rows.is_empty(),
        "default --type --group-by nvtx-parent on runtime-only trace must succeed"
    );
    let _keep = dir;
    Ok(())
}

#[test]
fn runtime_only_nvtx_parent_works_without_target_info_table() -> Result<()> {
    // Runtime kind attributes via direct globalTid containment from
    // the sidecar's `rt_rowid` join; it doesn't need the
    // `(deviceId, contextId) → processId` bridge that the GPU-side
    // kinds (kernel/memcpy/memset/sync) use. So `--type runtime
    // --group-by nvtx-parent` on a trace without
    // `TARGET_INFO_CUDA_CONTEXT_INFO` must succeed, not bail.
    use duckdb::{Connection, params};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&path)?;
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, globalTid BIGINT,
            textId BIGINT, text TEXT, domainId BIGINT, eventType BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT,
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT
        );
        "#,
    )?;
    let gtid: i64 = 12345i64 << 24;
    conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            0i64,
            1_000_000_000i64,
            gtid,
            None::<i64>,
            "step",
            0i64,
            60i64
        ],
    )?;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) VALUES (?, ?, ?, ?, ?)",
        params![100_000_000i64, 200_000_000i64, gtid, 7777i64, None::<i64>],
    )?;
    {
        let mut stmt = conn.prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'")?;
        let mut rows = stmt.query([])?;
        let mut tables = Vec::new();
        while let Some(r) = rows.next()? {
            tables.push(r.get::<_, String>(0)?);
        }
        for t in &tables {
            let out = path.join(format!("{t}.parquet"));
            let out_lit = out.to_string_lossy().replace('\x27', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{t}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
    }

    let outcome = veloq_nsys_query::stats::run(
        &path,
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Runtime]),
            group_by: parent_axis(),
            ..Default::default()
        },
    );
    let resp = match outcome {
        Ok(r) => r,
        Err(e) => {
            bail!("runtime-only nvtx-parent must not require TARGET_INFO_CUDA_CONTEXT_INFO: {e:#}")
        }
    };
    // At least one row — the runtime row attributes to "step".
    assert!(
        !resp.rows.is_empty(),
        "runtime row inside NVTX 'step' should produce at least one bucket"
    );
    let _keep = dir;
    Ok(())
}

#[test]
fn missing_target_info_table_bails_for_gpu_kinds() -> Result<()> {
    // Restored after the third review round: a trace that has an
    // NVTX range + a runtime row + a kernel that genuinely SHOULD
    // attribute (matching correlationIds) but lacks
    // `TARGET_INFO_CUDA_CONTEXT_INFO` must bail with a structured
    // error rather than silently degrade. Otherwise "inside NVTX but
    // missing bridge table" is indistinguishable from "not inside
    // any NVTX range".
    //
    // The fixture deliberately inserts the runtime row + matching
    // correlationId so the would-be attribution chain exists; the
    // only thing missing is the (device, context) → process_id
    // bridge.
    use duckdb::{Connection, params};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("test_pqtdir");
    std::fs::create_dir_all(&path)?;
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start BIGINT, "end" BIGINT, globalTid BIGINT,
            textId BIGINT, text TEXT, domainId BIGINT, eventType BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start BIGINT, "end" BIGINT,
            globalTid BIGINT, correlationId BIGINT, nameId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT, registersPerThread BIGINT,
            staticSharedMemory BIGINT, dynamicSharedMemory BIGINT,
            globalPid BIGINT,
            graphId BIGINT, graphNodeId BIGINT
        );
        "#,
    )?;
    let gtid: i64 = 12345i64 << 24;
    conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", globalTid, textId, text, domainId, eventType) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        params![
            0i64,
            1_000_000_000i64,
            gtid,
            None::<i64>,
            "step",
            0i64,
            60i64
        ],
    )?;
    // Runtime row inside the NVTX range, with a real correlationId.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME \
         (start, \"end\", globalTid, correlationId, nameId) VALUES (?, ?, ?, ?, ?)",
        params![100_000_000i64, 200_000_000i64, gtid, 7777i64, None::<i64>],
    )?;
    // Kernel with the matching correlationId. Would attribute to
    // "step" if the (device, context) → process_id bridge were
    // available.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            220_000_000i64,
            230_000_000i64,
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
            7777i64,
            32i64,
            0i64,
            0i64,
            0i64,
        ],
    )?;
    {
        let mut stmt = conn.prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'")?;
        let mut rows = stmt.query([])?;
        let mut tables = Vec::new();
        while let Some(r) = rows.next()? {
            tables.push(r.get::<_, String>(0)?);
        }
        for t in &tables {
            let out = path.join(format!("{t}.parquet"));
            let out_lit = out.to_string_lossy().replace('\x27', "''");
            conn.execute(
                &format!(r#"COPY (SELECT * FROM "{t}") TO '{out_lit}' (FORMAT PARQUET)"#),
                [],
            )?;
        }
    }

    let outcome = veloq_nsys_query::stats::run(
        &path,
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: parent_axis(),
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => bail!("expected structured preflight error for missing TARGET_INFO"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("TARGET_INFO_CUDA_CONTEXT_INFO") && msg.contains("requires"),
        "expected TARGET_INFO_CUDA_CONTEXT_INFO bail; got: {msg}"
    );
    let _keep = dir;
    Ok(())
}

#[test]
fn errors_on_missing_nvtx_events_table() -> Result<()> {
    // minimal_gpu has no NVTX_EVENTS table; --group-by nvtx-parent
    // must bail with a redirect rather than emit a silent empty
    // sentinel.
    let trace = fixture::minimal_gpu()?;
    let outcome = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: parent_axis(),
            ..Default::default()
        },
    );
    let err = match outcome {
        Ok(_) => return Err(anyhow!("expected reject for missing NVTX prereq, got Ok")),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("NVTX_EVENTS") && (msg.contains("not present") || msg.contains("requires")),
        "got: {msg}"
    );
    Ok(())
}

// ---------- parity with search --with-nvtx ---------------------------------

#[test]
fn nvtx_parent_attribution_agrees_with_search_with_nvtx() -> Result<()> {
    // The same kernel row must roll up under the same NVTX range
    // whether the agent asks for it via:
    //   - stats --group-by nvtx-parent (forward attribution; this WI), or
    //   - search --with-nvtx           (reverse attribution; nvtx_reverse.rs)
    // Both walk the rank-and-pick-innermost CTE (ROW_NUMBER OVER ...
    // ORDER BY n.start DESC, WHERE rn=1). The test pins that the two
    // surfaces resolve identically on the same fixture.
    let trace = fixture::nvtx_parent_attribution()?;

    // Forward: stats with the nvtx-parent axis + the per-kernel name
    // axis so each kernel row maps to one bucket.
    let stats_resp = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            group_by: GroupBy {
                nvtx_parent: true,
                stream: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    // Reverse: search returns per-event rows decorated via --with-nvtx.
    let search_resp = veloq_nsys_query::search::run(
        trace.path(),
        veloq_nsys_query::search::SearchRequest {
            kinds: KindFilter::Only(vec![EventKind::Kernel]),
            with_nvtx: true,
            limit: 100,
            ..Default::default()
        },
    )?;

    // Sum search-side total_ns per NVTX-parent name. Sentinel events
    // (nvtx_context is None on the search side) attribute to
    // "__no_nvtx__" to mirror the stats sentinel.
    let mut by_search: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for row in &search_resp.rows {
        let base = row.base();
        let bucket = base
            .nvtx_context
            .as_ref()
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "__no_nvtx__".to_string());
        *by_search.entry(bucket).or_insert(0) += base.duration_ns;
    }
    let mut by_stats: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for row in &stats_resp.rows {
        let bucket = row
            .nvtx_parent_name
            .clone()
            .ok_or_else(|| anyhow!("nvtx_parent_name missing on a parent-axis row"))?;
        *by_stats.entry(bucket).or_insert(0) += row.total_ns;
    }
    assert_eq!(
        by_stats, by_search,
        "stats nvtx-parent buckets must equal search --with-nvtx buckets row-for-row"
    );
    Ok(())
}

// ---------- graph + osrt all-sentinel --------------------------------------

#[test]
fn graph_rows_all_land_in_sentinel_under_nvtx_parent() -> Result<()> {
    // Graph events don't carry correlationId on the graph_trace table
    // by themselves — and we explicitly don't define an attribution
    // path for them. They must land in the sentinel under
    // --group-by nvtx-parent.
    let trace = fixture::with_graph_trace()?;
    let r = veloq_nsys_query::stats::run(
        trace.path(),
        StatsRequest {
            kinds: KindFilter::Only(vec![EventKind::Graph]),
            group_by: parent_axis(),
            ..Default::default()
        },
    )?;
    for row in &r.rows {
        assert_eq!(row.nvtx_parent_name.as_deref(), Some("__no_nvtx__"));
        assert_eq!(row.nvtx_parent_key.as_deref(), Some("nvtx:none"));
    }
    Ok(())
}
