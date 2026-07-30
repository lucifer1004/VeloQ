use anyhow::{Context, Result};
use duckdb::{Connection, params};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use tempfile::TempDir;

#[path = "../../nsys/veloq-nsys-query/tests/fixture.rs"]
mod nsys_fixture;

struct DaemonFixture {
    runtime: TempDir,
}

impl DaemonFixture {
    fn new() -> Result<Self> {
        let runtime = tempfile::tempdir().context("create isolated daemon runtime")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(runtime.path(), std::fs::Permissions::from_mode(0o700))
                .context("restrict isolated daemon runtime")?;
        }
        Ok(Self { runtime })
    }

    fn run(&self, args: &[&str]) -> Result<Output> {
        self.run_in(args, None)
    }

    fn run_in(&self, args: &[&str], cwd: Option<&Path>) -> Result<Output> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_veloq"));
        command
            .args(args)
            .env("XDG_RUNTIME_DIR", self.runtime.path());
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        command
            .output()
            .context("run veloq daemon contract command")
    }

    fn json_in(&self, args: &[&str], cwd: &Path) -> Result<(Output, Value)> {
        let output = self.run_in(args, Some(cwd))?;
        let value =
            serde_json::from_slice(&output.stdout).context("daemon command stdout must be JSON")?;
        Ok((output, value))
    }

    fn json(&self, args: &[&str]) -> Result<(Output, Value)> {
        let output = self.run(args)?;
        let value =
            serde_json::from_slice(&output.stdout).context("daemon command stdout must be JSON")?;
        Ok((output, value))
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_veloq"));
        command
            .env("XDG_RUNTIME_DIR", self.runtime.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }
}

impl Drop for DaemonFixture {
    fn drop(&mut self) {
        let _ = self.run(&["daemon", "stop", "--timeout-ms", "2000"]);
    }
}

#[test]
fn lifecycle_is_idempotent_bounded_and_reports_canonical_state() -> Result<()> {
    let fixture = DaemonFixture::new()?;

    let (status, stopped) = fixture.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert!(status.status.success());
    assert_lifecycle_envelope(&stopped, "status", "stopped")?;
    assert!(stopped.pointer("/data/auxiliary").is_some());

    let (start, ready) = fixture.json(&[
        "daemon",
        "start",
        "--timeout-ms",
        "5000",
        "--max-sessions",
        "3",
    ])?;
    assert!(
        start.status.success(),
        "daemon start failed: {}",
        String::from_utf8_lossy(&start.stderr)
    );
    assert_lifecycle_envelope(&ready, "start", "ready")?;
    assert_eq!(
        ready
            .pointer("/data/rows/0/limits/max_sessions")
            .and_then(Value::as_u64),
        Some(3)
    );
    assert!(ready.pointer("/data/auxiliary").is_none());

    let (status, running) = fixture.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert!(status.status.success());
    assert_lifecycle_envelope(&running, "status", "ready")?;
    assert_eq!(
        running
            .pointer("/data/auxiliary/sessions")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(0)
    );
    assert!(
        running
            .pointer("/data/auxiliary/evictions/sessions")
            .and_then(Value::as_u64)
            .is_some()
    );
    for field in [
        "resident_sessions",
        "resident_memory_estimate_bytes",
        "active_requests",
        "queued_requests",
        "query_workers_reserved",
        "query_memory_reserved_bytes",
        "exact_response_entries",
        "cache_hits",
        "cache_misses",
    ] {
        assert!(
            running
                .pointer(&format!("/data/rows/0/usage/{field}"))
                .and_then(Value::as_u64)
                .is_some(),
            "ready status usage must contain {field}"
        );
    }

    let (again, same_owner) = fixture.json(&[
        "daemon",
        "start",
        "--timeout-ms",
        "5000",
        "--max-sessions",
        "3",
    ])?;
    assert!(again.status.success());
    assert_eq!(
        same_owner
            .pointer("/data/rows/0/process_id")
            .and_then(Value::as_u64),
        ready
            .pointer("/data/rows/0/process_id")
            .and_then(Value::as_u64)
    );

    let (conflict, conflict_error) = fixture.json(&[
        "daemon",
        "start",
        "--timeout-ms",
        "2000",
        "--max-sessions",
        "4",
    ])?;
    assert!(!conflict.status.success());
    assert_eq!(
        conflict_error
            .pointer("/error/code")
            .and_then(Value::as_str),
        Some("daemon.config-conflict")
    );

    let (stop, stopped) = fixture.json(&["daemon", "stop", "--timeout-ms", "5000"])?;
    assert!(stop.status.success());
    assert_lifecycle_envelope(&stopped, "stop", "stopped")?;
    assert!(stopped.pointer("/data/auxiliary").is_none());

    let (again, stopped) = fixture.json(&["daemon", "stop", "--timeout-ms", "2000"])?;
    assert!(again.status.success());
    assert_lifecycle_envelope(&stopped, "stop", "stopped")
}

#[test]
fn simultaneous_equivalent_starts_converge_on_one_owner() -> Result<()> {
    let fixture = DaemonFixture::new()?;
    let args = [
        "daemon",
        "start",
        "--timeout-ms",
        "5000",
        "--max-sessions",
        "3",
    ];
    let first = fixture
        .command()
        .args(args)
        .spawn()
        .context("spawn first daemon start")?;
    let second = fixture
        .command()
        .args(args)
        .spawn()
        .context("spawn second daemon start")?;
    let first = first.wait_with_output().context("wait for first start")?;
    let second = second.wait_with_output().context("wait for second start")?;

    assert!(first.status.success());
    assert!(second.status.success());
    let first: Value =
        serde_json::from_slice(&first.stdout).context("first start stdout must be JSON")?;
    let second: Value =
        serde_json::from_slice(&second.stdout).context("second start stdout must be JSON")?;
    assert_eq!(
        first
            .pointer("/data/rows/0/process_id")
            .and_then(Value::as_u64),
        second
            .pointer("/data/rows/0/process_id")
            .and_then(Value::as_u64)
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn daemon_outlives_the_launcher_process_group() -> Result<()> {
    use std::os::unix::process::CommandExt;

    let fixture = DaemonFixture::new()?;
    let mut launcher = fixture.command();
    launcher
        .args(["daemon", "start", "--timeout-ms", "5000"])
        .process_group(0);
    let launcher = launcher.spawn().context("spawn isolated daemon launcher")?;
    let launcher_process_group = i32::try_from(launcher.id()).context("launcher PID overflow")?;
    let start = launcher
        .wait_with_output()
        .context("wait for isolated daemon launcher")?;
    assert!(
        start.status.success(),
        "daemon start failed: {}",
        String::from_utf8_lossy(&start.stderr)
    );
    let ready: Value =
        serde_json::from_slice(&start.stdout).context("daemon start stdout must be JSON")?;
    let daemon_pid = ready
        .pointer("/data/rows/0/process_id")
        .and_then(Value::as_i64)
        .and_then(|pid| i32::try_from(pid).ok())
        .context("daemon start must report a valid process ID")?;

    let daemon_process_group = unsafe { libc::getpgid(daemon_pid) };
    assert_ne!(
        daemon_process_group, -1,
        "ready daemon process must remain observable"
    );
    assert_ne!(
        daemon_process_group, launcher_process_group,
        "daemon must leave the launcher's terminal process group"
    );

    let (status, running) = fixture.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert!(status.status.success());
    assert_lifecycle_envelope(&running, "status", "ready")?;
    assert_eq!(
        running
            .pointer("/data/rows/0/process_id")
            .and_then(Value::as_i64),
        Some(i64::from(daemon_pid))
    );
    Ok(())
}

#[test]
fn routing_modes_preserve_safe_fallback_and_contextual_errors() -> Result<()> {
    let fixture = DaemonFixture::new()?;

    let (required, required_error) = fixture.json(&[
        "summary",
        "missing.nsys-rep",
        "--daemon",
        "required",
        "--daemon-connect-timeout-ms",
        "50",
    ])?;
    assert!(!required.status.success());
    assert_eq!(
        required_error
            .pointer("/error/code")
            .and_then(Value::as_str),
        Some("daemon.absent")
    );
    assert_eq!(
        required_error.get("command").and_then(Value::as_str),
        Some("nsys.summary")
    );
    assert_eq!(
        required_error
            .pointer("/trace/path")
            .and_then(Value::as_str),
        Some("missing.nsys-rep")
    );

    for mode in ["auto", "off"] {
        let (output, error) = fixture.json(&["summary", "missing.nsys-rep", "--daemon", mode])?;
        assert!(!output.status.success());
        assert_eq!(
            error.pointer("/error/code").and_then(Value::as_str),
            Some("nsys.data.trace-not-found"),
            "{mode} must preserve the independent one-shot result"
        );
    }

    let (invalid, error) =
        fixture.json(&["summary", "missing.nsys-rep", "--daemon", "sideways"])?;
    assert!(!invalid.status.success());
    assert_eq!(
        error.pointer("/error/code").and_then(Value::as_str),
        Some("daemon.invalid-config")
    );

    let (start, _) = fixture.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());
    let (required, source_error) =
        fixture.json(&["summary", "missing.nsys-rep", "--daemon", "required"])?;
    assert!(!required.status.success());
    assert_eq!(
        source_error.pointer("/error/code").and_then(Value::as_str),
        Some("nsys.data.trace-not-found")
    );

    let (required, unsupported) =
        fixture.json(&["hardware", "missing.nsys-rep", "--daemon", "required"])?;
    assert!(!required.status.success());
    assert_eq!(
        unsupported.pointer("/error/code").and_then(Value::as_str),
        Some("daemon.unsupported")
    );
    let (auto, fallback) = fixture.json(&["hardware", "missing.nsys-rep", "--daemon", "auto"])?;
    assert!(!auto.status.success());
    assert_eq!(
        fallback.pointer("/error/code").and_then(Value::as_str),
        Some("nsys.data.trace-not-found")
    );

    let required = fixture
        .command()
        .args(["summary", "missing.nsys-rep", "--daemon", "required"])
        .env("VELOQ_UNSTABLE", "occam-protocol-test")
        .output()
        .context("run environment-mismatched required query")?;
    let required_error: Value = serde_json::from_slice(&required.stdout)
        .context("environment-mismatched required stdout must be JSON")?;
    assert_eq!(
        required_error
            .pointer("/error/code")
            .and_then(Value::as_str),
        Some("daemon.incompatible")
    );

    let auto = fixture
        .command()
        .args(["summary", "missing.nsys-rep", "--daemon", "auto"])
        .env("VELOQ_UNSTABLE", "occam-protocol-test")
        .output()
        .context("run environment-mismatched auto query")?;
    let fallback: Value = serde_json::from_slice(&auto.stdout)
        .context("environment-mismatched auto stdout must be JSON")?;
    assert_eq!(
        fallback.pointer("/error/code").and_then(Value::as_str),
        Some("nsys.data.trace-not-found")
    );
    Ok(())
}

#[test]
fn daemon_summary_reuses_fresh_state_and_invalidates_replaced_evidence() -> Result<()> {
    let fixture = DaemonFixture::new()?;
    let (trace_root, trace) = build_minimal_trace()?;
    let relative_trace = trace
        .file_name()
        .and_then(|name| name.to_str())
        .context("synthetic trace path must have a UTF-8 file name")?;

    let (one_shot, expected) = fixture.json_in(
        &["summary", relative_trace, "--daemon", "off"],
        trace_root.path(),
    )?;
    assert!(one_shot.status.success());

    let (start, _) = fixture.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());
    for _ in 0..2 {
        let (daemon, actual) = fixture.json_in(
            &["summary", relative_trace, "--daemon", "required"],
            trace_root.path(),
        )?;
        assert!(daemon.status.success());
        assert_eq!(actual, expected);
    }

    let (status, snapshot) = fixture.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert!(status.status.success());
    assert_eq!(
        snapshot
            .pointer("/data/rows/0/usage/resident_sessions")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        snapshot
            .pointer("/data/rows/0/usage/exact_response_entries")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        snapshot
            .pointer("/data/rows/0/usage/cache_hits")
            .and_then(Value::as_u64),
        Some(1)
    );

    let meta_sidecar = veloq_core::artifact_dir_for(&trace).join("meta.bin");
    std::fs::write(&meta_sidecar, b"replaced sidecar")?;
    let (daemon, after_sidecar_replacement) = fixture.json_in(
        &["summary", relative_trace, "--daemon", "required"],
        trace_root.path(),
    )?;
    assert!(daemon.status.success());
    assert_eq!(after_sidecar_replacement, expected);

    let (clean, _) = fixture.json_in(&["clean", relative_trace], trace_root.path())?;
    assert!(clean.status.success());
    let (daemon, after_clean) = fixture.json_in(
        &["summary", relative_trace, "--daemon", "required"],
        trace_root.path(),
    )?;
    assert!(daemon.status.success());
    assert_eq!(after_clean, expected);

    replace_kernel_table(&trace, 300)?;
    let (one_shot, mutated_expected) = fixture.json_in(
        &["summary", relative_trace, "--daemon", "off"],
        trace_root.path(),
    )?;
    assert!(one_shot.status.success());
    assert_ne!(mutated_expected, expected);
    let (daemon, mutated_actual) = fixture.json_in(
        &["summary", relative_trace, "--daemon", "required"],
        trace_root.path(),
    )?;
    assert!(daemon.status.success());
    assert_eq!(mutated_actual, mutated_expected);

    let (status, snapshot) = fixture.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert!(status.status.success());
    assert!(
        snapshot
            .pointer("/data/auxiliary/evictions/freshness_invalidations")
            .and_then(Value::as_u64)
            .is_some_and(|count| count >= 3)
    );
    Ok(())
}

#[test]
fn cold_relative_daemon_query_uses_resolved_trace_for_trace_span_evidence() -> Result<()> {
    let fixture = DaemonFixture::new()?;
    let (trace_root, trace) = build_minimal_trace()?;
    let relative_trace = trace
        .file_name()
        .and_then(|name| name.to_str())
        .context("synthetic trace path must have a UTF-8 file name")?;
    let (start, _) = fixture.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());

    let (query, payload) = fixture.json_in(
        &["summary", relative_trace, "--daemon", "required"],
        trace_root.path(),
    )?;
    assert!(
        query.status.success(),
        "cold relative daemon query failed: {}",
        String::from_utf8_lossy(&query.stderr)
    );
    assert_eq!(
        payload.pointer("/trace/path").and_then(Value::as_str),
        Some(relative_trace)
    );
    assert!(
        payload.pointer("/trace_span/origin_ns").is_some(),
        "cold daemon output must re-read trace-span evidence through the resolved path"
    );
    Ok(())
}

#[test]
fn core_nsys_daemon_commands_match_one_shot_outputs() -> Result<()> {
    let daemon = DaemonFixture::new()?;
    let trace = nsys_fixture::nvtx_attribution()?;
    let trace_path = trace.path().to_string_lossy().into_owned();
    let commands = [
        vec!["summary", trace_path.as_str()],
        vec![
            "search",
            trace_path.as_str(),
            "--type",
            "kernel",
            "--limit",
            "1",
        ],
        vec!["inspect", trace_path.as_str(), "kernel:1"],
        vec!["correlate", trace_path.as_str(), "kernel:1"],
        vec![
            "stats",
            trace_path.as_str(),
            "--type",
            "kernel",
            "--all-devices",
            "--limit",
            "1",
        ],
        vec![
            "timeline",
            trace_path.as_str(),
            "--type",
            "kernel",
            "--interval",
            "50ms",
            "--all-devices",
            "--limit",
            "4",
        ],
        vec![
            "concurrency",
            trace_path.as_str(),
            "--all-devices",
            "--limit",
            "4",
        ],
        vec![
            "gaps",
            trace_path.as_str(),
            "--all-devices",
            "--min-duration",
            "1ms",
            "--limit",
            "4",
        ],
        vec![
            "slices",
            trace_path.as_str(),
            "--all-devices",
            "--limit",
            "4",
        ],
    ];

    let (start, _) = daemon.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());
    for command in commands {
        let mut one_shot_args = command.clone();
        one_shot_args.extend(["--daemon", "off"]);
        let one_shot = daemon.run(&one_shot_args)?;
        assert!(
            one_shot.status.success(),
            "one-shot command failed: {command:?}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&one_shot.stdout),
            String::from_utf8_lossy(&one_shot.stderr),
        );

        let mut daemon_args = command.clone();
        daemon_args.extend(["--daemon", "required"]);
        let routed = daemon.run(&daemon_args)?;
        assert!(
            routed.status.success(),
            "daemon command failed: {command:?}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&routed.stdout),
            String::from_utf8_lossy(&routed.stderr),
        );
        assert_eq!(
            routed.stdout, one_shot.stdout,
            "stdout differs for {command:?}"
        );
        assert_eq!(
            routed.stderr, one_shot.stderr,
            "stderr differs for {command:?}"
        );
    }

    for format in ["csv", "table"] {
        let commands = [
            vec![
                "search",
                trace_path.as_str(),
                "--type",
                "kernel",
                "--limit",
                "1",
            ],
            vec![
                "timeline",
                trace_path.as_str(),
                "--type",
                "kernel",
                "--interval",
                "25ms",
                "--all-devices",
                "--limit",
                "2",
            ],
            vec![
                "concurrency",
                trace_path.as_str(),
                "--all-devices",
                "--limit",
                "2",
            ],
            vec![
                "gaps",
                trace_path.as_str(),
                "--all-devices",
                "--min-duration",
                "500us",
                "--limit",
                "2",
            ],
        ];
        for command in commands {
            let one_shot = daemon.run(
                &command
                    .iter()
                    .copied()
                    .chain(["--format", format, "--daemon", "off"])
                    .collect::<Vec<_>>(),
            )?;
            let routed = daemon.run(
                &command
                    .iter()
                    .copied()
                    .chain(["--format", format, "--daemon", "required"])
                    .collect::<Vec<_>>(),
            )?;
            assert_eq!(
                routed.status.code(),
                one_shot.status.code(),
                "status differs for {command:?} ({format})"
            );
            assert_eq!(
                routed.stdout, one_shot.stdout,
                "stdout differs for {command:?} ({format})"
            );
            assert_eq!(
                routed.stderr, one_shot.stderr,
                "stderr differs for {command:?} ({format})"
            );
        }
    }
    Ok(())
}

#[test]
fn daemon_raw_stdout_and_handled_error_stderr_match_one_shot() -> Result<()> {
    let daemon = DaemonFixture::new()?;
    let (_trace_root, trace) = build_minimal_trace()?;
    let trace_path = trace.to_string_lossy().into_owned();
    let (start, _) = daemon.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());

    for command in [
        vec!["ncu-command", trace_path.as_str(), "kernel:1", "--print"],
        vec!["ncu-command", trace_path.as_str(), "runtime:1", "--print"],
    ] {
        let one_shot = daemon.run(
            &command
                .iter()
                .copied()
                .chain(["--daemon", "off"])
                .collect::<Vec<_>>(),
        )?;
        let routed = daemon.run(
            &command
                .iter()
                .copied()
                .chain(["--daemon", "required"])
                .collect::<Vec<_>>(),
        )?;
        assert_eq!(
            routed.status.code(),
            one_shot.status.code(),
            "status differs for {command:?}\none-shot stdout={}\none-shot stderr={}\nrouted stdout={}\nrouted stderr={}",
            String::from_utf8_lossy(&one_shot.stdout),
            String::from_utf8_lossy(&one_shot.stderr),
            String::from_utf8_lossy(&routed.stdout),
            String::from_utf8_lossy(&routed.stderr),
        );
        assert_eq!(routed.stdout, one_shot.stdout);
        assert_eq!(routed.stderr, one_shot.stderr);
    }
    Ok(())
}

#[test]
fn daemon_reuses_one_session_across_changing_scan_queries() -> Result<()> {
    let daemon = DaemonFixture::new()?;
    let trace = nsys_fixture::minimal_gpu()?;
    let trace_path = trace.path().to_string_lossy().into_owned();
    let sidecar = daemon.run(&[
        "gaps",
        trace_path.as_str(),
        "--all-devices",
        "--min-duration",
        "1ns",
        "--daemon",
        "off",
    ])?;
    assert!(sidecar.status.success());
    let (start, _) = daemon.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());

    let (summary, _) = daemon.json(&["summary", trace_path.as_str(), "--daemon", "required"])?;
    assert!(summary.status.success());
    let (stable_summary, _) =
        daemon.json(&["summary", trace_path.as_str(), "--daemon", "required"])?;
    assert!(stable_summary.status.success());
    let (_, snapshot) = daemon.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    let before = session_non_response_memory(&snapshot)?;
    let session_id = snapshot
        .pointer("/data/auxiliary/sessions/0/session_id")
        .and_then(Value::as_str)
        .context("daemon status must identify its resident session")?
        .to_string();

    let mut first_scan_memory = None;
    for (command_index, command) in [
        vec![
            "timeline",
            trace_path.as_str(),
            "--type",
            "kernel",
            "--interval",
            "8ms",
            "--process",
            "12345",
            "--device",
            "0",
            "--limit",
            "2",
        ],
        vec![
            "concurrency",
            trace_path.as_str(),
            "--process",
            "12345",
            "--device",
            "0",
            "--limit",
            "1",
        ],
        vec![
            "gaps",
            trace_path.as_str(),
            "--process",
            "12345",
            "--device",
            "0",
            "--min-duration",
            "100us",
            "--limit",
            "2",
        ],
    ]
    .into_iter()
    .enumerate()
    {
        let one_shot = daemon.run(
            &command
                .iter()
                .copied()
                .chain(["--daemon", "off"])
                .collect::<Vec<_>>(),
        )?;
        let output = daemon.run(
            &command
                .iter()
                .copied()
                .chain(["--daemon", "required"])
                .collect::<Vec<_>>(),
        )?;
        assert!(
            output.status.success(),
            "daemon scan query failed: {command:?}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.status.code(), one_shot.status.code());
        assert_eq!(
            output.stdout, one_shot.stdout,
            "resident scan stdout differs for {command:?}"
        );
        assert_eq!(
            output.stderr, one_shot.stderr,
            "resident scan stderr differs for {command:?}"
        );
        let (_, snapshot) = daemon.json(&["daemon", "status", "--timeout-ms", "2000"])?;
        let current = session_non_response_memory(&snapshot)?;
        assert_eq!(
            snapshot
                .pointer("/data/auxiliary/sessions/0/session_id")
                .and_then(Value::as_str),
            Some(session_id.as_str()),
            "changing scan commands must reuse the freshness-equivalent session"
        );
        if command_index == 0 {
            assert!(
                current > before,
                "the first scan query must account for retained query-engine state"
            );
            first_scan_memory = Some(current);
            let exact = daemon.run(
                &command
                    .iter()
                    .copied()
                    .chain(["--daemon", "required"])
                    .collect::<Vec<_>>(),
            )?;
            assert!(exact.status.success());
            let (_, after_exact) = daemon.json(&["daemon", "status", "--timeout-ms", "2000"])?;
            assert_eq!(
                session_non_response_memory(&after_exact)?,
                current,
                "an exact hit must not build the varying-query interval index"
            );
        } else if command_index == 1 {
            assert!(
                current > first_scan_memory.context("first scan memory missing")?,
                "the first changing scan miss must build and account the resident interval index"
            );
        }
    }

    let (_, snapshot) = daemon.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert_eq!(
        snapshot
            .pointer("/data/rows/0/usage/cache_hits")
            .and_then(Value::as_u64),
        Some(1),
        "only the deliberate exact repeat may count as a hit; changing commands remain misses"
    );
    Ok(())
}

#[test]
fn daemon_graph_replays_matches_one_shot_across_varying_queries() -> Result<()> {
    let daemon = DaemonFixture::new()?;
    let trace = nsys_fixture::with_graph_nodes()?;
    let trace_path = trace.path().to_string_lossy().into_owned();
    let (start, _) = daemon.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());

    let commands = [
        vec![
            "graph-replays",
            trace_path.as_str(),
            "--process",
            "12345",
            "--device",
            "0",
            "--sort",
            "start:asc",
            "--limit",
            "3",
        ],
        vec![
            "graph-replays",
            trace_path.as_str(),
            "--process",
            "12345",
            "--device",
            "0",
            "--from",
            "@200ms",
            "--to",
            "@216ms",
            "--sort",
            "sum:desc",
            "--top-nodes",
            "1",
            "--limit",
            "2",
        ],
        vec![
            "graph-replays",
            trace_path.as_str(),
            "--process",
            "12345",
            "--device",
            "0",
            "--nvtx",
            "frame",
            "--limit",
            "3",
        ],
    ];

    for command in &commands {
        let one_shot = daemon.run(
            &command
                .iter()
                .copied()
                .chain(["--daemon", "off"])
                .collect::<Vec<_>>(),
        )?;
        let routed = daemon.run(
            &command
                .iter()
                .copied()
                .chain(["--daemon", "required"])
                .collect::<Vec<_>>(),
        )?;
        assert!(
            one_shot.status.success(),
            "one-shot graph query failed: stdout={}\nstderr={}",
            String::from_utf8_lossy(&one_shot.stdout),
            String::from_utf8_lossy(&one_shot.stderr)
        );
        assert_eq!(routed.status.code(), one_shot.status.code());
        assert_eq!(routed.stdout, one_shot.stdout);
        assert_eq!(routed.stderr, one_shot.stderr);
    }

    let (_, before_exact) = daemon.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert_eq!(
        before_exact
            .pointer("/data/rows/0/usage/cache_hits")
            .and_then(Value::as_u64),
        Some(0)
    );

    let repeated = commands
        .last()
        .context("graph replay command set must not be empty")?;
    let repeat = daemon.run(
        &repeated
            .iter()
            .copied()
            .chain(["--daemon", "required"])
            .collect::<Vec<_>>(),
    )?;
    assert!(repeat.status.success());
    let (_, after_exact) = daemon.json(&["daemon", "status", "--timeout-ms", "2000"])?;
    assert_eq!(
        after_exact
            .pointer("/data/rows/0/usage/cache_hits")
            .and_then(Value::as_u64),
        Some(1)
    );
    Ok(())
}

#[test]
fn daemon_preserves_process_private_device_identity_and_ambiguity() -> Result<()> {
    let daemon = DaemonFixture::new()?;
    let trace = nsys_fixture::process_private_cuda_identity_collision()?;
    let trace_path = trace.path().to_string_lossy().into_owned();
    let (start, _) = daemon.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());

    let ambiguous = ["stats", trace_path.as_str(), "--device", "0"];
    let one_shot = daemon.run(
        &ambiguous
            .iter()
            .copied()
            .chain(["--daemon", "off"])
            .collect::<Vec<_>>(),
    )?;
    let routed = daemon.run(
        &ambiguous
            .iter()
            .copied()
            .chain(["--daemon", "required"])
            .collect::<Vec<_>>(),
    )?;
    assert!(!one_shot.status.success());
    assert_eq!(routed.status.code(), one_shot.status.code());
    assert_eq!(routed.stdout, one_shot.stdout);
    assert_eq!(routed.stderr, one_shot.stderr);

    let scoped = [
        "stats",
        trace_path.as_str(),
        "--process",
        "1001",
        "--device",
        "0",
    ];
    let one_shot = daemon.run(
        &scoped
            .iter()
            .copied()
            .chain(["--daemon", "off"])
            .collect::<Vec<_>>(),
    )?;
    let routed = daemon.run(
        &scoped
            .iter()
            .copied()
            .chain(["--daemon", "required"])
            .collect::<Vec<_>>(),
    )?;
    assert!(one_shot.status.success());
    assert_eq!(routed.status.code(), one_shot.status.code());
    assert_eq!(routed.stdout, one_shot.stdout);
    assert_eq!(routed.stderr, one_shot.stderr);
    Ok(())
}

#[test]
fn incompatible_live_owner_is_reported_without_replacement() -> Result<()> {
    use sysinfo::{Pid, ProcessesToUpdate, System};

    let fixture = DaemonFixture::new()?;
    let (seed_start, _) = fixture.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(seed_start.status.success());
    let root = fixture.runtime.path().join("veloq");
    let owner_path = root.join("owner-v1.json");
    let mut owner: Value =
        serde_json::from_slice(&std::fs::read(&owner_path).context("read seeded daemon owner")?)
            .context("parse seeded daemon owner")?;
    let (seed_stop, _) = fixture.json(&["daemon", "stop", "--timeout-ms", "5000"])?;
    assert!(seed_stop.status.success());

    let pid = Pid::from_u32(std::process::id());
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    let process_start_time = system
        .process(pid)
        .context("current test process must be observable")?
        .start_time();
    let owner = owner
        .as_object_mut()
        .context("seeded daemon owner must be a JSON object")?;
    owner.insert("phase".to_string(), Value::String("starting".to_string()));
    owner.insert("process_id".to_string(), Value::from(std::process::id()));
    owner.insert(
        "process_start_time".to_string(),
        Value::from(process_start_time),
    );
    owner.insert(
        "veloq_version".to_string(),
        Value::String("incompatible-test-version".to_string()),
    );
    std::fs::write(&owner_path, serde_json::to_vec(&owner)?)
        .context("write incompatible daemon owner")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&owner_path, std::fs::Permissions::from_mode(0o600))
            .context("restrict daemon owner")?;
    }

    let (status, status_payload) = fixture.json(&["daemon", "status", "--timeout-ms", "100"])?;
    assert!(status.status.success());
    assert_lifecycle_envelope(&status_payload, "status", "incompatible")?;

    let (start, start_error) = fixture.json(&["daemon", "start", "--timeout-ms", "100"])?;
    assert!(!start.status.success());
    assert_eq!(
        start_error.pointer("/error/code").and_then(Value::as_str),
        Some("daemon.incompatible")
    );

    let (auto, fallback) = fixture.json(&["summary", "missing.nsys-rep", "--daemon", "auto"])?;
    assert!(!auto.status.success());
    assert_eq!(
        fallback.pointer("/error/code").and_then(Value::as_str),
        Some("nsys.data.trace-not-found")
    );
    std::fs::remove_file(&owner_path).context("remove incompatible daemon owner")?;
    Ok(())
}

fn build_minimal_trace() -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir().context("create synthetic trace root")?;
    let connection = Connection::open_in_memory().context("open synthetic DuckDB")?;
    connection.execute_batch(
        r#"
        CREATE TABLE StringIds (id BIGINT PRIMARY KEY, value TEXT);
        CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
        CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT);
        CREATE TABLE PROCESSES (globalPid BIGINT, pid BIGINT, name TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
            deviceId BIGINT, contextId BIGINT, processId BIGINT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT,
            registersPerThread BIGINT,
            staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT,
            globalPid BIGINT,
            graphId BIGINT,
            graphNodeId BIGINT
        );
        "#,
    )?;
    connection.execute(
        "INSERT INTO StringIds (id, value) VALUES (?, ?)",
        params![1_i64, "daemon_kernel"],
    )?;
    connection.execute(
        "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?), (?, ?), (?, ?)",
        params![
            "EXPORT_SCHEMA_VERSION_MAJOR",
            "3",
            "EXPORT_SCHEMA_VERSION_MINOR",
            "0",
            "EXPORT_SCHEMA_VERSION_MICRO",
            "0",
        ],
    )?;
    connection.execute(
        "INSERT INTO PROCESSES (globalPid, pid, name) VALUES (?, ?, ?)",
        params![12345_i64, 12345_i64, "/opt/work/app"],
    )?;
    for (name, value) in [
        ("PROCESS_0:COMMAND", "/usr/bin/app"),
        ("PROCESS_0:ARGUMENT_0", "--size"),
        ("PROCESS_0:WORKING_DIR", "/workspace/case"),
    ] {
        connection.execute(
            "INSERT INTO META_DATA_CAPTURE (name, value) VALUES (?, ?)",
            params![name, value],
        )?;
    }
    connection.execute(
        "INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO \
         (deviceId, contextId, processId) VALUES (?, ?, ?)",
        params![0_i32, 1_i64, 12345_i64],
    )?;
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_i64, 200_i64, 0_i32, 1_i64, 7_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 128_i64,
            1_i64, 1_i64, 9_i64, 32_i64, 0_i64, 0_i64, 12345_i64,
        ],
    )?;

    let trace = dir.path().join("daemon_pqtdir");
    std::fs::create_dir(&trace)?;
    let mut tables = connection.prepare(
        "SELECT table_name FROM information_schema.tables \
         WHERE table_schema = 'main' AND table_type = 'BASE TABLE'",
    )?;
    let names = tables
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for name in names {
        let destination = trace.join(format!("{name}.parquet"));
        let destination = destination.to_string_lossy().replace('\'', "''");
        connection.execute_batch(&format!(
            "COPY (SELECT * FROM \"{name}\") TO '{destination}' (FORMAT PARQUET)"
        ))?;
    }
    Ok((dir, trace))
}

fn replace_kernel_table(trace: &Path, end_ns: i64) -> Result<()> {
    let connection = Connection::open_in_memory().context("open replacement DuckDB")?;
    connection.execute_batch(
        r#"
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start BIGINT, "end" BIGINT,
            deviceId BIGINT, contextId BIGINT, streamId BIGINT,
            shortName BIGINT, demangledName BIGINT,
            gridX BIGINT, gridY BIGINT, gridZ BIGINT,
            blockX BIGINT, blockY BIGINT, blockZ BIGINT,
            correlationId BIGINT,
            registersPerThread BIGINT,
            staticSharedMemory BIGINT,
            dynamicSharedMemory BIGINT,
            globalPid BIGINT,
            graphId BIGINT,
            graphNodeId BIGINT
        );
        "#,
    )?;
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL \
         (start, \"end\", deviceId, contextId, streamId, shortName, demangledName, \
          gridX, gridY, gridZ, blockX, blockY, blockZ, correlationId, \
          registersPerThread, staticSharedMemory, dynamicSharedMemory, globalPid) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            100_i64, end_ns, 0_i32, 1_i64, 7_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 128_i64,
            1_i64, 1_i64, 9_i64, 32_i64, 0_i64, 0_i64, 12345_i64,
        ],
    )?;
    let destination = trace.join("CUPTI_ACTIVITY_KIND_KERNEL.parquet");
    std::fs::remove_file(&destination)?;
    let destination = destination.to_string_lossy().replace('\'', "''");
    connection.execute_batch(&format!(
        "COPY CUPTI_ACTIVITY_KIND_KERNEL TO '{destination}' (FORMAT PARQUET)"
    ))?;
    Ok(())
}

#[test]
fn malformed_timeout_and_resource_values_are_contextual_daemon_errors() -> Result<()> {
    let fixture = DaemonFixture::new()?;
    for args in [
        vec!["daemon", "status", "--timeout-ms", "-1"],
        vec!["daemon", "start", "--max-sessions", "+1"],
        vec!["daemon", "start", "--max-concurrent-requests", "2"],
    ] {
        let (output, error) = fixture.json(&args)?;
        assert!(!output.status.success());
        assert_eq!(
            error.pointer("/error/code").and_then(Value::as_str),
            Some("daemon.invalid-config")
        );
        assert_eq!(error.get("command").and_then(Value::as_str), Some("daemon"));
    }

    let (output, error) = fixture.json(&[
        "summary",
        "missing.nsys-rep",
        "--daemon-connect-timeout-ms",
        "-1",
    ])?;
    assert!(!output.status.success());
    assert_eq!(
        error.pointer("/error/code").and_then(Value::as_str),
        Some("daemon.invalid-config")
    );
    assert_eq!(
        error.get("command").and_then(Value::as_str),
        Some("nsys.summary")
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn endpoint_and_owner_state_are_current_user_only() -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let fixture = DaemonFixture::new()?;
    let (start, _) = fixture.json(&["daemon", "start", "--timeout-ms", "5000"])?;
    assert!(start.status.success());

    let root = fixture.runtime.path().join("veloq");
    let owner_path = root.join("owner-v1.json");
    let owner: Value =
        serde_json::from_slice(&std::fs::read(&owner_path).context("read daemon owner record")?)
            .context("parse daemon owner record")?;
    let token = owner
        .get("token")
        .and_then(Value::as_str)
        .context("daemon owner record must carry a token")?;
    let socket_path = root.join(format!("daemon-v1-{token}.sock"));
    for path in [owner_path, socket_path] {
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("inspect {}", path.display()))?;
        assert_eq!(metadata.uid(), unsafe { libc::geteuid() });
        assert_eq!(
            metadata.permissions().mode() & 0o077,
            0,
            "{} must not grant group or other permissions",
            path.display()
        );
    }
    Ok(())
}

fn assert_lifecycle_envelope(value: &Value, operation: &str, state: &str) -> Result<()> {
    assert_eq!(value.get("schema").and_then(Value::as_str), Some("v1"));
    assert_eq!(
        value.pointer("/source/kind").and_then(Value::as_str),
        Some("veloq")
    );
    assert_eq!(
        value.pointer("/source/version").and_then(Value::as_str),
        Some("v1")
    );
    assert_eq!(value.get("command").and_then(Value::as_str), Some("daemon"));
    assert!(value.get("trace").is_none());
    assert!(value.get("trace_span").is_none());
    assert_eq!(
        value.pointer("/data/count").and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        value.pointer("/data/total_matched").and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        value.pointer("/data/rows/0/key").and_then(Value::as_str),
        Some("daemon|local")
    );
    assert_eq!(
        value
            .pointer("/data/rows/0/operation")
            .and_then(Value::as_str),
        Some(operation)
    );
    assert_eq!(
        value.pointer("/data/rows/0/state").and_then(Value::as_str),
        Some(state)
    );
    Ok(())
}

fn session_non_response_memory(snapshot: &Value) -> Result<u64> {
    let session = snapshot
        .pointer("/data/auxiliary/sessions/0")
        .context("daemon status must contain one resident session")?;
    let resident = session
        .get("resident_memory_estimate_bytes")
        .and_then(Value::as_u64)
        .context("session status must report resident memory")?;
    let exact = session
        .get("exact_response_bytes_estimate")
        .and_then(Value::as_u64)
        .context("session status must report exact response memory")?;
    resident
        .checked_sub(exact)
        .context("resident memory cannot be smaller than exact response memory")
}
