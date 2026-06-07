//! Generate an Nsight Compute command for one CUDA kernel event.
//!
//! This mirrors the practical Nsight Systems GUI handoff: pick a
//! kernel event, count earlier matching kernel launches, then rerun
//! the captured application under `ncu --kernel-name ... --launch-skip
//! ... --launch-count 1`.

use duckdb::types::Value;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use veloq_nsys_data::Trace;

use crate::query_sql::exec::SqlLabel;
use crate::{EventKind, NsysQueryError, NsysQueryResult, RowId};

const KERNEL_TABLE: &str = "CUPTI_ACTIVITY_KIND_KERNEL";
const KERNEL_COLUMN_SCAN_SQL: &str = "kernel-column scan";
const KERNEL_LOOKUP_SQL: &str = "kernel lookup";
const PRIOR_LAUNCH_COUNT_SQL: &str = "prior-launch count";
const PROCESS_COUNT_SQL: &str = "process count";
const PROCESS_NAME_SQL: &str = "process-name lookup";
const LAUNCH_RECIPE_SQL: &str = "launch-recipe scan";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum EnvPolicy {
    None,
    Safe,
    All,
}

impl EnvPolicy {
    pub fn parse(s: &str) -> NsysQueryResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "safe" => Ok(Self::Safe),
            "all" => Ok(Self::All),
            other => Err(NsysQueryError::ncu_command_unknown_env(other)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NcuCommandRequest {
    pub row_id: RowId,
    pub env_policy: EnvPolicy,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NcuCommandResponse {
    pub source_event: NcuSourceKernelEvent,
    pub launch_recipe: LaunchRecipeSummary,
    pub selector: NcuSelector,
    pub ncu: NcuInvocation,
    pub script: String,
    pub confidence: Confidence,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NcuSourceKernelEvent {
    pub row_id: RowId,
    pub short_name: Option<String>,
    pub demangled_name: Option<String>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_ns: i64,
    pub device_id: i32,
    pub context_id: i64,
    pub stream_id: i64,
    pub grid: [i64; 3],
    pub block: [i64; 3],
    pub static_shared_memory: Option<i64>,
    pub dynamic_shared_memory: Option<i64>,
    pub correlation_id: Option<i64>,
    pub global_pid: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<i64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LaunchRecipeSummary {
    pub process_index: u32,
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: String,
    pub env_policy: EnvPolicy,
    pub captured_env_count: usize,
    pub emitted_env_count: usize,
    pub redacted_env_count: usize,
    pub skipped_env_count: usize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NcuSelector {
    pub mode: String,
    pub kernel_name_base: String,
    pub kernel_name: String,
    pub launch_skip: i64,
    pub launch_count: i64,
    pub matching_launch_count_before_selected: i64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct NcuInvocation {
    pub working_dir: String,
    /// `ncu` argv only, not including `env` assignments or shell `cd`.
    pub argv: Vec<String>,
    pub export_path: String,
    /// Environment variable names emitted into the shell script.
    /// Values are present in `script`, not duplicated here.
    pub emitted_env: Vec<String>,
}

#[derive(Debug, Clone)]
struct KernelRow {
    row_id: RowId,
    start_ns: i64,
    end_ns: i64,
    device_id: i32,
    context_id: i64,
    stream_id: i64,
    short_name_id: Option<i64>,
    short_name: Option<String>,
    demangled_name_id: Option<i64>,
    demangled_name: Option<String>,
    grid: [i64; 3],
    block: [i64; 3],
    static_shared_memory: Option<i64>,
    dynamic_shared_memory: Option<i64>,
    correlation_id: Option<i64>,
    global_pid: Option<i64>,
    graph_id: Option<i64>,
    graph_node_id: Option<i64>,
}

#[derive(Debug, Clone)]
struct LaunchRecipe {
    process_index: u32,
    command: String,
    args: Vec<String>,
    working_dir: String,
    env: Vec<EnvVar>,
}

#[derive(Debug, Clone)]
struct LaunchRecipeBuilder {
    process_index: u32,
    command: Option<String>,
    args: BTreeMap<u32, String>,
    working_dir: Option<String>,
    env: Vec<EnvVar>,
}

#[derive(Debug, Clone)]
struct EnvVar {
    name: String,
    value: String,
}

#[derive(Debug)]
struct EnvSelection {
    emitted: Vec<EnvVar>,
    redacted_count: usize,
    skipped_count: usize,
}

pub fn run<P: AsRef<Path>>(path: P, req: NcuCommandRequest) -> NsysQueryResult<NcuCommandResponse> {
    if req.row_id.kind != EventKind::Kernel {
        return Err(NsysQueryError::ncu_command_row_id_kind(req.row_id));
    }

    // ncu-command is a narrow lookup verb — single kernel by rowid,
    // a few count/distinct queries, then META_DATA_CAPTURE reads.
    // All of those run against the parquetdir-backed
    // `nsight.<TABLE>` views in DuckDB.
    let trace = Trace::open(path).map_err(NsysQueryError::trace_open)?;
    if !trace.table_exists(KERNEL_TABLE) {
        return Err(NsysQueryError::NcuCommandKernelTableMissing);
    }
    if !trace.table_exists("META_DATA_CAPTURE") {
        return Err(NsysQueryError::NcuCommandMetadataTableMissing);
    }

    let kernel = selected_kernel(&trace, req.row_id)?;
    let (kernel_name_base, kernel_name, name_column, name_id) = selector_name(&kernel)?;
    let launch_skip = count_prior_matching_launches(&trace, &kernel, name_column, name_id)?;
    let same_name_processes = count_distinct_processes_for_name(&trace, name_column, name_id)?;

    let recipes = load_launch_recipes(&trace)?;
    let process_name = process_name_for_kernel(&trace, &kernel)?;
    let (recipe, mut warnings) = pick_launch_recipe(&recipes, process_name.as_deref())?;

    if kernel.graph_id.is_some() || kernel.graph_node_id.is_some() {
        warnings.push(
            "selected kernel appears to be inside a CUDA graph; NCU rerun behavior can differ from the NSys capture"
                .to_string(),
        );
    }
    if same_name_processes > 1 {
        warnings.push(format!(
            "kernel name `{kernel_name}` appears under {same_name_processes} processes in the trace; launch-skip follows NCU's name-filter semantics across target processes"
        ));
    }

    let env_selection = select_env(&recipe.env, req.env_policy);
    if env_selection.redacted_count > 0 {
        warnings.push(format!(
            "{} captured environment variable(s) were omitted because their names look sensitive",
            env_selection.redacted_count
        ));
    }
    if req.env_policy == EnvPolicy::None && !recipe.env.is_empty() {
        warnings.push(format!(
            "captured environment has {} variable(s), but --env none omits them from the generated script",
            recipe.env.len()
        ));
    }

    let export_path = format!("ncu-kernel-{}.ncu-rep", kernel.row_id.rowid);
    let argv = ncu_argv(
        &kernel_name_base,
        &kernel_name,
        launch_skip,
        &export_path,
        &recipe.command,
        &recipe.args,
    );
    let emitted_env_names: Vec<String> = env_selection
        .emitted
        .iter()
        .map(|e| e.name.clone())
        .collect();
    let script = render_script(
        &recipe.working_dir,
        &env_selection.emitted,
        &argv,
        &warnings,
    );
    let confidence = if warnings.is_empty() {
        Confidence::High
    } else if process_name.is_some() || recipes.len() == 1 {
        Confidence::Medium
    } else {
        Confidence::Low
    };

    Ok(NcuCommandResponse {
        source_event: NcuSourceKernelEvent {
            row_id: kernel.row_id,
            short_name: kernel.short_name.clone(),
            demangled_name: kernel.demangled_name.clone(),
            start_ns: kernel.start_ns,
            end_ns: kernel.end_ns,
            duration_ns: kernel.end_ns - kernel.start_ns,
            device_id: kernel.device_id,
            context_id: kernel.context_id,
            stream_id: kernel.stream_id,
            grid: kernel.grid,
            block: kernel.block,
            static_shared_memory: kernel.static_shared_memory,
            dynamic_shared_memory: kernel.dynamic_shared_memory,
            correlation_id: kernel.correlation_id,
            global_pid: kernel.global_pid,
            graph_id: kernel.graph_id,
            graph_node_id: kernel.graph_node_id,
        },
        launch_recipe: LaunchRecipeSummary {
            process_index: recipe.process_index,
            command: recipe.command.clone(),
            args: recipe.args.clone(),
            working_dir: recipe.working_dir.clone(),
            env_policy: req.env_policy,
            captured_env_count: recipe.env.len(),
            emitted_env_count: env_selection.emitted.len(),
            redacted_env_count: env_selection.redacted_count,
            skipped_env_count: env_selection.skipped_count,
        },
        selector: NcuSelector {
            mode: "name-skip".to_string(),
            kernel_name_base,
            kernel_name,
            launch_skip,
            launch_count: 1,
            matching_launch_count_before_selected: launch_skip,
        },
        ncu: NcuInvocation {
            working_dir: recipe.working_dir.clone(),
            argv,
            export_path,
            emitted_env: emitted_env_names,
        },
        script,
        confidence,
        warnings,
    })
}

fn selected_kernel(trace: &Trace, row_id: RowId) -> NsysQueryResult<KernelRow> {
    let columns = kernel_columns(trace)?;
    let smem_static = maybe_col(&columns, "staticSharedMemory");
    let smem_dyn = maybe_col(&columns, "dynamicSharedMemory");
    let corr = maybe_col(&columns, "correlationId");
    let gpid = maybe_col(&columns, "globalPid");
    let gid = maybe_col(&columns, "graphId");
    let gnid = maybe_col(&columns, "graphNodeId");
    let sql = format!(
        r#"
        SELECT
            t.start, t."end",
            CAST(t.deviceId AS INTEGER),
            CAST(t.contextId AS INTEGER),
            CAST(COALESCE(t.streamId, 0) AS INTEGER),
            CAST(t.shortName AS BIGINT),
            s_sh.value,
            CAST(t.demangledName AS BIGINT),
            s_dem.value,
            CAST(t.gridX AS INTEGER), CAST(t.gridY AS INTEGER), CAST(t.gridZ AS INTEGER),
            CAST(t.blockX AS INTEGER), CAST(t.blockY AS INTEGER), CAST(t.blockZ AS INTEGER),
            CAST({smem_static} AS BIGINT),
            CAST({smem_dyn} AS BIGINT),
            CAST({corr} AS BIGINT),
            CAST({gpid} AS BIGINT),
            CAST({gid} AS BIGINT),
            CAST({gnid} AS BIGINT)
        FROM nsight."{KERNEL_TABLE}" t
        LEFT JOIN nsight.StringIds s_sh  ON t.shortName = s_sh.id
        LEFT JOIN nsight.StringIds s_dem ON t.demangledName = s_dem.id
        WHERE t.rowid = ?
        "#
    );
    hydrate_selected_kernel(trace.conn(), row_id, &sql)
}

fn hydrate_selected_kernel(
    conn: &duckdb::Connection,
    row_id: RowId,
    sql: &str,
) -> NsysQueryResult<KernelRow> {
    let params = [Value::BigInt(row_id.rowid)];
    let Some(row) = query_ncu_optional_row(conn, sql, &params, KERNEL_LOOKUP_SQL, |r| {
        hydrate_kernel_row(row_id, r)
    })?
    else {
        return Err(NsysQueryError::ncu_command_kernel_not_found(row_id));
    };
    Ok(row)
}

fn hydrate_kernel_row(row_id: RowId, r: &duckdb::Row<'_>) -> Result<KernelRow, duckdb::Error> {
    let start_ns: i64 = r.get(0)?;
    let end_ns: i64 = r.get(1)?;
    Ok(KernelRow {
        row_id,
        start_ns,
        end_ns,
        device_id: r.get(2)?,
        context_id: r.get(3)?,
        stream_id: r.get(4)?,
        short_name_id: r.get(5)?,
        short_name: opt_string(r, 6)?,
        demangled_name_id: r.get(7)?,
        demangled_name: opt_string(r, 8)?,
        grid: [r.get(9)?, r.get(10)?, r.get(11)?],
        block: [r.get(12)?, r.get(13)?, r.get(14)?],
        static_shared_memory: r.get(15)?,
        dynamic_shared_memory: r.get(16)?,
        correlation_id: r.get(17)?,
        global_pid: r.get(18)?,
        graph_id: r.get(19)?,
        graph_node_id: r.get(20)?,
    })
}

fn query_ncu_rows<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    query: &'static str,
    hydrate: impl FnMut(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Vec<T>> {
    crate::query_sql::exec::query_rows(
        conn,
        sql,
        params,
        SqlLabel::new("ncu-command", query),
        hydrate,
    )
}

fn query_ncu_optional_row<T>(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    query: &'static str,
    hydrate: impl FnOnce(&duckdb::Row<'_>) -> Result<T, duckdb::Error>,
) -> NsysQueryResult<Option<T>> {
    crate::query_sql::exec::query_optional_row(
        conn,
        sql,
        params,
        SqlLabel::new("ncu-command", query),
        hydrate,
    )
}

fn selector_name(kernel: &KernelRow) -> NsysQueryResult<(String, String, &'static str, i64)> {
    if let (Some(id), Some(name)) = (kernel.short_name_id, kernel.short_name.clone()) {
        return Ok(("function".to_string(), name, "shortName", id));
    }
    if let (Some(id), Some(name)) = (kernel.demangled_name_id, kernel.demangled_name.clone()) {
        return Ok(("demangled".to_string(), name, "demangledName", id));
    }
    Err(NsysQueryError::ncu_command_kernel_name_missing(
        kernel.row_id,
    ))
}

fn count_prior_matching_launches(
    trace: &Trace,
    kernel: &KernelRow,
    name_column: &str,
    name_id: i64,
) -> NsysQueryResult<i64> {
    let sql = format!(
        r#"
        SELECT COUNT(*)
        FROM nsight."{KERNEL_TABLE}"
        WHERE {name_column} = ?
          AND (start < ? OR (start = ? AND rowid < ?))
        "#
    );
    query_single_i64(
        trace.conn(),
        &sql,
        &[
            Value::BigInt(name_id),
            Value::BigInt(kernel.start_ns),
            Value::BigInt(kernel.start_ns),
            Value::BigInt(kernel.row_id.rowid),
        ],
        PRIOR_LAUNCH_COUNT_SQL,
    )
}

fn count_distinct_processes_for_name(
    trace: &Trace,
    name_column: &str,
    name_id: i64,
) -> NsysQueryResult<i64> {
    let columns = kernel_columns(trace)?;
    if !columns.iter().any(|c| c == "globalPid") {
        return Ok(0);
    }
    let sql = format!(
        r#"
        SELECT COUNT(DISTINCT globalPid)
        FROM nsight."{KERNEL_TABLE}"
        WHERE {name_column} = ? AND globalPid IS NOT NULL
        "#
    );
    query_single_i64(
        trace.conn(),
        &sql,
        &[Value::BigInt(name_id)],
        PROCESS_COUNT_SQL,
    )
}

fn process_name_for_kernel(trace: &Trace, kernel: &KernelRow) -> NsysQueryResult<Option<String>> {
    let Some(global_pid) = kernel.global_pid else {
        return Ok(None);
    };
    if !trace.table_exists("PROCESSES") {
        return Ok(None);
    }
    let params = [Value::BigInt(global_pid)];
    let row = query_ncu_optional_row(
        trace.conn(),
        "SELECT name FROM nsight.PROCESSES WHERE globalPid = ? LIMIT 1",
        &params,
        PROCESS_NAME_SQL,
        |r| r.get::<_, Option<String>>(0),
    )?;
    Ok(row.flatten())
}

fn load_launch_recipes(trace: &Trace) -> NsysQueryResult<Vec<LaunchRecipe>> {
    load_launch_recipes_with_sql(
        trace.conn(),
        "SELECT name, value FROM nsight.META_DATA_CAPTURE \
         WHERE name LIKE 'PROCESS\\_%:%' ESCAPE '\\'",
    )
}

fn load_launch_recipes_with_sql(
    conn: &duckdb::Connection,
    sql: &str,
) -> NsysQueryResult<Vec<LaunchRecipe>> {
    let rows = query_ncu_rows(conn, sql, &[], LAUNCH_RECIPE_SQL, |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    let mut builders: BTreeMap<u32, LaunchRecipeBuilder> = BTreeMap::new();
    for (key, value) in rows {
        let Some((idx, field)) = split_process_key(&key) else {
            continue;
        };
        let builder = builders
            .entry(idx)
            .or_insert_with(|| LaunchRecipeBuilder::new(idx));
        match field {
            "COMMAND" => builder.command = Some(value),
            "WORKING_DIR" => builder.working_dir = Some(value),
            "ENVIRONMENT_VARIABLE" => {
                if let Some(env) = parse_env_assignment(&value) {
                    builder.env.push(env);
                }
            }
            other => {
                if let Some(arg_idx) = parse_argument_field(other) {
                    builder.args.insert(arg_idx, value);
                }
            }
        }
    }

    let mut out = Vec::new();
    for (_, b) in builders {
        if let Some(command) = b.command {
            let args: Vec<String> = b.args.into_values().collect();
            out.push(LaunchRecipe {
                process_index: b.process_index,
                command,
                args,
                working_dir: b.working_dir.unwrap_or_else(|| ".".to_string()),
                env: b.env,
            });
        }
    }
    if out.is_empty() {
        return Err(NsysQueryError::NcuCommandLaunchRecipeMissing);
    }
    Ok(out)
}

fn query_single_i64(
    conn: &duckdb::Connection,
    sql: &str,
    params: &[Value],
    query: &'static str,
) -> NsysQueryResult<i64> {
    query_ncu_optional_row(conn, sql, params, query, |row| row.get(0))?
        .ok_or(NsysQueryError::InternalNcuCommandSqlRowMissing { query })
}

fn pick_launch_recipe(
    recipes: &[LaunchRecipe],
    process_name: Option<&str>,
) -> NsysQueryResult<(LaunchRecipe, Vec<String>)> {
    let mut warnings = Vec::new();
    if let Some(name) = process_name {
        let matches: Vec<&LaunchRecipe> = recipes
            .iter()
            .filter(|r| commands_match_process(&r.command, name))
            .collect();
        if matches.len() == 1 {
            let Some(recipe) = matches.first() else {
                return Err(
                    NsysQueryError::InternalNcuCommandLaunchRecipeSelectionMissing {
                        selector: "process match",
                    },
                );
            };
            return Ok(((*recipe).clone(), warnings));
        }
        if matches.len() > 1 {
            return Err(NsysQueryError::ncu_command_ambiguous_process_recipe(name));
        }
        warnings.push(format!(
            "kernel process `{name}` did not match any META_DATA_CAPTURE command; falling back to launch recipe count"
        ));
    }

    if recipes.len() == 1 {
        let Some(recipe) = recipes.first() else {
            return Err(
                NsysQueryError::InternalNcuCommandLaunchRecipeSelectionMissing {
                    selector: "single recipe",
                },
            );
        };
        return Ok((recipe.clone(), warnings));
    }

    Err(NsysQueryError::NcuCommandAmbiguousLaunchRecipe)
}

fn ncu_argv(
    kernel_name_base: &str,
    kernel_name: &str,
    launch_skip: i64,
    export_path: &str,
    command: &str,
    args: &[String],
) -> Vec<String> {
    let mut argv = vec![
        "ncu".to_string(),
        "--kernel-name-base".to_string(),
        kernel_name_base.to_string(),
        "--kernel-name".to_string(),
        kernel_name.to_string(),
        "--launch-skip".to_string(),
        launch_skip.to_string(),
        "--launch-count".to_string(),
        "1".to_string(),
        "--target-processes".to_string(),
        "all".to_string(),
        "--export".to_string(),
        export_path.to_string(),
        "--force-overwrite".to_string(),
        "--".to_string(),
        command.to_string(),
    ];
    argv.extend(args.iter().cloned());
    argv
}

fn render_script(
    working_dir: &str,
    env: &[EnvVar],
    argv: &[String],
    warnings: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("#!/usr/bin/env bash\n");
    out.push_str("set -euo pipefail\n");
    for warning in warnings {
        out.push_str("# WARNING: ");
        out.push_str(&warning.replace('\n', " "));
        out.push('\n');
    }
    out.push_str("cd -- ");
    out.push_str(&shell_quote(working_dir));
    out.push_str("\n\n");

    let mut words = Vec::new();
    words.push("exec".to_string());
    if !env.is_empty() {
        words.push("env".to_string());
        for var in env {
            words.push(format!("{}={}", var.name, var.value));
        }
    }
    words.extend(argv.iter().cloned());
    out.push_str(&shell_command_lines(&words));
    out.push('\n');
    out
}

fn shell_command_lines(words: &[String]) -> String {
    let mut out = String::new();
    for (idx, word) in words.iter().enumerate() {
        if idx == 0 {
            out.push_str(&shell_quote(word));
        } else {
            out.push_str(" \\\n  ");
            out.push_str(&shell_quote(word));
        }
    }
    out
}

fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, '_' | '-' | '.' | '/' | ':' | '@' | '%' | '+' | '=')
        })
    {
        return s.to_string();
    }
    let mut out = String::from("'");
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn select_env(vars: &[EnvVar], policy: EnvPolicy) -> EnvSelection {
    let mut emitted = Vec::new();
    let mut redacted_count = 0usize;
    let mut skipped_count = 0usize;
    for var in vars {
        if !valid_env_name(&var.name) {
            skipped_count += 1;
            continue;
        }
        if is_sensitive_env(&var.name) {
            redacted_count += 1;
            continue;
        }
        let keep = match policy {
            EnvPolicy::None => false,
            EnvPolicy::Safe => is_safe_env(&var.name),
            EnvPolicy::All => true,
        };
        if keep {
            emitted.push(var.clone());
        } else {
            skipped_count += 1;
        }
    }
    EnvSelection {
        emitted,
        redacted_count,
        skipped_count,
    }
}

fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn is_sensitive_env(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    [
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "COOKIE",
        "AUTH",
        "API_KEY",
    ]
    .iter()
    .any(|needle| upper.contains(needle))
}

fn is_safe_env(name: &str) -> bool {
    matches!(
        name,
        "CUDA_VISIBLE_DEVICES"
            | "NVIDIA_VISIBLE_DEVICES"
            | "NVIDIA_DRIVER_CAPABILITIES"
            | "CUDA_HOME"
            | "PATH"
            | "LD_LIBRARY_PATH"
            | "LIBRARY_PATH"
            | "RANK"
            | "LOCAL_RANK"
            | "WORLD_SIZE"
            | "MASTER_ADDR"
            | "MASTER_PORT"
    ) || name.starts_with("NCCL_")
        || name.starts_with("UCX_")
        || name.starts_with("NVSHMEM_")
}

fn split_process_key(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("PROCESS_")?;
    let (idx, field) = rest.split_once(':')?;
    let idx = idx.parse::<u32>().ok()?;
    Some((idx, field))
}

fn parse_argument_field(field: &str) -> Option<u32> {
    field.strip_prefix("ARGUMENT_")?.parse::<u32>().ok()
}

fn parse_env_assignment(raw: &str) -> Option<EnvVar> {
    let (name, value) = raw.split_once('=')?;
    if name.is_empty() {
        return None;
    }
    Some(EnvVar {
        name: name.to_string(),
        value: unquote_nsys_value(value),
    })
}

fn unquote_nsys_value(value: &str) -> String {
    let Some(inner) = value.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
        return value.to_string();
    };
    let mut out = String::with_capacity(inner.len());
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            out.push(match ch {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                other => other,
            });
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        out.push('\\');
    }
    out
}

fn commands_match_process(command: &str, process_name: &str) -> bool {
    command == process_name || basename(command) == basename(process_name)
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

impl LaunchRecipeBuilder {
    fn new(process_index: u32) -> Self {
        Self {
            process_index,
            command: None,
            args: BTreeMap::new(),
            working_dir: None,
            env: Vec::new(),
        }
    }
}

fn kernel_columns(trace: &Trace) -> NsysQueryResult<Vec<String>> {
    // DuckDB's `DESCRIBE` returns the view's columns; sanitisation:
    // KERNEL_TABLE is a compile-time constant so interpolation is safe.
    let sql = format!(r#"DESCRIBE nsight."{KERNEL_TABLE}""#);
    load_kernel_columns(trace.conn(), &sql)
}

fn load_kernel_columns(conn: &duckdb::Connection, sql: &str) -> NsysQueryResult<Vec<String>> {
    query_ncu_rows(conn, sql, &[], KERNEL_COLUMN_SCAN_SQL, |r| r.get(0))
}

fn maybe_col(cols: &[String], col: &str) -> String {
    if cols.iter().any(|c| c == col) {
        format!("t.\"{col}\"")
    } else {
        "NULL".to_string()
    }
}

fn opt_string(row: &duckdb::Row<'_>, idx: usize) -> duckdb::Result<Option<String>> {
    row.get::<_, Option<String>>(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use veloq_core::VeloqDiagnostic;

    fn kernel_row_id() -> RowId {
        RowId::new(EventKind::Kernel, 7)
    }

    fn selected_kernel_hydration_sql(start_expr: &str) -> String {
        format!(
            "SELECT \
             {start_expr} AS start_ns, \
             20::BIGINT AS end_ns, \
             0::INTEGER AS device_id, \
             1::BIGINT AS context_id, \
             2::BIGINT AS stream_id, \
             3::BIGINT AS short_name_id, \
             'short' AS short_name, \
             4::BIGINT AS demangled_name_id, \
             'demangled' AS demangled_name, \
             1::BIGINT AS grid_x, \
             1::BIGINT AS grid_y, \
             1::BIGINT AS grid_z, \
             32::BIGINT AS block_x, \
             1::BIGINT AS block_y, \
             1::BIGINT AS block_z, \
             CAST(NULL AS BIGINT) AS static_shared_memory, \
             CAST(NULL AS BIGINT) AS dynamic_shared_memory, \
             CAST(NULL AS BIGINT) AS correlation_id, \
             CAST(NULL AS BIGINT) AS global_pid, \
             CAST(NULL AS BIGINT) AS graph_id, \
             CAST(NULL AS BIGINT) AS graph_node_id \
             WHERE ? IS NOT NULL"
        )
    }

    #[test]
    fn shell_quote_handles_spaces_and_single_quotes() {
        assert_eq!(shell_quote("/tmp/simple"), "/tmp/simple");
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
    }

    #[test]
    fn env_assignment_unquotes_nsys_values() -> Result<()> {
        let env = parse_env_assignment("CUDA_VISIBLE_DEVICES=\"0,1\"")
            .ok_or_else(|| anyhow::anyhow!("expected parsed env"))?;
        assert_eq!(env.name, "CUDA_VISIBLE_DEVICES");
        assert_eq!(env.value, "0,1");
        Ok(())
    }

    #[test]
    fn load_kernel_columns_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match load_kernel_columns(&conn, "SELECT * FROM") {
            Ok(cols) => anyhow::bail!(
                "malformed kernel-column SQL should not hydrate successfully: {cols:?}"
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some((
                "ncu-command",
                crate::SqlPhase::Prepare,
                KERNEL_COLUMN_SCAN_SQL
            ))
        );
        Ok(())
    }

    #[test]
    fn load_kernel_columns_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match load_kernel_columns(&conn, "SELECT CAST(? AS BIGINT) AS column_name") {
            Ok(cols) => {
                anyhow::bail!("unbound kernel-column SQL should not hydrate successfully: {cols:?}")
            }
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert_eq!(
            err.sql_parts(),
            Some((
                "ncu-command",
                crate::SqlPhase::Query,
                KERNEL_COLUMN_SCAN_SQL
            ))
        );
        Ok(())
    }

    #[test]
    fn load_kernel_columns_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match load_kernel_columns(&conn, "SELECT 1 AS column_name") {
            Ok(cols) => anyhow::bail!(
                "malformed kernel-column row should not hydrate successfully: {cols:?}"
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert_eq!(
            err.sql_parts(),
            Some(("ncu-command", crate::SqlPhase::Read, KERNEL_COLUMN_SCAN_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_selected_kernel_prepare_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_selected_kernel(&conn, kernel_row_id(), "SELECT * FROM") {
            Ok(row) => anyhow::bail!(
                "malformed kernel lookup SQL should not hydrate successfully: {row:?}"
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-prepare");
        assert_eq!(
            err.sql_parts(),
            Some(("ncu-command", crate::SqlPhase::Prepare, KERNEL_LOOKUP_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_selected_kernel_query_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;

        let err = match hydrate_selected_kernel(
            &conn,
            kernel_row_id(),
            "SELECT ? AS start_ns, ? AS end_ns",
        ) {
            Ok(row) => {
                anyhow::bail!("unbound kernel lookup SQL should not hydrate successfully: {row:?}")
            }
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-query");
        assert_eq!(
            err.sql_parts(),
            Some(("ncu-command", crate::SqlPhase::Query, KERNEL_LOOKUP_SQL))
        );
        Ok(())
    }

    #[test]
    fn hydrate_selected_kernel_read_error_is_typed() -> Result<()> {
        let conn = duckdb::Connection::open_in_memory()?;
        let sql = selected_kernel_hydration_sql("'not-a-start'");

        let err = match hydrate_selected_kernel(&conn, kernel_row_id(), &sql) {
            Ok(row) => anyhow::bail!(
                "malformed kernel lookup row should not hydrate successfully: {row:?}"
            ),
            Err(err) => err,
        };

        assert_eq!(err.code().as_str(), "nsys.query.sql-read");
        assert_eq!(
            err.sql_parts(),
            Some(("ncu-command", crate::SqlPhase::Read, KERNEL_LOOKUP_SQL))
        );
        Ok(())
    }
}
