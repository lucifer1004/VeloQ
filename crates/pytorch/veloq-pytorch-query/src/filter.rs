use crate::scope::RankScope;
use anyhow::{Context, Result};
use regex::Regex;
use std::collections::BTreeSet;
use veloq_core::time::DurationFilter;
use veloq_pytorch_data::{Event, EventType, TraceSet};

#[derive(Debug, Clone)]
pub struct EventFilterRequest {
    pub types: TypeSelection,
    pub name_glob: Option<String>,
    pub name_regex: Option<String>,
    pub duration: Option<DurationFilter>,
    pub time_window_ns: Option<(i64, i64)>,
    pub rank_scope: RankScope,
    pub device: Option<i64>,
    pub stream: Option<i64>,
    pub step: Option<i64>,
    pub is_comm: bool,
    pub limit: usize,
}

impl Default for EventFilterRequest {
    fn default() -> Self {
        Self {
            types: TypeSelection::All,
            name_glob: None,
            name_regex: None,
            duration: None,
            time_window_ns: None,
            rank_scope: RankScope::default(),
            device: None,
            stream: None,
            step: None,
            is_comm: false,
            limit: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeSelection {
    All,
    Only(BTreeSet<TypeToken>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeToken {
    Event(EventType),
    Comm,
}

pub fn parse_type_selection(raw: &str) -> Result<TypeSelection> {
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("all") {
        return Ok(TypeSelection::All);
    }
    let mut out = BTreeSet::new();
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let parsed = match token {
            "cpu-op" | "cpu_op" => TypeToken::Event(EventType::CpuOp),
            "annotation" => TypeToken::Event(EventType::Annotation),
            "step" => TypeToken::Event(EventType::Step),
            "runtime" => TypeToken::Event(EventType::Runtime),
            "driver" => TypeToken::Event(EventType::Driver),
            "kernel" => TypeToken::Event(EventType::Kernel),
            "memcpy" => TypeToken::Event(EventType::Memcpy),
            "memset" => TypeToken::Event(EventType::Memset),
            "memory" => TypeToken::Event(EventType::Memory),
            "python" => TypeToken::Event(EventType::Python),
            "comm" => TypeToken::Comm,
            "all" => return Ok(TypeSelection::All),
            other => anyhow::bail!(
                "unknown pytorch --type `{other}`; expected one of: cpu-op, annotation, step, runtime, driver, kernel, memcpy, memset, memory, python, comm, all"
            ),
        };
        out.insert(parsed);
    }
    if out.is_empty() {
        anyhow::bail!("--type must list at least one event type");
    }
    Ok(TypeSelection::Only(out))
}

pub(crate) fn require_rank_scope(trace: &TraceSet, scope: RankScope) -> Result<()> {
    if trace.is_multi_rank() && scope.rank.is_none() && !scope.all_ranks {
        anyhow::bail!("pytorch trace-set has multiple ranks; use `--rank <n>` or `--all-ranks`");
    }
    Ok(())
}

pub(crate) fn filtered_events<'a>(
    trace: &'a TraceSet,
    request: &EventFilterRequest,
) -> Result<Vec<&'a Event>> {
    if request.limit == 0 {
        anyhow::bail!("--limit must be at least 1");
    }
    let compiled = CompiledFilters::new(request)?;
    let mut events = trace
        .events
        .iter()
        .filter(|event| event_matches_type(event, &request.types))
        .filter(|event| !request.is_comm || event.is_comm)
        .filter(|event| event_matches_scope(event, request))
        .filter(|event| compiled.matches_name(&event.name))
        .filter(|event| {
            request
                .duration
                .is_none_or(|filter| duration_matches(event.duration_ns, filter))
        })
        .filter(|event| {
            request
                .time_window_ns
                .is_none_or(|(start, end)| event.end_ns > start && event.start_ns < end)
        })
        .collect::<Vec<_>>();
    events.sort_by_key(|event| (event.start_ns, event.stable_index));
    Ok(events)
}

pub(crate) fn event_matches_scope(event: &Event, request: &EventFilterRequest) -> bool {
    if let Some(rank) = request.rank_scope.rank
        && event.rank != Some(rank)
    {
        return false;
    }
    if let Some(device) = request.device
        && event.device_id != Some(device)
    {
        return false;
    }
    if let Some(stream) = request.stream
        && event.stream_id != Some(stream)
    {
        return false;
    }
    if let Some(step) = request.step
        && event.step != Some(step)
    {
        return false;
    }
    true
}

fn event_matches_type(event: &Event, selection: &TypeSelection) -> bool {
    match selection {
        TypeSelection::All => true,
        TypeSelection::Only(tokens) => tokens.iter().any(|token| match token {
            TypeToken::Event(event_type) => event.event_type == *event_type,
            TypeToken::Comm => event.is_comm,
        }),
    }
}

pub(crate) struct CompiledFilters {
    glob: Option<Regex>,
    regex: Option<Regex>,
}

impl CompiledFilters {
    pub(crate) fn new(request: &EventFilterRequest) -> Result<Self> {
        if request.name_glob.is_some() && request.name_regex.is_some() {
            anyhow::bail!("--name and --name-regex are mutually exclusive");
        }
        Ok(Self {
            glob: request
                .name_glob
                .as_deref()
                .map(glob_regex)
                .transpose()
                .context("invalid --name glob")?,
            regex: request
                .name_regex
                .as_deref()
                .map(Regex::new)
                .transpose()
                .context("invalid --name-regex")?,
        })
    }

    pub(crate) fn matches_name(&self, name: &str) -> bool {
        self.glob.as_ref().is_none_or(|regex| regex.is_match(name))
            && self.regex.as_ref().is_none_or(|regex| regex.is_match(name))
    }
}

fn glob_regex(pattern: &str) -> Result<Regex> {
    let mut out = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => out.push_str(".*"),
            '?' => out.push('.'),
            _ => out.push_str(&regex::escape(&ch.to_string())),
        }
    }
    out.push('$');
    Regex::new(&out).map_err(Into::into)
}

fn duration_matches(duration_ns: i64, filter: DurationFilter) -> bool {
    match filter {
        DurationFilter::Gt(ns) => duration_ns > ns,
        DurationFilter::Gte(ns) => duration_ns >= ns,
        DurationFilter::Lt(ns) => duration_ns < ns,
        DurationFilter::Lte(ns) => duration_ns <= ns,
        DurationFilter::Range { min_ns, max_ns } => duration_ns >= min_ns && duration_ns <= max_ns,
    }
}
