use crate::scope::RankScope;
use crate::{PytorchQueryError, PytorchQueryResult};
use std::collections::BTreeSet;
use veloq_core::time::DurationFilter;
use veloq_core::{AxisParentError, AxisUsage, LimitRef};
use veloq_pytorch_data::{EventType, QueryTrace};

const NO_AXES: &[&str] = &[];
const RANK_AXIS: &[&str] = &["rank"];
const DEVICE_AXIS: &[&str] = &["device"];
const RANK_DEVICE_AXES: &[&str] = &["rank", "device"];

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

pub fn parse_type_selection(raw: &str) -> PytorchQueryResult<TypeSelection> {
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
            other => return Err(PytorchQueryError::unknown_type(other)),
        };
        out.insert(parsed);
    }
    if out.is_empty() {
        return Err(PytorchQueryError::EmptyTypeSelection);
    }
    Ok(TypeSelection::Only(out))
}

pub(crate) fn require_rank_scope(trace: &QueryTrace, scope: RankScope) -> PytorchQueryResult<()> {
    if trace.is_multi_rank() && scope.rank.is_none() && !scope.all_ranks {
        return Err(PytorchQueryError::MultiRankRequiresScope);
    }
    Ok(())
}

pub(crate) fn validate_event_scope(
    trace: &QueryTrace,
    request: &EventFilterRequest,
) -> PytorchQueryResult<()> {
    let usage = event_filter_axis_usage(trace, request);
    if let Some(stream) = request.stream
        && let Err(err) = usage.validate_filter("stream", RANK_DEVICE_AXES)
    {
        return Err(stream_filter_parent_error(stream, &err));
    }

    if let Some(device) = request.device
        && usage.validate_filter("device", RANK_AXIS).is_err()
    {
        return Err(device_filter_parent_error(device));
    }

    require_rank_scope(trace, request.rank_scope)?;

    Ok(())
}

fn event_filter_axis_usage<'a>(trace: &QueryTrace, request: &EventFilterRequest) -> AxisUsage<'a> {
    match (
        !trace.is_multi_rank() || request.rank_scope.rank.is_some(),
        request.device.is_some(),
    ) {
        (true, true) => AxisUsage::new(RANK_DEVICE_AXES, NO_AXES),
        (true, false) => AxisUsage::new(RANK_AXIS, NO_AXES),
        (false, true) => AxisUsage::new(DEVICE_AXIS, NO_AXES),
        (false, false) => AxisUsage::new(NO_AXES, NO_AXES),
    }
}

fn stream_filter_parent_error(stream: i64, err: &AxisParentError) -> PytorchQueryError {
    match (err.missing_contains("rank"), err.missing_contains("device")) {
        (true, true) => PytorchQueryError::local_filter_parent_required(
            "stream",
            stream,
            "`--rank <n>` and `--device <id>` because stream ids are rank/device-local",
            "use `--rank <n> --device <id> --stream <id>` for one stream",
        ),
        (true, false) => PytorchQueryError::local_filter_parent_required(
            "stream",
            stream,
            "`--rank <n>` because stream ids are rank-local in multi-rank traces",
            "use `--rank <n> --device <id> --stream <id>` for one stream",
        ),
        (false, true) | (false, false) => PytorchQueryError::local_filter_parent_required(
            "stream",
            stream,
            "`--device <id>` because stream ids are device-local",
            "use `--device <id> --stream <id>` for one stream",
        ),
    }
}

fn device_filter_parent_error(device: i64) -> PytorchQueryError {
    PytorchQueryError::local_filter_parent_required(
        "device",
        device,
        "`--rank <n>` because device ids are rank-local in multi-rank traces",
        "use `--rank <n> --device <id>` for one rank/device scope",
    )
}

pub(crate) fn limit_ref(limit: usize) -> PytorchQueryResult<LimitRef> {
    LimitRef::new(limit).map_err(|_| PytorchQueryError::LimitTooSmall)
}
