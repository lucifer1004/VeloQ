use crate::scope::RankScope;
use crate::{PytorchQueryError, PytorchQueryResult};
use std::collections::BTreeSet;
use veloq_core::LimitRef;
use veloq_core::time::DurationFilter;
use veloq_pytorch_data::{EventType, QueryTrace};

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
    if let Some(stream) = request.stream {
        match (
            trace.is_multi_rank(),
            request.rank_scope.rank,
            request.device,
        ) {
            (true, None, None) => {
                return Err(PytorchQueryError::local_filter_parent_required(
                    "stream",
                    stream,
                    "`--rank <n>` and `--device <id>` because stream ids are rank/device-local",
                    "use `--rank <n> --device <id> --stream <id>` for one stream",
                ));
            }
            (true, None, Some(_)) => {
                return Err(PytorchQueryError::local_filter_parent_required(
                    "stream",
                    stream,
                    "`--rank <n>` because stream ids are rank-local in multi-rank traces",
                    "use `--rank <n> --device <id> --stream <id>` for one stream",
                ));
            }
            (_, _, None) => {
                return Err(PytorchQueryError::local_filter_parent_required(
                    "stream",
                    stream,
                    "`--device <id>` because stream ids are device-local",
                    "use `--device <id> --stream <id>` for one stream",
                ));
            }
            _ => {}
        }
    }

    if trace.is_multi_rank()
        && request.device.is_some()
        && request.rank_scope.rank.is_none()
        && let Some(device) = request.device
    {
        return Err(PytorchQueryError::local_filter_parent_required(
            "device",
            device,
            "`--rank <n>` because device ids are rank-local in multi-rank traces",
            "use `--rank <n> --device <id>` for one rank/device scope",
        ));
    }

    require_rank_scope(trace, request.rank_scope)?;

    Ok(())
}

pub(crate) fn limit_ref(limit: usize) -> PytorchQueryResult<LimitRef> {
    LimitRef::new(limit).map_err(|_| PytorchQueryError::LimitTooSmall)
}
