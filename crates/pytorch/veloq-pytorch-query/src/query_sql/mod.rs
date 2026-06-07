pub(crate) mod collectives;
pub(crate) mod event_filter;
pub(crate) mod event_row;
pub(crate) mod exec;
pub(crate) mod inspect;
pub(crate) mod sidecar;
pub(crate) mod slices;

use crate::filter::EventFilterRequest;
use crate::{PytorchQueryError, PytorchQueryResult};
use veloq_core::NameFilterRef;
use veloq_query::sql::{SqlFilter, name};

pub(crate) struct SqlQuery {
    pub(crate) sql: String,
    pub(crate) params: Vec<duckdb::types::Value>,
}

pub(crate) fn push_name_filter(
    filter: &mut SqlFilter,
    column: &str,
    request: &EventFilterRequest,
) -> PytorchQueryResult<()> {
    let name_filter =
        NameFilterRef::from_optional(request.name_glob.as_deref(), request.name_regex.as_deref())
            .map_err(PytorchQueryError::from_name_match)?;
    name_filter
        .compile_matcher()
        .map_err(PytorchQueryError::from_name_match)?;
    if let Some(fragment) = name::predicate(column, name_filter) {
        filter.push_fragment(fragment);
    }
    Ok(())
}
