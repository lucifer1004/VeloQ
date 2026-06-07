use veloq_core::{SortSpec, time::DurationFilter};
use veloq_nsys_query::RowId;

use crate::error::{NsysSourceError, NsysSourceResult};

/// Project a [`KindFilter`] into the comma-joined string form
/// `applied_scope.kind` carries. `All` is reported as `None` (the
/// default, equivalent to "no kind filter"); `Only(...)` joins the
/// kind names with commas.
pub(super) fn kinds_csv(kf: &veloq_nsys_query::KindFilter) -> Option<String> {
    use veloq_nsys_query::KindFilter;
    match kf {
        KindFilter::All => None,
        KindFilter::Only(v) => {
            if v.is_empty() {
                None
            } else {
                Some(
                    v.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<&str>>()
                        .join(","),
                )
            }
        }
    }
}

/// Parse the user's `--sort` string into a `SortSpec` for the request.
/// Returns `None` when the input is empty (lets the module pick its own
/// default), and an [`NsysSourceError`] on syntax problems.
pub(super) fn parse_sort_spec(s: &str) -> NsysSourceResult<Option<SortSpec>> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(None);
    }
    let spec = SortSpec::parse(t).map_err(|source| NsysSourceError::invalid_sort(s, source))?;
    Ok(Some(spec))
}

pub(super) fn parse_duration_filter(s: &str) -> NsysSourceResult<DurationFilter> {
    DurationFilter::parse(s).map_err(|source| NsysSourceError::invalid_duration(s, source))
}

pub(super) fn parse_row_id(s: &str) -> NsysSourceResult<RowId> {
    s.parse::<RowId>()
        .map_err(|source| NsysSourceError::invalid_row_id(s, source))
}
