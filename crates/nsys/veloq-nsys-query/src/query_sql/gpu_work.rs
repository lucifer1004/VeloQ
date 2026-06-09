//! Query-side adapter for NSys GPU work interval semantics.
//!
//! The data crate owns the source-of-truth table/label list. This
//! module maps that source-neutral definition to query-side
//! [`EventKind`] values and keeps compute/copy classification explicit.

use crate::{EventKind, NsysQueryError, NsysQueryResult};
use veloq_nsys_data::{GPU_WORK_INTERVAL_KINDS, GpuWorkKind, Trace};

#[derive(Debug, Clone)]
pub(crate) struct GpuWorkSet {
    kinds: Vec<EventKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuWorkClass {
    Compute,
    Copy,
}

impl GpuWorkSet {
    pub(crate) fn from_data_definition() -> NsysQueryResult<Self> {
        let kinds = GPU_WORK_INTERVAL_KINDS
            .iter()
            .map(event_kind_for_work)
            .collect::<NsysQueryResult<Vec<_>>>()?;
        Ok(Self { kinds })
    }

    pub(crate) fn kinds(&self) -> &[EventKind] {
        &self.kinds
    }

    pub(crate) fn contains(&self, kind: EventKind) -> bool {
        self.kinds.contains(&kind)
    }

    pub(crate) fn present_in(&self, trace: &Trace) -> Vec<EventKind> {
        self.kinds
            .iter()
            .copied()
            .filter(|kind| trace.table_exists(kind.table()))
            .collect()
    }

    pub(crate) fn class(&self, kind: EventKind) -> NsysQueryResult<GpuWorkClass> {
        if !self.contains(kind) {
            return Err(NsysQueryError::internal_unsupported_kind(
                "gpu-work",
                kind.as_str(),
            ));
        }
        match kind {
            EventKind::Kernel | EventKind::Graph => Ok(GpuWorkClass::Compute),
            EventKind::Memcpy | EventKind::Memset => Ok(GpuWorkClass::Copy),
            _ => Err(NsysQueryError::internal_unsupported_kind(
                "gpu-work-class",
                kind.as_str(),
            )),
        }
    }
}

fn event_kind_for_work(work: &GpuWorkKind) -> NsysQueryResult<EventKind> {
    let kind = EventKind::parse(work.label)
        .ok_or_else(|| NsysQueryError::internal_sql_kind_tag_invalid("gpu-work", work.label))?;
    if kind.table() != work.table {
        return Err(NsysQueryError::internal_unsupported_kind(
            "gpu-work", work.label,
        ));
    }
    Ok(kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_event_kinds_follow_data_definition() -> NsysQueryResult<()> {
        let work = GpuWorkSet::from_data_definition()?;
        assert_eq!(
            work.kinds(),
            vec![
                EventKind::Kernel,
                EventKind::Memcpy,
                EventKind::Memset,
                EventKind::Graph,
            ]
        );
        for work in GPU_WORK_INTERVAL_KINDS {
            let kind = event_kind_for_work(work)?;
            assert_eq!(kind.as_str(), work.label);
            assert_eq!(kind.table(), work.table);
        }
        Ok(())
    }

    #[test]
    fn compute_copy_classification_is_explicit() -> NsysQueryResult<()> {
        let work = GpuWorkSet::from_data_definition()?;
        assert_eq!(work.class(EventKind::Kernel)?, GpuWorkClass::Compute);
        assert_eq!(work.class(EventKind::Graph)?, GpuWorkClass::Compute);
        assert_eq!(work.class(EventKind::Memcpy)?, GpuWorkClass::Copy);
        assert_eq!(work.class(EventKind::Memset)?, GpuWorkClass::Copy);

        for kind in work.kinds() {
            work.class(*kind)?;
        }
        Ok(())
    }
}
