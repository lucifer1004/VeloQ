//! Type-safe "which event kinds is this request about" selector.
//!
//! A plain `kinds: Vec<EventKind>` field is ambiguous: an empty vec
//! could mean "all kinds" or "no kinds" depending on the command. With
//! this enum the meaning is explicit at the call site:
//!
//! ```ignore
//! KindFilter::All                                // every kind the command supports
//! KindFilter::Only(vec![EventKind::Kernel])      // just kernels
//! ```
//!
//! Each command's `run()` resolves `All` against its own allow-list
//! (stats only knows GPU kinds; search knows all six). The
//! `to_kinds(default_all)` helper centralises that step.

use crate::EventKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KindFilter {
    /// Every kind the command supports — resolved against the caller's
    /// `default_all` at the SQL boundary.
    All,
    /// An explicit subset. May still be filtered by table availability
    /// at SQL time, but `All` semantics are explicitly opted out of.
    Only(Vec<EventKind>),
}

impl Default for KindFilter {
    /// Default to `All` so library callers using `Request::default()`
    /// get a meaningful response rather than zero rows.
    fn default() -> Self {
        Self::All
    }
}

impl KindFilter {
    /// Resolve to a concrete kind list. `All` returns `default_all`;
    /// `Only` returns its contents.
    pub fn resolve(&self, default_all: &[EventKind]) -> Vec<EventKind> {
        match self {
            Self::All => default_all.to_vec(),
            Self::Only(v) => v.clone(),
        }
    }

    /// True if this filter would include `kind` once resolved against
    /// `default_all`.
    pub fn includes(&self, kind: EventKind, default_all: &[EventKind]) -> bool {
        match self {
            Self::All => default_all.contains(&kind),
            Self::Only(v) => v.contains(&kind),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_resolves_to_default() {
        let f = KindFilter::All;
        let allowed = &[EventKind::Kernel, EventKind::Memcpy];
        assert_eq!(
            f.resolve(allowed),
            vec![EventKind::Kernel, EventKind::Memcpy]
        );
    }

    #[test]
    fn only_returns_its_subset() {
        let f = KindFilter::Only(vec![EventKind::Kernel]);
        let allowed = &[EventKind::Kernel, EventKind::Memcpy, EventKind::Memset];
        assert_eq!(f.resolve(allowed), vec![EventKind::Kernel]);
    }

    #[test]
    fn default_is_all() {
        assert_eq!(KindFilter::default(), KindFilter::All);
    }
}
