#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRef {
    Any,
    Absolute { start_ns: i64, end_ns: i64 },
}

impl WindowRef {
    pub fn from_option(window: Option<(i64, i64)>) -> Self {
        match window {
            Some((start_ns, end_ns)) => Self::Absolute { start_ns, end_ns },
            None => Self::Any,
        }
    }

    pub fn bounds(self) -> Option<(i64, i64)> {
        match self {
            Self::Any => None,
            Self::Absolute { start_ns, end_ns } => Some((start_ns, end_ns)),
        }
    }

    /// Half-open interval overlap against this query window.
    ///
    /// Mirrors the canonical predicate used by SQL backends:
    /// `event_start < window_end AND event_end > window_start`.
    pub fn overlaps(self, event_start_ns: i64, event_end_ns: i64) -> bool {
        match self {
            Self::Any => true,
            Self::Absolute {
                start_ns: window_start_ns,
                end_ns: window_end_ns,
            } => event_start_ns < window_end_ns && event_end_ns > window_start_ns,
        }
    }

    /// Clip an event interval to this query window.
    ///
    /// Event end is normalised to at least `event_start_ns`, matching typed
    /// event paths that treat negative durations as empty rather than
    /// panicking.
    pub fn clipped_interval(self, event_start_ns: i64, event_end_ns: i64) -> Option<(i64, i64)> {
        let normalized_event_end_ns = event_end_ns.max(event_start_ns);
        let (window_start_ns, window_end_ns) = self
            .bounds()
            .unwrap_or((event_start_ns, normalized_event_end_ns));
        let start = event_start_ns.max(window_start_ns);
        let end = normalized_event_end_ns.min(window_end_ns);
        (end > start).then_some((start, end))
    }

    pub fn clipped_duration_ns(self, event_start_ns: i64, event_end_ns: i64) -> Option<i64> {
        self.clipped_interval(event_start_ns, event_end_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
}

/// Half-open interval overlap for an optional query window.
///
/// Mirrors the canonical predicate used by SQL backends:
/// `event_start < window_end AND event_end > window_start`.
pub fn interval_overlaps(start_ns: i64, end_ns: i64, window: Option<(i64, i64)>) -> bool {
    WindowRef::from_option(window).overlaps(start_ns, end_ns)
}

/// Clip an event interval to an optional query window.
///
/// Event end is normalised to at least `start_ns`, matching typed event
/// paths that treat negative durations as empty rather than panicking.
pub fn clipped_interval(
    start_ns: i64,
    end_ns: i64,
    window: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    WindowRef::from_option(window).clipped_interval(start_ns, end_ns)
}

pub fn clipped_interval_duration_ns(
    start_ns: i64,
    end_ns: i64,
    window: Option<(i64, i64)>,
) -> Option<i64> {
    WindowRef::from_option(window).clipped_duration_ns(start_ns, end_ns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_uses_half_open_intervals() {
        assert!(interval_overlaps(10, 20, Some((0, 11))));
        assert!(!interval_overlaps(10, 20, Some((20, 30))));
        assert!(!interval_overlaps(10, 20, Some((0, 10))));
        assert!(interval_overlaps(10, 20, None));
    }

    #[test]
    fn window_ref_preserves_optional_bounds() {
        assert_eq!(WindowRef::from_option(None), WindowRef::Any);
        assert_eq!(
            WindowRef::from_option(Some((10, 20))),
            WindowRef::Absolute {
                start_ns: 10,
                end_ns: 20
            }
        );
        assert_eq!(
            WindowRef::from_option(Some((10, 20))).bounds(),
            Some((10, 20))
        );
        assert_eq!(WindowRef::from_option(None).bounds(), None);
    }

    #[test]
    fn window_ref_overlaps_half_open_intervals() {
        let window = WindowRef::Absolute {
            start_ns: 10,
            end_ns: 20,
        };
        assert!(window.overlaps(0, 11));
        assert!(!window.overlaps(20, 30));
        assert!(!window.overlaps(0, 10));
        assert!(WindowRef::Any.overlaps(20, 10));
    }

    #[test]
    fn clipped_interval_applies_window_edges() {
        assert_eq!(clipped_interval(10, 20, Some((15, 25))), Some((15, 20)));
        assert_eq!(clipped_interval(10, 20, Some((0, 15))), Some((10, 15)));
        assert_eq!(clipped_interval(10, 20, Some((20, 30))), None);
    }

    #[test]
    fn window_ref_clips_interval_and_duration() {
        let window = WindowRef::Absolute {
            start_ns: 12,
            end_ns: 18,
        };
        assert_eq!(window.clipped_interval(10, 20), Some((12, 18)));
        assert_eq!(window.clipped_duration_ns(10, 20), Some(6));
        assert_eq!(WindowRef::Any.clipped_interval(10, 8), None);
    }

    #[test]
    fn clipped_duration_uses_normalized_end() {
        assert_eq!(clipped_interval_duration_ns(10, 8, None), None);
        assert_eq!(
            clipped_interval_duration_ns(10, 20, Some((12, 18))),
            Some(6)
        );
    }
}
