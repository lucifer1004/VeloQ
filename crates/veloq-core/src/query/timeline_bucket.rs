use super::time_window::WindowRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineBucket {
    pub start_ns: i64,
    pub interval_ns: i64,
}

impl TimelineBucket {
    pub fn new(start_ns: i64, interval_ns: i64) -> Self {
        Self {
            start_ns,
            interval_ns,
        }
    }

    pub fn first_start_for(origin_ns: i64, interval_ns: i64, event_start_ns: i64) -> i64 {
        let offset = event_start_ns.saturating_sub(origin_ns);
        origin_ns.saturating_add((offset / interval_ns) * interval_ns)
    }

    pub fn end_ns(self) -> i64 {
        self.start_ns.saturating_add(self.interval_ns)
    }

    pub fn key(self) -> String {
        timeline_bucket_key(self.start_ns, self.end_ns())
    }

    pub fn overlap_ns(
        self,
        event_start_ns: i64,
        event_end_ns: i64,
        window: WindowRef,
    ) -> Option<i64> {
        let (start_ns, end_ns) = window.clipped_interval(event_start_ns, event_end_ns)?;
        let overlap = end_ns
            .min(self.end_ns())
            .saturating_sub(start_ns.max(self.start_ns));
        (overlap > 0).then_some(overlap)
    }
}

pub fn timeline_bucket_key(start_ns: i64, end_ns: i64) -> String {
    format!("bucket|{start_ns}..{end_ns}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_uses_canonical_bucket_wire_format() {
        assert_eq!(
            TimelineBucket::new(100, 25).key(),
            "bucket|100..125".to_string()
        );
        assert_eq!(timeline_bucket_key(100, 125), "bucket|100..125".to_string());
    }

    #[test]
    fn first_start_aligns_to_origin_relative_interval() {
        assert_eq!(TimelineBucket::first_start_for(10, 5, 23), 20);
        assert_eq!(TimelineBucket::first_start_for(10, 5, 10), 10);
    }

    #[test]
    fn overlap_clips_to_bucket_and_query_window() {
        let bucket = TimelineBucket::new(100, 10);
        let window = WindowRef::from_option(Some((105, 120)));

        assert_eq!(bucket.overlap_ns(90, 108, window), Some(3));
        assert_eq!(bucket.overlap_ns(110, 115, window), None);
        assert_eq!(bucket.overlap_ns(95, 100, WindowRef::Any), None);
    }
}
