use thiserror::Error;

/// Source-neutral axis dependency validation.
///
/// VeloQ query flags use profile-specific axis names (`rank`, `device`,
/// `stream`, `context`, ...), but the safety rule is shared:
///
/// - filtering a child axis requires every parent axis to be fixed;
/// - projecting/grouping a child axis requires every parent axis to be
///   fixed or projected with it.
///
/// This module intentionally does not know CLI flags, error messages,
/// JSON wire fields, or source schemas. Source crates translate their
/// own flags/group-by axes into an [`AxisUsage`] and map
/// [`AxisParentError`] back into typed source errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisUsage<'a> {
    fixed: &'a [&'static str],
    projected: &'a [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{axis} is missing required parent axes")]
pub struct AxisParentError {
    axis: &'static str,
    missing_parents: Vec<&'static str>,
}

impl<'a> AxisUsage<'a> {
    pub fn new(fixed: &'a [&'static str], projected: &'a [&'static str]) -> Self {
        Self { fixed, projected }
    }

    pub fn validate_filter(
        self,
        axis: &'static str,
        parents: &[&'static str],
    ) -> Result<(), AxisParentError> {
        self.validate(axis, parents, AxisParentMode::Fixed)
    }

    pub fn validate_projection(
        self,
        axis: &'static str,
        parents: &[&'static str],
    ) -> Result<(), AxisParentError> {
        self.validate(axis, parents, AxisParentMode::FixedOrProjected)
    }

    fn validate(
        self,
        axis: &'static str,
        parents: &[&'static str],
        mode: AxisParentMode,
    ) -> Result<(), AxisParentError> {
        let missing_parents = parents
            .iter()
            .copied()
            .filter(|parent| match mode {
                AxisParentMode::Fixed => !self.is_fixed(parent),
                AxisParentMode::FixedOrProjected => {
                    !self.is_fixed(parent) && !self.is_projected(parent)
                }
            })
            .collect::<Vec<_>>();
        if missing_parents.is_empty() {
            Ok(())
        } else {
            Err(AxisParentError {
                axis,
                missing_parents,
            })
        }
    }

    fn is_fixed(self, axis: &str) -> bool {
        self.fixed.contains(&axis)
    }

    fn is_projected(self, axis: &str) -> bool {
        self.projected.contains(&axis)
    }
}

impl AxisParentError {
    pub fn axis(&self) -> &'static str {
        self.axis
    }

    pub fn missing_parents(&self) -> &[&'static str] {
        &self.missing_parents
    }

    pub fn missing_contains(&self, axis: &str) -> bool {
        self.missing_parents.contains(&axis)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxisParentMode {
    Fixed,
    FixedOrProjected,
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY: &[&str] = &[];
    const DEVICE: &[&str] = &["device"];
    const RANK_DEVICE: &[&str] = &["rank", "device"];

    #[test]
    fn child_filter_requires_fixed_parent() -> anyhow::Result<()> {
        let usage = AxisUsage::new(EMPTY, DEVICE);
        let err = usage
            .validate_filter("stream", DEVICE)
            .err()
            .ok_or_else(|| anyhow::anyhow!("projected parent must not satisfy a filter"))?;

        assert_eq!(err.axis(), "stream");
        assert_eq!(err.missing_parents(), DEVICE);
        Ok(())
    }

    #[test]
    fn child_projection_accepts_projected_parent() -> anyhow::Result<()> {
        let usage = AxisUsage::new(EMPTY, DEVICE);

        usage.validate_projection("stream", DEVICE)?;
        Ok(())
    }

    #[test]
    fn child_projection_accepts_fixed_parent() -> anyhow::Result<()> {
        let usage = AxisUsage::new(DEVICE, EMPTY);

        usage.validate_projection("stream", DEVICE)?;
        Ok(())
    }

    #[test]
    fn reports_every_missing_parent_in_order() -> anyhow::Result<()> {
        let usage = AxisUsage::new(EMPTY, EMPTY);
        let err = usage
            .validate_projection("stream", RANK_DEVICE)
            .err()
            .ok_or_else(|| anyhow::anyhow!("rank and device should both be missing"))?;

        assert_eq!(err.axis(), "stream");
        assert_eq!(err.missing_parents(), RANK_DEVICE);
        assert!(err.missing_contains("rank"));
        assert!(err.missing_contains("device"));
        Ok(())
    }
}
