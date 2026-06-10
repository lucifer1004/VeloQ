use crate::{NsysQueryError, NsysQueryResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrackKind {
    Gpu,
    CudaStreams,
    CudaStream,
    CudaApi,
    Nvtx,
    GapsOverlay,
}

impl TrackKind {
    pub(super) fn parse(raw: &str) -> NsysQueryResult<Self> {
        match raw {
            "gpu" => Ok(Self::Gpu),
            "cuda-streams" => Ok(Self::CudaStreams),
            "cuda-stream" => Ok(Self::CudaStream),
            "cuda-api" => Ok(Self::CudaApi),
            "nvtx" => Ok(Self::Nvtx),
            "gaps-overlay" => Ok(Self::GapsOverlay),
            _ => Err(NsysQueryError::VizTimelineUnknownTrackKind {
                kind: raw.to_string(),
            }),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::CudaStreams => "cuda-streams",
            Self::CudaStream => "cuda-stream",
            Self::CudaApi => "cuda-api",
            Self::Nvtx => "nvtx",
            Self::GapsOverlay => "gaps-overlay",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum DeviceSelector {
    All,
    One(i32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TrackSpec {
    pub(super) kind: TrackKind,
    pub(super) device: Option<DeviceSelector>,
    pub(super) stream: Option<i64>,
    pub(super) top: Option<usize>,
    pub(super) depth: Option<usize>,
}

impl TrackSpec {
    pub(super) fn parse(raw: &str) -> NsysQueryResult<Self> {
        let (kind_raw, selectors_raw) = match raw.split_once(':') {
            Some((kind, selectors)) => (kind.trim(), Some(selectors)),
            None => (raw.trim(), None),
        };
        let kind = TrackKind::parse(kind_raw)?;
        let mut spec = Self {
            kind,
            device: None,
            stream: None,
            top: None,
            depth: None,
        };
        if let Some(selectors) = selectors_raw {
            for selector in selectors.split(',') {
                let selector = selector.trim();
                if selector.is_empty() {
                    continue;
                }
                let (name, value) = selector.split_once('=').ok_or_else(|| {
                    NsysQueryError::VizTimelineInvalidSelector {
                        selector: selector.to_string(),
                    }
                })?;
                spec.apply_selector(name.trim(), value.trim())?;
            }
        }
        spec.validate()?;
        Ok(spec)
    }

    fn apply_selector(&mut self, name: &str, value: &str) -> NsysQueryResult<()> {
        match name {
            "device" if self.kind_accepts_selector("device") => {
                self.device = Some(parse_device_selector(value)?);
                Ok(())
            }
            "stream" if self.kind_accepts_selector("stream") => {
                self.stream = Some(parse_non_negative_i64("stream", value)?);
                Ok(())
            }
            "top" if self.kind_accepts_selector("top") => {
                self.top = Some(parse_positive_usize("top", value)?);
                Ok(())
            }
            "depth" if self.kind_accepts_selector("depth") => {
                self.depth = Some(parse_positive_usize("depth", value)?);
                Ok(())
            }
            _ => Err(NsysQueryError::VizTimelineUnknownSelector {
                kind: self.kind.as_str().to_string(),
                selector: name.to_string(),
            }),
        }
    }

    fn kind_accepts_selector(&self, selector: &str) -> bool {
        matches!(
            (self.kind, selector),
            (TrackKind::Gpu, "device")
                | (TrackKind::CudaStreams, "device" | "top")
                | (TrackKind::CudaStream, "device" | "stream")
                | (TrackKind::Nvtx, "depth")
                | (TrackKind::GapsOverlay, "device")
        )
    }

    fn validate(&self) -> NsysQueryResult<()> {
        if self.kind == TrackKind::CudaStream {
            match self.device {
                Some(DeviceSelector::One(_)) => {}
                Some(DeviceSelector::All) => {
                    return Err(NsysQueryError::VizTimelineCudaStreamDeviceAll);
                }
                None => return Err(NsysQueryError::VizTimelineCudaStreamDeviceRequired),
            }
            if self.stream.is_none() {
                return Err(NsysQueryError::VizTimelineCudaStreamStreamRequired);
            }
        }
        Ok(())
    }
}

fn parse_device_selector(value: &str) -> NsysQueryResult<DeviceSelector> {
    if value == "all" {
        return Ok(DeviceSelector::All);
    }
    let raw = parse_non_negative_i64("device", value)?;
    let device =
        i32::try_from(raw).map_err(|_| NsysQueryError::VizTimelineSelectorNonNegativeInt {
            selector: "device".to_string(),
        })?;
    Ok(DeviceSelector::One(device))
}

fn parse_non_negative_i64(selector: &str, value: &str) -> NsysQueryResult<i64> {
    let parsed =
        value
            .parse::<i64>()
            .map_err(|_| NsysQueryError::VizTimelineSelectorNonNegativeInt {
                selector: selector.to_string(),
            })?;
    if parsed < 0 {
        return Err(NsysQueryError::VizTimelineSelectorNonNegativeInt {
            selector: selector.to_string(),
        });
    }
    Ok(parsed)
}

fn parse_positive_usize(selector: &str, value: &str) -> NsysQueryResult<usize> {
    let parsed =
        value
            .parse::<usize>()
            .map_err(|_| NsysQueryError::VizTimelineSelectorPositiveInt {
                selector: selector.to_string(),
            })?;
    if parsed == 0 {
        return Err(NsysQueryError::VizTimelineSelectorPositiveInt {
            selector: selector.to_string(),
        });
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HighlightScope {
    Name,
    Instance,
}

impl HighlightScope {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        match raw {
            "name" => Ok(Self::Name),
            "instance" => Ok(Self::Instance),
            _ => Err(NsysQueryError::VizTimelineUnknownHighlightScope {
                scope: raw.to_string(),
            }),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Instance => "instance",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HighlightMetric {
    Duration,
    Count,
    MaxDuration,
}

impl HighlightMetric {
    fn parse(raw: &str) -> NsysQueryResult<Self> {
        match raw {
            "duration" | "total-duration" | "total_duration_ns" => Ok(Self::Duration),
            "count" | "instance-count" | "instance_count" => Ok(Self::Count),
            "max-duration" | "max_duration" | "max_duration_ns" => Ok(Self::MaxDuration),
            _ => Err(NsysQueryError::VizTimelineUnknownHighlightMetric {
                metric: raw.to_string(),
            }),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Duration => "total_duration_ns",
            Self::Count => "instance_count",
            Self::MaxDuration => "max_duration_ns",
        }
    }

    pub(super) fn score(
        self,
        total_duration_ns: i64,
        instance_count: usize,
        max_duration_ns: i64,
    ) -> i64 {
        match self {
            Self::Duration => total_duration_ns,
            Self::Count => i64::try_from(instance_count).unwrap_or(i64::MAX),
            Self::MaxDuration => max_duration_ns,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct KernelHighlightSpec {
    pub(super) raw: String,
    pub(super) top: usize,
    pub(super) scope: HighlightScope,
    pub(super) metric: HighlightMetric,
}

impl KernelHighlightSpec {
    pub(super) fn parse(raw: &str) -> NsysQueryResult<Self> {
        let mut top = None;
        let mut scope = HighlightScope::Name;
        let mut metric = HighlightMetric::Duration;
        for selector in raw.split(',') {
            let selector = selector.trim();
            if selector.is_empty() {
                continue;
            }
            let (name, value) = selector.split_once('=').ok_or_else(|| {
                NsysQueryError::VizTimelineInvalidSelector {
                    selector: selector.to_string(),
                }
            })?;
            match name.trim() {
                "top" => {
                    top = Some(parse_positive_usize("highlight-kernels.top", value.trim())?);
                }
                "scope" => {
                    scope = HighlightScope::parse(value.trim())?;
                }
                "by" | "metric" => {
                    metric = HighlightMetric::parse(value.trim())?;
                }
                other => {
                    return Err(NsysQueryError::VizTimelineUnknownSelector {
                        kind: "highlight-kernels".to_string(),
                        selector: other.to_string(),
                    });
                }
            }
        }
        Ok(Self {
            raw: raw.to_string(),
            top: top.ok_or(NsysQueryError::VizTimelineHighlightTopRequired)?,
            scope,
            metric,
        })
    }
}
