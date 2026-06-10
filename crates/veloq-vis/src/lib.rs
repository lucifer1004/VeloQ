//! Source-neutral static visualization primitives.
//!
//! Source crates own evidence extraction and track semantics. This crate
//! owns only the portable scene shape, a small SVG renderer, and
//! artifact-root publishing for generated report figures.

mod artifact;
mod error;
mod model;
mod render;

pub use artifact::write_svg_artifact;
pub use error::VisualizationError;
pub use model::{
    SvgRenderResult, SvgRenderSummary, VizAggregation, VizAxis, VizHighlight, VizInterval,
    VizLabelMode, VizLabelPolicy, VizRenderPolicy, VizRole, VizScene, VizSceneMetadata,
    VizTimeWindow, VizTrack, VizTrackMetadata, WrittenSvgArtifact,
};
pub use render::render_svg;
