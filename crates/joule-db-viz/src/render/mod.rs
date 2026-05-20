//! Rendering backends for visualization hints.
//!
//! Each renderer takes a [`VizHint`] and query data, producing output in a
//! specific format (Vega-Lite JSON, text summary, accessibility metadata).

#[cfg(feature = "vega")]
pub mod vega;

#[cfg(feature = "text-summary")]
pub mod text;

#[cfg(feature = "accessibility")]
pub mod accessibility;

use crate::error::VizResult;
use crate::hint::VizHint;

/// Output format from a renderer.
#[derive(Debug, Clone)]
pub enum RenderOutput {
    /// JSON string (Vega-Lite spec, etc.)
    Json(String),
    /// Plain text.
    Text(String),
    /// HTML fragment.
    Html(String),
}

/// Configuration for renderers.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Maximum width in pixels (for chart renderers).
    pub width: u32,
    /// Maximum height in pixels.
    pub height: u32,
    /// Whether to include data values inline (vs. reference).
    pub inline_data: bool,
    /// Color scheme name (e.g. "tableau10", "dark2").
    pub color_scheme: String,
    /// Whether to include interactive tooltips.
    pub interactive: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 600,
            height: 400,
            inline_data: true,
            color_scheme: "tableau10".to_string(),
            interactive: true,
        }
    }
}

/// Trait for visualization renderers.
pub trait Renderer {
    /// Render a visualization hint with data into an output format.
    fn render(
        &self,
        hint: &VizHint,
        columns: &[String],
        rows: &[Vec<serde_json::Value>],
        config: &RenderConfig,
    ) -> VizResult<RenderOutput>;
}
