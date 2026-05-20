//! joule-db-viz — Database-native visualization for JouleDB.
//!
//! This crate provides:
//! - **Inference**: Automatically determine the best chart type from query
//!   metadata and result data ([`VizInferencer`]).
//! - **Rendering**: Produce Vega-Lite specs, text summaries, and accessibility
//!   metadata from [`VizHint`]s.
//! - **Types**: Serializable hint types ([`ChartType`], [`SemanticType`],
//!   [`AxisMapping`]) that travel with query responses.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `infer` | yes | Inference engine |
//! | `vega` | yes | Vega-Lite JSON renderer |
//! | `text-summary` | yes | Natural language summaries |
//! | `accessibility` | yes | Alt-text and ARIA output |
//! | `sonification` | no | Auditory data mapping |
//! | `gpu` | no | GPU renderer stub (browser) |

pub mod column_classifier;
pub mod data_profile;
pub mod error;
pub mod hint;

#[cfg(feature = "infer")]
pub mod infer;

pub mod render;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use error::{VizError, VizResult};
pub use hint::{
    AccessibilityHint, AxisMapping, ChartType, ColumnRef, DensityStrategy, EnergyEfficiency,
    EnergyOverlay, SemanticType, SonificationHint, VizHint,
};

#[cfg(feature = "infer")]
pub use infer::{VizInferenceInput, VizInferencer};

pub use render::{RenderConfig, RenderOutput, Renderer};

#[cfg(feature = "vega")]
pub use render::vega::VegaRenderer;

#[cfg(feature = "text-summary")]
pub use render::text::TextRenderer;

#[cfg(feature = "accessibility")]
pub use render::accessibility::AccessibilityRenderer;

// ── GPU Renderer stub ───────────────────────────────────────────────────────
//
// The browser crate (`joule-db-browser`) imports `GpuRenderer` from this crate.
// The actual GPU implementation lives in `joule-db-browser`/`joule-db-gpu`.
// This stub provides the type so the import resolves.

/// GPU-accelerated chart renderer (stub).
///
/// The real implementation is provided by `joule-db-gpu` / `joule-db-browser`.
/// This stub exists so that `use joule_db_viz::GpuRenderer` resolves.
#[cfg(feature = "gpu")]
pub struct GpuRenderer {
    _private: (),
}

#[cfg(feature = "gpu")]
impl GpuRenderer {
    /// Create a new GPU renderer (stub — requires GPU backend integration).
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Render a chart using GPU acceleration (stub).
    pub async fn render_chart(&self, _chart_type: ChartType, _data: &[f32]) -> VizResult<()> {
        Err(VizError::GpuError(
            "GPU renderer stub — use joule-db-gpu for real implementation".to_string(),
        ))
    }
}

/// GPU-accelerated chart renderer (non-gpu builds).
///
/// Placeholder type so downstream code can reference `GpuRenderer`
/// without enabling the `gpu` feature.
#[cfg(not(feature = "gpu"))]
pub struct GpuRenderer {
    _private: (),
}

#[cfg(not(feature = "gpu"))]
impl GpuRenderer {
    /// Create a new GPU renderer (no-op without `gpu` feature).
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Render a chart (returns error without `gpu` feature).
    pub async fn render_chart(&self, _chart_type: ChartType, _data: &[f32]) -> VizResult<()> {
        Err(VizError::GpuError(
            "GPU renderer not available — enable the `gpu` feature".to_string(),
        ))
    }
}
