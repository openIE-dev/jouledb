//! Core visualization hint types.
//!
//! `VizHint` is the central abstraction — a lightweight, serializable metadata
//! structure describing HOW to visualize query results. It travels with the
//! `QueryResponse` over HTTP, WebSocket, and PgWire.

use serde::{Deserialize, Serialize};

use crate::data_profile::DataProfile;

/// Visualization hint attached to query results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VizHint {
    /// Recommended primary chart type.
    pub chart_type: ChartType,
    /// Alternative chart types ranked by relevance.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<ChartType>,
    /// Axis mappings: which columns map to x, y, color, size, label.
    pub axes: AxisMapping,
    /// Suggested title (derived from table name + aggregation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Confidence score for the inference (0.0–1.0).
    pub confidence: f32,
    /// Human-readable explanation of why this chart type was chosen.
    pub reasoning: String,
    /// Data characteristics that influenced the inference.
    pub data_profile: DataProfile,
    /// Energy visualization overlay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_overlay: Option<EnergyOverlay>,
    /// Accessibility metadata.
    pub accessibility: AccessibilityHint,
    /// Sonification parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sonification: Option<SonificationHint>,
    /// Distortion/accuracy warnings for the recommended chart.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// Density strategy hint for large datasets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density_strategy: Option<DensityStrategy>,
}

/// Density/aggregation strategy for large result sets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DensityStrategy {
    /// Server should aggregate before sending.
    ServerAggregate {
        /// Suggested maximum data points.
        target_points: usize,
    },
    /// Client should bin data for display.
    ClientBin {
        /// Suggested number of bins.
        bin_count: usize,
    },
    /// Downsample with LTTB or M4 algorithm.
    Downsample {
        /// Target number of output points.
        target_points: usize,
    },
}

/// Chart type enumeration covering common visualization forms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ChartType {
    /// Single scalar value display.
    Scalar,
    /// Tabular display (default fallback).
    Table,
    /// Line chart (time series).
    Line,
    /// Vertical bar chart (categorical comparisons).
    Bar,
    /// Horizontal bar chart.
    HorizontalBar,
    /// Stacked bar chart.
    StackedBar,
    /// Pie or donut chart.
    Pie,
    /// Scatter plot (2D or 3D).
    Scatter,
    /// Heatmap / matrix.
    Heatmap,
    /// Histogram (distribution).
    Histogram,
    /// Area chart (cumulative).
    Area,
    /// Sparkline (inline miniature).
    Sparkline,
    /// Gauge (single metric against a range).
    Gauge,
    /// Tree / hierarchical.
    Tree,
    /// Network / force-directed graph.
    ForceGraph,
    /// Geographic map.
    Map,
    /// Box plot / violin plot.
    BoxPlot,
    /// Energy efficiency dashboard.
    EnergyDashboard,
}

impl std::fmt::Display for ChartType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ChartType::Scalar => "scalar",
            ChartType::Table => "table",
            ChartType::Line => "line",
            ChartType::Bar => "bar",
            ChartType::HorizontalBar => "horizontal_bar",
            ChartType::StackedBar => "stacked_bar",
            ChartType::Pie => "pie",
            ChartType::Scatter => "scatter",
            ChartType::Heatmap => "heatmap",
            ChartType::Histogram => "histogram",
            ChartType::Area => "area",
            ChartType::Sparkline => "sparkline",
            ChartType::Gauge => "gauge",
            ChartType::Tree => "tree",
            ChartType::ForceGraph => "force_graph",
            ChartType::Map => "map",
            ChartType::BoxPlot => "box_plot",
            ChartType::EnergyDashboard => "energy_dashboard",
        };
        write!(f, "{}", s)
    }
}

/// Axis mappings describing which columns map to visual encodings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AxisMapping {
    /// Column for X axis (or category axis).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<ColumnRef>,
    /// Column for Y axis (or value axis).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<ColumnRef>,
    /// Column for Z axis (3D scatter).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<ColumnRef>,
    /// Column for color encoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<ColumnRef>,
    /// Column for size encoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<ColumnRef>,
    /// Column for label/tooltip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<ColumnRef>,
    /// Column for grouping/series.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<ColumnRef>,
}

/// Reference to a column in the result set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnRef {
    /// Column index in the result set.
    pub index: usize,
    /// Column name.
    pub name: String,
    /// Inferred semantic type.
    pub semantic_type: SemanticType,
}

/// Semantic classification of a column based on name and value heuristics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SemanticType {
    Timestamp,
    Date,
    NumericContinuous,
    NumericDiscrete,
    Categorical,
    Text,
    Boolean,
    Identifier,
    GeoLatitude,
    GeoLongitude,
    Currency,
    Percentage,
    Unknown,
}

/// Energy visualization overlay for energy-aware queries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnergyOverlay {
    /// Energy consumed in joules.
    pub energy_joules: f64,
    /// Power draw in watts.
    pub power_watts: f64,
    /// Efficiency classification.
    pub efficiency: EnergyEfficiency,
    /// Device that executed the query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Algorithm used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
}

/// Energy efficiency classification based on energy per row.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnergyEfficiency {
    /// < 1 mJ per row
    Excellent,
    /// 1–10 mJ per row
    Good,
    /// 10–100 mJ per row
    Fair,
    /// > 100 mJ per row
    Poor,
}

/// Accessibility metadata for screen readers and assistive technology.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AccessibilityHint {
    /// Human-readable alt text for the chart.
    pub alt_text: String,
    /// Longer description with data summary.
    pub description: String,
    /// ARIA role for the visualization element.
    pub aria_role: String,
}

/// Sonification parameters for auditory data representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SonificationHint {
    /// Column index mapped to pitch (frequency).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch_column: Option<usize>,
    /// Column index mapped to volume (amplitude).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume_column: Option<usize>,
    /// Base frequency in Hz.
    pub base_frequency: f64,
    /// Frequency range (min, max) in Hz.
    pub frequency_range: (f64, f64),
    /// Suggested MIDI instrument (General MIDI program number).
    pub midi_program: u8,
    /// Duration per data point in milliseconds.
    pub note_duration_ms: u32,
}
