//! Vega-Lite JSON specification renderer.
//!
//! Produces a complete Vega-Lite v5 spec from a [`VizHint`] and data.
//! Defaults to colorblind-safe palettes (Paul Tol categorical, viridis sequential).

use serde_json::{Value, json};

use crate::error::VizResult;
use crate::hint::{ChartType, VizHint};
use crate::render::{RenderConfig, RenderOutput, Renderer};

/// Renderer that produces Vega-Lite JSON specs.
pub struct VegaRenderer;

impl Renderer for VegaRenderer {
    fn render(
        &self,
        hint: &VizHint,
        columns: &[String],
        rows: &[Vec<serde_json::Value>],
        config: &RenderConfig,
    ) -> VizResult<RenderOutput> {
        let spec = build_vega_spec(hint, columns, rows, config)?;
        let json_str = serde_json::to_string_pretty(&spec)
            .map_err(|e| crate::error::VizError::SerializationError(e.to_string()))?;
        Ok(RenderOutput::Json(json_str))
    }
}

/// Build a complete Vega-Lite specification.
pub fn build_vega_spec(
    hint: &VizHint,
    columns: &[String],
    rows: &[Vec<serde_json::Value>],
    config: &RenderConfig,
) -> VizResult<Value> {
    let data_values = build_data_values(columns, rows);
    let mark = chart_type_to_mark(hint.chart_type);
    let encoding = build_encoding(hint);

    let mut spec = json!({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": config.width,
        "height": config.height,
        "mark": mark,
        "encoding": encoding,
    });

    if config.inline_data {
        spec["data"] = json!({ "values": data_values });
    }

    if let Some(title) = &hint.title {
        spec["title"] = json!(title);
    }

    // Apply colorblind-safe palette.
    let palette = color_palette(&config.color_scheme);
    spec["config"] = build_theme_config(&palette);

    if config.interactive {
        spec["selection"] = json!({
            "hover": {
                "type": "single",
                "on": "pointerover",
                "empty": "none"
            }
        });
    }

    Ok(spec)
}

/// Convert chart type to Vega-Lite mark specification.
fn chart_type_to_mark(chart_type: ChartType) -> Value {
    match chart_type {
        ChartType::Line => json!({"type": "line", "point": true}),
        ChartType::Bar | ChartType::StackedBar => json!("bar"),
        ChartType::HorizontalBar => json!("bar"),
        ChartType::Area => json!({"type": "area", "opacity": 0.7}),
        ChartType::Scatter => json!({"type": "point", "filled": true}),
        ChartType::Pie => json!({"type": "arc"}),
        ChartType::Heatmap => json!("rect"),
        ChartType::Histogram => json!({"type": "bar", "binSpacing": 0}),
        ChartType::Sparkline => json!({"type": "line", "strokeWidth": 1}),
        ChartType::BoxPlot => json!("boxplot"),
        ChartType::Scalar | ChartType::Gauge => json!({"type": "text", "fontSize": 32}),
        ChartType::EnergyDashboard => json!({"type": "line", "point": true, "strokeWidth": 2}),
        ChartType::Tree | ChartType::ForceGraph => json!({"type": "point", "filled": true}),
        ChartType::Map => json!({"type": "circle"}),
        ChartType::Table => json!("bar"), // Table is not charted via Vega-Lite.
    }
}

/// Build Vega-Lite encoding from axis mappings.
fn build_encoding(hint: &VizHint) -> Value {
    let mut encoding = json!({});

    if let Some(x) = &hint.axes.x {
        let x_type = semantic_to_vega_type(x.semantic_type);
        encoding["x"] = json!({
            "field": x.name,
            "type": x_type,
        });

        // Horizontal bar: swap x/y
        if hint.chart_type == ChartType::HorizontalBar {
            if let Some(y) = &hint.axes.y {
                let y_type = semantic_to_vega_type(y.semantic_type);
                encoding["x"] = json!({ "field": y.name, "type": y_type });
                encoding["y"] = json!({ "field": x.name, "type": x_type, "sort": "-x" });
                return encoding;
            }
        }

        // Histogram: add bin transform on x.
        if hint.chart_type == ChartType::Histogram {
            encoding["x"]["bin"] = json!(true);
            encoding["y"] = json!({"aggregate": "count"});
            return encoding;
        }
    }

    if let Some(y) = &hint.axes.y {
        let y_type = semantic_to_vega_type(y.semantic_type);
        encoding["y"] = json!({
            "field": y.name,
            "type": y_type,
        });
    }

    if let Some(color) = &hint.axes.color {
        let c_type = semantic_to_vega_type(color.semantic_type);
        // Pie charts use theta for value and color for category
        if hint.chart_type == ChartType::Pie {
            encoding["theta"] = encoding["y"].clone();
            encoding["color"] = json!({
                "field": color.name,
                "type": c_type,
            });
        } else {
            encoding["color"] = json!({
                "field": color.name,
                "type": c_type,
            });
        }
    }

    if let Some(size) = &hint.axes.size {
        encoding["size"] = json!({
            "field": size.name,
            "type": semantic_to_vega_type(size.semantic_type),
        });
    }

    if let Some(label) = &hint.axes.label {
        encoding["tooltip"] = json!([{
            "field": label.name,
            "type": semantic_to_vega_type(label.semantic_type),
        }]);
    }

    if let Some(group) = &hint.axes.group {
        if hint.chart_type == ChartType::StackedBar {
            encoding["color"] = json!({
                "field": group.name,
                "type": "nominal",
            });
        } else if matches!(hint.chart_type, ChartType::Line | ChartType::Area) {
            encoding["color"] = json!({
                "field": group.name,
                "type": "nominal",
            });
        }
    }

    encoding
}

/// Map SemanticType to Vega-Lite data type.
fn semantic_to_vega_type(st: crate::hint::SemanticType) -> &'static str {
    use crate::hint::SemanticType;
    match st {
        SemanticType::Timestamp | SemanticType::Date => "temporal",
        SemanticType::NumericContinuous | SemanticType::Currency | SemanticType::Percentage => {
            "quantitative"
        }
        SemanticType::NumericDiscrete => "quantitative",
        SemanticType::Categorical
        | SemanticType::Boolean
        | SemanticType::Identifier
        | SemanticType::Text => "nominal",
        SemanticType::GeoLatitude | SemanticType::GeoLongitude => "quantitative",
        SemanticType::Unknown => "nominal",
    }
}

/// Convert result rows into Vega-Lite data values format.
fn build_data_values(columns: &[String], rows: &[Vec<serde_json::Value>]) -> Vec<Value> {
    rows.iter()
        .map(|row| {
            let mut obj = serde_json::Map::new();
            for (i, col) in columns.iter().enumerate() {
                let val = row.get(i).cloned().unwrap_or(Value::Null);
                obj.insert(col.clone(), val);
            }
            Value::Object(obj)
        })
        .collect()
}

// ── Colorblind-safe palettes ─────────────────────────────────────────────────

/// Paul Tol's qualitative palette (colorblind-safe, up to 8 colors).
const TOL_QUALITATIVE: &[&str] = &[
    "#4477AA", // blue
    "#EE6677", // red
    "#228833", // green
    "#CCBB44", // yellow
    "#66CCEE", // cyan
    "#AA3377", // purple
    "#BBBBBB", // grey
    "#EE8866", // orange
];

/// IBM Design Language palette (colorblind-safe alternative).
const IBM_PALETTE: &[&str] = &[
    "#648FFF", // ultramarine
    "#DC267F", // magenta
    "#FE6100", // orange
    "#FFB000", // gold
    "#785EF0", // indigo
    "#22C55E", // green
    "#06B6D4", // teal
    "#94A3B8", // slate
];

/// JouleDB branded palette (energy theme).
const JOULEDB_PALETTE: &[&str] = &[
    "#FBBF24", // amber
    "#22C55E", // green
    "#3B82F6", // blue
    "#F97316", // orange
    "#A855F7", // purple
    "#EC4899", // pink
    "#06B6D4", // cyan
    "#84CC16", // lime
];

/// Resolve a named color scheme to a palette.
fn color_palette(scheme_name: &str) -> Vec<String> {
    let colors: &[&str] = match scheme_name {
        "tol" | "colorblind" => TOL_QUALITATIVE,
        "ibm" => IBM_PALETTE,
        "jouledb" => JOULEDB_PALETTE,
        _ => TOL_QUALITATIVE, // default to colorblind-safe
    };
    colors.iter().map(|s| (*s).to_string()).collect()
}

/// Build the Vega-Lite `config` object with our theme.
fn build_theme_config(palette: &[String]) -> Value {
    json!({
        "axis": {
            "labelColor": "#64748B",
            "titleColor": "#E2E8F0",
            "gridColor": "#1E293B",
            "domainColor": "#475569",
            "tickColor": "#475569"
        },
        "legend": {
            "labelColor": "#94A3B8",
            "titleColor": "#E2E8F0"
        },
        "title": {
            "color": "#F1F5F9"
        },
        "view": {
            "stroke": "transparent"
        },
        "background": "transparent",
        "range": {
            "category": palette
        }
    })
}
