//! Natural language text summary renderer.
//!
//! Produces human-readable summaries of query results.

use crate::error::VizResult;
use crate::hint::{ChartType, SemanticType, VizHint};
use crate::render::{RenderConfig, RenderOutput, Renderer};

/// Renderer that produces natural language summaries.
pub struct TextRenderer;

impl Renderer for TextRenderer {
    fn render(
        &self,
        hint: &VizHint,
        columns: &[String],
        rows: &[Vec<serde_json::Value>],
        _config: &RenderConfig,
    ) -> VizResult<RenderOutput> {
        let summary = build_summary(hint, columns, rows);
        Ok(RenderOutput::Text(summary))
    }
}

/// Build a natural language summary of the data.
pub fn build_summary(
    hint: &VizHint,
    columns: &[String],
    rows: &[Vec<serde_json::Value>],
) -> String {
    let profile = &hint.data_profile;
    let mut parts: Vec<String> = Vec::new();

    // Opening sentence
    match hint.chart_type {
        ChartType::Scalar => {
            if let Some(val) = rows.first().and_then(|r| r.first()) {
                parts.push(format!(
                    "The result is a single value: {}.",
                    format_value(val)
                ));
            }
        }
        ChartType::Table => {
            parts.push(format!(
                "Query returned {} rows across {} columns.",
                profile.row_count, profile.col_count,
            ));
        }
        _ => {
            let title = hint.title.as_deref().unwrap_or("the query results");
            parts.push(format!(
                "Showing {} as a {} with {} data points.",
                title,
                hint.chart_type.to_string().replace('_', " "),
                profile.row_count,
            ));
        }
    }

    // Numeric summaries
    for col in &profile.columns {
        if matches!(
            col.semantic_type,
            SemanticType::NumericContinuous | SemanticType::Currency | SemanticType::Percentage
        ) {
            if let (Some(min), Some(max)) = (col.min_value, col.max_value) {
                let col_values: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.get(col.index))
                    .filter_map(|v| v.as_f64())
                    .collect();

                if !col_values.is_empty() {
                    let sum: f64 = col_values.iter().sum();
                    let avg = sum / col_values.len() as f64;

                    parts.push(format!(
                        "{} ranges from {:.2} to {:.2} (avg {:.2}).",
                        col.name, min, max, avg,
                    ));
                }
            }
        }
    }

    // Time series trend
    if profile.is_time_series {
        let numeric_col = profile.columns.iter().find(|c| {
            matches!(
                c.semantic_type,
                SemanticType::NumericContinuous | SemanticType::Currency
            )
        });

        if let Some(nc) = numeric_col {
            let values: Vec<f64> = rows
                .iter()
                .filter_map(|r| r.get(nc.index))
                .filter_map(|v| v.as_f64())
                .collect();

            if values.len() >= 2 {
                let first = values[0];
                let last = values[values.len() - 1];
                if first != 0.0 {
                    let pct_change = ((last - first) / first.abs()) * 100.0;
                    let direction = if pct_change > 0.0 {
                        "increased"
                    } else {
                        "decreased"
                    };
                    parts.push(format!(
                        "{} {} by {:.1}% over the period.",
                        nc.name,
                        direction,
                        pct_change.abs(),
                    ));
                }
            }
        }
    }

    // Categorical breakdown
    if profile.categorical_count >= 1 {
        let cat_col = profile
            .columns
            .iter()
            .find(|c| c.semantic_type == SemanticType::Categorical);

        if let Some(cc) = cat_col {
            parts.push(format!(
                "{} has {} distinct categories.",
                cc.name, cc.distinct_count,
            ));
        }
    }

    // Energy note
    if let Some(energy) = &hint.energy_overlay {
        parts.push(format!(
            "Energy consumed: {:.4} J at {:.1} W ({} efficiency).",
            energy.energy_joules,
            energy.power_watts,
            format!("{:?}", energy.efficiency).to_lowercase(),
        ));
    }

    parts.join(" ")
}

/// Format a JSON value for display in text.
fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 {
                    format!("{}", f as i64)
                } else {
                    format!("{:.2}", f)
                }
            } else {
                n.to_string()
            }
        }
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        _ => v.to_string(),
    }
}
