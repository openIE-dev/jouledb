//! Accessibility renderer for screen readers and assistive technology.
//!
//! Produces alt-text, ARIA metadata, and data summaries optimized
//! for non-visual consumption.

use crate::error::VizResult;
use crate::hint::{AccessibilityHint, ChartType, SemanticType, VizHint};
use crate::render::{RenderConfig, RenderOutput, Renderer};

/// Renderer for accessibility metadata.
pub struct AccessibilityRenderer;

impl Renderer for AccessibilityRenderer {
    fn render(
        &self,
        hint: &VizHint,
        columns: &[String],
        rows: &[Vec<serde_json::Value>],
        _config: &RenderConfig,
    ) -> VizResult<RenderOutput> {
        let a11y = build_accessibility_output(hint, columns, rows);
        let json = serde_json::to_string_pretty(&a11y)
            .map_err(|e| crate::error::VizError::SerializationError(e.to_string()))?;
        Ok(RenderOutput::Json(json))
    }
}

/// Build enhanced accessibility output with data summary.
pub fn build_accessibility_output(
    hint: &VizHint,
    columns: &[String],
    rows: &[Vec<serde_json::Value>],
) -> AccessibilityOutput {
    let profile = &hint.data_profile;

    // Build a data table for screen readers
    let data_table = build_data_table(columns, rows);

    // Build key findings
    let findings = build_key_findings(hint, rows);

    AccessibilityOutput {
        hint: hint.accessibility.clone(),
        data_table,
        key_findings: findings,
        row_count: profile.row_count,
        column_count: profile.col_count,
        chart_type: hint.chart_type.to_string(),
    }
}

/// Structured accessibility output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccessibilityOutput {
    /// Core accessibility hint.
    pub hint: AccessibilityHint,
    /// Tabular data for screen reader navigation.
    pub data_table: DataTable,
    /// Key findings in plain language.
    pub key_findings: Vec<String>,
    /// Total rows.
    pub row_count: usize,
    /// Total columns.
    pub column_count: usize,
    /// Chart type name.
    pub chart_type: String,
}

/// Simple table structure for screen readers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataTable {
    /// Column headers.
    pub headers: Vec<String>,
    /// Row data as strings.
    pub rows: Vec<Vec<String>>,
}

/// Build a screen-reader-friendly data table.
fn build_data_table(columns: &[String], rows: &[Vec<serde_json::Value>]) -> DataTable {
    let max_rows = 20; // Limit for screen reader usability
    let display_rows: Vec<Vec<String>> = rows
        .iter()
        .take(max_rows)
        .map(|row| row.iter().map(|v| value_to_string(v)).collect())
        .collect();

    DataTable {
        headers: columns.to_vec(),
        rows: display_rows,
    }
}

/// Extract key findings for narration.
fn build_key_findings(hint: &VizHint, rows: &[Vec<serde_json::Value>]) -> Vec<String> {
    let mut findings = Vec::new();
    let profile = &hint.data_profile;

    // Size context
    findings.push(format!(
        "Dataset contains {} rows and {} columns.",
        profile.row_count, profile.col_count
    ));

    // Chart recommendation
    let chart_name = hint.chart_type.to_string().replace('_', " ");
    findings.push(format!(
        "Recommended visualization: {} (confidence: {:.0}%).",
        chart_name,
        hint.confidence * 100.0
    ));

    // Numeric extremes
    for col in &profile.columns {
        if matches!(
            col.semantic_type,
            SemanticType::NumericContinuous | SemanticType::Currency
        ) {
            if let (Some(min), Some(max)) = (col.min_value, col.max_value) {
                findings.push(format!(
                    "{}: minimum {:.2}, maximum {:.2}.",
                    col.name, min, max
                ));
            }
        }
    }

    // Time series trend
    if profile.is_time_series {
        if let Some(nc) = profile.columns.iter().find(|c| {
            matches!(
                c.semantic_type,
                SemanticType::NumericContinuous | SemanticType::Currency
            )
        }) {
            let values: Vec<f64> = rows
                .iter()
                .filter_map(|r| r.get(nc.index))
                .filter_map(|v| v.as_f64())
                .collect();

            if values.len() >= 2 {
                let first = values[0];
                let last = values[values.len() - 1];
                if last > first {
                    findings.push(format!("{} shows an upward trend.", nc.name));
                } else if last < first {
                    findings.push(format!("{} shows a downward trend.", nc.name));
                } else {
                    findings.push(format!("{} remains stable.", nc.name));
                }
            }
        }
    }

    // Energy context
    if let Some(energy) = &hint.energy_overlay {
        findings.push(format!(
            "Query energy: {:.4} joules at {:.1} watts.",
            energy.energy_joules, energy.power_watts
        ));
    }

    findings
}

/// Convert a JSON value to a screen-reader-friendly string.
fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "empty".to_string(),
        serde_json::Value::Bool(b) => if *b { "yes" } else { "no" }.to_string(),
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
        serde_json::Value::String(s) => s.clone(),
        _ => v.to_string(),
    }
}
