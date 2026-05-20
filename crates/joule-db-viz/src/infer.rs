//! Visualization inference engine.
//!
//! [`VizInferencer`] takes query metadata and result data, profiles the columns,
//! and returns a [`VizHint`] recommending the best chart type.
//!
//! The engine uses a **weighted scoring** approach (inspired by Draco) rather than
//! a decision tree: every chart type is evaluated against the data profile and
//! query metadata, scored on perceptual accuracy (Cleveland & McGill), data fit,
//! and SQL intent signals. The highest-scoring chart wins.

use crate::data_profile::{DataProfile, build_profile};
use crate::hint::{
    AccessibilityHint, AxisMapping, ChartType, ColumnRef, DensityStrategy, EnergyEfficiency,
    EnergyOverlay, SemanticType, SonificationHint, VizHint,
};

/// Input for visualization inference, decoupled from server types.
#[derive(Debug, Clone)]
pub struct VizInferenceInput {
    /// Column names.
    pub columns: Vec<String>,
    /// Sample rows (JSON values per column).
    pub sample_rows: Vec<Vec<serde_json::Value>>,
    /// Total row count (may exceed sample_rows.len()).
    pub total_rows: usize,
    /// Whether the query uses GROUP BY.
    pub has_group_by: bool,
    /// Whether the query uses ORDER BY.
    pub has_order_by: bool,
    /// Columns in the GROUP BY clause.
    pub group_by_columns: Vec<String>,
    /// Columns in the ORDER BY clause.
    pub order_by_columns: Vec<String>,
    /// Whether the query uses aggregate functions.
    pub has_aggregates: bool,
    /// Names of aggregate functions used (e.g. "SUM", "COUNT").
    pub aggregate_functions: Vec<String>,
    /// Whether the query uses DISTINCT.
    pub has_distinct: bool,
    /// Source table name (if single-table query).
    pub source_table: Option<String>,
    /// Energy consumed in joules (from QueryResponse).
    pub energy_joules: Option<f64>,
    /// Power draw in watts.
    pub power_watts: Option<f64>,
    /// Device that executed the query.
    pub device_target: Option<String>,
    /// Algorithm used.
    pub algorithm_type: Option<String>,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
}

impl Default for VizInferenceInput {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            sample_rows: Vec::new(),
            total_rows: 0,
            has_group_by: false,
            has_order_by: false,
            group_by_columns: Vec::new(),
            order_by_columns: Vec::new(),
            has_aggregates: false,
            aggregate_functions: Vec::new(),
            has_distinct: false,
            source_table: None,
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: None,
            execution_time_ms: 0,
        }
    }
}

/// Inference engine that produces [`VizHint`] from query metadata + data.
pub struct VizInferencer;

/// Internal: scored candidate chart type.
struct ScoredChart {
    chart_type: ChartType,
    score: f32,
    reasoning: String,
}

/// All chart types we score.
const ALL_CHART_TYPES: &[ChartType] = &[
    ChartType::Scalar,
    ChartType::Table,
    ChartType::Line,
    ChartType::Bar,
    ChartType::HorizontalBar,
    ChartType::StackedBar,
    ChartType::Pie,
    ChartType::Scatter,
    ChartType::Heatmap,
    ChartType::Histogram,
    ChartType::Area,
    ChartType::Sparkline,
    ChartType::Gauge,
    ChartType::Tree,
    ChartType::ForceGraph,
    ChartType::Map,
    ChartType::BoxPlot,
    ChartType::EnergyDashboard,
];

impl VizInferencer {
    /// Infer the best visualization for the given input.
    pub fn infer(input: &VizInferenceInput) -> VizHint {
        let profile = build_profile(
            &input.columns,
            &input.sample_rows,
            input.has_group_by,
            input.has_aggregates,
        );

        let (chart_type, confidence, alternatives, reasoning) =
            Self::score_all_charts(&profile, input);
        let axes = Self::build_axes(&profile, chart_type);
        let title = Self::suggest_title(input, chart_type);
        let energy_overlay = Self::build_energy_overlay(input, &profile);
        let accessibility = Self::build_accessibility(&profile, chart_type, &title);
        let sonification = Self::build_sonification(&profile);
        let warnings = Self::detect_warnings(&profile, chart_type, input);
        let density_strategy = Self::suggest_density_strategy(input, &profile, chart_type);

        VizHint {
            chart_type,
            alternatives,
            axes,
            title,
            confidence,
            reasoning,
            data_profile: profile,
            energy_overlay,
            accessibility,
            sonification,
            warnings,
            density_strategy,
        }
    }

    // ── Weighted scorer ──────────────────────────────────────────────────

    /// Score all chart types and return the winner with alternatives.
    fn score_all_charts(
        profile: &DataProfile,
        input: &VizInferenceInput,
    ) -> (ChartType, f32, Vec<ChartType>, String) {
        let mut candidates: Vec<ScoredChart> = ALL_CHART_TYPES
            .iter()
            .map(|&ct| Self::score_chart(ct, profile, input))
            .filter(|s| s.score > 0.0)
            .collect();

        // Sort descending by score.
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if candidates.is_empty() {
            return (
                ChartType::Table,
                0.5,
                vec![],
                "No suitable chart type found; defaulting to table.".to_string(),
            );
        }

        let best = &candidates[0];
        let chart_type = best.chart_type;
        let reasoning = best.reasoning.clone();

        // Confidence: based on absolute score and gap to runner-up.
        let runner_up_score = candidates.get(1).map(|c| c.score).unwrap_or(0.0);
        let margin = best.score - runner_up_score;
        // Base confidence from score, boosted by margin.
        let confidence = (best.score * 0.7 + margin.min(0.3) * 1.0).min(0.99);

        // Alternatives: next best chart types (up to 3, score > 0.2).
        let alternatives: Vec<ChartType> = candidates
            .iter()
            .skip(1)
            .filter(|c| c.score > 0.2)
            .take(3)
            .map(|c| c.chart_type)
            .collect();

        (chart_type, confidence, alternatives, reasoning)
    }

    /// Score a single chart type against the data profile and input.
    fn score_chart(
        chart_type: ChartType,
        profile: &DataProfile,
        input: &VizInferenceInput,
    ) -> ScoredChart {
        let (score, reasoning) = match chart_type {
            ChartType::Scalar => Self::score_scalar(profile, input),
            ChartType::Table => Self::score_table(profile),
            ChartType::Line => Self::score_line(profile, input),
            ChartType::Bar => Self::score_bar(profile, input),
            ChartType::HorizontalBar => Self::score_horizontal_bar(profile, input),
            ChartType::StackedBar => Self::score_stacked_bar(profile, input),
            ChartType::Pie => Self::score_pie(profile, input),
            ChartType::Scatter => Self::score_scatter(profile, input),
            ChartType::Heatmap => Self::score_heatmap(profile, input),
            ChartType::Histogram => Self::score_histogram(profile, input),
            ChartType::Area => Self::score_area(profile, input),
            ChartType::Sparkline => Self::score_sparkline(profile, input),
            ChartType::Gauge => Self::score_gauge(profile, input),
            ChartType::Tree => Self::score_tree(profile),
            ChartType::ForceGraph => Self::score_force_graph(profile),
            ChartType::Map => Self::score_map(profile),
            ChartType::BoxPlot => Self::score_box_plot(profile, input),
            ChartType::EnergyDashboard => Self::score_energy_dashboard(profile, input),
        };
        ScoredChart {
            chart_type,
            score,
            reasoning,
        }
    }

    // ── Per-chart-type scoring functions ──────────────────────────────────

    fn score_scalar(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count == 0 {
            return (0.0, String::new());
        }
        if profile.row_count == 1 && profile.col_count == 1 {
            return (0.98, "Single value result — scalar display.".into());
        }
        if profile.row_count == 1 && profile.col_count <= 4 && profile.numeric_count >= 1 {
            return (
                0.88,
                "Single row with few numeric columns — multi-metric scalar.".into(),
            );
        }
        // COUNT(*) with single aggregate
        if profile.row_count == 1 && input.has_aggregates && input.aggregate_functions.len() == 1 {
            return (0.90, "Single aggregate result.".into());
        }
        (0.0, String::new())
    }

    fn score_table(profile: &DataProfile) -> (f32, String) {
        if profile.row_count == 0 {
            return (0.92, "Empty result — table display.".into());
        }
        // Table is the universal fallback — always viable but low priority.
        let mut score: f32 = 0.30;
        // Boost for wide results (many columns of mixed types).
        if profile.col_count > 6 {
            score += 0.15;
        }
        // Boost for small row counts where a chart isn't warranted.
        if profile.row_count <= 3 && profile.col_count >= 3 {
            score += 0.20;
        }
        (score, "Table is always available as a fallback.".into())
    }

    fn score_line(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if !profile.is_time_series && profile.timestamp_count == 0 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        if profile.is_time_series {
            score += 0.70;
            reasons.push("time series detected");
        }
        if input.has_order_by {
            score += 0.10;
            reasons.push("ORDER BY present");
        }
        if input.has_group_by && group_by_is_temporal(profile, input) {
            score += 0.15;
            reasons.push("GROUP BY timestamp");
        }
        if profile.numeric_count >= 1 {
            score += 0.05;
        }
        // Multi-series: categorical grouping column present.
        if profile.categorical_count >= 1 && profile.timestamp_count >= 1 {
            score += 0.03;
            reasons.push("multi-series line");
        }

        (
            score.min(0.98),
            format!("Line chart: {}", reasons.join(", ")),
        )
    }

    fn score_bar(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count == 0 || profile.row_count == 1 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        if profile.categorical_count >= 1 && profile.numeric_count >= 1 {
            score += 0.65;
            reasons.push("categorical + numeric columns");

            // Prefer vertical bar for moderate category counts.
            let cat = first_categorical(profile);
            if let Some(cat) = cat {
                if cat.distinct_count <= 10 {
                    score += 0.10;
                    reasons.push("moderate category count");
                }
            }
        }

        if input.has_group_by && input.has_aggregates {
            score += 0.10;
            reasons.push("GROUP BY + aggregate");
        }

        // Penalize if time series (line is better).
        if profile.is_time_series {
            score -= 0.25;
        }

        (
            score.max(0.0).min(0.95),
            format!("Bar chart: {}", reasons.join(", ")),
        )
    }

    fn score_horizontal_bar(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count == 0 || profile.row_count == 1 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        if profile.categorical_count >= 1 && profile.numeric_count >= 1 {
            score += 0.55;
            reasons.push("categorical + numeric");

            let cat = first_categorical(profile);
            if let Some(cat) = cat {
                // Horizontal bar shines for many categories (labels need space).
                if cat.distinct_count > 10 {
                    score += 0.25;
                    reasons.push("many categories (>10) — horizontal labels readable");
                } else {
                    // Still acceptable but vertical bar is better.
                    score -= 0.10;
                }
            }
        }

        if input.has_group_by && input.has_aggregates {
            score += 0.05;
        }

        if profile.is_time_series {
            score -= 0.30;
        }

        (
            score.max(0.0).min(0.95),
            format!("Horizontal bar: {}", reasons.join(", ")),
        )
    }

    fn score_stacked_bar(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count <= 1 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // StackedBar needs at least 2 categorical dimensions or group-by with category.
        if input.group_by_columns.len() >= 2 {
            score += 0.70;
            reasons.push("multiple GROUP BY columns");
        }
        if profile.categorical_count >= 2 && profile.numeric_count >= 1 {
            score += 0.60;
            reasons.push("2+ categorical + numeric");
        } else if profile.categorical_count >= 1 && profile.numeric_count >= 1 {
            score += 0.30;
        }

        if input.has_aggregates {
            score += 0.05;
        }
        if profile.is_time_series {
            score -= 0.15;
        }

        (
            score.max(0.0).min(0.90),
            format!("Stacked bar: {}", reasons.join(", ")),
        )
    }

    fn score_pie(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count <= 1 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // Pie requires categorical + numeric, few categories.
        if profile.categorical_count >= 1 && profile.numeric_count >= 1 {
            let cat = first_categorical(profile);
            if let Some(cat) = cat {
                if cat.distinct_count <= 8 {
                    score += 0.65;
                    reasons.push("few categories (≤8)");
                } else {
                    // Too many slices — pie becomes misleading.
                    return (0.0, String::new());
                }
            }
        }

        if input.has_group_by && input.has_aggregates {
            score += 0.10;
            reasons.push("part-to-whole aggregation");
        }

        // Perceptual accuracy penalty: angle encoding is less accurate than position.
        // Cleveland & McGill rank: position > length > angle.
        score -= 0.05;

        if profile.is_time_series {
            score -= 0.30;
        }

        (
            score.max(0.0).min(0.85),
            format!("Pie chart: {}", reasons.join(", ")),
        )
    }

    fn score_scatter(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count <= 1 || profile.numeric_count < 2 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // Two or more numeric columns.
        if profile.numeric_count >= 2 {
            score += 0.65;
            reasons.push("2+ numeric columns");
        }

        // Penalize for wide all-numeric results (heatmap is better).
        if profile.col_count > 4 && profile.numeric_count == profile.col_count {
            score -= 0.20;
            reasons.push("wide all-numeric (heatmap preferred)");
        }

        // Boost if columns are correlated (scatter reveals the relationship).
        if !profile.correlation_pairs.is_empty() {
            let max_r = profile
                .correlation_pairs
                .iter()
                .map(|p| p.r.abs())
                .fold(0.0_f64, f64::max);
            if max_r > 0.5 {
                score += 0.15;
                reasons.push("strong numeric correlation detected");
            }
        }

        // Third numeric column can encode size (bubble chart).
        if profile.numeric_count >= 3 {
            score += 0.05;
            reasons.push("3rd numeric for size encoding");
        }

        // Penalize if time series (line is usually better).
        if profile.is_time_series {
            score -= 0.30;
        }

        // Penalize if has_group_by with aggregates (bar is usually better).
        if input.has_group_by && input.has_aggregates {
            score -= 0.15;
        }

        (
            score.max(0.0).min(0.95),
            format!("Scatter plot: {}", reasons.join(", ")),
        )
    }

    fn score_heatmap(profile: &DataProfile, _input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count <= 1 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // Many numeric columns (>5) — matrix view.
        if profile.col_count > 5 && profile.numeric_count == profile.col_count {
            score += 0.75;
            reasons.push("all-numeric wide result (correlation matrix candidate)");
        }

        // Two categorical + one numeric (pivot table / cross-tab).
        if profile.categorical_count >= 2 && profile.numeric_count >= 1 {
            score += 0.70;
            reasons.push("two categorical + numeric (cross-tab)");
        }

        // Three columns: cat + cat + numeric (e.g., hour × day_of_week × count).
        if profile.col_count == 3 && profile.categorical_count >= 1 && profile.numeric_count >= 1 {
            score += 0.10;
        }

        (
            score.max(0.0).min(0.90),
            format!("Heatmap: {}", reasons.join(", ")),
        )
    }

    fn score_histogram(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count <= 2 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // Single numeric column, no grouping — distribution analysis.
        if profile.col_count == 1 && profile.numeric_count == 1 && !input.has_group_by {
            score += 0.80;
            reasons.push("single numeric column — distribution");
        }

        // Skewed data benefits from histogram.
        let first_numeric = profile
            .columns
            .iter()
            .find(|c| is_numeric(&c.semantic_type));
        if let Some(col) = first_numeric {
            if let Some(skew) = col.skewness {
                if skew.abs() > 1.0 {
                    score += 0.10;
                    reasons.push("skewed distribution");
                }
            }
        }

        // DISTINCT on a single numeric column suggests distribution interest.
        if input.has_distinct && profile.numeric_count == 1 {
            score += 0.10;
        }

        // Penalize if has_group_by (bar chart is better for grouped data).
        if input.has_group_by {
            score -= 0.30;
        }

        (
            score.max(0.0).min(0.90),
            format!("Histogram: {}", reasons.join(", ")),
        )
    }

    fn score_area(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if !profile.is_time_series {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // Area is a variant of Line — always scores slightly below Line.
        score += 0.60;
        reasons.push("time series (area variant)");

        if input.has_order_by {
            score += 0.05;
        }

        // Cumulative/monotonic numeric data is a strong signal for area.
        let numeric = profile
            .columns
            .iter()
            .find(|c| is_numeric(&c.semantic_type));
        if let Some(col) = numeric {
            if col.is_monotonic {
                score += 0.15;
                reasons.push("monotonic values — cumulative area");
            }
        }

        (
            score.min(0.85),
            format!("Area chart: {}", reasons.join(", ")),
        )
    }

    fn score_sparkline(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if !profile.is_time_series {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        score += 0.45;
        reasons.push("time series");

        // Sparklines are best for small, inline displays.
        if profile.row_count <= 60 {
            score += 0.10;
            reasons.push("compact dataset (≤60 points)");
        }
        if input.has_order_by {
            score += 0.03;
        }

        // Always below Line.
        (
            score.min(0.70),
            format!("Sparkline: {}", reasons.join(", ")),
        )
    }

    fn score_gauge(profile: &DataProfile, _input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count != 1 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        if profile.col_count == 1 && profile.numeric_count == 1 {
            score += 0.75;
            reasons.push("single metric");

            // Boost for percentage/ratio semantics.
            let col = &profile.columns[0];
            if col.semantic_type == SemanticType::Percentage {
                score += 0.15;
                reasons.push("percentage value — gauge natural");
            }
        }

        // Gauge is a scalar variant — always below Scalar.
        (score.min(0.92), format!("Gauge: {}", reasons.join(", ")))
    }

    fn score_tree(profile: &DataProfile) -> (f32, String) {
        // Detect hierarchical pattern: columns named id + parent_id.
        let has_id = profile.columns.iter().any(|c| {
            let lower = c.name.to_lowercase();
            lower == "id" || lower.ends_with("_id")
        });
        let has_parent = profile.columns.iter().any(|c| {
            let lower = c.name.to_lowercase();
            lower == "parent_id" || lower == "parent" || lower.ends_with("_parent_id")
        });

        if has_id && has_parent && profile.row_count >= 2 {
            return (0.82, "Hierarchical data (id + parent_id pattern).".into());
        }
        (0.0, String::new())
    }

    fn score_force_graph(profile: &DataProfile) -> (f32, String) {
        // Detect graph pattern: source + target columns.
        let has_source = profile.columns.iter().any(|c| {
            let lower = c.name.to_lowercase();
            lower == "source" || lower == "source_id" || lower == "from" || lower == "from_id"
        });
        let has_target = profile.columns.iter().any(|c| {
            let lower = c.name.to_lowercase();
            lower == "target" || lower == "target_id" || lower == "to" || lower == "to_id"
        });

        if has_source && has_target && profile.row_count >= 2 {
            let mut score = 0.82;
            let mut reason = "Graph data (source + target pattern)".to_string();
            // Weight column boosts.
            let has_weight = profile.columns.iter().any(|c| {
                let lower = c.name.to_lowercase();
                lower == "weight" || lower == "value" || lower == "strength"
            });
            if has_weight {
                score += 0.05;
                reason.push_str(" with edge weights");
            }
            return (score, format!("{reason}."));
        }
        (0.0, String::new())
    }

    fn score_map(profile: &DataProfile) -> (f32, String) {
        if profile.geo_count >= 2 {
            let mut score: f32 = 0.90;
            let reason;
            if profile.numeric_count >= 1 {
                score += 0.05;
                reason = "Geographic coordinates + value column.";
            } else {
                reason = "Geographic coordinates detected.";
            }
            return (score.min(0.97), reason.into());
        }
        (0.0, String::new())
    }

    fn score_box_plot(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        if profile.row_count <= 3 {
            return (0.0, String::new());
        }
        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        // Categorical + numeric — distribution comparison across groups.
        if profile.categorical_count >= 1 && profile.numeric_count >= 1 {
            score += 0.55;
            reasons.push("categorical + numeric");

            // Boost for skewed data or outlier presence.
            let numeric = profile
                .columns
                .iter()
                .find(|c| is_numeric(&c.semantic_type));
            if let Some(col) = numeric {
                if let (Some(q1), Some(q3), Some(min), Some(max)) =
                    (col.q1, col.q3, col.min_value, col.max_value)
                {
                    let iqr = q3 - q1;
                    let has_outliers = min < q1 - 1.5 * iqr || max > q3 + 1.5 * iqr;
                    if has_outliers {
                        score += 0.15;
                        reasons.push("outliers detected (IQR test)");
                    }
                }
                if let Some(skew) = col.skewness {
                    if skew.abs() > 1.0 {
                        score += 0.10;
                        reasons.push("skewed distribution");
                    }
                }
            }
        }

        // Penalize for grouped queries (bar is more natural).
        if input.has_group_by && input.has_aggregates {
            score -= 0.15;
        }

        (
            score.max(0.0).min(0.85),
            format!("Box plot: {}", reasons.join(", ")),
        )
    }

    fn score_energy_dashboard(profile: &DataProfile, input: &VizInferenceInput) -> (f32, String) {
        // Energy dashboard when energy telemetry columns are present in the DATA itself.
        let has_energy_cols = profile.columns.iter().any(|c| {
            let lower = c.name.to_lowercase();
            lower.contains("energy")
                || lower.contains("joules")
                || lower.contains("watt")
                || lower.contains("power")
        });

        if has_energy_cols && profile.is_time_series && profile.row_count > 1 {
            return (
                0.88,
                "Energy telemetry time series — energy dashboard.".into(),
            );
        }
        if has_energy_cols && profile.row_count > 1 {
            return (
                0.65,
                "Energy columns present — energy dashboard candidate.".into(),
            );
        }
        (0.0, String::new())
    }

    // ── Distortion warnings ──────────────────────────────────────────────

    fn detect_warnings(
        profile: &DataProfile,
        chart_type: ChartType,
        input: &VizInferenceInput,
    ) -> Vec<String> {
        let mut warnings = Vec::new();

        // Pie chart with many slices.
        if chart_type == ChartType::Pie {
            if let Some(cat) = first_categorical(profile) {
                if cat.distinct_count > 6 {
                    warnings.push(format!(
                        "Pie chart with {} categories may be hard to read; consider bar chart.",
                        cat.distinct_count
                    ));
                }
            }
        }

        // Skewed data on linear scale.
        for col in &profile.columns {
            if is_numeric(&col.semantic_type) {
                if let Some(skew) = col.skewness {
                    if skew.abs() > 2.0 {
                        warnings.push(format!(
                            "Column '{}' is highly skewed (skewness={:.1}); consider log scale.",
                            col.name, skew
                        ));
                    }
                }
            }
        }

        // Large dataset without aggregation.
        if input.total_rows > 10_000 && !input.has_aggregates {
            warnings.push(format!(
                "Result set has {} rows; server-side aggregation recommended.",
                input.total_rows
            ));
        }

        // Scatter with potential overplotting.
        if chart_type == ChartType::Scatter && profile.row_count > 5000 {
            warnings.push(
                "Large scatter plot may have overplotting; consider hexbin or density.".into(),
            );
        }

        // Bar chart not starting at zero.
        if matches!(chart_type, ChartType::Bar | ChartType::HorizontalBar) {
            let numeric = profile
                .columns
                .iter()
                .find(|c| is_numeric(&c.semantic_type));
            if let Some(col) = numeric {
                if let Some(min) = col.min_value {
                    if min > 0.0 {
                        if let Some(max) = col.max_value {
                            // If min is > 30% of max, the truncated axis may exaggerate differences.
                            if min > max * 0.3 {
                                warnings.push(format!(
                                    "Column '{}' range ({:.0}–{:.0}) — bar chart axis should start at 0.",
                                    col.name, min, max
                                ));
                            }
                        }
                    }
                }
            }
        }

        warnings
    }

    // ── Density strategy ─────────────────────────────────────────────────

    fn suggest_density_strategy(
        input: &VizInferenceInput,
        _profile: &DataProfile,
        chart_type: ChartType,
    ) -> Option<DensityStrategy> {
        let rows = input.total_rows;
        if rows <= 2_000 {
            return None; // Small enough to render directly.
        }

        match chart_type {
            ChartType::Line | ChartType::Area | ChartType::Sparkline => {
                // Time series: LTTB/M4 downsampling.
                Some(DensityStrategy::Downsample {
                    target_points: 1_000,
                })
            }
            ChartType::Scatter => {
                if rows > 10_000 {
                    Some(DensityStrategy::ClientBin { bin_count: 100 })
                } else {
                    None
                }
            }
            ChartType::Histogram => Some(DensityStrategy::ServerAggregate { target_points: 50 }),
            _ => {
                if rows > 10_000 {
                    Some(DensityStrategy::ServerAggregate {
                        target_points: 2_000,
                    })
                } else {
                    None
                }
            }
        }
    }

    // ── Axis mappings ────────────────────────────────────────────────────

    /// Build axis mappings based on chart type and column profiles.
    fn build_axes(profile: &DataProfile, chart_type: ChartType) -> AxisMapping {
        let mut axes = AxisMapping::default();

        if profile.columns.is_empty() {
            return axes;
        }

        match chart_type {
            ChartType::Scalar | ChartType::Gauge => {
                if let Some(c) = profile.columns.first() {
                    axes.y = Some(col_ref(c));
                }
            }
            ChartType::Line | ChartType::Area | ChartType::Sparkline => {
                let temporal = profile
                    .columns
                    .iter()
                    .find(|c| is_temporal(&c.semantic_type));
                let numeric = profile
                    .columns
                    .iter()
                    .find(|c| is_numeric(&c.semantic_type));
                let group_col = profile
                    .columns
                    .iter()
                    .find(|c| is_categorical(&c.semantic_type));

                axes.x = temporal.or(profile.columns.first()).map(col_ref);
                axes.y = numeric.map(col_ref);
                axes.group = group_col.map(col_ref);
            }
            ChartType::Bar | ChartType::HorizontalBar | ChartType::StackedBar => {
                let category = profile
                    .columns
                    .iter()
                    .find(|c| is_categorical(&c.semantic_type));
                let numeric = profile
                    .columns
                    .iter()
                    .find(|c| is_numeric(&c.semantic_type));
                let group_col = profile
                    .columns
                    .iter()
                    .filter(|c| is_categorical(&c.semantic_type))
                    .nth(1);

                axes.x = category.or(profile.columns.first()).map(col_ref);
                axes.y = numeric.map(col_ref);
                axes.color = group_col.map(col_ref);
            }
            ChartType::Pie => {
                let category = profile
                    .columns
                    .iter()
                    .find(|c| is_categorical(&c.semantic_type));
                let numeric = profile
                    .columns
                    .iter()
                    .find(|c| is_numeric(&c.semantic_type));

                axes.color = category.or(profile.columns.first()).map(col_ref);
                axes.y = numeric.map(col_ref);
            }
            ChartType::Scatter => {
                let numerics: Vec<_> = profile
                    .columns
                    .iter()
                    .filter(|c| is_numeric(&c.semantic_type))
                    .collect();

                if let Some(c) = numerics.first() {
                    axes.x = Some(col_ref(c));
                }
                if let Some(c) = numerics.get(1) {
                    axes.y = Some(col_ref(c));
                }
                if let Some(c) = numerics.get(2) {
                    axes.size = Some(col_ref(c));
                }
                let label = profile.columns.iter().find(|c| {
                    is_categorical(&c.semantic_type)
                        || matches!(c.semantic_type, SemanticType::Text)
                });
                axes.label = label.map(col_ref);
            }
            ChartType::Heatmap => {
                if profile.columns.len() >= 2 {
                    axes.x = Some(col_ref(&profile.columns[0]));
                    axes.y = Some(col_ref(&profile.columns[1]));
                }
                if let Some(c) = profile.columns.get(2) {
                    axes.color = Some(col_ref(c));
                }
            }
            ChartType::Histogram => {
                let numeric = profile
                    .columns
                    .iter()
                    .find(|c| is_numeric(&c.semantic_type));
                axes.x = numeric.or(profile.columns.first()).map(col_ref);
            }
            ChartType::Map => {
                let lat = profile
                    .columns
                    .iter()
                    .find(|c| c.semantic_type == SemanticType::GeoLatitude);
                let lon = profile
                    .columns
                    .iter()
                    .find(|c| c.semantic_type == SemanticType::GeoLongitude);
                let value = profile.columns.iter().find(|c| {
                    is_numeric(&c.semantic_type)
                        && c.semantic_type != SemanticType::GeoLatitude
                        && c.semantic_type != SemanticType::GeoLongitude
                });

                axes.x = lon.map(col_ref);
                axes.y = lat.map(col_ref);
                axes.size = value.map(col_ref);
            }
            ChartType::BoxPlot => {
                let category = profile
                    .columns
                    .iter()
                    .find(|c| is_categorical(&c.semantic_type));
                let numeric = profile
                    .columns
                    .iter()
                    .find(|c| is_numeric(&c.semantic_type));

                axes.x = category.map(col_ref);
                axes.y = numeric.map(col_ref);
            }
            ChartType::Tree | ChartType::ForceGraph => {
                // First two columns: id/source + parent/target.
                if let Some(c) = profile.columns.first() {
                    axes.x = Some(col_ref(c));
                }
                if let Some(c) = profile.columns.get(1) {
                    axes.y = Some(col_ref(c));
                }
                // Third column: label or weight.
                if let Some(c) = profile.columns.get(2) {
                    axes.label = Some(col_ref(c));
                }
            }
            ChartType::EnergyDashboard => {
                // Timestamp for X, first energy/power column for Y.
                let temporal = profile
                    .columns
                    .iter()
                    .find(|c| is_temporal(&c.semantic_type));
                let energy_col = profile.columns.iter().find(|c| {
                    let lower = c.name.to_lowercase();
                    lower.contains("energy") || lower.contains("power") || lower.contains("watt")
                });
                axes.x = temporal.or(profile.columns.first()).map(col_ref);
                axes.y = energy_col
                    .or_else(|| {
                        profile
                            .columns
                            .iter()
                            .find(|c| is_numeric(&c.semantic_type))
                    })
                    .map(col_ref);
            }
            _ => {
                // Table and any unknown types.
                if let Some(c) = profile.columns.first() {
                    axes.x = Some(col_ref(c));
                }
                if let Some(c) = profile.columns.get(1) {
                    axes.y = Some(col_ref(c));
                }
            }
        }

        axes
    }

    /// Generate a suggested title from query metadata.
    fn suggest_title(input: &VizInferenceInput, chart_type: ChartType) -> Option<String> {
        let table = input.source_table.as_deref().unwrap_or("results");

        if !input.aggregate_functions.is_empty() && !input.group_by_columns.is_empty() {
            let aggs = input.aggregate_functions.join(", ");
            let groups = input.group_by_columns.join(", ");
            return Some(format!("{aggs} by {groups}"));
        }

        if input.has_group_by && !input.group_by_columns.is_empty() {
            let groups = input.group_by_columns.join(", ");
            return Some(format!("{table} by {groups}"));
        }

        match chart_type {
            ChartType::Scalar => Some(format!("{table} value")),
            ChartType::Table => None,
            _ => Some(table.to_string()),
        }
    }

    /// Build energy overlay if telemetry is available.
    fn build_energy_overlay(
        input: &VizInferenceInput,
        profile: &DataProfile,
    ) -> Option<EnergyOverlay> {
        let energy = input.energy_joules?;
        let power = input.power_watts?;

        let energy_per_row = if profile.row_count > 0 {
            energy / profile.row_count as f64
        } else {
            energy
        };

        let efficiency = if energy_per_row < 0.001 {
            EnergyEfficiency::Excellent
        } else if energy_per_row < 0.01 {
            EnergyEfficiency::Good
        } else if energy_per_row < 0.1 {
            EnergyEfficiency::Fair
        } else {
            EnergyEfficiency::Poor
        };

        Some(EnergyOverlay {
            energy_joules: energy,
            power_watts: power,
            efficiency,
            device: input.device_target.clone(),
            algorithm: input.algorithm_type.clone(),
        })
    }

    /// Generate accessibility metadata.
    fn build_accessibility(
        profile: &DataProfile,
        chart_type: ChartType,
        title: &Option<String>,
    ) -> AccessibilityHint {
        let chart_name = chart_type.to_string().replace('_', " ");
        let title_str = title.as_deref().unwrap_or("query results");

        let alt_text = format!(
            "{chart_name} showing {title_str} with {rows} rows and {cols} columns",
            rows = profile.row_count,
            cols = profile.col_count,
        );

        let mut description = format!("A {chart_name} visualization of {title_str}. ");

        if profile.numeric_count > 0 {
            let numeric_names: Vec<&str> = profile
                .columns
                .iter()
                .filter(|c| is_numeric(&c.semantic_type))
                .map(|c| c.name.as_str())
                .collect();
            description.push_str(&format!("Numeric columns: {}. ", numeric_names.join(", ")));
        }

        if profile.is_time_series {
            description.push_str("Data represents a time series. ");
        }

        let aria_role = match chart_type {
            ChartType::Table => "table".to_string(),
            _ => "img".to_string(),
        };

        AccessibilityHint {
            alt_text,
            description,
            aria_role,
        }
    }

    /// Build sonification hints for auditory representation.
    fn build_sonification(profile: &DataProfile) -> Option<SonificationHint> {
        if !profile.is_time_series && profile.numeric_count < 2 {
            return None;
        }

        let pitch_column = profile
            .columns
            .iter()
            .find(|c| is_numeric(&c.semantic_type))
            .map(|c| c.index);

        let volume_column = profile
            .columns
            .iter()
            .filter(|c| is_numeric(&c.semantic_type))
            .nth(1)
            .map(|c| c.index);

        Some(SonificationHint {
            pitch_column,
            volume_column,
            base_frequency: 220.0,
            frequency_range: (110.0, 880.0),
            midi_program: 0,
            note_duration_ms: 200,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Check if a GROUP BY column is temporal.
fn group_by_is_temporal(profile: &DataProfile, input: &VizInferenceInput) -> bool {
    input.group_by_columns.iter().any(|g| {
        profile.columns.iter().any(|c| {
            c.name.eq_ignore_ascii_case(g)
                && matches!(
                    c.semantic_type,
                    SemanticType::Timestamp | SemanticType::Date
                )
        })
    })
}

/// Get the first categorical column profile.
fn first_categorical(profile: &DataProfile) -> Option<&crate::data_profile::ColumnProfile> {
    profile
        .columns
        .iter()
        .find(|c| matches!(c.semantic_type, SemanticType::Categorical))
}

fn is_temporal(st: &SemanticType) -> bool {
    matches!(st, SemanticType::Timestamp | SemanticType::Date)
}

fn is_numeric(st: &SemanticType) -> bool {
    matches!(
        st,
        SemanticType::NumericContinuous
            | SemanticType::NumericDiscrete
            | SemanticType::Currency
            | SemanticType::Percentage
    )
}

fn is_categorical(st: &SemanticType) -> bool {
    matches!(
        st,
        SemanticType::Categorical | SemanticType::Boolean | SemanticType::Text
    )
}

fn col_ref(cp: &crate::data_profile::ColumnProfile) -> ColumnRef {
    ColumnRef {
        index: cp.index,
        name: cp.name.clone(),
        semantic_type: cp.semantic_type,
    }
}
