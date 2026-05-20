//! Comprehensive tests for joule-db-viz crate.

use serde_json::json;

use joule_db_viz::column_classifier::classify_column;
use joule_db_viz::data_profile::build_profile;
use joule_db_viz::render::accessibility::build_accessibility_output;
use joule_db_viz::render::text::build_summary;
use joule_db_viz::render::vega::build_vega_spec;
use joule_db_viz::render::{RenderConfig, Renderer};
use joule_db_viz::{
    ChartType, EnergyEfficiency, SemanticType, VizError, VizInferenceInput, VizInferencer,
    VizResult,
};

// ═══════════════════════════════════════════════════════════════════════════
// Column classifier tests (15)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classify_timestamp_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(
        classify_column("created_at", &vals),
        SemanticType::Timestamp
    );
    assert_eq!(
        classify_column("updated_at", &vals),
        SemanticType::Timestamp
    );
    assert_eq!(
        classify_column("event_timestamp", &vals),
        SemanticType::Timestamp
    );
    assert_eq!(classify_column("log_time", &vals), SemanticType::Timestamp);
}

#[test]
fn classify_date_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("date", &vals), SemanticType::Date);
    assert_eq!(classify_column("birth_date", &vals), SemanticType::Date);
}

#[test]
fn classify_identifier_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("id", &vals), SemanticType::Identifier);
    assert_eq!(classify_column("user_id", &vals), SemanticType::Identifier);
    assert_eq!(classify_column("uuid", &vals), SemanticType::Identifier);
}

#[test]
fn classify_latitude_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("lat", &vals), SemanticType::GeoLatitude);
    assert_eq!(
        classify_column("latitude", &vals),
        SemanticType::GeoLatitude
    );
    assert_eq!(
        classify_column("store_lat", &vals),
        SemanticType::GeoLatitude
    );
}

#[test]
fn classify_longitude_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("lon", &vals), SemanticType::GeoLongitude);
    assert_eq!(
        classify_column("longitude", &vals),
        SemanticType::GeoLongitude
    );
    assert_eq!(
        classify_column("store_lng", &vals),
        SemanticType::GeoLongitude
    );
}

#[test]
fn classify_currency_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("price", &vals), SemanticType::Currency);
    assert_eq!(classify_column("revenue", &vals), SemanticType::Currency);
    assert_eq!(classify_column("total_cost", &vals), SemanticType::Currency);
}

#[test]
fn classify_percentage_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("percent", &vals), SemanticType::Percentage);
    assert_eq!(
        classify_column("click_rate", &vals),
        SemanticType::Percentage
    );
    assert_eq!(
        classify_column("conversion_pct", &vals),
        SemanticType::Percentage
    );
}

#[test]
fn classify_boolean_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("is_active", &vals), SemanticType::Boolean);
    assert_eq!(
        classify_column("has_subscription", &vals),
        SemanticType::Boolean
    );
    assert_eq!(classify_column("active", &vals), SemanticType::Boolean);
}

#[test]
fn classify_categorical_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("status", &vals), SemanticType::Categorical);
    assert_eq!(
        classify_column("category", &vals),
        SemanticType::Categorical
    );
    assert_eq!(classify_column("region", &vals), SemanticType::Categorical);
    assert_eq!(classify_column("country", &vals), SemanticType::Categorical);
}

#[test]
fn classify_text_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(classify_column("description", &vals), SemanticType::Text);
    assert_eq!(classify_column("name", &vals), SemanticType::Text);
    assert_eq!(classify_column("email", &vals), SemanticType::Text);
}

#[test]
fn classify_aggregation_by_name() {
    let vals: Vec<&serde_json::Value> = vec![];
    assert_eq!(
        classify_column("count", &vals),
        SemanticType::NumericContinuous
    );
    assert_eq!(
        classify_column("sum", &vals),
        SemanticType::NumericContinuous
    );
    assert_eq!(
        classify_column("avg", &vals),
        SemanticType::NumericContinuous
    );
    assert_eq!(
        classify_column("total_count", &vals),
        SemanticType::NumericContinuous
    );
}

#[test]
fn classify_boolean_by_values() {
    let v1 = json!(true);
    let v2 = json!(false);
    let v3 = json!(true);
    let vals: Vec<&serde_json::Value> = vec![&v1, &v2, &v3];
    assert_eq!(classify_column("foo", &vals), SemanticType::Boolean);
}

#[test]
fn classify_numeric_by_values() {
    let v1 = json!(1.5);
    let v2 = json!(2.7);
    let v3 = json!(3.9);
    let vals: Vec<&serde_json::Value> = vec![&v1, &v2, &v3];
    assert_eq!(
        classify_column("foo", &vals),
        SemanticType::NumericContinuous
    );
}

#[test]
fn classify_timestamp_by_values() {
    let v1 = json!("2024-01-15T10:30:00Z");
    let v2 = json!("2024-01-16T11:00:00Z");
    let vals: Vec<&serde_json::Value> = vec![&v1, &v2];
    assert_eq!(classify_column("foo", &vals), SemanticType::Timestamp);
}

#[test]
fn classify_uuid_by_values() {
    let v1 = json!("550e8400-e29b-41d4-a716-446655440000");
    let v2 = json!("6ba7b810-9dad-11d1-80b4-00c04fd430c8");
    let vals: Vec<&serde_json::Value> = vec![&v1, &v2];
    assert_eq!(classify_column("foo", &vals), SemanticType::Identifier);
}

// ═══════════════════════════════════════════════════════════════════════════
// Data profile tests (5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn profile_empty_result() {
    let profile = build_profile(&[], &[], false, false);
    assert_eq!(profile.row_count, 0);
    assert_eq!(profile.col_count, 0);
    assert!(!profile.is_time_series);
}

#[test]
fn profile_numeric_columns() {
    let cols = vec!["price".to_string(), "quantity".to_string()];
    let rows = vec![
        vec![json!(10.0), json!(5)],
        vec![json!(20.0), json!(3)],
        vec![json!(30.0), json!(8)],
    ];
    let profile = build_profile(&cols, &rows, false, false);
    assert_eq!(profile.numeric_count, 2);
    assert_eq!(profile.row_count, 3);
    assert_eq!(profile.col_count, 2);
}

#[test]
fn profile_time_series_detected() {
    let cols = vec!["created_at".to_string(), "revenue".to_string()];
    let rows = vec![
        vec![json!("2024-01-01T00:00:00Z"), json!(100)],
        vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        vec![json!("2024-03-01T00:00:00Z"), json!(300)],
    ];
    let profile = build_profile(&cols, &rows, false, false);
    assert!(profile.is_time_series);
    assert_eq!(profile.timestamp_count, 1);
}

#[test]
fn profile_categorical_columns() {
    let cols = vec!["status".to_string(), "count".to_string()];
    let rows = vec![
        vec![json!("active"), json!(42)],
        vec![json!("inactive"), json!(18)],
        vec![json!("pending"), json!(7)],
    ];
    let profile = build_profile(&cols, &rows, true, true);
    assert_eq!(profile.categorical_count, 1);
    assert!(profile.has_group_by);
    assert!(profile.has_aggregates);
}

#[test]
fn profile_geo_columns() {
    let cols = vec!["lat".to_string(), "lon".to_string(), "revenue".to_string()];
    let rows = vec![
        vec![json!(40.7128), json!(-74.0060), json!(1000)],
        vec![json!(34.0522), json!(-118.2437), json!(2000)],
    ];
    let profile = build_profile(&cols, &rows, false, false);
    assert_eq!(profile.geo_count, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Inference tests (25)
// ═══════════════════════════════════════════════════════════════════════════

fn make_input(columns: Vec<&str>, rows: Vec<Vec<serde_json::Value>>) -> VizInferenceInput {
    VizInferenceInput {
        columns: columns.into_iter().map(|s| s.to_string()).collect(),
        total_rows: rows.len(),
        sample_rows: rows,
        ..Default::default()
    }
}

#[test]
fn infer_empty_result_is_table() {
    let input = make_input(vec!["id", "name"], vec![]);
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Table);
    assert!(hint.confidence >= 0.8);
}

#[test]
fn infer_single_value_is_scalar() {
    let input = make_input(vec!["count"], vec![vec![json!(42)]]);
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Scalar);
    assert!(hint.confidence >= 0.9);
}

#[test]
fn infer_single_row_few_cols_is_scalar() {
    let input = make_input(
        vec!["count", "sum", "avg"],
        vec![vec![json!(100), json!(5000), json!(50.0)]],
    );
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Scalar);
}

#[test]
fn infer_group_by_sum_is_bar_or_pie() {
    let mut input = make_input(
        vec!["region", "revenue"],
        vec![
            vec![json!("North"), json!(1000)],
            vec![json!("South"), json!(800)],
            vec![json!("East"), json!(1200)],
        ],
    );
    input.has_group_by = true;
    input.has_aggregates = true;
    input.group_by_columns = vec!["region".to_string()];
    input.aggregate_functions = vec!["SUM".to_string()];

    let hint = VizInferencer::infer(&input);
    assert!(
        matches!(hint.chart_type, ChartType::Pie | ChartType::Bar),
        "Expected Pie or Bar, got {:?}",
        hint.chart_type
    );
}

#[test]
fn infer_group_by_timestamp_is_line() {
    let mut input = make_input(
        vec!["created_at", "count"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(10)],
            vec![json!("2024-02-01T00:00:00Z"), json!(20)],
            vec![json!("2024-03-01T00:00:00Z"), json!(30)],
        ],
    );
    input.has_group_by = true;
    input.has_aggregates = true;
    input.group_by_columns = vec!["created_at".to_string()];
    input.aggregate_functions = vec!["COUNT".to_string()];

    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Line);
}

#[test]
fn infer_order_by_timestamp_is_line() {
    let mut input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
            vec![json!("2024-03-01T00:00:00Z"), json!(300)],
        ],
    );
    input.has_order_by = true;
    input.order_by_columns = vec!["created_at".to_string()];

    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Line);
}

#[test]
fn infer_two_col_category_numeric_is_bar() {
    let input = make_input(
        vec!["category", "count"],
        vec![
            vec![json!("A"), json!(10)],
            vec![json!("B"), json!(20)],
            vec![json!("C"), json!(30)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert!(
        matches!(hint.chart_type, ChartType::Bar | ChartType::HorizontalBar),
        "Expected Bar or HorizontalBar, got {:?}",
        hint.chart_type
    );
}

#[test]
fn infer_two_col_numeric_numeric_is_scatter() {
    let input = make_input(
        vec!["height_cm", "weight_kg"],
        vec![
            vec![json!(170), json!(70)],
            vec![json!(165), json!(60)],
            vec![json!(180), json!(85)],
            vec![json!(175), json!(75)],
            vec![json!(160), json!(55)],
            vec![json!(190), json!(90)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Scatter);
}

#[test]
fn infer_two_col_timestamp_numeric_is_line() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
            vec![json!("2024-03-01T00:00:00Z"), json!(300)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Line);
}

#[test]
fn infer_three_col_lat_lon_value_is_map() {
    let input = make_input(
        vec!["lat", "lon", "revenue"],
        vec![
            vec![json!(40.7128), json!(-74.0060), json!(1000)],
            vec![json!(34.0522), json!(-118.2437), json!(2000)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Map);
}

#[test]
fn infer_three_numeric_is_scatter() {
    let input = make_input(
        vec!["x_val", "y_val", "z_val"],
        vec![
            vec![json!(1.0), json!(2.0), json!(3.0)],
            vec![json!(4.0), json!(5.0), json!(6.0)],
            vec![json!(7.0), json!(8.0), json!(9.0)],
            vec![json!(10.0), json!(11.0), json!(12.0)],
            vec![json!(13.0), json!(14.0), json!(15.0)],
            vec![json!(16.0), json!(17.0), json!(18.0)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Scatter);
}

#[test]
fn infer_many_numeric_cols_is_heatmap() {
    let input = make_input(
        vec!["c1", "c2", "c3", "c4", "c5", "c6"],
        vec![
            vec![json!(1), json!(2), json!(3), json!(4), json!(5), json!(6)],
            vec![
                json!(7),
                json!(8),
                json!(9),
                json!(10),
                json!(11),
                json!(12),
            ],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::Heatmap);
}

#[test]
fn infer_many_categories_is_horizontal_bar() {
    let mut rows = Vec::new();
    for i in 0..15 {
        rows.push(vec![json!(format!("item_{}", i)), json!(i * 10)]);
    }
    let input = make_input(vec!["category", "count"], rows);
    let hint = VizInferencer::infer(&input);
    assert_eq!(hint.chart_type, ChartType::HorizontalBar);
}

#[test]
fn infer_confidence_decreases_for_ambiguous_data() {
    let input = make_input(
        vec!["a", "b", "c", "d", "e"],
        vec![vec![
            json!("x"),
            json!(1),
            json!("y"),
            json!(true),
            json!(null),
        ]],
    );
    let hint = VizInferencer::infer(&input);
    assert!(
        hint.confidence <= 0.85,
        "Confidence should be moderate for mixed data"
    );
}

#[test]
fn infer_alternatives_are_non_empty_for_common_charts() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert!(
        !hint.alternatives.is_empty(),
        "Line chart should have alternatives"
    );
}

#[test]
fn infer_energy_overlay_present() {
    let mut input = make_input(vec!["count"], vec![vec![json!(42)]]);
    input.energy_joules = Some(0.005);
    input.power_watts = Some(15.0);
    input.device_target = Some("Apple M4 Max".to_string());
    input.algorithm_type = Some("scan".to_string());

    let hint = VizInferencer::infer(&input);
    assert!(hint.energy_overlay.is_some());
    let overlay = hint.energy_overlay.unwrap();
    assert_eq!(overlay.energy_joules, 0.005);
    assert_eq!(overlay.power_watts, 15.0);
    assert_eq!(overlay.device.as_deref(), Some("Apple M4 Max"));
}

#[test]
fn infer_energy_efficiency_classification() {
    // Excellent: < 1 mJ per row
    let mut input = make_input(vec!["revenue"], vec![vec![json!(100)]; 10]);
    input.energy_joules = Some(0.001); // 0.1 mJ per row
    input.power_watts = Some(10.0);

    let hint = VizInferencer::infer(&input);
    let overlay = hint.energy_overlay.unwrap();
    assert_eq!(overlay.efficiency, EnergyEfficiency::Excellent);
}

#[test]
fn infer_energy_efficiency_poor() {
    let mut input = make_input(vec!["revenue"], vec![vec![json!(100)]]);
    input.energy_joules = Some(1.0); // 1000 mJ per row
    input.power_watts = Some(50.0);

    let hint = VizInferencer::infer(&input);
    let overlay = hint.energy_overlay.unwrap();
    assert_eq!(overlay.efficiency, EnergyEfficiency::Poor);
}

#[test]
fn infer_no_energy_overlay_when_missing() {
    let input = make_input(vec!["count"], vec![vec![json!(42)]]);
    let hint = VizInferencer::infer(&input);
    assert!(hint.energy_overlay.is_none());
}

#[test]
fn infer_title_from_aggregation() {
    let mut input = make_input(
        vec!["region", "revenue"],
        vec![
            vec![json!("North"), json!(1000)],
            vec![json!("South"), json!(800)],
        ],
    );
    input.has_group_by = true;
    input.has_aggregates = true;
    input.group_by_columns = vec!["region".to_string()];
    input.aggregate_functions = vec!["SUM".to_string()];

    let hint = VizInferencer::infer(&input);
    assert!(hint.title.is_some());
    let title = hint.title.unwrap();
    assert!(
        title.contains("SUM"),
        "Title should mention aggregation: {}",
        title
    );
    assert!(
        title.contains("region"),
        "Title should mention group: {}",
        title
    );
}

#[test]
fn infer_accessibility_always_present() {
    let input = make_input(vec!["x", "y"], vec![vec![json!(1), json!(2)]]);
    let hint = VizInferencer::infer(&input);
    assert!(!hint.accessibility.alt_text.is_empty());
    assert!(!hint.accessibility.description.is_empty());
    assert!(!hint.accessibility.aria_role.is_empty());
}

#[test]
fn infer_accessibility_table_role() {
    let input = make_input(
        vec!["a", "b", "c", "d", "e", "f"],
        vec![vec![
            json!("x"),
            json!(1),
            json!("y"),
            json!(2),
            json!("z"),
            json!(3),
        ]],
    );
    let hint = VizInferencer::infer(&input);
    if hint.chart_type == ChartType::Table {
        assert_eq!(hint.accessibility.aria_role, "table");
    }
}

#[test]
fn infer_sonification_for_timeseries() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
            vec![json!("2024-03-01T00:00:00Z"), json!(300)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert!(hint.sonification.is_some());
    let son = hint.sonification.unwrap();
    assert!(son.pitch_column.is_some());
    assert!(son.base_frequency > 0.0);
}

#[test]
fn infer_stacked_bar_for_multi_group() {
    let mut input = make_input(
        vec!["region", "category", "revenue"],
        vec![
            vec![json!("North"), json!("Electronics"), json!(1000)],
            vec![json!("South"), json!("Clothing"), json!(800)],
        ],
    );
    input.has_group_by = true;
    input.has_aggregates = true;
    input.group_by_columns = vec!["region".to_string(), "category".to_string()];
    input.aggregate_functions = vec!["SUM".to_string()];

    let hint = VizInferencer::infer(&input);
    assert!(
        matches!(
            hint.chart_type,
            ChartType::StackedBar | ChartType::Bar | ChartType::Pie
        ),
        "Expected StackedBar/Bar/Pie for multi-group, got {:?}",
        hint.chart_type
    );
}

#[test]
fn infer_axes_line_chart() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    assert!(hint.axes.x.is_some());
    assert!(hint.axes.y.is_some());
    assert_eq!(hint.axes.x.as_ref().unwrap().name, "created_at");
    assert_eq!(hint.axes.y.as_ref().unwrap().name, "revenue");
}

// ═══════════════════════════════════════════════════════════════════════════
// Vega-Lite renderer tests (10)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn vega_spec_has_schema() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();

    let spec = build_vega_spec(
        &hint,
        &["created_at".into(), "revenue".into()],
        &input.sample_rows,
        &config,
    )
    .unwrap();

    assert_eq!(
        spec["$schema"],
        "https://vega.github.io/schema/vega-lite/v5.json"
    );
}

#[test]
fn vega_line_chart_mark() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();

    let spec = build_vega_spec(
        &hint,
        &["created_at".into(), "revenue".into()],
        &input.sample_rows,
        &config,
    )
    .unwrap();

    assert!(
        spec["mark"]["type"] == "line",
        "Expected line mark, got {:?}",
        spec["mark"]
    );
}

#[test]
fn vega_bar_chart_mark() {
    let input = make_input(
        vec!["category", "count"],
        vec![
            vec![json!("A"), json!(10)],
            vec![json!("B"), json!(20)],
            vec![json!("C"), json!(30)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();
    let cols: Vec<String> = vec!["category".into(), "count".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert!(
        spec["mark"] == "bar" || spec["mark"]["type"] == "bar",
        "Expected bar mark, got {:?}",
        spec["mark"]
    );
}

#[test]
fn vega_scatter_mark() {
    let input = make_input(
        vec!["height_cm", "weight_kg"],
        vec![
            vec![json!(170), json!(70)],
            vec![json!(165), json!(60)],
            vec![json!(180), json!(85)],
            vec![json!(175), json!(75)],
            vec![json!(160), json!(55)],
            vec![json!(190), json!(90)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();
    let cols: Vec<String> = vec!["height_cm".into(), "weight_kg".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert_eq!(spec["mark"]["type"], "point");
}

#[test]
fn vega_spec_has_encoding() {
    let input = make_input(
        vec!["category", "count"],
        vec![vec![json!("A"), json!(10)], vec![json!("B"), json!(20)]],
    );
    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();
    let cols: Vec<String> = vec!["category".into(), "count".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert!(spec["encoding"].is_object(), "Spec should have encoding");
}

#[test]
fn vega_spec_inline_data() {
    let input = make_input(vec!["x", "y"], vec![vec![json!(1), json!(2)]]);
    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();
    let cols: Vec<String> = vec!["x".into(), "y".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert!(
        spec["data"]["values"].is_array(),
        "Should have inline data values"
    );
}

#[test]
fn vega_spec_no_inline_data_when_disabled() {
    let input = make_input(vec!["x", "y"], vec![vec![json!(1), json!(2)]]);
    let hint = VizInferencer::infer(&input);
    let mut config = RenderConfig::default();
    config.inline_data = false;
    let cols: Vec<String> = vec!["x".into(), "y".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert!(
        spec.get("data").is_none(),
        "Should not have data when inline_data=false"
    );
}

#[test]
fn vega_spec_has_title_when_available() {
    let mut input = make_input(
        vec!["region", "revenue"],
        vec![
            vec![json!("North"), json!(1000)],
            vec![json!("South"), json!(800)],
        ],
    );
    input.has_group_by = true;
    input.has_aggregates = true;
    input.group_by_columns = vec!["region".to_string()];
    input.aggregate_functions = vec!["SUM".to_string()];

    let hint = VizInferencer::infer(&input);
    let config = RenderConfig::default();
    let cols: Vec<String> = vec!["region".into(), "revenue".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert!(
        spec.get("title").is_some(),
        "Should have title for grouped query"
    );
}

#[test]
fn vega_spec_width_height() {
    let input = make_input(vec!["x"], vec![vec![json!(1)]]);
    let hint = VizInferencer::infer(&input);
    let mut config = RenderConfig::default();
    config.width = 800;
    config.height = 600;
    let cols: Vec<String> = vec!["x".into()];

    let spec = build_vega_spec(&hint, &cols, &input.sample_rows, &config).unwrap();
    assert_eq!(spec["width"], 800);
    assert_eq!(spec["height"], 600);
}

#[test]
fn vega_spec_is_valid_json() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let renderer = joule_db_viz::VegaRenderer;
    let config = RenderConfig::default();
    let cols: Vec<String> = vec!["created_at".into(), "revenue".into()];

    let output = renderer
        .render(&hint, &cols, &input.sample_rows, &config)
        .unwrap();
    if let joule_db_viz::RenderOutput::Json(json_str) = output {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
        assert!(parsed.is_ok(), "Vega output must be valid JSON");
    } else {
        panic!("VegaRenderer should produce Json output");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Text summary renderer tests (6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn text_summary_scalar() {
    let input = make_input(vec!["count"], vec![vec![json!(42)]]);
    let hint = VizInferencer::infer(&input);
    let summary = build_summary(&hint, &["count".into()], &input.sample_rows);
    assert!(
        summary.contains("42"),
        "Summary should contain the scalar value"
    );
}

#[test]
fn text_summary_table() {
    let input = make_input(
        vec!["status", "count"],
        vec![
            vec![json!("active"), json!(42)],
            vec![json!("inactive"), json!(18)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["status".into(), "count".into()];
    let summary = build_summary(&hint, &cols, &input.sample_rows);
    assert!(!summary.is_empty(), "Summary should not be empty");
}

#[test]
fn text_summary_time_series_trend() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100.0)],
            vec![json!("2024-02-01T00:00:00Z"), json!(150.0)],
            vec![json!("2024-03-01T00:00:00Z"), json!(200.0)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["created_at".into(), "revenue".into()];
    let summary = build_summary(&hint, &cols, &input.sample_rows);
    assert!(
        summary.contains("increased") || summary.contains("trend"),
        "Summary should mention trend: {}",
        summary
    );
}

#[test]
fn text_summary_numeric_range() {
    let input = make_input(
        vec!["price"],
        vec![vec![json!(10.0)], vec![json!(20.0)], vec![json!(30.0)]],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["price".into()];
    let summary = build_summary(&hint, &cols, &input.sample_rows);
    assert!(
        summary.contains("10") && summary.contains("30"),
        "Summary should show range: {}",
        summary
    );
}

#[test]
fn text_summary_energy_info() {
    let mut input = make_input(vec!["count"], vec![vec![json!(42)]]);
    input.energy_joules = Some(0.005);
    input.power_watts = Some(15.0);

    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["count".into()];
    let summary = build_summary(&hint, &cols, &input.sample_rows);
    assert!(
        summary.contains("Energy") || summary.contains("energy") || summary.contains("joules"),
        "Summary should mention energy: {}",
        summary
    );
}

#[test]
fn text_summary_in_english() {
    let input = make_input(
        vec!["category", "count"],
        vec![vec![json!("A"), json!(10)], vec![json!("B"), json!(20)]],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["category".into(), "count".into()];
    let summary = build_summary(&hint, &cols, &input.sample_rows);
    // Should be valid English (contains words, no raw JSON)
    assert!(summary.contains(" "), "Summary should be natural language");
    assert!(!summary.starts_with('{'), "Summary should not be raw JSON");
}

// ═══════════════════════════════════════════════════════════════════════════
// Accessibility renderer tests (5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn accessibility_alt_text_includes_data_info() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["created_at".into(), "revenue".into()];

    let output = build_accessibility_output(&hint, &cols, &input.sample_rows);
    assert!(
        output.hint.alt_text.contains("2 rows"),
        "Alt text should mention row count: {}",
        output.hint.alt_text
    );
    assert!(
        output.hint.alt_text.contains("2 columns"),
        "Alt text should mention col count: {}",
        output.hint.alt_text
    );
}

#[test]
fn accessibility_aria_role_correct() {
    // Table chart → "table" role
    let input = make_input(
        vec!["a", "b", "c", "d", "e", "f"],
        vec![vec![
            json!("x"),
            json!(1),
            json!("y"),
            json!(2),
            json!("z"),
            json!(3),
        ]],
    );
    let hint = VizInferencer::infer(&input);
    if hint.chart_type == ChartType::Table {
        assert_eq!(hint.accessibility.aria_role, "table");
    } else {
        assert_eq!(hint.accessibility.aria_role, "img");
    }
}

#[test]
fn accessibility_data_table_headers() {
    let input = make_input(
        vec!["status", "count"],
        vec![
            vec![json!("active"), json!(42)],
            vec![json!("inactive"), json!(18)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["status".into(), "count".into()];

    let output = build_accessibility_output(&hint, &cols, &input.sample_rows);
    assert_eq!(output.data_table.headers, vec!["status", "count"]);
    assert_eq!(output.data_table.rows.len(), 2);
}

#[test]
fn accessibility_key_findings_non_empty() {
    let input = make_input(vec!["revenue"], vec![vec![json!(100)], vec![json!(200)]]);
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["revenue".into()];

    let output = build_accessibility_output(&hint, &cols, &input.sample_rows);
    assert!(!output.key_findings.is_empty(), "Should have key findings");
    assert!(
        output.key_findings.iter().any(|f| f.contains("rows")),
        "Findings should mention row count"
    );
}

#[test]
fn accessibility_output_serializable() {
    let input = make_input(
        vec!["category", "count"],
        vec![vec![json!("A"), json!(10)], vec![json!("B"), json!(20)]],
    );
    let hint = VizInferencer::infer(&input);
    let cols: Vec<String> = vec!["category".into(), "count".into()];

    let output = build_accessibility_output(&hint, &cols, &input.sample_rows);
    let json = serde_json::to_string(&output);
    assert!(json.is_ok(), "Accessibility output should be serializable");
}

// ═══════════════════════════════════════════════════════════════════════════
// Serialization tests (4)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn viz_hint_serializes_to_json() {
    let input = make_input(
        vec!["created_at", "revenue"],
        vec![
            vec![json!("2024-01-01T00:00:00Z"), json!(100)],
            vec![json!("2024-02-01T00:00:00Z"), json!(200)],
        ],
    );
    let hint = VizInferencer::infer(&input);
    let json = serde_json::to_string_pretty(&hint);
    assert!(json.is_ok(), "VizHint should serialize to JSON");
}

#[test]
fn viz_hint_roundtrip() {
    let input = make_input(
        vec!["category", "count"],
        vec![vec![json!("A"), json!(10)], vec![json!("B"), json!(20)]],
    );
    let hint = VizInferencer::infer(&input);
    let json_str = serde_json::to_string(&hint).unwrap();
    let deserialized: joule_db_viz::VizHint = serde_json::from_str(&json_str).unwrap();
    assert_eq!(hint, deserialized, "VizHint should survive JSON roundtrip");
}

#[test]
fn chart_type_serde_snake_case() {
    let json = serde_json::to_string(&ChartType::HorizontalBar).unwrap();
    assert_eq!(json, "\"horizontal_bar\"");

    let json = serde_json::to_string(&ChartType::EnergyDashboard).unwrap();
    assert_eq!(json, "\"energy_dashboard\"");

    let json = serde_json::to_string(&ChartType::ForceGraph).unwrap();
    assert_eq!(json, "\"force_graph\"");
}

#[test]
fn semantic_type_serde_snake_case() {
    let json = serde_json::to_string(&SemanticType::NumericContinuous).unwrap();
    assert_eq!(json, "\"numeric_continuous\"");

    let json = serde_json::to_string(&SemanticType::GeoLatitude).unwrap();
    assert_eq!(json, "\"geo_latitude\"");
}

// ═══════════════════════════════════════════════════════════════════════════
// Error type tests (2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn viz_error_display() {
    let err = VizError::IncompatibleData {
        chart_type: "pie".to_string(),
        reason: "no numeric columns".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("pie"), "Error should mention chart type");
    assert!(msg.contains("no numeric"), "Error should mention reason");
}

#[test]
fn viz_error_variants() {
    let _: VizError = VizError::RenderError("test".to_string());
    let _: VizError = VizError::GpuError("test".to_string());
    let _: VizError = VizError::SerializationError("test".to_string());
    let _: VizError = VizError::InvalidConfig("test".to_string());
    let _: VizError = VizError::DataError("test".to_string());
}

// ═══════════════════════════════════════════════════════════════════════════
// GpuRenderer stub tests (2)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gpu_renderer_stub_exists() {
    let _renderer = joule_db_viz::GpuRenderer::new();
}

#[tokio::test]
async fn gpu_renderer_stub_returns_error() {
    let renderer = joule_db_viz::GpuRenderer::new();
    let result = renderer
        .render_chart(ChartType::Bar, &[1.0, 2.0, 3.0])
        .await;
    assert!(result.is_err(), "Stub GPU renderer should return error");
}
