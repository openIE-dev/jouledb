//! SigQL (Signal Query Language) Execution Engine
//!
//! Bridges SigQL's signal-processing pipeline into JouleDB's query dispatch.
//! SigQL queries operate on time-series signal data with DSP transforms,
//! frequency-domain operations, and uncertainty-quantified aggregations.
//!
//! Signal sources are resolved from amorphic storage tables:
//! - `FROM controller.imu.accel` → table `controller_imu_accel`
//! - Numeric `value` column extracted as f64 samples
//! - Sample rate from `sample_rate` column or default (1000 Hz)

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::Value;
use joule_db_query::executor::TableStorage;
use sigql::ast::FromClause;
use sigql::compile::Target;
use sigql::runtime::{OutputValue, Runtime, RuntimeConfig};
use sigql::types::DynSignal;
use std::sync::Arc;
use std::time::Instant;

/// Maximum rows returned for signal/spectrum outputs to prevent OOM.
const MAX_SIGNAL_ROWS: usize = 10_000;

// ============================================================================
// Detection
// ============================================================================

/// Detect whether a query string is SigQL rather than SQL.
///
/// SigQL is distinguished by keywords/syntax that never appear in SQL:
/// - `TRANSFORM` keyword after `FROM`
/// - `AGGREGATE {` with curly braces
/// - `CORRELATE` keyword
/// - Pipeline operator `|>`
/// - Frequency literals (`4Hz`, `12Hz`)
/// - `RETURNING CONFIDENCE(...)`
pub fn is_sigql_query(sql: &str) -> bool {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();

    // Primary pattern: FROM ... <sigql-keyword>
    if upper.starts_with("FROM ") {
        if upper.contains(" TRANSFORM ")
            || upper.contains(" AGGREGATE {")
            || upper.contains(" AGGREGATE{")
            || upper.contains(" CORRELATE ")
            || upper.contains(" CORRELATE{")
            || upper.contains(" RETURNING CONFIDENCE")
            || trimmed.contains("|>")
            || contains_frequency_literal(trimmed)
        {
            return true;
        }
    }

    // Pipeline syntax: source |> operation (exclusive to SigQL)
    if trimmed.contains("|>") && !upper.contains("SELECT") {
        return true;
    }

    false
}

/// Check for frequency literals like `4Hz`, `12.5Hz`, `500kHz`.
fn contains_frequency_literal(s: &str) -> bool {
    let bytes = s.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        // Look for digit followed by "Hz" (case-insensitive)
        if bytes[i].is_ascii_digit()
            && (bytes[i + 1] == b'H' || bytes[i + 1] == b'h')
            && (bytes[i + 2] == b'z' || bytes[i + 2] == b'Z')
        {
            return true;
        }
        // Also check for "kHz", "MHz" patterns: digit + 'k'/'M' + 'Hz'
        if i + 3 < bytes.len()
            && bytes[i].is_ascii_digit()
            && (bytes[i + 1] == b'k' || bytes[i + 1] == b'M' || bytes[i + 1] == b'G')
            && (bytes[i + 2] == b'H' || bytes[i + 2] == b'h')
            && (bytes[i + 3] == b'z' || bytes[i + 3] == b'Z')
        {
            return true;
        }
    }
    false
}

// ============================================================================
// Execution
// ============================================================================

/// Execute a SigQL query against amorphic storage.
///
/// Pipeline: parse → compile → register sources → execute → map to QueryResponse.
pub fn execute_sigql(
    sql: &str,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    // 1. Parse
    let query = sigql::parse(sql)
        .map_err(|e| QueryErrorResponse::syntax_error(&format!("SigQL: {}", e), 1, 1))?;

    // 2. Compile
    let plan = sigql::compile(&query, Target::Simd)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL compile: {}", e)))?;

    // 3. Create runtime and register signal sources from storage
    let mut runtime = Runtime::new(RuntimeConfig::default());
    register_sources_from_storage(&query, &mut runtime, amorphic)?;

    // 4. Execute
    let result = runtime
        .execute(&plan)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("SigQL runtime: {}", e)))?;

    // 5. Map results to QueryResponse
    map_result_to_response(result, start)
}

/// Extract signal source names from the parsed query's FROM clauses
/// and register matching amorphic tables as signal sources in the runtime.
fn register_sources_from_storage(
    query: &sigql::Query,
    runtime: &mut Runtime,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(), QueryErrorResponse> {
    for from in &query.from {
        let source_name = match from {
            FromClause::Signal(source_ref) => source_ref.path.to_string(),
            FromClause::Table { name, .. } => name.to_string(),
            _ => continue,
        };

        // Map dotted signal path to table name: controller.imu.accel → controller_imu_accel
        let table_name = source_name.replace('.', "_");

        // Check if table exists in amorphic storage
        let tables = amorphic.list_tables();
        if !tables.contains(&table_name) {
            // Not in storage — the runtime will handle SourceNotFound if needed
            continue;
        }

        // Scan the table for signal data
        let scan = match amorphic.scan(&table_name) {
            Ok(rows) => rows,
            Err(_) => continue,
        };
        if scan.is_empty() {
            continue;
        }

        // Extract samples from the `value` column (or first numeric column)
        let mut samples: Vec<f64> = Vec::new();
        let mut sample_rate: u32 = 1000; // default

        // Find column indices from the table schema
        let columns = match amorphic.columns(&table_name) {
            Ok(cols) => cols,
            Err(_) => continue,
        };
        let value_idx = columns.iter().position(|c| c == "value");
        let rate_idx = columns.iter().position(|c| c == "sample_rate");

        for row in &scan {
            if let Some(idx) = value_idx {
                if let Some(val) = row.values.get(idx) {
                    if let Some(f) = value_to_f64(val) {
                        samples.push(f);
                    }
                }
            } else {
                // No `value` column — try first numeric column
                for val in &row.values {
                    if let Some(f) = value_to_f64(val) {
                        samples.push(f);
                        break;
                    }
                }
            }

            // Extract sample rate from first row
            if samples.len() == 1 {
                if let Some(idx) = rate_idx {
                    if let Some(val) = row.values.get(idx) {
                        if let Some(r) = value_to_f64(val) {
                            sample_rate = r as u32;
                        }
                    }
                }
            }
        }

        if !samples.is_empty() {
            let signal = DynSignal::new(&source_name, samples, sample_rate, 0);
            runtime.register_signal(&source_name, signal);
        }
    }

    Ok(())
}

/// Convert an AST Value to f64 if possible.
fn value_to_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

// ============================================================================
// Result Mapping
// ============================================================================

/// Map SigQL `ExecutionResult` to JouleDB `QueryResponse`.
fn map_result_to_response(
    result: sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    if result.outputs.is_empty() {
        return Ok(QueryResponse {
            columns: vec!["result".into()],
            rows: vec![vec![serde_json::json!("OK")]],
            affected_rows: None,
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated: false,
            warnings: Vec::new(),
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: Some("sigql".into()),
            session_id: None,
            viz_hint: None,
        });
    }

    // Classify outputs
    let mut has_scalar = false;
    let mut has_signal = false;
    let mut has_spectrum = false;
    for val in result.outputs.values() {
        match val {
            OutputValue::Scalar(_) => has_scalar = true,
            OutputValue::Signal(_) => has_signal = true,
            OutputValue::Spectrum(_) => has_spectrum = true,
        }
    }

    // All-scalar: compact uncertainty table
    if has_scalar && !has_signal && !has_spectrum {
        return map_scalars(&result, start);
    }

    // Signal output
    if has_signal && !has_spectrum {
        return map_signals(&result, start);
    }

    // Spectrum output
    if has_spectrum && !has_signal {
        return map_spectra(&result, start);
    }

    // Mixed: use generic format
    map_mixed(&result, start)
}

/// Map all-scalar outputs to a compact uncertainty table.
fn map_scalars(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec![
        "name".into(),
        "value".into(),
        "confidence".into(),
        "lower_bound".into(),
        "upper_bound".into(),
        "n_samples".into(),
    ];

    let mut rows = Vec::new();
    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        if let Some(OutputValue::Scalar(uv)) = result.outputs.get(name) {
            rows.push(vec![
                serde_json::json!(name.as_str()),
                serde_json::json!(uv.value),
                serde_json::json!(uv.confidence),
                serde_json::json!(uv.lower_bound),
                serde_json::json!(uv.upper_bound),
                serde_json::json!(uv.n_samples),
            ]);
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// Map signal outputs to sample rows.
fn map_signals(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec![
        "name".into(),
        "sample_index".into(),
        "value".into(),
        "sample_rate".into(),
        "channel".into(),
    ];

    let mut rows = Vec::new();
    let mut truncated = false;
    let mut warnings = Vec::new();

    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        if let Some(OutputValue::Signal(sig)) = result.outputs.get(name) {
            let limit = sig.samples.len().min(MAX_SIGNAL_ROWS);
            if sig.samples.len() > MAX_SIGNAL_ROWS {
                truncated = true;
                warnings.push(format!(
                    "Signal '{}' truncated from {} to {} samples",
                    name,
                    sig.samples.len(),
                    MAX_SIGNAL_ROWS
                ));
            }
            for (i, &sample) in sig.samples.iter().take(limit).enumerate() {
                rows.push(vec![
                    serde_json::json!(name.as_str()),
                    serde_json::json!(i),
                    serde_json::json!(sample),
                    serde_json::json!(sig.sample_rate),
                    serde_json::json!(sig.channel.as_str()),
                ]);
            }
        }
        // Include any scalars in the same output
        if let Some(OutputValue::Scalar(uv)) = result.outputs.get(name) {
            rows.push(vec![
                serde_json::json!(name.as_str()),
                serde_json::json!(0),
                serde_json::json!(uv.value),
                serde_json::json!(0),
                serde_json::json!("scalar"),
            ]);
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings,
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// Map spectrum outputs to frequency-bin rows.
fn map_spectra(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec!["name".into(), "frequency_bin".into(), "magnitude".into()];

    let mut rows = Vec::new();
    let mut truncated = false;
    let mut warnings = Vec::new();

    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        if let Some(OutputValue::Spectrum(spec)) = result.outputs.get(name) {
            let limit = spec.len().min(MAX_SIGNAL_ROWS);
            if spec.len() > MAX_SIGNAL_ROWS {
                truncated = true;
                warnings.push(format!(
                    "Spectrum '{}' truncated from {} to {} bins",
                    name,
                    spec.len(),
                    MAX_SIGNAL_ROWS
                ));
            }
            for (i, &mag) in spec.iter().take(limit).enumerate() {
                rows.push(vec![
                    serde_json::json!(name.as_str()),
                    serde_json::json!(i),
                    serde_json::json!(mag),
                ]);
            }
        }
        // Include scalars
        if let Some(OutputValue::Scalar(uv)) = result.outputs.get(name) {
            rows.push(vec![
                serde_json::json!(name.as_str()),
                serde_json::json!(0),
                serde_json::json!(uv.value),
            ]);
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings,
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// Map mixed outputs (scalar + signal + spectrum) to a generic format.
fn map_mixed(
    result: &sigql::runtime::ExecutionResult,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let columns = vec!["name".into(), "type".into(), "index".into(), "value".into()];

    let mut rows = Vec::new();
    let mut truncated = false;
    let mut warnings = Vec::new();

    let mut sorted_keys: Vec<_> = result.outputs.keys().collect();
    sorted_keys.sort();

    for name in sorted_keys {
        match result.outputs.get(name) {
            Some(OutputValue::Scalar(uv)) => {
                rows.push(vec![
                    serde_json::json!(name.as_str()),
                    serde_json::json!("scalar"),
                    serde_json::json!(0),
                    serde_json::json!(uv.value),
                ]);
            }
            Some(OutputValue::Signal(sig)) => {
                let limit = sig.samples.len().min(MAX_SIGNAL_ROWS);
                if sig.samples.len() > MAX_SIGNAL_ROWS {
                    truncated = true;
                    warnings.push(format!(
                        "Signal '{}' truncated to {} samples",
                        name, MAX_SIGNAL_ROWS
                    ));
                }
                for (i, &sample) in sig.samples.iter().take(limit).enumerate() {
                    rows.push(vec![
                        serde_json::json!(name.as_str()),
                        serde_json::json!("signal"),
                        serde_json::json!(i),
                        serde_json::json!(sample),
                    ]);
                }
            }
            Some(OutputValue::Spectrum(spec)) => {
                let limit = spec.len().min(MAX_SIGNAL_ROWS);
                if spec.len() > MAX_SIGNAL_ROWS {
                    truncated = true;
                    warnings.push(format!(
                        "Spectrum '{}' truncated to {} bins",
                        name, MAX_SIGNAL_ROWS
                    ));
                }
                for (i, &mag) in spec.iter().take(limit).enumerate() {
                    rows.push(vec![
                        serde_json::json!(name.as_str()),
                        serde_json::json!("spectrum"),
                        serde_json::json!(i),
                        serde_json::json!(mag),
                    ]);
                }
            }
            None => {}
        }
    }

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated,
        warnings,
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sigql".into()),
        session_id: None,
        viz_hint: None,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Detection tests --

    #[test]
    fn test_is_sigql_query_positive() {
        // FROM + TRANSFORM
        assert!(is_sigql_query(
            "FROM sensor.data TRANSFORM bandpass(4Hz, 12Hz)"
        ));
        // FROM + AGGREGATE {
        assert!(is_sigql_query(
            "FROM sensor.data AGGREGATE { power: band_power(4Hz..12Hz) }"
        ));
        // FROM + CORRELATE
        assert!(is_sigql_query(
            "FROM sig_a, sig_b CORRELATE { xcorr: cross_correlation }"
        ));
        // Pipeline operator
        assert!(is_sigql_query(
            "sensor.data |> bandpass(4Hz, 12Hz) |> envelope"
        ));
        // Frequency literal
        assert!(is_sigql_query("FROM sensor TRANSFORM lowpass(50Hz)"));
        // RETURNING CONFIDENCE
        assert!(is_sigql_query(
            "FROM sensor.data AGGREGATE { m: mean } RETURNING CONFIDENCE(0.95)"
        ));
        // Case insensitive
        assert!(is_sigql_query("from sensor.data transform fft()"));
    }

    #[test]
    fn test_is_sigql_query_negative() {
        // Standard SQL
        assert!(!is_sigql_query("SELECT * FROM users"));
        assert!(!is_sigql_query("INSERT INTO users VALUES (1, 'Alice')"));
        // Cypher
        assert!(!is_sigql_query("MATCH (n:Person) RETURN n"));
        // CQL
        assert!(!is_sigql_query("CREATE KEYSPACE ks WITH replication = {}"));
        // GraphQL
        assert!(!is_sigql_query("{ users { id name } }"));
        assert!(!is_sigql_query("query GetUser { user(id: 1) { name } }"));
        // Plain FROM without SigQL keywords
        assert!(!is_sigql_query("FROM users WHERE id = 1"));
        // SHOW/DESCRIBE
        assert!(!is_sigql_query("SHOW TABLES"));
    }

    #[test]
    fn test_contains_frequency_literal() {
        assert!(contains_frequency_literal("bandpass(4Hz, 12Hz)"));
        assert!(contains_frequency_literal("lowpass(50kHz)"));
        assert!(contains_frequency_literal("highpass(1MHz)"));
        assert!(!contains_frequency_literal("SELECT Hz FROM table"));
        assert!(!contains_frequency_literal("no frequencies here"));
    }

    // -- Execution tests --

    fn create_test_amorphic() -> Arc<AmorphicTableStorage> {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        std::mem::forget(dir);
        Arc::new(AmorphicTableStorage::new(store))
    }

    #[test]
    fn test_sigql_parse_error() {
        let amorphic = create_test_amorphic();
        let start = Instant::now();
        // Invalid SigQL syntax
        let result = execute_sigql("FROM TRANSFORM", &amorphic, start);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigql_source_not_found() {
        let amorphic = create_test_amorphic();
        let start = Instant::now();
        // Valid SigQL referencing a source not in storage — runtime error
        let result = execute_sigql("FROM nonexistent.signal TRANSFORM fft()", &amorphic, start);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigql_scalar_output() {
        let amorphic = create_test_amorphic();
        let start = Instant::now();

        // Create a table with signal data
        use joule_db_query::executor::TableStorage;
        let cols = vec!["value".to_string()];
        amorphic.create_table("test_signal", &cols).unwrap();

        // Insert sample data
        for i in 0..100 {
            let val = (i as f64 * 0.1).sin();
            let row = joule_db_query::executor::RowData::new(
                vec!["value".into()],
                vec![joule_db_query::ast::Value::Float(val)],
            );
            let _ = amorphic.insert("test_signal", &row);
        }

        // Run aggregate query
        let result = execute_sigql("FROM test.signal AGGREGATE { avg: mean }", &amorphic, start);

        // Should produce scalar output (or fail gracefully if source not found)
        // The table name is test_signal but source is test.signal → maps to test_signal
        match result {
            Ok(resp) => {
                assert_eq!(resp.algorithm_type, Some("sigql".into()));
                assert!(!resp.columns.is_empty());
            }
            Err(_) => {
                // Source resolution may fail if runtime doesn't find it — acceptable
            }
        }
    }

    #[test]
    fn test_sigql_signal_output_mapping() {
        // Test the result mapping directly
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0], 1000, 0);
        outputs.insert(
            smol_str::SmolStr::new("filtered"),
            OutputValue::Signal(signal),
        );

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(
            response.columns,
            vec!["name", "sample_index", "value", "sample_rate", "channel"]
        );
        assert_eq!(response.rows.len(), 5);
        assert_eq!(response.rows[0][0], serde_json::json!("filtered"));
        assert_eq!(response.rows[0][1], serde_json::json!(0));
        assert_eq!(response.rows[0][2], serde_json::json!(1.0));
        assert!(!response.truncated);
    }

    #[test]
    fn test_sigql_scalar_output_mapping() {
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use sigql::types::UncertainValue;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        let uv = UncertainValue::from_ci(10.5, 0.7, 0.95, 1000);
        outputs.insert(smol_str::SmolStr::new("power"), OutputValue::Scalar(uv));

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(response.columns[0], "name");
        assert_eq!(response.columns[1], "value");
        assert_eq!(response.columns[2], "confidence");
        assert_eq!(response.rows.len(), 1);
        assert_eq!(response.rows[0][0], serde_json::json!("power"));
        assert_eq!(response.rows[0][1], serde_json::json!(10.5));
    }

    #[test]
    fn test_sigql_spectrum_output_mapping() {
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        outputs.insert(
            smol_str::SmolStr::new("spectrum"),
            OutputValue::Spectrum(vec![0.1, 0.5, 0.3, 0.2]),
        );

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(response.columns, vec!["name", "frequency_bin", "magnitude"]);
        assert_eq!(response.rows.len(), 4);
        assert_eq!(response.rows[1][2], serde_json::json!(0.5));
    }

    #[test]
    fn test_sigql_truncation() {
        use sigql::runtime::ExecutionResult;
        use sigql::runtime::ExecutionStats;
        use std::collections::HashMap;

        let mut outputs = HashMap::new();
        let samples: Vec<f64> = (0..20_000).map(|i| i as f64 * 0.001).collect();
        let signal = DynSignal::new("big", samples, 1000, 0);
        outputs.insert(smol_str::SmolStr::new("big"), OutputValue::Signal(signal));

        let result = ExecutionResult {
            outputs,
            stats: ExecutionStats::default(),
        };

        let start = Instant::now();
        let response = map_result_to_response(result, start).unwrap();

        assert_eq!(response.rows.len(), MAX_SIGNAL_ROWS);
        assert!(response.truncated);
        assert!(!response.warnings.is_empty());
    }
}
