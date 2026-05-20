//! Module dispatch registry for joule-web.
//!
//! Provides a [`ModuleRegistry`] that maps module names to callable functions,
//! enabling CLI tools, MCP servers, and benchmarks to dynamically invoke any
//! joule-web module by name with JSON arguments.

use std::collections::HashMap;
use std::time::Instant;

// ── Types ────────────────────────────────────────────────────────

/// Metadata and callable entry for a single joule-web module.
pub struct ModuleEntry {
    /// Module name (e.g. "black_scholes").
    pub name: &'static str,
    /// One-line description from the module doc comment.
    pub doc: &'static str,
    /// Domain category (e.g. "financial", "bioinformatics").
    pub domain: &'static str,
    /// Execute the module's primary operation with JSON args, return JSON result.
    pub run: fn(&serde_json::Value) -> Result<serde_json::Value, String>,
}

/// Result of a module invocation, including energy telemetry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InvokeResult {
    /// Module name that was invoked.
    pub module: String,
    /// The operation result as JSON.
    pub result: serde_json::Value,
    /// Wall-clock duration in seconds.
    pub duration_secs: f64,
    /// Estimated energy in joules (wall-time × estimated watts).
    pub energy_joules: f64,
}

/// Registry of all dispatchable joule-web modules.
pub struct ModuleRegistry {
    modules: HashMap<&'static str, ModuleEntry>,
}

impl ModuleRegistry {
    /// Build the default registry with all registered modules.
    pub fn new() -> Self {
        let mut modules = HashMap::new();
        register_all(&mut modules);
        Self { modules }
    }

    /// Number of registered modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// List all module entries, sorted by name.
    pub fn list(&self) -> Vec<&ModuleEntry> {
        let mut entries: Vec<_> = self.modules.values().collect();
        entries.sort_by_key(|e| e.name);
        entries
    }

    /// List modules filtered by domain.
    pub fn list_domain(&self, domain: &str) -> Vec<&ModuleEntry> {
        let mut entries: Vec<_> = self.modules.values().filter(|e| e.domain == domain).collect();
        entries.sort_by_key(|e| e.name);
        entries
    }

    /// Get all unique domain names.
    pub fn domains(&self) -> Vec<&'static str> {
        let mut domains: Vec<_> = self.modules.values().map(|e| e.domain).collect();
        domains.sort();
        domains.dedup();
        domains
    }

    /// Look up a module by name.
    pub fn get(&self, name: &str) -> Option<&ModuleEntry> {
        self.modules.get(name)
    }

    /// Invoke a module by name with JSON arguments.
    /// Returns the result with energy telemetry.
    pub fn invoke(
        &self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<InvokeResult, String> {
        let entry = self.modules.get(name).ok_or_else(|| format!("unknown module: {name}"))?;
        let start = Instant::now();
        let result = (entry.run)(args)?;
        let elapsed = start.elapsed();
        let duration_secs = elapsed.as_secs_f64();
        // Estimate energy: assume ~5W for CPU-bound pure computation on modern hardware
        // Real measurement comes from joule-db-energy in the bench harness
        let energy_joules = duration_secs * 5.0;
        Ok(InvokeResult {
            module: name.to_string(),
            result,
            duration_secs,
            energy_joules,
        })
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────

macro_rules! register_module {
    ($map:expr, $name:expr, $doc:expr, $domain:expr, $run_fn:expr) => {
        $map.insert(
            $name,
            ModuleEntry {
                name: $name,
                doc: $doc,
                domain: $domain,
                run: $run_fn,
            },
        );
    };
}

fn get_f64(args: &serde_json::Value, key: &str) -> Result<f64, String> {
    args.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing or invalid f64 parameter: {key}"))
}

fn get_str<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing or invalid string parameter: {key}"))
}

fn get_f64_array(args: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    args.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
        .ok_or_else(|| format!("missing or invalid f64 array parameter: {key}"))
}

fn get_usize(args: &serde_json::Value, key: &str) -> Result<usize, String> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("missing or invalid usize parameter: {key}"))
}

// ── Registration ─────────────────────────────────────────────────

fn register_all(m: &mut HashMap<&'static str, ModuleEntry>) {
    // ── Financial ────────────────────────────────────────────
    register_module!(m, "black_scholes", "Black-Scholes European option pricing", "financial", run_black_scholes);
    register_module!(m, "bond_price", "Bond pricing and yield calculation", "financial", run_bond_price);
    register_module!(m, "var_model", "Value-at-Risk models for portfolio risk", "financial", run_var_model);
    register_module!(m, "monte_carlo_risk", "Monte Carlo risk simulation (GBM paths)", "financial", run_monte_carlo_risk);
    register_module!(m, "portfolio_optimize", "Mean-variance portfolio optimization", "financial", run_portfolio_optimize);

    // ── Bioinformatics ───────────────────────────────────────
    register_module!(m, "needleman_wunsch", "Global sequence alignment (Needleman-Wunsch)", "bioinformatics", run_needleman_wunsch);
    register_module!(m, "smith_waterman", "Local sequence alignment (Smith-Waterman)", "bioinformatics", run_smith_waterman);
    register_module!(m, "hardy_weinberg", "Hardy-Weinberg equilibrium test", "bioinformatics", run_hardy_weinberg);
    register_module!(m, "kmer_count", "K-mer frequency counting", "bioinformatics", run_kmer_count);

    // ── GIS / Geospatial ─────────────────────────────────────
    register_module!(m, "great_circle", "Great circle distance and bearing", "geospatial", run_great_circle);
    register_module!(m, "geohash_encode", "Geohash encoding/decoding", "geospatial", run_geohash_encode);
    register_module!(m, "geo_coord", "Coordinate conversion (lat/lon, ECEF, DMS)", "geospatial", run_geo_coord);

    // ── Post-Quantum Crypto ──────────────────────────────────
    register_module!(m, "kyber_kem", "CRYSTALS-Kyber key encapsulation", "post-quantum", run_kyber_kem);
    register_module!(m, "dilithium_sign", "CRYSTALS-Dilithium digital signatures", "post-quantum", run_dilithium_sign);
    register_module!(m, "hash_tree", "Merkle hash tree construction", "post-quantum", run_hash_tree);

    // ── CAD/CAM ──────────────────────────────────────────────
    register_module!(m, "bezier_curve", "Bézier curve evaluation and subdivision", "cad", run_bezier_curve);
    register_module!(m, "nurbs_curve", "NURBS curve evaluation", "cad", run_nurbs_curve);

    // ── Healthcare ───────────────────────────────────────────
    register_module!(m, "icd_code", "ICD-10 code lookup and validation", "healthcare", run_icd_code);
    register_module!(m, "dose_calc", "Medication dosage calculation", "healthcare", run_dose_calc);

    // ── Core Web / ML ────────────────────────────────────────
    register_module!(m, "fft_engine", "Fast Fourier Transform", "core", run_fft_engine);
    register_module!(m, "kmeans", "K-means clustering", "core", run_kmeans);
    register_module!(m, "stats_engine", "Descriptive statistics engine", "core", run_stats_engine);
    register_module!(m, "json_patch", "RFC 6902 JSON Patch", "core", run_json_patch);
    register_module!(m, "markdown", "Markdown to HTML parser", "core", run_markdown);
}

// ── Run functions ────────────────────────────────────────────────

fn run_black_scholes(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::black_scholes::{BsModel, OptionKind};
    let spot = get_f64(args, "spot")?;
    let strike = get_f64(args, "strike")?;
    let rate = get_f64(args, "rate")?;
    let vol = get_f64(args, "vol")?;
    let expiry = get_f64(args, "expiry")?;
    let is_call = args.get("is_call").and_then(|v| v.as_bool()).unwrap_or(true);
    let model = BsModel::new(spot, strike, rate, vol, expiry);
    let kind = if is_call { OptionKind::Call } else { OptionKind::Put };
    let price = model.price(kind).map_err(|e| e.to_string())?;
    let greeks = model.greeks(kind).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "price": price,
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "theta": greeks.theta,
        "vega": greeks.vega,
        "rho": greeks.rho,
    }))
}

fn run_bond_price(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::bond_price::{BondPriceConfig, BondPrice};
    let face = get_f64(args, "face_value").unwrap_or(1000.0);
    let coupon = get_f64(args, "coupon_rate")?;
    let yield_rate = get_f64(args, "yield_rate")?;
    let maturity = get_f64(args, "maturity_years")?;
    let cfg = BondPriceConfig::new()
        .with_face_value(face)
        .with_coupon_rate(coupon)
        .with_yield_rate(yield_rate)
        .with_maturity_years(maturity);
    let pricer = BondPrice::new(cfg).map_err(|e| e.to_string())?;
    let price = pricer.clean_price();
    let duration = pricer.modified_duration();
    Ok(serde_json::json!({
        "clean_price": price,
        "modified_duration": duration,
        "summary": format!("{:?}", pricer.summary()),
    }))
}

fn run_var_model(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::var_model::{VarConfig, generate_var_report};
    let confidence = get_f64(args, "confidence").unwrap_or(0.95);
    let pnl = get_f64_array(args, "returns")?;
    let cfg = VarConfig::new().with_confidence(confidence);
    let report = generate_var_report(&pnl, &cfg);
    Ok(serde_json::json!({
        "historical_var": report.historical_var,
        "parametric_var": report.parametric_var,
        "cvar": report.cvar,
        "sample_count": report.sample_count,
    }))
}

fn run_monte_carlo_risk(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::monte_carlo_risk::{GbmConfig, GbmPath};
    let s0 = get_f64(args, "initial_price").unwrap_or(100.0);
    let drift = get_f64(args, "drift").unwrap_or(0.05);
    let vol = get_f64(args, "volatility").unwrap_or(0.2);
    let steps = get_usize(args, "steps").unwrap_or(252);
    let n_paths = get_usize(args, "n_paths").unwrap_or(1000);
    let cfg = GbmConfig::new()
        .with_drift(drift)
        .with_volatility(vol)
        .with_steps(steps);
    let mut gbm = GbmPath::new(cfg);
    let terminals = gbm.generate_terminals(s0, n_paths);
    let mean_terminal: f64 = terminals.iter().sum::<f64>() / terminals.len() as f64;
    let min = terminals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = terminals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(serde_json::json!({
        "mean_terminal": mean_terminal,
        "min_terminal": min,
        "max_terminal": max,
        "n_paths": n_paths,
        "steps": steps,
    }))
}

fn run_portfolio_optimize(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::portfolio_optimize::{PortfolioConfig, mean_variance_optimize};
    let returns = get_f64_array(args, "returns")?;
    let n_assets = get_usize(args, "n_assets")?;
    let risk_free = get_f64(args, "risk_free_rate").unwrap_or(0.02);
    let target_return = get_f64(args, "target_return")?;
    let cfg = PortfolioConfig::new().with_risk_free_rate(risk_free);
    let result = mean_variance_optimize(&returns, n_assets, target_return, &cfg);
    Ok(serde_json::json!({
        "weights": result.weights,
        "expected_return": result.expected_return,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
    }))
}

fn run_needleman_wunsch(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::needleman_wunsch::{NeedlemanWunsch, NwConfig};
    let seq_a = get_str(args, "seq_a")?;
    let seq_b = get_str(args, "seq_b")?;
    let cfg = NwConfig::new();
    let aligner = NeedlemanWunsch::new(cfg);
    let result = aligner.align(seq_a.as_bytes(), seq_b.as_bytes()).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "score": result.score,
        "aligned_a": String::from_utf8_lossy(&result.aligned_a),
        "aligned_b": String::from_utf8_lossy(&result.aligned_b),
        "length": result.length(),
        "matches": result.matches(),
    }))
}

fn run_smith_waterman(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::smith_waterman::{SmithWaterman, SwConfig};
    let seq_a = get_str(args, "seq_a")?;
    let seq_b = get_str(args, "seq_b")?;
    let cfg = SwConfig::new();
    let aligner = SmithWaterman::new(cfg);
    let result = aligner.align(seq_a.as_bytes(), seq_b.as_bytes()).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "score": result.score,
        "aligned_a": String::from_utf8_lossy(&result.aligned_a),
        "aligned_b": String::from_utf8_lossy(&result.aligned_b),
        "length": result.length(),
        "matches": result.matches(),
    }))
}

fn run_hardy_weinberg(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::hardy_weinberg::{GenotypeCounts, HweTester};
    let n_aa = args.get("n_aa").and_then(|v| v.as_u64()).unwrap_or(50);
    let n_ab = args.get("n_ab").and_then(|v| v.as_u64()).unwrap_or(40);
    let n_bb = args.get("n_bb").and_then(|v| v.as_u64()).unwrap_or(10);
    let counts = GenotypeCounts::new(n_aa, n_ab, n_bb);
    let tester = HweTester::new();
    let result = tester.test(&counts);
    Ok(serde_json::json!({
        "freq_a": counts.freq_a(),
        "freq_b": counts.freq_b(),
        "observed_het": counts.observed_het(),
        "expected_het": counts.expected_het(),
        "chi_squared": result.chi_squared,
        "p_value": result.p_value,
    }))
}

fn run_kmer_count(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::kmer_count::KmerCounter;
    let sequence = get_str(args, "sequence")?;
    let k = get_usize(args, "k").unwrap_or(3);
    let counter = KmerCounter::new(k);
    let spectrum = counter.count(sequence.as_bytes()).map_err(|e| e.to_string())?;
    let top = spectrum.top_n(10);
    let entries: Vec<_> = top.iter().map(|(kmer, count)| {
        serde_json::json!({ "kmer": String::from_utf8_lossy(kmer), "count": count })
    }).collect();
    Ok(serde_json::json!({ "top_kmers": entries, "distinct": spectrum.distinct() }))
}

fn run_great_circle(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::great_circle::{GreatCircle, GreatCircleConfig};
    let lat1 = get_f64(args, "lat1")?;
    let lon1 = get_f64(args, "lon1")?;
    let lat2 = get_f64(args, "lat2")?;
    let lon2 = get_f64(args, "lon2")?;
    let cfg = GreatCircleConfig::new();
    let gc = GreatCircle::new(cfg).map_err(|e| e.to_string())?;
    // The template API operates on internal data via push(); haversine() uses that data.
    // Feed the 4 coords as data points and use summary().
    let gc = gc.with_data(vec![lat1, lon1, lat2, lon2]);
    let dist = gc.haversine();
    let bearing = gc.initial_bearing();
    Ok(serde_json::json!({ "distance_km": dist, "bearing_deg": bearing }))
}

fn run_geohash_encode(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::geohash_encode;
    let lat = get_f64(args, "lat")?;
    let lon = get_f64(args, "lon")?;
    let precision = get_usize(args, "precision").unwrap_or(8);
    let hash = geohash_encode::encode(lat, lon, precision).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "geohash": hash }))
}

fn run_geo_coord(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::geo_coord::GeoCoord;
    let lat = get_f64(args, "lat")?;
    let lon = get_f64(args, "lon")?;
    let coord = GeoCoord::new(lat, lon).map_err(|e| e.to_string())?;
    let dms = coord.to_dms();
    let ecef = coord.to_ecef();
    Ok(serde_json::json!({
        "lat_rad": coord.lat_rad(),
        "lon_rad": coord.lon_rad(),
        "ecef": [ecef.x, ecef.y, ecef.z],
        "dms_lat": format!("{}", dms.lat),
        "dms_lon": format!("{}", dms.lon),
    }))
}

fn run_kyber_kem(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::kyber_kem::{KyberKem, KyberKemConfig};
    let security = get_usize(args, "security_level").unwrap_or(3);
    let cfg = KyberKemConfig::new().with_security_level(security);
    let kem = KyberKem::new(cfg).map_err(|e| e.to_string())?;
    let keys = kem.keygen();
    let summary = kem.summary();
    Ok(serde_json::json!({
        "keygen_output_len": keys.len(),
        "security_level": security,
        "summary": format!("{:?}", summary),
    }))
}

fn run_dilithium_sign(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::dilithium_sign::{DilithiumSign, DilithiumSignConfig};
    let security = get_usize(args, "security_level").unwrap_or(3);
    let cfg = DilithiumSignConfig::new().with_security_level(security);
    let signer = DilithiumSign::new(cfg).map_err(|e| e.to_string())?;
    let keys = signer.keygen();
    let sig = signer.sign();
    let valid = signer.verify();
    Ok(serde_json::json!({
        "keygen_output_len": keys.len(),
        "signature_len": sig.len(),
        "verify": valid,
        "security_level": security,
    }))
}

fn run_hash_tree(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::hash_tree::{HashTree, HashTreeConfig};
    let leaves = get_usize(args, "leaves").unwrap_or(8);
    let cfg = HashTreeConfig::new().with_leaf_count(leaves);
    let tree = HashTree::new(cfg).map_err(|e| e.to_string())?;
    let levels = tree.build_tree();
    let auth = tree.auth_path();
    let valid = tree.verify_path();
    Ok(serde_json::json!({
        "levels": levels.len(),
        "auth_path_len": auth.len(),
        "verify": valid,
        "leaves": leaves,
    }))
}

fn run_bezier_curve(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::bezier_curve::{BezierCurve, BezierCurveConfig};
    let degree = get_usize(args, "degree").unwrap_or(3);
    let cfg = BezierCurveConfig::new().with_degree(degree);
    let curve = BezierCurve::new(cfg).map_err(|e| e.to_string())?;
    let pts = curve.evaluate();
    let arc = curve.arc_length();
    Ok(serde_json::json!({
        "evaluate_len": pts.len(),
        "arc_length": arc,
        "degree": degree,
    }))
}

fn run_nurbs_curve(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::nurbs_curve::{NurbsCurve, WeightedPoint};
    // Build a simple curve from control points or use defaults
    let degree = get_usize(args, "degree").unwrap_or(2);
    let n_pts = get_usize(args, "n_points").unwrap_or(4);
    let t = get_f64(args, "t").unwrap_or(0.5);
    // Generate evenly spaced control points
    let control_points: Vec<WeightedPoint> = (0..n_pts)
        .map(|i| WeightedPoint::new(i as f64, (i as f64).sin(), 0.0, 1.0))
        .collect();
    let knots = NurbsCurve::clamped_uniform_knots(n_pts, degree);
    let curve = NurbsCurve::new(control_points, knots, degree).map_err(|e| e.to_string())?;
    let pt = curve.evaluate(t).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "point": [pt.x, pt.y, pt.z],
        "t": t,
        "degree": degree,
        "n_control_points": n_pts,
    }))
}

fn run_icd_code(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::icd_code::{IcdCode, IcdCodeConfig};
    let _code = get_str(args, "code")?;
    let cfg = IcdCodeConfig::new();
    let icd = IcdCode::new(cfg).map_err(|e| e.to_string())?;
    let desc = icd.lookup();
    let valid = icd.is_valid();
    let parent = icd.parent_code();
    Ok(serde_json::json!({
        "description": desc,
        "is_valid": valid,
        "parent_code": parent,
    }))
}

fn run_dose_calc(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::dose_calc::{DoseCalc, DoseCalcConfig};
    let weight_kg = get_f64(args, "weight_kg")?;
    let cfg = DoseCalcConfig::new().with_weight_kg(weight_kg);
    let calc = DoseCalc::new(cfg).map_err(|e| e.to_string())?;
    let weight_dose = calc.weight_based_dose();
    let bsa = calc.bsa_dose();
    let crcl = calc.creatinine_clearance();
    Ok(serde_json::json!({
        "weight_based_dose": weight_dose,
        "bsa_dose": bsa,
        "creatinine_clearance": crcl,
    }))
}

fn run_fft_engine(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::fft_engine::{self, Complex};
    let data = get_f64_array(args, "data")?;
    let spectrum = fft_engine::rfft(&data).map_err(|e| e.to_string())?;
    let magnitudes = fft_engine::magnitude_spectrum(&spectrum);
    let power = fft_engine::power_spectrum(&spectrum);
    Ok(serde_json::json!({
        "magnitudes": magnitudes,
        "power_spectrum": power,
        "n": data.len(),
        "n_bins": spectrum.len(),
    }))
}

fn run_kmeans(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::kmeans::{KMeansConfig, kmeans};
    let k = get_usize(args, "k")?;
    let flat_data = get_f64_array(args, "data")?;
    let dims = get_usize(args, "dims").unwrap_or(2);
    // Convert flat array to Vec<Vec<f64>>
    let data: Vec<Vec<f64>> = flat_data.chunks(dims)
        .filter(|c| c.len() == dims)
        .map(|c| c.to_vec())
        .collect();
    let cfg = KMeansConfig { k, max_iter: 100, tol: 1e-6, init: crate::kmeans::InitMethod::KMeansPlusPlus, seed: 42 };
    let result = kmeans(&data, &cfg);
    Ok(serde_json::json!({
        "centroids": result.centroids,
        "assignments": result.assignments,
        "iterations": result.iterations,
        "inertia": result.inertia,
        "k": k,
    }))
}

fn run_stats_engine(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::stats_engine;
    let data = get_f64_array(args, "data")?;
    let summary = stats_engine::summarize(&data);
    Ok(serde_json::json!({
        "mean": summary.mean,
        "median": summary.median,
        "std_dev": summary.std_dev,
        "min": summary.min,
        "max": summary.max,
        "count": summary.count,
        "variance": summary.variance,
        "skewness": summary.skewness,
        "kurtosis": summary.kurtosis,
    }))
}

fn run_json_patch(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::json_patch;
    let mut doc = args.get("document").cloned().ok_or("missing parameter: document")?;
    let patches_val = args.get("patches").cloned().ok_or("missing parameter: patches")?;
    let ops = json_patch::parse_patch(&patches_val).map_err(|e| e.to_string())?;
    json_patch::apply(&mut doc, &ops).map_err(|e| e.to_string())?;
    Ok(doc)
}

fn run_markdown(args: &serde_json::Value) -> Result<serde_json::Value, String> {
    use crate::markdown;
    let input = get_str(args, "input")?;
    let ast = markdown::parse_markdown(input);
    let html = markdown::render_html(&ast);
    Ok(serde_json::json!({ "html": html }))
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let reg = ModuleRegistry::new();
        assert!(reg.len() >= 20);
    }

    #[test]
    fn test_list_sorted() {
        let reg = ModuleRegistry::new();
        let list = reg.list();
        for w in list.windows(2) {
            assert!(w[0].name <= w[1].name);
        }
    }

    #[test]
    fn test_domains() {
        let reg = ModuleRegistry::new();
        let domains = reg.domains();
        assert!(domains.contains(&"financial"));
        assert!(domains.contains(&"bioinformatics"));
        assert!(domains.contains(&"geospatial"));
    }

    #[test]
    fn test_invoke_stats_engine() {
        let reg = ModuleRegistry::new();
        let args = serde_json::json!({ "data": [1.0, 2.0, 3.0, 4.0, 5.0] });
        let result = reg.invoke("stats_engine", &args).unwrap();
        assert!(result.duration_secs >= 0.0);
        assert!(result.energy_joules >= 0.0);
        let mean = result.result["mean"].as_f64().unwrap();
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_invoke_unknown() {
        let reg = ModuleRegistry::new();
        let args = serde_json::json!({});
        assert!(reg.invoke("nonexistent_module", &args).is_err());
    }

    #[test]
    fn test_get_module() {
        let reg = ModuleRegistry::new();
        let entry = reg.get("black_scholes").unwrap();
        assert_eq!(entry.domain, "financial");
    }
}
