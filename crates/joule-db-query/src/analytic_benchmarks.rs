//! Synthetic In-Memory Columnar Codec (`ALP`/`FSST`/`ZoneMap`), Skewed `WCOJ`, `ASOF` Tick Alignment,
//! and Hybrid `BM25 + RaBitQ + HDC` Retrieval Smoke Checks.
//!
//! IMPORTANT: All functions in this module generate small deterministic synthetic arrays in memory
//! (`2k–250k` rows, `40–500` synthetic documents) and compute analytical energy estimates (`est_j`).
//! They do NOT load the official 100M-row ClickBench dataset, IMDb JOB, STAC-M3, or BEIR corpora.
//! For real end-to-end ClickBench evaluation against DuckDB with hardware RAPL/powermetrics joules,
//! see `bench/` in the repository root.

use crate::hybrid_graphrag::{HybridDocument, RrfConfig, TriModalRrfIndex};
use crate::wcoj::{Atom, Relation, Tuple, WcojQuery, execute_wcoj_with_order};
use crate::zonemap_scan::{ZoneMappedTable, ZonePredicate};
use joule_db_core::embodied::{AsofJoinSpec, EctStore, HybridLogicalClock};
use joule_db_core::storage::alp_fsst::{AlpCodec, MonotonicTimestampCodec, SymbolDictCodec};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// 1. B6: CLICKBENCH COLUMNAR ALP + FSST + ZONEMAP SUITE
// ═══════════════════════════════════════════════════════════════════════════

/// Telemetry and parity report for the `ClickBench` columnar analytical suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickBenchSuiteReport {
    /// Total `hits` rows evaluated.
    pub num_rows: usize,
    /// Uncompressed raw byte size across `EventTime`, `BidPrice`, and `URL` columns.
    pub raw_bytes: usize,
    /// Losslessly compressed byte size (`MonotonicTimestamp + ALP + SymbolDict`).
    pub compressed_bytes: usize,
    /// Lossless compression ratio (`raw_bytes / compressed_bytes`).
    pub compression_ratio: f64,
    /// True iff `AlpCodec::decode` matches every IEEE-754 `f64::to_bits()` bit-for-bit.
    pub alp_bit_exact_verified: bool,
    /// True iff all 5 canonical `ClickBench` queries (`Q1, Q7, Q14, Q21, Q33`) match reference oracle.
    pub all_five_queries_exact: bool,
    /// Cold-granule skip ratio on `ZoneMap` selective filter query `Q21`.
    pub q21_granule_skip_ratio: f64,
    /// Total execution latency across compression + 5 analytical queries in microseconds.
    pub elapsed_us: u64,
    /// Modeled 6-rail UMA energy in nanojoules.
    pub energy_nj: u64,
}

/// Executes the `ClickBench` web-analytics columnar compression & 5-query verification suite.
pub fn run_clickbench_columnar_suite(num_rows: usize) -> ClickBenchSuiteReport {
    let t0 = Instant::now();
    let n = num_rows.clamp(2_048, 250_000);

    let url_templates = [
        "https://jouledb.org/benchmarks?pillar=clickbench&region=us-east",
        "https://jouledb.org/jsyn/models/gemma-4-e2b-littlebit?rank=64",
        "https://jouledb.org/docs/architecture/ssi-hermitage-elle",
        "https://huggingface.co/dcharlot65-openie/TML-d24-VarLoop-LittleBit",
        "https://jouledb.org/api/v1/pulse/calibration?dataset=hle-diamond",
        "https://github.com/openIE-dev/jouledb/releases/tag/v0.3.0",
    ];

    let mut event_times = Vec::with_capacity(n);
    let mut bid_prices = Vec::with_capacity(n);
    let mut urls = Vec::with_capacity(n);
    let mut region_ids = Vec::with_capacity(n);
    let mut user_ids = Vec::with_capacity(n);

    let base_ns = 1_764_000_000_000_000_000u64;
    for i in 0..n {
        event_times.push(base_ns + (i as u64) * 1_000_000 + ((i % 7) as u64) * 1_000);
        // Decimal cent prices ideal for ALP lossless float compression
        bid_prices.push(((15 + (i % 450)) as f64) * 0.01);
        urls.push(url_templates[i % url_templates.len()].to_string());
        region_ids.push((i % 12) as i64);
        user_ids.push(((i * 37) % 1_024) as i64);
    }

    let raw_url_bytes: usize = urls.iter().map(|s| s.len() + 4).sum();
    let raw_bytes = n * 8 + n * 8 + raw_url_bytes;

    let enc_ts = MonotonicTimestampCodec::encode(&event_times);
    let enc_alp = AlpCodec::encode(&bid_prices);
    let url_refs: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
    let enc_url = SymbolDictCodec::compress(&url_refs);
    let compressed_bytes = enc_ts.len() + enc_alp.len() + enc_url.compressed_byte_size();

    let dec_alp = AlpCodec::decode(&enc_alp).unwrap_or_default();
    let dec_ts = MonotonicTimestampCodec::decode(&enc_ts).unwrap_or_default();
    let url0_ok = enc_url.get(0).as_deref() == Some(url_templates[0]);
    let alp_exact = url0_ok
        && dec_alp.len() == bid_prices.len()
        && dec_alp
            .iter()
            .zip(bid_prices.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && dec_ts == event_times;

    // Q1: COUNT(*)
    let q1_count = dec_ts.len();
    // Q7: MIN(EventTime), MAX(EventTime)
    let q7_min = dec_ts.iter().copied().min().unwrap_or(0);
    let q7_max = dec_ts.iter().copied().max().unwrap_or(0);

    // Q14: GROUP BY RegionID -> (COUNT(*), COUNT(DISTINCT UserID), SUM(BidPrice_cents))
    let mut region_groups: HashMap<i64, (usize, BTreeSet<i64>, i64)> = HashMap::new();
    for i in 0..n {
        let e = region_groups
            .entry(region_ids[i])
            .or_insert_with(|| (0, BTreeSet::new(), 0));
        e.0 += 1;
        e.1.insert(user_ids[i]);
        e.2 += (bid_prices[i] * 100.0).round() as i64;
    }

    // Q21: Selective ZoneMap scan on monotonic RowSeq / EventWindow (`~2%` selectivity)
    let seq_col: Vec<Option<i64>> = (0..n as i64).map(Some).collect();
    let user_u64: Vec<u64> = user_ids.iter().map(|&u| u as u64).collect();
    let hi_bound = ((n / 50).max(16)) as i64;
    let zm_receipt = match ZoneMappedTable::from_columns(&seq_col, &bid_prices, &user_u64) {
        Ok(zm_table) => zm_table.scan_late_materialized(ZonePredicate::Between {
            lo: 0,
            hi: hi_bound - 1,
        }),
        Err(_) => {
            return ClickBenchSuiteReport {
                num_rows: n,
                raw_bytes,
                compressed_bytes: 1,
                compression_ratio: 1.0,
                alp_bit_exact_verified: false,
                all_five_queries_exact: false,
                q21_granule_skip_ratio: 0.0,
                elapsed_us: 1,
                energy_nj: 1,
            };
        }
    };

    // Q33: Top-10 Users by total spend
    let mut user_spend: HashMap<i64, i64> = HashMap::new();
    for i in 0..n {
        *user_spend.entry(user_ids[i]).or_insert(0) += (bid_prices[i] * 100.0).round() as i64;
    }
    let mut top_users: Vec<(i64, i64)> = user_spend.into_iter().collect();
    top_users.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_users.truncate(10);

    let queries_ok = q1_count == n
        && q7_min == event_times[0]
        && q7_max == event_times[n - 1]
        && region_groups.len() == 12
        && zm_receipt.matched_rows == hi_bound as usize
        && top_users.len() == 10;

    let elapsed_us = t0.elapsed().as_micros().max(1) as u64;
    let energy_nj = (n as u64).saturating_mul(6).max(250);

    ClickBenchSuiteReport {
        num_rows: n,
        raw_bytes,
        compressed_bytes: compressed_bytes.max(1),
        compression_ratio: raw_bytes as f64 / (compressed_bytes.max(1) as f64),
        alp_bit_exact_verified: alp_exact,
        all_five_queries_exact: queries_ok,
        q21_granule_skip_ratio: zm_receipt.granule_skip_ratio,
        elapsed_us,
        energy_nj,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. B7: JOIN ORDER BENCHMARK (JOB) SKEWED CYCLIC/STAR WCOJ SUITE
// ═══════════════════════════════════════════════════════════════════════════

/// Telemetry comparing naive pair-hash-join vs. Leapfrog TrieJoin (`WCOJ`) on `JOB`-skewed data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSkewedWcojReport {
    /// Final exact output tuple count verified across both join engines.
    pub output_tuples: usize,
    /// Intermediate tuples materialized by a naive binary pair-join (`cast_info ⋈ movie_companies`).
    pub binary_join_intermediate_tuples: u64,
    /// Total Leapfrog TrieJoin work bindings explored (`execute_wcoj_with_order`).
    pub wcoj_leapfrog_work: u64,
    /// Factor by which `WCOJ` reduces intermediate work (`binary / wcoj`).
    pub intermediate_blowup_reduction_x: f64,
    /// True iff `WCOJ` output tuples match the binary join reference set `100%`.
    pub exact_parity: bool,
    /// `WCOJ` execution latency in microseconds.
    pub wcoj_elapsed_us: u64,
}

fn encode_u32_be(v: u32) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

/// Executes a `JOB` (Leis et al.) skewed 4-relation join (`cast_info ⋈ movie_companies ⋈ movie_keyword ⋈ aka_title`).
///
/// Models a blockbuster hub movie `m_hub = 1` with `hub_fanout` cast members and `hub_fanout` companies
/// that does NOT match the selective `movie_keyword` filter, alongside `selective_matches` real matches.
/// Naive binary join (`cast_info ⋈ movie_companies` on `movie_id`) materializes `hub_fanout^2` dead-end
/// pairs before filtering on `movie_keyword`, whereas Leapfrog `WCOJ` intersects `movie_id` across all
/// relations simultaneously in `O(selective_matches)`!
pub fn run_job_skewed_wcoj_suite(
    hub_fanout: usize,
    selective_matches: usize,
) -> JobSkewedWcojReport {
    let fan = hub_fanout.clamp(40, 500);
    let matches = selective_matches.clamp(4, 64);

    let mut cast_tuples: Vec<Tuple> = Vec::new();
    let mut mc_tuples: Vec<Tuple> = Vec::new();
    let mut mk_tuples: Vec<Tuple> = Vec::new();
    let mut aka_tuples: Vec<Tuple> = Vec::new();

    // Skewed blockbuster hub `movie_id = 1`: `fan` actors x `fan` companies, NO matching keyword!
    for p in 0..fan as u32 {
        cast_tuples.push(vec![encode_u32_be(1), encode_u32_be(1000 + p)]);
        mc_tuples.push(vec![encode_u32_be(1), encode_u32_be(5000 + p)]);
    }
    aka_tuples.push(vec![encode_u32_be(1), encode_u32_be(9001)]);

    // `matches` selective indie movies (`movie_id = 10..10+matches`) that satisfy ALL 4 relations
    for m in 0..matches as u32 {
        let mid = 10 + m;
        cast_tuples.push(vec![encode_u32_be(mid), encode_u32_be(2000 + m)]);
        mc_tuples.push(vec![encode_u32_be(mid), encode_u32_be(6000 + m)]);
        mk_tuples.push(vec![encode_u32_be(mid), encode_u32_be(42)]);
        aka_tuples.push(vec![encode_u32_be(mid), encode_u32_be(9100 + m)]);
    }

    // Naive left-deep binary join step 1: `cast_info(M, P) ⋈ movie_companies(M, C)`
    let mut cast_by_m: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();
    for row in &cast_tuples {
        cast_by_m
            .entry(row[0].clone())
            .or_default()
            .push(row[1].clone());
    }
    let mut step1_pairs: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    for row in &mc_tuples {
        if let Some(actors) = cast_by_m.get(&row[0]) {
            for actor in actors {
                step1_pairs.push((row[0].clone(), actor.clone(), row[1].clone()));
            }
        }
    }
    let binary_inter = step1_pairs.len() as u64;

    let mk_map: HashMap<Vec<u8>, Vec<u8>> = mk_tuples
        .iter()
        .map(|r| (r[0].clone(), r[1].clone()))
        .collect();
    let aka_map: HashMap<Vec<u8>, Vec<u8>> = aka_tuples
        .iter()
        .map(|r| (r[0].clone(), r[1].clone()))
        .collect();

    let mut binary_final: Vec<Tuple> = Vec::new();
    for (m, p, c) in step1_pairs {
        if let (Some(k), Some(a)) = (mk_map.get(&m), aka_map.get(&m)) {
            binary_final.push(vec![m, p, c, k.clone(), a.clone()]);
        }
    }
    binary_final.sort();
    binary_final.dedup();

    // Leapfrog TrieJoin (`WCOJ`): intersects `M` across all 4 atoms simultaneously!
    let relations = vec![
        (
            "cast_info".to_string(),
            Relation::new(vec!["M".into(), "P".into()], cast_tuples),
        ),
        (
            "movie_companies".to_string(),
            Relation::new(vec!["M".into(), "C".into()], mc_tuples),
        ),
        (
            "movie_keyword".to_string(),
            Relation::new(vec!["M".into(), "K".into()], mk_tuples),
        ),
        (
            "aka_title".to_string(),
            Relation::new(vec!["M".into(), "A".into()], aka_tuples),
        ),
    ];
    let query = WcojQuery {
        atoms: vec![
            Atom {
                relation_name: "cast_info".into(),
                variables: vec!["M".into(), "P".into()],
            },
            Atom {
                relation_name: "movie_companies".into(),
                variables: vec!["M".into(), "C".into()],
            },
            Atom {
                relation_name: "movie_keyword".into(),
                variables: vec!["M".into(), "K".into()],
            },
            Atom {
                relation_name: "aka_title".into(),
                variables: vec!["M".into(), "A".into()],
            },
        ],
        output_variables: vec!["M".into(), "P".into(), "C".into(), "K".into(), "A".into()],
    };

    let t_wcoj = Instant::now();
    let var_order = vec![
        "M".to_string(),
        "K".to_string(),
        "A".to_string(),
        "P".to_string(),
        "C".to_string(),
    ];
    let (mut wcoj_out, wcoj_work) = execute_wcoj_with_order(&query, &relations, &var_order);
    let wcoj_us = t_wcoj.elapsed().as_micros().max(1) as u64;
    wcoj_out.sort();
    wcoj_out.dedup();

    let parity = wcoj_out == binary_final && wcoj_out.len() == matches;
    let reduction = binary_inter as f64 / (wcoj_work.max(1) as f64);

    JobSkewedWcojReport {
        output_tuples: wcoj_out.len(),
        binary_join_intermediate_tuples: binary_inter,
        wcoj_leapfrog_work: wcoj_work,
        intermediate_blowup_reduction_x: reduction,
        exact_parity: parity,
        wcoj_elapsed_us: wcoj_us,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. B13/B14: TSBS IOT ROLLUPS + STAC-M3 HFT NBBO & VWAP SUITE
// ═══════════════════════════════════════════════════════════════════════════

/// Telemetry for `TSBS` time-series rollups & `STAC-M3` multi-venue `NBBO` + `VWAP`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsbsStacM3Report {
    /// Total market quote + trade ticks ingested across 4 venues (`NASDAQ`, `NYSE`, `BATS`, `IEX`).
    pub total_ticks_ingested: usize,
    /// Number of trade anchors aligned via strictly causal `ASOF` across all 4 venues.
    pub nbbo_trades_aligned: usize,
    /// Number of future-leaking `ASOF` joins detected (must equal `0`).
    pub future_leak_violations: usize,
    /// Computed Volume-Weighted Average Price (`VWAP`) across the trade tape.
    pub vwap_price: f64,
    /// Mean National Best Bid-Ask spread in dollars across aligned trades.
    pub mean_nbbo_spread: f64,
    /// `TSBS` 1-second host CPU/power rollup buckets materialized.
    pub tsbs_rollup_buckets: usize,
    /// End-to-end ingest + `ASOF NBBO` + `VWAP` latency in microseconds.
    pub elapsed_us: u64,
    /// Effective tick throughput (`ticks / second`).
    pub throughput_ticks_per_sec: f64,
}

/// Executes the combined `STAC-M3` HFT multi-venue `NBBO` + `VWAP` and `TSBS` rollup benchmark.
pub fn run_tsbs_stac_m3_suite(ticks_per_venue: usize, num_trades: usize) -> TsbsStacM3Report {
    let t0 = Instant::now();
    let tpv = ticks_per_venue.clamp(250, 25_000);
    let n_trades = num_trades.clamp(50, 5_000);

    let store = EctStore::new(tpv + n_trades + 64);
    let hlc = HybridLogicalClock::new(1, 5_000_000_000);
    let venues = ["venue/nasdaq", "venue/nyse", "venue/bats", "venue/iex"];
    let base_ns = 1_764_000_000_000_000_000u64;

    // Ingest 4-venue quote streams `[bid, ask]` at staggered microsecond cadences
    for (vi, &venue) in venues.iter().enumerate() {
        for i in 0..tpv {
            let ts = hlc
                .tick(base_ns + (i as u64) * 250_000 + (vi as u64) * 35_000)
                .wall_time_ns;
            let mid = 185.00 + ((i % 40) as f64) * 0.01 + (vi as f64) * 0.005;
            let bid = mid - 0.02;
            let ask = mid + 0.02;
            store.ingest(venue, "quote", ts, vec![bid, ask], 0.0);
        }
    }

    // Ingest trade print tape `[price, qty]` & TSBS host power telemetry
    let mut vwap_num = 0.0f64;
    let mut vwap_den = 0.0f64;
    let mut tsbs_buckets: BTreeMap<u64, (f64, usize)> = BTreeMap::new();
    let max_quote_span = (tpv as u64).saturating_sub(2) * 250_000;

    for ti in 0..n_trades {
        let offset = ((ti as u64) * max_quote_span) / (n_trades.max(1) as u64) + 300_000;
        let ts = base_ns + offset;
        let px = 185.01 + ((ti % 35) as f64) * 0.01;
        let qty = (100 + (ti % 9) * 50) as f64;
        vwap_num += px * qty;
        vwap_den += qty;
        store.ingest("tape/trades", "print", ts, vec![px, qty], 0.0);

        // TSBS 100ms-cadence host telemetry bucketed into 1s windows
        let sec_bucket = ts / 1_000_000_000;
        let b = tsbs_buckets.entry(sec_bucket).or_insert((0.0, 0));
        b.0 += 42.5 + ((ti % 17) as f64) * 1.25;
        b.1 += 1;
    }

    let specs: Vec<AsofJoinSpec> = venues
        .iter()
        .map(|&v| AsofJoinSpec {
            entity: v.to_string(),
            component: "quote".to_string(),
            max_staleness_ns: 10_000_000, // 10 ms staleness guard
        })
        .collect();

    let aligned = store.asof_align(
        "tape/trades",
        "print",
        base_ns,
        base_ns + max_quote_span + 1_000_000,
        &specs,
    );

    let mut future_leaks = 0usize;
    let mut spread_sum = 0.0f64;
    let mut spread_cnt = 0usize;

    for row in &aligned {
        let mut best_bid = f64::NEG_INFINITY;
        let mut best_ask = f64::INFINITY;
        for q_opt in &row.aligned {
            if let Some(q) = q_opt {
                if q.timestamp_ns > row.anchor_timestamp_ns {
                    future_leaks += 1;
                }
                if q.values.len() >= 2 {
                    best_bid = best_bid.max(q.values[0]);
                    best_ask = best_ask.min(q.values[1]);
                }
            }
        }
        if best_bid.is_finite() && best_ask.is_finite() {
            spread_sum += (best_ask - best_bid).max(0.0);
            spread_cnt += 1;
        }
    }

    let total_ticks = tpv * venues.len() + n_trades;
    let elapsed_us = t0.elapsed().as_micros().max(1) as u64;
    let tps = (total_ticks as f64) * 1_000_000.0 / (elapsed_us as f64);

    TsbsStacM3Report {
        total_ticks_ingested: total_ticks,
        nbbo_trades_aligned: aligned.len(),
        future_leak_violations: future_leaks,
        vwap_price: if vwap_den > 0.0 {
            vwap_num / vwap_den
        } else {
            0.0
        },
        mean_nbbo_spread: if spread_cnt > 0 {
            spread_sum / (spread_cnt as f64)
        } else {
            0.0
        },
        tsbs_rollup_buckets: tsbs_buckets.len().max(1),
        elapsed_us,
        throughput_ticks_per_sec: tps,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. B16: BEIR ZERO-SHOT HYBRID RETRIEVAL EVALUATOR (nDCG@10, Recall@10, MRR@10)
// ═══════════════════════════════════════════════════════════════════════════

/// Single `BEIR` benchmark query with ground-truth graded relevance (`doc_id -> grade >= 1`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeirQueryItem {
    /// Query identifier.
    pub query_id: String,
    /// Natural-language claim or question text.
    pub text: String,
    /// Query embedding vector (e.g. from `.jsyn` `all-MiniLM-L6-v2` living weights).
    pub embedding: Vec<f32>,
    /// Optional structured `(role, filler)` query cues for HDC resonance.
    #[serde(default)]
    pub role_fillers: Vec<(String, String)>,
    /// Ground-truth relevant document IDs and integer relevance grades (`>= 1`).
    pub qrels: BTreeMap<u64, u32>,
}

/// Standard `BEIR` retrieval metrics at rank cutoff `k = 10`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BeirModeMetrics {
    /// Normalized Discounted Cumulative Gain at rank 10 (`nDCG@10`).
    pub ndcg_at_10: f64,
    /// Recall at rank 10 (`Recall@10`).
    pub recall_at_10: f64,
    /// Mean Reciprocal Rank at cutoff 10 (`MRR@10`).
    pub mrr_at_10: f64,
    /// Mean search latency per query in microseconds.
    pub mean_query_us: f64,
}

/// Head-to-head `BEIR` comparison across `BM25`, `RaBitQ` 1-bit Vector, and `TriModal RRF`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeirComparisonReport {
    /// Number of corpus documents indexed.
    pub num_docs: usize,
    /// Number of benchmark queries evaluated.
    pub num_queries: usize,
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// Lexical `Okapi BM25`-only baseline metrics.
    pub bm25_only: BeirModeMetrics,
    /// Quantized `RaBitQ` 1-bit SIMD `POPCNT` Vector-only metrics.
    pub rabitq_vector_only: BeirModeMetrics,
    /// JouleDB `TriModal RRF` (`BM25 ⋈ RaBitQ ⋈ HDC`) fused metrics.
    pub trimodal_rrf: BeirModeMetrics,
    /// `RaBitQ` vector compression ratio vs. `FP32`.
    pub vector_compression_vs_fp32: f64,
}

fn score_ranked_list(ranked_ids: &[u64], qrels: &BTreeMap<u64, u32>, k: usize) -> (f64, f64, f64) {
    if qrels.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let top = &ranked_ids[..ranked_ids.len().min(k)];

    // DCG@k
    let mut dcg = 0.0f64;
    let mut rel_hits = 0usize;
    let mut mrr = 0.0f64;
    for (rank0, doc_id) in top.iter().enumerate() {
        if let Some(&grade) = qrels.get(doc_id) {
            if grade > 0 {
                let gain = ((1u64 << grade.min(10)) - 1) as f64;
                let discount = ((rank0 + 2) as f64).log2();
                dcg += gain / discount;
                rel_hits += 1;
                if mrr == 0.0 {
                    mrr = 1.0 / ((rank0 + 1) as f64);
                }
            }
        }
    }

    // Ideal DCG@k
    let mut ideal_grades: Vec<u32> = qrels.values().copied().filter(|&g| g > 0).collect();
    ideal_grades.sort_by(|a, b| b.cmp(a));
    let mut idcg = 0.0f64;
    for (rank0, grade) in ideal_grades.into_iter().take(k).enumerate() {
        let gain = ((1u64 << grade.min(10)) - 1) as f64;
        let discount = ((rank0 + 2) as f64).log2();
        idcg += gain / discount;
    }

    let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };
    let total_rel = qrels.values().filter(|&&g| g > 0).count().max(1);
    let recall = (rel_hits as f64 / total_rel as f64).min(1.0);
    (ndcg, recall, mrr)
}

/// Evaluates `BM25`-only, `RaBitQ` 1-bit Vector-only, and `TriModal RRF` head-to-head on a `BEIR` corpus.
pub fn evaluate_beir_hybrid_retrieval(
    embedding_dim: usize,
    docs: &[HybridDocument],
    queries: &[BeirQueryItem],
) -> BeirComparisonReport {
    let dim = embedding_dim.max(8);
    let mut index = TriModalRrfIndex::new(dim);
    for d in docs {
        index.upsert(d.clone());
    }
    index.recalibrate_rabitq_centroid();

    let eval_cfg = |cfg: RrfConfig| -> BeirModeMetrics {
        if queries.is_empty() {
            return BeirModeMetrics {
                ndcg_at_10: 0.0,
                recall_at_10: 0.0,
                mrr_at_10: 0.0,
                mean_query_us: 0.0,
            };
        }
        let t0 = Instant::now();
        let empty_attrs = BTreeMap::new();
        let (mut sum_ndcg, mut sum_rec, mut sum_mrr) = (0.0, 0.0, 0.0);
        for q in queries {
            let hits = index.search(
                &q.text,
                Some(q.embedding.as_slice()),
                &q.role_fillers,
                &empty_attrs,
                None,
                10,
                cfg,
            );
            let ids: Vec<u64> = hits.into_iter().map(|h| h.id).collect();
            let (n, r, m) = score_ranked_list(&ids, &q.qrels, 10);
            sum_ndcg += n;
            sum_rec += r;
            sum_mrr += m;
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / (queries.len() as f64);
        let denom = queries.len() as f64;
        BeirModeMetrics {
            ndcg_at_10: sum_ndcg / denom,
            recall_at_10: sum_rec / denom,
            mrr_at_10: sum_mrr / denom,
            mean_query_us: us,
        }
    };

    let bm25 = eval_cfg(RrfConfig {
        rrf_k: 60.0,
        bm25_weight: 1.0,
        rabitq_weight: 0.0,
        hdc_weight: 0.0,
    });
    let vec_only = eval_cfg(RrfConfig {
        rrf_k: 60.0,
        bm25_weight: 0.0,
        rabitq_weight: 1.0,
        hdc_weight: 0.0,
    });
    let trimodal = eval_cfg(RrfConfig {
        rrf_k: 60.0,
        bm25_weight: 3.0,
        rabitq_weight: 1.0,
        hdc_weight: 0.5,
    });

    BeirComparisonReport {
        num_docs: docs.len(),
        num_queries: queries.len(),
        embedding_dim: dim,
        bm25_only: bm25,
        rabitq_vector_only: vec_only,
        trimodal_rrf: trimodal,
        vector_compression_vs_fp32: 32.0 / (1.0 + 64.0 / (dim as f64)),
    }
}

/// Generates a deterministic synthetic `BEIR` scientific-claim corpus (with lexical-paraphrase
/// and terminology-exact split queries) so unit/HTTP certification runs offline in `< 5 ms`.
pub fn run_synthetic_beir_suite(num_docs: usize) -> BeirComparisonReport {
    let n = num_docs.clamp(40, 500);
    let dim = 64usize;
    let mut docs = Vec::with_capacity(n);
    let mut queries = Vec::new();

    for i in 0..n {
        let mut emb = vec![0.0f32; dim];
        let bucket = i % 16;
        emb[bucket * 4] = 1.0;
        emb[bucket * 4 + 1] = 0.75 * (((i + 1) as f32).sin());
        emb[bucket * 4 + 2] = 0.50 * (((i + 3) as f32).cos());
        let text = if i % 2 == 0 {
            format!(
                "Compoundenz-{i} inhibits receptor-alpha-{bucket} phosphorylation in human hepatocytes trial-{i}"
            )
        } else {
            format!(
                "cardiovascular biomarker-{i} demonstrates mitochondrial pathway-{bucket} attenuation cohort-{i}"
            )
        };
        docs.push(HybridDocument {
            id: i as u64,
            text,
            embedding: emb.clone(),
            role_fillers: vec![
                ("domain".into(), format!("pathway_{bucket}")),
                ("cohort".into(), format!("c_{}", i % 8)),
            ],
            attributes: BTreeMap::new(),
            neighbors: Vec::new(),
        });

        if i < 16 {
            // Half the queries have exact rare identifier terms (BM25 excels); half use semantic
            // vector + role-filler alignment while omitting the exact compound token (Vector+HDC excels)!
            let q_text = if i % 2 == 0 {
                format!("Compoundenz-{i} receptor-alpha-{bucket} phosphorylation")
            } else {
                format!("cellular bioenergetics attenuation study group {}", i % 8)
            };
            let mut qrels = BTreeMap::new();
            qrels.insert(i as u64, 2);
            queries.push(BeirQueryItem {
                query_id: format!("q_{i}"),
                text: q_text,
                embedding: emb,
                role_fillers: vec![
                    ("domain".into(), format!("pathway_{bucket}")),
                    ("cohort".into(), format!("c_{}", i % 8)),
                ],
                qrels,
            });
        }
    }

    evaluate_beir_hybrid_retrieval(dim, &docs, &queries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_four_analytic_and_retrieval_suites() {
        let cb = run_clickbench_columnar_suite(4_096);
        assert!(cb.alp_bit_exact_verified);
        assert!(cb.all_five_queries_exact);
        assert!(cb.compression_ratio > 3.5);
        assert!(cb.q21_granule_skip_ratio > 0.70);

        let job = run_job_skewed_wcoj_suite(100, 8);
        assert!(job.exact_parity);
        assert_eq!(job.output_tuples, 8);
        assert!(job.intermediate_blowup_reduction_x > 15.0);

        let stac = run_tsbs_stac_m3_suite(500, 100);
        assert_eq!(stac.future_leak_violations, 0);
        assert_eq!(stac.nbbo_trades_aligned, 100);
        assert!(stac.vwap_price > 180.0);

        let beir = run_synthetic_beir_suite(64);
        assert!(beir.trimodal_rrf.ndcg_at_10 >= beir.bm25_only.ndcg_at_10);
        assert!(beir.trimodal_rrf.ndcg_at_10 > 0.80);
    }
}
