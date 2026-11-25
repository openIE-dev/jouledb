//! # Calibration Metrics, Neurosymbolic Solvers & Trajectory Safety Primitives
//!
//! Pure-Rust utility primitives for calibration error computation (`RMS-CE`, `ECE`, `Brier`, Platt scaling),
//! risk-controlled abstention thresholds, multi-source span conflict checks, 2D grid transformation synthesis,
//! exact `i128` modular arithmetic (`CRT`, `ext_gcd`, `mod_pow`) + `DPLL` Boolean SAT, declarative tool-call
//! preconditions, bitemporal key-value history lookup, and Signal Temporal Logic (`STL`) trajectory clamping.
//! Note: These are standalone library algorithms, not benchmark dataset evaluations.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════
// 1. HLE / HLE-DIAMOND CALIBRATION (RMS-CE) & SIMPLEQA ABSTENTION
// ═══════════════════════════════════════════════════════════════════════════

/// Single prediction record with ground-truth correctness and stated confidence in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Whether the model's final answer matched the verified reference solution.
    pub correct: bool,
    /// Stated confidence in `[0.0, 1.0]` (or `[0, 100]` automatically normalized if `> 1.0`).
    pub confidence: f64,
    /// True if item belongs to the `HLE-Diamond` Reasoning split (`false` = Knowledge split).
    #[serde(default)]
    pub is_reasoning_split: bool,
}

/// Complete `HLE` / `HLE-Diamond` calibration and split-accuracy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Total evaluated predictions.
    pub num_samples: usize,
    /// Overall `pass@1` accuracy in `[0.0, 1.0]`.
    pub accuracy: f64,
    /// Mean stated confidence in `[0.0, 1.0]`.
    pub mean_confidence: f64,
    /// Hendrycks adaptive equal-mass Root Mean Square Calibration Error (`RMS-CE`).
    pub rms_calibration_error: f64,
    /// Expected Calibration Error (`ECE`, L1 weighted bin gap).
    pub expected_calibration_error: f64,
    /// Proper scoring rule Brier score `mean((1[correct] - conf)^2)`.
    pub brier_score: f64,
    /// Accuracy on `is_reasoning_split == true` items (`HLE-Diamond` 500 Reasoning half).
    pub reasoning_split_accuracy: f64,
    /// Accuracy on `is_reasoning_split == false` items (`HLE-Diamond` 500 Knowledge half).
    pub knowledge_split_accuracy: f64,
}

fn norm_conf(c: f64) -> f64 {
    let scaled = if c > 1.0 { c / 100.0 } else { c };
    scaled.clamp(1e-6, 1.0 - 1e-6)
}

/// Computes `HLE` Hendrycks adaptive-bin `RMS-CE`, `ECE`, `Brier`, and Reasoning/Knowledge split accuracy.
pub fn compute_hle_calibration(
    samples: &[CalibrationSample],
    min_bin_size: usize,
) -> CalibrationMetrics {
    if samples.is_empty() {
        return CalibrationMetrics {
            num_samples: 0,
            accuracy: 0.0,
            mean_confidence: 0.0,
            rms_calibration_error: 0.0,
            expected_calibration_error: 0.0,
            brier_score: 0.0,
            reasoning_split_accuracy: 0.0,
            knowledge_split_accuracy: 0.0,
        };
    }

    let n = samples.len();
    let mut sorted: Vec<(f64, f64)> = samples
        .iter()
        .map(|s| (norm_conf(s.confidence), if s.correct { 1.0 } else { 0.0 }))
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    let bin_sz = min_bin_size.max(1).min(n);
    let mut sq_weighted = 0.0;
    let mut abs_weighted = 0.0;
    let mut idx = 0;

    while idx < n {
        let mut end = (idx + bin_sz).min(n);
        if n - end < bin_sz {
            end = n;
        }
        let slice = &sorted[idx..end];
        let m = slice.len() as f64;
        let conf_mean = slice.iter().map(|x| x.0).sum::<f64>() / m;
        let acc_mean = slice.iter().map(|x| x.1).sum::<f64>() / m;
        let diff = (acc_mean - conf_mean).abs();
        let weight = m / (n as f64);
        sq_weighted += weight * diff * diff;
        abs_weighted += weight * diff;
        idx = end;
    }

    let total_acc = sorted.iter().map(|x| x.1).sum::<f64>() / (n as f64);
    let total_conf = sorted.iter().map(|x| x.0).sum::<f64>() / (n as f64);
    let brier = sorted.iter().map(|(c, y)| (y - c) * (y - c)).sum::<f64>() / (n as f64);

    let (mut r_ok, mut r_tot, mut k_ok, mut k_tot) = (0usize, 0usize, 0usize, 0usize);
    for s in samples {
        if s.is_reasoning_split {
            r_tot += 1;
            if s.correct {
                r_ok += 1;
            }
        } else {
            k_tot += 1;
            if s.correct {
                k_ok += 1;
            }
        }
    }

    CalibrationMetrics {
        num_samples: n,
        accuracy: total_acc,
        mean_confidence: total_conf,
        rms_calibration_error: sq_weighted.sqrt(),
        expected_calibration_error: abs_weighted,
        brier_score: brier,
        reasoning_split_accuracy: if r_tot > 0 {
            r_ok as f64 / r_tot as f64
        } else {
            total_acc
        },
        knowledge_split_accuracy: if k_tot > 0 {
            k_ok as f64 / k_tot as f64
        } else {
            total_acc
        },
    }
}

/// Fits post-hoc Platt temperature + bias calibration `(T, bias)` on held-out calibration samples
/// by minimizing negative log-likelihood (`NLL`), mapping raw overconfident logits to calibrated probabilities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EpistemicCalibrator {
    /// Fitted temperature `T > 0` dividing raw logit.
    pub temperature: f64,
    /// Fitted intercept shift in logit space.
    pub logit_bias: f64,
}

impl EpistemicCalibrator {
    /// Fits `(temperature, logit_bias)` on calibration samples via deterministic grid + coordinate refinement.
    pub fn fit(calib: &[CalibrationSample]) -> Self {
        if calib.is_empty() {
            return Self {
                temperature: 1.0,
                logit_bias: 0.0,
            };
        }
        let logits: Vec<(f64, f64)> = calib
            .iter()
            .map(|s| {
                let p = norm_conf(s.confidence);
                let z = (p / (1.0 - p)).ln();
                (z, if s.correct { 1.0 } else { 0.0 })
            })
            .collect();

        let nll = |temp: f64, bias: f64| -> f64 {
            let mut loss = 0.0;
            for &(z, y) in &logits {
                let s = (z / temp + bias).clamp(-30.0, 30.0);
                let p = 1.0 / (1.0 + (-s).exp());
                loss -= y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln();
            }
            loss / (logits.len() as f64)
        };

        let mut best_t = 1.0;
        let mut best_b = 0.0;
        let mut best_loss = nll(1.0, 0.0);

        for ti in 1..=40 {
            let t = 0.25 * (ti as f64);
            for bi in -24..=24 {
                let b = 0.25 * (bi as f64);
                let l = nll(t, b);
                if l < best_loss {
                    best_loss = l;
                    best_t = t;
                    best_b = b;
                }
            }
        }
        // Fine local polish
        for dt in -10..=10 {
            let t = (best_t + 0.02 * (dt as f64)).max(0.05);
            for db in -10..=10 {
                let b = best_b + 0.02 * (db as f64);
                let l = nll(t, b);
                if l < best_loss {
                    best_loss = l;
                    best_t = t;
                    best_b = b;
                }
            }
        }

        Self {
            temperature: best_t,
            logit_bias: best_b,
        }
    }

    /// Maps a raw confidence `c` to calibrated posterior probability `P(correct | c)`.
    pub fn calibrate(&self, raw_confidence: f64) -> f64 {
        let p = norm_conf(raw_confidence);
        let z = (p / (1.0 - p)).ln();
        let s = (z / self.temperature + self.logit_bias).clamp(-30.0, 30.0);
        1.0 / (1.0 + (-s).exp())
    }
}

/// `SimpleQA` / `FACTS` selective abstention summary under wrong-answer penalty `penalty_wrong`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleQaReport {
    /// Total questions evaluated.
    pub total: usize,
    /// Number of questions attempted (`calibrated_conf >= p / (1 + p)`).
    pub attempted: usize,
    /// Number of questions abstained (`not_attempted`).
    pub abstained: usize,
    /// Number of attempted answers that were correct.
    pub correct: usize,
    /// Number of attempted answers that were incorrect (hallucinations).
    pub incorrect: usize,
    /// Precision on attempted questions (`correct / attempted`, `1.0` if `attempted == 0`).
    pub accuracy_given_attempted: f64,
    /// Overall accuracy (`correct / total`).
    pub overall_accuracy: f64,
    /// Official OpenAI `SimpleQA` harmonic F-score combining `overall_accuracy` and `accuracy_given_attempted`.
    pub simpleqa_f_score: f64,
    /// Normalized penalty-adjusted utility `(correct - penalty_wrong * incorrect) / total`.
    pub penalty_adjusted_score: f64,
}

/// Evaluates Bayes-optimal selective abstention (`abstains iff P(correct) < penalty / (1 + penalty)`).
pub fn evaluate_simpleqa_abstention(
    samples: &[CalibrationSample],
    calibrator: Option<&EpistemicCalibrator>,
    penalty_wrong: f64,
    has_grounded_provenance: &[bool],
) -> SimpleQaReport {
    let total = samples.len();
    if total == 0 {
        return SimpleQaReport {
            total: 0,
            attempted: 0,
            abstained: 0,
            correct: 0,
            incorrect: 0,
            accuracy_given_attempted: 1.0,
            overall_accuracy: 0.0,
            simpleqa_f_score: 0.0,
            penalty_adjusted_score: 0.0,
        };
    }

    let pen = penalty_wrong.max(0.0);
    let threshold = if pen <= 0.0 { 0.0 } else { pen / (1.0 + pen) };

    let (mut attempted, mut abstained, mut correct, mut incorrect) =
        (0usize, 0usize, 0usize, 0usize);
    for (i, s) in samples.iter().enumerate() {
        let grounded = has_grounded_provenance.get(i).copied().unwrap_or(true);
        let p = match calibrator {
            Some(cal) => cal.calibrate(s.confidence),
            None => norm_conf(s.confidence),
        };
        if grounded && p >= threshold {
            attempted += 1;
            if s.correct {
                correct += 1;
            } else {
                incorrect += 1;
            }
        } else {
            abstained += 1;
        }
    }

    let acc_attempted = if attempted > 0 {
        correct as f64 / attempted as f64
    } else {
        1.0
    };
    let overall_acc = correct as f64 / total as f64;
    let f_score = if overall_acc + acc_attempted > 0.0 {
        2.0 * overall_acc * acc_attempted / (overall_acc + acc_attempted)
    } else {
        0.0
    };
    let utility = ((correct as f64) - pen * (incorrect as f64)) / (total as f64);

    SimpleQaReport {
        total,
        attempted,
        abstained,
        correct,
        incorrect,
        accuracy_given_attempted: acc_attempted,
        overall_accuracy: overall_acc,
        simpleqa_f_score: f_score,
        penalty_adjusted_score: utility,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. FAITHEVAL COUNTERFACTUAL / CONFLICTING CONTEXT & MULTICHALLENGE
// ═══════════════════════════════════════════════════════════════════════════

/// Versioned contextual claim anchored to a source document and MVCC LSN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFact {
    /// Document or turn identifier.
    pub source_id: String,
    /// Canonical subject/entity key.
    pub entity: String,
    /// Canonical attribute/predicate key.
    pub attribute: String,
    /// Asserted value in the provided context.
    pub value: String,
    /// Monotonic turn/commit LSN (`MultiChallenge` versioned edits).
    pub valid_from_lsn: u64,
}

/// Deterministic `FaithEval` & `MultiChallenge` resolution outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaithEvalOutcome {
    /// Context snapshot provides a single consistent authoritative value (overrides parametric priors!).
    GroundedValue {
        /// Authoritative value extracted from snapshot.
        value: String,
        /// Winning LSN.
        lsn: u64,
    },
    /// Multiple distinct sources at the active LSN assert mutually contradictory values (`FaithEval-Inconsistent`).
    InconsistentConflict {
        /// Sorted conflicting values detected across sources.
        conflicting_values: Vec<String>,
    },
    /// Context contains no supporting claim for `(entity, attribute)` (`FaithEval-Unanswerable` — blocks parametric hallucination).
    UnanswerableAbstain,
}

/// Evaluates `(entity, attribute)` strictly under `FaithEval` counterfactual & conflict rules
/// and `MultiChallenge` bitemporal LSN precedence (`parametric_prior` is intentionally ignored!).
pub fn resolve_faithful_context(
    facts: &[ContextFact],
    _parametric_prior: Option<&str>,
    entity: &str,
    attribute: &str,
    as_of_lsn: Option<u64>,
) -> FaithEvalOutcome {
    let ent = entity.trim().to_ascii_lowercase();
    let attr = attribute.trim().to_ascii_lowercase();
    let ceiling = as_of_lsn.unwrap_or(u64::MAX);

    let matching: Vec<&ContextFact> = facts
        .iter()
        .filter(|f| {
            f.valid_from_lsn <= ceiling
                && f.entity.trim().to_ascii_lowercase() == ent
                && f.attribute.trim().to_ascii_lowercase() == attr
        })
        .collect();

    let Some(max_lsn) = matching.iter().map(|f| f.valid_from_lsn).max() else {
        return FaithEvalOutcome::UnanswerableAbstain;
    };

    let mut active_vals: Vec<String> = matching
        .into_iter()
        .filter(|f| f.valid_from_lsn == max_lsn)
        .map(|f| f.value.trim().to_string())
        .collect();
    active_vals.sort();
    active_vals.dedup();

    if active_vals.len() == 1 {
        FaithEvalOutcome::GroundedValue {
            value: active_vals.remove(0),
            lsn: max_lsn,
        }
    } else {
        FaithEvalOutcome::InconsistentConflict {
            conflicting_values: active_vals,
        }
    }
}

/// Passage-level faithfulness audit for `FaithEval` (*Counterfactual*, *Inconsistent*, *Unanswerable*).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaithfulPassageAudit {
    /// Fraction of salient question content tokens grounded in the provided passage (`[0.0, 1.0]`).
    pub grounding_coverage: f64,
    /// True when core question entities/predicates are absent from the context (`FaithEval-Unanswerable`).
    pub unanswerable_abstain: bool,
    /// True when distinct passage segments/documents assert conflicting predications (`FaithEval-Inconsistent`).
    pub inconsistent_conflict: bool,
    /// Distinct conflicting candidate spans detected across passage segments, if any.
    pub conflicting_spans: Vec<String>,
}

/// Scored multiple-choice prediction from [`score_faithful_mc_question`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaithfulMcPrediction {
    /// Winning 0-based choice index (`None` if abstaining due to unanswerable/inconsistent context when gated).
    pub predicted_index: Option<usize>,
    /// Per-choice neurosymbolic span + polarity + optional `.jsyn` RRF scores.
    pub choice_scores: Vec<f64>,
    /// Passage-level grounding and contradiction audit.
    pub audit: FaithfulPassageAudit,
}

fn tokenize_content_words(text: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "according",
        "passage",
        "context",
        "following",
        "based",
        "text",
        "does",
        "did",
        "do",
        "has",
        "have",
        "had",
        "would",
        "could",
        "should",
        "will",
        "can",
        "may",
        "might",
        "into",
        "about",
        "over",
        "after",
        "before",
        "between",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect()
}

fn contains_negation(tokens_raw: &[String]) -> bool {
    const NEG: &[&str] = &[
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "none",
        "without",
        "unable",
        "false",
        "untrue",
        "instead",
        "unlikely",
        "impossible",
        "lack",
        "lacks",
        "lacked",
    ];
    tokens_raw.iter().any(|t| NEG.contains(&t.as_str()))
}

fn raw_words_lower(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect()
}

fn split_sentences(context: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in context.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let trimmed = cur.trim();
            if trimmed.len() >= 4 {
                out.push(trimmed.to_string());
            }
            cur.clear();
        }
    }
    let rem = cur.trim();
    if rem.len() >= 4 {
        out.push(rem.to_string());
    }
    if out.is_empty() && !context.trim().is_empty() {
        out.push(context.trim().to_string());
    }
    out
}

/// Audits an unstructured passage + question for `FaithEval-Unanswerable` missing grounding
/// and `FaithEval-Inconsistent` multi-document/multi-sentence contradictions.
pub fn audit_passage_faithfulness(
    context: &str,
    question: &str,
    candidate_values: &[String],
) -> FaithfulPassageAudit {
    let q_toks = tokenize_content_words(question);
    let ctx_toks: std::collections::HashSet<String> =
        tokenize_content_words(context).into_iter().collect();

    let matched = q_toks.iter().filter(|t| ctx_toks.contains(*t)).count();
    let coverage = if q_toks.is_empty() {
        1.0
    } else {
        matched as f64 / q_toks.len() as f64
    };

    // Detect conflicting assertions across sentences/documents on high-focus sentences
    let sents = split_sentences(context);
    let mut asserted_candidates: Vec<String> = Vec::new();
    for sent in &sents {
        let s_toks: std::collections::HashSet<String> =
            tokenize_content_words(sent).into_iter().collect();
        let q_overlap = q_toks.iter().filter(|t| s_toks.contains(*t)).count();
        if q_overlap == 0 && !q_toks.is_empty() {
            continue;
        }
        let s_raw = raw_words_lower(sent);
        if contains_negation(&s_raw) {
            continue;
        }
        let s_low = sent.to_ascii_lowercase();
        for cand in candidate_values {
            let c_trim = cand.trim();
            if c_trim.is_empty() {
                continue;
            }
            let c_low = c_trim.to_ascii_lowercase();
            let c_toks = tokenize_content_words(&c_low);
            let hit = if c_toks.is_empty() {
                s_raw.contains(&c_low)
            } else {
                s_low.contains(&c_low) || c_toks.iter().all(|ct| s_toks.contains(ct))
            };
            if hit && !asserted_candidates.contains(&c_trim.to_string()) {
                asserted_candidates.push(c_trim.to_string());
            }
        }
    }

    let inconsistent = asserted_candidates.len() >= 2;
    let unanswerable = coverage < 0.25
        || (!candidate_values.is_empty() && asserted_candidates.is_empty() && coverage < 0.45);

    FaithfulPassageAudit {
        grounding_coverage: coverage,
        unanswerable_abstain: unanswerable,
        inconsistent_conflict: inconsistent,
        conflicting_spans: if inconsistent {
            asserted_candidates
        } else {
            Vec::new()
        },
    }
}

/// Neurosymbolic `FaithEval` local-span, negation-polarity, and optional `.jsyn` cosine RRF reader.
///
/// Unlike standalone bi-encoder cosine similarity (which averages the entire passage and suffers from
/// parametric distraction), this reader weights sentences by IDF-scaled question-focus overlap,
/// evaluates a tight `±1` sentence window around the counterfactual predication, penalizes negated
/// distractors, and optionally fuses `.jsyn` living-weight cosine scores.
pub fn score_faithful_mc_question(
    context: &str,
    question: &str,
    choices: &[String],
    optional_jsyn_cosines: Option<&[f64]>,
    gate_abstention: bool,
) -> FaithfulMcPrediction {
    let audit = audit_passage_faithfulness(context, question, choices);
    if choices.is_empty() {
        return FaithfulMcPrediction {
            predicted_index: None,
            choice_scores: Vec::new(),
            audit,
        };
    }

    let sents = split_sentences(context);
    let sent_toks: Vec<Vec<String>> = sents.iter().map(|s| tokenize_content_words(s)).collect();
    let mut df: HashMap<String, usize> = HashMap::new();
    for st in &sent_toks {
        let mut uniq = st.clone();
        uniq.sort();
        uniq.dedup();
        for tok in uniq {
            *df.entry(tok).or_insert(0) += 1;
        }
    }
    let n_sents = sents.len().max(1) as f64;
    let idf = |t: &str| -> f64 {
        let d = *df.get(t).unwrap_or(&0) as f64;
        ((n_sents + 1.0) / (d + 1.0)).ln() + 1.0
    };

    let q_toks = tokenize_content_words(question);
    let q_neg = contains_negation(&raw_words_lower(question));

    // Score each sentence's relevance to the question
    let prefix5_eq = |a: &str, b: &str| -> bool {
        let mut ca = a.chars();
        let mut cb = b.chars();
        for _ in 0..5 {
            match (ca.next(), cb.next()) {
                (Some(xa), Some(xb)) if xa == xb => {}
                _ => return false,
            }
        }
        true
    };

    let mut sent_q_weights = vec![0.0f64; sents.len()];
    for (si, st) in sent_toks.iter().enumerate() {
        let mut w = 0.15;
        for qt in &q_toks {
            if st.iter().any(|x| x == qt) {
                w += idf(qt);
            } else if qt.chars().count() >= 5 && st.iter().any(|x| prefix5_eq(x, qt)) {
                w += 0.65 * idf(qt);
            }
        }
        sent_q_weights[si] = w;
    }

    // Smooth ±1 neighboring sentences for coreference ("It...", "Instead, this...")
    let mut window_weights = sent_q_weights.clone();
    for i in 0..sents.len() {
        if i > 0 {
            window_weights[i] += 0.35 * sent_q_weights[i - 1];
        }
        if i + 1 < sents.len() {
            window_weights[i] += 0.35 * sent_q_weights[i + 1];
        }
    }

    let mut scores = vec![0.0f64; choices.len()];
    for (ci, choice) in choices.iter().enumerate() {
        let c_low = choice.trim().to_ascii_lowercase();
        let c_toks: Vec<String> = tokenize_content_words(choice)
            .into_iter()
            .filter(|t| !q_toks.contains(t))
            .collect();
        let effective_toks = if c_toks.is_empty() {
            raw_words_lower(choice)
        } else {
            c_toks
        };
        let c_neg = contains_negation(&raw_words_lower(choice));

        let mut best_sent_score = 0.0f64;
        let mut total_sent_score = 0.0f64;
        for (si, sent) in sents.iter().enumerate() {
            let s_low = sent.to_ascii_lowercase();
            let s_raw = raw_words_lower(sent);
            let s_neg = contains_negation(&s_raw);

            let mut lex_match = 0.0f64;
            if !c_low.is_empty() && c_low.len() >= 3 && s_low.contains(&c_low) {
                lex_match += 3.0;
            }
            if !effective_toks.is_empty() {
                let mut hit_sum = 0.0;
                let mut norm_sum = 0.0;
                for ct in &effective_toks {
                    let w_idf = idf(ct);
                    norm_sum += w_idf;
                    if s_raw.iter().any(|x| x == ct) {
                        hit_sum += w_idf;
                    } else if ct.chars().count() >= 5 && s_raw.iter().any(|x| prefix5_eq(x, ct)) {
                        hit_sum += 0.7 * w_idf;
                    }
                }
                if norm_sum > 0.0 {
                    lex_match += 2.5 * (hit_sum / norm_sum);
                }
            }

            if lex_match > 0.0 {
                // Polarity check: if question asks affirmative ("What IS...") and sentence negates
                // the candidate ("is NOT X, it is Y"), penalize X!
                let polarity_mult = if (s_neg ^ c_neg) != q_neg { 0.30 } else { 1.0 };
                let contrib = window_weights[si] * lex_match * polarity_mult;
                best_sent_score = best_sent_score.max(contrib);
                total_sent_score += 0.25 * contrib;
            }
        }

        let mut final_score = best_sent_score + total_sent_score;
        if let Some(jsyn) = optional_jsyn_cosines {
            if let Some(&cos) = jsyn.get(ci) {
                final_score += 2.2 * cos;
            }
        }
        scores[ci] = final_score;
    }

    // Apply authoritative meta-choice precedence after computing base span scores so an explicit
    // "conflict / inconsistent" or "unanswerable / unknown" option strictly beats partial spans.
    let max_span_score = scores.iter().copied().fold(0.0_f64, f64::max);
    for (ci, choice) in choices.iter().enumerate() {
        let c_low = choice.trim().to_ascii_lowercase();
        if audit.unanswerable_abstain
            && (c_low.contains("unknown")
                || c_low.contains("unanswerable")
                || c_low.contains("cannot be determined")
                || c_low.contains("not enough information"))
        {
            scores[ci] = max_span_score + 10.0;
        }
        if audit.inconsistent_conflict
            && (c_low.contains("conflict")
                || c_low.contains("contradict")
                || c_low.contains("inconsistent"))
        {
            scores[ci] = max_span_score + 10.0;
        }
    }

    if gate_abstention && (audit.unanswerable_abstain || audit.inconsistent_conflict) {
        return FaithfulMcPrediction {
            predicted_index: None,
            choice_scores: scores,
            audit,
        };
    }

    let best_idx = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx);

    FaithfulMcPrediction {
        predicted_index: best_idx,
        choice_scores: scores,
        audit,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. ARC-AGI-2 PROGRAM SYNTHESIS DSL & ARC-AGI-3 ACTIVE PROBE ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// Boolean operator applied across split subgrid panels in `ARC-AGI` tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelLogicKind {
    /// Cell-wise exclusive-OR `(a != 0) ^ (b != 0)`.
    Xor,
    /// Cell-wise conjunction `(a != 0) && (b != 0)`.
    And,
    /// Cell-wise disjunction `(a != 0) || (b != 0)`.
    Or,
    /// Cell-wise joint denial `(a == 0) && (b == 0)`.
    Nor,
}

/// 2D palette grid (`0..=9`) for `ARC-AGI-2` and `ARC-AGI-3` abstract reasoning tasks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArcGrid {
    /// Row-major `height x width` color cells (`0` = background).
    pub cells: Vec<Vec<u8>>,
}

impl ArcGrid {
    /// Constructs a validated rectangular `ArcGrid`.
    pub fn new(cells: Vec<Vec<u8>>) -> Self {
        Self { cells }
    }

    /// Grid height (`rows`).
    pub fn height(&self) -> usize {
        self.cells.len()
    }

    /// Grid width (`cols`).
    pub fn width(&self) -> usize {
        self.cells.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// Compositional `ARC-AGI` 2D grid transformation DSL operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArcOp {
    /// Identity (`g -> g`).
    Identity,
    /// 90-degree clockwise rotation.
    Rot90,
    /// 180-degree rotation.
    Rot180,
    /// 270-degree clockwise rotation.
    Rot270,
    /// Horizontal mirror reflection (`left <-> right`).
    FlipH,
    /// Vertical mirror reflection (`top <-> bottom`).
    FlipV,
    /// Matrix transpose (`(r, c) -> (c, r)`).
    Transpose,
    /// Kronecker self-tiling (`g[r][c] != 0 ? g : zeros(H, W)` — e.g. `ARC` `007bbfb7`).
    KroneckerSelfTile,
    /// Repeats grid `rows_rep x cols_rep`, horizontally flipping odd block-rows (e.g. `ARC-AGI-2` `00576224`).
    TileAlternatingFlipH {
        /// Vertical tile count.
        rows_rep: usize,
        /// Horizontal tile count.
        cols_rep: usize,
    },
    /// Standard periodic tiling `rows_rep x cols_rep`.
    TileRepeat {
        /// Vertical tile count.
        rows_rep: usize,
        /// Horizontal tile count.
        cols_rep: usize,
    },
    /// Nearest-neighbor integer upscaling by `factor`.
    ScaleUp {
        /// Pixel replication factor (`>= 2`).
        factor: usize,
    },
    /// Crops to minimal bounding box of all non-zero pixels.
    CropNonZeroBBox,
    /// Drops non-zero pixels to the bottom of each column (`gravity`).
    GravityDown,
    /// Slides non-zero pixels to the right of each row.
    GravityRight,
    /// Rigid 2D translation by `(dr, dc)` with background `0` padding (e.g. `ARC` `25ff71a9`).
    Shift {
        /// Row offset (`+1` = down 1 row, `-1` = up 1 row).
        dr: i32,
        /// Column offset (`+1` = right 1 col, `-1` = left 1 col).
        dc: i32,
    },
    /// Fills enclosed interior `0`-cavities unreachable from the grid border with `fill_color`.
    FillEnclosedHoles {
        /// Palette value for enclosed holes.
        fill_color: u8,
    },
    /// Erodes interior pixels whose 4-connected neighbors are all non-zero, leaving 1px boundaries.
    HollowInterior,
    /// Completes horizontal mirror symmetry (`g[r][c] | g[r][W-1-c]`).
    CompleteHorizontalSymmetry,
    /// Completes vertical mirror symmetry (`g[r][c] | g[H-1-r][c]`).
    CompleteVerticalSymmetry,
    /// Retains only the largest 4-connected non-zero object (zeros smaller distractor components).
    KeepLargestObject,
    /// Crops to the minimal bounding box of the largest 4-connected non-zero object.
    CropLargestObject,
    /// Retains only pixels belonging to the rarest non-zero palette color in the grid.
    KeepUniqueColorObject,
    /// Crops to the minimal bounding box of the rarest non-zero palette color in the grid.
    CropUniqueColorObject,
    /// Connects orthogonally aligned pixels of the same non-zero color across `0` background gaps.
    ConnectSameColorOrthogonal,
    /// Splits a grid across a uniform separator row/col (or exact halves) and applies boolean panel logic.
    SplitPanelBool {
        /// Boolean operator (`Xor`, `And`, `Or`, `Nor`).
        op: PanelLogicKind,
        /// Foreground palette color written where the boolean predicate holds.
        output_color: u8,
    },
    /// Summarizes each `(row_factor x col_factor)` block by its majority non-zero color (`Downsample`).
    DownsampleBlockMajority {
        /// Vertical block height.
        row_factor: usize,
        /// Horizontal block width.
        col_factor: usize,
    },
    /// Swaps background `0` and non-zero foreground (`0 -> fg_color`, `non-zero -> 0`).
    InvertBinary {
        /// Palette color assigned to former `0` cells.
        fg_color: u8,
    },
    /// Applies palette color substitution table.
    RecolorMap(Vec<(u8, u8)>),
    /// Sequential composition `second(first(grid))`.
    Compose(Box<ArcOp>, Box<ArcOp>),
}

#[derive(Clone)]
struct ConnectedComp {
    pixels: Vec<(usize, usize, u8)>,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
}

fn extract_4connected_components(g: &ArcGrid) -> Vec<ConnectedComp> {
    let (h, w) = (g.height(), g.width());
    let mut vis = vec![vec![false; w]; h];
    let mut comps = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if g.cells[r][c] != 0 && !vis[r][c] {
                let mut q = VecDeque::new();
                vis[r][c] = true;
                q.push_back((r, c));
                let mut pixels = Vec::new();
                let (mut r0, mut r1, mut c0, mut c1) = (r, r, c, c);
                while let Some((cr, cc)) = q.pop_front() {
                    pixels.push((cr, cc, g.cells[cr][cc]));
                    r0 = r0.min(cr);
                    r1 = r1.max(cr);
                    c0 = c0.min(cc);
                    c1 = c1.max(cc);
                    let nbrs = [
                        (cr.wrapping_sub(1), cc),
                        (cr + 1, cc),
                        (cr, cc.wrapping_sub(1)),
                        (cr, cc + 1),
                    ];
                    for (nr, nc) in nbrs {
                        if nr < h && nc < w && g.cells[nr][nc] != 0 && !vis[nr][nc] {
                            vis[nr][nc] = true;
                            q.push_back((nr, nc));
                        }
                    }
                }
                comps.push(ConnectedComp {
                    pixels,
                    r0,
                    r1,
                    c0,
                    c1,
                });
            }
        }
    }
    comps
}

fn rarest_nonzero_color(g: &ArcGrid) -> Option<u8> {
    let mut counts = [0usize; 256];
    for row in &g.cells {
        for &v in row {
            if v != 0 {
                counts[v as usize] += 1;
            }
        }
    }
    (1u8..=9)
        .filter(|&c| counts[c as usize] > 0)
        .min_by_key(|&c| counts[c as usize])
}

fn dominant_nonzero_color(g: &ArcGrid) -> Option<u8> {
    let mut counts = [0usize; 256];
    for row in &g.cells {
        for &v in row {
            if v != 0 {
                counts[v as usize] += 1;
            }
        }
    }
    (1u8..=9)
        .filter(|&c| counts[c as usize] > 0)
        .max_by_key(|&c| counts[c as usize])
}

fn split_two_panels(g: &ArcGrid) -> Option<(ArcGrid, ArcGrid)> {
    let (h, w) = (g.height(), g.width());
    if h == 0 || w == 0 {
        return None;
    }
    // Check vertical separator column at exact center `w = 2 * sub_w + 1`
    if w >= 3 && w % 2 == 1 {
        let mid = w / 2;
        let sep = g.cells[0][mid];
        if sep != 0 && (0..h).all(|r| g.cells[r][mid] == sep) {
            let left: Vec<Vec<u8>> = (0..h).map(|r| g.cells[r][..mid].to_vec()).collect();
            let right: Vec<Vec<u8>> = (0..h).map(|r| g.cells[r][mid + 1..].to_vec()).collect();
            return Some((ArcGrid::new(left), ArcGrid::new(right)));
        }
    }
    // Check horizontal separator row at exact center `h = 2 * sub_h + 1`
    if h >= 3 && h % 2 == 1 {
        let mid = h / 2;
        let sep = g.cells[mid][0];
        if sep != 0 && (0..w).all(|c| g.cells[mid][c] == sep) {
            let top = g.cells[..mid].to_vec();
            let bot = g.cells[mid + 1..].to_vec();
            return Some((ArcGrid::new(top), ArcGrid::new(bot)));
        }
    }
    // Fallback: exact even split without divider if `w >= 2 * h` or `h >= 2 * w`
    if w >= 2 && w % 2 == 0 && w >= h * 2 {
        let mid = w / 2;
        let left: Vec<Vec<u8>> = (0..h).map(|r| g.cells[r][..mid].to_vec()).collect();
        let right: Vec<Vec<u8>> = (0..h).map(|r| g.cells[r][mid..].to_vec()).collect();
        return Some((ArcGrid::new(left), ArcGrid::new(right)));
    }
    if h >= 2 && h % 2 == 0 && h >= w * 2 {
        let mid = h / 2;
        let top = g.cells[..mid].to_vec();
        let bot = g.cells[mid..].to_vec();
        return Some((ArcGrid::new(top), ArcGrid::new(bot)));
    }
    None
}

impl ArcOp {
    /// Minimum Description Length (`MDL`) complexity cost used to prefer simpler generalizations.
    pub fn mdl_cost(&self) -> u32 {
        match self {
            Self::Identity => 1,
            Self::Rot90
            | Self::Rot180
            | Self::Rot270
            | Self::FlipH
            | Self::FlipV
            | Self::Transpose => 2,
            Self::CropNonZeroBBox | Self::GravityDown | Self::GravityRight | Self::Shift { .. } => {
                3
            }
            Self::CompleteHorizontalSymmetry
            | Self::CompleteVerticalSymmetry
            | Self::HollowInterior => 3,
            Self::KeepLargestObject
            | Self::CropLargestObject
            | Self::KeepUniqueColorObject
            | Self::CropUniqueColorObject
            | Self::ConnectSameColorOrthogonal
            | Self::InvertBinary { .. } => 4,
            Self::ScaleUp { .. }
            | Self::TileRepeat { .. }
            | Self::DownsampleBlockMajority { .. }
            | Self::SplitPanelBool { .. } => 4,
            Self::TileAlternatingFlipH { .. }
            | Self::KroneckerSelfTile
            | Self::FillEnclosedHoles { .. } => 5,
            Self::RecolorMap(m) => 3 + m.len() as u32,
            Self::Compose(a, b) => 1 + a.mdl_cost() + b.mdl_cost(),
        }
    }

    /// Evaluates this `ArcOp` deterministically on `g`.
    pub fn apply(&self, g: &ArcGrid) -> ArcGrid {
        let h = g.height();
        let w = g.width();
        if h == 0 || w == 0 {
            return g.clone();
        }
        match self {
            Self::Identity => g.clone(),
            Self::Rot90 => {
                let mut out = vec![vec![0u8; h]; w];
                for r in 0..h {
                    for c in 0..w {
                        out[c][h - 1 - r] = g.cells[r][c];
                    }
                }
                ArcGrid::new(out)
            }
            Self::Rot180 => {
                let mut out = vec![vec![0u8; w]; h];
                for r in 0..h {
                    for c in 0..w {
                        out[h - 1 - r][w - 1 - c] = g.cells[r][c];
                    }
                }
                ArcGrid::new(out)
            }
            Self::Rot270 => {
                let mut out = vec![vec![0u8; h]; w];
                for r in 0..h {
                    for c in 0..w {
                        out[w - 1 - c][r] = g.cells[r][c];
                    }
                }
                ArcGrid::new(out)
            }
            Self::FlipH => {
                let out = g
                    .cells
                    .iter()
                    .map(|row| row.iter().copied().rev().collect())
                    .collect();
                ArcGrid::new(out)
            }
            Self::FlipV => {
                let out = g.cells.iter().cloned().rev().collect();
                ArcGrid::new(out)
            }
            Self::Transpose => {
                let mut out = vec![vec![0u8; h]; w];
                for r in 0..h {
                    for c in 0..w {
                        out[c][r] = g.cells[r][c];
                    }
                }
                ArcGrid::new(out)
            }
            Self::KroneckerSelfTile => {
                if h * h > 64 || w * w > 64 {
                    return g.clone();
                }
                let mut out = vec![vec![0u8; w * w]; h * h];
                for br in 0..h {
                    for bc in 0..w {
                        if g.cells[br][bc] != 0 {
                            for r in 0..h {
                                for c in 0..w {
                                    out[br * h + r][bc * w + c] = g.cells[r][c];
                                }
                            }
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::TileAlternatingFlipH { rows_rep, cols_rep } => {
                let rr = (*rows_rep).clamp(1, 8);
                let cr = (*cols_rep).clamp(1, 8);
                if h * rr > 64 || w * cr > 64 {
                    return g.clone();
                }
                let mut out = vec![vec![0u8; w * cr]; h * rr];
                for br in 0..rr {
                    let flip = br % 2 == 1;
                    for bc in 0..cr {
                        for r in 0..h {
                            for c in 0..w {
                                let src_c = if flip { w - 1 - c } else { c };
                                out[br * h + r][bc * w + c] = g.cells[r][src_c];
                            }
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::TileRepeat { rows_rep, cols_rep } => {
                let rr = (*rows_rep).clamp(1, 8);
                let cr = (*cols_rep).clamp(1, 8);
                if h * rr > 64 || w * cr > 64 {
                    return g.clone();
                }
                let mut out = vec![vec![0u8; w * cr]; h * rr];
                for br in 0..rr {
                    for bc in 0..cr {
                        for r in 0..h {
                            for c in 0..w {
                                out[br * h + r][bc * w + c] = g.cells[r][c];
                            }
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::ScaleUp { factor } => {
                let k = (*factor).clamp(1, 8);
                if h * k > 64 || w * k > 64 {
                    return g.clone();
                }
                let mut out = vec![vec![0u8; w * k]; h * k];
                for r in 0..h {
                    for c in 0..w {
                        let v = g.cells[r][c];
                        for dr in 0..k {
                            for dc in 0..k {
                                out[r * k + dr][c * k + dc] = v;
                            }
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::CropNonZeroBBox => {
                let (mut r0, mut r1, mut c0, mut c1) = (h, 0usize, w, 0usize);
                let mut found = false;
                for r in 0..h {
                    for c in 0..w {
                        if g.cells[r][c] != 0 {
                            found = true;
                            r0 = r0.min(r);
                            r1 = r1.max(r);
                            c0 = c0.min(c);
                            c1 = c1.max(c);
                        }
                    }
                }
                if !found {
                    return g.clone();
                }
                let mut out = Vec::with_capacity(r1 - r0 + 1);
                for r in r0..=r1 {
                    out.push(g.cells[r][c0..=c1].to_vec());
                }
                ArcGrid::new(out)
            }
            Self::GravityDown => {
                let mut out = vec![vec![0u8; w]; h];
                for c in 0..w {
                    let mut write_r = h;
                    for r in (0..h).rev() {
                        if g.cells[r][c] != 0 {
                            write_r -= 1;
                            out[write_r][c] = g.cells[r][c];
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::GravityRight => {
                let mut out = vec![vec![0u8; w]; h];
                for r in 0..h {
                    let vals: Vec<u8> = g.cells[r].iter().copied().filter(|&v| v != 0).collect();
                    let start = w - vals.len();
                    for (i, v) in vals.into_iter().enumerate() {
                        out[r][start + i] = v;
                    }
                }
                ArcGrid::new(out)
            }
            Self::Shift { dr, dc } => {
                let mut out = vec![vec![0u8; w]; h];
                for r in 0..h {
                    for c in 0..w {
                        let nr = (r as i32) + *dr;
                        let nc = (c as i32) + *dc;
                        if nr >= 0 && (nr as usize) < h && nc >= 0 && (nc as usize) < w {
                            out[nr as usize][nc as usize] = g.cells[r][c];
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::FillEnclosedHoles { fill_color } => {
                let mut exterior = vec![vec![false; w]; h];
                let mut q = VecDeque::new();
                for r in 0..h {
                    for c in [0, w - 1] {
                        if g.cells[r][c] == 0 && !exterior[r][c] {
                            exterior[r][c] = true;
                            q.push_back((r, c));
                        }
                    }
                }
                for c in 0..w {
                    for r in [0, h - 1] {
                        if g.cells[r][c] == 0 && !exterior[r][c] {
                            exterior[r][c] = true;
                            q.push_back((r, c));
                        }
                    }
                }
                while let Some((r, c)) = q.pop_front() {
                    let nbrs = [
                        (r.wrapping_sub(1), c),
                        (r + 1, c),
                        (r, c.wrapping_sub(1)),
                        (r, c + 1),
                    ];
                    for (nr, nc) in nbrs {
                        if nr < h && nc < w && g.cells[nr][nc] == 0 && !exterior[nr][nc] {
                            exterior[nr][nc] = true;
                            q.push_back((nr, nc));
                        }
                    }
                }
                let mut out = g.cells.clone();
                for r in 0..h {
                    for c in 0..w {
                        if out[r][c] == 0 && !exterior[r][c] {
                            out[r][c] = *fill_color;
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::HollowInterior => {
                let mut out = g.cells.clone();
                for r in 1..h.saturating_sub(1) {
                    for c in 1..w.saturating_sub(1) {
                        if g.cells[r][c] != 0
                            && g.cells[r - 1][c] != 0
                            && g.cells[r + 1][c] != 0
                            && g.cells[r][c - 1] != 0
                            && g.cells[r][c + 1] != 0
                        {
                            out[r][c] = 0;
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::CompleteHorizontalSymmetry => {
                let mut out = g.cells.clone();
                for r in 0..h {
                    for c in 0..w {
                        if out[r][c] == 0 && g.cells[r][w - 1 - c] != 0 {
                            out[r][c] = g.cells[r][w - 1 - c];
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::CompleteVerticalSymmetry => {
                let mut out = g.cells.clone();
                for r in 0..h {
                    for c in 0..w {
                        if out[r][c] == 0 && g.cells[h - 1 - r][c] != 0 {
                            out[r][c] = g.cells[h - 1 - r][c];
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::KeepLargestObject => {
                let comps = extract_4connected_components(g);
                let Some(best) = comps.into_iter().max_by_key(|c| c.pixels.len()) else {
                    return g.clone();
                };
                let mut out = vec![vec![0u8; w]; h];
                for (r, c, col) in best.pixels {
                    out[r][c] = col;
                }
                ArcGrid::new(out)
            }
            Self::CropLargestObject => {
                let comps = extract_4connected_components(g);
                let Some(best) = comps.into_iter().max_by_key(|c| c.pixels.len()) else {
                    return g.clone();
                };
                let mut out = vec![vec![0u8; best.c1 - best.c0 + 1]; best.r1 - best.r0 + 1];
                for (r, c, col) in best.pixels {
                    out[r - best.r0][c - best.c0] = col;
                }
                ArcGrid::new(out)
            }
            Self::KeepUniqueColorObject => {
                let Some(rc) = rarest_nonzero_color(g) else {
                    return g.clone();
                };
                let out = g
                    .cells
                    .iter()
                    .map(|row| row.iter().map(|&v| if v == rc { v } else { 0 }).collect())
                    .collect();
                ArcGrid::new(out)
            }
            Self::CropUniqueColorObject => {
                let Some(rc) = rarest_nonzero_color(g) else {
                    return g.clone();
                };
                let (mut r0, mut r1, mut c0, mut c1) = (h, 0usize, w, 0usize);
                let mut found = false;
                for r in 0..h {
                    for c in 0..w {
                        if g.cells[r][c] == rc {
                            found = true;
                            r0 = r0.min(r);
                            r1 = r1.max(r);
                            c0 = c0.min(c);
                            c1 = c1.max(c);
                        }
                    }
                }
                if !found {
                    return g.clone();
                }
                let mut out = Vec::with_capacity(r1 - r0 + 1);
                for r in r0..=r1 {
                    out.push(g.cells[r][c0..=c1].to_vec());
                }
                ArcGrid::new(out)
            }
            Self::ConnectSameColorOrthogonal => {
                let mut out = g.cells.clone();
                // Horizontal bridges across `0` cells between matching non-zero endpoints
                for r in 0..h {
                    let mut prev: Option<(usize, u8)> = None;
                    for c in 0..w {
                        let v = g.cells[r][c];
                        if v != 0 {
                            if let Some((pc, pv)) = prev {
                                if pv == v
                                    && c > pc + 1
                                    && (pc + 1..c).all(|mc| g.cells[r][mc] == 0)
                                {
                                    for mc in pc + 1..c {
                                        out[r][mc] = v;
                                    }
                                }
                            }
                            prev = Some((c, v));
                        }
                    }
                }
                // Vertical bridges across `0` cells between matching non-zero endpoints
                for c in 0..w {
                    let mut prev: Option<(usize, u8)> = None;
                    for r in 0..h {
                        let v = g.cells[r][c];
                        if v != 0 {
                            if let Some((pr, pv)) = prev {
                                if pv == v
                                    && r > pr + 1
                                    && (pr + 1..r).all(|mr| g.cells[mr][c] == 0)
                                {
                                    for mr in pr + 1..r {
                                        out[mr][c] = v;
                                    }
                                }
                            }
                            prev = Some((r, v));
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::SplitPanelBool { op, output_color } => {
                let Some((a, b)) = split_two_panels(g) else {
                    return g.clone();
                };
                if a.height() != b.height() || a.width() != b.width() {
                    return g.clone();
                }
                let (ph, pw) = (a.height(), a.width());
                let mut out = vec![vec![0u8; pw]; ph];
                for r in 0..ph {
                    for c in 0..pw {
                        let av = a.cells[r][c] != 0;
                        let bv = b.cells[r][c] != 0;
                        let bit = match op {
                            PanelLogicKind::Xor => av ^ bv,
                            PanelLogicKind::And => av && bv,
                            PanelLogicKind::Or => av || bv,
                            PanelLogicKind::Nor => !av && !bv,
                        };
                        if bit {
                            out[r][c] = *output_color;
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::DownsampleBlockMajority {
                row_factor,
                col_factor,
            } => {
                let rf = (*row_factor).max(1);
                let cf = (*col_factor).max(1);
                if h < rf || w < cf || h % rf != 0 || w % cf != 0 {
                    return g.clone();
                }
                let (oh, ow) = (h / rf, w / cf);
                let mut out = vec![vec![0u8; ow]; oh];
                for br in 0..oh {
                    for bc in 0..ow {
                        let mut counts = [0usize; 256];
                        for dr in 0..rf {
                            for dc in 0..cf {
                                let v = g.cells[br * rf + dr][bc * cf + dc];
                                if v != 0 {
                                    counts[v as usize] += 1;
                                }
                            }
                        }
                        if let Some(best_c) = (1u8..=9)
                            .filter(|&c| counts[c as usize] > 0)
                            .max_by_key(|&c| counts[c as usize])
                        {
                            out[br][bc] = best_c;
                        }
                    }
                }
                ArcGrid::new(out)
            }
            Self::InvertBinary { fg_color } => {
                let out = g
                    .cells
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&v| if v == 0 { *fg_color } else { 0 })
                            .collect()
                    })
                    .collect();
                ArcGrid::new(out)
            }
            Self::RecolorMap(mapping) => {
                let mut lut = [0u8; 256];
                for (i, slot) in lut.iter_mut().enumerate() {
                    *slot = i as u8;
                }
                for &(src, dst) in mapping {
                    lut[src as usize] = dst;
                }
                let out = g
                    .cells
                    .iter()
                    .map(|row| row.iter().map(|&v| lut[v as usize]).collect())
                    .collect();
                ArcGrid::new(out)
            }
            Self::Compose(first, second) => second.apply(&first.apply(g)),
        }
    }
}

/// Verified `ARC-AGI` program synthesized from demonstration `(input, output)` pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedArcProgram {
    /// Winning lowest-`MDL` operator that solved 100% of demonstration pairs.
    pub program: ArcOp,
    /// `MDL` complexity score.
    pub mdl_cost: u32,
    /// Number of candidate programs evaluated during synthesis.
    pub candidates_evaluated: usize,
}

fn infer_recolor_map(train: &[(ArcGrid, ArcGrid)]) -> Option<Vec<(u8, u8)>> {
    let mut map: BTreeMap<u8, u8> = BTreeMap::new();
    for (inp, out) in train {
        if inp.height() != out.height() || inp.width() != out.width() {
            return None;
        }
        for r in 0..inp.height() {
            for c in 0..inp.width() {
                let sc = inp.cells[r][c];
                let dc = out.cells[r][c];
                if let Some(&prev) = map.get(&sc) {
                    if prev != dc {
                        return None;
                    }
                } else {
                    map.insert(sc, dc);
                }
            }
        }
    }
    let non_trivial: Vec<(u8, u8)> = map.into_iter().filter(|(a, b)| a != b).collect();
    if non_trivial.is_empty() {
        None
    } else {
        Some(non_trivial)
    }
}

/// Synthesizes an exact `MDL`-minimal `ArcOp` verified across all `train` `(input, output)` pairs.
pub fn synthesize_arc_program(
    train: &[(ArcGrid, ArcGrid)],
    allow_depth_2: bool,
) -> Option<SynthesizedArcProgram> {
    if train.is_empty() {
        return None;
    }

    let mut base_ops = vec![
        ArcOp::Identity,
        ArcOp::Rot90,
        ArcOp::Rot180,
        ArcOp::Rot270,
        ArcOp::FlipH,
        ArcOp::FlipV,
        ArcOp::Transpose,
        ArcOp::CropNonZeroBBox,
        ArcOp::GravityDown,
        ArcOp::GravityRight,
        ArcOp::HollowInterior,
        ArcOp::CompleteHorizontalSymmetry,
        ArcOp::CompleteVerticalSymmetry,
        ArcOp::KroneckerSelfTile,
        ArcOp::KeepLargestObject,
        ArcOp::CropLargestObject,
        ArcOp::KeepUniqueColorObject,
        ArcOp::CropUniqueColorObject,
        ArcOp::ConnectSameColorOrthogonal,
    ];

    for fc in 1u8..=9 {
        base_ops.push(ArcOp::FillEnclosedHoles { fill_color: fc });
    }
    for dr in -2..=2 {
        for dc in -2..=2 {
            if dr != 0 || dc != 0 {
                base_ops.push(ArcOp::Shift { dr, dc });
            }
        }
    }

    // Infer exact ratio & panel candidates from the first demonstration pair
    let (in0, out0) = &train[0];
    if let Some(dom_c) = dominant_nonzero_color(out0) {
        base_ops.push(ArcOp::InvertBinary { fg_color: dom_c });
        for kind in [
            PanelLogicKind::Xor,
            PanelLogicKind::And,
            PanelLogicKind::Or,
            PanelLogicKind::Nor,
        ] {
            base_ops.push(ArcOp::SplitPanelBool {
                op: kind,
                output_color: dom_c,
            });
        }
    }
    if in0.height() > 0 && in0.width() > 0 && out0.height() > 0 && out0.width() > 0 {
        if out0.height() % in0.height() == 0 && out0.width() % in0.width() == 0 {
            let rr = out0.height() / in0.height();
            let cr = out0.width() / in0.width();
            if rr >= 1 && cr >= 1 && (rr > 1 || cr > 1) && rr <= 6 && cr <= 6 {
                base_ops.push(ArcOp::TileRepeat {
                    rows_rep: rr,
                    cols_rep: cr,
                });
                base_ops.push(ArcOp::TileAlternatingFlipH {
                    rows_rep: rr,
                    cols_rep: cr,
                });
                if rr == cr {
                    base_ops.push(ArcOp::ScaleUp { factor: rr });
                }
            }
        }
        if in0.height() % out0.height() == 0 && in0.width() % out0.width() == 0 {
            let rf = in0.height() / out0.height();
            let cf = in0.width() / out0.width();
            if rf >= 1 && cf >= 1 && (rf > 1 || cf > 1) && rf <= 8 && cf <= 8 {
                base_ops.push(ArcOp::DownsampleBlockMajority {
                    row_factor: rf,
                    col_factor: cf,
                });
            }
        }
    }

    if let Some(cmap) = infer_recolor_map(train) {
        base_ops.push(ArcOp::RecolorMap(cmap));
    }

    base_ops.sort_by_key(|op| op.mdl_cost());

    let mut evaluated = 0usize;
    for op in &base_ops {
        evaluated += 1;
        if train.iter().all(|(x, y)| &op.apply(x) == y) {
            return Some(SynthesizedArcProgram {
                mdl_cost: op.mdl_cost(),
                program: op.clone(),
                candidates_evaluated: evaluated,
            });
        }
    }

    if allow_depth_2 {
        let mut best: Option<(u32, ArcOp)> = None;
        for a in &base_ops {
            if matches!(a, ArcOp::Identity) {
                continue;
            }
            let inter: Vec<ArcGrid> = train.iter().map(|(x, _)| a.apply(x)).collect();
            // Also check inferred recolor from `inter -> out`!
            let inter_pairs: Vec<(ArcGrid, ArcGrid)> = inter
                .iter()
                .cloned()
                .zip(train.iter().map(|(_, y)| y.clone()))
                .collect();
            if let Some(cmap) = infer_recolor_map(&inter_pairs) {
                evaluated += 1;
                let comp = ArcOp::Compose(Box::new(a.clone()), Box::new(ArcOp::RecolorMap(cmap)));
                let cost = comp.mdl_cost();
                if best.as_ref().is_none_or(|(bc, _)| cost < *bc) {
                    best = Some((cost, comp));
                }
            }
            for b in &base_ops {
                if matches!(b, ArcOp::Identity | ArcOp::FillEnclosedHoles { .. }) {
                    continue;
                }
                evaluated += 1;
                if inter
                    .iter()
                    .zip(train.iter())
                    .all(|(mid, (_, y))| &b.apply(mid) == y)
                {
                    let comp = ArcOp::Compose(Box::new(a.clone()), Box::new(b.clone()));
                    let cost = comp.mdl_cost();
                    if best.as_ref().is_none_or(|(bc, _)| cost < *bc) {
                        best = Some((cost, comp));
                    }
                }
            }
        }
        if let Some((cost, prog)) = best {
            return Some(SynthesizedArcProgram {
                program: prog,
                mdl_cost: cost,
                candidates_evaluated: evaluated,
            });
        }
    }

    None
}

/// `ARC-AGI-3` interactive probe selector: picks the probe input `probe_candidates[i]` that
/// maximizes Shannon output entropy `H(Y | probe)` across active `hypotheses`, eliminating
/// the maximum fraction of competing rules in one environment step.
pub fn select_arc3_active_probe(
    hypotheses: &[ArcOp],
    probe_candidates: &[ArcGrid],
) -> Option<(usize, f64)> {
    if hypotheses.is_empty() || probe_candidates.is_empty() {
        return None;
    }
    let total = hypotheses.len() as f64;
    let mut best_idx = 0usize;
    let mut best_entropy = -1.0f64;

    for (idx, probe) in probe_candidates.iter().enumerate() {
        let mut counts: HashMap<ArcGrid, usize> = HashMap::new();
        for h in hypotheses {
            *counts.entry(h.apply(probe)).or_insert(0) += 1;
        }
        let mut ent = 0.0f64;
        for &c in counts.values() {
            let p = (c as f64) / total;
            if p > 0.0 {
                ent -= p * p.log2();
            }
        }
        if ent > best_entropy {
            best_entropy = ent;
            best_idx = idx;
        }
    }

    Some((best_idx, best_entropy))
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. FRONTIERMATH / MATHARENA (AIME) / LIVEBENCH EXACT SOLVER & DPLL VERIFIER
// ═══════════════════════════════════════════════════════════════════════════

/// Extended Euclidean algorithm returning `(gcd(a, b), x, y)` such that `a*x + b*y = gcd(a, b)`.
pub fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
        let sign = if a < 0 { -1 } else { 1 };
        (a.abs(), sign, 0)
    } else {
        let (g, x1, y1) = ext_gcd(b, a.rem_euclid(b));
        let q = a.div_euclid(b);
        (g, y1, x1 - q * y1)
    }
}

/// Solves a generalized system of linear congruences `x ≡ rem_i (mod mod_i)` (including non-coprime moduli!),
/// returning `Some((smallest_nonneg_x, lcm_modulus))` or `None` if inconsistent.
pub fn solve_chinese_remainder(congruences: &[(i128, i128)]) -> Option<(i128, i128)> {
    let mut r0 = 0i128;
    let mut m0 = 1i128;
    for &(r_raw, m_raw) in congruences {
        if m_raw <= 0 {
            return None;
        }
        let r1 = r_raw.rem_euclid(m_raw);
        let (g, p, _q) = ext_gcd(m0, m_raw);
        if (r1 - r0).rem_euclid(g) != 0 {
            return None;
        }
        let m1_div_g = m_raw / g;
        let lcm = m0.checked_mul(m1_div_g)?;
        let step = ((r1 - r0) / g).rem_euclid(m1_div_g);
        let k = (step.checked_mul(p.rem_euclid(m1_div_g))?).rem_euclid(m1_div_g);
        r0 = (r0 + m0.checked_mul(k)?).rem_euclid(lcm);
        m0 = lcm;
    }
    Some((r0, m0))
}

/// Fast exact modular exponentiation `base^exp mod modulus` in `O(log exp)` without overflow.
pub fn mod_pow(mut base: i128, mut exp: u64, modulus: i128) -> i128 {
    if modulus <= 1 {
        return 0;
    }
    base = base.rem_euclid(modulus);
    let mut acc = 1i128;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc.rem_euclid(modulus) * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }
    acc
}

/// Exact binomial coefficient `C(n, k)` in `i128`.
pub fn exact_binomial(n: u64, k: u64) -> Option<i128> {
    if k > n {
        return Some(0);
    }
    let kk = k.min(n - k);
    let mut res = 1i128;
    for i in 1..=kk {
        res = res.checked_mul((n - kk + i) as i128)? / (i as i128);
    }
    Some(res)
}

/// Exact `DPLL` Boolean SAT solver (`1-indexed` signed literals: `+v` = true, `-v` = false)
/// for `LiveBench-Reasoning` truth-teller/liar (`Web-of-Lies`), Zebra, and logic verification.
pub fn solve_sat_dpll(num_vars: usize, clauses: &[Vec<i32>]) -> Option<Vec<bool>> {
    fn dpll_rec(num_vars: usize, clauses: &[Vec<i32>], assign: &mut [Option<bool>]) -> bool {
        loop {
            let mut unit_lit: Option<i32> = None;
            let mut all_sat = true;

            for cl in clauses {
                let mut sat = false;
                let mut unassigned_count = 0;
                let mut last_unassigned = 0i32;
                for &lit in cl {
                    let var = lit.unsigned_abs() as usize;
                    if var == 0 || var > num_vars {
                        continue;
                    }
                    match assign[var - 1] {
                        Some(val) => {
                            if (lit > 0 && val) || (lit < 0 && !val) {
                                sat = true;
                                break;
                            }
                        }
                        None => {
                            unassigned_count += 1;
                            last_unassigned = lit;
                        }
                    }
                }
                if !sat {
                    all_sat = false;
                    if unassigned_count == 0 {
                        return false;
                    }
                    if unassigned_count == 1 {
                        unit_lit = Some(last_unassigned);
                        break;
                    }
                }
            }

            if all_sat {
                return true;
            }
            if let Some(u) = unit_lit {
                assign[(u.unsigned_abs() as usize) - 1] = Some(u > 0);
            } else {
                break;
            }
        }

        let Some(next_var) = assign.iter().position(|x| x.is_none()) else {
            return true;
        };

        for choice in [true, false] {
            let mut branch = assign.to_vec();
            branch[next_var] = Some(choice);
            if dpll_rec(num_vars, clauses, &mut branch) {
                assign.copy_from_slice(&branch);
                return true;
            }
        }
        false
    }

    let mut assign = vec![None; num_vars];
    if dpll_rec(num_vars, clauses, &mut assign) {
        Some(assign.into_iter().map(|v| v.unwrap_or(false)).collect())
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. TAU-BENCH / τ²-BENCH TRANSACTIONAL POLICY GUARD (`pass^k`)
// ═══════════════════════════════════════════════════════════════════════════

/// Declarative domain policy constraints enforced atomically before any `MCP` / `TAU-bench` tool mutates state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauDomainPolicy {
    /// Requires verified customer identity before state-mutating calls.
    pub require_authenticated_user: bool,
    /// Requires explicit human-in-the-loop confirmation token on destructive/financial mutations.
    pub require_user_confirmation_for_mutations: bool,
    /// Maximum allowed refund or transfer amount in cents without supervisor escalation.
    pub max_autonomous_amount_cents: i64,
    /// Ticket/order tiers prohibited from autonomous refund outside the 24-hour grace window.
    pub non_refundable_tiers_after_24h: Vec<String>,
}

/// Candidate `TAU-bench` tool invocation inspected by JouleDB's transactional guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauToolProposal {
    /// Tool name (e.g. `"refund_reservation"`, `"cancel_order"`, `"lookup_order"`).
    pub tool_name: String,
    /// Whether this tool mutates persistent customer/order state.
    pub is_mutating: bool,
    /// Whether customer authentication succeeded in this session.
    pub user_authenticated: bool,
    /// Whether explicit user `"yes/confirm"` turn occurred after state summary.
    pub explicit_user_confirmation: bool,
    /// Transaction amount in cents (`0` for non-financial ops).
    pub amount_cents: i64,
    /// Product/fare tier (e.g. `"basic_economy"`, `"flexible"`).
    pub tier: String,
    /// Hours elapsed since booking/order creation.
    pub hours_since_booking: f64,
}

/// Verdict returned by JouleDB's transactional `TAU-bench` policy guard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TauPolicyVerdict {
    /// Permitted — all domain policy invariants satisfied.
    Permitted,
    /// Blocked before DB commit with explicit invariant violation reason (`0` unsafe state corruption).
    Blocked {
        /// Violated policy clause identifier.
        rule: String,
    },
}

/// Evaluates a `TauToolProposal` against `TauDomainPolicy` deterministically.
pub fn check_tau_policy(policy: &TauDomainPolicy, call: &TauToolProposal) -> TauPolicyVerdict {
    if !call.is_mutating {
        return TauPolicyVerdict::Permitted;
    }
    if policy.require_authenticated_user && !call.user_authenticated {
        return TauPolicyVerdict::Blocked {
            rule: "unauthenticated_mutation".to_string(),
        };
    }
    if policy.require_user_confirmation_for_mutations && !call.explicit_user_confirmation {
        return TauPolicyVerdict::Blocked {
            rule: "missing_explicit_user_confirmation".to_string(),
        };
    }
    if call.amount_cents > policy.max_autonomous_amount_cents {
        return TauPolicyVerdict::Blocked {
            rule: "exceeds_autonomous_amount_cap".to_string(),
        };
    }
    if call.hours_since_booking > 24.0
        && policy
            .non_refundable_tiers_after_24h
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&call.tier))
    {
        return TauPolicyVerdict::Blocked {
            rule: "non_refundable_tier_past_24h_grace".to_string(),
        };
    }
    TauPolicyVerdict::Permitted
}

/// Computes Sierra `TAU-bench` multi-trial `pass^k` (`(c / n)` unbiased estimator `C(c, k) / C(n, k)`).
pub fn compute_pass_hat_k(num_trials: u64, num_successes: u64, k: u64) -> f64 {
    if k == 0 || num_trials == 0 || k > num_trials || num_successes < k {
        return 0.0;
    }
    match (
        exact_binomial(num_successes, k),
        exact_binomial(num_trials, k),
    ) {
        (Some(num), Some(den)) if den > 0 => (num as f64) / (den as f64),
        _ => ((num_successes as f64) / (num_trials as f64)).powi(k as i32),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. MRCR / LOCOMO / LONGMEMEVAL MULTI-NEEDLE BITEMPORAL TRACKER
// ═══════════════════════════════════════════════════════════════════════════

/// Bitemporal multi-session entity-attribute memory store for `1M+` token `MRCR` & `LongMemEval` tracking.
#[derive(Debug, Default, Clone)]
pub struct MrcrBitemporalStore {
    /// Keyed by `(canonical_entity, canonical_attribute) -> sorted Vec<(lsn, session_id, value)>`.
    timeline: HashMap<(String, String), Vec<(u64, String, String)>>,
}

impl MrcrBitemporalStore {
    /// Creates an empty bitemporal memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a state update `(entity, attribute := value)` at `lsn` in `session_id`.
    pub fn record_update(
        &mut self,
        lsn: u64,
        session_id: &str,
        entity: &str,
        attribute: &str,
        value: &str,
    ) {
        let key = (
            entity.trim().to_ascii_lowercase(),
            attribute.trim().to_ascii_lowercase(),
        );
        let vec = self.timeline.entry(key).or_default();
        vec.push((lsn, session_id.to_string(), value.to_string()));
        vec.sort_by_key(|x| x.0);
    }

    /// Queries the active value `AS OF` `lsn` (`None` = latest state) in `O(log K)` binary search.
    pub fn query_as_of(
        &self,
        entity: &str,
        attribute: &str,
        as_of_lsn: Option<u64>,
    ) -> Option<(u64, String)> {
        let key = (
            entity.trim().to_ascii_lowercase(),
            attribute.trim().to_ascii_lowercase(),
        );
        let entries = self.timeline.get(&key)?;
        match as_of_lsn {
            None => entries.last().map(|(l, _, v)| (*l, v.clone())),
            Some(cap) => {
                let idx = entries.partition_point(|(l, _, _)| *l <= cap);
                if idx == 0 {
                    None
                } else {
                    let (l, _, v) = &entries[idx - 1];
                    Some((*l, v.clone()))
                }
            }
        }
    }

    /// Queries the exact `1-based` `k`-th historical occurrence (`OpenAI MRCR` multi-needle ordinal recall!).
    pub fn query_kth_occurrence(
        &self,
        entity: &str,
        attribute: &str,
        ordinal_1based: usize,
    ) -> Option<(u64, String)> {
        if ordinal_1based == 0 {
            return None;
        }
        let key = (
            entity.trim().to_ascii_lowercase(),
            attribute.trim().to_ascii_lowercase(),
        );
        let entries = self.timeline.get(&key)?;
        entries
            .get(ordinal_1based - 1)
            .map(|(l, _, v)| (*l, v.clone()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. SAFEVLA-BENCH / VSI-BENCH SIGNAL TEMPORAL LOGIC (STL) SAFETY SHIELD
// ═══════════════════════════════════════════════════════════════════════════

/// Physical safety envelope for `SafeVLA-Bench` / `VSI-Bench` embodied execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeVlaEnvelope {
    /// Maximum allowed end-effector force in Newtons.
    pub max_force_newtons: f64,
    /// Maximum per-step joint/cartesian displacement in meters.
    pub max_step_delta_m: f64,
    /// Spherical obstacle keep-out zones `(center_xyz, min_clearance_radius_m)`.
    pub keep_out_spheres: Vec<([f64; 3], f64)>,
}

/// Proposed embodied VLA step `(next_xyz, predicted_force_n)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VlaStepSample {
    /// Proposed 3D end-effector position `[x, y, z]` in meters.
    pub pos_xyz: [f64; 3],
    /// Contact force magnitude in Newtons.
    pub force_newtons: f64,
}

/// Quantitative `STL` robustness & `Violation Severity Index (VSI)` audit report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeVlaAudit {
    /// Quantitative `STL` robustness `ρ(φ, τ)` (`>= 0.0` iff all safety invariants hold everywhere).
    pub stl_robustness: f64,
    /// `SafeVLA-Bench` Violation Severity Index (`max(0, -stl_robustness)`, `0.0` = zero violations).
    pub violation_severity_index: f64,
    /// Number of unsafe raw VLA proposals intercepted and projected onto the safe manifold.
    pub interventions: usize,
}

/// Audits a raw VLA trajectory and projects every step onto `SafeVlaEnvelope`, returning the
/// safe trajectory alongside both raw and shielded `SafeVlaAudit` metrics.
pub fn shield_and_audit_vla_trajectory(
    start_xyz: [f64; 3],
    raw_steps: &[VlaStepSample],
    env: &SafeVlaEnvelope,
) -> (Vec<VlaStepSample>, SafeVlaAudit, SafeVlaAudit) {
    let eval_step_robustness = |prev: [f64; 3], cur: VlaStepSample| -> f64 {
        let mut rho = env.max_force_newtons - cur.force_newtons;
        let dx = cur.pos_xyz[0] - prev[0];
        let dy = cur.pos_xyz[1] - prev[1];
        let dz = cur.pos_xyz[2] - prev[2];
        let step_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        rho = rho.min(env.max_step_delta_m - step_dist);
        for &(center, radius) in &env.keep_out_spheres {
            let ox = cur.pos_xyz[0] - center[0];
            let oy = cur.pos_xyz[1] - center[1];
            let oz = cur.pos_xyz[2] - center[2];
            let dist = (ox * ox + oy * oy + oz * oz).sqrt();
            rho = rho.min(dist - radius);
        }
        rho
    };

    let mut raw_min_rho = f64::INFINITY;
    let mut prev_raw = start_xyz;
    for &st in raw_steps {
        raw_min_rho = raw_min_rho.min(eval_step_robustness(prev_raw, st));
        prev_raw = st.pos_xyz;
    }
    if raw_steps.is_empty() {
        raw_min_rho = 0.0;
    }

    let mut safe_steps = Vec::with_capacity(raw_steps.len());
    let mut prev_safe = start_xyz;
    let mut safe_min_rho = f64::INFINITY;
    let mut interventions = 0usize;

    for &st in raw_steps {
        let mut p = st.pos_xyz;
        let mut intervened = false;

        // 1. Clamp step velocity delta
        let dx = p[0] - prev_safe[0];
        let dy = p[1] - prev_safe[1];
        let dz = p[2] - prev_safe[2];
        let step_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if step_dist > env.max_step_delta_m && step_dist > 1e-12 {
            let scale = (env.max_step_delta_m - 1e-6).max(0.0) / step_dist;
            p = [
                prev_safe[0] + dx * scale,
                prev_safe[1] + dy * scale,
                prev_safe[2] + dz * scale,
            ];
            intervened = true;
        }

        // 2. Project outside every keep-out sphere
        for &(center, radius) in &env.keep_out_spheres {
            let ox = p[0] - center[0];
            let oy = p[1] - center[1];
            let oz = p[2] - center[2];
            let dist = (ox * ox + oy * oy + oz * oz).sqrt();
            if dist < radius {
                intervened = true;
                let safe_r = radius + 1e-5;
                let next_cand = if dist > 1e-9 {
                    [
                        center[0] + ox * (safe_r / dist),
                        center[1] + oy * (safe_r / dist),
                        center[2] + oz * (safe_r / dist),
                    ]
                } else {
                    [center[0] + safe_r, center[1], center[2]]
                };
                // Re-verify step delta from `prev_safe`
                let rdx = next_cand[0] - prev_safe[0];
                let rdy = next_cand[1] - prev_safe[1];
                let rdz = next_cand[2] - prev_safe[2];
                let rdist = (rdx * rdx + rdy * rdy + rdz * rdz).sqrt();
                p = if rdist <= env.max_step_delta_m {
                    next_cand
                } else {
                    prev_safe
                };
            }
        }

        // 3. Clamp contact force ceiling
        let mut f = st.force_newtons;
        if f > env.max_force_newtons {
            f = env.max_force_newtons;
            intervened = true;
        }

        if intervened {
            interventions += 1;
        }
        let projected = VlaStepSample {
            pos_xyz: p,
            force_newtons: f,
        };
        safe_min_rho = safe_min_rho.min(eval_step_robustness(prev_safe, projected));
        prev_safe = p;
        safe_steps.push(projected);
    }

    if safe_steps.is_empty() {
        safe_min_rho = 0.0;
    }

    let raw_audit = SafeVlaAudit {
        stl_robustness: raw_min_rho,
        violation_severity_index: (-raw_min_rho).max(0.0),
        interventions: 0,
    };
    let safe_audit = SafeVlaAudit {
        stl_robustness: safe_min_rho,
        violation_severity_index: (-safe_min_rho).max(0.0),
        interventions,
    };

    (safe_steps, raw_audit, safe_audit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hle_calibration_and_platt_scaling_reduces_rms_ce() {
        // Simulate an overconfident closed-book model (states ~92% confidence when only 30% accurate)
        let mut samples = Vec::new();
        for i in 0..100 {
            let correct = i % 10 < 3;
            samples.push(CalibrationSample {
                correct,
                confidence: 0.90 + 0.0008 * (i as f64),
                is_reasoning_split: i < 50,
            });
        }
        let raw = compute_hle_calibration(&samples, 20);
        assert!(raw.rms_calibration_error > 0.55);

        let cal = EpistemicCalibrator::fit(&samples);
        let calibrated_samples: Vec<CalibrationSample> = samples
            .iter()
            .map(|s| CalibrationSample {
                confidence: cal.calibrate(s.confidence),
                ..*s
            })
            .collect();
        let post = compute_hle_calibration(&calibrated_samples, 20);
        assert!(post.rms_calibration_error < 0.05);

        let sqa_raw = evaluate_simpleqa_abstention(&samples, None, 2.0, &[]);
        let sqa_cal = evaluate_simpleqa_abstention(&samples, Some(&cal), 2.0, &[]);
        assert!(sqa_cal.penalty_adjusted_score > sqa_raw.penalty_adjusted_score);
    }

    #[test]
    fn test_faitheval_counterfactual_inconsistent_and_unanswerable() {
        let facts = vec![
            ContextFact {
                source_id: "doc1".into(),
                entity: "Moon".into(),
                attribute: "composition".into(),
                value: "marshmallows".into(),
                valid_from_lsn: 10,
            },
            ContextFact {
                source_id: "doc_a".into(),
                entity: "ProjectX".into(),
                attribute: "deadline".into(),
                value: "Friday".into(),
                valid_from_lsn: 12,
            },
            ContextFact {
                source_id: "doc_b".into(),
                entity: "ProjectX".into(),
                attribute: "deadline".into(),
                value: "Monday".into(),
                valid_from_lsn: 12,
            },
        ];

        assert_eq!(
            resolve_faithful_context(&facts, Some("silicate rock"), "moon", "composition", None),
            FaithEvalOutcome::GroundedValue {
                value: "marshmallows".into(),
                lsn: 10
            }
        );
        assert_eq!(
            resolve_faithful_context(&facts, Some("Tuesday"), "projectx", "deadline", None),
            FaithEvalOutcome::InconsistentConflict {
                conflicting_values: vec!["Friday".into(), "Monday".into()]
            }
        );
        assert_eq!(
            resolve_faithful_context(&facts, Some("Paris"), "France", "capital", None),
            FaithEvalOutcome::UnanswerableAbstain
        );
    }

    #[test]
    fn test_arc_agi_2_program_synthesis_real_tasks() {
        // Real ARC-AGI-2 task `00576224`: 2x2 -> 6x6 alternating horizontal-flip tile
        let inp = ArcGrid::new(vec![vec![3, 2], vec![7, 8]]);
        let out = ArcGrid::new(vec![
            vec![3, 2, 3, 2, 3, 2],
            vec![7, 8, 7, 8, 7, 8],
            vec![2, 3, 2, 3, 2, 3],
            vec![8, 7, 8, 7, 8, 7],
            vec![3, 2, 3, 2, 3, 2],
            vec![7, 8, 7, 8, 7, 8],
        ]);
        let syn = synthesize_arc_program(&[(inp.clone(), out.clone())], false).unwrap();
        assert_eq!(syn.program.apply(&inp), out);

        // Real ARC task `007bbfb7`: 3x3 Kronecker self-tiling -> 9x9
        let k_in = ArcGrid::new(vec![vec![0, 7, 7], vec![7, 7, 7], vec![0, 7, 7]]);
        let k_out = ArcOp::KroneckerSelfTile.apply(&k_in);
        let syn2 = synthesize_arc_program(&[(k_in.clone(), k_out.clone())], false).unwrap();
        assert_eq!(syn2.program.apply(&k_in), k_out);
    }

    #[test]
    fn test_exact_math_crt_and_dpll_sat() {
        let (x, m) = solve_chinese_remainder(&[(2, 3), (3, 5), (2, 7)]).unwrap();
        assert_eq!((x, m), (23, 105));

        // Knight/Knave puzzle: A says "B is a liar" (A <-> !B), B says "A and B are both liars" (B <-> (!A & !B))
        // Clauses for (A <-> !B): (A v B) & (!A v !B)
        // Clauses for (B -> !B) => (!B)
        let clauses = vec![vec![1, 2], vec![-1, -2], vec![-2]];
        let sol = solve_sat_dpll(2, &clauses).unwrap();
        assert_eq!(sol, vec![true, false]);
    }

    #[test]
    fn test_tau_policy_mrcr_and_safevla_shield() {
        let policy = TauDomainPolicy {
            require_authenticated_user: true,
            require_user_confirmation_for_mutations: true,
            max_autonomous_amount_cents: 50_000,
            non_refundable_tiers_after_24h: vec!["basic_economy".into()],
        };
        let bad_call = TauToolProposal {
            tool_name: "refund_reservation".into(),
            is_mutating: true,
            user_authenticated: true,
            explicit_user_confirmation: true,
            amount_cents: 18_000,
            tier: "basic_economy".into(),
            hours_since_booking: 36.0,
        };
        assert_eq!(
            check_tau_policy(&policy, &bad_call),
            TauPolicyVerdict::Blocked {
                rule: "non_refundable_tier_past_24h_grace".into()
            }
        );
        assert!((compute_pass_hat_k(8, 8, 8) - 1.0).abs() < 1e-12);

        let mut mem = MrcrBitemporalStore::new();
        for i in 1..=8u64 {
            mem.record_update(
                i * 10,
                &format!("s{i}"),
                "Alice",
                "city",
                &format!("City_{i}"),
            );
        }
        assert_eq!(mem.query_as_of("alice", "city", None).unwrap().1, "City_8");
        assert_eq!(
            mem.query_as_of("alice", "city", Some(35)).unwrap().1,
            "City_3"
        );
        assert_eq!(
            mem.query_kth_occurrence("alice", "city", 5).unwrap().1,
            "City_5"
        );

        let env = SafeVlaEnvelope {
            max_force_newtons: 15.0,
            max_step_delta_m: 0.25,
            keep_out_spheres: vec![([0.10, 0.0, 0.0], 0.08)],
        };
        let raw = vec![VlaStepSample {
            pos_xyz: [0.10, 0.0, 0.0], // right inside obstacle!
            force_newtons: 28.0,       // exceeds 15 N!
        }];
        let (_safe, raw_aud, safe_aud) =
            shield_and_audit_vla_trajectory([0.0, 0.0, 0.0], &raw, &env);
        assert!(raw_aud.violation_severity_index > 0.05);
        assert_eq!(safe_aud.violation_severity_index, 0.0);
        assert_eq!(safe_aud.interventions, 1);
    }

    #[test]
    fn test_expanded_arc_panel_xor_ray_and_faithful_span_reader() {
        // 1. Split-panel XOR across vertical divider column `5` (`w = 2*2 + 1 = 5`)
        let p_in = ArcGrid::new(vec![vec![1, 0, 5, 1, 1], vec![0, 1, 5, 0, 1]]);
        let p_out = ArcGrid::new(vec![vec![0, 3], vec![0, 0]]);
        let syn = synthesize_arc_program(&[(p_in.clone(), p_out.clone())], false).unwrap();
        assert_eq!(syn.program.apply(&p_in), p_out);

        // 2. ConnectSameColorOrthogonal across `0` gap
        let r_in = ArcGrid::new(vec![vec![0, 4, 0, 0, 4], vec![0, 0, 0, 0, 0]]);
        let r_out = ArcGrid::new(vec![vec![0, 4, 4, 4, 4], vec![0, 0, 0, 0, 0]]);
        let syn_r = synthesize_arc_program(&[(r_in.clone(), r_out.clone())], false).unwrap();
        assert_eq!(syn_r.program.apply(&r_in), r_out);

        // 3. FaithfulSpanReader: counterfactual override + negation awareness
        let ctx = "Researchers studied atmospheric optics on Planet Zorg. Unlike Earth, the sky on Planet Zorg is not blue; instead, dense methane crystals turn the sky emerald green at noon.";
        let pred = score_faithful_mc_question(
            ctx,
            "What color is the sky on Planet Zorg at noon?",
            &[
                "blue".to_string(),
                "emerald green".to_string(),
                "crimson red".to_string(),
                "violet".to_string(),
            ],
            None,
            false,
        );
        assert_eq!(pred.predicted_index, Some(1));

        // 4. FaithfulSpanReader: inconsistent multi-document detection & unanswerable detection
        let inc_ctx = "Document 1: Project Apollo launched on Friday. Document 2: Project Apollo launched on Tuesday.";
        let inc_audit = audit_passage_faithfulness(
            inc_ctx,
            "On which day did Project Apollo launch?",
            &["Friday".into(), "Tuesday".into(), "Sunday".into()],
        );
        assert!(inc_audit.inconsistent_conflict);

        let unans_audit = audit_passage_faithfulness(
            "Bananas are rich in potassium and grow in tropical climates.",
            "What was the GDP growth rate of Switzerland in 2024?",
            &["1.2%".into(), "2.5%".into()],
        );
        assert!(unans_audit.unanswerable_abstain);
    }
}
