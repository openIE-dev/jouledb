//! Performance audit model (Lighthouse-style): named audits with scoring
//! thresholds, weighted category scores, overall score, and recommendations.

// ── Audit Result ─────────────────────────────────────────────────

/// A single audit result.
#[derive(Debug, Clone)]
pub struct AuditResult {
    pub id: String,
    pub title: String,
    /// Score between 0.0 and 1.0.
    pub score: f64,
    pub display_value: String,
}

impl AuditResult {
    pub fn new(id: &str, title: &str, score: f64, display_value: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            score: score.clamp(0.0, 1.0),
            display_value: display_value.to_string(),
        }
    }

    /// Color rating: green (>=0.9), orange (>=0.5), red (<0.5).
    pub fn rating(&self) -> Rating {
        if self.score >= 0.9 {
            Rating::Green
        } else if self.score >= 0.5 {
            Rating::Orange
        } else {
            Rating::Red
        }
    }
}

/// Traffic-light rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rating {
    Green,
    Orange,
    Red,
}

// ── Audit Category ───────────────────────────────────────────────

/// A category of audits with a weight for scoring.
#[derive(Debug, Clone)]
pub struct AuditCategory {
    pub name: String,
    pub weight: f64,
    pub audit_ids: Vec<String>,
}

impl AuditCategory {
    pub fn new(name: &str, weight: f64, audit_ids: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            weight,
            audit_ids: audit_ids.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Compute the weighted score for this category given audit results.
    pub fn score(&self, audits: &[AuditResult]) -> f64 {
        let relevant: Vec<f64> = self
            .audit_ids
            .iter()
            .filter_map(|id| audits.iter().find(|a| a.id == *id).map(|a| a.score))
            .collect();

        if relevant.is_empty() {
            return 0.0;
        }

        relevant.iter().sum::<f64>() / relevant.len() as f64
    }
}

// ── Standard Metrics ─────────────────────────────────────────────

/// Compute a score for First Contentful Paint (FCP) in milliseconds.
/// Good: <1800ms (1.0), Needs improvement: <3000ms (0.5-0.9), Poor: >=3000ms
pub fn score_fcp(ms: f64) -> AuditResult {
    let score = if ms <= 1800.0 {
        1.0
    } else if ms <= 3000.0 {
        1.0 - (ms - 1800.0) / (3000.0 - 1800.0) * 0.5
    } else if ms <= 6000.0 {
        0.5 - (ms - 3000.0) / (6000.0 - 3000.0) * 0.5
    } else {
        0.0
    };
    AuditResult::new("fcp", "First Contentful Paint", score, &format!("{:.1} s", ms / 1000.0))
}

/// Compute a score for Largest Contentful Paint (LCP) in milliseconds.
/// Good: <2500ms, Needs improvement: <4000ms, Poor: >=4000ms
pub fn score_lcp(ms: f64) -> AuditResult {
    let score = if ms <= 2500.0 {
        1.0
    } else if ms <= 4000.0 {
        1.0 - (ms - 2500.0) / (4000.0 - 2500.0) * 0.5
    } else if ms <= 8000.0 {
        0.5 - (ms - 4000.0) / (8000.0 - 4000.0) * 0.5
    } else {
        0.0
    };
    AuditResult::new("lcp", "Largest Contentful Paint", score, &format!("{:.1} s", ms / 1000.0))
}

/// Compute a score for Time to Interactive (TTI) in milliseconds.
/// Good: <3800ms, Needs improvement: <7300ms, Poor: >=7300ms
pub fn score_tti(ms: f64) -> AuditResult {
    let score = if ms <= 3800.0 {
        1.0
    } else if ms <= 7300.0 {
        1.0 - (ms - 3800.0) / (7300.0 - 3800.0) * 0.5
    } else if ms <= 15000.0 {
        0.5 - (ms - 7300.0) / (15000.0 - 7300.0) * 0.5
    } else {
        0.0
    };
    AuditResult::new("tti", "Time to Interactive", score, &format!("{:.1} s", ms / 1000.0))
}

/// Compute a score for Cumulative Layout Shift (CLS).
/// Good: <0.1, Needs improvement: <0.25, Poor: >=0.25
pub fn score_cls(value: f64) -> AuditResult {
    let score = if value <= 0.1 {
        1.0
    } else if value <= 0.25 {
        1.0 - (value - 0.1) / (0.25 - 0.1) * 0.5
    } else if value <= 1.0 {
        0.5 - (value - 0.25) / (1.0 - 0.25) * 0.5
    } else {
        0.0
    };
    AuditResult::new("cls", "Cumulative Layout Shift", score, &format!("{:.3}", value))
}

/// Compute a score for Total Blocking Time (TBT) in milliseconds.
/// Good: <200ms, Needs improvement: <600ms, Poor: >=600ms
pub fn score_tbt(ms: f64) -> AuditResult {
    let score = if ms <= 200.0 {
        1.0
    } else if ms <= 600.0 {
        1.0 - (ms - 200.0) / (600.0 - 200.0) * 0.5
    } else if ms <= 1500.0 {
        0.5 - (ms - 600.0) / (1500.0 - 600.0) * 0.5
    } else {
        0.0
    };
    AuditResult::new("tbt", "Total Blocking Time", score, &format!("{:.0} ms", ms))
}

// ── Report ───────────────────────────────────────────────────────

/// A performance report containing audits and categories.
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub audits: Vec<AuditResult>,
    pub categories: Vec<AuditCategory>,
}

impl PerformanceReport {
    pub fn new(audits: Vec<AuditResult>, categories: Vec<AuditCategory>) -> Self {
        Self { audits, categories }
    }

    /// Create a standard performance report from raw metrics.
    pub fn from_metrics(fcp_ms: f64, lcp_ms: f64, tti_ms: f64, cls: f64, tbt_ms: f64) -> Self {
        let audits = vec![
            score_fcp(fcp_ms),
            score_lcp(lcp_ms),
            score_tti(tti_ms),
            score_cls(cls),
            score_tbt(tbt_ms),
        ];

        let categories = vec![
            AuditCategory::new("Performance", 1.0, &["fcp", "lcp", "tti", "cls", "tbt"]),
        ];

        Self { audits, categories }
    }

    /// Compute the weighted score for a single category (0.0 - 1.0).
    pub fn category_score(&self, category_name: &str) -> Option<f64> {
        self.categories
            .iter()
            .find(|c| c.name == category_name)
            .map(|c| c.score(&self.audits))
    }

    /// Compute the overall score on a 0-100 scale, weighted by category weights.
    pub fn overall_score(&self) -> f64 {
        let total_weight: f64 = self.categories.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = self
            .categories
            .iter()
            .map(|c| c.score(&self.audits) * c.weight)
            .sum();

        (weighted_sum / total_weight * 100.0).round()
    }

    /// Generate recommendations: suggest improving the lowest-scoring audits.
    pub fn recommendations(&self) -> Vec<String> {
        let mut sorted: Vec<&AuditResult> = self.audits.iter().collect();
        sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        sorted
            .iter()
            .filter(|a| a.score < 0.9)
            .map(|a| {
                let urgency = match a.rating() {
                    Rating::Red => "Critical",
                    Rating::Orange => "Moderate",
                    Rating::Green => "Low",
                };
                format!(
                    "[{}] Improve '{}' (current: {}, score: {:.0}%)",
                    urgency,
                    a.title,
                    a.display_value,
                    a.score * 100.0,
                )
            })
            .collect()
    }

    /// Get an audit by id.
    pub fn audit(&self, id: &str) -> Option<&AuditResult> {
        self.audits.iter().find(|a| a.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fcp_good() {
        let a = score_fcp(1000.0);
        assert_eq!(a.rating(), Rating::Green);
        assert!((a.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fcp_poor() {
        let a = score_fcp(8000.0);
        assert_eq!(a.rating(), Rating::Red);
    }

    #[test]
    fn test_lcp_scoring() {
        let good = score_lcp(2000.0);
        assert_eq!(good.rating(), Rating::Green);

        let moderate = score_lcp(3500.0);
        assert_eq!(moderate.rating(), Rating::Orange);

        let poor = score_lcp(10000.0);
        assert_eq!(poor.rating(), Rating::Red);
    }

    #[test]
    fn test_cls_scoring() {
        let good = score_cls(0.05);
        assert_eq!(good.rating(), Rating::Green);

        let bad = score_cls(0.5);
        assert_eq!(bad.rating(), Rating::Red);
    }

    #[test]
    fn test_tbt_scoring() {
        let good = score_tbt(100.0);
        assert_eq!(good.rating(), Rating::Green);

        let poor = score_tbt(2000.0);
        assert_eq!(poor.rating(), Rating::Red);
    }

    #[test]
    fn test_tti_scoring() {
        let good = score_tti(3000.0);
        assert_eq!(good.rating(), Rating::Green);
    }

    #[test]
    fn test_category_score() {
        let report = PerformanceReport::from_metrics(1000.0, 2000.0, 3000.0, 0.05, 100.0);
        let perf = report.category_score("Performance").unwrap();
        assert!(perf >= 0.9, "all good metrics should give high score, got {}", perf);
    }

    #[test]
    fn test_overall_score_perfect() {
        let report = PerformanceReport::from_metrics(500.0, 1000.0, 2000.0, 0.01, 50.0);
        let score = report.overall_score();
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_overall_score_poor() {
        let report = PerformanceReport::from_metrics(8000.0, 10000.0, 20000.0, 2.0, 3000.0);
        let score = report.overall_score();
        assert!(score < 10.0, "all poor metrics, got {}", score);
    }

    #[test]
    fn test_recommendations() {
        let report = PerformanceReport::from_metrics(5000.0, 2000.0, 3000.0, 0.05, 100.0);
        let recs = report.recommendations();
        assert!(!recs.is_empty());
        // FCP should be in recommendations
        assert!(recs.iter().any(|r| r.contains("First Contentful Paint")));
    }

    #[test]
    fn test_no_recommendations_when_perfect() {
        let report = PerformanceReport::from_metrics(500.0, 1000.0, 2000.0, 0.01, 50.0);
        let recs = report.recommendations();
        assert!(recs.is_empty());
    }

    #[test]
    fn test_audit_lookup() {
        let report = PerformanceReport::from_metrics(1000.0, 2000.0, 3000.0, 0.1, 150.0);
        let fcp = report.audit("fcp").unwrap();
        assert_eq!(fcp.id, "fcp");
        assert!(report.audit("nonexistent").is_none());
    }

    #[test]
    fn test_rating_boundaries() {
        let green = AuditResult::new("t", "T", 0.9, "");
        assert_eq!(green.rating(), Rating::Green);

        let orange = AuditResult::new("t", "T", 0.5, "");
        assert_eq!(orange.rating(), Rating::Orange);

        let red = AuditResult::new("t", "T", 0.49, "");
        assert_eq!(red.rating(), Rating::Red);
    }

    #[test]
    fn test_score_clamped() {
        let a = AuditResult::new("t", "T", 1.5, "");
        assert!((a.score - 1.0).abs() < f64::EPSILON);

        let b = AuditResult::new("t", "T", -0.5, "");
        assert!((b.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiple_categories() {
        let audits = vec![
            AuditResult::new("a1", "Audit 1", 0.8, ""),
            AuditResult::new("a2", "Audit 2", 0.6, ""),
            AuditResult::new("a3", "Audit 3", 1.0, ""),
        ];
        let categories = vec![
            AuditCategory::new("Cat A", 2.0, &["a1", "a2"]),
            AuditCategory::new("Cat B", 1.0, &["a3"]),
        ];
        let report = PerformanceReport::new(audits, categories);
        // Cat A: (0.8+0.6)/2 = 0.7, Cat B: 1.0
        // Overall: (0.7*2 + 1.0*1)/3 * 100 = 80
        let score = report.overall_score();
        assert!((score - 80.0).abs() < 1.0, "expected ~80, got {}", score);
    }
}
