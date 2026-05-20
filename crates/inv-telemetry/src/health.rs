use std::time::Instant;

use serde::Serialize;

/// Overall health status of a subsystem or node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}

impl HealthStatus {
    /// Whether the status is `Healthy`.
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Whether the status is `Unhealthy`.
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy { .. })
    }
}

/// Health report for a single subsystem.
#[derive(Debug, Clone, Serialize)]
pub struct SubsystemHealth {
    pub name: String,
    pub status: HealthStatus,
    pub latency_ms: f64,
    pub message: Option<String>,
}

/// A health check for a subsystem.
///
/// Implementors define a name and a check function that returns the
/// subsystem's current health status. Checks should be fast (< 100ms).
pub trait HealthCheck: Send + Sync {
    /// Name of the subsystem being checked.
    fn name(&self) -> &str;

    /// Execute the health check and return the result.
    fn check(&self) -> SubsystemHealth;
}

/// Aggregated health report across all subsystems.
#[derive(Debug, Clone, Serialize)]
pub struct AggregatedHealth {
    pub status: HealthStatus,
    pub subsystems: Vec<SubsystemHealth>,
    pub total_latency_ms: f64,
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub unhealthy_count: usize,
}

/// Aggregates health checks from multiple subsystems into a single report.
///
/// The overall status is determined by the worst subsystem:
/// - All healthy → Healthy
/// - Any degraded (none unhealthy) → Degraded
/// - Any unhealthy → Unhealthy
pub struct HealthAggregator {
    checks: Vec<Box<dyn HealthCheck>>,
}

impl HealthAggregator {
    /// Create an empty health aggregator.
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Register a health check.
    pub fn register(&mut self, check: Box<dyn HealthCheck>) {
        self.checks.push(check);
    }

    /// Number of registered checks.
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Run all health checks and produce an aggregated report.
    pub fn check_all(&self) -> AggregatedHealth {
        let start = Instant::now();
        let mut subsystems = Vec::with_capacity(self.checks.len());
        let mut healthy = 0usize;
        let mut degraded = 0usize;
        let mut unhealthy = 0usize;

        for check in &self.checks {
            let result = check.check();
            match &result.status {
                HealthStatus::Healthy => healthy += 1,
                HealthStatus::Degraded { .. } => degraded += 1,
                HealthStatus::Unhealthy { .. } => unhealthy += 1,
            }
            subsystems.push(result);
        }

        let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let status = if unhealthy > 0 {
            let names: Vec<&str> = subsystems
                .iter()
                .filter(|s| s.status.is_unhealthy())
                .map(|s| s.name.as_str())
                .collect();
            HealthStatus::Unhealthy {
                reason: format!("{} subsystem(s) unhealthy: {}", unhealthy, names.join(", ")),
            }
        } else if degraded > 0 {
            let names: Vec<&str> = subsystems
                .iter()
                .filter(|s| matches!(s.status, HealthStatus::Degraded { .. }))
                .map(|s| s.name.as_str())
                .collect();
            HealthStatus::Degraded {
                reason: format!("{} subsystem(s) degraded: {}", degraded, names.join(", ")),
            }
        } else {
            HealthStatus::Healthy
        };

        AggregatedHealth {
            status,
            subsystems,
            total_latency_ms,
            healthy_count: healthy,
            degraded_count: degraded,
            unhealthy_count: unhealthy,
        }
    }

    /// Quick check: are all subsystems healthy?
    pub fn is_healthy(&self) -> bool {
        self.check_all().status.is_healthy()
    }
}

impl Default for HealthAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysHealthy;
    impl HealthCheck for AlwaysHealthy {
        fn name(&self) -> &str {
            "always_healthy"
        }
        fn check(&self) -> SubsystemHealth {
            SubsystemHealth {
                name: "always_healthy".to_string(),
                status: HealthStatus::Healthy,
                latency_ms: 0.1,
                message: None,
            }
        }
    }

    struct AlwaysDegraded;
    impl HealthCheck for AlwaysDegraded {
        fn name(&self) -> &str {
            "degraded_check"
        }
        fn check(&self) -> SubsystemHealth {
            SubsystemHealth {
                name: "degraded_check".to_string(),
                status: HealthStatus::Degraded {
                    reason: "high latency".to_string(),
                },
                latency_ms: 50.0,
                message: Some("p99 > 500ms".to_string()),
            }
        }
    }

    struct AlwaysUnhealthy;
    impl HealthCheck for AlwaysUnhealthy {
        fn name(&self) -> &str {
            "unhealthy_check"
        }
        fn check(&self) -> SubsystemHealth {
            SubsystemHealth {
                name: "unhealthy_check".to_string(),
                status: HealthStatus::Unhealthy {
                    reason: "connection refused".to_string(),
                },
                latency_ms: 100.0,
                message: Some("database unreachable".to_string()),
            }
        }
    }

    #[test]
    fn empty_aggregator_is_healthy() {
        let agg = HealthAggregator::new();
        assert!(agg.is_healthy());
        let report = agg.check_all();
        assert_eq!(report.healthy_count, 0);
    }

    #[test]
    fn all_healthy() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysHealthy));
        agg.register(Box::new(AlwaysHealthy));

        let report = agg.check_all();
        assert!(report.status.is_healthy());
        assert_eq!(report.healthy_count, 2);
        assert_eq!(report.degraded_count, 0);
        assert_eq!(report.unhealthy_count, 0);
    }

    #[test]
    fn degraded_overrides_healthy() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysHealthy));
        agg.register(Box::new(AlwaysDegraded));

        let report = agg.check_all();
        assert!(matches!(report.status, HealthStatus::Degraded { .. }));
        assert_eq!(report.healthy_count, 1);
        assert_eq!(report.degraded_count, 1);
    }

    #[test]
    fn unhealthy_overrides_all() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysHealthy));
        agg.register(Box::new(AlwaysDegraded));
        agg.register(Box::new(AlwaysUnhealthy));

        let report = agg.check_all();
        assert!(report.status.is_unhealthy());
        assert_eq!(report.unhealthy_count, 1);
    }

    #[test]
    fn check_count() {
        let mut agg = HealthAggregator::new();
        assert_eq!(agg.check_count(), 0);
        agg.register(Box::new(AlwaysHealthy));
        agg.register(Box::new(AlwaysDegraded));
        assert_eq!(agg.check_count(), 2);
    }

    #[test]
    fn subsystem_details_preserved() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysUnhealthy));

        let report = agg.check_all();
        assert_eq!(report.subsystems.len(), 1);
        assert_eq!(report.subsystems[0].name, "unhealthy_check");
        assert_eq!(
            report.subsystems[0].message.as_deref(),
            Some("database unreachable")
        );
    }

    #[test]
    fn health_status_variants() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Healthy.is_unhealthy());

        let degraded = HealthStatus::Degraded {
            reason: "slow".into(),
        };
        assert!(!degraded.is_healthy());
        assert!(!degraded.is_unhealthy());

        let unhealthy = HealthStatus::Unhealthy {
            reason: "down".into(),
        };
        assert!(!unhealthy.is_healthy());
        assert!(unhealthy.is_unhealthy());
    }

    #[test]
    fn aggregated_health_serializes() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysHealthy));
        let report = agg.check_all();
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["status"]["status"], "healthy");
        assert!(json["subsystems"].is_array());
    }

    #[test]
    fn total_latency_measured() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysHealthy));
        let report = agg.check_all();
        assert!(report.total_latency_ms >= 0.0);
    }

    #[test]
    fn is_healthy_shortcut() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysHealthy));
        assert!(agg.is_healthy());

        let mut agg2 = HealthAggregator::new();
        agg2.register(Box::new(AlwaysUnhealthy));
        assert!(!agg2.is_healthy());
    }

    #[test]
    fn default_creates_empty() {
        let agg = HealthAggregator::default();
        assert_eq!(agg.check_count(), 0);
    }

    #[test]
    fn multiple_unhealthy_listed() {
        let mut agg = HealthAggregator::new();
        agg.register(Box::new(AlwaysUnhealthy));
        agg.register(Box::new(AlwaysUnhealthy));

        let report = agg.check_all();
        if let HealthStatus::Unhealthy { reason } = &report.status {
            assert!(reason.contains("2 subsystem(s) unhealthy"));
        } else {
            panic!("Expected unhealthy status");
        }
    }
}
