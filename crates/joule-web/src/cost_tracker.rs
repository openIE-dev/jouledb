//! Cost/resource tracking — per-request cost attribution, resource usage meters,
//! budget alerts, cost allocation by tenant/service/endpoint, billing period rollups.
//!
//! Pure-Rust replacement for cloud cost management SDKs and billing libraries.

use std::collections::HashMap;
use std::fmt;

// ── Cost entry ───────────────────────────────────────────────────

/// A single cost entry attributed to a request or operation.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEntry {
    pub timestamp_s: u64,
    pub tenant: String,
    pub service: String,
    pub endpoint: String,
    pub resource: ResourceType,
    pub quantity: f64,
    pub unit_cost: f64,
}

impl CostEntry {
    pub fn new(
        timestamp_s: u64,
        tenant: impl Into<String>,
        service: impl Into<String>,
        endpoint: impl Into<String>,
        resource: ResourceType,
        quantity: f64,
        unit_cost: f64,
    ) -> Self {
        Self {
            timestamp_s,
            tenant: tenant.into(),
            service: service.into(),
            endpoint: endpoint.into(),
            resource,
            quantity,
            unit_cost,
        }
    }

    /// Total cost for this entry.
    pub fn total_cost(&self) -> f64 {
        self.quantity * self.unit_cost
    }
}

/// Type of resource being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Compute,
    Memory,
    Storage,
    Network,
    ApiCall,
    Custom,
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::Compute => write!(f, "compute"),
            ResourceType::Memory => write!(f, "memory"),
            ResourceType::Storage => write!(f, "storage"),
            ResourceType::Network => write!(f, "network"),
            ResourceType::ApiCall => write!(f, "api-call"),
            ResourceType::Custom => write!(f, "custom"),
        }
    }
}

// ── Resource usage meter ─────────────────────────────────────────

/// Tracks cumulative resource usage.
#[derive(Debug, Clone)]
pub struct UsageMeter {
    pub name: String,
    pub resource: ResourceType,
    pub unit: String,
    cumulative: f64,
    count: u64,
}

impl UsageMeter {
    pub fn new(name: impl Into<String>, resource: ResourceType, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            resource,
            unit: unit.into(),
            cumulative: 0.0,
            count: 0,
        }
    }

    pub fn record(&mut self, quantity: f64) {
        self.cumulative += quantity;
        self.count += 1;
    }

    pub fn cumulative(&self) -> f64 { self.cumulative }
    pub fn count(&self) -> u64 { self.count }

    pub fn average(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.cumulative / self.count as f64 }
    }

    pub fn reset(&mut self) {
        self.cumulative = 0.0;
        self.count = 0;
    }
}

// ── Budget ───────────────────────────────────────────────────────

/// Budget definition with alert thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    pub name: String,
    pub limit: f64,
    /// Alert at these percentage thresholds (e.g., 0.50, 0.80, 0.90).
    pub alert_thresholds: Vec<f64>,
}

impl Budget {
    pub fn new(name: impl Into<String>, limit: f64, alert_thresholds: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            limit,
            alert_thresholds,
        }
    }

    /// Check which alert thresholds have been breached given current spend.
    pub fn breached_thresholds(&self, current_spend: f64) -> Vec<f64> {
        if self.limit <= 0.0 { return Vec::new(); }
        let ratio = current_spend / self.limit;
        self.alert_thresholds.iter()
            .filter(|&&t| ratio >= t)
            .copied()
            .collect()
    }

    /// Remaining budget.
    pub fn remaining(&self, current_spend: f64) -> f64 {
        self.limit - current_spend
    }

    /// Whether the budget is exceeded.
    pub fn is_exceeded(&self, current_spend: f64) -> bool {
        current_spend > self.limit
    }
}

// ── Budget alert ─────────────────────────────────────────────────

/// A generated budget alert.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetAlert {
    pub budget_name: String,
    pub threshold: f64,
    pub current_spend: f64,
    pub limit: f64,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "info"),
            AlertSeverity::Warning => write!(f, "warning"),
            AlertSeverity::Critical => write!(f, "critical"),
        }
    }
}

// ── Billing period ───────────────────────────────────────────────

/// A billing period definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BillingPeriod {
    pub name: String,
    pub start_s: u64,
    pub end_s: u64,
}

impl BillingPeriod {
    pub fn new(name: impl Into<String>, start_s: u64, end_s: u64) -> Self {
        assert!(end_s > start_s, "end must be after start");
        Self { name: name.into(), start_s, end_s }
    }

    pub fn contains(&self, timestamp_s: u64) -> bool {
        timestamp_s >= self.start_s && timestamp_s < self.end_s
    }

    pub fn duration_s(&self) -> u64 {
        self.end_s - self.start_s
    }
}

// ── Rollup ───────────────────────────────────────────────────────

/// Cost rollup for a billing period.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodRollup {
    pub period: BillingPeriod,
    pub total_cost: f64,
    pub by_tenant: Vec<(String, f64)>,
    pub by_service: Vec<(String, f64)>,
    pub by_resource: Vec<(ResourceType, f64)>,
    pub entry_count: usize,
}

// ── Cost Tracker ─────────────────────────────────────────────────

/// Central cost tracker that collects entries and generates reports.
#[derive(Debug, Clone)]
pub struct CostTracker {
    entries: Vec<CostEntry>,
    budgets: HashMap<String, Budget>,
    meters: HashMap<String, UsageMeter>,
    /// Tracks which thresholds have already fired to avoid duplicate alerts.
    fired_alerts: HashMap<String, Vec<f64>>,
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            budgets: HashMap::new(),
            meters: HashMap::new(),
            fired_alerts: HashMap::new(),
        }
    }

    /// Record a cost entry.
    pub fn record(&mut self, entry: CostEntry) {
        self.entries.push(entry);
    }

    /// Add a budget.
    pub fn add_budget(&mut self, budget: Budget) {
        self.budgets.insert(budget.name.clone(), budget);
    }

    /// Add a usage meter.
    pub fn add_meter(&mut self, meter: UsageMeter) {
        self.meters.insert(meter.name.clone(), meter);
    }

    /// Record usage on a meter.
    pub fn record_usage(&mut self, meter_name: &str, quantity: f64) -> bool {
        if let Some(meter) = self.meters.get_mut(meter_name) {
            meter.record(quantity);
            true
        } else {
            false
        }
    }

    pub fn meter(&self, name: &str) -> Option<&UsageMeter> {
        self.meters.get(name)
    }

    /// Total cost across all entries.
    pub fn total_cost(&self) -> f64 {
        self.entries.iter().map(|e| e.total_cost()).sum()
    }

    /// Total cost filtered by time range.
    pub fn cost_in_range(&self, start_s: u64, end_s: u64) -> f64 {
        self.entries.iter()
            .filter(|e| e.timestamp_s >= start_s && e.timestamp_s < end_s)
            .map(|e| e.total_cost())
            .sum()
    }

    /// Cost breakdown by tenant.
    pub fn cost_by_tenant(&self) -> Vec<(String, f64)> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for entry in &self.entries {
            *map.entry(entry.tenant.clone()).or_insert(0.0) += entry.total_cost();
        }
        let mut result: Vec<(String, f64)> = map.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Cost breakdown by service.
    pub fn cost_by_service(&self) -> Vec<(String, f64)> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for entry in &self.entries {
            *map.entry(entry.service.clone()).or_insert(0.0) += entry.total_cost();
        }
        let mut result: Vec<(String, f64)> = map.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Cost breakdown by endpoint.
    pub fn cost_by_endpoint(&self) -> Vec<(String, f64)> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for entry in &self.entries {
            *map.entry(entry.endpoint.clone()).or_insert(0.0) += entry.total_cost();
        }
        let mut result: Vec<(String, f64)> = map.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Cost breakdown by resource type.
    pub fn cost_by_resource(&self) -> Vec<(ResourceType, f64)> {
        let mut map: HashMap<ResourceType, f64> = HashMap::new();
        for entry in &self.entries {
            *map.entry(entry.resource).or_insert(0.0) += entry.total_cost();
        }
        let mut result: Vec<(ResourceType, f64)> = map.into_iter().collect();
        result.sort_by(|a, b| format!("{}", a.0).cmp(&format!("{}", b.0)));
        result
    }

    /// Check budgets and generate new alerts (deduplicating previously fired ones).
    pub fn check_budgets(&mut self) -> Vec<BudgetAlert> {
        let total = self.total_cost();
        let mut alerts = Vec::new();

        for (name, budget) in &self.budgets {
            let breached = budget.breached_thresholds(total);
            let fired = self.fired_alerts.entry(name.clone()).or_default();

            for threshold in breached {
                if !fired.contains(&threshold) {
                    let severity = if threshold >= 1.0 {
                        AlertSeverity::Critical
                    } else if threshold >= 0.8 {
                        AlertSeverity::Warning
                    } else {
                        AlertSeverity::Info
                    };
                    alerts.push(BudgetAlert {
                        budget_name: name.clone(),
                        threshold,
                        current_spend: total,
                        limit: budget.limit,
                        severity,
                    });
                    fired.push(threshold);
                }
            }
        }

        alerts.sort_by(|a, b| a.budget_name.cmp(&b.budget_name));
        alerts
    }

    /// Generate a rollup for a billing period.
    pub fn rollup(&self, period: &BillingPeriod) -> PeriodRollup {
        let period_entries: Vec<&CostEntry> = self.entries.iter()
            .filter(|e| period.contains(e.timestamp_s))
            .collect();

        let total_cost: f64 = period_entries.iter().map(|e| e.total_cost()).sum();
        let entry_count = period_entries.len();

        let mut by_tenant_map: HashMap<String, f64> = HashMap::new();
        let mut by_service_map: HashMap<String, f64> = HashMap::new();
        let mut by_resource_map: HashMap<ResourceType, f64> = HashMap::new();

        for entry in &period_entries {
            let cost = entry.total_cost();
            *by_tenant_map.entry(entry.tenant.clone()).or_insert(0.0) += cost;
            *by_service_map.entry(entry.service.clone()).or_insert(0.0) += cost;
            *by_resource_map.entry(entry.resource).or_insert(0.0) += cost;
        }

        let mut by_tenant: Vec<(String, f64)> = by_tenant_map.into_iter().collect();
        by_tenant.sort_by(|a, b| a.0.cmp(&b.0));

        let mut by_service: Vec<(String, f64)> = by_service_map.into_iter().collect();
        by_service.sort_by(|a, b| a.0.cmp(&b.0));

        let mut by_resource: Vec<(ResourceType, f64)> = by_resource_map.into_iter().collect();
        by_resource.sort_by(|a, b| format!("{}", a.0).cmp(&format!("{}", b.0)));

        PeriodRollup {
            period: period.clone(),
            total_cost,
            by_tenant,
            by_service,
            by_resource,
            entry_count,
        }
    }

    /// Number of entries recorded.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Prune entries older than the given timestamp.
    pub fn prune_before(&mut self, timestamp_s: u64) {
        self.entries.retain(|e| e.timestamp_s >= timestamp_s);
    }

    /// Cost per tenant for a specific service.
    pub fn cost_by_tenant_for_service(&self, service: &str) -> Vec<(String, f64)> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for entry in &self.entries {
            if entry.service == service {
                *map.entry(entry.tenant.clone()).or_insert(0.0) += entry.total_cost();
            }
        }
        let mut result: Vec<(String, f64)> = map.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(ts: u64, tenant: &str, service: &str, resource: ResourceType, qty: f64, unit_cost: f64) -> CostEntry {
        CostEntry::new(ts, tenant, service, "/api/v1", resource, qty, unit_cost)
    }

    #[test]
    fn test_cost_entry_total() {
        let e = CostEntry::new(100, "acme", "api", "/v1", ResourceType::Compute, 10.0, 0.5);
        assert!((e.total_cost() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_resource_type_display() {
        assert_eq!(format!("{}", ResourceType::Compute), "compute");
        assert_eq!(format!("{}", ResourceType::Memory), "memory");
        assert_eq!(format!("{}", ResourceType::Storage), "storage");
        assert_eq!(format!("{}", ResourceType::Network), "network");
        assert_eq!(format!("{}", ResourceType::ApiCall), "api-call");
        assert_eq!(format!("{}", ResourceType::Custom), "custom");
    }

    #[test]
    fn test_usage_meter() {
        let mut meter = UsageMeter::new("cpu", ResourceType::Compute, "seconds");
        meter.record(10.0);
        meter.record(20.0);
        assert!((meter.cumulative() - 30.0).abs() < 1e-10);
        assert_eq!(meter.count(), 2);
        assert!((meter.average() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_usage_meter_reset() {
        let mut meter = UsageMeter::new("mem", ResourceType::Memory, "MB");
        meter.record(100.0);
        meter.reset();
        assert!((meter.cumulative() - 0.0).abs() < 1e-10);
        assert_eq!(meter.count(), 0);
    }

    #[test]
    fn test_usage_meter_average_empty() {
        let meter = UsageMeter::new("mem", ResourceType::Memory, "MB");
        assert!((meter.average() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_breached() {
        let budget = Budget::new("monthly", 100.0, vec![0.5, 0.8, 1.0]);
        assert!(budget.breached_thresholds(40.0).is_empty());
        assert_eq!(budget.breached_thresholds(55.0), vec![0.5]);
        assert_eq!(budget.breached_thresholds(85.0), vec![0.5, 0.8]);
        assert_eq!(budget.breached_thresholds(105.0), vec![0.5, 0.8, 1.0]);
    }

    #[test]
    fn test_budget_remaining() {
        let budget = Budget::new("monthly", 100.0, vec![]);
        assert!((budget.remaining(40.0) - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_exceeded() {
        let budget = Budget::new("monthly", 100.0, vec![]);
        assert!(!budget.is_exceeded(99.0));
        assert!(budget.is_exceeded(101.0));
    }

    #[test]
    fn test_budget_zero_limit() {
        let budget = Budget::new("zero", 0.0, vec![0.5]);
        assert!(budget.breached_thresholds(10.0).is_empty());
    }

    #[test]
    fn test_billing_period() {
        let bp = BillingPeriod::new("march", 1000, 2000);
        assert!(bp.contains(1000));
        assert!(bp.contains(1500));
        assert!(!bp.contains(2000));
        assert!(!bp.contains(999));
        assert_eq!(bp.duration_s(), 1000);
    }

    #[test]
    #[should_panic(expected = "end must be after start")]
    fn test_billing_period_bad_range() {
        BillingPeriod::new("bad", 2000, 1000);
    }

    #[test]
    fn test_cost_tracker_total() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "acme", "api", ResourceType::Memory, 5.0, 2.0));
        assert!((tracker.total_cost() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_tracker_by_tenant() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "beta", "api", ResourceType::Compute, 5.0, 1.0));
        tracker.record(make_entry(300, "acme", "api", ResourceType::Compute, 3.0, 1.0));
        let by_tenant = tracker.cost_by_tenant();
        assert_eq!(by_tenant.len(), 2);
        assert_eq!(by_tenant[0].0, "acme");
        assert!((by_tenant[0].1 - 13.0).abs() < 1e-10);
        assert_eq!(by_tenant[1].0, "beta");
        assert!((by_tenant[1].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_tracker_by_service() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "acme", "db", ResourceType::Compute, 5.0, 1.0));
        let by_svc = tracker.cost_by_service();
        assert_eq!(by_svc.len(), 2);
    }

    #[test]
    fn test_cost_tracker_by_endpoint() {
        let mut tracker = CostTracker::new();
        tracker.record(CostEntry::new(100, "t", "s", "/a", ResourceType::Compute, 1.0, 1.0));
        tracker.record(CostEntry::new(200, "t", "s", "/b", ResourceType::Compute, 2.0, 1.0));
        let by_ep = tracker.cost_by_endpoint();
        assert_eq!(by_ep.len(), 2);
    }

    #[test]
    fn test_cost_tracker_by_resource() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "acme", "api", ResourceType::Network, 5.0, 1.0));
        let by_res = tracker.cost_by_resource();
        assert_eq!(by_res.len(), 2);
    }

    #[test]
    fn test_cost_in_range() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "acme", "api", ResourceType::Compute, 5.0, 1.0));
        tracker.record(make_entry(300, "acme", "api", ResourceType::Compute, 3.0, 1.0));
        assert!((tracker.cost_in_range(150, 250) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_alerts() {
        let mut tracker = CostTracker::new();
        tracker.add_budget(Budget::new("monthly", 100.0, vec![0.5, 0.8, 1.0]));
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 55.0, 1.0));
        let alerts = tracker.check_budgets();
        assert_eq!(alerts.len(), 1);
        assert!((alerts[0].threshold - 0.5).abs() < 1e-10);
        assert_eq!(alerts[0].severity, AlertSeverity::Info);
    }

    #[test]
    fn test_budget_alerts_dedup() {
        let mut tracker = CostTracker::new();
        tracker.add_budget(Budget::new("monthly", 100.0, vec![0.5, 0.8]));
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 55.0, 1.0));
        let alerts1 = tracker.check_budgets();
        assert_eq!(alerts1.len(), 1);
        // Same state, should not re-fire.
        let alerts2 = tracker.check_budgets();
        assert!(alerts2.is_empty());
    }

    #[test]
    fn test_budget_alert_escalation() {
        let mut tracker = CostTracker::new();
        tracker.add_budget(Budget::new("monthly", 100.0, vec![0.5, 0.8, 1.0]));
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 85.0, 1.0));
        let alerts = tracker.check_budgets();
        assert_eq!(alerts.len(), 2); // 0.5 and 0.8
        let warning_alert = alerts.iter().find(|a| (a.threshold - 0.8).abs() < 1e-10).unwrap();
        assert_eq!(warning_alert.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_rollup() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "beta", "db", ResourceType::Memory, 5.0, 2.0));
        tracker.record(make_entry(500, "acme", "api", ResourceType::Compute, 3.0, 1.0));

        let period = BillingPeriod::new("test", 50, 300);
        let rollup = tracker.rollup(&period);
        assert!((rollup.total_cost - 20.0).abs() < 1e-10);
        assert_eq!(rollup.entry_count, 2);
        assert_eq!(rollup.by_tenant.len(), 2);
    }

    #[test]
    fn test_prune_before() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "a", "s", ResourceType::Compute, 1.0, 1.0));
        tracker.record(make_entry(200, "a", "s", ResourceType::Compute, 1.0, 1.0));
        tracker.record(make_entry(300, "a", "s", ResourceType::Compute, 1.0, 1.0));
        tracker.prune_before(200);
        assert_eq!(tracker.entry_count(), 2);
    }

    #[test]
    fn test_meters_via_tracker() {
        let mut tracker = CostTracker::new();
        tracker.add_meter(UsageMeter::new("cpu", ResourceType::Compute, "seconds"));
        assert!(tracker.record_usage("cpu", 10.0));
        assert!(!tracker.record_usage("nonexistent", 5.0));
        let m = tracker.meter("cpu").unwrap();
        assert!((m.cumulative() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_by_tenant_for_service() {
        let mut tracker = CostTracker::new();
        tracker.record(make_entry(100, "acme", "api", ResourceType::Compute, 10.0, 1.0));
        tracker.record(make_entry(200, "beta", "api", ResourceType::Compute, 5.0, 1.0));
        tracker.record(make_entry(300, "acme", "db", ResourceType::Compute, 20.0, 1.0));
        let result = tracker.cost_by_tenant_for_service("api");
        assert_eq!(result.len(), 2);
        let acme = result.iter().find(|(t, _)| t == "acme").unwrap();
        assert!((acme.1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(format!("{}", AlertSeverity::Info), "info");
        assert_eq!(format!("{}", AlertSeverity::Warning), "warning");
        assert_eq!(format!("{}", AlertSeverity::Critical), "critical");
    }

    #[test]
    fn test_default_tracker() {
        let tracker = CostTracker::default();
        assert_eq!(tracker.entry_count(), 0);
        assert!((tracker.total_cost() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rollup_empty_period() {
        let tracker = CostTracker::new();
        let period = BillingPeriod::new("empty", 1000, 2000);
        let rollup = tracker.rollup(&period);
        assert!((rollup.total_cost - 0.0).abs() < 1e-10);
        assert_eq!(rollup.entry_count, 0);
    }
}
