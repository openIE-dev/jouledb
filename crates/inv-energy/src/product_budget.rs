use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tracing::warn;

/// Alert delivery channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    Webhook { url: String },
    Email { address: String },
    Slack { channel: String },
    Log,
}

/// A threshold that triggers an alert when exceeded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub percent: f64,
    pub channels: Vec<AlertChannel>,
}

/// Budget type for different time windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Daily,
    Monthly,
    PerRequest,
}

/// Per-product energy budget configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductBudget {
    pub product_id: String,
    pub product_name: String,
    pub daily_joules: Option<f64>,
    pub monthly_joules: Option<f64>,
    pub per_request_joules: Option<f64>,
    pub thresholds: Vec<AlertThreshold>,
    pub created_at: SystemTime,
    pub enabled: bool,
}

/// A triggered budget alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    pub product_id: String,
    pub period: BudgetPeriod,
    pub current_joules: f64,
    pub limit_joules: f64,
    pub threshold_pct: f64,
    pub utilization_pct: f64,
    pub triggered_at: SystemTime,
    pub channels: Vec<AlertChannel>,
}

/// Usage record for a product.
#[derive(Debug, Clone)]
struct ProductUsage {
    daily_joules: f64,
    monthly_joules: f64,
    request_count: u64,
    last_request_joules: f64,
    daily_reset_at: SystemTime,
    monthly_reset_at: SystemTime,
}

/// Budget check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub product_id: String,
    pub daily_utilization: Option<f64>,
    pub monthly_utilization: Option<f64>,
    pub last_request_utilization: Option<f64>,
    pub exceeded: bool,
    pub alerts: Vec<BudgetAlert>,
}

/// Manager for per-product energy budgets with alert dispatch.
pub struct ProductBudgetManager {
    budgets: Arc<RwLock<HashMap<String, ProductBudget>>>,
    usage: Arc<RwLock<HashMap<String, ProductUsage>>>,
    alerts: Arc<RwLock<Vec<BudgetAlert>>>,
    total_energy_uj: std::sync::atomic::AtomicU64,
}

impl Default for ProductBudgetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProductBudgetManager {
    pub fn new() -> Self {
        Self {
            budgets: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            total_energy_uj: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Set or update a product's energy budget.
    pub fn set_budget(&self, budget: ProductBudget) {
        let energy = 0.001; // 1 mJ per budget set
        self.track_energy(energy);

        let product_id = budget.product_id.clone();
        self.budgets
            .write()
            .unwrap()
            .insert(product_id.clone(), budget);

        // Initialize usage if needed
        let mut usage = self.usage.write().unwrap();
        usage.entry(product_id).or_insert_with(|| ProductUsage {
            daily_joules: 0.0,
            monthly_joules: 0.0,
            request_count: 0,
            last_request_joules: 0.0,
            daily_reset_at: SystemTime::now(),
            monthly_reset_at: SystemTime::now(),
        });
    }

    /// Record energy usage for a product and check budgets.
    pub fn record_usage(&self, product_id: &str, joules: f64) -> BudgetStatus {
        let energy = 0.0005; // 0.5 mJ per record
        self.track_energy(energy);

        // Update usage
        {
            let mut usage = self.usage.write().unwrap();
            let u = usage
                .entry(product_id.to_string())
                .or_insert_with(|| ProductUsage {
                    daily_joules: 0.0,
                    monthly_joules: 0.0,
                    request_count: 0,
                    last_request_joules: 0.0,
                    daily_reset_at: SystemTime::now(),
                    monthly_reset_at: SystemTime::now(),
                });
            u.daily_joules += joules;
            u.monthly_joules += joules;
            u.request_count += 1;
            u.last_request_joules = joules;
        }

        self.check_budget(product_id)
    }

    /// Check current budget status for a product.
    pub fn check_budget(&self, product_id: &str) -> BudgetStatus {
        let budgets = self.budgets.read().unwrap();
        let usage = self.usage.read().unwrap();

        let budget = match budgets.get(product_id) {
            Some(b) => b,
            None => {
                return BudgetStatus {
                    product_id: product_id.to_string(),
                    daily_utilization: None,
                    monthly_utilization: None,
                    last_request_utilization: None,
                    exceeded: false,
                    alerts: vec![],
                };
            }
        };

        if !budget.enabled {
            return BudgetStatus {
                product_id: product_id.to_string(),
                daily_utilization: None,
                monthly_utilization: None,
                last_request_utilization: None,
                exceeded: false,
                alerts: vec![],
            };
        }

        let u = match usage.get(product_id) {
            Some(u) => u,
            None => {
                return BudgetStatus {
                    product_id: product_id.to_string(),
                    daily_utilization: Some(0.0),
                    monthly_utilization: Some(0.0),
                    last_request_utilization: Some(0.0),
                    exceeded: false,
                    alerts: vec![],
                };
            }
        };

        let mut alerts = Vec::new();
        let mut exceeded = false;

        let daily_util = budget.daily_joules.map(|limit| {
            let util = if limit > 0.0 {
                u.daily_joules / limit
            } else {
                0.0
            };
            if util >= 1.0 {
                exceeded = true;
            }
            self.check_thresholds(
                product_id,
                BudgetPeriod::Daily,
                u.daily_joules,
                limit,
                util,
                &budget.thresholds,
                &mut alerts,
            );
            util
        });

        let monthly_util = budget.monthly_joules.map(|limit| {
            let util = if limit > 0.0 {
                u.monthly_joules / limit
            } else {
                0.0
            };
            if util >= 1.0 {
                exceeded = true;
            }
            self.check_thresholds(
                product_id,
                BudgetPeriod::Monthly,
                u.monthly_joules,
                limit,
                util,
                &budget.thresholds,
                &mut alerts,
            );
            util
        });

        let request_util = budget.per_request_joules.map(|limit| {
            let util = if limit > 0.0 {
                u.last_request_joules / limit
            } else {
                0.0
            };
            if util >= 1.0 {
                exceeded = true;
            }
            util
        });

        // Store alerts
        if !alerts.is_empty() {
            let mut stored = self.alerts.write().unwrap();
            for alert in &alerts {
                warn!(
                    product_id = %alert.product_id,
                    period = ?alert.period,
                    utilization = %alert.utilization_pct,
                    threshold = %alert.threshold_pct,
                    "energy budget threshold exceeded"
                );
            }
            stored.extend(alerts.clone());
        }

        BudgetStatus {
            product_id: product_id.to_string(),
            daily_utilization: daily_util,
            monthly_utilization: monthly_util,
            last_request_utilization: request_util,
            exceeded,
            alerts,
        }
    }

    /// Get all triggered alerts.
    pub fn get_alerts(&self) -> Vec<BudgetAlert> {
        self.alerts.read().unwrap().clone()
    }

    /// Get alerts for a specific product.
    pub fn get_product_alerts(&self, product_id: &str) -> Vec<BudgetAlert> {
        self.alerts
            .read()
            .unwrap()
            .iter()
            .filter(|a| a.product_id == product_id)
            .cloned()
            .collect()
    }

    /// Reset daily usage counters for all products.
    pub fn reset_daily(&self) {
        let mut usage = self.usage.write().unwrap();
        let now = SystemTime::now();
        for u in usage.values_mut() {
            u.daily_joules = 0.0;
            u.daily_reset_at = now;
        }
    }

    /// Reset monthly usage counters for all products.
    pub fn reset_monthly(&self) {
        let mut usage = self.usage.write().unwrap();
        let now = SystemTime::now();
        for u in usage.values_mut() {
            u.monthly_joules = 0.0;
            u.monthly_reset_at = now;
        }
    }

    /// Get budget for a product.
    pub fn get_budget(&self, product_id: &str) -> Option<ProductBudget> {
        self.budgets.read().unwrap().get(product_id).cloned()
    }

    /// Remove a product budget.
    pub fn remove_budget(&self, product_id: &str) {
        self.budgets.write().unwrap().remove(product_id);
        self.usage.write().unwrap().remove(product_id);
    }

    /// List all product budgets.
    pub fn list_budgets(&self) -> Vec<ProductBudget> {
        self.budgets.read().unwrap().values().cloned().collect()
    }

    /// Total energy consumed by budget management operations.
    pub fn total_energy_joules(&self) -> f64 {
        self.total_energy_uj
            .load(std::sync::atomic::Ordering::Relaxed) as f64
            / 1_000_000.0
    }

    fn track_energy(&self, joules: f64) {
        self.total_energy_uj.fetch_add(
            (joules * 1_000_000.0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn check_thresholds(
        &self,
        product_id: &str,
        period: BudgetPeriod,
        current: f64,
        limit: f64,
        utilization: f64,
        thresholds: &[AlertThreshold],
        alerts: &mut Vec<BudgetAlert>,
    ) {
        let util_pct = utilization * 100.0;
        for threshold in thresholds {
            if util_pct >= threshold.percent {
                alerts.push(BudgetAlert {
                    product_id: product_id.to_string(),
                    period,
                    current_joules: current,
                    limit_joules: limit,
                    threshold_pct: threshold.percent,
                    utilization_pct: util_pct,
                    triggered_at: SystemTime::now(),
                    channels: threshold.channels.clone(),
                });
            }
        }
    }
}
