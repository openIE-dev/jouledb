//! Subscription lifecycle — plans, billing intervals, proration, and metered usage.
//!
//! Replaces Stripe Billing / Chargebee / Recurly with a pure-Rust subscription
//! domain model. No HTTP calls — only models plans, subscriptions, and billing math.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Subscription domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscriptionError {
    /// Invalid state transition.
    InvalidTransition { from: SubscriptionStatus, to: SubscriptionStatus },
    /// Plan not found.
    PlanNotFound(String),
    /// Subscription not found.
    NotFound(String),
    /// Cannot downgrade during trial.
    CannotChangeDuringStatus(SubscriptionStatus),
    /// Negative usage.
    InvalidUsage,
    /// Already cancelled.
    AlreadyCancelled,
}

impl std::fmt::Display for SubscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from:?} to {to:?}")
            }
            Self::PlanNotFound(id) => write!(f, "plan not found: {id}"),
            Self::NotFound(id) => write!(f, "subscription not found: {id}"),
            Self::CannotChangeDuringStatus(s) => {
                write!(f, "cannot change plan during status {s:?}")
            }
            Self::InvalidUsage => write!(f, "usage amount must be non-negative"),
            Self::AlreadyCancelled => write!(f, "subscription is already cancelled"),
        }
    }
}

impl std::error::Error for SubscriptionError {}

// ── Billing Interval ────────────────────────────────────────────

/// Billing frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BillingInterval {
    Weekly,
    Monthly,
    Yearly,
}

impl BillingInterval {
    /// Number of days in this interval (approximate for billing math).
    pub fn days(&self) -> i64 {
        match self {
            Self::Weekly => 7,
            Self::Monthly => 30,
            Self::Yearly => 365,
        }
    }
}

// ── Plan ────────────────────────────────────────────────────────

/// A subscription plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub name: String,
    pub amount_cents: i64,
    pub interval: BillingInterval,
    pub trial_days: u32,
    /// If set, this plan includes metered usage with a base allowance.
    pub metered: Option<MeteredConfig>,
}

/// Configuration for metered/usage-based billing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredConfig {
    /// Units included in the base plan price.
    pub included_units: u64,
    /// Cost per additional unit in cents.
    pub overage_cost_cents: i64,
    /// Unit label (e.g., "API calls", "GB").
    pub unit_label: String,
}

impl Plan {
    /// Create a new flat-rate plan.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        amount_cents: i64,
        interval: BillingInterval,
        trial_days: u32,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            amount_cents,
            interval,
            trial_days,
            metered: None,
        }
    }

    /// Create a metered plan.
    pub fn with_metered(mut self, config: MeteredConfig) -> Self {
        self.metered = Some(config);
        self
    }

    /// Daily rate in cents (for proration math).
    pub fn daily_rate_cents(&self) -> f64 {
        self.amount_cents as f64 / self.interval.days() as f64
    }
}

// ── Subscription Status ─────────────────────────────────────────

/// Lifecycle states for a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    Trialing,
    Active,
    PastDue,
    Cancelled,
    Expired,
}

impl SubscriptionStatus {
    /// Valid transitions from this status.
    fn valid_transitions(&self) -> &[SubscriptionStatus] {
        match self {
            Self::Trialing => &[Self::Active, Self::Cancelled, Self::Expired],
            Self::Active => &[Self::PastDue, Self::Cancelled, Self::Expired],
            Self::PastDue => &[Self::Active, Self::Cancelled, Self::Expired],
            Self::Cancelled => &[],
            Self::Expired => &[],
        }
    }

    /// Whether this status allows plan changes.
    pub fn allows_plan_change(&self) -> bool {
        matches!(self, Self::Active)
    }
}

// ── Subscription ────────────────────────────────────────────────

/// A subscription instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub id: String,
    pub plan_id: String,
    pub status: SubscriptionStatus,
    pub current_period_start: DateTime<Utc>,
    pub current_period_end: DateTime<Utc>,
    pub cancel_at_period_end: bool,
    pub created_at: DateTime<Utc>,
    /// Reported usage for metered plans in the current period.
    pub usage_units: u64,
}

impl Subscription {
    /// Create a new subscription, starting in trialing if the plan has trial days.
    pub fn new(id: impl Into<String>, plan: &Plan, now: DateTime<Utc>) -> Self {
        let (status, period_end) = if plan.trial_days > 0 {
            (
                SubscriptionStatus::Trialing,
                now + Duration::days(plan.trial_days as i64),
            )
        } else {
            (
                SubscriptionStatus::Active,
                now + Duration::days(plan.interval.days()),
            )
        };

        Self {
            id: id.into(),
            plan_id: plan.id.clone(),
            status,
            current_period_start: now,
            current_period_end: period_end,
            cancel_at_period_end: false,
            created_at: now,
            usage_units: 0,
        }
    }

    /// Transition to a new status.
    pub fn transition(&mut self, to: SubscriptionStatus) -> Result<(), SubscriptionError> {
        if self.status.valid_transitions().contains(&to) {
            self.status = to;
            Ok(())
        } else {
            Err(SubscriptionError::InvalidTransition {
                from: self.status,
                to,
            })
        }
    }

    /// Activate the subscription after trial (sets period to plan interval).
    pub fn activate(&mut self, plan: &Plan, now: DateTime<Utc>) -> Result<(), SubscriptionError> {
        self.transition(SubscriptionStatus::Active)?;
        self.current_period_start = now;
        self.current_period_end = now + Duration::days(plan.interval.days());
        self.usage_units = 0;
        Ok(())
    }

    /// Cancel the subscription at end of period.
    pub fn cancel_at_end(&mut self) -> Result<(), SubscriptionError> {
        if self.status == SubscriptionStatus::Cancelled {
            return Err(SubscriptionError::AlreadyCancelled);
        }
        self.cancel_at_period_end = true;
        Ok(())
    }

    /// Cancel the subscription immediately.
    pub fn cancel_now(&mut self) -> Result<(), SubscriptionError> {
        self.transition(SubscriptionStatus::Cancelled)
    }

    /// Report metered usage.
    pub fn report_usage(&mut self, units: u64) -> Result<(), SubscriptionError> {
        if self.status != SubscriptionStatus::Active && self.status != SubscriptionStatus::Trialing
        {
            return Err(SubscriptionError::CannotChangeDuringStatus(self.status));
        }
        self.usage_units += units;
        Ok(())
    }

    /// Remaining days in current period.
    pub fn remaining_days(&self, now: DateTime<Utc>) -> i64 {
        let diff = self.current_period_end - now;
        diff.num_days().max(0)
    }

    /// Renew the subscription for another period.
    pub fn renew(&mut self, plan: &Plan) -> Result<(), SubscriptionError> {
        if self.cancel_at_period_end {
            self.status = SubscriptionStatus::Cancelled;
            return Err(SubscriptionError::AlreadyCancelled);
        }
        if self.status != SubscriptionStatus::Active {
            return Err(SubscriptionError::CannotChangeDuringStatus(self.status));
        }
        self.current_period_start = self.current_period_end;
        self.current_period_end =
            self.current_period_start + Duration::days(plan.interval.days());
        self.usage_units = 0;
        Ok(())
    }
}

// ── Proration ───────────────────────────────────────────────────

/// Proration result when changing plans mid-period.
#[derive(Debug, Clone, PartialEq)]
pub struct Proration {
    /// Credit from remaining days on old plan (positive).
    pub credit_cents: f64,
    /// Charge for remaining days on new plan (positive).
    pub charge_cents: f64,
    /// Net amount: charge - credit (negative means customer gets credit).
    pub net_cents: f64,
    /// Days remaining when the change happens.
    pub remaining_days: i64,
}

/// Calculate proration when switching plans mid-period.
pub fn calculate_proration(
    old_plan: &Plan,
    new_plan: &Plan,
    remaining_days: i64,
) -> Proration {
    let credit = old_plan.daily_rate_cents() * remaining_days as f64;
    let charge = new_plan.daily_rate_cents() * remaining_days as f64;
    Proration {
        credit_cents: credit,
        charge_cents: charge,
        net_cents: charge - credit,
        remaining_days,
    }
}

/// Compute overage charge for metered usage.
pub fn compute_overage(plan: &Plan, usage: u64) -> i64 {
    match &plan.metered {
        Some(config) => {
            if usage > config.included_units {
                let overage = usage - config.included_units;
                overage as i64 * config.overage_cost_cents
            } else {
                0
            }
        }
        None => 0,
    }
}

// ── Plan Change ─────────────────────────────────────────────────

/// Change a subscription's plan (upgrade or downgrade).
pub fn change_plan(
    sub: &mut Subscription,
    old_plan: &Plan,
    new_plan: &Plan,
    now: DateTime<Utc>,
) -> Result<Proration, SubscriptionError> {
    if !sub.status.allows_plan_change() {
        return Err(SubscriptionError::CannotChangeDuringStatus(sub.status));
    }
    let remaining = sub.remaining_days(now);
    let proration = calculate_proration(old_plan, new_plan, remaining);
    sub.plan_id = new_plan.id.clone();
    Ok(proration)
}

// ── Subscription Store ──────────────────────────────────────────

/// In-memory subscription store with plans.
#[derive(Debug, Default)]
pub struct SubscriptionStore {
    pub plans: HashMap<String, Plan>,
    pub subscriptions: HashMap<String, Subscription>,
}

impl SubscriptionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a plan.
    pub fn add_plan(&mut self, plan: Plan) {
        self.plans.insert(plan.id.clone(), plan);
    }

    /// Look up a plan.
    pub fn get_plan(&self, id: &str) -> Option<&Plan> {
        self.plans.get(id)
    }

    /// Create a subscription for a plan.
    pub fn subscribe(
        &mut self,
        sub_id: impl Into<String>,
        plan_id: &str,
        now: DateTime<Utc>,
    ) -> Result<&Subscription, SubscriptionError> {
        let plan = self
            .plans
            .get(plan_id)
            .ok_or_else(|| SubscriptionError::PlanNotFound(plan_id.to_string()))?
            .clone();
        let id = sub_id.into();
        let sub = Subscription::new(id.clone(), &plan, now);
        self.subscriptions.insert(id.clone(), sub);
        Ok(self.subscriptions.get(&id).unwrap())
    }

    /// Get a subscription.
    pub fn get_subscription(&self, id: &str) -> Option<&Subscription> {
        self.subscriptions.get(id)
    }

    /// Get a mutable subscription.
    pub fn get_subscription_mut(&mut self, id: &str) -> Option<&mut Subscription> {
        self.subscriptions.get_mut(id)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap()
    }

    fn basic_plan() -> Plan {
        Plan::new("pro", "Pro Plan", 2999, BillingInterval::Monthly, 0)
    }

    fn trial_plan() -> Plan {
        Plan::new("starter", "Starter Plan", 999, BillingInterval::Monthly, 14)
    }

    fn yearly_plan() -> Plan {
        Plan::new("enterprise", "Enterprise", 29999, BillingInterval::Yearly, 0)
    }

    #[test]
    fn create_subscription_active() {
        let plan = basic_plan();
        let sub = Subscription::new("sub-1", &plan, now());
        assert_eq!(sub.status, SubscriptionStatus::Active);
        assert_eq!(sub.plan_id, "pro");
        assert_eq!(sub.current_period_start, now());
        assert_eq!(sub.current_period_end, now() + Duration::days(30));
    }

    #[test]
    fn create_subscription_with_trial() {
        let plan = trial_plan();
        let sub = Subscription::new("sub-2", &plan, now());
        assert_eq!(sub.status, SubscriptionStatus::Trialing);
        assert_eq!(sub.current_period_end, now() + Duration::days(14));
    }

    #[test]
    fn activate_after_trial() {
        let plan = trial_plan();
        let mut sub = Subscription::new("sub-3", &plan, now());
        assert_eq!(sub.status, SubscriptionStatus::Trialing);
        let activate_time = now() + Duration::days(14);
        sub.activate(&plan, activate_time).unwrap();
        assert_eq!(sub.status, SubscriptionStatus::Active);
        assert_eq!(sub.current_period_start, activate_time);
    }

    #[test]
    fn cancel_at_period_end() {
        let plan = basic_plan();
        let mut sub = Subscription::new("sub-4", &plan, now());
        sub.cancel_at_end().unwrap();
        assert!(sub.cancel_at_period_end);
        assert_eq!(sub.status, SubscriptionStatus::Active);
    }

    #[test]
    fn cancel_immediately() {
        let plan = basic_plan();
        let mut sub = Subscription::new("sub-5", &plan, now());
        sub.cancel_now().unwrap();
        assert_eq!(sub.status, SubscriptionStatus::Cancelled);
    }

    #[test]
    fn invalid_transition() {
        let plan = basic_plan();
        let mut sub = Subscription::new("sub-6", &plan, now());
        sub.cancel_now().unwrap();
        let result = sub.transition(SubscriptionStatus::Active);
        assert!(result.is_err());
    }

    #[test]
    fn proration_upgrade() {
        let old = Plan::new("basic", "Basic", 1000, BillingInterval::Monthly, 0);
        let new = Plan::new("pro", "Pro", 3000, BillingInterval::Monthly, 0);
        let proration = calculate_proration(&old, &new, 15);
        // Old daily: 1000/30 = 33.33, credit = 500
        // New daily: 3000/30 = 100.0, charge = 1500
        assert!(proration.net_cents > 0.0); // upgrade costs more
        assert_eq!(proration.remaining_days, 15);
        let expected_credit = 1000.0 / 30.0 * 15.0;
        let expected_charge = 3000.0 / 30.0 * 15.0;
        assert!((proration.credit_cents - expected_credit).abs() < 0.01);
        assert!((proration.charge_cents - expected_charge).abs() < 0.01);
    }

    #[test]
    fn proration_downgrade() {
        let old = Plan::new("pro", "Pro", 3000, BillingInterval::Monthly, 0);
        let new = Plan::new("basic", "Basic", 1000, BillingInterval::Monthly, 0);
        let proration = calculate_proration(&old, &new, 15);
        assert!(proration.net_cents < 0.0); // downgrade gives credit
    }

    #[test]
    fn change_plan_mid_period() {
        let old = basic_plan();
        let new = yearly_plan();
        let mut sub = Subscription::new("sub-7", &old, now());
        let mid = now() + Duration::days(15);
        let proration = change_plan(&mut sub, &old, &new, mid).unwrap();
        assert_eq!(sub.plan_id, "enterprise");
        assert_eq!(proration.remaining_days, 15);
    }

    #[test]
    fn metered_usage_and_overage() {
        let plan = Plan::new("api", "API Plan", 5000, BillingInterval::Monthly, 0)
            .with_metered(MeteredConfig {
                included_units: 1000,
                overage_cost_cents: 5,
                unit_label: "API calls".to_string(),
            });
        let mut sub = Subscription::new("sub-8", &plan, now());
        sub.report_usage(800).unwrap();
        assert_eq!(compute_overage(&plan, sub.usage_units), 0);
        sub.report_usage(500).unwrap();
        assert_eq!(sub.usage_units, 1300);
        assert_eq!(compute_overage(&plan, sub.usage_units), 300 * 5);
    }

    #[test]
    fn renew_subscription() {
        let plan = basic_plan();
        let mut sub = Subscription::new("sub-9", &plan, now());
        let old_end = sub.current_period_end;
        sub.renew(&plan).unwrap();
        assert_eq!(sub.current_period_start, old_end);
        assert_eq!(
            sub.current_period_end,
            old_end + Duration::days(30)
        );
    }

    #[test]
    fn renew_cancelled_at_end() {
        let plan = basic_plan();
        let mut sub = Subscription::new("sub-10", &plan, now());
        sub.cancel_at_end().unwrap();
        let result = sub.renew(&plan);
        assert!(result.is_err());
        assert_eq!(sub.status, SubscriptionStatus::Cancelled);
    }

    #[test]
    fn store_subscribe_and_lookup() {
        let mut store = SubscriptionStore::new();
        store.add_plan(basic_plan());
        store.subscribe("sub-11", "pro", now()).unwrap();
        let sub = store.get_subscription("sub-11").unwrap();
        assert_eq!(sub.plan_id, "pro");
        assert_eq!(sub.status, SubscriptionStatus::Active);
    }

    #[test]
    fn store_plan_not_found() {
        let mut store = SubscriptionStore::new();
        let result = store.subscribe("sub-12", "nonexistent", now());
        assert!(matches!(result, Err(SubscriptionError::PlanNotFound(_))));
    }

    #[test]
    fn remaining_days() {
        let plan = basic_plan();
        let sub = Subscription::new("sub-13", &plan, now());
        let mid = now() + Duration::days(20);
        assert_eq!(sub.remaining_days(mid), 10);
    }

    #[test]
    fn daily_rate_yearly() {
        let plan = yearly_plan();
        let daily = plan.daily_rate_cents();
        assert!((daily - 29999.0 / 365.0).abs() < 0.01);
    }

    #[test]
    fn cannot_report_usage_when_cancelled() {
        let plan = basic_plan();
        let mut sub = Subscription::new("sub-14", &plan, now());
        sub.cancel_now().unwrap();
        let result = sub.report_usage(100);
        assert!(matches!(
            result,
            Err(SubscriptionError::CannotChangeDuringStatus(
                SubscriptionStatus::Cancelled
            ))
        ));
    }
}
