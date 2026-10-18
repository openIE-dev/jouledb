//! Shared domain types for cross-crate bridges.
//!
//! These types are used by multiple productivity domain crates to communicate
//! without creating circular dependencies. Each domain crate can convert
//! between its internal types and these canonical shared types.

use serde::{Deserialize, Serialize};

// ── Delivery / Notification ────────────────────────────────────────────────

/// Notification delivery channel (shared by joule-notify, joule-email).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeliveryChannel {
    InApp,
    Push,
    Email,
    Sms,
    Webhook,
}

/// Source module that emitted an activity or event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceModule {
    Calendar,
    Commerce,
    Crm,
    Email,
    Finance,
    Forms,
    Hr,
    Inventory,
    Learn,
    Notify,
    Pay,
    Recruit,
    Security,
    Sign,
    Social,
    Support,
    Survey,
    Tasks,
    Wiki,
    Custom(String),
}

/// Kind of activity (shared by joule-crm, joule-notify).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivityKind {
    Call,
    Email,
    Meeting,
    Note,
    Task,
}

// ── Employment / HR / Recruit ──────────────────────────────────────────────

/// Employment type (shared by joule-hr, joule-recruit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmploymentType {
    FullTime,
    PartTime,
    Contract,
    Internship,
    Temporary,
}

/// Pay period for salary ranges (shared by joule-hr, joule-recruit, joule-finance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayPeriod {
    Annual,
    Monthly,
    Hourly,
}

/// Salary range in cents (shared by joule-hr, joule-recruit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SalaryRange {
    pub min_cents: i64,
    pub max_cents: i64,
    pub period: PayPeriod,
}

// ── Finance / Pay ──────────────────────────────────────────────────────────

// ── Email / Contact ────────────────────────────────────────────────────────

/// Email address with optional display name (shared by joule-email, joule-notify, joule-crm).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmailAddress {
    pub name: Option<String>,
    pub address: String,
}

impl EmailAddress {
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            name: None,
            address: address.into(),
        }
    }

    pub fn with_name(name: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            address: address.into(),
        }
    }
}

impl std::fmt::Display for EmailAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(name) => write!(f, "{} <{}>", name, self.address),
            None => write!(f, "{}", self.address),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_address_display() {
        let plain = EmailAddress::new("a@b.com");
        assert_eq!(format!("{}", plain), "a@b.com");

        let named = EmailAddress::with_name("Alice", "a@b.com");
        assert_eq!(format!("{}", named), "Alice <a@b.com>");
    }

    #[test]
    fn salary_range_serde() {
        let r = SalaryRange {
            min_cents: 80_000_00,
            max_cents: 120_000_00,
            period: PayPeriod::Annual,
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: SalaryRange = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.min_cents, r.min_cents);
        assert_eq!(parsed.period, PayPeriod::Annual);
    }

    #[test]
    fn delivery_channel_variants() {
        let channels = [
            DeliveryChannel::InApp,
            DeliveryChannel::Push,
            DeliveryChannel::Email,
            DeliveryChannel::Sms,
            DeliveryChannel::Webhook,
        ];
        assert_eq!(channels.len(), 5);
    }

    #[test]
    fn source_module_custom() {
        let m = SourceModule::Custom("my-plugin".into());
        assert_eq!(m, SourceModule::Custom("my-plugin".into()));
    }
}
