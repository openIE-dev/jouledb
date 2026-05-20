//! Tax calculation — rates, rules, VAT reverse charge, exemptions, and rounding.
//!
//! Replaces tax libraries (TaxJar, Avalara SDK) with a pure-Rust tax engine
//! supporting multi-jurisdiction rules, inclusive/exclusive pricing, and exemptions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Tax domain errors.
#[derive(Debug, Clone, PartialEq)]
pub enum TaxError {
    /// Tax rate out of valid range.
    InvalidRate(f64),
    /// No applicable rule found.
    NoApplicableRule { category: String, region: String },
}

impl std::fmt::Display for TaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRate(r) => write!(f, "invalid tax rate: {r}"),
            Self::NoApplicableRule { category, region } => {
                write!(f, "no tax rule for category={category}, region={region}")
            }
        }
    }
}

impl std::error::Error for TaxError {}

// ── Tax Rate ────────────────────────────────────────────────────

/// A named tax rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxRate {
    /// Rate as a fraction (e.g. 0.07 for 7%).
    pub rate: f64,
    /// Human-readable name (e.g. "FL Sales Tax").
    pub name: String,
    /// Jurisdiction identifier (e.g. "US-FL", "DE", "GB").
    pub jurisdiction: String,
}

impl TaxRate {
    pub fn new(rate: f64, name: impl Into<String>, jurisdiction: impl Into<String>) -> Result<Self, TaxError> {
        if rate < 0.0 || rate > 1.0 {
            return Err(TaxError::InvalidRate(rate));
        }
        Ok(Self {
            rate,
            name: name.into(),
            jurisdiction: jurisdiction.into(),
        })
    }

    /// Rate as a percentage (e.g. 7.0 for 7%).
    pub fn percentage(&self) -> f64 {
        self.rate * 100.0
    }
}

// ── Tax Rule ────────────────────────────────────────────────────

/// A rule matching a product category and region to a tax rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxRule {
    /// Product category this rule applies to (e.g. "physical", "digital", "food").
    pub category: String,
    /// Region code (e.g. "US-FL", "DE"). Empty string means "default".
    pub region: String,
    /// The tax rate to apply.
    pub tax_rate: TaxRate,
}

impl TaxRule {
    pub fn new(category: impl Into<String>, region: impl Into<String>, tax_rate: TaxRate) -> Self {
        Self {
            category: category.into(),
            region: region.into(),
            tax_rate,
        }
    }

    /// Check if this rule matches the given category and region.
    pub fn matches(&self, category: &str, region: &str) -> bool {
        self.category == category && (self.region == region || self.region.is_empty())
    }
}

// ── Pricing Mode ────────────────────────────────────────────────

/// Whether prices include or exclude tax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingMode {
    /// Prices are exclusive of tax — tax is added on top.
    TaxExclusive,
    /// Prices already include tax — tax is extracted.
    TaxInclusive,
}

// ── Tax Exemption Certificate ───────────────────────────────────

/// A certificate that exempts a buyer from tax.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxExemptionCertificate {
    pub id: String,
    pub holder_name: String,
    /// Jurisdictions covered by this exemption.
    pub jurisdictions: Vec<String>,
    /// Product categories covered.
    pub categories: Vec<String>,
    pub valid: bool,
}

impl TaxExemptionCertificate {
    /// Check if this certificate exempts the given category in the given jurisdiction.
    pub fn covers(&self, category: &str, jurisdiction: &str) -> bool {
        self.valid
            && (self.categories.is_empty() || self.categories.iter().any(|c| c == category))
            && (self.jurisdictions.is_empty() || self.jurisdictions.iter().any(|j| j == jurisdiction))
    }
}

// ── Taxable Item ────────────────────────────────────────────────

/// An item for tax calculation.
#[derive(Debug, Clone)]
pub struct TaxableItem {
    pub description: String,
    pub amount_cents: i64,
    pub category: String,
    pub quantity: f64,
}

impl TaxableItem {
    pub fn new(
        description: impl Into<String>,
        amount_cents: i64,
        category: impl Into<String>,
        quantity: f64,
    ) -> Self {
        Self {
            description: description.into(),
            amount_cents,
            category: category.into(),
            quantity,
        }
    }

    /// Total before tax.
    pub fn line_total_cents(&self) -> i64 {
        (self.amount_cents as f64 * self.quantity).round() as i64
    }
}

// ── Tax Line ────────────────────────────────────────────────────

/// Result of computing tax for a single item.
#[derive(Debug, Clone)]
pub struct TaxLine {
    pub description: String,
    pub subtotal_cents: i64,
    pub tax_cents: i64,
    pub tax_rate_name: String,
    pub tax_rate: f64,
    pub jurisdiction: String,
}

// ── Tax Summary ─────────────────────────────────────────────────

/// Aggregated tax grouped by rate.
#[derive(Debug, Clone, Default)]
pub struct TaxSummary {
    /// Map of (rate_name, jurisdiction) -> total tax cents.
    pub by_rate: HashMap<(String, String), TaxSummaryEntry>,
    pub total_tax_cents: i64,
    pub total_subtotal_cents: i64,
}

/// A single entry in the summary.
#[derive(Debug, Clone)]
pub struct TaxSummaryEntry {
    pub rate_name: String,
    pub jurisdiction: String,
    pub rate: f64,
    pub taxable_amount_cents: i64,
    pub tax_cents: i64,
}

// ── Tax Calculator ──────────────────────────────────────────────

/// Configurable tax calculator with rules and exemptions.
#[derive(Debug, Clone)]
pub struct TaxCalculator {
    rules: Vec<TaxRule>,
    exemptions: Vec<TaxExemptionCertificate>,
    pricing_mode: PricingMode,
    /// If true, B2B intra-EU transactions reverse charge VAT.
    pub vat_reverse_charge: bool,
}

impl TaxCalculator {
    pub fn new(pricing_mode: PricingMode) -> Self {
        Self {
            rules: Vec::new(),
            exemptions: Vec::new(),
            pricing_mode,
            vat_reverse_charge: false,
        }
    }

    /// Add a tax rule.
    pub fn add_rule(&mut self, rule: TaxRule) {
        self.rules.push(rule);
    }

    /// Add a tax exemption certificate.
    pub fn add_exemption(&mut self, cert: TaxExemptionCertificate) {
        self.exemptions.push(cert);
    }

    /// Find the best matching rule for a category and region.
    fn find_rule(&self, category: &str, region: &str) -> Option<&TaxRule> {
        // Exact match first, then default region.
        self.rules
            .iter()
            .find(|r| r.category == category && r.region == region)
            .or_else(|| {
                self.rules
                    .iter()
                    .find(|r| r.category == category && r.region.is_empty())
            })
    }

    /// Check if any exemption covers this item.
    fn is_exempt(&self, category: &str, jurisdiction: &str) -> bool {
        self.exemptions
            .iter()
            .any(|cert| cert.covers(category, jurisdiction))
    }

    /// Compute tax for a single item in a given region.
    pub fn compute_item_tax(
        &self,
        item: &TaxableItem,
        region: &str,
    ) -> TaxLine {
        let rule = self.find_rule(&item.category, region);
        let line_total = item.line_total_cents();

        let (rate, rate_name, jurisdiction) = match rule {
            Some(r) => {
                if self.is_exempt(&item.category, &r.tax_rate.jurisdiction) {
                    (0.0, "Exempt".to_string(), r.tax_rate.jurisdiction.clone())
                } else if self.vat_reverse_charge {
                    (0.0, "Reverse Charge".to_string(), r.tax_rate.jurisdiction.clone())
                } else {
                    (r.tax_rate.rate, r.tax_rate.name.clone(), r.tax_rate.jurisdiction.clone())
                }
            }
            None => (0.0, "None".to_string(), region.to_string()),
        };

        let (subtotal_cents, tax_cents) = match self.pricing_mode {
            PricingMode::TaxExclusive => {
                let tax = round_half_up(line_total as f64 * rate);
                (line_total, tax)
            }
            PricingMode::TaxInclusive => {
                // Extract tax from inclusive price: tax = price - price / (1 + rate)
                let net = round_half_up(line_total as f64 / (1.0 + rate));
                let tax = line_total - net;
                (net, tax)
            }
        };

        TaxLine {
            description: item.description.clone(),
            subtotal_cents,
            tax_cents,
            tax_rate_name: rate_name,
            tax_rate: rate,
            jurisdiction,
        }
    }

    /// Compute tax for multiple items and return a summary.
    pub fn compute_tax(
        &self,
        items: &[TaxableItem],
        region: &str,
    ) -> (Vec<TaxLine>, TaxSummary) {
        let mut lines = Vec::new();
        let mut summary = TaxSummary::default();

        for item in items {
            let line = self.compute_item_tax(item, region);

            let key = (line.tax_rate_name.clone(), line.jurisdiction.clone());
            let entry = summary.by_rate.entry(key.clone()).or_insert_with(|| {
                TaxSummaryEntry {
                    rate_name: line.tax_rate_name.clone(),
                    jurisdiction: line.jurisdiction.clone(),
                    rate: line.tax_rate,
                    taxable_amount_cents: 0,
                    tax_cents: 0,
                }
            });
            entry.taxable_amount_cents += line.subtotal_cents;
            entry.tax_cents += line.tax_cents;

            summary.total_subtotal_cents += line.subtotal_cents;
            summary.total_tax_cents += line.tax_cents;

            lines.push(line);
        }

        (lines, summary)
    }
}

/// Round to the nearest cent using half-up rounding.
fn round_half_up(value: f64) -> i64 {
    (value + 0.5f64.copysign(value) * f64::EPSILON).round() as i64
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sales_tax_rule(category: &str, region: &str, rate: f64, name: &str) -> TaxRule {
        TaxRule::new(
            category,
            region,
            TaxRate::new(rate, name, region).unwrap(),
        )
    }

    #[test]
    fn tax_rate_validation() {
        assert!(TaxRate::new(0.07, "Sales Tax", "US-FL").is_ok());
        assert!(TaxRate::new(-0.01, "Bad", "XX").is_err());
        assert!(TaxRate::new(1.5, "Bad", "XX").is_err());
    }

    #[test]
    fn tax_rate_percentage() {
        let rate = TaxRate::new(0.07, "FL Sales Tax", "US-FL").unwrap();
        assert!((rate.percentage() - 7.0).abs() < 0.001);
    }

    #[test]
    fn exclusive_tax_calculation() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("physical", "US-FL", 0.07, "FL Sales Tax"));

        let item = TaxableItem::new("Widget", 1000, "physical", 1.0);
        let line = calc.compute_item_tax(&item, "US-FL");
        assert_eq!(line.subtotal_cents, 1000);
        assert_eq!(line.tax_cents, 70);
    }

    #[test]
    fn inclusive_tax_calculation() {
        let mut calc = TaxCalculator::new(PricingMode::TaxInclusive);
        calc.add_rule(sales_tax_rule("physical", "DE", 0.19, "DE MwSt"));

        let item = TaxableItem::new("Widget", 11900, "physical", 1.0);
        let line = calc.compute_item_tax(&item, "DE");
        assert_eq!(line.subtotal_cents, 10000);
        assert_eq!(line.tax_cents, 1900);
    }

    #[test]
    fn no_rule_means_no_tax() {
        let calc = TaxCalculator::new(PricingMode::TaxExclusive);
        let item = TaxableItem::new("Mystery", 5000, "unknown", 1.0);
        let line = calc.compute_item_tax(&item, "XX");
        assert_eq!(line.tax_cents, 0);
    }

    #[test]
    fn default_region_fallback() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("digital", "", 0.10, "Default Digital Tax"));

        let item = TaxableItem::new("E-book", 999, "digital", 1.0);
        let line = calc.compute_item_tax(&item, "US-CA");
        assert_eq!(line.tax_cents, 100); // 10% of 999 rounded
    }

    #[test]
    fn exemption_certificate() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("physical", "US-FL", 0.07, "FL Sales Tax"));
        calc.add_exemption(TaxExemptionCertificate {
            id: "cert_1".into(),
            holder_name: "Nonprofit Inc".into(),
            jurisdictions: vec!["US-FL".into()],
            categories: vec!["physical".into()],
            valid: true,
        });

        let item = TaxableItem::new("Equipment", 50000, "physical", 1.0);
        let line = calc.compute_item_tax(&item, "US-FL");
        assert_eq!(line.tax_cents, 0);
        assert_eq!(line.tax_rate_name, "Exempt");
    }

    #[test]
    fn vat_reverse_charge() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("digital", "DE", 0.19, "DE VAT"));
        calc.vat_reverse_charge = true;

        let item = TaxableItem::new("SaaS", 10000, "digital", 1.0);
        let line = calc.compute_item_tax(&item, "DE");
        assert_eq!(line.tax_cents, 0);
        assert_eq!(line.tax_rate_name, "Reverse Charge");
    }

    #[test]
    fn tax_summary_groups_by_rate() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("physical", "US-FL", 0.07, "FL Sales Tax"));
        calc.add_rule(sales_tax_rule("food", "US-FL", 0.0, "Food Exempt"));

        let items = vec![
            TaxableItem::new("Widget", 1000, "physical", 2.0),
            TaxableItem::new("Gadget", 2000, "physical", 1.0),
            TaxableItem::new("Bread", 500, "food", 1.0),
        ];

        let (lines, summary) = calc.compute_tax(&items, "US-FL");
        assert_eq!(lines.len(), 3);
        assert_eq!(summary.total_tax_cents, 280); // 7% of 4000
        assert_eq!(summary.total_subtotal_cents, 4500);
    }

    #[test]
    fn quantity_multiplication() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("physical", "US-TX", 0.0825, "TX Sales Tax"));

        let item = TaxableItem::new("Book", 1299, "physical", 3.0);
        let line = calc.compute_item_tax(&item, "US-TX");
        assert_eq!(line.subtotal_cents, 3897);
        // 3897 * 0.0825 = 321.5025, rounds to 322
        assert_eq!(line.tax_cents, 322);
    }

    #[test]
    fn invalid_exemption_not_applied() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("physical", "US-FL", 0.07, "FL Tax"));
        calc.add_exemption(TaxExemptionCertificate {
            id: "cert_expired".into(),
            holder_name: "Ex Corp".into(),
            jurisdictions: vec!["US-FL".into()],
            categories: vec!["physical".into()],
            valid: false,
        });

        let item = TaxableItem::new("Stuff", 10000, "physical", 1.0);
        let line = calc.compute_item_tax(&item, "US-FL");
        assert_eq!(line.tax_cents, 700);
    }

    #[test]
    fn rounding_half_up() {
        // 9.99 * 0.07 = 0.6993 -> round to 70 cents
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("physical", "US", 0.07, "Tax"));

        let item = TaxableItem::new("Item", 999, "physical", 1.0);
        let line = calc.compute_item_tax(&item, "US");
        assert_eq!(line.tax_cents, 70);
    }

    #[test]
    fn exemption_covers_specific_category() {
        let cert = TaxExemptionCertificate {
            id: "c1".into(),
            holder_name: "Org".into(),
            jurisdictions: vec!["US-FL".into()],
            categories: vec!["food".into()],
            valid: true,
        };
        assert!(cert.covers("food", "US-FL"));
        assert!(!cert.covers("physical", "US-FL"));
        assert!(!cert.covers("food", "US-TX"));
    }

    #[test]
    fn exact_region_over_default() {
        let mut calc = TaxCalculator::new(PricingMode::TaxExclusive);
        calc.add_rule(sales_tax_rule("digital", "", 0.10, "Default"));
        calc.add_rule(sales_tax_rule("digital", "US-CA", 0.0725, "CA Tax"));

        let item = TaxableItem::new("App", 1000, "digital", 1.0);
        let line = calc.compute_item_tax(&item, "US-CA");
        assert_eq!(line.tax_rate_name, "CA Tax");
        assert_eq!(line.tax_cents, 73); // 7.25% of 1000
    }
}
