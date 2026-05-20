//! Invoice generation — line items, tax, totals, and structured text export.
//!
//! Replaces invoice libraries (jsPDF-Invoice, html-pdf) with a pure-Rust
//! domain model that can render to structured text or feed any template engine.

use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// Invoice domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvoiceError {
    /// No line items on the invoice.
    EmptyInvoice,
    /// Invoice is in a terminal state and cannot be modified.
    NotEditable(InvoiceStatus),
    /// Duplicate invoice number.
    DuplicateNumber(String),
    /// Invalid transition.
    InvalidTransition { from: InvoiceStatus, to: InvoiceStatus },
}

impl std::fmt::Display for InvoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInvoice => write!(f, "invoice has no line items"),
            Self::NotEditable(s) => write!(f, "invoice not editable in status {s:?}"),
            Self::DuplicateNumber(n) => write!(f, "duplicate invoice number: {n}"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from:?} to {to:?}")
            }
        }
    }
}

impl std::error::Error for InvoiceError {}

// ── Address ─────────────────────────────────────────────────────

/// A postal address for invoice parties.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Address {
    pub name: String,
    pub line1: String,
    pub line2: Option<String>,
    pub city: String,
    pub state: Option<String>,
    pub postal_code: String,
    pub country: String,
}

impl Address {
    pub fn new(
        name: impl Into<String>,
        line1: impl Into<String>,
        city: impl Into<String>,
        postal_code: impl Into<String>,
        country: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            line1: line1.into(),
            line2: None,
            city: city.into(),
            state: None,
            postal_code: postal_code.into(),
            country: country.into(),
        }
    }

    /// Format the address as multi-line text.
    pub fn format(&self) -> String {
        let mut lines = vec![self.name.clone(), self.line1.clone()];
        if let Some(l2) = &self.line2 {
            lines.push(l2.clone());
        }
        let mut city_line = self.city.clone();
        if let Some(st) = &self.state {
            city_line.push_str(", ");
            city_line.push_str(st);
        }
        city_line.push(' ');
        city_line.push_str(&self.postal_code);
        lines.push(city_line);
        lines.push(self.country.clone());
        lines.join("\n")
    }
}

// ── Line Item ───────────────────────────────────────────────────

/// A single line item on an invoice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem {
    pub description: String,
    pub quantity: f64,
    pub unit_price_cents: i64,
    /// Tax rate as a fraction (e.g. 0.07 for 7%).
    pub tax_rate: f64,
}

impl LineItem {
    pub fn new(
        description: impl Into<String>,
        quantity: f64,
        unit_price_cents: i64,
        tax_rate: f64,
    ) -> Self {
        Self {
            description: description.into(),
            quantity,
            unit_price_cents,
            tax_rate,
        }
    }

    /// Subtotal for this line item in cents (before tax).
    pub fn subtotal_cents(&self) -> i64 {
        (self.quantity * self.unit_price_cents as f64).round() as i64
    }

    /// Tax amount for this line item in cents.
    pub fn tax_cents(&self) -> i64 {
        (self.subtotal_cents() as f64 * self.tax_rate).round() as i64
    }

    /// Total including tax in cents.
    pub fn total_cents(&self) -> i64 {
        self.subtotal_cents() + self.tax_cents()
    }
}

// ── Invoice Status ──────────────────────────────────────────────

/// Lifecycle states for an invoice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InvoiceStatus {
    Draft,
    Sent,
    Paid,
    Overdue,
    Void,
}

impl InvoiceStatus {
    /// Valid next states.
    pub fn valid_transitions(self) -> &'static [InvoiceStatus] {
        match self {
            Self::Draft => &[Self::Sent, Self::Void],
            Self::Sent => &[Self::Paid, Self::Overdue, Self::Void],
            Self::Overdue => &[Self::Paid, Self::Void],
            Self::Paid => &[],
            Self::Void => &[],
        }
    }

    pub fn can_transition_to(self, next: Self) -> bool {
        self.valid_transitions().contains(&next)
    }

    /// Whether the invoice can be edited (add/remove items).
    pub fn is_editable(self) -> bool {
        self == Self::Draft
    }
}

// ── Invoice ─────────────────────────────────────────────────────

/// A full invoice document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    pub id: String,
    pub number: String,
    pub date: NaiveDate,
    pub due_date: NaiveDate,
    pub from: Address,
    pub to: Address,
    pub line_items: Vec<LineItem>,
    pub notes: Option<String>,
    pub status: InvoiceStatus,
    pub currency: String,
}

impl Invoice {
    /// Create a new draft invoice.
    pub fn new(
        id: impl Into<String>,
        number: impl Into<String>,
        date: NaiveDate,
        due_date: NaiveDate,
        from: Address,
        to: Address,
        currency: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            number: number.into(),
            date,
            due_date,
            from,
            to,
            line_items: Vec::new(),
            notes: None,
            status: InvoiceStatus::Draft,
            currency: currency.into().to_uppercase(),
        }
    }

    /// Add a line item. Only allowed in Draft status.
    pub fn add_item(&mut self, item: LineItem) -> Result<(), InvoiceError> {
        if !self.status.is_editable() {
            return Err(InvoiceError::NotEditable(self.status));
        }
        self.line_items.push(item);
        Ok(())
    }

    /// Remove a line item by index.
    pub fn remove_item(&mut self, index: usize) -> Result<LineItem, InvoiceError> {
        if !self.status.is_editable() {
            return Err(InvoiceError::NotEditable(self.status));
        }
        if index < self.line_items.len() {
            Ok(self.line_items.remove(index))
        } else {
            Err(InvoiceError::EmptyInvoice)
        }
    }

    /// Subtotal in cents (sum of line item subtotals).
    pub fn subtotal_cents(&self) -> i64 {
        self.line_items.iter().map(|li| li.subtotal_cents()).sum()
    }

    /// Total tax in cents.
    pub fn tax_cents(&self) -> i64 {
        self.line_items.iter().map(|li| li.tax_cents()).sum()
    }

    /// Grand total in cents.
    pub fn total_cents(&self) -> i64 {
        self.subtotal_cents() + self.tax_cents()
    }

    /// Transition to a new status.
    pub fn transition(&mut self, next: InvoiceStatus) -> Result<(), InvoiceError> {
        if !self.status.can_transition_to(next) {
            return Err(InvoiceError::InvalidTransition {
                from: self.status,
                to: next,
            });
        }
        if next == InvoiceStatus::Sent && self.line_items.is_empty() {
            return Err(InvoiceError::EmptyInvoice);
        }
        self.status = next;
        Ok(())
    }

    /// Check if the invoice is overdue relative to `today`.
    pub fn is_overdue(&self, today: NaiveDate) -> bool {
        self.status == InvoiceStatus::Sent && today > self.due_date
    }

    /// Set notes.
    pub fn set_notes(&mut self, notes: impl Into<String>) {
        self.notes = Some(notes.into());
    }

    /// Export as structured text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("INVOICE #{}\n", self.number));
        out.push_str(&format!("Date: {}\n", self.date));
        out.push_str(&format!("Due:  {}\n\n", self.due_date));

        out.push_str("From:\n");
        for line in self.from.format().lines() {
            out.push_str(&format!("  {line}\n"));
        }
        out.push('\n');

        out.push_str("To:\n");
        for line in self.to.format().lines() {
            out.push_str(&format!("  {line}\n"));
        }
        out.push('\n');

        out.push_str(&format!(
            "{:<40} {:>6} {:>10} {:>10}\n",
            "Description", "Qty", "Unit", "Total"
        ));
        out.push_str(&"-".repeat(70));
        out.push('\n');

        for item in &self.line_items {
            out.push_str(&format!(
                "{:<40} {:>6.1} {:>10} {:>10}\n",
                item.description,
                item.quantity,
                format_cents(item.unit_price_cents),
                format_cents(item.total_cents()),
            ));
        }

        out.push_str(&"-".repeat(70));
        out.push('\n');
        out.push_str(&format!("{:>58} {:>10}\n", "Subtotal:", format_cents(self.subtotal_cents())));
        out.push_str(&format!("{:>58} {:>10}\n", "Tax:", format_cents(self.tax_cents())));
        out.push_str(&format!("{:>58} {:>10}\n", "TOTAL:", format_cents(self.total_cents())));

        if let Some(notes) = &self.notes {
            out.push_str(&format!("\nNotes: {notes}\n"));
        }

        out
    }
}

/// Format cents as dollars string.
fn format_cents(cents: i64) -> String {
    let dollars = cents / 100;
    let remainder = (cents % 100).unsigned_abs();
    if cents < 0 {
        format!("-{}.{:02}", dollars.unsigned_abs(), remainder)
    } else {
        format!("{dollars}.{remainder:02}")
    }
}

// ── Sequential Number Generator ─────────────────────────────────

/// Generates sequential invoice numbers with a prefix.
#[derive(Debug)]
pub struct InvoiceNumberGenerator {
    prefix: String,
    next: u64,
    width: usize,
}

impl InvoiceNumberGenerator {
    /// Create a generator. Numbers will be like "INV-0001", "INV-0002", etc.
    pub fn new(prefix: impl Into<String>, start: u64, width: usize) -> Self {
        Self {
            prefix: prefix.into(),
            next: start,
            width,
        }
    }

    /// Get the next invoice number.
    pub fn next_number(&mut self) -> String {
        let n = self.next;
        self.next += 1;
        format!("{}-{:0>width$}", self.prefix, n, width = self.width)
    }

    /// Peek at the next number without consuming it.
    pub fn peek(&self) -> String {
        format!("{}-{:0>width$}", self.prefix, self.next, width = self.width)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_address(name: &str) -> Address {
        Address::new(name, "123 Main St", "Springfield", "62704", "US")
    }

    fn sample_invoice() -> Invoice {
        Invoice::new(
            "inv_1",
            "INV-0001",
            NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 3, 31).unwrap(),
            sample_address("Seller Co"),
            sample_address("Buyer LLC"),
            "USD",
        )
    }

    #[test]
    fn line_item_calculations() {
        let item = LineItem::new("Widget", 3.0, 1000, 0.07);
        assert_eq!(item.subtotal_cents(), 3000);
        assert_eq!(item.tax_cents(), 210);
        assert_eq!(item.total_cents(), 3210);
    }

    #[test]
    fn invoice_totals() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("A", 2.0, 500, 0.10)).unwrap();
        inv.add_item(LineItem::new("B", 1.0, 2000, 0.0)).unwrap();
        assert_eq!(inv.subtotal_cents(), 3000);
        assert_eq!(inv.tax_cents(), 100);
        assert_eq!(inv.total_cents(), 3100);
    }

    #[test]
    fn cannot_add_item_after_sent() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("X", 1.0, 100, 0.0)).unwrap();
        inv.transition(InvoiceStatus::Sent).unwrap();
        assert!(inv.add_item(LineItem::new("Y", 1.0, 200, 0.0)).is_err());
    }

    #[test]
    fn cannot_send_empty_invoice() {
        let mut inv = sample_invoice();
        assert!(inv.transition(InvoiceStatus::Sent).is_err());
    }

    #[test]
    fn status_transitions() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("Z", 1.0, 100, 0.0)).unwrap();
        inv.transition(InvoiceStatus::Sent).unwrap();
        inv.transition(InvoiceStatus::Paid).unwrap();
        assert_eq!(inv.status, InvoiceStatus::Paid);
    }

    #[test]
    fn overdue_check() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("Z", 1.0, 100, 0.0)).unwrap();
        inv.transition(InvoiceStatus::Sent).unwrap();
        assert!(!inv.is_overdue(NaiveDate::from_ymd_opt(2026, 3, 15).unwrap()));
        assert!(inv.is_overdue(NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()));
    }

    #[test]
    fn void_is_terminal() {
        let mut inv = sample_invoice();
        inv.transition(InvoiceStatus::Void).unwrap();
        assert!(inv.transition(InvoiceStatus::Sent).is_err());
    }

    #[test]
    fn sequential_numbering() {
        let mut numberer = InvoiceNumberGenerator::new("INV", 1, 4);
        assert_eq!(numberer.next_number(), "INV-0001");
        assert_eq!(numberer.next_number(), "INV-0002");
        assert_eq!(numberer.peek(), "INV-0003");
        assert_eq!(numberer.next_number(), "INV-0003");
    }

    #[test]
    fn text_export_contains_key_fields() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("Consulting", 10.0, 15000, 0.0)).unwrap();
        inv.set_notes("Net 30");
        let text = inv.to_text();
        assert!(text.contains("INVOICE #INV-0001"));
        assert!(text.contains("Consulting"));
        assert!(text.contains("1500.00")); // total
        assert!(text.contains("Net 30"));
    }

    #[test]
    fn address_formatting() {
        let mut addr = sample_address("ACME Inc");
        addr.state = Some("IL".into());
        addr.line2 = Some("Suite 100".into());
        let fmt = addr.format();
        assert!(fmt.contains("ACME Inc"));
        assert!(fmt.contains("Suite 100"));
        assert!(fmt.contains("IL"));
    }

    #[test]
    fn remove_line_item() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("A", 1.0, 100, 0.0)).unwrap();
        inv.add_item(LineItem::new("B", 1.0, 200, 0.0)).unwrap();
        let removed = inv.remove_item(0).unwrap();
        assert_eq!(removed.description, "A");
        assert_eq!(inv.line_items.len(), 1);
    }

    #[test]
    fn serialization_round_trip() {
        let mut inv = sample_invoice();
        inv.add_item(LineItem::new("Test", 1.0, 999, 0.05)).unwrap();
        let json = serde_json::to_string(&inv).unwrap();
        let restored: Invoice = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.total_cents(), inv.total_cents());
    }

    #[test]
    fn format_cents_negative() {
        assert_eq!(format_cents(-150), "-1.50");
        assert_eq!(format_cents(0), "0.00");
        assert_eq!(format_cents(99), "0.99");
    }
}
