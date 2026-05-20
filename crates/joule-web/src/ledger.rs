//! Double-entry ledger — accounts (asset/liability/equity/revenue/expense),
//! journal entries, trial balance, income statement, balance sheet, account
//! hierarchy, and posting validation (debits = credits).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from ledger operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    /// Journal entry debits do not equal credits.
    UnbalancedEntry { debits: i64, credits: i64 },
    /// Account not found.
    AccountNotFound(String),
    /// Duplicate account code.
    DuplicateAccount(String),
    /// Entry references a non-existent account.
    InvalidAccountRef(String),
    /// Amount must be positive.
    InvalidAmount(i64),
    /// Entry has no line items.
    EmptyEntry,
    /// Parent account not found.
    ParentNotFound(String),
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnbalancedEntry { debits, credits } => {
                write!(f, "unbalanced entry: debits={debits}, credits={credits}")
            }
            Self::AccountNotFound(code) => write!(f, "account not found: {code}"),
            Self::DuplicateAccount(code) => write!(f, "duplicate account: {code}"),
            Self::InvalidAccountRef(code) => write!(f, "invalid account reference: {code}"),
            Self::InvalidAmount(amt) => write!(f, "invalid amount: {amt}"),
            Self::EmptyEntry => write!(f, "entry has no line items"),
            Self::ParentNotFound(code) => write!(f, "parent account not found: {code}"),
        }
    }
}

impl std::error::Error for LedgerError {}

// ── Account Types ───────────────────────────────────────────────────────────

/// The five fundamental account types in double-entry accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccountType {
    Asset,
    Liability,
    Equity,
    Revenue,
    Expense,
}

impl AccountType {
    /// Whether this account type normally carries a debit balance.
    pub fn is_debit_normal(&self) -> bool {
        matches!(self, AccountType::Asset | AccountType::Expense)
    }

    /// Whether this account type normally carries a credit balance.
    pub fn is_credit_normal(&self) -> bool {
        !self.is_debit_normal()
    }
}

impl fmt::Display for AccountType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asset => write!(f, "Asset"),
            Self::Liability => write!(f, "Liability"),
            Self::Equity => write!(f, "Equity"),
            Self::Revenue => write!(f, "Revenue"),
            Self::Expense => write!(f, "Expense"),
        }
    }
}

// ── Account ─────────────────────────────────────────────────────────────────

/// An account in the chart of accounts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// Unique account code (e.g., "1000", "2100").
    pub code: String,
    /// Human-readable name.
    pub name: String,
    /// Account classification.
    pub account_type: AccountType,
    /// Optional parent account code for hierarchy.
    pub parent_code: Option<String>,
    /// Running debit total (in minor units, e.g. cents).
    pub total_debits: i64,
    /// Running credit total (in minor units).
    pub total_credits: i64,
}

impl Account {
    /// Create a new account.
    pub fn new(
        code: impl Into<String>,
        name: impl Into<String>,
        account_type: AccountType,
    ) -> Self {
        Self {
            code: code.into(),
            name: name.into(),
            account_type,
            parent_code: None,
            total_debits: 0,
            total_credits: 0,
        }
    }

    /// Set a parent account for hierarchy.
    pub fn with_parent(mut self, parent_code: impl Into<String>) -> Self {
        self.parent_code = Some(parent_code.into());
        self
    }

    /// Current balance, computed based on normal balance direction.
    /// Debit-normal accounts: debits - credits.
    /// Credit-normal accounts: credits - debits.
    pub fn balance(&self) -> i64 {
        if self.account_type.is_debit_normal() {
            self.total_debits - self.total_credits
        } else {
            self.total_credits - self.total_debits
        }
    }
}

// ── Journal Entry ───────────────────────────────────────────────────────────

/// A single line item within a journal entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LineItem {
    /// Account code to post to.
    pub account_code: String,
    /// Debit amount (minor units). Zero if this is a credit line.
    pub debit: i64,
    /// Credit amount (minor units). Zero if this is a debit line.
    pub credit: i64,
}

impl LineItem {
    /// Create a debit line item.
    pub fn debit(account_code: impl Into<String>, amount: i64) -> Self {
        Self {
            account_code: account_code.into(),
            debit: amount,
            credit: 0,
        }
    }

    /// Create a credit line item.
    pub fn credit(account_code: impl Into<String>, amount: i64) -> Self {
        Self {
            account_code: account_code.into(),
            debit: 0,
            credit: amount,
        }
    }
}

/// A complete journal entry with multiple line items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Sequential entry ID.
    pub id: u64,
    /// Description/memo.
    pub description: String,
    /// Unix timestamp.
    pub timestamp: u64,
    /// Line items (must balance: sum debits = sum credits).
    pub lines: Vec<LineItem>,
}

impl JournalEntry {
    /// Total debits in this entry.
    pub fn total_debits(&self) -> i64 {
        self.lines.iter().map(|l| l.debit).sum()
    }

    /// Total credits in this entry.
    pub fn total_credits(&self) -> i64 {
        self.lines.iter().map(|l| l.credit).sum()
    }

    /// Whether this entry is balanced.
    pub fn is_balanced(&self) -> bool {
        self.total_debits() == self.total_credits()
    }
}

// ── Trial Balance ───────────────────────────────────────────────────────────

/// A row in the trial balance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialBalanceRow {
    pub account_code: String,
    pub account_name: String,
    pub account_type: AccountType,
    pub debit_balance: i64,
    pub credit_balance: i64,
}

/// Complete trial balance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialBalance {
    pub rows: Vec<TrialBalanceRow>,
    pub total_debits: i64,
    pub total_credits: i64,
}

impl TrialBalance {
    /// Whether total debits equal total credits.
    pub fn is_balanced(&self) -> bool {
        self.total_debits == self.total_credits
    }
}

// ── Financial Statements ────────────────────────────────────────────────────

/// Income statement (Revenue - Expenses = Net Income).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomeStatement {
    pub revenue_items: Vec<(String, i64)>,
    pub expense_items: Vec<(String, i64)>,
    pub total_revenue: i64,
    pub total_expenses: i64,
    pub net_income: i64,
}

/// Balance sheet (Assets = Liabilities + Equity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceSheet {
    pub asset_items: Vec<(String, i64)>,
    pub liability_items: Vec<(String, i64)>,
    pub equity_items: Vec<(String, i64)>,
    pub total_assets: i64,
    pub total_liabilities: i64,
    pub total_equity: i64,
    /// Whether A = L + E holds.
    pub is_balanced: bool,
}

// ── Ledger ──────────────────────────────────────────────────────────────────

/// The double-entry ledger engine.
#[derive(Debug, Clone)]
pub struct Ledger {
    /// Chart of accounts, keyed by account code.
    accounts: HashMap<String, Account>,
    /// Posted journal entries in order.
    entries: Vec<JournalEntry>,
    /// Next entry ID.
    next_id: u64,
}

impl Ledger {
    /// Create a new empty ledger.
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            entries: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an account to the chart of accounts.
    pub fn add_account(&mut self, account: Account) -> Result<(), LedgerError> {
        if self.accounts.contains_key(&account.code) {
            return Err(LedgerError::DuplicateAccount(account.code.clone()));
        }
        if let Some(parent) = &account.parent_code {
            if !self.accounts.contains_key(parent) {
                return Err(LedgerError::ParentNotFound(parent.clone()));
            }
        }
        self.accounts.insert(account.code.clone(), account);
        Ok(())
    }

    /// Get an account by its code.
    pub fn get_account(&self, code: &str) -> Option<&Account> {
        self.accounts.get(code)
    }

    /// Get the number of accounts.
    pub fn account_count(&self) -> usize {
        self.accounts.len()
    }

    /// Get the number of posted entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get all children of a given parent account code.
    pub fn children_of(&self, parent_code: &str) -> Vec<&Account> {
        self.accounts
            .values()
            .filter(|a| a.parent_code.as_deref() == Some(parent_code))
            .collect()
    }

    /// Post a journal entry. Validates balance and account existence.
    pub fn post_entry(
        &mut self,
        description: impl Into<String>,
        timestamp: u64,
        lines: Vec<LineItem>,
    ) -> Result<u64, LedgerError> {
        if lines.is_empty() {
            return Err(LedgerError::EmptyEntry);
        }

        // Validate amounts
        for line in &lines {
            if line.debit < 0 {
                return Err(LedgerError::InvalidAmount(line.debit));
            }
            if line.credit < 0 {
                return Err(LedgerError::InvalidAmount(line.credit));
            }
        }

        // Validate account references
        for line in &lines {
            if !self.accounts.contains_key(&line.account_code) {
                return Err(LedgerError::InvalidAccountRef(line.account_code.clone()));
            }
        }

        let total_debits: i64 = lines.iter().map(|l| l.debit).sum();
        let total_credits: i64 = lines.iter().map(|l| l.credit).sum();

        if total_debits != total_credits {
            return Err(LedgerError::UnbalancedEntry {
                debits: total_debits,
                credits: total_credits,
            });
        }

        // Post to accounts
        for line in &lines {
            let account = self.accounts.get_mut(&line.account_code).unwrap();
            account.total_debits += line.debit;
            account.total_credits += line.credit;
        }

        let id = self.next_id;
        self.next_id += 1;

        let entry = JournalEntry {
            id,
            description: description.into(),
            timestamp,
            lines,
        };
        self.entries.push(entry);
        Ok(id)
    }

    /// Generate a trial balance.
    pub fn trial_balance(&self) -> TrialBalance {
        let mut rows = Vec::new();
        let mut total_debits = 0i64;
        let mut total_credits = 0i64;

        // Sort by account code for deterministic output
        let mut codes: Vec<&String> = self.accounts.keys().collect();
        codes.sort();

        for code in codes {
            let acct = &self.accounts[code];
            let balance = acct.balance();
            let (db, cr) = if acct.account_type.is_debit_normal() {
                if balance >= 0 {
                    (balance, 0i64)
                } else {
                    (0i64, -balance)
                }
            } else if balance >= 0 {
                (0i64, balance)
            } else {
                (-balance, 0i64)
            };

            total_debits += db;
            total_credits += cr;

            rows.push(TrialBalanceRow {
                account_code: acct.code.clone(),
                account_name: acct.name.clone(),
                account_type: acct.account_type,
                debit_balance: db,
                credit_balance: cr,
            });
        }

        TrialBalance {
            rows,
            total_debits,
            total_credits,
        }
    }

    /// Generate an income statement.
    pub fn income_statement(&self) -> IncomeStatement {
        let mut revenue_items = Vec::new();
        let mut expense_items = Vec::new();

        let mut codes: Vec<&String> = self.accounts.keys().collect();
        codes.sort();

        for code in codes {
            let acct = &self.accounts[code];
            let balance = acct.balance();
            match acct.account_type {
                AccountType::Revenue => {
                    revenue_items.push((acct.name.clone(), balance));
                }
                AccountType::Expense => {
                    expense_items.push((acct.name.clone(), balance));
                }
                _ => {}
            }
        }

        let total_revenue: i64 = revenue_items.iter().map(|(_, b)| *b).sum();
        let total_expenses: i64 = expense_items.iter().map(|(_, b)| *b).sum();

        IncomeStatement {
            revenue_items,
            expense_items,
            total_revenue,
            total_expenses,
            net_income: total_revenue - total_expenses,
        }
    }

    /// Generate a balance sheet.
    pub fn balance_sheet(&self) -> BalanceSheet {
        let mut asset_items = Vec::new();
        let mut liability_items = Vec::new();
        let mut equity_items = Vec::new();

        let mut codes: Vec<&String> = self.accounts.keys().collect();
        codes.sort();

        for code in codes {
            let acct = &self.accounts[code];
            let balance = acct.balance();
            match acct.account_type {
                AccountType::Asset => asset_items.push((acct.name.clone(), balance)),
                AccountType::Liability => liability_items.push((acct.name.clone(), balance)),
                AccountType::Equity => equity_items.push((acct.name.clone(), balance)),
                _ => {}
            }
        }

        let total_assets: i64 = asset_items.iter().map(|(_, b)| *b).sum();
        let total_liabilities: i64 = liability_items.iter().map(|(_, b)| *b).sum();
        let total_equity: i64 = equity_items.iter().map(|(_, b)| *b).sum();

        BalanceSheet {
            asset_items,
            liability_items,
            equity_items,
            total_assets,
            total_liabilities,
            total_equity,
            is_balanced: total_assets == total_liabilities + total_equity,
        }
    }

    /// Get all entries that touched a given account.
    pub fn entries_for_account(&self, code: &str) -> Vec<&JournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.lines.iter().any(|l| l.account_code == code))
            .collect()
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_basic_ledger() -> Ledger {
        let mut ledger = Ledger::new();
        ledger.add_account(Account::new("1000", "Cash", AccountType::Asset)).unwrap();
        ledger.add_account(Account::new("2000", "Accounts Payable", AccountType::Liability)).unwrap();
        ledger.add_account(Account::new("3000", "Retained Earnings", AccountType::Equity)).unwrap();
        ledger.add_account(Account::new("4000", "Sales Revenue", AccountType::Revenue)).unwrap();
        ledger.add_account(Account::new("5000", "Rent Expense", AccountType::Expense)).unwrap();
        ledger
    }

    #[test]
    fn test_add_account() {
        let mut ledger = Ledger::new();
        ledger.add_account(Account::new("1000", "Cash", AccountType::Asset)).unwrap();
        assert_eq!(ledger.account_count(), 1);
    }

    #[test]
    fn test_duplicate_account_error() {
        let mut ledger = Ledger::new();
        ledger.add_account(Account::new("1000", "Cash", AccountType::Asset)).unwrap();
        let err = ledger.add_account(Account::new("1000", "Cash 2", AccountType::Asset)).unwrap_err();
        assert_eq!(err, LedgerError::DuplicateAccount("1000".to_string()));
    }

    #[test]
    fn test_post_balanced_entry() {
        let mut ledger = setup_basic_ledger();
        let id = ledger.post_entry(
            "Cash sale",
            1000,
            vec![
                LineItem::debit("1000", 5000),
                LineItem::credit("4000", 5000),
            ],
        ).unwrap();
        assert_eq!(id, 1);
        assert_eq!(ledger.entry_count(), 1);
    }

    #[test]
    fn test_unbalanced_entry_error() {
        let mut ledger = setup_basic_ledger();
        let err = ledger.post_entry(
            "Bad entry",
            1000,
            vec![
                LineItem::debit("1000", 5000),
                LineItem::credit("4000", 3000),
            ],
        ).unwrap_err();
        assert_eq!(err, LedgerError::UnbalancedEntry { debits: 5000, credits: 3000 });
    }

    #[test]
    fn test_empty_entry_error() {
        let mut ledger = setup_basic_ledger();
        let err = ledger.post_entry("Empty", 1000, vec![]).unwrap_err();
        assert_eq!(err, LedgerError::EmptyEntry);
    }

    #[test]
    fn test_invalid_account_ref() {
        let mut ledger = setup_basic_ledger();
        let err = ledger.post_entry(
            "Bad ref",
            1000,
            vec![
                LineItem::debit("9999", 100),
                LineItem::credit("4000", 100),
            ],
        ).unwrap_err();
        assert_eq!(err, LedgerError::InvalidAccountRef("9999".to_string()));
    }

    #[test]
    fn test_negative_amount_error() {
        let mut ledger = setup_basic_ledger();
        let err = ledger.post_entry(
            "Negative",
            1000,
            vec![
                LineItem::debit("1000", -100),
                LineItem::credit("4000", -100),
            ],
        ).unwrap_err();
        assert_eq!(err, LedgerError::InvalidAmount(-100));
    }

    #[test]
    fn test_account_balance() {
        let mut ledger = setup_basic_ledger();
        ledger.post_entry("Sale", 1000, vec![
            LineItem::debit("1000", 10000),
            LineItem::credit("4000", 10000),
        ]).unwrap();
        let cash = ledger.get_account("1000").unwrap();
        assert_eq!(cash.balance(), 10000); // Asset: debit-normal
        let revenue = ledger.get_account("4000").unwrap();
        assert_eq!(revenue.balance(), 10000); // Revenue: credit-normal
    }

    #[test]
    fn test_trial_balance_is_balanced() {
        let mut ledger = setup_basic_ledger();
        ledger.post_entry("Sale", 1000, vec![
            LineItem::debit("1000", 10000),
            LineItem::credit("4000", 10000),
        ]).unwrap();
        ledger.post_entry("Rent", 2000, vec![
            LineItem::debit("5000", 3000),
            LineItem::credit("1000", 3000),
        ]).unwrap();
        let tb = ledger.trial_balance();
        assert!(tb.is_balanced());
        assert_eq!(tb.total_debits, tb.total_credits);
    }

    #[test]
    fn test_income_statement() {
        let mut ledger = setup_basic_ledger();
        ledger.post_entry("Sale", 1000, vec![
            LineItem::debit("1000", 10000),
            LineItem::credit("4000", 10000),
        ]).unwrap();
        ledger.post_entry("Rent", 2000, vec![
            LineItem::debit("5000", 3000),
            LineItem::credit("1000", 3000),
        ]).unwrap();
        let is = ledger.income_statement();
        assert_eq!(is.total_revenue, 10000);
        assert_eq!(is.total_expenses, 3000);
        assert_eq!(is.net_income, 7000);
    }

    #[test]
    fn test_balance_sheet() {
        let mut ledger = setup_basic_ledger();
        // Owner investment
        ledger.post_entry("Investment", 1000, vec![
            LineItem::debit("1000", 50000),
            LineItem::credit("3000", 50000),
        ]).unwrap();
        let bs = ledger.balance_sheet();
        assert_eq!(bs.total_assets, 50000);
        assert_eq!(bs.total_equity, 50000);
        assert!(bs.is_balanced);
    }

    #[test]
    fn test_account_hierarchy() {
        let mut ledger = Ledger::new();
        ledger.add_account(Account::new("1000", "Assets", AccountType::Asset)).unwrap();
        ledger.add_account(
            Account::new("1100", "Cash", AccountType::Asset).with_parent("1000"),
        ).unwrap();
        ledger.add_account(
            Account::new("1200", "Receivables", AccountType::Asset).with_parent("1000"),
        ).unwrap();
        let children = ledger.children_of("1000");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_parent_not_found() {
        let mut ledger = Ledger::new();
        let err = ledger.add_account(
            Account::new("1100", "Cash", AccountType::Asset).with_parent("9999"),
        ).unwrap_err();
        assert_eq!(err, LedgerError::ParentNotFound("9999".to_string()));
    }

    #[test]
    fn test_entries_for_account() {
        let mut ledger = setup_basic_ledger();
        ledger.post_entry("Sale 1", 1000, vec![
            LineItem::debit("1000", 5000),
            LineItem::credit("4000", 5000),
        ]).unwrap();
        ledger.post_entry("Rent", 2000, vec![
            LineItem::debit("5000", 2000),
            LineItem::credit("1000", 2000),
        ]).unwrap();
        let cash_entries = ledger.entries_for_account("1000");
        assert_eq!(cash_entries.len(), 2);
        let rent_entries = ledger.entries_for_account("5000");
        assert_eq!(rent_entries.len(), 1);
    }

    #[test]
    fn test_journal_entry_is_balanced() {
        let entry = JournalEntry {
            id: 1,
            description: "test".to_string(),
            timestamp: 0,
            lines: vec![
                LineItem::debit("1000", 100),
                LineItem::credit("4000", 100),
            ],
        };
        assert!(entry.is_balanced());
    }

    #[test]
    fn test_account_type_display() {
        assert_eq!(format!("{}", AccountType::Asset), "Asset");
        assert_eq!(format!("{}", AccountType::Revenue), "Revenue");
    }

    #[test]
    fn test_account_type_normal_balance() {
        assert!(AccountType::Asset.is_debit_normal());
        assert!(AccountType::Expense.is_debit_normal());
        assert!(AccountType::Liability.is_credit_normal());
        assert!(AccountType::Equity.is_credit_normal());
        assert!(AccountType::Revenue.is_credit_normal());
    }

    #[test]
    fn test_multiple_entries_accumulate() {
        let mut ledger = setup_basic_ledger();
        ledger.post_entry("Sale 1", 100, vec![
            LineItem::debit("1000", 1000),
            LineItem::credit("4000", 1000),
        ]).unwrap();
        ledger.post_entry("Sale 2", 200, vec![
            LineItem::debit("1000", 2000),
            LineItem::credit("4000", 2000),
        ]).unwrap();
        let cash = ledger.get_account("1000").unwrap();
        assert_eq!(cash.balance(), 3000);
    }

    #[test]
    fn test_ledger_error_display() {
        let err = LedgerError::UnbalancedEntry { debits: 100, credits: 50 };
        let msg = format!("{err}");
        assert!(msg.contains("debits=100"));
        assert!(msg.contains("credits=50"));
    }

    #[test]
    fn test_default_ledger() {
        let ledger = Ledger::default();
        assert_eq!(ledger.account_count(), 0);
        assert_eq!(ledger.entry_count(), 0);
    }
}
