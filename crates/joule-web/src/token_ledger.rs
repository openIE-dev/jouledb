//! Token/fungible asset ledger — accounts with balances, transfer with
//! validation, mint/burn, transaction history, allowance/approval (ERC20-like),
//! batch transfer, and balance snapshots.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from token ledger operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenError {
    /// Insufficient balance for transfer/burn.
    InsufficientBalance { account: String, available: u64, requested: u64 },
    /// Account not found.
    AccountNotFound(String),
    /// Duplicate account.
    DuplicateAccount(String),
    /// Transfer amount must be > 0.
    ZeroAmount,
    /// Cannot transfer to self.
    SelfTransfer(String),
    /// Allowance exceeded.
    AllowanceExceeded { owner: String, spender: String, allowance: u64, requested: u64 },
    /// Overflow in balance computation.
    Overflow,
    /// Snapshot not found.
    SnapshotNotFound(u64),
}

impl fmt::Display for TokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientBalance { account, available, requested } => {
                write!(f, "insufficient balance for {account}: have {available}, need {requested}")
            }
            Self::AccountNotFound(a) => write!(f, "account not found: {a}"),
            Self::DuplicateAccount(a) => write!(f, "duplicate account: {a}"),
            Self::ZeroAmount => write!(f, "amount must be greater than zero"),
            Self::SelfTransfer(a) => write!(f, "cannot transfer to self: {a}"),
            Self::AllowanceExceeded { owner, spender, allowance, requested } => {
                write!(
                    f,
                    "allowance exceeded: {spender} allowed {allowance} from {owner}, requested {requested}"
                )
            }
            Self::Overflow => write!(f, "balance overflow"),
            Self::SnapshotNotFound(id) => write!(f, "snapshot not found: {id}"),
        }
    }
}

impl std::error::Error for TokenError {}

// ── Transaction ─────────────────────────────────────────────────────────────

/// Type of token transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxType {
    /// Tokens minted (created).
    Mint,
    /// Tokens burned (destroyed).
    Burn,
    /// Tokens transferred between accounts.
    Transfer,
    /// Transfer on behalf of another (via allowance).
    TransferFrom,
}

impl fmt::Display for TxType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mint => write!(f, "mint"),
            Self::Burn => write!(f, "burn"),
            Self::Transfer => write!(f, "transfer"),
            Self::TransferFrom => write!(f, "transfer_from"),
        }
    }
}

/// A recorded token transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransaction {
    /// Sequential transaction ID.
    pub id: u64,
    /// Transaction type.
    pub tx_type: TxType,
    /// Source account (None for mint).
    pub from: Option<String>,
    /// Destination account (None for burn).
    pub to: Option<String>,
    /// Amount transferred.
    pub amount: u64,
    /// Unix timestamp.
    pub timestamp: u64,
}

// ── Account ─────────────────────────────────────────────────────────────────

/// A token account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAccount {
    /// Unique account address/ID.
    pub address: String,
    /// Current balance.
    pub balance: u64,
    /// Allowances granted to other accounts: spender -> amount.
    allowances: HashMap<String, u64>,
}

impl TokenAccount {
    /// Create a new account with zero balance.
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            balance: 0,
            allowances: HashMap::new(),
        }
    }

    /// Get the allowance granted to a spender.
    pub fn allowance(&self, spender: &str) -> u64 {
        self.allowances.get(spender).copied().unwrap_or(0)
    }

    /// Set the allowance for a spender.
    pub fn approve(&mut self, spender: impl Into<String>, amount: u64) {
        self.allowances.insert(spender.into(), amount);
    }
}

// ── Snapshot ────────────────────────────────────────────────────────────────

/// A snapshot of all account balances at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceSnapshot {
    /// Snapshot ID.
    pub id: u64,
    /// Unix timestamp of the snapshot.
    pub timestamp: u64,
    /// Account balances at snapshot time.
    pub balances: Vec<(String, u64)>,
    /// Total supply at snapshot time.
    pub total_supply: u64,
}

// ── Token Ledger ────────────────────────────────────────────────────────────

/// A fungible token ledger with ERC20-like semantics.
#[derive(Debug, Clone)]
pub struct TokenLedger {
    /// Token name.
    pub name: String,
    /// Token symbol.
    pub symbol: String,
    /// Decimal places (e.g. 18 for ETH-like, 6 for USDC-like).
    pub decimals: u8,
    /// Accounts keyed by address.
    accounts: HashMap<String, TokenAccount>,
    /// Transaction history.
    transactions: Vec<TokenTransaction>,
    /// Next transaction ID.
    next_tx_id: u64,
    /// Total supply of tokens.
    pub total_supply: u64,
    /// Snapshots.
    snapshots: Vec<BalanceSnapshot>,
    /// Next snapshot ID.
    next_snapshot_id: u64,
}

impl TokenLedger {
    /// Create a new token ledger.
    pub fn new(
        name: impl Into<String>,
        symbol: impl Into<String>,
        decimals: u8,
    ) -> Self {
        Self {
            name: name.into(),
            symbol: symbol.into(),
            decimals,
            accounts: HashMap::new(),
            transactions: Vec::new(),
            next_tx_id: 1,
            total_supply: 0,
            snapshots: Vec::new(),
            next_snapshot_id: 1,
        }
    }

    /// Create or retrieve an account.
    pub fn create_account(&mut self, address: impl Into<String>) -> Result<(), TokenError> {
        let addr = address.into();
        if self.accounts.contains_key(&addr) {
            return Err(TokenError::DuplicateAccount(addr));
        }
        self.accounts.insert(addr.clone(), TokenAccount::new(addr));
        Ok(())
    }

    /// Ensure an account exists, creating it if necessary.
    pub fn ensure_account(&mut self, address: impl Into<String>) {
        let addr = address.into();
        if !self.accounts.contains_key(&addr) {
            self.accounts.insert(addr.clone(), TokenAccount::new(addr));
        }
    }

    /// Get an account by address.
    pub fn get_account(&self, address: &str) -> Option<&TokenAccount> {
        self.accounts.get(address)
    }

    /// Get the balance of an account.
    pub fn balance_of(&self, address: &str) -> u64 {
        self.accounts.get(address).map(|a| a.balance).unwrap_or(0)
    }

    /// Number of accounts.
    pub fn account_count(&self) -> usize {
        self.accounts.len()
    }

    /// Mint new tokens to an account.
    pub fn mint(
        &mut self,
        to: &str,
        amount: u64,
        timestamp: u64,
    ) -> Result<u64, TokenError> {
        if amount == 0 {
            return Err(TokenError::ZeroAmount);
        }
        self.ensure_account(to);
        let account = self.accounts.get_mut(to).unwrap();
        account.balance = account.balance.checked_add(amount).ok_or(TokenError::Overflow)?;
        self.total_supply = self.total_supply.checked_add(amount).ok_or(TokenError::Overflow)?;

        let tx_id = self.next_tx_id;
        self.next_tx_id += 1;
        self.transactions.push(TokenTransaction {
            id: tx_id,
            tx_type: TxType::Mint,
            from: None,
            to: Some(to.to_string()),
            amount,
            timestamp,
        });
        Ok(tx_id)
    }

    /// Burn tokens from an account.
    pub fn burn(
        &mut self,
        from: &str,
        amount: u64,
        timestamp: u64,
    ) -> Result<u64, TokenError> {
        if amount == 0 {
            return Err(TokenError::ZeroAmount);
        }
        let account = self
            .accounts
            .get_mut(from)
            .ok_or_else(|| TokenError::AccountNotFound(from.to_string()))?;
        if account.balance < amount {
            return Err(TokenError::InsufficientBalance {
                account: from.to_string(),
                available: account.balance,
                requested: amount,
            });
        }
        account.balance -= amount;
        self.total_supply -= amount;

        let tx_id = self.next_tx_id;
        self.next_tx_id += 1;
        self.transactions.push(TokenTransaction {
            id: tx_id,
            tx_type: TxType::Burn,
            from: Some(from.to_string()),
            to: None,
            amount,
            timestamp,
        });
        Ok(tx_id)
    }

    /// Transfer tokens between accounts.
    pub fn transfer(
        &mut self,
        from: &str,
        to: &str,
        amount: u64,
        timestamp: u64,
    ) -> Result<u64, TokenError> {
        if amount == 0 {
            return Err(TokenError::ZeroAmount);
        }
        if from == to {
            return Err(TokenError::SelfTransfer(from.to_string()));
        }

        // Check balance
        let from_balance = self
            .accounts
            .get(from)
            .ok_or_else(|| TokenError::AccountNotFound(from.to_string()))?
            .balance;
        if from_balance < amount {
            return Err(TokenError::InsufficientBalance {
                account: from.to_string(),
                available: from_balance,
                requested: amount,
            });
        }

        self.ensure_account(to);

        // Perform transfer
        self.accounts.get_mut(from).unwrap().balance -= amount;
        let to_acct = self.accounts.get_mut(to).unwrap();
        to_acct.balance = to_acct.balance.checked_add(amount).ok_or(TokenError::Overflow)?;

        let tx_id = self.next_tx_id;
        self.next_tx_id += 1;
        self.transactions.push(TokenTransaction {
            id: tx_id,
            tx_type: TxType::Transfer,
            from: Some(from.to_string()),
            to: Some(to.to_string()),
            amount,
            timestamp,
        });
        Ok(tx_id)
    }

    /// Approve a spender to transfer up to `amount` on behalf of the owner.
    pub fn approve(
        &mut self,
        owner: &str,
        spender: &str,
        amount: u64,
    ) -> Result<(), TokenError> {
        let account = self
            .accounts
            .get_mut(owner)
            .ok_or_else(|| TokenError::AccountNotFound(owner.to_string()))?;
        account.approve(spender, amount);
        Ok(())
    }

    /// Get the allowance a spender has from an owner.
    pub fn allowance(&self, owner: &str, spender: &str) -> u64 {
        self.accounts
            .get(owner)
            .map(|a| a.allowance(spender))
            .unwrap_or(0)
    }

    /// Transfer tokens on behalf of the owner using allowance.
    pub fn transfer_from(
        &mut self,
        spender: &str,
        owner: &str,
        to: &str,
        amount: u64,
        timestamp: u64,
    ) -> Result<u64, TokenError> {
        if amount == 0 {
            return Err(TokenError::ZeroAmount);
        }
        if owner == to {
            return Err(TokenError::SelfTransfer(owner.to_string()));
        }

        // Check allowance
        let current_allowance = self.allowance(owner, spender);
        if current_allowance < amount {
            return Err(TokenError::AllowanceExceeded {
                owner: owner.to_string(),
                spender: spender.to_string(),
                allowance: current_allowance,
                requested: amount,
            });
        }

        // Check balance
        let owner_balance = self
            .accounts
            .get(owner)
            .ok_or_else(|| TokenError::AccountNotFound(owner.to_string()))?
            .balance;
        if owner_balance < amount {
            return Err(TokenError::InsufficientBalance {
                account: owner.to_string(),
                available: owner_balance,
                requested: amount,
            });
        }

        self.ensure_account(to);

        // Decrease allowance
        self.accounts
            .get_mut(owner)
            .unwrap()
            .approve(spender, current_allowance - amount);

        // Perform transfer
        self.accounts.get_mut(owner).unwrap().balance -= amount;
        let to_acct = self.accounts.get_mut(to).unwrap();
        to_acct.balance = to_acct.balance.checked_add(amount).ok_or(TokenError::Overflow)?;

        let tx_id = self.next_tx_id;
        self.next_tx_id += 1;
        self.transactions.push(TokenTransaction {
            id: tx_id,
            tx_type: TxType::TransferFrom,
            from: Some(owner.to_string()),
            to: Some(to.to_string()),
            amount,
            timestamp,
        });
        Ok(tx_id)
    }

    /// Batch transfer to multiple recipients.
    pub fn batch_transfer(
        &mut self,
        from: &str,
        recipients: &[(String, u64)],
        timestamp: u64,
    ) -> Result<Vec<u64>, TokenError> {
        // Pre-validate total amount
        let total: u64 = recipients
            .iter()
            .map(|(_, amt)| *amt)
            .try_fold(0u64, |acc, amt| acc.checked_add(amt))
            .ok_or(TokenError::Overflow)?;

        let from_balance = self
            .accounts
            .get(from)
            .ok_or_else(|| TokenError::AccountNotFound(from.to_string()))?
            .balance;
        if from_balance < total {
            return Err(TokenError::InsufficientBalance {
                account: from.to_string(),
                available: from_balance,
                requested: total,
            });
        }

        let mut tx_ids = Vec::new();
        for (to, amount) in recipients {
            let id = self.transfer(from, to, *amount, timestamp)?;
            tx_ids.push(id);
        }
        Ok(tx_ids)
    }

    /// Get all transactions.
    pub fn transactions(&self) -> &[TokenTransaction] {
        &self.transactions
    }

    /// Get transactions for a specific account.
    pub fn transactions_for(&self, address: &str) -> Vec<&TokenTransaction> {
        self.transactions
            .iter()
            .filter(|tx| {
                tx.from.as_deref() == Some(address) || tx.to.as_deref() == Some(address)
            })
            .collect()
    }

    /// Take a balance snapshot.
    pub fn take_snapshot(&mut self, timestamp: u64) -> u64 {
        let id = self.next_snapshot_id;
        self.next_snapshot_id += 1;

        let mut balances: Vec<(String, u64)> = self
            .accounts
            .iter()
            .map(|(addr, acct)| (addr.clone(), acct.balance))
            .collect();
        balances.sort_by(|a, b| a.0.cmp(&b.0));

        self.snapshots.push(BalanceSnapshot {
            id,
            timestamp,
            balances,
            total_supply: self.total_supply,
        });
        id
    }

    /// Get a snapshot by ID.
    pub fn get_snapshot(&self, id: u64) -> Option<&BalanceSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Get the balance of an account at a specific snapshot.
    pub fn balance_at_snapshot(&self, address: &str, snapshot_id: u64) -> Result<u64, TokenError> {
        let snapshot = self
            .snapshots
            .iter()
            .find(|s| s.id == snapshot_id)
            .ok_or(TokenError::SnapshotNotFound(snapshot_id))?;
        Ok(snapshot
            .balances
            .iter()
            .find(|(addr, _)| addr == address)
            .map(|(_, bal)| *bal)
            .unwrap_or(0))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_ledger() -> TokenLedger {
        let mut ledger = TokenLedger::new("TestToken", "TST", 18);
        ledger.ensure_account("alice");
        ledger.ensure_account("bob");
        ledger.mint("alice", 10000, 100).unwrap();
        ledger
    }

    #[test]
    fn test_create_account() {
        let mut ledger = TokenLedger::new("T", "T", 18);
        ledger.create_account("alice").unwrap();
        assert_eq!(ledger.account_count(), 1);
    }

    #[test]
    fn test_duplicate_account_error() {
        let mut ledger = TokenLedger::new("T", "T", 18);
        ledger.create_account("alice").unwrap();
        let err = ledger.create_account("alice").unwrap_err();
        assert_eq!(err, TokenError::DuplicateAccount("alice".to_string()));
    }

    #[test]
    fn test_mint_tokens() {
        let ledger = setup_ledger();
        assert_eq!(ledger.balance_of("alice"), 10000);
        assert_eq!(ledger.total_supply, 10000);
    }

    #[test]
    fn test_mint_zero_error() {
        let mut ledger = TokenLedger::new("T", "T", 18);
        ledger.ensure_account("alice");
        let err = ledger.mint("alice", 0, 0).unwrap_err();
        assert_eq!(err, TokenError::ZeroAmount);
    }

    #[test]
    fn test_burn_tokens() {
        let mut ledger = setup_ledger();
        ledger.burn("alice", 3000, 200).unwrap();
        assert_eq!(ledger.balance_of("alice"), 7000);
        assert_eq!(ledger.total_supply, 7000);
    }

    #[test]
    fn test_burn_insufficient() {
        let mut ledger = setup_ledger();
        let err = ledger.burn("alice", 99999, 200).unwrap_err();
        assert!(matches!(err, TokenError::InsufficientBalance { .. }));
    }

    #[test]
    fn test_transfer() {
        let mut ledger = setup_ledger();
        ledger.transfer("alice", "bob", 4000, 200).unwrap();
        assert_eq!(ledger.balance_of("alice"), 6000);
        assert_eq!(ledger.balance_of("bob"), 4000);
    }

    #[test]
    fn test_transfer_insufficient() {
        let mut ledger = setup_ledger();
        let err = ledger.transfer("alice", "bob", 99999, 200).unwrap_err();
        assert!(matches!(err, TokenError::InsufficientBalance { .. }));
    }

    #[test]
    fn test_self_transfer_error() {
        let mut ledger = setup_ledger();
        let err = ledger.transfer("alice", "alice", 100, 200).unwrap_err();
        assert_eq!(err, TokenError::SelfTransfer("alice".to_string()));
    }

    #[test]
    fn test_zero_transfer_error() {
        let mut ledger = setup_ledger();
        let err = ledger.transfer("alice", "bob", 0, 200).unwrap_err();
        assert_eq!(err, TokenError::ZeroAmount);
    }

    #[test]
    fn test_approve_and_allowance() {
        let mut ledger = setup_ledger();
        ledger.approve("alice", "charlie", 5000).unwrap();
        assert_eq!(ledger.allowance("alice", "charlie"), 5000);
    }

    #[test]
    fn test_transfer_from() {
        let mut ledger = setup_ledger();
        ledger.ensure_account("charlie");
        ledger.approve("alice", "charlie", 5000).unwrap();
        ledger.transfer_from("charlie", "alice", "bob", 3000, 300).unwrap();
        assert_eq!(ledger.balance_of("alice"), 7000);
        assert_eq!(ledger.balance_of("bob"), 3000);
        assert_eq!(ledger.allowance("alice", "charlie"), 2000);
    }

    #[test]
    fn test_transfer_from_exceeds_allowance() {
        let mut ledger = setup_ledger();
        ledger.ensure_account("charlie");
        ledger.approve("alice", "charlie", 100).unwrap();
        let err = ledger.transfer_from("charlie", "alice", "bob", 500, 300).unwrap_err();
        assert!(matches!(err, TokenError::AllowanceExceeded { .. }));
    }

    #[test]
    fn test_batch_transfer() {
        let mut ledger = setup_ledger();
        ledger.ensure_account("charlie");
        let ids = ledger.batch_transfer(
            "alice",
            &[
                ("bob".to_string(), 2000),
                ("charlie".to_string(), 3000),
            ],
            400,
        ).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(ledger.balance_of("alice"), 5000);
        assert_eq!(ledger.balance_of("bob"), 2000);
        assert_eq!(ledger.balance_of("charlie"), 3000);
    }

    #[test]
    fn test_batch_transfer_insufficient() {
        let mut ledger = setup_ledger();
        let err = ledger.batch_transfer(
            "alice",
            &[
                ("bob".to_string(), 6000),
                ("charlie".to_string(), 6000),
            ],
            400,
        ).unwrap_err();
        assert!(matches!(err, TokenError::InsufficientBalance { .. }));
    }

    #[test]
    fn test_transaction_history() {
        let mut ledger = setup_ledger();
        ledger.transfer("alice", "bob", 500, 200).unwrap();
        assert_eq!(ledger.transactions().len(), 2); // 1 mint + 1 transfer
    }

    #[test]
    fn test_transactions_for_account() {
        let mut ledger = setup_ledger();
        ledger.transfer("alice", "bob", 500, 200).unwrap();
        let alice_txs = ledger.transactions_for("alice");
        assert_eq!(alice_txs.len(), 2); // mint + transfer
        let bob_txs = ledger.transactions_for("bob");
        assert_eq!(bob_txs.len(), 1); // transfer only
    }

    #[test]
    fn test_balance_snapshot() {
        let mut ledger = setup_ledger();
        let snap_id = ledger.take_snapshot(500);
        ledger.transfer("alice", "bob", 5000, 600).unwrap();
        // Current balance changed
        assert_eq!(ledger.balance_of("alice"), 5000);
        // Snapshot preserved old balance
        assert_eq!(ledger.balance_at_snapshot("alice", snap_id).unwrap(), 10000);
    }

    #[test]
    fn test_snapshot_not_found() {
        let ledger = setup_ledger();
        let err = ledger.balance_at_snapshot("alice", 999).unwrap_err();
        assert_eq!(err, TokenError::SnapshotNotFound(999));
    }

    #[test]
    fn test_tx_type_display() {
        assert_eq!(format!("{}", TxType::Mint), "mint");
        assert_eq!(format!("{}", TxType::Transfer), "transfer");
    }

    #[test]
    fn test_token_error_display() {
        let err = TokenError::InsufficientBalance {
            account: "alice".to_string(),
            available: 100,
            requested: 500,
        };
        let msg = format!("{err}");
        assert!(msg.contains("alice"));
        assert!(msg.contains("100"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn test_ensure_account_idempotent() {
        let mut ledger = TokenLedger::new("T", "T", 18);
        ledger.ensure_account("alice");
        ledger.ensure_account("alice"); // no error
        assert_eq!(ledger.account_count(), 1);
    }
}
