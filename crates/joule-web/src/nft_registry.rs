//! NFT registry — token minting with unique ID, ownership tracking, transfer,
//! metadata (name, description, attributes), collection management, royalty
//! tracking, ownership history, and enumeration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from NFT registry operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NftError {
    /// Token not found.
    TokenNotFound(u64),
    /// Not the owner of the token.
    NotOwner { token_id: u64, caller: String, owner: String },
    /// Collection not found.
    CollectionNotFound(String),
    /// Duplicate collection.
    DuplicateCollection(String),
    /// Token already exists.
    DuplicateToken(u64),
    /// Cannot transfer to self.
    SelfTransfer { token_id: u64 },
    /// Invalid royalty percentage (must be 0..=10000 basis points).
    InvalidRoyalty(u32),
    /// Not approved for transfer.
    NotApproved { token_id: u64, operator: String },
}

impl fmt::Display for NftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenNotFound(id) => write!(f, "token not found: {id}"),
            Self::NotOwner { token_id, caller, owner } => {
                write!(f, "token {token_id}: caller {caller} is not owner {owner}")
            }
            Self::CollectionNotFound(name) => write!(f, "collection not found: {name}"),
            Self::DuplicateCollection(name) => write!(f, "duplicate collection: {name}"),
            Self::DuplicateToken(id) => write!(f, "duplicate token ID: {id}"),
            Self::SelfTransfer { token_id } => {
                write!(f, "cannot transfer token {token_id} to self")
            }
            Self::InvalidRoyalty(bps) => {
                write!(f, "invalid royalty: {bps} basis points (max 10000)")
            }
            Self::NotApproved { token_id, operator } => {
                write!(f, "token {token_id}: operator {operator} not approved")
            }
        }
    }
}

impl std::error::Error for NftError {}

// ── Metadata ────────────────────────────────────────────────────────────────

/// An attribute on an NFT (trait_type -> value).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NftAttribute {
    pub trait_type: String,
    pub value: String,
}

impl NftAttribute {
    pub fn new(trait_type: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            trait_type: trait_type.into(),
            value: value.into(),
        }
    }
}

/// NFT metadata following OpenSea/ERC721 conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NftMetadata {
    pub name: String,
    pub description: String,
    pub image_uri: Option<String>,
    pub external_url: Option<String>,
    pub attributes: Vec<NftAttribute>,
}

impl NftMetadata {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            image_uri: None,
            external_url: None,
            attributes: Vec::new(),
        }
    }

    pub fn with_image(mut self, uri: impl Into<String>) -> Self {
        self.image_uri = Some(uri.into());
        self
    }

    pub fn with_attribute(mut self, trait_type: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push(NftAttribute::new(trait_type, value));
        self
    }
}

// ── Ownership Record ────────────────────────────────────────────────────────

/// A record of an ownership transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnershipRecord {
    pub from: Option<String>,
    pub to: String,
    pub timestamp: u64,
}

// ── Royalty ──────────────────────────────────────────────────────────────────

/// Royalty information for a token.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Royalty {
    /// Recipient address for royalty payments.
    pub recipient: String,
    /// Royalty in basis points (1 bp = 0.01%, max 10000 = 100%).
    pub basis_points: u32,
}

impl Royalty {
    pub fn new(recipient: impl Into<String>, basis_points: u32) -> Result<Self, NftError> {
        if basis_points > 10000 {
            return Err(NftError::InvalidRoyalty(basis_points));
        }
        Ok(Self {
            recipient: recipient.into(),
            basis_points,
        })
    }

    /// Calculate royalty amount from a sale price.
    pub fn calculate(&self, sale_price: u64) -> u64 {
        (sale_price as u128 * self.basis_points as u128 / 10000) as u64
    }
}

// ── NFT Token ───────────────────────────────────────────────────────────────

/// A non-fungible token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NftToken {
    pub token_id: u64,
    pub owner: String,
    pub collection: String,
    pub metadata: NftMetadata,
    pub royalty: Option<Royalty>,
    /// Approved operator for this specific token (if any).
    pub approved: Option<String>,
    /// Ownership history.
    pub history: Vec<OwnershipRecord>,
    /// Unix timestamp of minting.
    pub minted_at: u64,
}

// ── Collection ──────────────────────────────────────────────────────────────

/// An NFT collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub symbol: String,
    pub creator: String,
    pub description: String,
    /// Default royalty for tokens in this collection.
    pub default_royalty: Option<Royalty>,
    /// Token IDs in this collection.
    pub token_ids: Vec<u64>,
    /// Maximum supply (None = unlimited).
    pub max_supply: Option<u64>,
}

impl Collection {
    pub fn new(
        name: impl Into<String>,
        symbol: impl Into<String>,
        creator: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            symbol: symbol.into(),
            creator: creator.into(),
            description: String::new(),
            default_royalty: None,
            token_ids: Vec::new(),
            max_supply: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_max_supply(mut self, max: u64) -> Self {
        self.max_supply = Some(max);
        self
    }

    pub fn with_royalty(mut self, royalty: Royalty) -> Self {
        self.default_royalty = Some(royalty);
        self
    }

    pub fn supply(&self) -> u64 {
        self.token_ids.len() as u64
    }
}

// ── Registry ────────────────────────────────────────────────────────────────

/// The NFT registry — manages collections, tokens, ownership, and transfers.
#[derive(Debug, Clone)]
pub struct NftRegistry {
    /// All tokens keyed by token ID.
    tokens: HashMap<u64, NftToken>,
    /// Collections keyed by name.
    collections: HashMap<String, Collection>,
    /// Operator approvals: owner -> (operator -> approved_for_all).
    operator_approvals: HashMap<String, HashMap<String, bool>>,
    /// Next auto-generated token ID.
    next_token_id: u64,
}

impl NftRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            collections: HashMap::new(),
            operator_approvals: HashMap::new(),
            next_token_id: 1,
        }
    }

    /// Register a collection.
    pub fn create_collection(&mut self, collection: Collection) -> Result<(), NftError> {
        if self.collections.contains_key(&collection.name) {
            return Err(NftError::DuplicateCollection(collection.name.clone()));
        }
        self.collections.insert(collection.name.clone(), collection);
        Ok(())
    }

    /// Get a collection by name.
    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    /// Number of collections.
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    /// Mint an NFT into a collection.
    pub fn mint(
        &mut self,
        collection_name: &str,
        owner: impl Into<String>,
        metadata: NftMetadata,
        timestamp: u64,
    ) -> Result<u64, NftError> {
        let collection = self
            .collections
            .get_mut(collection_name)
            .ok_or_else(|| NftError::CollectionNotFound(collection_name.to_string()))?;

        let token_id = self.next_token_id;
        self.next_token_id += 1;

        let owner_str = owner.into();
        let royalty = collection.default_royalty.clone();

        collection.token_ids.push(token_id);

        let token = NftToken {
            token_id,
            owner: owner_str.clone(),
            collection: collection_name.to_string(),
            metadata,
            royalty,
            approved: None,
            history: vec![OwnershipRecord {
                from: None,
                to: owner_str,
                timestamp,
            }],
            minted_at: timestamp,
        };
        self.tokens.insert(token_id, token);
        Ok(token_id)
    }

    /// Mint with a specific token ID.
    pub fn mint_with_id(
        &mut self,
        token_id: u64,
        collection_name: &str,
        owner: impl Into<String>,
        metadata: NftMetadata,
        timestamp: u64,
    ) -> Result<(), NftError> {
        if self.tokens.contains_key(&token_id) {
            return Err(NftError::DuplicateToken(token_id));
        }

        let collection = self
            .collections
            .get_mut(collection_name)
            .ok_or_else(|| NftError::CollectionNotFound(collection_name.to_string()))?;

        let owner_str = owner.into();
        let royalty = collection.default_royalty.clone();

        collection.token_ids.push(token_id);

        let token = NftToken {
            token_id,
            owner: owner_str.clone(),
            collection: collection_name.to_string(),
            metadata,
            royalty,
            approved: None,
            history: vec![OwnershipRecord {
                from: None,
                to: owner_str,
                timestamp,
            }],
            minted_at: timestamp,
        };
        self.tokens.insert(token_id, token);

        if token_id >= self.next_token_id {
            self.next_token_id = token_id + 1;
        }
        Ok(())
    }

    /// Get a token by ID.
    pub fn get_token(&self, token_id: u64) -> Option<&NftToken> {
        self.tokens.get(&token_id)
    }

    /// Get the owner of a token.
    pub fn owner_of(&self, token_id: u64) -> Result<&str, NftError> {
        self.tokens
            .get(&token_id)
            .map(|t| t.owner.as_str())
            .ok_or(NftError::TokenNotFound(token_id))
    }

    /// Total number of tokens.
    pub fn total_supply(&self) -> usize {
        self.tokens.len()
    }

    /// Get all tokens owned by an address.
    pub fn tokens_of_owner(&self, owner: &str) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .tokens
            .values()
            .filter(|t| t.owner == owner)
            .map(|t| t.token_id)
            .collect();
        ids.sort();
        ids
    }

    /// Number of tokens owned by an address.
    pub fn balance_of(&self, owner: &str) -> usize {
        self.tokens.values().filter(|t| t.owner == owner).count()
    }

    /// Approve a specific operator for a specific token.
    pub fn approve(
        &mut self,
        caller: &str,
        operator: impl Into<String>,
        token_id: u64,
    ) -> Result<(), NftError> {
        let token = self
            .tokens
            .get_mut(&token_id)
            .ok_or(NftError::TokenNotFound(token_id))?;
        if token.owner != caller {
            return Err(NftError::NotOwner {
                token_id,
                caller: caller.to_string(),
                owner: token.owner.clone(),
            });
        }
        token.approved = Some(operator.into());
        Ok(())
    }

    /// Set approval for all tokens from an owner to an operator.
    pub fn set_approval_for_all(
        &mut self,
        owner: impl Into<String>,
        operator: impl Into<String>,
        approved: bool,
    ) {
        let owner_str = owner.into();
        let operator_str = operator.into();
        let entry = self.operator_approvals.entry(owner_str).or_default();
        entry.insert(operator_str, approved);
    }

    /// Check if an operator is approved for all tokens of an owner.
    pub fn is_approved_for_all(&self, owner: &str, operator: &str) -> bool {
        self.operator_approvals
            .get(owner)
            .and_then(|ops| ops.get(operator))
            .copied()
            .unwrap_or(false)
    }

    /// Check if a caller can transfer a token.
    fn is_authorized(&self, caller: &str, token: &NftToken) -> bool {
        if token.owner == caller {
            return true;
        }
        if token.approved.as_deref() == Some(caller) {
            return true;
        }
        self.is_approved_for_all(&token.owner, caller)
    }

    /// Transfer an NFT from one owner to another.
    pub fn transfer(
        &mut self,
        caller: &str,
        to: impl Into<String>,
        token_id: u64,
        timestamp: u64,
    ) -> Result<(), NftError> {
        let to_str = to.into();

        let token = self
            .tokens
            .get(&token_id)
            .ok_or(NftError::TokenNotFound(token_id))?;

        if token.owner == to_str {
            return Err(NftError::SelfTransfer { token_id });
        }

        if !self.is_authorized(caller, token) {
            return Err(NftError::NotApproved {
                token_id,
                operator: caller.to_string(),
            });
        }

        let from_str = token.owner.clone();

        let token = self.tokens.get_mut(&token_id).unwrap();
        token.owner = to_str.clone();
        token.approved = None; // Clear approval on transfer

        token.history.push(OwnershipRecord {
            from: Some(from_str),
            to: to_str,
            timestamp,
        });

        Ok(())
    }

    /// Get the ownership history of a token.
    pub fn ownership_history(&self, token_id: u64) -> Result<&[OwnershipRecord], NftError> {
        let token = self
            .tokens
            .get(&token_id)
            .ok_or(NftError::TokenNotFound(token_id))?;
        Ok(&token.history)
    }

    /// Calculate royalty for a sale.
    pub fn royalty_info(&self, token_id: u64, sale_price: u64) -> Result<Option<(String, u64)>, NftError> {
        let token = self
            .tokens
            .get(&token_id)
            .ok_or(NftError::TokenNotFound(token_id))?;
        match &token.royalty {
            Some(r) => Ok(Some((r.recipient.clone(), r.calculate(sale_price)))),
            None => Ok(None),
        }
    }

    /// Set royalty on a token (only by current owner).
    pub fn set_royalty(
        &mut self,
        caller: &str,
        token_id: u64,
        royalty: Royalty,
    ) -> Result<(), NftError> {
        let token = self
            .tokens
            .get_mut(&token_id)
            .ok_or(NftError::TokenNotFound(token_id))?;
        if token.owner != caller {
            return Err(NftError::NotOwner {
                token_id,
                caller: caller.to_string(),
                owner: token.owner.clone(),
            });
        }
        token.royalty = Some(royalty);
        Ok(())
    }

    /// Enumerate all token IDs (sorted).
    pub fn all_token_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.tokens.keys().copied().collect();
        ids.sort();
        ids
    }
}

impl Default for NftRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_registry() -> NftRegistry {
        let mut reg = NftRegistry::new();
        let collection = Collection::new("CryptoPunks", "PUNK", "creator1")
            .with_description("Test collection")
            .with_royalty(Royalty::new("creator1", 250).unwrap());
        reg.create_collection(collection).unwrap();
        reg
    }

    #[test]
    fn test_create_collection() {
        let reg = setup_registry();
        assert_eq!(reg.collection_count(), 1);
        let col = reg.get_collection("CryptoPunks").unwrap();
        assert_eq!(col.symbol, "PUNK");
    }

    #[test]
    fn test_duplicate_collection_error() {
        let mut reg = setup_registry();
        let err = reg.create_collection(Collection::new("CryptoPunks", "X", "y")).unwrap_err();
        assert_eq!(err, NftError::DuplicateCollection("CryptoPunks".to_string()));
    }

    #[test]
    fn test_mint_nft() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("Punk #1", "A rare punk");
        let id = reg.mint("CryptoPunks", "alice", meta, 1000).unwrap();
        assert_eq!(id, 1);
        assert_eq!(reg.total_supply(), 1);
        assert_eq!(reg.owner_of(id).unwrap(), "alice");
    }

    #[test]
    fn test_mint_with_specific_id() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("Punk #42", "Custom ID");
        reg.mint_with_id(42, "CryptoPunks", "bob", meta, 1000).unwrap();
        assert_eq!(reg.owner_of(42).unwrap(), "bob");
    }

    #[test]
    fn test_duplicate_token_id_error() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("A", "B");
        reg.mint_with_id(1, "CryptoPunks", "alice", meta.clone(), 100).unwrap();
        let err = reg.mint_with_id(1, "CryptoPunks", "bob", meta, 200).unwrap_err();
        assert_eq!(err, NftError::DuplicateToken(1));
    }

    #[test]
    fn test_transfer_nft() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        reg.transfer("alice", "bob", id, 200).unwrap();
        assert_eq!(reg.owner_of(id).unwrap(), "bob");
    }

    #[test]
    fn test_transfer_not_owner() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        let err = reg.transfer("charlie", "bob", id, 200).unwrap_err();
        assert!(matches!(err, NftError::NotApproved { .. }));
    }

    #[test]
    fn test_self_transfer_error() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        let err = reg.transfer("alice", "alice", id, 200).unwrap_err();
        assert_eq!(err, NftError::SelfTransfer { token_id: id });
    }

    #[test]
    fn test_approve_and_transfer() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        reg.approve("alice", "operator", id).unwrap();
        reg.transfer("operator", "bob", id, 200).unwrap();
        assert_eq!(reg.owner_of(id).unwrap(), "bob");
    }

    #[test]
    fn test_approval_cleared_on_transfer() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        reg.approve("alice", "operator", id).unwrap();
        reg.transfer("alice", "bob", id, 200).unwrap();
        let token = reg.get_token(id).unwrap();
        assert!(token.approved.is_none());
    }

    #[test]
    fn test_approval_for_all() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        reg.set_approval_for_all("alice", "marketplace", true);
        assert!(reg.is_approved_for_all("alice", "marketplace"));
        reg.transfer("marketplace", "bob", id, 200).unwrap();
        assert_eq!(reg.owner_of(id).unwrap(), "bob");
    }

    #[test]
    fn test_ownership_history() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        reg.transfer("alice", "bob", id, 200).unwrap();
        reg.transfer("bob", "charlie", id, 300).unwrap();
        let history = reg.ownership_history(id).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].to, "alice");
        assert_eq!(history[1].to, "bob");
        assert_eq!(history[2].to, "charlie");
    }

    #[test]
    fn test_royalty_calculation() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        let id = reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        let info = reg.royalty_info(id, 10000).unwrap().unwrap();
        assert_eq!(info.0, "creator1");
        assert_eq!(info.1, 250); // 2.5% of 10000
    }

    #[test]
    fn test_invalid_royalty() {
        let err = Royalty::new("someone", 10001).unwrap_err();
        assert_eq!(err, NftError::InvalidRoyalty(10001));
    }

    #[test]
    fn test_tokens_of_owner() {
        let mut reg = setup_registry();
        for i in 0..3 {
            let meta = NftMetadata::new(format!("T{i}"), "D");
            reg.mint("CryptoPunks", "alice", meta, 100).unwrap();
        }
        let meta = NftMetadata::new("T3", "D");
        reg.mint("CryptoPunks", "bob", meta, 100).unwrap();
        assert_eq!(reg.tokens_of_owner("alice").len(), 3);
        assert_eq!(reg.balance_of("bob"), 1);
    }

    #[test]
    fn test_metadata_with_attributes() {
        let meta = NftMetadata::new("Cool NFT", "Very cool")
            .with_image("https://example.com/img.png")
            .with_attribute("rarity", "legendary")
            .with_attribute("color", "gold");
        assert_eq!(meta.attributes.len(), 2);
        assert_eq!(meta.image_uri.as_deref(), Some("https://example.com/img.png"));
    }

    #[test]
    fn test_collection_with_max_supply() {
        let col = Collection::new("Limited", "LMT", "creator")
            .with_max_supply(100);
        assert_eq!(col.max_supply, Some(100));
        assert_eq!(col.supply(), 0);
    }

    #[test]
    fn test_all_token_ids_sorted() {
        let mut reg = setup_registry();
        let meta = NftMetadata::new("T", "D");
        reg.mint_with_id(5, "CryptoPunks", "a", meta.clone(), 100).unwrap();
        reg.mint_with_id(2, "CryptoPunks", "b", meta.clone(), 100).unwrap();
        reg.mint_with_id(8, "CryptoPunks", "c", meta, 100).unwrap();
        assert_eq!(reg.all_token_ids(), vec![2, 5, 8]);
    }

    #[test]
    fn test_nft_error_display() {
        let err = NftError::NotOwner {
            token_id: 1,
            caller: "bob".to_string(),
            owner: "alice".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("bob"));
        assert!(msg.contains("alice"));
    }

    #[test]
    fn test_default_registry() {
        let reg = NftRegistry::default();
        assert_eq!(reg.total_supply(), 0);
        assert_eq!(reg.collection_count(), 0);
    }
}
