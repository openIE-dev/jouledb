use std::fmt;

use serde::{Deserialize, Serialize};

/// CAIP-10 wallet address for cross-chain identity.
///
/// Format: `{namespace}:{chainId}:{address}`
/// Examples:
/// - Base (EVM):   `eip155:8453:0x1234...abcd`
/// - Solana:       `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp:7S3P4...`
/// - Polygon:      `eip155:137:0x5678...efgh`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WalletAddress(String);

impl WalletAddress {
    /// Create a new wallet address from a CAIP-10 string.
    ///
    /// Validates basic structure: at least 3 colon-separated parts.
    pub fn new(caip10: impl Into<String>) -> Result<Self, WalletAddressError> {
        let s: String = caip10.into();
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 3 {
            return Err(WalletAddressError::InvalidFormat(s));
        }
        if parts[0].is_empty() || parts[1].is_empty() || parts[2].is_empty() {
            return Err(WalletAddressError::InvalidFormat(s));
        }
        Ok(Self(s))
    }

    /// Create a Base (EVM) wallet address from a hex address.
    pub fn base(hex_address: &str) -> Result<Self, WalletAddressError> {
        if !hex_address.starts_with("0x") || hex_address.len() != 42 {
            return Err(WalletAddressError::InvalidAddress(hex_address.to_string()));
        }
        Self::new(format!("eip155:8453:{hex_address}"))
    }

    /// The namespace (e.g., "eip155", "solana").
    pub fn namespace(&self) -> &str {
        self.0.split(':').next().unwrap_or("")
    }

    /// The chain reference (e.g., "8453" for Base, "137" for Polygon).
    pub fn chain_ref(&self) -> &str {
        self.0.split(':').nth(1).unwrap_or("")
    }

    /// The account address (everything after the second colon).
    pub fn address(&self) -> &str {
        let first_colon = self.0.find(':').unwrap_or(0);
        let second_colon = self.0[first_colon + 1..].find(':').unwrap_or(0) + first_colon + 1;
        &self.0[second_colon + 1..]
    }

    /// Whether this is a Base L2 wallet.
    pub fn is_base(&self) -> bool {
        self.0.starts_with("eip155:8453:")
    }

    /// Whether this is a Solana wallet.
    pub fn is_solana(&self) -> bool {
        self.0.starts_with("solana:")
    }

    /// Whether this is an EVM-compatible wallet (any EIP-155 chain).
    pub fn is_evm(&self) -> bool {
        self.0.starts_with("eip155:")
    }

    /// The full CAIP-10 string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WalletAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Errors for wallet address parsing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WalletAddressError {
    #[error("invalid CAIP-10 format: {0}")]
    InvalidFormat(String),
    #[error("invalid address: {0}")]
    InvalidAddress(String),
}

/// How a request is paid for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PaymentMethod {
    /// Traditional JWT/API-key org, billed monthly via invoices.
    Subscription { org_id: String },
    /// x402 pay-per-use: each request carries a payment signature.
    PayPerUse { wallet: WalletAddress },
    /// Prepaid USDC balance: deducted per-request from off-chain ledger.
    Prepaid {
        org_id: String,
        wallet: WalletAddress,
    },
    /// Free tier (health, auth, public endpoints).
    Free,
}

/// Settlement network for x402 payments.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettlementNetwork {
    /// Base L2 (eip155:8453) — Coinbase-native, default.
    #[default]
    Base,
    /// Solana mainnet — lowest cost, fastest finality.
    Solana,
    /// Polygon PoS (eip155:137) — wide DeFi ecosystem.
    Polygon,
}

impl SettlementNetwork {
    /// CAIP-2 network identifier.
    pub fn caip2(&self) -> &'static str {
        match self {
            Self::Base => "eip155:8453",
            Self::Solana => "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
            Self::Polygon => "eip155:137",
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Base => "Base",
            Self::Solana => "Solana",
            Self::Polygon => "Polygon",
        }
    }

    /// USDC contract address on this network.
    pub fn usdc_address(&self) -> &'static str {
        match self {
            Self::Base => "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            Self::Solana => "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            Self::Polygon => "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
        }
    }

    /// Approximate transaction cost in USD.
    pub fn approx_tx_cost_usd(&self) -> f64 {
        match self {
            Self::Base => 0.001,
            Self::Solana => 0.00025,
            Self::Polygon => 0.001,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wallet_address_base() {
        let addr = WalletAddress::base("0x1234567890abcdef1234567890abcdef12345678").unwrap();
        assert!(addr.is_base());
        assert!(addr.is_evm());
        assert!(!addr.is_solana());
        assert_eq!(addr.namespace(), "eip155");
        assert_eq!(addr.chain_ref(), "8453");
        assert_eq!(addr.address(), "0x1234567890abcdef1234567890abcdef12345678");
    }

    #[test]
    fn wallet_address_solana() {
        let addr =
            WalletAddress::new("solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp:7S3P4HxJED").unwrap();
        assert!(addr.is_solana());
        assert!(!addr.is_evm());
        assert_eq!(addr.namespace(), "solana");
    }

    #[test]
    fn wallet_address_invalid_format() {
        assert!(WalletAddress::new("not-a-caip10").is_err());
        assert!(WalletAddress::new("eip155:").is_err());
        assert!(WalletAddress::new("::").is_err());
    }

    #[test]
    fn wallet_address_invalid_hex() {
        assert!(WalletAddress::base("not-hex").is_err());
        assert!(WalletAddress::base("0x123").is_err()); // too short
    }

    #[test]
    fn payment_method_serde_roundtrip() {
        let method = PaymentMethod::PayPerUse {
            wallet: WalletAddress::base("0x1234567890abcdef1234567890abcdef12345678").unwrap(),
        };
        let json = serde_json::to_string(&method).unwrap();
        let deserialized: PaymentMethod = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, PaymentMethod::PayPerUse { .. }));
    }

    #[test]
    fn settlement_network_caip2() {
        assert_eq!(SettlementNetwork::Base.caip2(), "eip155:8453");
        assert_eq!(SettlementNetwork::Polygon.caip2(), "eip155:137");
        assert!(SettlementNetwork::Solana.caip2().starts_with("solana:"));
    }

    #[test]
    fn settlement_network_default_is_base() {
        assert_eq!(SettlementNetwork::default(), SettlementNetwork::Base);
    }

    #[test]
    fn wallet_display() {
        let addr = WalletAddress::base("0x1234567890abcdef1234567890abcdef12345678").unwrap();
        assert_eq!(
            addr.to_string(),
            "eip155:8453:0x1234567890abcdef1234567890abcdef12345678"
        );
    }
}
