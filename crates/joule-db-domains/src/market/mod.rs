//! JouleDB Market Link
//!
//! VSA Encoder for High-Frequency Trading data.
//! Converts market data structures into HDC Hypervectors for zero-latency networking and storage.

pub use joule_db_hdc::{BinaryHV, BundleAccumulator};

/// Standard dimension for market hypervectors
pub const DIMENSION: usize = 10000;

/// A trade event in the market
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: Side,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, Eq, Hash)]
pub enum OptionType {
    Call,
    Put,
}

/// Greeks for options pricing sensitivity
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Greeks {
    /// Delta: Price sensitivity to underlying. Range [-1.0, +1.0]
    pub delta: f64,
    /// Gamma: Delta sensitivity to underlying. Range [0.0, ~0.1]
    pub gamma: f64,
    /// Theta: Time decay per day. Range [-1.0, 0.0] typically negative
    pub theta: f64,
    /// Vega: Volatility sensitivity. Range [0.0, ~1.0]
    pub vega: f64,
}

impl Greeks {
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64) -> Self {
        Self {
            delta,
            gamma,
            theta,
            vega,
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }
}

/// A trade event for an Option Contract
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptionTrade {
    pub symbol: String, // Underlying (e.g., AAPL)
    pub price: f64,     // Premium
    pub quantity: f64,
    pub side: Side,
    pub strike: f64,
    pub option_type: OptionType,
    pub expiration_epoch: u64, // Unix timestamp days
    /// Greeks: Price sensitivities (passed in from external pricing engine)
    pub greeks: Greeks,
}

/// A trade event for a Futures Contract
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FutureTrade {
    pub symbol: String, // Underlying resource (e.g., CL, ES)
    pub price: f64,
    pub quantity: f64,
    pub side: Side,
    pub delivery_epoch: u64, // Delivery date
}

// ============================================================================
// MarketLink Encoder (macro-generated)
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// The VSA Encoder for market data
    pub struct MarketLink {
        seed: 42,
        dimension: 10000,
        fields: ["symbol", "price", "quantity", "side", "strike", "option_type",
                 "expiration", "delta", "gamma", "theta", "vega", "delivery", "context"],
        scalars: ["price", "quantity", "strike", "expiration", "delivery",
                   "delta", "gamma", "theta", "vega"],
        enums: {
            side_vectors: Side => [Side::Buy, Side::Sell],
            option_type_vectors: OptionType => [OptionType::Call, OptionType::Put]
        },
        dynamic: {
            symbol_vectors: "symbol"
        },
    }
}

impl MarketLink {
    /// Encode a Trade struct into a single Hypervector
    ///
    /// H(Trade) = [H(Symbol_Field) * H(Symbol_Val)] + [H(Price_Field) * H(Price_Val)] + ...
    pub fn encode_trade(&mut self, trade: &Trade) -> BinaryHV {
        // 1. Encode Symbol (Look up + Bind)
        let symbol_vec = self.symbol_vectors(&trade.symbol);
        // bind is O(N/64) - fast SIMD XOR
        let bound_symbol = self.field_vectors["symbol"].bind(&symbol_vec);

        // 2. Encode Price (Hash-based for Orthogonality)
        // We use hashing to ensure distinct prices are orthogonal (Sim ~ 0.5).
        // This allows precise querying of specific price levels.
        let price_bytes = trade.price.to_le_bytes();
        let price_vec = BinaryHV::from_hash(&price_bytes, DIMENSION);
        let bound_price = self.field_vectors["price"].bind(&price_vec);

        // 3. Encode Quantity (Word-Aligned Permutation)
        let qty_shift = (trade.quantity * 1.0) as usize;
        let qty_vec = self.scalar_bases["quantity"].permute_words(qty_shift);
        let bound_qty = self.field_vectors["quantity"].bind(&qty_vec);

        // 4. Encode Side (Pre-computed Lookup)
        let bound_side = self.field_vectors["side"].bind(&self.side_vectors[&trade.side]);

        // Superpose all fields using BundleAccumulator
        // This is the bottleneck: O(N) bitwise accumulation.
        let mut accumulator = BundleAccumulator::new(DIMENSION);
        accumulator.add(&bound_symbol);
        accumulator.add(&bound_price);
        accumulator.add(&bound_qty);
        // 5. Context Vector (Constant for all trades to ensure Odd number N=5 for Majority Vote)
        // This avoids density drift from tie-breaking on N=4.
        accumulator.add(&self.field_vectors["context"]);

        accumulator.threshold()
    }

    /// Encode an Option contract trade with Greeks
    ///
    /// Adds Strike, Type, Expiration, and Greeks (Delta, Gamma, Theta, Vega) dimensions.
    pub fn encode_option_trade(&mut self, trade: &OptionTrade) -> BinaryHV {
        // 1. Symbol
        let symbol_vec = self.symbol_vectors(&trade.symbol);
        let bound_symbol = self.field_vectors["symbol"].bind(&symbol_vec);

        // 2. Price (Premium)
        let price_bytes = trade.price.to_le_bytes();
        let price_vec = BinaryHV::from_hash(&price_bytes, DIMENSION);
        let bound_price = self.field_vectors["price"].bind(&price_vec);

        // 3. Quantity
        let qty_shift = (trade.quantity * 1.0) as usize;
        let qty_vec = self.scalar_bases["quantity"].permute_words(qty_shift);
        let bound_qty = self.field_vectors["quantity"].bind(&qty_vec);

        // 4. Side
        let bound_side = self.field_vectors["side"].bind(&self.side_vectors[&trade.side]);

        // 5. Strike Price (scalar encoding for range queries)
        let strike_shift = trade.strike as usize;
        let strike_vec = self.scalar_bases["strike"].permute_words(strike_shift);
        let bound_strike = self.field_vectors["strike"].bind(&strike_vec);

        // 6. Option Type
        let bound_type =
            self.field_vectors["option_type"].bind(&self.option_type_vectors[&trade.option_type]);

        // 7. Expiration (Scalar encoding of epoch days)
        let expiry_vec =
            self.scalar_bases["expiration"].permute_words(trade.expiration_epoch as usize);
        let bound_expiry = self.field_vectors["expiration"].bind(&expiry_vec);

        // 8. Delta: Signed permutation encoding
        // Scale [-1.0, +1.0] to [0, 2000] for permutation
        let delta_shift = ((trade.greeks.delta + 1.0) * 1000.0).clamp(0.0, 2000.0) as usize;
        let delta_vec = self.scalar_bases["delta"].permute_words(delta_shift);
        let bound_delta = self.field_vectors["delta"].bind(&delta_vec);

        // 9. Gamma: Log-scale permutation encoding
        // Gamma ranges 0.001 to 0.1+, use log scale to compress range
        let gamma_shift = if trade.greeks.gamma < 0.0001 {
            0
        } else {
            let log_gamma = trade.greeks.gamma.log10();
            ((log_gamma + 4.0) * 75.0).clamp(0.0, 300.0) as usize
        };
        let gamma_vec = self.scalar_bases["gamma"].permute_words(gamma_shift);
        let bound_gamma = self.field_vectors["gamma"].bind(&gamma_vec);

        // 10. Theta: Bucketed absolute value encoding
        // Theta is typically negative; bucket by decay magnitude
        let abs_theta = trade.greeks.theta.abs();
        let theta_shift = if abs_theta < 0.01 {
            0
        } else if abs_theta < 0.05 {
            50
        } else if abs_theta < 0.1 {
            100
        } else if abs_theta < 0.5 {
            150
        } else if abs_theta < 1.0 {
            200
        } else {
            250
        };
        let theta_vec = self.scalar_bases["theta"].permute_words(theta_shift);
        let bound_theta = self.field_vectors["theta"].bind(&theta_vec);

        // 11. Vega: Linear permutation encoding
        // Scale [0.0, 1.0] to [0, 100] for permutation
        let vega_shift = (trade.greeks.vega * 100.0).clamp(0.0, 100.0) as usize;
        let vega_vec = self.scalar_bases["vega"].permute_words(vega_shift);
        let bound_vega = self.field_vectors["vega"].bind(&vega_vec);

        // Bundle all 12 fields (odd count ensures clean majority voting)
        let mut accumulator = BundleAccumulator::new(DIMENSION);
        accumulator.add(&bound_symbol);
        accumulator.add(&bound_price);
        accumulator.add(&bound_qty);
        accumulator.add(&bound_side);
        accumulator.add(&bound_strike);
        accumulator.add(&bound_type);
        accumulator.add(&bound_expiry);
        accumulator.add(&bound_delta);
        accumulator.add(&bound_gamma);
        accumulator.add(&bound_theta);
        accumulator.add(&bound_vega);
        accumulator.add(&self.field_vectors["context"]);

        accumulator.threshold()
    }

    /// Encode a delta probe for querying options by delta threshold
    pub fn encode_delta_probe(&self, delta: f64) -> BinaryHV {
        let delta_shift = ((delta + 1.0) * 1000.0).clamp(0.0, 2000.0) as usize;
        let delta_vec = self.scalar_bases["delta"].permute_words(delta_shift);
        self.field_vectors["delta"].bind(&delta_vec)
    }

    /// Encode a gamma probe for querying high-gamma options
    pub fn encode_gamma_probe(&self, gamma: f64) -> BinaryHV {
        let gamma_shift = if gamma < 0.0001 {
            0
        } else {
            let log_gamma = gamma.log10();
            ((log_gamma + 4.0) * 75.0).clamp(0.0, 300.0) as usize
        };
        let gamma_vec = self.scalar_bases["gamma"].permute_words(gamma_shift);
        self.field_vectors["gamma"].bind(&gamma_vec)
    }

    /// Encode a Future Trade
    /// Similar to Spot Trade but with a Delivery Date
    pub fn encode_future_trade(&mut self, trade: &FutureTrade) -> BinaryHV {
        // 1. Symbol
        let symbol_vec = self.symbol_vectors(&trade.symbol);
        let bound_symbol = self.field_vectors["symbol"].bind(&symbol_vec);

        // 2. Price
        let price_bytes = trade.price.to_le_bytes();
        let price_vec = BinaryHV::from_hash(&price_bytes, DIMENSION);
        let bound_price = self.field_vectors["price"].bind(&price_vec);

        // 3. Quantity
        let qty_shift = (trade.quantity * 1.0) as usize;
        let qty_vec = self.scalar_bases["quantity"].permute_words(qty_shift);
        let bound_qty = self.field_vectors["quantity"].bind(&qty_vec);

        // 4. Side
        let bound_side = self.field_vectors["side"].bind(&self.side_vectors[&trade.side]);

        // 5. Delivery Date (Scalar)
        let delivery_vec =
            self.scalar_bases["delivery"].permute_words(trade.delivery_epoch as usize);
        let bound_delivery = self.field_vectors["delivery"].bind(&delivery_vec);

        let mut accumulator = BundleAccumulator::new(DIMENSION);
        accumulator.add(&bound_symbol);
        accumulator.add(&bound_price);
        accumulator.add(&bound_qty);
        accumulator.add(&bound_side);
        accumulator.add(&bound_delivery);

        accumulator.add(&self.field_vectors["context"]);

        accumulator.threshold()
    }

    /// Public helper to encode just the Price term: H(Field_Price) * H(Price_Val)
    /// Used by HolographicOrderBook for querying coherence at a specific price.
    pub fn encode_price_term(&self, price: f64) -> BinaryHV {
        let price_bytes = price.to_le_bytes();
        let price_vec = BinaryHV::from_hash(&price_bytes, DIMENSION);
        self.field_vectors["price"].bind(&price_vec)
    }
}

pub mod order_book;
pub use order_book::HolographicOrderBook;

pub mod network;
pub use network::{HolographicBroadcaster, HolographicReceiver};

pub mod prediction;
pub use prediction::{MarketEvent, MarketPredictor};

pub mod analysis;
pub use analysis::{MarketAnalyzer, RSI, VWAP};

pub mod risk;
pub use risk::{ExpirationRisk, PortfolioRiskSummary, RiskAggregator, SymbolRisk};

pub mod strategy;
pub use strategy::{HolographicStrategyBook, MultiLegStrategy, StrategyType};

pub mod vol_surface;
pub use vol_surface::VolatilitySurface;

pub mod pnl_attribution;
pub use pnl_attribution::{GreekPnL, PnLAttributionEngine, PortfolioPnLSnapshot};

pub mod correlation;
pub use correlation::{CorrelationMatrix, MoveDirection};

pub mod margin;
pub use margin::{HedgeProposal, MarginOptimizer, MarginResult, MarginRules, PositionLeg};
