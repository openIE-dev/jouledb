//! Market Prediction Engine
//!
//! Uses Predictive Memory (Markov/N-gram) to forecast market moves.
//! Tokenizes continuous trade flow into discrete regimes for pattern matching.

use super::{OptionTrade, OptionType, Side, Trade};
use joule_db_hdc::predictive::{NGramPredictor, Prediction, QueryPredictor};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Represents a discrete market event for prediction
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MarketEvent {
    PriceUp(String),
    PriceDown(String),
    PriceStable(String),
    Buy(String),
    Sell(String),
    WhaleBuy(String),
    WhaleSell(String),
    // Options Flow Events
    CallBuy(String),
    PutBuy(String),
    CallSell(String),
    PutSell(String),
}

impl MarketEvent {
    pub fn to_token(&self) -> String {
        match self {
            Self::PriceUp(s) => format!("{}:UP", s),
            Self::PriceDown(s) => format!("{}:DOWN", s),
            Self::PriceStable(s) => format!("{}:FLAT", s),
            Self::Buy(s) => format!("{}:BUY", s),
            Self::Sell(s) => format!("{}:SELL", s),
            Self::WhaleBuy(s) => format!("{}:WHALE_BUY", s),
            Self::WhaleSell(s) => format!("{}:WHALE_SELL", s),
            // Options
            Self::CallBuy(s) => format!("{}:OPT_CALL_BUY", s),
            Self::PutBuy(s) => format!("{}:OPT_PUT_BUY", s),
            Self::CallSell(s) => format!("{}:OPT_CALL_SELL", s),
            Self::PutSell(s) => format!("{}:OPT_PUT_SELL", s),
        }
    }
}

/// A predictor specialized for high-frequency market data
pub struct MarketPredictor {
    /// Underlying N-Gram predictor
    predictor: NGramPredictor,
    /// Last seen price for each symbol to determine direction
    last_prices: Arc<RwLock<HashMap<String, f64>>>,
    /// Reverse lookup for hashes to event names (for debugging/decoding)
    token_registry: Arc<RwLock<HashMap<u64, String>>>,
}

impl MarketPredictor {
    pub fn new(ngram_context: usize) -> Self {
        Self {
            predictor: NGramPredictor::new(ngram_context),
            last_prices: Arc::new(RwLock::new(HashMap::new())),
            token_registry: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Observe a trade and update the internal model
    pub fn observe(&self, trade: &Trade) {
        let events = self.tokenize(trade);
        self.feed(events);
    }

    /// Observe an option trade
    pub fn observe_option(&self, trade: &OptionTrade) {
        let events = self.tokenize_option(trade);
        self.feed(events);
    }

    fn feed(&self, events: Vec<MarketEvent>) {
        for event in events {
            let token = event.to_token();
            let hash = QueryPredictor::hash_query(&token);

            // Register token for decoding
            {
                let mut registry = self.token_registry.write().unwrap();
                registry.entry(hash).or_insert(token.clone());
            }

            // Feed to predictor
            self.predictor.observe(&token);
        }
    }

    /// Predict the next likely market events
    pub fn predict_next(&self, top_k: usize) -> Vec<(String, f64)> {
        let predictions = self.predictor.predict(top_k);
        let registry = self.token_registry.read().unwrap();

        predictions
            .into_iter()
            .filter_map(|p| {
                registry
                    .get(&p.hash)
                    .map(|token| (token.clone(), p.probability))
            })
            .collect()
    }

    /// Convert a continuous trade into discrete events
    fn tokenize(&self, trade: &Trade) -> Vec<MarketEvent> {
        let mut events = Vec::new();
        let mut prices = self.last_prices.write().unwrap();
        let last_price = prices.get(&trade.symbol).cloned();

        // 1. Price Direction
        if let Some(prev) = last_price {
            if trade.price > prev {
                events.push(MarketEvent::PriceUp(trade.symbol.clone()));
            } else if trade.price < prev {
                events.push(MarketEvent::PriceDown(trade.symbol.clone()));
            } else {
                events.push(MarketEvent::PriceStable(trade.symbol.clone()));
            }
        }
        prices.insert(trade.symbol.clone(), trade.price);

        // 2. Side / Whales
        // Simple heuristic: Qty > 1000 is a "Whale"
        let is_whale = trade.quantity > 1000.0;
        match (trade.side.clone(), is_whale) {
            (Side::Buy, true) => events.push(MarketEvent::WhaleBuy(trade.symbol.clone())),
            (Side::Sell, true) => events.push(MarketEvent::WhaleSell(trade.symbol.clone())),
            (Side::Buy, false) => events.push(MarketEvent::Buy(trade.symbol.clone())),
            (Side::Sell, false) => events.push(MarketEvent::Sell(trade.symbol.clone())),
        }

        events
    }

    fn tokenize_option(&self, trade: &OptionTrade) -> Vec<MarketEvent> {
        let mut events = Vec::new();
        // Just track flow type for now
        // A "Call Buy" is bullish. A "Put Buy" is bearish.
        match (trade.side.clone(), trade.option_type.clone()) {
            (Side::Buy, OptionType::Call) => {
                events.push(MarketEvent::CallBuy(trade.symbol.clone()))
            }
            (Side::Buy, OptionType::Put) => events.push(MarketEvent::PutBuy(trade.symbol.clone())),
            (Side::Sell, OptionType::Call) => {
                events.push(MarketEvent::CallSell(trade.symbol.clone()))
            }
            (Side::Sell, OptionType::Put) => {
                events.push(MarketEvent::PutSell(trade.symbol.clone()))
            }
        }
        events
    }
}
