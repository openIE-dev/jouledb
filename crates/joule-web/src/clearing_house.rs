//! Clearing House — CCP novation, multilateral netting, default fund
//! waterfall, margin collection, position management, and member tiering.
//!
//! Pure-Rust central counterparty clearing model implementing novation,
//! tiered membership, default fund waterfall, and position tracking.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ClearingError {
    MemberNotFound(String),
    InsufficientMargin(String),
    InvalidTrade(String),
    DefaultTriggered(String),
    NovationFailed(String),
    ConfigError(String),
}

impl fmt::Display for ClearingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemberNotFound(s) => write!(f, "member not found: {s}"),
            Self::InsufficientMargin(s) => write!(f, "insufficient margin: {s}"),
            Self::InvalidTrade(s) => write!(f, "invalid trade: {s}"),
            Self::DefaultTriggered(s) => write!(f, "default triggered: {s}"),
            Self::NovationFailed(s) => write!(f, "novation failed: {s}"),
            Self::ConfigError(s) => write!(f, "config error: {s}"),
        }
    }
}

impl std::error::Error for ClearingError {}

// ── Member tier ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemberTier {
    /// Direct clearing member — full access.
    General,
    /// Individual clearing member — limited product set.
    Individual,
    /// Non-clearing member — clears through a general member.
    NonClearing,
}

impl fmt::Display for MemberTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => write!(f, "GCM"),
            Self::Individual => write!(f, "ICM"),
            Self::NonClearing => write!(f, "NCM"),
        }
    }
}

// ── Trade side ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl fmt::Display for TradeSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

// ── Clearing member ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ClearingMember {
    pub id: String,
    pub name: String,
    pub tier: MemberTier,
    pub default_fund_contribution: f64,
    pub margin_deposited: f64,
    pub is_active: bool,
    pub sponsor_id: Option<String>,
}

impl ClearingMember {
    pub fn new(id: &str, name: &str, tier: MemberTier) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            tier,
            default_fund_contribution: 0.0,
            margin_deposited: 0.0,
            is_active: true,
            sponsor_id: None,
        }
    }

    pub fn with_sponsor(mut self, sponsor: &str) -> Self {
        self.sponsor_id = Some(sponsor.to_string());
        self
    }

    pub fn with_default_fund(mut self, amount: f64) -> Self {
        self.default_fund_contribution = amount;
        self
    }

    pub fn with_margin(mut self, amount: f64) -> Self {
        self.margin_deposited = amount;
        self
    }

    /// Total resources available to cover losses.
    pub fn total_resources(&self) -> f64 {
        self.margin_deposited + self.default_fund_contribution
    }
}

impl fmt::Display for ClearingMember {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}, tier={})", self.name, self.id, self.tier)
    }
}

// ── Position ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Position {
    pub member_id: String,
    pub security_id: String,
    pub net_quantity: f64,
    pub avg_price: f64,
    pub mark_price: f64,
    pub realized_pnl: f64,
}

impl Position {
    pub fn new(member_id: &str, security_id: &str) -> Self {
        Self {
            member_id: member_id.to_string(),
            security_id: security_id.to_string(),
            net_quantity: 0.0,
            avg_price: 0.0,
            mark_price: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// Unrealized P&L at current mark.
    pub fn unrealized_pnl(&self) -> f64 {
        self.net_quantity * (self.mark_price - self.avg_price)
    }

    /// Notional exposure.
    pub fn notional(&self) -> f64 {
        self.net_quantity.abs() * self.mark_price
    }

    /// Apply a trade fill to this position.
    pub fn apply_fill(&mut self, side: TradeSide, quantity: f64, price: f64) {
        let signed_qty = match side {
            TradeSide::Buy => quantity,
            TradeSide::Sell => -quantity,
        };

        let new_qty = self.net_quantity + signed_qty;

        // If crossing zero, realize P&L on the closed portion
        if self.net_quantity != 0.0
            && new_qty.signum() != self.net_quantity.signum()
            && new_qty != 0.0
        {
            let closed = self.net_quantity.abs().min(signed_qty.abs());
            self.realized_pnl += closed * (price - self.avg_price) * self.net_quantity.signum();
            self.avg_price = price;
        } else if self.net_quantity.signum() == signed_qty.signum() || self.net_quantity == 0.0 {
            // Adding to position: weighted average
            let old_notional = self.net_quantity.abs() * self.avg_price;
            let add_notional = quantity * price;
            let total_qty = self.net_quantity.abs() + quantity;
            if total_qty > 0.0 {
                self.avg_price = (old_notional + add_notional) / total_qty;
            }
        } else {
            // Reducing position
            let closed = quantity.min(self.net_quantity.abs());
            self.realized_pnl += closed * (price - self.avg_price) * self.net_quantity.signum();
        }

        self.net_quantity = new_qty;
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Position({}, {}: qty={:.2}, avg={:.4}, upnl={:.2})",
            self.member_id,
            self.security_id,
            self.net_quantity,
            self.avg_price,
            self.unrealized_pnl(),
        )
    }
}

// ── Novated trade ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NovatedTrade {
    pub original_trade_id: u64,
    pub buy_member: String,
    pub sell_member: String,
    pub security_id: String,
    pub quantity: f64,
    pub price: f64,
    pub ccp_buy_leg_id: u64,
    pub ccp_sell_leg_id: u64,
}

impl fmt::Display for NovatedTrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Novated(#{}: {} <-> CCP <-> {}, {} x{:.2}@{:.4})",
            self.original_trade_id,
            self.buy_member,
            self.sell_member,
            self.security_id,
            self.quantity,
            self.price,
        )
    }
}

// ── Default waterfall layer ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WaterfallLayer {
    pub name: String,
    pub available: f64,
    pub used: f64,
}

impl WaterfallLayer {
    pub fn new(name: &str, available: f64) -> Self {
        Self {
            name: name.to_string(),
            available,
            used: 0.0,
        }
    }

    pub fn remaining(&self) -> f64 {
        self.available - self.used
    }

    /// Absorb loss up to available. Returns unabsorbed remainder.
    pub fn absorb(&mut self, loss: f64) -> f64 {
        let absorbable = loss.min(self.remaining());
        self.used += absorbable;
        loss - absorbable
    }
}

impl fmt::Display for WaterfallLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2}/{:.2} used",
            self.name, self.used, self.available
        )
    }
}

// ── Net obligation ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ClearingObligation {
    pub member_id: String,
    pub security_id: String,
    pub net_quantity: f64,
    pub net_cash: f64,
}

impl fmt::Display for ClearingObligation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Obligation({}, {}: qty={:.2}, cash={:.2})",
            self.member_id, self.security_id, self.net_quantity, self.net_cash
        )
    }
}

// ── Clearing house ──────────────────────────────────────────────

pub struct ClearingHouse {
    members: HashMap<String, ClearingMember>,
    positions: Vec<Position>,
    novated_trades: Vec<NovatedTrade>,
    next_leg_id: u64,
    ccp_skin_in_the_game: f64,
}

impl ClearingHouse {
    pub fn new() -> Self {
        Self {
            members: HashMap::new(),
            positions: Vec::new(),
            novated_trades: Vec::new(),
            next_leg_id: 1,
            ccp_skin_in_the_game: 0.0,
        }
    }

    pub fn with_ccp_capital(mut self, amount: f64) -> Self {
        self.ccp_skin_in_the_game = amount;
        self
    }

    /// Register a clearing member.
    pub fn add_member(&mut self, member: ClearingMember) -> Result<(), ClearingError> {
        if member.tier == MemberTier::NonClearing && member.sponsor_id.is_none() {
            return Err(ClearingError::ConfigError(
                "NCM requires a sponsor".into(),
            ));
        }
        self.members.insert(member.id.clone(), member);
        Ok(())
    }

    pub fn get_member(&self, id: &str) -> Option<&ClearingMember> {
        self.members.get(id)
    }

    /// Deposit margin for a member.
    pub fn deposit_margin(&mut self, member_id: &str, amount: f64) -> Result<(), ClearingError> {
        let m = self
            .members
            .get_mut(member_id)
            .ok_or_else(|| ClearingError::MemberNotFound(member_id.into()))?;
        m.margin_deposited += amount;
        Ok(())
    }

    /// Novate a bilateral trade into two CCP legs.
    pub fn novate(
        &mut self,
        trade_id: u64,
        buyer: &str,
        seller: &str,
        security_id: &str,
        quantity: f64,
        price: f64,
    ) -> Result<NovatedTrade, ClearingError> {
        if !self.members.contains_key(buyer) {
            return Err(ClearingError::MemberNotFound(buyer.into()));
        }
        if !self.members.contains_key(seller) {
            return Err(ClearingError::MemberNotFound(seller.into()));
        }
        if quantity <= 0.0 || price < 0.0 {
            return Err(ClearingError::InvalidTrade("invalid qty/price".into()));
        }

        let buy_leg = self.next_leg_id;
        self.next_leg_id += 1;
        let sell_leg = self.next_leg_id;
        self.next_leg_id += 1;

        // Update positions
        self.update_position(buyer, security_id, TradeSide::Buy, quantity, price);
        self.update_position(seller, security_id, TradeSide::Sell, quantity, price);

        let novated = NovatedTrade {
            original_trade_id: trade_id,
            buy_member: buyer.to_string(),
            sell_member: seller.to_string(),
            security_id: security_id.to_string(),
            quantity,
            price,
            ccp_buy_leg_id: buy_leg,
            ccp_sell_leg_id: sell_leg,
        };
        self.novated_trades.push(novated.clone());
        Ok(novated)
    }

    fn update_position(
        &mut self,
        member_id: &str,
        security_id: &str,
        side: TradeSide,
        quantity: f64,
        price: f64,
    ) {
        let pos = self.positions.iter_mut().find(|p| {
            p.member_id == member_id && p.security_id == security_id
        });

        if let Some(p) = pos {
            p.apply_fill(side, quantity, price);
        } else {
            let mut p = Position::new(member_id, security_id);
            p.apply_fill(side, quantity, price);
            self.positions.push(p);
        }
    }

    /// Multilateral netting: compute net obligations per member.
    pub fn multilateral_net(&self) -> Vec<ClearingObligation> {
        let mut nets: HashMap<(String, String), (f64, f64)> = HashMap::new();

        for pos in &self.positions {
            let key = (pos.member_id.clone(), pos.security_id.clone());
            let entry = nets.entry(key).or_insert((0.0, 0.0));
            entry.0 += pos.net_quantity;
            entry.1 += pos.net_quantity * pos.avg_price;
        }

        nets.into_iter()
            .map(|((mid, sid), (qty, cash))| ClearingObligation {
                member_id: mid,
                security_id: sid,
                net_quantity: qty,
                net_cash: cash,
            })
            .collect()
    }

    /// Default waterfall: allocate loss through layers.
    pub fn default_waterfall(&self, defaulter_id: &str, loss: f64) -> Result<Vec<WaterfallLayer>, ClearingError> {
        let defaulter = self
            .members
            .get(defaulter_id)
            .ok_or_else(|| ClearingError::MemberNotFound(defaulter_id.into()))?;

        let mut layers = vec![
            WaterfallLayer::new("Defaulter Margin", defaulter.margin_deposited),
            WaterfallLayer::new("Defaulter Default Fund", defaulter.default_fund_contribution),
            WaterfallLayer::new("CCP Skin-in-the-Game", self.ccp_skin_in_the_game),
        ];

        // Surviving members' default fund
        let surviving_df: f64 = self
            .members
            .values()
            .filter(|m| m.id != defaulter_id && m.is_active)
            .map(|m| m.default_fund_contribution)
            .sum();
        layers.push(WaterfallLayer::new("Surviving Members DF", surviving_df));

        let mut remaining = loss;
        for layer in &mut layers {
            remaining = layer.absorb(remaining);
            if remaining <= 0.0 {
                break;
            }
        }

        Ok(layers)
    }

    /// Collect required margin from a member. Returns shortfall if any.
    pub fn collect_margin(
        &mut self,
        member_id: &str,
        required: f64,
    ) -> Result<f64, ClearingError> {
        let m = self
            .members
            .get(member_id)
            .ok_or_else(|| ClearingError::MemberNotFound(member_id.into()))?;
        let shortfall = (required - m.margin_deposited).max(0.0);
        Ok(shortfall)
    }

    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    pub fn novated_trades(&self) -> &[NovatedTrade] {
        &self.novated_trades
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    pub fn total_default_fund(&self) -> f64 {
        self.members.values().map(|m| m.default_fund_contribution).sum()
    }
}

impl fmt::Display for ClearingHouse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClearingHouse(members={}, positions={}, trades={})",
            self.members.len(),
            self.positions.len(),
            self.novated_trades.len(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_house() -> ClearingHouse {
        let mut ch = ClearingHouse::new().with_ccp_capital(1_000_000.0);
        ch.add_member(
            ClearingMember::new("M1", "Acme", MemberTier::General)
                .with_default_fund(500_000.0)
                .with_margin(1_000_000.0),
        ).unwrap();
        ch.add_member(
            ClearingMember::new("M2", "Beta", MemberTier::General)
                .with_default_fund(400_000.0)
                .with_margin(800_000.0),
        ).unwrap();
        ch
    }

    #[test]
    fn test_add_member() {
        let mut ch = ClearingHouse::new();
        ch.add_member(ClearingMember::new("M1", "Acme", MemberTier::General)).unwrap();
        assert_eq!(ch.member_count(), 1);
    }

    #[test]
    fn test_ncm_requires_sponsor() {
        let mut ch = ClearingHouse::new();
        let r = ch.add_member(ClearingMember::new("N1", "Small", MemberTier::NonClearing));
        assert!(r.is_err());
    }

    #[test]
    fn test_ncm_with_sponsor_ok() {
        let mut ch = ClearingHouse::new();
        ch.add_member(ClearingMember::new("M1", "Acme", MemberTier::General)).unwrap();
        ch.add_member(
            ClearingMember::new("N1", "Small", MemberTier::NonClearing).with_sponsor("M1"),
        ).unwrap();
        assert_eq!(ch.member_count(), 2);
    }

    #[test]
    fn test_deposit_margin() {
        let mut ch = setup_house();
        ch.deposit_margin("M1", 500_000.0).unwrap();
        assert!((ch.get_member("M1").unwrap().margin_deposited - 1_500_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_novation() {
        let mut ch = setup_house();
        let n = ch.novate(1, "M1", "M2", "AAPL", 100.0, 150.0).unwrap();
        assert_eq!(n.buy_member, "M1");
        assert_eq!(n.sell_member, "M2");
        assert_eq!(ch.novated_trades().len(), 1);
    }

    #[test]
    fn test_novation_unknown_member() {
        let mut ch = setup_house();
        let r = ch.novate(1, "UNKNOWN", "M2", "AAPL", 100.0, 150.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_position_tracking_after_novation() {
        let mut ch = setup_house();
        ch.novate(1, "M1", "M2", "AAPL", 100.0, 150.0).unwrap();
        assert_eq!(ch.positions().len(), 2);
        let m1_pos = ch.positions().iter().find(|p| p.member_id == "M1").unwrap();
        assert!((m1_pos.net_quantity - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_multilateral_netting() {
        let mut ch = setup_house();
        ch.novate(1, "M1", "M2", "AAPL", 100.0, 150.0).unwrap();
        ch.novate(2, "M2", "M1", "AAPL", 40.0, 155.0).unwrap();
        let obligations = ch.multilateral_net();
        assert!(!obligations.is_empty());
    }

    #[test]
    fn test_default_waterfall_covers_loss() {
        let ch = setup_house();
        let layers = ch.default_waterfall("M1", 200_000.0).unwrap();
        let total_used: f64 = layers.iter().map(|l| l.used).sum();
        assert!((total_used - 200_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_default_waterfall_exhausts_layers() {
        let ch = setup_house();
        let layers = ch.default_waterfall("M1", 10_000_000.0).unwrap();
        let total_remaining: f64 = layers.iter().map(|l| l.remaining()).sum();
        assert!(total_remaining < 1e-9);
    }

    #[test]
    fn test_waterfall_layer_absorb() {
        let mut layer = WaterfallLayer::new("Test", 100.0);
        let remainder = layer.absorb(60.0);
        assert!((remainder).abs() < 1e-9);
        assert!((layer.used - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_waterfall_layer_partial_absorb() {
        let mut layer = WaterfallLayer::new("Test", 100.0);
        let remainder = layer.absorb(150.0);
        assert!((remainder - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_collect_margin_no_shortfall() {
        let mut ch = setup_house();
        let shortfall = ch.collect_margin("M1", 500_000.0).unwrap();
        assert!((shortfall).abs() < 1e-9);
    }

    #[test]
    fn test_collect_margin_with_shortfall() {
        let mut ch = setup_house();
        let shortfall = ch.collect_margin("M1", 2_000_000.0).unwrap();
        assert!((shortfall - 1_000_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_total_default_fund() {
        let ch = setup_house();
        assert!((ch.total_default_fund() - 900_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new("M1", "AAPL");
        pos.apply_fill(TradeSide::Buy, 100.0, 150.0);
        pos.mark_price = 160.0;
        assert!((pos.unrealized_pnl() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_member_total_resources() {
        let m = ClearingMember::new("M1", "Acme", MemberTier::General)
            .with_default_fund(500.0)
            .with_margin(1000.0);
        assert!((m.total_resources() - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_impls() {
        let ch = ClearingHouse::new();
        assert!(format!("{ch}").contains("ClearingHouse"));
        assert_eq!(format!("{}", MemberTier::General), "GCM");
        assert_eq!(format!("{}", TradeSide::Buy), "BUY");
    }
}
