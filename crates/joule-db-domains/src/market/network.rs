use super::strategy::{MultiLegStrategy, StrategyType};
use super::{BinaryHV, DIMENSION, Greeks, HolographicOrderBook, Trade};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread;

/// Top-level message enum for the holographic protocol
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum HolographicMessage {
    Delta(HolographicDelta),
    Prediction(PredictionPacket),
    Strategy(StrategyPacket),
    RiskUpdate(RiskUpdatePacket),
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PredictionPacket {
    pub symbol: String,
    pub token: String,
    pub probability: f64,
}

/// A strategy update packet for multi-leg strategies
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct StrategyPacket {
    pub strategy_id: u128,
    pub symbol: String,
    pub strategy_type: StrategyType,
    pub vector: Vec<u8>, // Holographic encoding of strategy
    pub is_add: bool,
    pub leg_count: usize,
}

/// A risk update packet for portfolio risk changes
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct RiskUpdatePacket {
    pub symbol: String,
    pub delta_change: f64,
    pub gamma_change: f64,
    pub theta_change: f64,
    pub vega_change: f64,
    pub notional_change: f64,
    pub timestamp: u64,
}

/// A Holographic Delta Packet
/// 1.25 KB fixed size. Fits in standard MTU (1500 bytes).
/// Contains the Hypervector difference to apply to the book.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct HolographicDelta {
    pub symbol_hash: u64, // Topic/Channel
    pub vector: Vec<u8>,  // The compressed hypervector
    pub is_bid: bool,
    pub is_add: bool, // true = add, false = remove
}

pub struct HolographicBroadcaster {
    socket: UdpSocket,
    target_addr: String,
}

impl HolographicBroadcaster {
    pub fn new(bind_addr: &str, target_addr: &str) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(bind_addr)?;
        socket.set_broadcast(true)?;

        // If target is multicast, set TTL
        if let Ok(addr) = target_addr.parse::<std::net::SocketAddr>() {
            if addr.ip().is_multicast() {
                socket.set_multicast_ttl_v4(1)?;
            }
        }

        Ok(Self {
            socket,
            target_addr: target_addr.to_string(),
        })
    }

    /// Broadcast a trade update as a Holographic Delta
    /// This DOES NOT send the trade details (price/qty).
    /// It sends the mathematical STATE CHANGE.
    /// Receivers can update their local book without knowing the content.
    pub fn broadcast_update(
        &self,
        link: &mut super::MarketLink,
        trade: &Trade,
        is_add: bool,
    ) -> std::io::Result<()> {
        let hv = link.encode_trade(trade);

        // Compress/Serialize the vector
        // BinaryHV internal vec is Vec<u32> or similar. We need bytes.
        // For now, we assume we can serialize BinaryHV.
        // If BinaryHV isn't generic serialize, we might need to extract.
        // Let's assume serde works for BinaryHV (derived in joule-db-novel).

        // Wait, `BinaryHV` in `joule-db-novel` likely has `Vec<u32>`.
        // Let's rely on Serde.

        let mut hasher = DefaultHasher::new();
        trade.symbol.hash(&mut hasher);
        let symbol_hash = hasher.finish();

        let delta = HolographicDelta {
            symbol_hash,
            vector: bincode::serde::encode_to_vec(&hv, bincode::config::standard())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
            is_bid: trade.side == super::Side::Buy,
            is_add,
        };

        let message = HolographicMessage::Delta(delta);
        let payload = bincode::serde::encode_to_vec(&message, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        // Check MTU safety? 10000 bits = 1250 bytes. + Overhead. Should fit.

        self.socket.send_to(&payload, &self.target_addr)?;
        Ok(())
    }

    /// Broadcast an Option Trade update
    pub fn broadcast_option_update(
        &self,
        link: &mut super::MarketLink,
        trade: &super::OptionTrade,
        is_add: bool,
    ) -> std::io::Result<()> {
        let hv = link.encode_option_trade(trade);

        let mut hasher = DefaultHasher::new();
        trade.symbol.hash(&mut hasher);
        let symbol_hash = hasher.finish();

        let delta = HolographicDelta {
            symbol_hash,
            vector: bincode::serde::encode_to_vec(&hv, bincode::config::standard())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
            is_bid: trade.side == super::Side::Buy,
            is_add,
        };

        let message = HolographicMessage::Delta(delta);
        let payload = bincode::serde::encode_to_vec(&message, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.socket.send_to(&payload, &self.target_addr)?;
        Ok(())
    }

    /// Broadcast a Future Trade update
    pub fn broadcast_future_update(
        &self,
        link: &mut super::MarketLink,
        trade: &super::FutureTrade,
        is_add: bool,
    ) -> std::io::Result<()> {
        let hv = link.encode_future_trade(trade);

        let mut hasher = DefaultHasher::new();
        trade.symbol.hash(&mut hasher);
        let symbol_hash = hasher.finish();

        let delta = HolographicDelta {
            symbol_hash,
            vector: bincode::serde::encode_to_vec(&hv, bincode::config::standard())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
            is_bid: trade.side == super::Side::Buy,
            is_add,
        };

        let message = HolographicMessage::Delta(delta);
        let payload = bincode::serde::encode_to_vec(&message, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.socket.send_to(&payload, &self.target_addr)?;
        Ok(())
    }

    /// Broadcast a market prediction
    pub fn broadcast_prediction(
        &self,
        symbol: &str,
        token: &str,
        probability: f64,
    ) -> std::io::Result<()> {
        let packet = PredictionPacket {
            symbol: symbol.to_string(),
            token: token.to_string(),
            probability,
        };
        let message = HolographicMessage::Prediction(packet);
        let payload = bincode::serde::encode_to_vec(&message, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.socket.send_to(&payload, &self.target_addr)?;
        Ok(())
    }

    /// Broadcast a multi-leg strategy update
    pub fn broadcast_strategy_update(
        &self,
        strategy: &MultiLegStrategy,
        strategy_hv: &BinaryHV,
        is_add: bool,
    ) -> std::io::Result<()> {
        let packet = StrategyPacket {
            strategy_id: strategy.id,
            symbol: strategy.symbol.clone(),
            strategy_type: strategy.strategy_type,
            vector: bincode::serde::encode_to_vec(strategy_hv, bincode::config::standard())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
            is_add,
            leg_count: strategy.legs.len(),
        };

        let message = HolographicMessage::Strategy(packet);
        let payload = bincode::serde::encode_to_vec(&message, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.socket.send_to(&payload, &self.target_addr)?;
        Ok(())
    }

    /// Broadcast a risk update (Greeks change)
    pub fn broadcast_risk_update(
        &self,
        symbol: &str,
        greeks: &Greeks,
        notional: f64,
        is_add: bool,
    ) -> std::io::Result<()> {
        let multiplier = if is_add { 1.0 } else { -1.0 };

        let packet = RiskUpdatePacket {
            symbol: symbol.to_string(),
            delta_change: greeks.delta * multiplier,
            gamma_change: greeks.gamma * multiplier,
            theta_change: greeks.theta * multiplier,
            vega_change: greeks.vega * multiplier,
            notional_change: notional * multiplier,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                .as_secs(),
        };

        let message = HolographicMessage::RiskUpdate(packet);
        let payload = bincode::serde::encode_to_vec(&message, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.socket.send_to(&payload, &self.target_addr)?;
        Ok(())
    }
}

pub struct HolographicReceiver {
    socket: UdpSocket,
    book: Arc<Mutex<HolographicOrderBook>>,
}

impl HolographicReceiver {
    pub fn new(bind_addr: &str, book: Arc<Mutex<HolographicOrderBook>>) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(bind_addr)?;
        Ok(Self { socket, book })
    }

    /// Join a multicast group
    pub fn join_multicast(&self, multi_addr: Ipv4Addr, interface: Ipv4Addr) -> std::io::Result<()> {
        self.socket.join_multicast_v4(&multi_addr, &interface)
    }

    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.socket.local_addr()
    }

    pub fn start_listening<F>(&self, on_prediction: F)
    where
        F: Fn(String, String, f64) + Send + 'static,
    {
        let socket = self.socket.try_clone().unwrap();
        let book = self.book.clone();

        thread::spawn(move || {
            let mut buf = [0u8; 4096]; // Increased buffer for safety
            loop {
                match socket.recv_from(&mut buf) {
                    Ok((amt, _src)) => {
                        let payload = &buf[..amt];

                        // Try deserializing as new Message enum
                        if let Ok((msg, _)) = bincode::serde::decode_from_slice::<HolographicMessage, _>(payload, bincode::config::standard()) {
                            match msg {
                                HolographicMessage::Delta(delta) => {
                                    let mut b_guard = book.lock().unwrap();
                                    // Deserialize the HV component
                                    if let Ok((hv, _)) = bincode::serde::decode_from_slice::<BinaryHV, _>(&delta.vector, bincode::config::standard())
                                    {
                                        let target_bundle = if delta.is_bid {
                                            &mut b_guard.bids
                                        } else {
                                            &mut b_guard.asks
                                        };

                                        if delta.is_add {
                                            target_bundle.add(&hv);
                                        } else {
                                            target_bundle.subtract(&hv);
                                        }
                                    }
                                }
                                HolographicMessage::Prediction(pred) => {
                                    on_prediction(pred.symbol, pred.token, pred.probability);
                                }
                                HolographicMessage::Strategy(_strategy) => {
                                    // Strategy updates handled by separate strategy book
                                    // Could integrate with HolographicStrategyBook here
                                }
                                HolographicMessage::RiskUpdate(_risk) => {
                                    // Risk updates handled by separate risk aggregator
                                    // Could integrate with RiskAggregator here
                                }
                            }
                        }
                        // Backward compatibility or legacy packet fallthrough could go here
                        // but since we control both ends in this repo, we enforce the new format.
                    }
                    Err(e) => eprintln!("UDP Recv Error: {}", e),
                }
            }
        });
    }
}
