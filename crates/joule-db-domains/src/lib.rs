//! # JouleDB Domain-Specific HDC Modules
//!
//! Domain-specific hyperdimensional computing encoders for various industries.
//! Each module implements specialized encoding strategies for domain data types.
//!
//! Enable individual domains via feature flags (e.g., `features = ["market", "health"]`).

#![allow(missing_docs)]

#[cfg(feature = "adtech")]
pub mod adtech;

#[cfg(feature = "agri")]
pub mod agri;

#[cfg(feature = "auto")]
pub mod auto;

#[cfg(feature = "cyber")]
pub mod cyber;

#[cfg(feature = "edu")]
pub mod edu;

#[cfg(feature = "energy")]
pub mod energy;

#[cfg(feature = "gaming")]
pub mod gaming;

#[cfg(feature = "genomics")]
pub mod genomics;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "health")]
pub mod health;

#[cfg(feature = "insurance")]
pub mod insurance;

#[cfg(feature = "iot")]
pub mod iot;

#[cfg(feature = "legal")]
pub mod legal;

#[cfg(feature = "market")]
pub mod market;

#[cfg(feature = "media")]
pub mod media;

#[cfg(feature = "multimodal")]
pub mod multimodal;

#[cfg(feature = "retail")]
pub mod retail;

#[cfg(feature = "spatial")]
pub mod spatial;

#[cfg(feature = "supply")]
pub mod supply;

#[cfg(feature = "telecom")]
pub mod telecom;

#[cfg(feature = "temporal")]
pub mod temporal;
