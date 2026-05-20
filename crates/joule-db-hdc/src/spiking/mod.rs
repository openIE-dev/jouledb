//! Spiking Neural Networks (SNN)
//!
//! Event-driven neural networks for temporal data processing.
//!
//! # Overview
//!
//! SNNs process information using discrete spike events rather than
//! continuous values. This enables:
//!
//! - Temporal pattern recognition
//! - Energy-efficient processing
//! - Event-driven computation
//! - Biological plausibility
//!
//! # Example
//!
//! ```rust
//! use joule_db_hdc::spiking::{SpikeNeuralNetwork, SpikeEvent};
//!
//! let mut snn = SpikeNeuralNetwork::new(10, 1.0);
//!
//! // Add synaptic connections
//! snn.add_synapse(0, 1, 0.5, 1.0).unwrap();
//!
//! // Add input spike
//! snn.add_input_spike(0, 0.0, 1.0);
//!
//! // Run simulation
//! let spikes = snn.update();
//! ```

mod network;
mod neuron;
mod temporal;

pub use network::{SNNError, SNNStats, SpikeNeuralNetwork, Synapse};
pub use neuron::{Neuron, NeuronType, SpikeEvent};
pub use temporal::{DecodingType, EncodingType, TemporalDecoder, TemporalEncoder};
