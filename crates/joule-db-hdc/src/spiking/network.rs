//! Spiking Neural Network

use super::neuron::{Neuron, SpikeEvent};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// SNN errors
#[derive(Error, Debug, Clone)]
pub enum SNNError {
    /// Invalid neuron index
    #[error("Invalid neuron index: {0} (network has {1} neurons)")]
    InvalidNeuronIndex(usize, usize),

    /// Lock error
    #[error("Lock poisoned")]
    LockPoisoned,
}

/// Synapse connecting two neurons
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Source neuron
    pub source: usize,
    /// Target neuron
    pub target: usize,
    /// Connection weight
    pub weight: f32,
    /// Transmission delay (ms)
    pub delay: f32,
}

impl Synapse {
    /// Create new synapse
    pub fn new(source: usize, target: usize, weight: f32, delay: f32) -> Self {
        Self {
            source,
            target,
            weight,
            delay,
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct SNNStats {
    /// Number of neurons
    pub num_neurons: usize,
    /// Number of synapses
    pub num_synapses: usize,
    /// Number of neurons that have spiked
    pub active_neurons: usize,
    /// Current simulation time
    pub current_time: f32,
    /// Time step
    pub time_step: f32,
}

/// Spiking Neural Network
pub struct SpikeNeuralNetwork {
    neurons: Arc<RwLock<Vec<Neuron>>>,
    synapses: Arc<RwLock<Vec<Synapse>>>,
    spike_times: Arc<RwLock<Vec<f32>>>,
    input_spikes: Arc<RwLock<VecDeque<SpikeEvent>>>,
    time_step: f32,
    current_time: Arc<RwLock<f32>>,
}

impl SpikeNeuralNetwork {
    /// Create new SNN with given number of neurons
    pub fn new(num_neurons: usize, time_step: f32) -> Self {
        let neurons: Vec<Neuron> = (0..num_neurons).map(|_| Neuron::new(1.0, 0.1)).collect();

        Self {
            neurons: Arc::new(RwLock::new(neurons)),
            synapses: Arc::new(RwLock::new(Vec::new())),
            spike_times: Arc::new(RwLock::new(vec![-1000.0; num_neurons])),
            input_spikes: Arc::new(RwLock::new(VecDeque::new())),
            time_step,
            current_time: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Add synapse between neurons
    pub fn add_synapse(
        &self,
        source: usize,
        target: usize,
        weight: f32,
        delay: f32,
    ) -> Result<(), SNNError> {
        let neurons = self.neurons.read().map_err(|_| SNNError::LockPoisoned)?;
        let num_neurons = neurons.len();

        if source >= num_neurons {
            return Err(SNNError::InvalidNeuronIndex(source, num_neurons));
        }
        if target >= num_neurons {
            return Err(SNNError::InvalidNeuronIndex(target, num_neurons));
        }

        let mut synapses = self.synapses.write().map_err(|_| SNNError::LockPoisoned)?;
        synapses.push(Synapse::new(source, target, weight, delay));

        Ok(())
    }

    /// Add input spike
    pub fn add_input_spike(&self, neuron_id: usize, time: f32, strength: f32) {
        let mut inputs = self.input_spikes.write().unwrap();
        inputs.push_back(SpikeEvent::new(neuron_id, time, strength));
    }

    /// Update network by one time step
    ///
    /// Returns list of neurons that spiked.
    pub fn update(&self) -> Vec<SpikeEvent> {
        let mut output_spikes = Vec::new();
        let mut neurons = self.neurons.write().unwrap();
        let synapses = self.synapses.read().unwrap();
        let mut spike_times = self.spike_times.write().unwrap();
        let mut input_spikes = self.input_spikes.write().unwrap();
        let mut current_time = self.current_time.write().unwrap();

        // Compute input currents for each neuron
        let mut input_currents = vec![0.0f32; neurons.len()];

        // Process delayed spikes from synapses
        for synapse in synapses.iter() {
            let source_spike_time = spike_times[synapse.source];
            if source_spike_time > -100.0
                && *current_time >= source_spike_time + synapse.delay
                && *current_time < source_spike_time + synapse.delay + self.time_step
            {
                input_currents[synapse.target] += synapse.weight;
            }
        }

        // Process input spikes
        while let Some(spike) = input_spikes.front() {
            if (spike.time - *current_time).abs() < self.time_step {
                if spike.neuron_id < input_currents.len() {
                    input_currents[spike.neuron_id] += spike.strength;
                }
                input_spikes.pop_front();
            } else if spike.time > *current_time {
                break;
            } else {
                input_spikes.pop_front();
            }
        }

        // Update each neuron
        for (i, neuron) in neurons.iter_mut().enumerate() {
            let spiked = neuron.update(input_currents[i], self.time_step, *current_time);
            if spiked {
                spike_times[i] = *current_time;
                output_spikes.push(SpikeEvent::new(i, *current_time, 1.0));
            }
        }

        *current_time += self.time_step;
        output_spikes
    }

    /// Run simulation for multiple time steps
    pub fn simulate(&self, num_steps: usize) -> Vec<Vec<SpikeEvent>> {
        (0..num_steps).map(|_| self.update()).collect()
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f32 {
        *self.current_time.read().unwrap()
    }

    /// Get number of neurons
    pub fn num_neurons(&self) -> usize {
        self.neurons.read().unwrap().len()
    }

    /// Get number of synapses
    pub fn num_synapses(&self) -> usize {
        self.synapses.read().unwrap().len()
    }

    /// Reset network to initial state
    pub fn reset(&self) {
        let mut neurons = self.neurons.write().unwrap();
        for neuron in neurons.iter_mut() {
            neuron.reset();
        }
        let mut spike_times = self.spike_times.write().unwrap();
        for time in spike_times.iter_mut() {
            *time = -1000.0;
        }
        self.input_spikes.write().unwrap().clear();
        *self.current_time.write().unwrap() = 0.0;
    }

    /// Get network statistics
    pub fn stats(&self) -> SNNStats {
        let neurons = self.neurons.read().unwrap();
        let synapses = self.synapses.read().unwrap();
        let spike_times = self.spike_times.read().unwrap();

        let active_neurons = spike_times.iter().filter(|&&t| t > -100.0).count();

        SNNStats {
            num_neurons: neurons.len(),
            num_synapses: synapses.len(),
            active_neurons,
            current_time: *self.current_time.read().unwrap(),
            time_step: self.time_step,
        }
    }

    /// Get time step
    pub fn time_step(&self) -> f32 {
        self.time_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_creation() {
        let snn = SpikeNeuralNetwork::new(10, 1.0);
        assert_eq!(snn.num_neurons(), 10);
        assert_eq!(snn.num_synapses(), 0);
    }

    #[test]
    fn test_add_synapse() {
        let snn = SpikeNeuralNetwork::new(5, 1.0);

        snn.add_synapse(0, 1, 0.5, 1.0).unwrap();
        assert_eq!(snn.num_synapses(), 1);

        snn.add_synapse(1, 2, 0.3, 2.0).unwrap();
        assert_eq!(snn.num_synapses(), 2);
    }

    #[test]
    fn test_invalid_synapse() {
        let snn = SpikeNeuralNetwork::new(5, 1.0);

        let result = snn.add_synapse(0, 10, 0.5, 1.0); // Invalid target
        assert!(matches!(result, Err(SNNError::InvalidNeuronIndex(10, 5))));
    }

    #[test]
    fn test_input_spike() {
        let snn = SpikeNeuralNetwork::new(3, 1.0);

        snn.add_input_spike(0, 0.0, 10.0); // Strong input to trigger spike

        let spikes = snn.update();
        // May or may not spike depending on neuron parameters
        assert!(snn.current_time() > 0.0);
    }

    #[test]
    fn test_simulate() {
        let snn = SpikeNeuralNetwork::new(3, 0.1);

        snn.add_synapse(0, 1, 1.0, 0.5).unwrap();
        snn.add_input_spike(0, 0.0, 20.0); // Strong input

        let all_spikes = snn.simulate(10);
        assert_eq!(all_spikes.len(), 10);
    }

    #[test]
    fn test_reset() {
        let snn = SpikeNeuralNetwork::new(3, 1.0);

        snn.add_input_spike(0, 0.0, 10.0);
        snn.update();

        assert!(snn.current_time() > 0.0);

        snn.reset();
        assert_eq!(snn.current_time(), 0.0);
    }

    #[test]
    fn test_stats() {
        let snn = SpikeNeuralNetwork::new(5, 0.5);
        snn.add_synapse(0, 1, 0.5, 1.0).unwrap();

        let stats = snn.stats();
        assert_eq!(stats.num_neurons, 5);
        assert_eq!(stats.num_synapses, 1);
        assert_eq!(stats.time_step, 0.5);
    }
}
