//! Neuron Models
//!
//! Leaky integrate-and-fire and other neuron models.

/// Neuron type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronType {
    /// Leaky integrate-and-fire (simplest model)
    LeakyIntegrateFire,
    /// Izhikevich model (biologically accurate)
    Izhikevich,
    /// Hodgkin-Huxley model (most accurate)
    HodgkinHuxley,
}

/// Neuron state
#[derive(Clone, Debug)]
pub struct Neuron {
    membrane_potential: f32,
    threshold: f32,
    leak: f32,
    last_spike_time: f32,
    refractory_period: f32,
    neuron_type: NeuronType,
}

impl Neuron {
    /// Create new neuron with given parameters
    pub fn new(threshold: f32, leak: f32) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold,
            leak,
            last_spike_time: -1000.0, // Far in the past
            refractory_period: 2.0,   // 2ms refractory period
            neuron_type: NeuronType::LeakyIntegrateFire,
        }
    }

    /// Create neuron with custom refractory period
    pub fn with_refractory(threshold: f32, leak: f32, refractory_period: f32) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold,
            leak,
            last_spike_time: -1000.0,
            refractory_period,
            neuron_type: NeuronType::LeakyIntegrateFire,
        }
    }

    /// Update neuron state using Leaky Integrate-and-Fire dynamics
    ///
    /// Returns true if the neuron spiked.
    pub fn update(&mut self, input_current: f32, dt: f32, current_time: f32) -> bool {
        // Leaky integration: dV/dt = -V/τ + I
        let tau = 1.0 / self.leak.max(0.001);
        self.membrane_potential *= (-dt / tau).exp();
        self.membrane_potential += input_current * dt;

        // Check refractory period
        let can_spike = (current_time - self.last_spike_time) >= self.refractory_period;

        // Check for spike
        if self.membrane_potential >= self.threshold && can_spike {
            self.membrane_potential = 0.0; // Reset
            self.last_spike_time = current_time;
            return true;
        }

        false
    }

    /// Get membrane potential
    pub fn potential(&self) -> f32 {
        self.membrane_potential
    }

    /// Set membrane potential directly
    pub fn set_potential(&mut self, potential: f32) {
        self.membrane_potential = potential;
    }

    /// Get threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get last spike time
    pub fn last_spike_time(&self) -> f32 {
        self.last_spike_time
    }

    /// Get neuron type
    pub fn neuron_type(&self) -> NeuronType {
        self.neuron_type
    }

    /// Reset neuron to initial state
    pub fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.last_spike_time = -1000.0;
    }
}

impl Default for Neuron {
    fn default() -> Self {
        Self::new(1.0, 0.1)
    }
}

/// Spike event representing a neural spike
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeEvent {
    /// ID of the neuron that spiked
    pub neuron_id: usize,
    /// Time of the spike
    pub time: f32,
    /// Spike strength/weight
    pub strength: f32,
}

impl SpikeEvent {
    /// Create new spike event
    pub fn new(neuron_id: usize, time: f32, strength: f32) -> Self {
        Self {
            neuron_id,
            time,
            strength,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new(1.0, 0.1);
        assert_eq!(neuron.threshold(), 1.0);
        assert_eq!(neuron.potential(), 0.0);
    }

    #[test]
    fn test_neuron_integration() {
        let mut neuron = Neuron::new(1.0, 0.1);

        // Apply constant input
        for i in 0..10 {
            neuron.update(0.5, 0.1, i as f32 * 0.1);
        }

        // Potential should increase
        assert!(neuron.potential() > 0.0);
    }

    #[test]
    fn test_neuron_spike() {
        let mut neuron = Neuron::new(0.5, 0.01); // Low threshold, low leak

        // Apply strong input to cause spike
        let spiked = neuron.update(10.0, 0.1, 0.0);
        assert!(spiked);

        // After spike, potential should be reset
        assert_eq!(neuron.potential(), 0.0);
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = Neuron::with_refractory(0.1, 0.01, 2.0);

        // Cause first spike
        let spiked1 = neuron.update(10.0, 0.1, 0.0);
        assert!(spiked1);

        // Try to spike again during refractory period
        neuron.set_potential(10.0);
        let spiked2 = neuron.update(0.0, 0.1, 1.0); // 1.0 < 2.0 refractory
        assert!(!spiked2);

        // After refractory period
        neuron.set_potential(10.0);
        let spiked3 = neuron.update(0.0, 0.1, 3.0); // 3.0 > 2.0 refractory
        assert!(spiked3);
    }

    #[test]
    fn test_neuron_reset() {
        let mut neuron = Neuron::new(1.0, 0.1);
        neuron.update(10.0, 0.1, 0.0);

        neuron.reset();
        assert_eq!(neuron.potential(), 0.0);
    }

    #[test]
    fn test_spike_event() {
        let event = SpikeEvent::new(5, 10.0, 1.5);
        assert_eq!(event.neuron_id, 5);
        assert_eq!(event.time, 10.0);
        assert_eq!(event.strength, 1.5);
    }
}
