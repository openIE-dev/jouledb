//! Audio bus/routing system with hierarchy, effects chains, sends, and snapshots.
//!
//! Buses: named audio paths with volume, mute/solo, effects chains,
//! send/return for shared effects, side-chain ducking, and snapshot
//! save/restore.

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────

/// Unique bus identifier.
pub type BusId = u64;

/// Unique effect identifier.
pub type EffectId = u64;

/// Unique snapshot identifier.
pub type SnapshotId = u64;

/// An audio effect in a bus chain.
#[derive(Debug, Clone, PartialEq)]
pub enum BusEffect {
    /// Volume gain (multiplier).
    Gain(f64),
    /// Low-pass filter with cutoff frequency.
    LowPass { cutoff: f64 },
    /// High-pass filter with cutoff frequency.
    HighPass { cutoff: f64 },
    /// Compressor (threshold, ratio, attack_ms, release_ms).
    Compressor { threshold: f64, ratio: f64, attack_ms: f64, release_ms: f64 },
    /// Limiter (ceiling level).
    Limiter { ceiling: f64 },
    /// EQ band (center_freq, gain_db, bandwidth).
    EqBand { center_freq: f64, gain_db: f64, bandwidth: f64 },
}

impl BusEffect {
    /// Apply this effect to a buffer of samples.
    pub fn process(&self, samples: &mut [f64]) {
        match self {
            BusEffect::Gain(g) => {
                for s in samples.iter_mut() { *s *= *g; }
            }
            BusEffect::LowPass { cutoff } => {
                // Simple one-pole approximation
                let alpha = (cutoff / 22050.0).clamp(0.01, 0.99);
                let mut prev = 0.0;
                for s in samples.iter_mut() {
                    prev += alpha * (*s - prev);
                    *s = prev;
                }
            }
            BusEffect::HighPass { cutoff } => {
                let alpha = (cutoff / 22050.0).clamp(0.01, 0.99);
                let mut prev = 0.0;
                for s in samples.iter_mut() {
                    prev += alpha * (*s - prev);
                    *s = *s - prev;
                }
            }
            BusEffect::Compressor { threshold, ratio, .. } => {
                for s in samples.iter_mut() {
                    let level = s.abs();
                    if level > *threshold {
                        let excess = level - threshold;
                        let compressed = threshold + excess / ratio;
                        *s = s.signum() * compressed;
                    }
                }
            }
            BusEffect::Limiter { ceiling } => {
                for s in samples.iter_mut() {
                    if s.abs() > *ceiling {
                        *s = s.signum() * *ceiling;
                    }
                }
            }
            BusEffect::EqBand { gain_db, .. } => {
                let linear_gain = 10.0f64.powf(*gain_db / 20.0);
                for s in samples.iter_mut() { *s *= linear_gain; }
            }
        }
    }
}

/// A send to another bus (auxiliary bus).
#[derive(Debug, Clone, PartialEq)]
pub struct BusSend {
    pub target_bus: BusId,
    pub send_level: f64,
    pub pre_fader: bool,
}

/// Side-chain configuration for ducking.
#[derive(Debug, Clone, PartialEq)]
pub struct SideChain {
    /// Bus that triggers the ducking.
    pub source_bus: BusId,
    /// Amount to reduce volume (0.0-1.0).
    pub duck_amount: f64,
    /// Threshold above which ducking activates.
    pub threshold: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Current envelope value (internal state).
    pub envelope: f64,
}

impl SideChain {
    pub fn new(source_bus: BusId, duck_amount: f64, threshold: f64) -> Self {
        Self {
            source_bus,
            duck_amount: duck_amount.clamp(0.0, 1.0),
            threshold,
            attack_ms: 10.0,
            release_ms: 100.0,
            envelope: 0.0,
        }
    }
}

/// An audio bus.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBus {
    pub id: BusId,
    pub name: String,
    pub volume: f64,
    pub mute: bool,
    pub solo: bool,
    pub parent: Option<BusId>,
    pub effects: Vec<(EffectId, BusEffect)>,
    pub sends: Vec<BusSend>,
    pub side_chain: Option<SideChain>,
}

impl AudioBus {
    pub fn new(id: BusId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            volume: 1.0,
            mute: false,
            solo: false,
            parent: None,
            effects: Vec::new(),
            sends: Vec::new(),
            side_chain: None,
        }
    }
}

/// Snapshot of all bus states.
#[derive(Debug, Clone, PartialEq)]
pub struct BusSnapshot {
    pub id: SnapshotId,
    pub name: String,
    pub states: Vec<BusState>,
}

/// State of a single bus in a snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct BusState {
    pub bus_id: BusId,
    pub volume: f64,
    pub mute: bool,
    pub solo: bool,
}

// ── Audio Bus System ───────────────────────────────────────────

/// Audio bus routing system.
#[derive(Debug, Clone)]
pub struct AudioBusSystem {
    buses: HashMap<BusId, AudioBus>,
    snapshots: HashMap<SnapshotId, BusSnapshot>,
    next_bus_id: BusId,
    next_effect_id: EffectId,
    next_snapshot_id: SnapshotId,
    master_bus_id: Option<BusId>,
}

impl AudioBusSystem {
    /// Create a new bus system.
    pub fn new() -> Self {
        Self {
            buses: HashMap::new(),
            snapshots: HashMap::new(),
            next_bus_id: 1,
            next_effect_id: 1,
            next_snapshot_id: 1,
            master_bus_id: None,
        }
    }

    /// Create a bus.
    pub fn add_bus(&mut self, name: &str) -> BusId {
        let id = self.next_bus_id;
        self.next_bus_id += 1;
        self.buses.insert(id, AudioBus::new(id, name));
        id
    }

    /// Create the master bus (or return existing).
    pub fn add_master_bus(&mut self) -> BusId {
        if let Some(id) = self.master_bus_id {
            return id;
        }
        let id = self.add_bus("master");
        self.master_bus_id = Some(id);
        id
    }

    /// Remove a bus.
    pub fn remove_bus(&mut self, id: BusId) -> Option<AudioBus> {
        if self.master_bus_id == Some(id) {
            self.master_bus_id = None;
        }
        // Remove as parent from children
        let children: Vec<BusId> = self.buses.values()
            .filter(|b| b.parent == Some(id))
            .map(|b| b.id)
            .collect();
        for child_id in children {
            if let Some(child) = self.buses.get_mut(&child_id) {
                child.parent = None;
            }
        }
        self.buses.remove(&id)
    }

    /// Get a bus by ID.
    pub fn get_bus(&self, id: BusId) -> Option<&AudioBus> {
        self.buses.get(&id)
    }

    /// Get a mutable bus by ID.
    pub fn get_bus_mut(&mut self, id: BusId) -> Option<&mut AudioBus> {
        self.buses.get_mut(&id)
    }

    /// Set bus volume.
    pub fn set_volume(&mut self, id: BusId, volume: f64) -> bool {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.volume = volume.clamp(0.0, 2.0);
            true
        } else {
            false
        }
    }

    /// Set bus mute.
    pub fn set_mute(&mut self, id: BusId, mute: bool) -> bool {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.mute = mute;
            true
        } else {
            false
        }
    }

    /// Set bus solo.
    pub fn set_solo(&mut self, id: BusId, solo: bool) -> bool {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.solo = solo;
            true
        } else {
            false
        }
    }

    /// Set parent bus (creates hierarchy).
    pub fn set_parent(&mut self, child_id: BusId, parent_id: BusId) -> bool {
        if child_id == parent_id { return false; }
        if !self.buses.contains_key(&parent_id) { return false; }
        // Check for cycle: parent chain should not include child
        let mut current = Some(parent_id);
        while let Some(cid) = current {
            if cid == child_id { return false; }
            current = self.buses.get(&cid).and_then(|b| b.parent);
        }
        if let Some(bus) = self.buses.get_mut(&child_id) {
            bus.parent = Some(parent_id);
            true
        } else {
            false
        }
    }

    /// Add an effect to a bus's chain.
    pub fn add_effect(&mut self, bus_id: BusId, effect: BusEffect) -> Option<EffectId> {
        let eid = self.next_effect_id;
        self.next_effect_id += 1;
        if let Some(bus) = self.buses.get_mut(&bus_id) {
            bus.effects.push((eid, effect));
            Some(eid)
        } else {
            None
        }
    }

    /// Remove an effect from a bus.
    pub fn remove_effect(&mut self, bus_id: BusId, effect_id: EffectId) -> bool {
        if let Some(bus) = self.buses.get_mut(&bus_id) {
            let before = bus.effects.len();
            bus.effects.retain(|(id, _)| *id != effect_id);
            bus.effects.len() < before
        } else {
            false
        }
    }

    /// Add a send from one bus to another.
    pub fn add_send(&mut self, from_bus: BusId, to_bus: BusId,
                    level: f64, pre_fader: bool) -> bool {
        if !self.buses.contains_key(&to_bus) { return false; }
        if let Some(bus) = self.buses.get_mut(&from_bus) {
            bus.sends.push(BusSend {
                target_bus: to_bus,
                send_level: level.clamp(0.0, 1.0),
                pre_fader,
            });
            true
        } else {
            false
        }
    }

    /// Set side-chain on a bus.
    pub fn set_side_chain(&mut self, bus_id: BusId, side_chain: SideChain) -> bool {
        if !self.buses.contains_key(&side_chain.source_bus) { return false; }
        if let Some(bus) = self.buses.get_mut(&bus_id) {
            bus.side_chain = Some(side_chain);
            true
        } else {
            false
        }
    }

    /// Number of buses.
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Get children of a bus.
    pub fn children(&self, parent_id: BusId) -> Vec<BusId> {
        let mut children: Vec<BusId> = self.buses.values()
            .filter(|b| b.parent == Some(parent_id))
            .map(|b| b.id)
            .collect();
        children.sort();
        children
    }

    /// Process all buses with given input buffers.
    /// Returns processed output per bus.
    pub fn process(&mut self, inputs: &HashMap<BusId, Vec<f64>>) -> HashMap<BusId, Vec<f64>> {
        // Get topological order (children before parents)
        let order = self.topological_order();
        let mut outputs: HashMap<BusId, Vec<f64>> = HashMap::new();

        // Snapshot side-chain envelopes (from previous bus states)
        let side_chain_levels: HashMap<BusId, f64> = {
            let mut levels = HashMap::new();
            for bus in self.buses.values() {
                if let Some(ref sc) = bus.side_chain {
                    // Compute level from source bus input
                    if let Some(src_buf) = inputs.get(&sc.source_bus) {
                        let peak: f64 = src_buf.iter().map(|s| s.abs()).fold(0.0, f64::max);
                        levels.insert(bus.id, peak);
                    }
                }
            }
            levels
        };

        for &bus_id in &order {
            let bus = match self.buses.get(&bus_id) {
                Some(b) => b.clone(),
                None => continue,
            };

            if bus.mute { continue; }

            // Start with input or silence
            let buf_len = inputs.values().next().map(|v| v.len()).unwrap_or(256);
            let mut buffer = inputs.get(&bus_id).cloned().unwrap_or_else(|| vec![0.0; buf_len]);

            // Mix in children outputs
            let child_ids = self.children(bus_id);
            for child_id in &child_ids {
                if let Some(child_buf) = outputs.get(child_id) {
                    for i in 0..buffer.len().min(child_buf.len()) {
                        buffer[i] += child_buf[i];
                    }
                }
            }

            // Apply effects chain
            let effects: Vec<BusEffect> = bus.effects.iter().map(|(_, e)| e.clone()).collect();
            for effect in &effects {
                effect.process(&mut buffer);
            }

            // Apply volume
            for s in &mut buffer {
                *s *= bus.volume;
            }

            // Apply side-chain ducking
            if let Some(ref sc) = bus.side_chain {
                if let Some(&level) = side_chain_levels.get(&bus_id) {
                    if level > sc.threshold {
                        let duck = 1.0 - sc.duck_amount;
                        for s in &mut buffer {
                            *s *= duck;
                        }
                    }
                }
            }

            outputs.insert(bus_id, buffer);
        }

        outputs
    }

    /// Save a snapshot of all bus states.
    pub fn save_snapshot(&mut self, name: &str) -> SnapshotId {
        let id = self.next_snapshot_id;
        self.next_snapshot_id += 1;
        let states: Vec<BusState> = self.buses.values().map(|b| BusState {
            bus_id: b.id,
            volume: b.volume,
            mute: b.mute,
            solo: b.solo,
        }).collect();
        self.snapshots.insert(id, BusSnapshot {
            id,
            name: name.to_string(),
            states,
        });
        id
    }

    /// Restore a snapshot.
    pub fn restore_snapshot(&mut self, id: SnapshotId) -> bool {
        let snapshot = match self.snapshots.get(&id) {
            Some(s) => s.clone(),
            None => return false,
        };
        for state in &snapshot.states {
            if let Some(bus) = self.buses.get_mut(&state.bus_id) {
                bus.volume = state.volume;
                bus.mute = state.mute;
                bus.solo = state.solo;
            }
        }
        true
    }

    /// Get snapshot by ID.
    pub fn get_snapshot(&self, id: SnapshotId) -> Option<&BusSnapshot> {
        self.snapshots.get(&id)
    }

    /// Number of snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Delete a snapshot.
    pub fn delete_snapshot(&mut self, id: SnapshotId) -> bool {
        self.snapshots.remove(&id).is_some()
    }

    /// Topological order: children before parents.
    fn topological_order(&self) -> Vec<BusId> {
        let mut in_degree: HashMap<BusId, usize> = HashMap::new();
        for bus in self.buses.values() {
            in_degree.entry(bus.id).or_insert(0);
            if let Some(pid) = bus.parent {
                *in_degree.entry(pid).or_insert(0) += 1;
            }
        }
        let mut queue: Vec<BusId> = in_degree.iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        queue.sort();
        let mut result = Vec::new();
        while let Some(n) = queue.pop() {
            result.push(n);
            // Find parent of n
            if let Some(bus) = self.buses.get(&n) {
                if let Some(pid) = bus.parent {
                    if let Some(deg) = in_degree.get_mut(&pid) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(pid);
                            queue.sort();
                        }
                    }
                }
            }
        }
        result
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_system() {
        let sys = AudioBusSystem::new();
        assert_eq!(sys.bus_count(), 0);
    }

    #[test]
    fn test_add_bus() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("sfx");
        assert_eq!(sys.bus_count(), 1);
        assert_eq!(sys.get_bus(id).unwrap().name, "sfx");
    }

    #[test]
    fn test_add_master_bus() {
        let mut sys = AudioBusSystem::new();
        let master = sys.add_master_bus();
        let master2 = sys.add_master_bus();
        assert_eq!(master, master2); // Should return same ID
    }

    #[test]
    fn test_remove_bus() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("sfx");
        sys.remove_bus(id);
        assert_eq!(sys.bus_count(), 0);
    }

    #[test]
    fn test_set_volume() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("music");
        sys.set_volume(id, 0.5);
        assert!((sys.get_bus(id).unwrap().volume - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_volume_clamped() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("music");
        sys.set_volume(id, 5.0);
        assert!((sys.get_bus(id).unwrap().volume - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bus_hierarchy() {
        let mut sys = AudioBusSystem::new();
        let master = sys.add_bus("master");
        let sfx = sys.add_bus("sfx");
        let music = sys.add_bus("music");
        assert!(sys.set_parent(sfx, master));
        assert!(sys.set_parent(music, master));
        let children = sys.children(master);
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_hierarchy_no_self_parent() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("test");
        assert!(!sys.set_parent(id, id));
    }

    #[test]
    fn test_hierarchy_no_cycle() {
        let mut sys = AudioBusSystem::new();
        let a = sys.add_bus("a");
        let b = sys.add_bus("b");
        sys.set_parent(a, b);
        assert!(!sys.set_parent(b, a)); // Would create cycle
    }

    #[test]
    fn test_add_effect() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("sfx");
        let eid = sys.add_effect(id, BusEffect::Gain(0.8)).unwrap();
        assert!(eid > 0);
        assert_eq!(sys.get_bus(id).unwrap().effects.len(), 1);
    }

    #[test]
    fn test_remove_effect() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("sfx");
        let eid = sys.add_effect(id, BusEffect::Gain(0.8)).unwrap();
        assert!(sys.remove_effect(id, eid));
        assert_eq!(sys.get_bus(id).unwrap().effects.len(), 0);
    }

    #[test]
    fn test_add_send() {
        let mut sys = AudioBusSystem::new();
        let sfx = sys.add_bus("sfx");
        let reverb = sys.add_bus("reverb");
        assert!(sys.add_send(sfx, reverb, 0.5, false));
        assert_eq!(sys.get_bus(sfx).unwrap().sends.len(), 1);
    }

    #[test]
    fn test_gain_effect_process() {
        let effect = BusEffect::Gain(0.5);
        let mut buf = vec![1.0, 0.8, 0.6];
        effect.process(&mut buf);
        assert!((buf[0] - 0.5).abs() < 1e-10);
        assert!((buf[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_limiter_effect() {
        let effect = BusEffect::Limiter { ceiling: 0.5 };
        let mut buf = vec![1.0, -0.8, 0.3];
        effect.process(&mut buf);
        assert!((buf[0] - 0.5).abs() < 1e-10);
        assert!((buf[1] - (-0.5)).abs() < 1e-10);
        assert!((buf[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_compressor_effect() {
        let effect = BusEffect::Compressor {
            threshold: 0.5, ratio: 4.0, attack_ms: 1.0, release_ms: 10.0,
        };
        let mut buf = vec![0.3, 0.9, -0.7];
        effect.process(&mut buf);
        assert!((buf[0] - 0.3).abs() < 1e-10); // Below threshold
        assert!(buf[1] < 0.9); // Compressed
    }

    #[test]
    fn test_process_single_bus() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("test");
        sys.add_effect(id, BusEffect::Gain(0.5));
        let mut inputs = HashMap::new();
        inputs.insert(id, vec![1.0, 1.0, 1.0, 1.0]);
        let outputs = sys.process(&inputs);
        let buf = outputs.get(&id).unwrap();
        for s in buf {
            assert!((*s - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_process_hierarchy() {
        let mut sys = AudioBusSystem::new();
        let master = sys.add_bus("master");
        let sfx = sys.add_bus("sfx");
        sys.set_parent(sfx, master);

        let mut inputs = HashMap::new();
        inputs.insert(sfx, vec![0.5; 4]);
        let outputs = sys.process(&inputs);
        // Master should get sfx output
        let master_buf = outputs.get(&master).unwrap();
        assert!(master_buf.iter().any(|s| s.abs() > 0.01));
    }

    #[test]
    fn test_muted_bus() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("test");
        sys.set_mute(id, true);
        let mut inputs = HashMap::new();
        inputs.insert(id, vec![1.0; 4]);
        let outputs = sys.process(&inputs);
        assert!(outputs.get(&id).is_none()); // Muted bus produces no output
    }

    #[test]
    fn test_side_chain_ducking() {
        let mut sys = AudioBusSystem::new();
        let voice = sys.add_bus("voice");
        let music = sys.add_bus("music");
        sys.set_side_chain(music, SideChain::new(voice, 0.8, 0.1));

        let mut inputs = HashMap::new();
        inputs.insert(voice, vec![0.9; 4]); // Loud voice
        inputs.insert(music, vec![0.5; 4]);
        let outputs = sys.process(&inputs);
        let music_buf = outputs.get(&music).unwrap();
        // Music should be ducked
        for s in music_buf {
            assert!(*s < 0.2);
        }
    }

    #[test]
    fn test_save_snapshot() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("test");
        sys.set_volume(id, 0.7);
        let sid = sys.save_snapshot("snapshot1");
        assert_eq!(sys.snapshot_count(), 1);
        let snap = sys.get_snapshot(sid).unwrap();
        assert_eq!(snap.name, "snapshot1");
    }

    #[test]
    fn test_restore_snapshot() {
        let mut sys = AudioBusSystem::new();
        let id = sys.add_bus("test");
        sys.set_volume(id, 0.7);
        let sid = sys.save_snapshot("snap");
        sys.set_volume(id, 0.3);
        assert!(sys.restore_snapshot(sid));
        assert!((sys.get_bus(id).unwrap().volume - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_delete_snapshot() {
        let mut sys = AudioBusSystem::new();
        let sid = sys.save_snapshot("snap");
        assert!(sys.delete_snapshot(sid));
        assert_eq!(sys.snapshot_count(), 0);
    }

    #[test]
    fn test_remove_bus_clears_children() {
        let mut sys = AudioBusSystem::new();
        let parent = sys.add_bus("parent");
        let child = sys.add_bus("child");
        sys.set_parent(child, parent);
        sys.remove_bus(parent);
        assert!(sys.get_bus(child).unwrap().parent.is_none());
    }

    #[test]
    fn test_lowpass_effect() {
        let effect = BusEffect::LowPass { cutoff: 100.0 };
        let mut buf = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let orig_energy: f64 = buf.iter().map(|s| s * s).sum();
        effect.process(&mut buf);
        let filt_energy: f64 = buf.iter().map(|s| s * s).sum();
        // Low-pass should reduce high-frequency oscillation
        assert!(filt_energy < orig_energy);
    }

    #[test]
    fn test_eq_band_boost() {
        let effect = BusEffect::EqBand {
            center_freq: 1000.0, gain_db: 6.0, bandwidth: 1.0,
        };
        let mut buf = vec![0.5; 4];
        effect.process(&mut buf);
        // +6dB ≈ 2x
        assert!(buf[0] > 0.9);
    }
}
