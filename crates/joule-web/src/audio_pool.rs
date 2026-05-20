//! Voice/channel pool management for audio playback.
//!
//! Fixed pool of N voices with priority-based allocation, stealing
//! policies, one-shot vs looping, voice states, virtual voices, and
//! pool statistics.

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────

/// Unique voice identifier.
pub type VoiceId = u64;

/// Unique sound identifier.
pub type SoundId = u64;

/// Voice playback state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceState {
    /// Voice is available for allocation.
    Stopped,
    /// Voice is actively producing audio.
    Playing,
    /// Voice is paused (can be resumed).
    Paused,
    /// Voice is fading out before stopping.
    Stopping,
}

/// Playback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackMode {
    /// Play once and stop.
    OneShot,
    /// Loop continuously.
    Looping,
}

/// Voice stealing policy when pool is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StealPolicy {
    /// Steal the oldest voice.
    Oldest,
    /// Steal the quietest voice.
    Quietest,
    /// Steal the lowest-priority voice.
    LowestPriority,
}

/// Priority level (higher = more important).
pub type Priority = u32;

/// A voice in the audio pool.
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    pub id: VoiceId,
    pub sound_id: SoundId,
    pub state: VoiceState,
    pub mode: PlaybackMode,
    pub priority: Priority,
    pub volume: f64,
    pub position: usize,
    pub age: u64,
    pub fade_out_remaining: usize,
    pub fade_out_total: usize,
    pub is_virtual: bool,
}

impl Voice {
    fn new(id: VoiceId, sound_id: SoundId, priority: Priority, mode: PlaybackMode) -> Self {
        Self {
            id,
            sound_id,
            state: VoiceState::Playing,
            mode,
            priority,
            volume: 1.0,
            position: 0,
            age: 0,
            fade_out_remaining: 0,
            fade_out_total: 0,
            is_virtual: false,
        }
    }
}

/// Virtual voice — tracked but not rendered.
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualVoice {
    pub sound_id: SoundId,
    pub priority: Priority,
    pub mode: PlaybackMode,
    pub volume: f64,
    pub position: usize,
    pub age: u64,
}

/// Pool statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    pub total_voices: usize,
    pub active_voices: usize,
    pub paused_voices: usize,
    pub stopping_voices: usize,
    pub free_voices: usize,
    pub virtual_voices: usize,
    pub total_allocations: u64,
    pub total_steals: u64,
}

// ── Audio Pool ─────────────────────────────────────────────────

/// Fixed-size voice pool for audio playback management.
#[derive(Debug, Clone)]
pub struct AudioPool {
    voices: Vec<Voice>,
    virtual_voices: Vec<VirtualVoice>,
    pool_size: usize,
    steal_policy: StealPolicy,
    next_voice_id: VoiceId,
    fade_out_samples: usize,
    current_tick: u64,
    total_allocations: u64,
    total_steals: u64,
    max_virtual: usize,
}

impl AudioPool {
    /// Create a new audio pool with a fixed number of voices.
    pub fn new(pool_size: usize) -> Self {
        let voices: Vec<Voice> = (0..pool_size).map(|i| {
            Voice {
                id: (i + 1) as VoiceId,
                sound_id: 0,
                state: VoiceState::Stopped,
                mode: PlaybackMode::OneShot,
                priority: 0,
                volume: 0.0,
                position: 0,
                age: 0,
                fade_out_remaining: 0,
                fade_out_total: 0,
                is_virtual: false,
            }
        }).collect();

        Self {
            voices,
            virtual_voices: Vec::new(),
            pool_size,
            steal_policy: StealPolicy::LowestPriority,
            next_voice_id: (pool_size + 1) as VoiceId,
            fade_out_samples: 256,
            current_tick: 0,
            total_allocations: 0,
            total_steals: 0,
            max_virtual: 64,
        }
    }

    /// Set the voice stealing policy.
    pub fn set_steal_policy(&mut self, policy: StealPolicy) {
        self.steal_policy = policy;
    }

    /// Set fade-out duration in samples.
    pub fn set_fade_out_samples(&mut self, samples: usize) {
        self.fade_out_samples = samples;
    }

    /// Set maximum virtual voices.
    pub fn set_max_virtual(&mut self, max: usize) {
        self.max_virtual = max;
    }

    /// Pool size.
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Allocate a voice for a sound. Returns the voice ID, or None if
    /// no voice is available and stealing fails.
    pub fn allocate(&mut self, sound_id: SoundId, priority: Priority,
                    mode: PlaybackMode) -> Option<VoiceId> {
        self.current_tick += 1;
        self.total_allocations += 1;

        // First check virtual voices — promote if this sound was virtual
        let virtual_idx = self.virtual_voices.iter().position(|v| v.sound_id == sound_id);
        let (position, age) = if let Some(idx) = virtual_idx {
            let vv = self.virtual_voices.remove(idx);
            (vv.position, vv.age)
        } else {
            (0, self.current_tick)
        };

        // Find a free (stopped) voice
        if let Some(voice) = self.voices.iter_mut().find(|v| v.state == VoiceState::Stopped) {
            let vid = self.next_voice_id;
            self.next_voice_id += 1;
            *voice = Voice::new(vid, sound_id, priority, mode);
            voice.position = position;
            voice.age = age;
            return Some(vid);
        }

        // No free voice: attempt to steal
        if let Some(idx) = self.find_steal_candidate(priority) {
            let stolen = &self.voices[idx];
            // Move stolen to virtual if possible
            if self.virtual_voices.len() < self.max_virtual {
                self.virtual_voices.push(VirtualVoice {
                    sound_id: stolen.sound_id,
                    priority: stolen.priority,
                    mode: stolen.mode,
                    volume: stolen.volume,
                    position: stolen.position,
                    age: stolen.age,
                });
            }

            let vid = self.next_voice_id;
            self.next_voice_id += 1;
            self.voices[idx] = Voice::new(vid, sound_id, priority, mode);
            self.voices[idx].position = position;
            self.voices[idx].age = age;
            self.total_steals += 1;
            return Some(vid);
        }

        // Cannot allocate — add to virtual voices
        if self.virtual_voices.len() < self.max_virtual {
            self.virtual_voices.push(VirtualVoice {
                sound_id,
                priority,
                mode,
                volume: 1.0,
                position: 0,
                age: self.current_tick,
            });
        }

        None
    }

    /// Find the best voice to steal for the given priority.
    fn find_steal_candidate(&self, new_priority: Priority) -> Option<usize> {
        let playing: Vec<(usize, &Voice)> = self.voices.iter().enumerate()
            .filter(|(_, v)| v.state == VoiceState::Playing || v.state == VoiceState::Paused)
            .collect();

        if playing.is_empty() { return None; }

        match self.steal_policy {
            StealPolicy::Oldest => {
                playing.iter()
                    .filter(|(_, v)| v.priority <= new_priority)
                    .min_by_key(|(_, v)| v.age)
                    .map(|(i, _)| *i)
            }
            StealPolicy::Quietest => {
                playing.iter()
                    .filter(|(_, v)| v.priority <= new_priority)
                    .min_by(|(_, a), (_, b)| {
                        a.volume.partial_cmp(&b.volume).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| *i)
            }
            StealPolicy::LowestPriority => {
                playing.iter()
                    .min_by_key(|(_, v)| v.priority)
                    .filter(|(_, v)| v.priority <= new_priority)
                    .map(|(i, _)| *i)
            }
        }
    }

    /// Release a voice (initiate fade-out and stop).
    pub fn release(&mut self, voice_id: VoiceId) -> bool {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            if voice.state == VoiceState::Playing || voice.state == VoiceState::Paused {
                if self.fade_out_samples > 0 {
                    voice.state = VoiceState::Stopping;
                    voice.fade_out_remaining = self.fade_out_samples;
                    voice.fade_out_total = self.fade_out_samples;
                } else {
                    voice.state = VoiceState::Stopped;
                }
                return true;
            }
        }
        false
    }

    /// Immediately stop a voice.
    pub fn stop(&mut self, voice_id: VoiceId) -> bool {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            voice.state = VoiceState::Stopped;
            return true;
        }
        false
    }

    /// Pause a voice.
    pub fn pause(&mut self, voice_id: VoiceId) -> bool {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            if voice.state == VoiceState::Playing {
                voice.state = VoiceState::Paused;
                return true;
            }
        }
        false
    }

    /// Resume a paused voice.
    pub fn resume(&mut self, voice_id: VoiceId) -> bool {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            if voice.state == VoiceState::Paused {
                voice.state = VoiceState::Playing;
                return true;
            }
        }
        false
    }

    /// Set volume for a voice.
    pub fn set_volume(&mut self, voice_id: VoiceId, volume: f64) -> bool {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            voice.volume = volume.clamp(0.0, 1.0);
            return true;
        }
        false
    }

    /// Get a voice by ID.
    pub fn get_voice(&self, voice_id: VoiceId) -> Option<&Voice> {
        self.voices.iter().find(|v| v.id == voice_id)
    }

    /// Get all active voice IDs (playing, paused, or stopping).
    pub fn active_voice_ids(&self) -> Vec<VoiceId> {
        self.voices.iter()
            .filter(|v| v.state != VoiceState::Stopped)
            .map(|v| v.id)
            .collect()
    }

    /// Update the pool: advance positions, handle fade-outs, promote virtuals.
    pub fn update(&mut self, frames: usize) {
        self.current_tick += 1;

        for voice in &mut self.voices {
            match voice.state {
                VoiceState::Playing => {
                    voice.position += frames;
                }
                VoiceState::Stopping => {
                    voice.position += frames;
                    if voice.fade_out_remaining <= frames {
                        voice.fade_out_remaining = 0;
                        voice.state = VoiceState::Stopped;
                    } else {
                        voice.fade_out_remaining -= frames;
                    }
                }
                _ => {}
            }
        }

        // Advance virtual voices
        for vv in &mut self.virtual_voices {
            vv.position += frames;
        }

        // Try to promote virtual voices to free slots
        let mut promoted = Vec::new();
        for i in 0..self.virtual_voices.len() {
            if let Some(slot) = self.voices.iter_mut().find(|v| v.state == VoiceState::Stopped) {
                let vv = &self.virtual_voices[i];
                let vid = self.next_voice_id;
                self.next_voice_id += 1;
                *slot = Voice::new(vid, vv.sound_id, vv.priority, vv.mode);
                slot.volume = vv.volume;
                slot.position = vv.position;
                slot.age = vv.age;
                promoted.push(i);
            } else {
                break;
            }
        }
        // Remove promoted virtuals in reverse order to maintain indices
        for &idx in promoted.iter().rev() {
            self.virtual_voices.remove(idx);
        }
    }

    /// Compute fade-out gain for a stopping voice.
    pub fn fade_out_gain(voice: &Voice) -> f64 {
        if voice.state != VoiceState::Stopping || voice.fade_out_total == 0 {
            return 1.0;
        }
        voice.fade_out_remaining as f64 / voice.fade_out_total as f64
    }

    /// Stop all voices.
    pub fn stop_all(&mut self) {
        for voice in &mut self.voices {
            voice.state = VoiceState::Stopped;
        }
        self.virtual_voices.clear();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let mut active = 0;
        let mut paused = 0;
        let mut stopping = 0;
        let mut free = 0;
        for v in &self.voices {
            match v.state {
                VoiceState::Playing => active += 1,
                VoiceState::Paused => paused += 1,
                VoiceState::Stopping => stopping += 1,
                VoiceState::Stopped => free += 1,
            }
        }
        PoolStats {
            total_voices: self.pool_size,
            active_voices: active,
            paused_voices: paused,
            stopping_voices: stopping,
            free_voices: free,
            virtual_voices: self.virtual_voices.len(),
            total_allocations: self.total_allocations,
            total_steals: self.total_steals,
        }
    }

    /// Number of virtual voices.
    pub fn virtual_voice_count(&self) -> usize {
        self.virtual_voices.len()
    }

    /// Get virtual voices.
    pub fn virtual_voices(&self) -> &[VirtualVoice] {
        &self.virtual_voices
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_pool() {
        let pool = AudioPool::new(32);
        assert_eq!(pool.pool_size(), 32);
        let stats = pool.stats();
        assert_eq!(stats.free_voices, 32);
        assert_eq!(stats.active_voices, 0);
    }

    #[test]
    fn test_allocate_voice() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot);
        assert!(vid.is_some());
        let stats = pool.stats();
        assert_eq!(stats.active_voices, 1);
        assert_eq!(stats.free_voices, 3);
    }

    #[test]
    fn test_allocate_multiple() {
        let mut pool = AudioPool::new(4);
        for i in 0..4 {
            let vid = pool.allocate(i + 1, 5, PlaybackMode::OneShot);
            assert!(vid.is_some());
        }
        let stats = pool.stats();
        assert_eq!(stats.active_voices, 4);
        assert_eq!(stats.free_voices, 0);
    }

    #[test]
    fn test_steal_lowest_priority() {
        let mut pool = AudioPool::new(2);
        pool.set_steal_policy(StealPolicy::LowestPriority);
        pool.allocate(1, 5, PlaybackMode::OneShot); // priority 5
        pool.allocate(2, 3, PlaybackMode::OneShot); // priority 3
        // Pool full, allocate with higher priority
        let vid = pool.allocate(3, 10, PlaybackMode::OneShot);
        assert!(vid.is_some());
        let stats = pool.stats();
        assert_eq!(stats.total_steals, 1);
    }

    #[test]
    fn test_steal_oldest() {
        let mut pool = AudioPool::new(2);
        pool.set_steal_policy(StealPolicy::Oldest);
        let v1 = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        let _v2 = pool.allocate(2, 5, PlaybackMode::OneShot).unwrap();
        // Pool full, v1 is oldest
        let v3 = pool.allocate(3, 5, PlaybackMode::OneShot);
        assert!(v3.is_some());
        // v1 should be stolen (it was older)
        assert!(pool.get_voice(v1).is_none());
    }

    #[test]
    fn test_steal_quietest() {
        let mut pool = AudioPool::new(2);
        pool.set_steal_policy(StealPolicy::Quietest);
        let v1 = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        let v2 = pool.allocate(2, 5, PlaybackMode::OneShot).unwrap();
        pool.set_volume(v1, 0.1);
        pool.set_volume(v2, 0.9);
        // v1 is quietest → should be stolen
        let v3 = pool.allocate(3, 5, PlaybackMode::OneShot);
        assert!(v3.is_some());
        assert!(pool.get_voice(v1).is_none());
    }

    #[test]
    fn test_cannot_steal_higher_priority() {
        let mut pool = AudioPool::new(2);
        pool.set_steal_policy(StealPolicy::LowestPriority);
        pool.set_max_virtual(0);
        pool.allocate(1, 10, PlaybackMode::OneShot);
        pool.allocate(2, 10, PlaybackMode::OneShot);
        // Try with lower priority
        let vid = pool.allocate(3, 1, PlaybackMode::OneShot);
        assert!(vid.is_none());
    }

    #[test]
    fn test_release_with_fadeout() {
        let mut pool = AudioPool::new(4);
        pool.set_fade_out_samples(100);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.release(vid);
        let voice = pool.get_voice(vid).unwrap();
        assert_eq!(voice.state, VoiceState::Stopping);
        assert_eq!(voice.fade_out_remaining, 100);
    }

    #[test]
    fn test_release_no_fadeout() {
        let mut pool = AudioPool::new(4);
        pool.set_fade_out_samples(0);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.release(vid);
        let voice = pool.get_voice(vid).unwrap();
        assert_eq!(voice.state, VoiceState::Stopped);
    }

    #[test]
    fn test_stop_voice() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.stop(vid);
        let voice = pool.get_voice(vid).unwrap();
        assert_eq!(voice.state, VoiceState::Stopped);
    }

    #[test]
    fn test_pause_resume() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.pause(vid);
        assert_eq!(pool.get_voice(vid).unwrap().state, VoiceState::Paused);
        pool.resume(vid);
        assert_eq!(pool.get_voice(vid).unwrap().state, VoiceState::Playing);
    }

    #[test]
    fn test_set_volume() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.set_volume(vid, 0.7);
        assert!((pool.get_voice(vid).unwrap().volume - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_volume_clamped() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.set_volume(vid, 5.0);
        assert!((pool.get_voice(vid).unwrap().volume - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_advances_position() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.update(256);
        assert_eq!(pool.get_voice(vid).unwrap().position, 256);
    }

    #[test]
    fn test_update_completes_fadeout() {
        let mut pool = AudioPool::new(4);
        pool.set_fade_out_samples(100);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.release(vid);
        pool.update(100);
        let voice = pool.get_voice(vid).unwrap();
        assert_eq!(voice.state, VoiceState::Stopped);
    }

    #[test]
    fn test_fade_out_gain() {
        let mut pool = AudioPool::new(4);
        pool.set_fade_out_samples(100);
        let vid = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.release(vid);
        let gain = AudioPool::fade_out_gain(pool.get_voice(vid).unwrap());
        assert!((gain - 1.0).abs() < 1e-6);
        pool.update(50);
        let gain = AudioPool::fade_out_gain(pool.get_voice(vid).unwrap());
        assert!((gain - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_stop_all() {
        let mut pool = AudioPool::new(4);
        pool.allocate(1, 5, PlaybackMode::OneShot);
        pool.allocate(2, 5, PlaybackMode::OneShot);
        pool.stop_all();
        let stats = pool.stats();
        assert_eq!(stats.active_voices, 0);
        assert_eq!(stats.free_voices, 4);
    }

    #[test]
    fn test_virtual_voices() {
        let mut pool = AudioPool::new(2);
        pool.set_steal_policy(StealPolicy::LowestPriority);
        pool.allocate(1, 10, PlaybackMode::OneShot);
        pool.allocate(2, 10, PlaybackMode::OneShot);
        // This will steal and make the stolen voice virtual
        pool.allocate(3, 15, PlaybackMode::OneShot);
        assert!(pool.virtual_voice_count() > 0);
    }

    #[test]
    fn test_virtual_voice_promotion() {
        let mut pool = AudioPool::new(2);
        pool.set_steal_policy(StealPolicy::LowestPriority);
        let v1 = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        pool.allocate(2, 5, PlaybackMode::OneShot);
        // Steal v1's slot
        pool.allocate(3, 10, PlaybackMode::OneShot);
        assert_eq!(pool.virtual_voice_count(), 1);
        // Stop a voice to free a slot
        pool.stop(v1); // v1 already stolen, but let's stop another
        let active = pool.active_voice_ids();
        if let Some(&first) = active.first() {
            pool.stop(first);
        }
        // Update should promote virtual
        pool.update(1);
        assert_eq!(pool.virtual_voice_count(), 0);
    }

    #[test]
    fn test_active_voice_ids() {
        let mut pool = AudioPool::new(4);
        let v1 = pool.allocate(1, 5, PlaybackMode::OneShot).unwrap();
        let v2 = pool.allocate(2, 5, PlaybackMode::OneShot).unwrap();
        let active = pool.active_voice_ids();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&v1));
        assert!(active.contains(&v2));
    }

    #[test]
    fn test_stats() {
        let mut pool = AudioPool::new(4);
        pool.allocate(1, 5, PlaybackMode::OneShot);
        pool.allocate(2, 5, PlaybackMode::OneShot);
        let vid = pool.allocate(3, 5, PlaybackMode::OneShot).unwrap();
        pool.pause(vid);
        let stats = pool.stats();
        assert_eq!(stats.total_voices, 4);
        assert_eq!(stats.active_voices, 2);
        assert_eq!(stats.paused_voices, 1);
        assert_eq!(stats.free_voices, 1);
        assert_eq!(stats.total_allocations, 3);
    }

    #[test]
    fn test_looping_mode() {
        let mut pool = AudioPool::new(4);
        let vid = pool.allocate(1, 5, PlaybackMode::Looping).unwrap();
        let voice = pool.get_voice(vid).unwrap();
        assert_eq!(voice.mode, PlaybackMode::Looping);
    }

    #[test]
    fn test_pool_stats_struct() {
        let stats = PoolStats {
            total_voices: 32,
            active_voices: 10,
            paused_voices: 2,
            stopping_voices: 1,
            free_voices: 19,
            virtual_voices: 5,
            total_allocations: 100,
            total_steals: 3,
        };
        assert_eq!(stats.total_voices, 32);
        assert_eq!(stats.total_allocations, 100);
    }
}
