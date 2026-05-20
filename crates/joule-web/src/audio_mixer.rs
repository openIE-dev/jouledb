//! Multi-channel audio mixer with volume, pan, mute/solo, groups, and metering.
//!
//! Mixes N channels down to stereo output. Supports channel groups (music,
//! SFX, voice) with group volume, fade over time, peak metering, and
//! clipping detection.

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────

/// Unique channel identifier.
pub type ChannelId = u64;

/// Unique group identifier.
pub type GroupId = u64;

/// Fade curve type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FadeCurve {
    Linear,
    Exponential,
}

/// Fade state for a channel.
#[derive(Debug, Clone, PartialEq)]
pub struct FadeState {
    pub from_volume: f32,
    pub to_volume: f32,
    pub curve: FadeCurve,
    pub duration_samples: usize,
    pub elapsed_samples: usize,
}

impl FadeState {
    /// Current volume of the fade at current elapsed position.
    pub fn current_volume(&self) -> f32 {
        if self.duration_samples == 0 {
            return self.to_volume;
        }
        let t = (self.elapsed_samples as f32 / self.duration_samples as f32).clamp(0.0, 1.0);
        match self.curve {
            FadeCurve::Linear => self.from_volume + (self.to_volume - self.from_volume) * t,
            FadeCurve::Exponential => {
                let from = self.from_volume.max(0.001);
                let to = self.to_volume.max(0.001);
                (from.ln() + (to.ln() - from.ln()) * t).exp()
            }
        }
    }

    /// Whether the fade is complete.
    pub fn is_done(&self) -> bool {
        self.elapsed_samples >= self.duration_samples
    }

    /// Advance by given number of samples.
    pub fn advance(&mut self, samples: usize) {
        self.elapsed_samples = (self.elapsed_samples + samples).min(self.duration_samples);
    }
}

/// A mixer channel.
#[derive(Debug, Clone, PartialEq)]
pub struct MixerChannel {
    pub id: ChannelId,
    pub name: String,
    pub volume: f32,
    pub pan: f32,
    pub mute: bool,
    pub solo: bool,
    pub group_id: Option<GroupId>,
    pub fade: Option<FadeState>,
    pub peak_left: f32,
    pub peak_right: f32,
    pub clipping: bool,
}

impl MixerChannel {
    pub fn new(id: ChannelId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            volume: 1.0,
            pan: 0.0,
            mute: false,
            solo: false,
            group_id: None,
            fade: None,
            peak_left: 0.0,
            peak_right: 0.0,
            clipping: false,
        }
    }

    /// Effective volume accounting for fade.
    pub fn effective_volume(&self) -> f32 {
        match &self.fade {
            Some(fade) => fade.current_volume(),
            None => self.volume,
        }
    }

    /// Compute stereo gain from pan.
    /// Pan: -1.0 = full left, 0.0 = center, 1.0 = full right.
    pub fn stereo_gains(&self) -> (f32, f32) {
        let p = self.pan.clamp(-1.0, 1.0);
        let left = ((1.0 - p) / 2.0).sqrt();
        let right = ((1.0 + p) / 2.0).sqrt();
        (left, right)
    }
}

/// A channel group.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelGroup {
    pub id: GroupId,
    pub name: String,
    pub volume: f32,
    pub mute: bool,
}

impl ChannelGroup {
    pub fn new(id: GroupId, name: &str) -> Self {
        Self { id, name: name.to_string(), volume: 1.0, mute: false }
    }
}

/// The master channel output.
#[derive(Debug, Clone, PartialEq)]
pub struct MasterChannel {
    pub volume: f32,
    pub mute: bool,
    pub peak_left: f32,
    pub peak_right: f32,
    pub clipping: bool,
}

impl Default for MasterChannel {
    fn default() -> Self {
        Self { volume: 1.0, mute: false, peak_left: 0.0, peak_right: 0.0, clipping: false }
    }
}

// ── Audio Mixer ────────────────────────────────────────────────

/// Multi-channel audio mixer.
#[derive(Debug, Clone)]
pub struct AudioMixer {
    channels: HashMap<ChannelId, MixerChannel>,
    groups: HashMap<GroupId, ChannelGroup>,
    master: MasterChannel,
    next_channel_id: ChannelId,
    next_group_id: GroupId,
    sample_rate: u32,
}

impl AudioMixer {
    /// Create a new mixer with the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            channels: HashMap::new(),
            groups: HashMap::new(),
            master: MasterChannel::default(),
            next_channel_id: 1,
            next_group_id: 1,
            sample_rate,
        }
    }

    /// Add a channel to the mixer.
    pub fn add_channel(&mut self, name: &str) -> ChannelId {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        self.channels.insert(id, MixerChannel::new(id, name));
        id
    }

    /// Remove a channel.
    pub fn remove_channel(&mut self, id: ChannelId) -> Option<MixerChannel> {
        self.channels.remove(&id)
    }

    /// Get a reference to a channel.
    pub fn get_channel(&self, id: ChannelId) -> Option<&MixerChannel> {
        self.channels.get(&id)
    }

    /// Get a mutable reference to a channel.
    pub fn get_channel_mut(&mut self, id: ChannelId) -> Option<&mut MixerChannel> {
        self.channels.get_mut(&id)
    }

    /// Set volume for a channel (clamped to 0.0-1.0).
    pub fn set_volume(&mut self, id: ChannelId, volume: f32) -> bool {
        if let Some(ch) = self.channels.get_mut(&id) {
            ch.volume = volume.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Set pan for a channel (clamped to -1.0 to 1.0).
    pub fn set_pan(&mut self, id: ChannelId, pan: f32) -> bool {
        if let Some(ch) = self.channels.get_mut(&id) {
            ch.pan = pan.clamp(-1.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Toggle mute for a channel.
    pub fn set_mute(&mut self, id: ChannelId, mute: bool) -> bool {
        if let Some(ch) = self.channels.get_mut(&id) {
            ch.mute = mute;
            true
        } else {
            false
        }
    }

    /// Toggle solo for a channel.
    pub fn set_solo(&mut self, id: ChannelId, solo: bool) -> bool {
        if let Some(ch) = self.channels.get_mut(&id) {
            ch.solo = solo;
            true
        } else {
            false
        }
    }

    /// Create a channel group.
    pub fn add_group(&mut self, name: &str) -> GroupId {
        let id = self.next_group_id;
        self.next_group_id += 1;
        self.groups.insert(id, ChannelGroup::new(id, name));
        id
    }

    /// Get a reference to a group.
    pub fn get_group(&self, id: GroupId) -> Option<&ChannelGroup> {
        self.groups.get(&id)
    }

    /// Set group volume.
    pub fn set_group_volume(&mut self, id: GroupId, volume: f32) -> bool {
        if let Some(g) = self.groups.get_mut(&id) {
            g.volume = volume.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Set group mute.
    pub fn set_group_mute(&mut self, id: GroupId, mute: bool) -> bool {
        if let Some(g) = self.groups.get_mut(&id) {
            g.mute = mute;
            true
        } else {
            false
        }
    }

    /// Assign a channel to a group.
    pub fn assign_to_group(&mut self, channel_id: ChannelId, group_id: GroupId) -> bool {
        if !self.groups.contains_key(&group_id) { return false; }
        if let Some(ch) = self.channels.get_mut(&channel_id) {
            ch.group_id = Some(group_id);
            true
        } else {
            false
        }
    }

    /// Start a fade on a channel.
    pub fn fade_channel(&mut self, id: ChannelId, to_volume: f32,
                        duration_ms: f32, curve: FadeCurve) -> bool {
        if let Some(ch) = self.channels.get_mut(&id) {
            let duration_samples = (duration_ms / 1000.0 * self.sample_rate as f32) as usize;
            ch.fade = Some(FadeState {
                from_volume: ch.effective_volume(),
                to_volume: to_volume.clamp(0.0, 1.0),
                curve,
                duration_samples,
                elapsed_samples: 0,
            });
            true
        } else {
            false
        }
    }

    /// Get master channel reference.
    pub fn master(&self) -> &MasterChannel {
        &self.master
    }

    /// Set master volume.
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master.volume = volume.clamp(0.0, 1.0);
    }

    /// Set master mute.
    pub fn set_master_mute(&mut self, mute: bool) {
        self.master.mute = mute;
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Mix all channels to a stereo output buffer.
    /// `inputs` maps channel ID to mono sample buffer.
    /// Returns interleaved stereo output [L, R, L, R, ...].
    pub fn mix(&mut self, inputs: &HashMap<ChannelId, Vec<f32>>, frames: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; frames * 2];

        // Determine if any channel has solo enabled
        let any_solo = self.channels.values().any(|ch| ch.solo);

        // Snapshot group data to avoid borrow issues
        let groups: HashMap<GroupId, (f32, bool)> = self.groups.iter()
            .map(|(&gid, g)| (gid, (g.volume, g.mute)))
            .collect();

        for ch in self.channels.values_mut() {
            // Skip muted channels
            if ch.mute { continue; }
            // If any channel is solo'd, skip non-solo channels
            if any_solo && !ch.solo { continue; }
            // Check group mute
            if let Some(gid) = ch.group_id {
                if let Some((_, group_mute)) = groups.get(&gid) {
                    if *group_mute { continue; }
                }
            }

            let input = match inputs.get(&ch.id) {
                Some(buf) => buf,
                None => continue,
            };

            let (gain_l, gain_r) = ch.stereo_gains();
            let group_volume = ch.group_id
                .and_then(|gid| groups.get(&gid))
                .map(|(v, _)| *v)
                .unwrap_or(1.0);

            // Process per-sample with fade
            let mut peak_l: f32 = 0.0;
            let mut peak_r: f32 = 0.0;
            let mut clipping = false;

            for frame in 0..frames {
                let vol = ch.effective_volume() * group_volume;
                let sample = if frame < input.len() { input[frame] } else { 0.0 };
                let left = sample * vol * gain_l;
                let right = sample * vol * gain_r;

                output[frame * 2] += left;
                output[frame * 2 + 1] += right;

                peak_l = peak_l.max(left.abs());
                peak_r = peak_r.max(right.abs());
                if left.abs() > 1.0 || right.abs() > 1.0 {
                    clipping = true;
                }

                // Advance fade
                if let Some(ref mut fade) = ch.fade {
                    fade.advance(1);
                }
            }

            ch.peak_left = peak_l;
            ch.peak_right = peak_r;
            ch.clipping = clipping;

            // If fade is done, commit volume and clear fade
            let fade_done = ch.fade.as_ref().is_some_and(|f| f.is_done());
            if fade_done {
                let final_vol = ch.fade.as_ref().unwrap().to_volume;
                ch.volume = final_vol;
                ch.fade = None;
            }
        }

        // Apply master
        if self.master.mute {
            output.fill(0.0);
            self.master.peak_left = 0.0;
            self.master.peak_right = 0.0;
            self.master.clipping = false;
        } else {
            let mut master_peak_l: f32 = 0.0;
            let mut master_peak_r: f32 = 0.0;
            let mut master_clip = false;
            for frame in 0..frames {
                output[frame * 2] *= self.master.volume;
                output[frame * 2 + 1] *= self.master.volume;
                master_peak_l = master_peak_l.max(output[frame * 2].abs());
                master_peak_r = master_peak_r.max(output[frame * 2 + 1].abs());
                if output[frame * 2].abs() > 1.0 || output[frame * 2 + 1].abs() > 1.0 {
                    master_clip = true;
                }
            }
            self.master.peak_left = master_peak_l;
            self.master.peak_right = master_peak_r;
            self.master.clipping = master_clip;
        }

        output
    }

    /// Reset peak meters on all channels and master.
    pub fn reset_peaks(&mut self) {
        for ch in self.channels.values_mut() {
            ch.peak_left = 0.0;
            ch.peak_right = 0.0;
            ch.clipping = false;
        }
        self.master.peak_left = 0.0;
        self.master.peak_right = 0.0;
        self.master.clipping = false;
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mixer() -> AudioMixer {
        AudioMixer::new(44100)
    }

    #[test]
    fn test_create_mixer() {
        let m = make_mixer();
        assert_eq!(m.channel_count(), 0);
        assert_eq!(m.group_count(), 0);
    }

    #[test]
    fn test_add_channel() {
        let mut m = make_mixer();
        let id = m.add_channel("drums");
        assert_eq!(m.channel_count(), 1);
        let ch = m.get_channel(id).unwrap();
        assert_eq!(ch.name, "drums");
        assert!((ch.volume - 1.0).abs() < 1e-6);
        assert!((ch.pan - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_remove_channel() {
        let mut m = make_mixer();
        let id = m.add_channel("drums");
        let removed = m.remove_channel(id);
        assert!(removed.is_some());
        assert_eq!(m.channel_count(), 0);
    }

    #[test]
    fn test_set_volume() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        m.set_volume(id, 0.5);
        assert!((m.get_channel(id).unwrap().volume - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_volume_clamped() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        m.set_volume(id, 2.0);
        assert!((m.get_channel(id).unwrap().volume - 1.0).abs() < 1e-6);
        m.set_volume(id, -1.0);
        assert!((m.get_channel(id).unwrap().volume - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_pan() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        m.set_pan(id, -0.5);
        assert!((m.get_channel(id).unwrap().pan - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_pan_clamped() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        m.set_pan(id, 5.0);
        assert!((m.get_channel(id).unwrap().pan - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mute_solo() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        m.set_mute(id, true);
        assert!(m.get_channel(id).unwrap().mute);
        m.set_solo(id, true);
        assert!(m.get_channel(id).unwrap().solo);
    }

    #[test]
    fn test_mix_single_channel() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        let input = vec![0.5f32; 4];
        let mut inputs = HashMap::new();
        inputs.insert(id, input);
        let out = m.mix(&inputs, 4);
        assert_eq!(out.len(), 8); // 4 frames * 2 channels
        // Center pan, volume 1.0 → equal left/right
        for frame in 0..4 {
            assert!((out[frame * 2] - out[frame * 2 + 1]).abs() < 1e-6);
            assert!(out[frame * 2] > 0.0);
        }
    }

    #[test]
    fn test_mix_muted_channel() {
        let mut m = make_mixer();
        let id = m.add_channel("ch1");
        m.set_mute(id, true);
        let input = vec![1.0f32; 4];
        let mut inputs = HashMap::new();
        inputs.insert(id, input);
        let out = m.mix(&inputs, 4);
        assert!(out.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn test_solo_behavior() {
        let mut m = make_mixer();
        let ch1 = m.add_channel("ch1");
        let ch2 = m.add_channel("ch2");
        m.set_solo(ch1, true);
        let mut inputs = HashMap::new();
        inputs.insert(ch1, vec![0.5f32; 4]);
        inputs.insert(ch2, vec![0.5f32; 4]);
        let out = m.mix(&inputs, 4);
        // Only ch1 should contribute
        let expected_gain_l = ((1.0f32 - 0.0) / 2.0).sqrt();
        for frame in 0..4 {
            assert!((out[frame * 2] - 0.5 * expected_gain_l).abs() < 1e-5);
        }
    }

    #[test]
    fn test_stereo_panning_left() {
        let ch = MixerChannel::new(1, "test");
        let mut ch_l = ch.clone();
        ch_l.pan = -1.0;
        let (l, r) = ch_l.stereo_gains();
        assert!(l > r + 0.1);
        assert!(r < 0.1);
    }

    #[test]
    fn test_stereo_panning_right() {
        let mut ch = MixerChannel::new(1, "test");
        ch.pan = 1.0;
        let (l, r) = ch.stereo_gains();
        assert!(r > l + 0.1);
        assert!(l < 0.1);
    }

    #[test]
    fn test_channel_group() {
        let mut m = make_mixer();
        let gid = m.add_group("sfx");
        let ch = m.add_channel("explosion");
        assert!(m.assign_to_group(ch, gid));
        assert_eq!(m.get_channel(ch).unwrap().group_id, Some(gid));
    }

    #[test]
    fn test_group_volume() {
        let mut m = make_mixer();
        let gid = m.add_group("music");
        m.set_group_volume(gid, 0.5);
        let ch = m.add_channel("bgm");
        m.assign_to_group(ch, gid);

        let mut inputs = HashMap::new();
        inputs.insert(ch, vec![1.0f32; 4]);
        let out = m.mix(&inputs, 4);
        // Volume 1.0 * group 0.5 → ~0.5 effective, split across stereo
        let max_sample = out.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_sample < 0.6);
    }

    #[test]
    fn test_group_mute() {
        let mut m = make_mixer();
        let gid = m.add_group("sfx");
        m.set_group_mute(gid, true);
        let ch = m.add_channel("shot");
        m.assign_to_group(ch, gid);

        let mut inputs = HashMap::new();
        inputs.insert(ch, vec![1.0f32; 4]);
        let out = m.mix(&inputs, 4);
        assert!(out.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn test_master_volume() {
        let mut m = make_mixer();
        m.set_master_volume(0.25);
        let ch = m.add_channel("ch1");
        let mut inputs = HashMap::new();
        inputs.insert(ch, vec![1.0f32; 4]);
        let out = m.mix(&inputs, 4);
        let max_sample = out.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_sample < 0.3);
    }

    #[test]
    fn test_master_mute() {
        let mut m = make_mixer();
        m.set_master_mute(true);
        let ch = m.add_channel("ch1");
        let mut inputs = HashMap::new();
        inputs.insert(ch, vec![1.0f32; 4]);
        let out = m.mix(&inputs, 4);
        assert!(out.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn test_clipping_detection() {
        let mut m = make_mixer();
        let ch1 = m.add_channel("ch1");
        let ch2 = m.add_channel("ch2");
        let mut inputs = HashMap::new();
        inputs.insert(ch1, vec![0.9f32; 4]);
        inputs.insert(ch2, vec![0.9f32; 4]);
        m.mix(&inputs, 4);
        assert!(m.master().clipping);
    }

    #[test]
    fn test_peak_metering() {
        let mut m = make_mixer();
        let ch = m.add_channel("ch1");
        let input = vec![0.0, 0.3, 0.7, 0.2];
        let mut inputs = HashMap::new();
        inputs.insert(ch, input);
        m.mix(&inputs, 4);
        assert!(m.get_channel(ch).unwrap().peak_left > 0.4);
    }

    #[test]
    fn test_fade_linear() {
        let mut fade = FadeState {
            from_volume: 1.0,
            to_volume: 0.0,
            curve: FadeCurve::Linear,
            duration_samples: 100,
            elapsed_samples: 50,
        };
        let v = fade.current_volume();
        assert!((v - 0.5).abs() < 1e-5);
        fade.advance(50);
        assert!(fade.is_done());
    }

    #[test]
    fn test_fade_exponential() {
        let fade = FadeState {
            from_volume: 1.0,
            to_volume: 0.01,
            curve: FadeCurve::Exponential,
            duration_samples: 100,
            elapsed_samples: 50,
        };
        let v = fade.current_volume();
        // Exponential midpoint should be geometric mean
        assert!(v > 0.01 && v < 1.0);
    }

    #[test]
    fn test_reset_peaks() {
        let mut m = make_mixer();
        let ch = m.add_channel("ch1");
        let mut inputs = HashMap::new();
        inputs.insert(ch, vec![0.5f32; 4]);
        m.mix(&inputs, 4);
        assert!(m.get_channel(ch).unwrap().peak_left > 0.0);
        m.reset_peaks();
        assert!((m.get_channel(ch).unwrap().peak_left - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fade_channel() {
        let mut m = make_mixer();
        let ch = m.add_channel("ch1");
        m.fade_channel(ch, 0.0, 100.0, FadeCurve::Linear);
        assert!(m.get_channel(ch).unwrap().fade.is_some());
    }

    #[test]
    fn test_mix_no_input_for_channel() {
        let mut m = make_mixer();
        m.add_channel("ch1");
        let inputs = HashMap::new();
        let out = m.mix(&inputs, 4);
        assert!(out.iter().all(|s| s.abs() < 1e-6));
    }
}
