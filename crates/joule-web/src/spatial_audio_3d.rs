//! 3D positional audio: listener, sources, distance attenuation, Doppler,
//! stereo panning, occlusion, and priority-based voice limiting.

use std::collections::HashMap;

// ── Vector Math ────────────────────────────────────────────────

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::zero()
        } else {
            Self { x: self.x / len, y: self.y / len, z: self.z / len }
        }
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn scale(&self, s: f64) -> Vec3 {
        Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

// ── Listener ───────────────────────────────────────────────────

/// The audio listener (camera/head).
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub velocity: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::zero(),
            forward: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            velocity: Vec3::zero(),
        }
    }
}

impl Listener {
    /// Right vector derived from forward and up.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(&self.up).normalized()
    }
}

// ── Distance Attenuation ───────────────────────────────────────

/// Distance attenuation model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceModel {
    /// Linear: gain = 1 - rolloff * (distance - ref) / (max - ref)
    Linear { ref_distance: f64, max_distance: f64, rolloff: f64 },
    /// Inverse: gain = ref / (ref + rolloff * (distance - ref))
    Inverse { ref_distance: f64, rolloff: f64 },
    /// Exponential: gain = (distance / ref)^(-rolloff)
    Exponential { ref_distance: f64, rolloff: f64 },
}

impl DistanceModel {
    /// Compute gain for a given distance.
    pub fn attenuation(&self, distance: f64) -> f64 {
        match self {
            DistanceModel::Linear { ref_distance, max_distance, rolloff } => {
                let d = distance.clamp(*ref_distance, *max_distance);
                let range = *max_distance - *ref_distance;
                if range < 1e-12 { return 1.0; }
                (1.0 - rolloff * (d - ref_distance) / range).max(0.0)
            }
            DistanceModel::Inverse { ref_distance, rolloff } => {
                let d = distance.max(*ref_distance);
                *ref_distance / (*ref_distance + rolloff * (d - ref_distance))
            }
            DistanceModel::Exponential { ref_distance, rolloff } => {
                let d = distance.max(*ref_distance);
                if *ref_distance < 1e-12 { return 0.0; }
                (d / ref_distance).powf(-rolloff)
            }
        }
    }
}

// ── Audio Source ────────────────────────────────────────────────

/// Unique source identifier.
pub type SourceId = u64;

/// Cone directionality for a source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceCone {
    /// Inner cone half-angle in radians (full volume within).
    pub inner_angle: f64,
    /// Outer cone half-angle in radians (attenuated beyond).
    pub outer_angle: f64,
    /// Gain at the outer boundary (0.0-1.0).
    pub outer_gain: f64,
}

/// A 3D audio source.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSource3D {
    pub id: SourceId,
    pub position: Vec3,
    pub velocity: Vec3,
    pub direction: Vec3,
    pub cone: Option<SourceCone>,
    pub distance_model: DistanceModel,
    pub volume: f64,
    pub priority: u32,
    pub occlusion: f64,
    pub active: bool,
}

impl AudioSource3D {
    pub fn new(id: SourceId, position: Vec3) -> Self {
        Self {
            id,
            position,
            velocity: Vec3::zero(),
            direction: Vec3::new(0.0, 0.0, -1.0),
            cone: None,
            distance_model: DistanceModel::Inverse { ref_distance: 1.0, rolloff: 1.0 },
            volume: 1.0,
            priority: 0,
            occlusion: 0.0,
            active: true,
        }
    }
}

// ── Spatial Result ─────────────────────────────────────────────

/// Computed spatial parameters for a source.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialResult {
    pub source_id: SourceId,
    pub gain: f64,
    pub pan: f64,
    pub doppler_pitch: f64,
    pub distance: f64,
}

// ── Spatial Audio Engine ───────────────────────────────────────

/// Speed of sound in meters per second.
const SPEED_OF_SOUND: f64 = 343.0;

/// 3D spatial audio engine.
#[derive(Debug, Clone)]
pub struct SpatialAudio3D {
    listener: Listener,
    sources: HashMap<SourceId, AudioSource3D>,
    next_source_id: SourceId,
    max_voices: usize,
    doppler_factor: f64,
}

impl SpatialAudio3D {
    /// Create with a voice limit.
    pub fn new(max_voices: usize) -> Self {
        Self {
            listener: Listener::default(),
            sources: HashMap::new(),
            next_source_id: 1,
            max_voices,
            doppler_factor: 1.0,
        }
    }

    /// Set the listener state.
    pub fn set_listener(&mut self, listener: Listener) {
        self.listener = listener;
    }

    /// Get the listener.
    pub fn listener(&self) -> &Listener {
        &self.listener
    }

    /// Set doppler factor (0 = disabled).
    pub fn set_doppler_factor(&mut self, factor: f64) {
        self.doppler_factor = factor.max(0.0);
    }

    /// Add a source, returning its ID.
    pub fn add_source(&mut self, position: Vec3) -> SourceId {
        let id = self.next_source_id;
        self.next_source_id += 1;
        self.sources.insert(id, AudioSource3D::new(id, position));
        id
    }

    /// Remove a source.
    pub fn remove_source(&mut self, id: SourceId) -> Option<AudioSource3D> {
        self.sources.remove(&id)
    }

    /// Get a source by ID.
    pub fn get_source(&self, id: SourceId) -> Option<&AudioSource3D> {
        self.sources.get(&id)
    }

    /// Get a mutable source by ID.
    pub fn get_source_mut(&mut self, id: SourceId) -> Option<&mut AudioSource3D> {
        self.sources.get_mut(&id)
    }

    /// Number of sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Compute azimuth angle (radians) from listener to source.
    pub fn compute_azimuth(&self, source_pos: &Vec3) -> f64 {
        let to_source = source_pos.sub(&self.listener.position);
        let right = self.listener.right();
        let fwd = self.listener.forward.normalized();
        let projected_fwd = to_source.dot(&fwd);
        let projected_right = to_source.dot(&right);
        projected_right.atan2(projected_fwd)
    }

    /// Compute elevation angle (radians) from listener to source.
    pub fn compute_elevation(&self, source_pos: &Vec3) -> f64 {
        let to_source = source_pos.sub(&self.listener.position);
        let len = to_source.length();
        if len < 1e-12 { return 0.0; }
        let up = self.listener.up.normalized();
        let proj_up = to_source.dot(&up);
        (proj_up / len).clamp(-1.0, 1.0).asin()
    }

    /// Compute stereo pan from azimuth: -1.0 = left, 1.0 = right.
    pub fn azimuth_to_pan(azimuth: f64) -> f64 {
        azimuth.sin().clamp(-1.0, 1.0)
    }

    /// Compute doppler pitch shift.
    pub fn compute_doppler(&self, source: &AudioSource3D) -> f64 {
        if self.doppler_factor < 1e-12 { return 1.0; }
        let to_listener = self.listener.position.sub(&source.position);
        let dist = to_listener.length();
        if dist < 1e-12 { return 1.0; }
        let dir = to_listener.normalized();
        let v_listener = self.listener.velocity.dot(&dir);
        let v_source = source.velocity.dot(&dir);

        let vs = SPEED_OF_SOUND;
        // Standard Doppler: pitch = (c + v_listener_toward) / (c - v_source_toward)
        // v_source is positive when source moves toward listener (along dir),
        // so negate for the denominator.
        let num = vs + v_listener * self.doppler_factor;
        let den = vs - v_source * self.doppler_factor;
        if den.abs() < 1e-12 { return 1.0; }
        (num / den).clamp(0.5, 2.0)
    }

    /// Compute cone gain for a directional source.
    pub fn compute_cone_gain(source: &AudioSource3D, listener_pos: &Vec3) -> f64 {
        let cone = match &source.cone {
            Some(c) => c,
            None => return 1.0,
        };
        let to_listener = listener_pos.sub(&source.position);
        let dist = to_listener.length();
        if dist < 1e-12 { return 1.0; }
        let dir = to_listener.normalized();
        let src_dir = source.direction.normalized();
        let angle = dir.dot(&src_dir).clamp(-1.0, 1.0).acos();

        if angle <= cone.inner_angle {
            1.0
        } else if angle >= cone.outer_angle {
            cone.outer_gain
        } else {
            let range = cone.outer_angle - cone.inner_angle;
            if range < 1e-12 { return 1.0; }
            let t = (angle - cone.inner_angle) / range;
            1.0 + (cone.outer_gain - 1.0) * t
        }
    }

    /// Process all sources, returning spatial parameters for active voices.
    /// Priority-limited to max_voices.
    pub fn process(&self) -> Vec<SpatialResult> {
        let mut active: Vec<&AudioSource3D> = self.sources.values()
            .filter(|s| s.active)
            .collect();

        // Sort by priority (descending), then by distance (ascending)
        let listener_pos = self.listener.position;
        active.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| {
                    let da = a.position.sub(&listener_pos).length();
                    let db = b.position.sub(&listener_pos).length();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        active.truncate(self.max_voices);

        active.iter().map(|src| {
            let distance = src.position.sub(&self.listener.position).length();
            let dist_gain = src.distance_model.attenuation(distance);
            let cone_gain = Self::compute_cone_gain(src, &self.listener.position);
            let occlusion_gain = 1.0 - src.occlusion.clamp(0.0, 1.0);
            let gain = src.volume * dist_gain * cone_gain * occlusion_gain;
            let azimuth = self.compute_azimuth(&src.position);
            let pan = Self::azimuth_to_pan(azimuth);
            let doppler_pitch = self.compute_doppler(src);

            SpatialResult {
                source_id: src.id,
                gain,
                pan,
                doppler_pitch,
                distance,
            }
        }).collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(0.0, 0.0, 5.0);
        let n = v.normalized();
        assert!((n.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(a.dot(&b).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_listener_right() {
        let listener = Listener::default();
        let r = listener.right();
        // forward (0,0,-1) cross up (0,1,0) = (1,0,0)
        assert!((r.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_linear() {
        let model = DistanceModel::Linear { ref_distance: 1.0, max_distance: 10.0, rolloff: 1.0 };
        assert!((model.attenuation(1.0) - 1.0).abs() < 1e-10);
        assert!((model.attenuation(10.0) - 0.0).abs() < 1e-10);
        assert!((model.attenuation(5.5) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_distance_inverse() {
        let model = DistanceModel::Inverse { ref_distance: 1.0, rolloff: 1.0 };
        assert!((model.attenuation(1.0) - 1.0).abs() < 1e-10);
        assert!((model.attenuation(2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_distance_exponential() {
        let model = DistanceModel::Exponential { ref_distance: 1.0, rolloff: 2.0 };
        assert!((model.attenuation(1.0) - 1.0).abs() < 1e-10);
        assert!((model.attenuation(2.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_add_remove_source() {
        let mut sa = SpatialAudio3D::new(32);
        let id = sa.add_source(Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(sa.source_count(), 1);
        sa.remove_source(id);
        assert_eq!(sa.source_count(), 0);
    }

    #[test]
    fn test_azimuth_front() {
        let sa = SpatialAudio3D::new(32);
        // Source directly in front
        let az = sa.compute_azimuth(&Vec3::new(0.0, 0.0, -5.0));
        assert!(az.abs() < 0.1);
    }

    #[test]
    fn test_azimuth_right() {
        let sa = SpatialAudio3D::new(32);
        // Source to the right
        let az = sa.compute_azimuth(&Vec3::new(5.0, 0.0, 0.0));
        assert!((az - PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_pan_from_azimuth() {
        assert!((SpatialAudio3D::azimuth_to_pan(0.0) - 0.0).abs() < 1e-10);
        assert!((SpatialAudio3D::azimuth_to_pan(PI / 2.0) - 1.0).abs() < 1e-6);
        assert!((SpatialAudio3D::azimuth_to_pan(-PI / 2.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_doppler_stationary() {
        let sa = SpatialAudio3D::new(32);
        let src = AudioSource3D::new(1, Vec3::new(5.0, 0.0, 0.0));
        let pitch = sa.compute_doppler(&src);
        assert!((pitch - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_doppler_approaching() {
        let mut sa = SpatialAudio3D::new(32);
        sa.set_doppler_factor(1.0);
        let mut src = AudioSource3D::new(1, Vec3::new(100.0, 0.0, 0.0));
        src.velocity = Vec3::new(-50.0, 0.0, 0.0); // approaching
        let pitch = sa.compute_doppler(&src);
        assert!(pitch > 1.0);
    }

    #[test]
    fn test_doppler_receding() {
        let mut sa = SpatialAudio3D::new(32);
        sa.set_doppler_factor(1.0);
        let mut src = AudioSource3D::new(1, Vec3::new(100.0, 0.0, 0.0));
        src.velocity = Vec3::new(50.0, 0.0, 0.0); // receding
        let pitch = sa.compute_doppler(&src);
        assert!(pitch < 1.0);
    }

    #[test]
    fn test_cone_gain_inside_inner() {
        let mut src = AudioSource3D::new(1, Vec3::zero());
        src.direction = Vec3::new(0.0, 0.0, -1.0);
        src.cone = Some(SourceCone {
            inner_angle: PI / 4.0,
            outer_angle: PI / 2.0,
            outer_gain: 0.0,
        });
        let listener_pos = Vec3::new(0.0, 0.0, -5.0);
        let gain = SpatialAudio3D::compute_cone_gain(&src, &listener_pos);
        assert!((gain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cone_gain_outside_outer() {
        let mut src = AudioSource3D::new(1, Vec3::zero());
        src.direction = Vec3::new(0.0, 0.0, -1.0);
        src.cone = Some(SourceCone {
            inner_angle: PI / 4.0,
            outer_angle: PI / 2.0,
            outer_gain: 0.0,
        });
        let listener_pos = Vec3::new(0.0, 0.0, 5.0); // behind
        let gain = SpatialAudio3D::compute_cone_gain(&src, &listener_pos);
        assert!((gain - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_occlusion() {
        let mut sa = SpatialAudio3D::new(32);
        let id = sa.add_source(Vec3::new(5.0, 0.0, 0.0));
        sa.get_source_mut(id).unwrap().occlusion = 0.5;
        let results = sa.process();
        let r = results.iter().find(|r| r.source_id == id).unwrap();
        // 50% occlusion → gain reduced
        assert!(r.gain < 0.6);
    }

    #[test]
    fn test_priority_voice_limit() {
        let mut sa = SpatialAudio3D::new(2);
        let id1 = sa.add_source(Vec3::new(1.0, 0.0, 0.0));
        sa.get_source_mut(id1).unwrap().priority = 10;
        let id2 = sa.add_source(Vec3::new(2.0, 0.0, 0.0));
        sa.get_source_mut(id2).unwrap().priority = 5;
        let _id3 = sa.add_source(Vec3::new(3.0, 0.0, 0.0));
        // Only 2 voices max, third should be dropped
        let results = sa.process();
        assert_eq!(results.len(), 2);
        // Highest priority should be present
        assert!(results.iter().any(|r| r.source_id == id1));
    }

    #[test]
    fn test_elevation() {
        let sa = SpatialAudio3D::new(32);
        let elev = sa.compute_elevation(&Vec3::new(0.0, 5.0, 0.0));
        assert!((elev - PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_process_inactive_source() {
        let mut sa = SpatialAudio3D::new(32);
        let id = sa.add_source(Vec3::new(5.0, 0.0, 0.0));
        sa.get_source_mut(id).unwrap().active = false;
        let results = sa.process();
        assert!(results.is_empty());
    }

    #[test]
    fn test_distance_zero() {
        let sa = SpatialAudio3D::new(32);
        let az = sa.compute_azimuth(&Vec3::zero());
        // At same position, result defined (atan2(0,0) = 0)
        assert!(az.abs() < 1e-6);
    }
}
