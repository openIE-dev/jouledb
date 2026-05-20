// gpu_particles.rs — High-count particle system with CPU-side compute simulation
// Part of joule-web: Particles & VFX cluster

use std::collections::VecDeque;

/// 3D vector for particle math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    pub fn distance_sq(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// Single particle in the pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub age: f32,
    pub size: f32,
    pub color: Color,
    pub rotation: f32,
    pub alive: bool,
    pub energy_cost_uj: f64,
}

impl Particle {
    pub fn dead() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            lifetime: 0.0,
            age: 0.0,
            size: 0.0,
            color: Color::WHITE,
            rotation: 0.0,
            alive: false,
            energy_cost_uj: 0.0,
        }
    }

    /// Normalized age in [0, 1].
    pub fn normalized_age(&self) -> f32 {
        if self.lifetime <= 0.0 { 1.0 } else { (self.age / self.lifetime).min(1.0) }
    }
}

/// Spawn mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpawnMode {
    /// Continuous emission at a fixed rate (particles per second).
    Continuous { rate: f32 },
    /// Instantaneous burst of N particles.
    Burst { count: u32 },
}

/// Template for newly spawned particles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpawnParams {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub size: f32,
    pub color: Color,
    pub rotation: f32,
}

impl Default for SpawnParams {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::new(0.0, 1.0, 0.0),
            lifetime: 2.0,
            size: 0.1,
            color: Color::WHITE,
            rotation: 0.0,
        }
    }
}

/// Sort criterion for alpha-blended particles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortMode {
    /// No sorting.
    None,
    /// Sort by distance to camera (back-to-front).
    BackToFront,
    /// Sort by distance to camera (front-to-back).
    FrontToBack,
    /// Sort by age (oldest first).
    OldestFirst,
}

/// Energy tracking for the entire system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyStats {
    pub total_energy_uj: f64,
    pub spawn_energy_uj: f64,
    pub update_energy_uj: f64,
    pub sort_energy_uj: f64,
}

impl EnergyStats {
    pub const ZERO: Self = Self {
        total_energy_uj: 0.0,
        spawn_energy_uj: 0.0,
        update_energy_uj: 0.0,
        sort_energy_uj: 0.0,
    };
}

/// GPU-style particle system simulated on CPU.
pub struct GpuParticleSystem {
    particles: Vec<Particle>,
    dead_list: VecDeque<usize>,
    capacity: usize,
    alive_count: usize,
    sort_mode: SortMode,
    sorted_indices: Vec<usize>,
    emission_accumulator: f32,
    energy_stats: EnergyStats,
    /// Cost in microjoules per particle per spawn.
    energy_per_spawn_uj: f64,
    /// Cost in microjoules per particle per update step.
    energy_per_update_uj: f64,
    /// Cost in microjoules per comparison during sort.
    energy_per_sort_cmp_uj: f64,
    gravity: Vec3,
}

impl GpuParticleSystem {
    /// Create a particle system with given fixed capacity.
    pub fn new(capacity: usize) -> Self {
        let mut dead_list = VecDeque::with_capacity(capacity);
        for i in (0..capacity).rev() {
            dead_list.push_back(i);
        }
        Self {
            particles: vec![Particle::dead(); capacity],
            dead_list,
            capacity,
            alive_count: 0,
            sort_mode: SortMode::None,
            sorted_indices: Vec::new(),
            emission_accumulator: 0.0,
            energy_stats: EnergyStats::ZERO,
            energy_per_spawn_uj: 0.05,
            energy_per_update_uj: 0.02,
            energy_per_sort_cmp_uj: 0.001,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn alive_count(&self) -> usize {
        self.alive_count
    }

    pub fn dead_count(&self) -> usize {
        self.dead_list.len()
    }

    pub fn set_sort_mode(&mut self, mode: SortMode) {
        self.sort_mode = mode;
    }

    pub fn sort_mode(&self) -> SortMode {
        self.sort_mode
    }

    pub fn set_gravity(&mut self, g: Vec3) {
        self.gravity = g;
    }

    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn energy_stats(&self) -> EnergyStats {
        self.energy_stats
    }

    pub fn reset_energy_stats(&mut self) {
        self.energy_stats = EnergyStats::ZERO;
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn sorted_indices(&self) -> &[usize] {
        &self.sorted_indices
    }

    /// Spawn a single particle if capacity allows. Returns the index or None.
    pub fn spawn_one(&mut self, params: &SpawnParams) -> Option<usize> {
        let idx = self.dead_list.pop_front()?;
        self.particles[idx] = Particle {
            position: params.position,
            velocity: params.velocity,
            lifetime: params.lifetime,
            age: 0.0,
            size: params.size,
            color: params.color,
            rotation: params.rotation,
            alive: true,
            energy_cost_uj: 0.0,
        };
        self.alive_count += 1;
        self.energy_stats.spawn_energy_uj += self.energy_per_spawn_uj;
        self.energy_stats.total_energy_uj += self.energy_per_spawn_uj;
        self.particles[idx].energy_cost_uj += self.energy_per_spawn_uj;
        Some(idx)
    }

    /// Spawn a burst of particles. Returns how many were actually spawned.
    pub fn spawn_burst(&mut self, params: &SpawnParams, count: u32) -> u32 {
        let mut spawned = 0u32;
        for _ in 0..count {
            if self.spawn_one(params).is_some() {
                spawned += 1;
            } else {
                break;
            }
        }
        spawned
    }

    /// Continuous emission: accumulates fractional particles from rate*dt.
    pub fn emit_continuous(&mut self, params: &SpawnParams, rate: f32, dt: f32) -> u32 {
        self.emission_accumulator += rate * dt;
        let to_spawn = self.emission_accumulator.floor() as u32;
        self.emission_accumulator -= to_spawn as f32;
        self.spawn_burst(params, to_spawn)
    }

    /// Update all alive particles by dt seconds.
    pub fn update(&mut self, dt: f32) {
        let gravity = self.gravity;
        let cost_per = self.energy_per_update_uj;

        for i in 0..self.capacity {
            if !self.particles[i].alive {
                continue;
            }
            let p = &mut self.particles[i];
            // Integrate velocity with gravity
            p.velocity = p.velocity + gravity * dt;
            p.position = p.position + p.velocity * dt;
            p.age += dt;
            p.energy_cost_uj += cost_per;

            // Kill expired
            if p.age >= p.lifetime {
                p.alive = false;
                self.dead_list.push_back(i);
                self.alive_count -= 1;
            }
        }
        let energy = cost_per * self.alive_count as f64;
        self.energy_stats.update_energy_uj += energy;
        self.energy_stats.total_energy_uj += energy;
    }

    /// Sort alive particles for alpha blending relative to a camera position.
    pub fn sort(&mut self, camera_pos: Vec3) {
        self.sorted_indices.clear();
        for i in 0..self.capacity {
            if self.particles[i].alive {
                self.sorted_indices.push(i);
            }
        }

        let particles = &self.particles;
        let mut cmp_count = 0u64;

        match self.sort_mode {
            SortMode::None => {}
            SortMode::BackToFront => {
                self.sorted_indices.sort_by(|&a, &b| {
                    cmp_count += 1;
                    let da = particles[a].position.distance_sq(camera_pos);
                    let db = particles[b].position.distance_sq(camera_pos);
                    db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortMode::FrontToBack => {
                self.sorted_indices.sort_by(|&a, &b| {
                    cmp_count += 1;
                    let da = particles[a].position.distance_sq(camera_pos);
                    let db = particles[b].position.distance_sq(camera_pos);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortMode::OldestFirst => {
                self.sorted_indices.sort_by(|&a, &b| {
                    cmp_count += 1;
                    particles[b]
                        .age
                        .partial_cmp(&particles[a].age)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let sort_energy = self.energy_per_sort_cmp_uj * cmp_count as f64;
        self.energy_stats.sort_energy_uj += sort_energy;
        self.energy_stats.total_energy_uj += sort_energy;
    }

    /// Kill all particles, returning them to the dead list.
    pub fn clear(&mut self) {
        for i in 0..self.capacity {
            if self.particles[i].alive {
                self.particles[i].alive = false;
                self.dead_list.push_back(i);
            }
        }
        self.alive_count = 0;
        self.sorted_indices.clear();
        self.emission_accumulator = 0.0;
    }

    /// Get a specific particle by index.
    pub fn get_particle(&self, idx: usize) -> Option<&Particle> {
        self.particles.get(idx)
    }

    /// Iterate over alive particles.
    pub fn alive_particles(&self) -> impl Iterator<Item = (usize, &Particle)> {
        self.particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.alive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> SpawnParams {
        SpawnParams::default()
    }

    #[test]
    fn test_new_system_capacity() {
        let sys = GpuParticleSystem::new(1000);
        assert_eq!(sys.capacity(), 1000);
        assert_eq!(sys.alive_count(), 0);
        assert_eq!(sys.dead_count(), 1000);
    }

    #[test]
    fn test_spawn_one_particle() {
        let mut sys = GpuParticleSystem::new(100);
        let idx = sys.spawn_one(&default_params());
        assert!(idx.is_some());
        assert_eq!(sys.alive_count(), 1);
        assert_eq!(sys.dead_count(), 99);
        let p = sys.get_particle(idx.unwrap()).unwrap();
        assert!(p.alive);
        assert!((p.age - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_spawn_burst() {
        let mut sys = GpuParticleSystem::new(50);
        let spawned = sys.spawn_burst(&default_params(), 30);
        assert_eq!(spawned, 30);
        assert_eq!(sys.alive_count(), 30);
    }

    #[test]
    fn test_spawn_burst_over_capacity() {
        let mut sys = GpuParticleSystem::new(10);
        let spawned = sys.spawn_burst(&default_params(), 20);
        assert_eq!(spawned, 10);
        assert_eq!(sys.alive_count(), 10);
        assert_eq!(sys.dead_count(), 0);
    }

    #[test]
    fn test_spawn_returns_none_at_capacity() {
        let mut sys = GpuParticleSystem::new(1);
        assert!(sys.spawn_one(&default_params()).is_some());
        assert!(sys.spawn_one(&default_params()).is_none());
    }

    #[test]
    fn test_continuous_emission() {
        let mut sys = GpuParticleSystem::new(100);
        // 10 particles/sec * 0.5s = 5 particles
        let spawned = sys.emit_continuous(&default_params(), 10.0, 0.5);
        assert_eq!(spawned, 5);
        assert_eq!(sys.alive_count(), 5);
    }

    #[test]
    fn test_continuous_emission_accumulator() {
        let mut sys = GpuParticleSystem::new(100);
        // 3 particles/sec * 0.1s = 0.3 -> 0 particles, accumulator = 0.3
        let s1 = sys.emit_continuous(&default_params(), 3.0, 0.1);
        assert_eq!(s1, 0);
        // Another 0.1s -> 0.3+0.3 = 0.6 -> 0 particles
        let s2 = sys.emit_continuous(&default_params(), 3.0, 0.1);
        assert_eq!(s2, 0);
        // Another 0.2s -> 0.6+0.6 = 1.2 -> 1 particle, accumulator = 0.2
        let s3 = sys.emit_continuous(&default_params(), 3.0, 0.2);
        assert_eq!(s3, 1);
    }

    #[test]
    fn test_update_integrates_velocity() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_gravity(Vec3::ZERO);
        let params = SpawnParams {
            velocity: Vec3::new(1.0, 0.0, 0.0),
            lifetime: 5.0,
            ..Default::default()
        };
        let idx = sys.spawn_one(&params).unwrap();
        sys.update(1.0);
        let p = sys.get_particle(idx).unwrap();
        assert!((p.position.x - 1.0).abs() < 1e-5);
        assert!((p.age - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_applies_gravity() {
        let mut sys = GpuParticleSystem::new(10);
        let params = SpawnParams {
            velocity: Vec3::ZERO,
            lifetime: 5.0,
            ..Default::default()
        };
        let idx = sys.spawn_one(&params).unwrap();
        sys.update(1.0);
        let p = sys.get_particle(idx).unwrap();
        // After 1s of gravity: vy = -9.81, y = -9.81
        assert!((p.velocity.y - (-9.81)).abs() < 1e-3);
        assert!((p.position.y - (-9.81)).abs() < 1e-3);
    }

    #[test]
    fn test_particle_dies_when_age_exceeds_lifetime() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_gravity(Vec3::ZERO);
        let params = SpawnParams { lifetime: 1.0, ..Default::default() };
        sys.spawn_one(&params);
        assert_eq!(sys.alive_count(), 1);
        sys.update(1.5);
        assert_eq!(sys.alive_count(), 0);
        assert_eq!(sys.dead_count(), 10);
    }

    #[test]
    fn test_dead_particles_recycled() {
        let mut sys = GpuParticleSystem::new(2);
        sys.set_gravity(Vec3::ZERO);
        let params = SpawnParams { lifetime: 0.5, ..Default::default() };
        sys.spawn_burst(&params, 2);
        assert_eq!(sys.alive_count(), 2);
        assert_eq!(sys.dead_count(), 0);
        // Kill them
        sys.update(1.0);
        assert_eq!(sys.alive_count(), 0);
        assert_eq!(sys.dead_count(), 2);
        // Respawn
        let spawned = sys.spawn_burst(&params, 2);
        assert_eq!(spawned, 2);
        assert_eq!(sys.alive_count(), 2);
    }

    #[test]
    fn test_sort_back_to_front() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_gravity(Vec3::ZERO);
        sys.set_sort_mode(SortMode::BackToFront);
        let near = SpawnParams {
            position: Vec3::new(1.0, 0.0, 0.0),
            lifetime: 5.0,
            ..Default::default()
        };
        let far = SpawnParams {
            position: Vec3::new(10.0, 0.0, 0.0),
            lifetime: 5.0,
            ..Default::default()
        };
        let near_idx = sys.spawn_one(&near).unwrap();
        let far_idx = sys.spawn_one(&far).unwrap();
        sys.sort(Vec3::ZERO);
        let indices = sys.sorted_indices();
        assert_eq!(indices.len(), 2);
        // Farthest first for back-to-front
        assert_eq!(indices[0], far_idx);
        assert_eq!(indices[1], near_idx);
    }

    #[test]
    fn test_sort_front_to_back() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_gravity(Vec3::ZERO);
        sys.set_sort_mode(SortMode::FrontToBack);
        let near = SpawnParams {
            position: Vec3::new(1.0, 0.0, 0.0),
            lifetime: 5.0,
            ..Default::default()
        };
        let far = SpawnParams {
            position: Vec3::new(10.0, 0.0, 0.0),
            lifetime: 5.0,
            ..Default::default()
        };
        let near_idx = sys.spawn_one(&near).unwrap();
        let far_idx = sys.spawn_one(&far).unwrap();
        sys.sort(Vec3::ZERO);
        let indices = sys.sorted_indices();
        assert_eq!(indices[0], near_idx);
        assert_eq!(indices[1], far_idx);
    }

    #[test]
    fn test_sort_oldest_first() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_gravity(Vec3::ZERO);
        sys.set_sort_mode(SortMode::OldestFirst);
        let params = SpawnParams { lifetime: 10.0, ..Default::default() };
        let first = sys.spawn_one(&params).unwrap();
        sys.update(1.0);
        let second = sys.spawn_one(&params).unwrap();
        sys.sort(Vec3::ZERO);
        let indices = sys.sorted_indices();
        assert_eq!(indices[0], first);
        assert_eq!(indices[1], second);
    }

    #[test]
    fn test_sort_mode_none() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_sort_mode(SortMode::None);
        sys.spawn_burst(&default_params(), 5);
        sys.sort(Vec3::ZERO);
        assert_eq!(sys.sorted_indices().len(), 5);
    }

    #[test]
    fn test_clear() {
        let mut sys = GpuParticleSystem::new(50);
        sys.spawn_burst(&default_params(), 30);
        assert_eq!(sys.alive_count(), 30);
        sys.clear();
        assert_eq!(sys.alive_count(), 0);
        assert_eq!(sys.dead_count(), 50);
    }

    #[test]
    fn test_energy_tracking_spawn() {
        let mut sys = GpuParticleSystem::new(10);
        sys.reset_energy_stats();
        sys.spawn_burst(&default_params(), 5);
        let stats = sys.energy_stats();
        assert!(stats.spawn_energy_uj > 0.0);
        assert!((stats.spawn_energy_uj - 5.0 * 0.05).abs() < 1e-9);
        assert!((stats.total_energy_uj - stats.spawn_energy_uj).abs() < 1e-9);
    }

    #[test]
    fn test_energy_tracking_update() {
        let mut sys = GpuParticleSystem::new(10);
        sys.spawn_burst(&default_params(), 5);
        sys.reset_energy_stats();
        sys.update(0.01);
        let stats = sys.energy_stats();
        assert!(stats.update_energy_uj > 0.0);
    }

    #[test]
    fn test_energy_tracking_sort() {
        let mut sys = GpuParticleSystem::new(100);
        sys.set_sort_mode(SortMode::BackToFront);
        sys.spawn_burst(&default_params(), 50);
        sys.reset_energy_stats();
        sys.sort(Vec3::ZERO);
        let stats = sys.energy_stats();
        assert!(stats.sort_energy_uj > 0.0);
    }

    #[test]
    fn test_per_particle_energy_cost() {
        let mut sys = GpuParticleSystem::new(10);
        sys.set_gravity(Vec3::ZERO);
        let params = SpawnParams { lifetime: 10.0, ..Default::default() };
        let idx = sys.spawn_one(&params).unwrap();
        sys.update(0.1);
        let p = sys.get_particle(idx).unwrap();
        assert!(p.energy_cost_uj > 0.0);
    }

    #[test]
    fn test_normalized_age() {
        let p = Particle {
            lifetime: 4.0,
            age: 2.0,
            ..Particle::dead()
        };
        assert!((p.normalized_age() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_age_zero_lifetime() {
        let p = Particle { lifetime: 0.0, age: 1.0, ..Particle::dead() };
        assert!((p.normalized_age() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_alive_particles_iterator() {
        let mut sys = GpuParticleSystem::new(20);
        sys.spawn_burst(&default_params(), 7);
        let count = sys.alive_particles().count();
        assert_eq!(count, 7);
    }

    #[test]
    fn test_vec3_distance_sq() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 3.0);
        // (3^2 + 4^2 + 0) = 25
        assert!((a.distance_sq(b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_capacity() {
        let mut sys = GpuParticleSystem::new(10_000);
        let spawned = sys.spawn_burst(&default_params(), 10_000);
        assert_eq!(spawned, 10_000);
        assert_eq!(sys.alive_count(), 10_000);
        assert_eq!(sys.dead_count(), 0);
    }
}
