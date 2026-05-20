//! Weather simulation and rendering effects: weather states, smooth transitions,
//! rain/snow particles, fog, wind, puddle accumulation, wet surface darkening,
//! and thunder/lightning flashes.
//!
//! Pure Rust — all particle physics and state interpolation on CPU.

// ── Vec3 / Vec2 / Color ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn add(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn scale(&self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
    }
    pub fn lerp(&self, other: &Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
    pub fn scale(&self, s: f32) -> Color {
        Color::new(self.r * s, self.g * s, self.b * s, self.a)
    }
}

// ── Weather state ────────────────────────────────────────────────

/// Discrete weather states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherState {
    Clear,
    Cloudy,
    Overcast,
    Rain,
    Snow,
    Fog,
    Storm,
}

impl WeatherState {
    /// All weather states for iteration.
    pub fn all() -> &'static [WeatherState] {
        &[
            WeatherState::Clear,
            WeatherState::Cloudy,
            WeatherState::Overcast,
            WeatherState::Rain,
            WeatherState::Snow,
            WeatherState::Fog,
            WeatherState::Storm,
        ]
    }
}

/// Continuous weather parameters that can be interpolated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeatherParams {
    pub cloud_coverage: f32,
    pub precipitation_intensity: f32,
    pub fog_density: f32,
    pub fog_height: f32,
    pub wind_strength: f32,
    pub wind_direction: Vec2,
    pub wet_surface_amount: f32,
    pub snow_accumulation_rate: f32,
    pub lightning_probability: f32,
    pub ambient_brightness: f32,
}

impl WeatherParams {
    pub fn from_state(state: WeatherState) -> Self {
        match state {
            WeatherState::Clear => Self {
                cloud_coverage: 0.1,
                precipitation_intensity: 0.0,
                fog_density: 0.0,
                fog_height: 0.0,
                wind_strength: 0.1,
                wind_direction: Vec2::new(1.0, 0.0),
                wet_surface_amount: 0.0,
                snow_accumulation_rate: 0.0,
                lightning_probability: 0.0,
                ambient_brightness: 1.0,
            },
            WeatherState::Cloudy => Self {
                cloud_coverage: 0.5,
                precipitation_intensity: 0.0,
                fog_density: 0.01,
                fog_height: 50.0,
                wind_strength: 0.2,
                wind_direction: Vec2::new(1.0, 0.3),
                wet_surface_amount: 0.0,
                snow_accumulation_rate: 0.0,
                lightning_probability: 0.0,
                ambient_brightness: 0.85,
            },
            WeatherState::Overcast => Self {
                cloud_coverage: 0.9,
                precipitation_intensity: 0.0,
                fog_density: 0.03,
                fog_height: 80.0,
                wind_strength: 0.3,
                wind_direction: Vec2::new(0.8, 0.5),
                wet_surface_amount: 0.0,
                snow_accumulation_rate: 0.0,
                lightning_probability: 0.0,
                ambient_brightness: 0.65,
            },
            WeatherState::Rain => Self {
                cloud_coverage: 0.85,
                precipitation_intensity: 0.7,
                fog_density: 0.05,
                fog_height: 100.0,
                wind_strength: 0.5,
                wind_direction: Vec2::new(0.7, 0.7),
                wet_surface_amount: 0.8,
                snow_accumulation_rate: 0.0,
                lightning_probability: 0.0,
                ambient_brightness: 0.5,
            },
            WeatherState::Snow => Self {
                cloud_coverage: 0.8,
                precipitation_intensity: 0.5,
                fog_density: 0.04,
                fog_height: 120.0,
                wind_strength: 0.3,
                wind_direction: Vec2::new(0.5, 0.5),
                wet_surface_amount: 0.0,
                snow_accumulation_rate: 0.02,
                lightning_probability: 0.0,
                ambient_brightness: 0.7,
            },
            WeatherState::Fog => Self {
                cloud_coverage: 0.7,
                precipitation_intensity: 0.0,
                fog_density: 0.15,
                fog_height: 200.0,
                wind_strength: 0.05,
                wind_direction: Vec2::new(1.0, 0.0),
                wet_surface_amount: 0.3,
                snow_accumulation_rate: 0.0,
                lightning_probability: 0.0,
                ambient_brightness: 0.6,
            },
            WeatherState::Storm => Self {
                cloud_coverage: 1.0,
                precipitation_intensity: 1.0,
                fog_density: 0.08,
                fog_height: 150.0,
                wind_strength: 1.0,
                wind_direction: Vec2::new(0.6, 0.8),
                wet_surface_amount: 1.0,
                snow_accumulation_rate: 0.0,
                lightning_probability: 0.3,
                ambient_brightness: 0.35,
            },
        }
    }

    /// Interpolate between two parameter sets.
    pub fn lerp(&self, other: &WeatherParams, t: f32) -> WeatherParams {
        let t = t.clamp(0.0, 1.0);
        let lr = |a: f32, b: f32| a + (b - a) * t;
        WeatherParams {
            cloud_coverage: lr(self.cloud_coverage, other.cloud_coverage),
            precipitation_intensity: lr(self.precipitation_intensity, other.precipitation_intensity),
            fog_density: lr(self.fog_density, other.fog_density),
            fog_height: lr(self.fog_height, other.fog_height),
            wind_strength: lr(self.wind_strength, other.wind_strength),
            wind_direction: Vec2::new(
                lr(self.wind_direction.x, other.wind_direction.x),
                lr(self.wind_direction.y, other.wind_direction.y),
            ),
            wet_surface_amount: lr(self.wet_surface_amount, other.wet_surface_amount),
            snow_accumulation_rate: lr(self.snow_accumulation_rate, other.snow_accumulation_rate),
            lightning_probability: lr(self.lightning_probability, other.lightning_probability),
            ambient_brightness: lr(self.ambient_brightness, other.ambient_brightness),
        }
    }
}

// ── Particles ────────────────────────────────────────────────────

/// A single precipitation particle (rain drop or snowflake).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub size: f32,
}

impl Particle {
    pub fn new(position: Vec3, velocity: Vec3, lifetime: f32, size: f32) -> Self {
        Self {
            position,
            velocity,
            lifetime,
            max_lifetime: lifetime,
            size,
        }
    }

    pub fn alive(&self) -> bool {
        self.lifetime > 0.0
    }

    pub fn alpha(&self) -> f32 {
        if self.max_lifetime < 1e-6 {
            return 0.0;
        }
        (self.lifetime / self.max_lifetime).clamp(0.0, 1.0)
    }

    /// Update particle position by dt seconds.
    pub fn update(&mut self, dt: f32, wind: &Vec3, gravity: f32) {
        let accel = Vec3::new(wind.x, -gravity, wind.z);
        self.velocity = self.velocity.add(&accel.scale(dt));
        self.position = self.position.add(&self.velocity.scale(dt));
        self.lifetime -= dt;
    }
}

/// Particle emitter for precipitation.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    pub particles: Vec<Particle>,
    pub max_particles: usize,
    pub emit_rate: f32,
    pub emit_accumulator: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub ground_y: f32,
}

impl ParticleEmitter {
    pub fn new(max_particles: usize, bounds_min: Vec3, bounds_max: Vec3, ground_y: f32) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles),
            max_particles,
            emit_rate: 100.0,
            emit_accumulator: 0.0,
            bounds_min,
            bounds_max,
            ground_y,
        }
    }

    pub fn active_count(&self) -> usize {
        self.particles.iter().filter(|p| p.alive()).count()
    }

    /// Emit new particles and update existing ones.
    pub fn update(&mut self, dt: f32, wind: &Vec3, gravity: f32, velocity_base: Vec3, particle_size: f32, lifetime: f32) {
        // Update existing
        for p in &mut self.particles {
            if p.alive() {
                p.update(dt, wind, gravity);
                if p.position.y < self.ground_y {
                    p.lifetime = 0.0;
                }
            }
        }

        // Remove dead
        self.particles.retain(|p| p.alive());

        // Emit new
        self.emit_accumulator += self.emit_rate * dt;
        let to_emit = self.emit_accumulator.floor() as usize;
        self.emit_accumulator -= to_emit as f32;

        let range_x = self.bounds_max.x - self.bounds_min.x;
        let range_z = self.bounds_max.z - self.bounds_min.z;

        for i in 0..to_emit {
            if self.particles.len() >= self.max_particles {
                break;
            }
            let frac = if to_emit > 1 { i as f32 / (to_emit - 1) as f32 } else { 0.5 };
            let hash_val = ((frac * 12345.6789).sin() * 43758.5453).fract();
            let hash_val2 = ((frac * 98765.4321 + 0.5).sin() * 23421.6312).fract();
            let px = self.bounds_min.x + hash_val.abs() * range_x;
            let pz = self.bounds_min.z + hash_val2.abs() * range_z;
            let py = self.bounds_max.y;
            let vel_jitter = Vec3::new(
                (hash_val - 0.5) * 0.5,
                0.0,
                (hash_val2 - 0.5) * 0.5,
            );
            let vel = velocity_base.add(&vel_jitter);
            self.particles.push(Particle::new(
                Vec3::new(px, py, pz),
                vel,
                lifetime,
                particle_size,
            ));
        }
    }
}

// ── Splash effect ────────────────────────────────────────────────

/// Rain splash at impact point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Splash {
    pub position: Vec3,
    pub age: f32,
    pub max_age: f32,
    pub radius: f32,
}

impl Splash {
    pub fn new(position: Vec3, max_age: f32, radius: f32) -> Self {
        Self { position, age: 0.0, max_age, radius }
    }

    pub fn alive(&self) -> bool {
        self.age < self.max_age
    }

    pub fn alpha(&self) -> f32 {
        if self.max_age < 1e-6 {
            return 0.0;
        }
        (1.0 - self.age / self.max_age).clamp(0.0, 1.0)
    }

    pub fn current_radius(&self) -> f32 {
        if self.max_age < 1e-6 {
            return self.radius;
        }
        self.radius * (self.age / self.max_age).clamp(0.0, 1.0) * 2.0
    }

    pub fn update(&mut self, dt: f32) {
        self.age += dt;
    }
}

// ── Fog ──────────────────────────────────────────────────────────

/// Distance fog + height fog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogConfig {
    pub distance_density: f32,
    pub height_density: f32,
    pub height_falloff: f32,
    pub color: Color,
}

impl Default for FogConfig {
    fn default() -> Self {
        Self {
            distance_density: 0.02,
            height_density: 0.01,
            height_falloff: 0.05,
            color: Color::new(0.7, 0.75, 0.8, 1.0),
        }
    }
}

impl FogConfig {
    /// Compute fog factor at given distance and height. Returns [0, 1].
    pub fn fog_factor(&self, distance: f32, height: f32) -> f32 {
        let dist_fog = 1.0 - (-self.distance_density * distance).exp();
        let height_fog = self.height_density * (-self.height_falloff * height.max(0.0)).exp();
        (dist_fog + height_fog).clamp(0.0, 1.0)
    }

    /// Apply fog to an object color.
    pub fn apply(&self, object_color: &Color, distance: f32, height: f32) -> Color {
        let f = self.fog_factor(distance, height);
        object_color.lerp(&self.color, f)
    }
}

// ── Wet surface ──────────────────────────────────────────────────

/// Wet surface darkening effect.
pub fn wet_surface_darken(color: &Color, wetness: f32) -> Color {
    let darken = 1.0 - wetness.clamp(0.0, 1.0) * 0.4;
    color.scale(darken)
}

/// Puddle accumulation on flat surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PuddleState {
    pub depth: f32,
    pub max_depth: f32,
    pub reflectivity: f32,
}

impl PuddleState {
    pub fn new(max_depth: f32) -> Self {
        Self {
            depth: 0.0,
            max_depth,
            reflectivity: 0.0,
        }
    }

    pub fn accumulate(&mut self, rain_intensity: f32, dt: f32) {
        self.depth = (self.depth + rain_intensity * dt * 0.1).min(self.max_depth);
        self.reflectivity = (self.depth / self.max_depth.max(0.01)).clamp(0.0, 1.0);
    }

    pub fn evaporate(&mut self, rate: f32, dt: f32) {
        self.depth = (self.depth - rate * dt).max(0.0);
        self.reflectivity = (self.depth / self.max_depth.max(0.01)).clamp(0.0, 1.0);
    }

    pub fn fill_fraction(&self) -> f32 {
        (self.depth / self.max_depth.max(0.01)).clamp(0.0, 1.0)
    }
}

// ── Snow accumulation ────────────────────────────────────────────

/// Snow depth on a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnowAccumulation {
    pub depth: f32,
    pub max_depth: f32,
    pub melt_rate: f32,
}

impl SnowAccumulation {
    pub fn new(max_depth: f32) -> Self {
        Self {
            depth: 0.0,
            max_depth,
            melt_rate: 0.005,
        }
    }

    pub fn accumulate(&mut self, rate: f32, dt: f32) {
        self.depth = (self.depth + rate * dt).min(self.max_depth);
    }

    pub fn melt(&mut self, temperature: f32, dt: f32) {
        if temperature > 0.0 {
            let rate = self.melt_rate * temperature;
            self.depth = (self.depth - rate * dt).max(0.0);
        }
    }

    pub fn coverage(&self) -> f32 {
        (self.depth / self.max_depth.max(0.01)).clamp(0.0, 1.0)
    }

    pub fn apply_to_color(&self, surface_color: &Color) -> Color {
        let snow = Color::new(0.95, 0.97, 1.0, 1.0);
        surface_color.lerp(&snow, self.coverage())
    }
}

// ── Lightning ────────────────────────────────────────────────────

/// Lightning flash state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightningFlash {
    pub active: bool,
    pub intensity: f32,
    pub duration: f32,
    pub elapsed: f32,
}

impl LightningFlash {
    pub fn new() -> Self {
        Self {
            active: false,
            intensity: 0.0,
            duration: 0.2,
            elapsed: 0.0,
        }
    }

    pub fn trigger(&mut self, intensity: f32) {
        self.active = true;
        self.intensity = intensity;
        self.elapsed = 0.0;
    }

    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.active = false;
            self.intensity = 0.0;
        }
    }

    /// Current flash brightness (flickers).
    pub fn current_brightness(&self) -> f32 {
        if !self.active {
            return 0.0;
        }
        let t = self.elapsed / self.duration.max(1e-6);
        // Sharp flicker pattern
        let flicker = if t < 0.1 {
            1.0
        } else if t < 0.2 {
            0.3
        } else if t < 0.35 {
            0.9
        } else {
            (1.0 - t).max(0.0)
        };
        flicker * self.intensity
    }

    /// Apply flash to ambient brightness.
    pub fn apply_to_scene(&self, base_brightness: f32) -> f32 {
        base_brightness + self.current_brightness()
    }
}

impl Default for LightningFlash {
    fn default() -> Self {
        Self::new()
    }
}

// ── Weather system ───────────────────────────────────────────────

/// Main weather simulation controller.
#[derive(Debug, Clone)]
pub struct WeatherSystem {
    pub current_state: WeatherState,
    pub target_state: WeatherState,
    pub current_params: WeatherParams,
    pub transition_time: f32,
    pub transition_duration: f32,
    pub rain_emitter: Option<ParticleEmitter>,
    pub snow_emitter: Option<ParticleEmitter>,
    pub splashes: Vec<Splash>,
    pub fog: FogConfig,
    pub puddle: PuddleState,
    pub snow: SnowAccumulation,
    pub lightning: LightningFlash,
    pub time_since_last_lightning: f32,
}

impl WeatherSystem {
    pub fn new(initial_state: WeatherState) -> Self {
        let params = WeatherParams::from_state(initial_state);
        Self {
            current_state: initial_state,
            target_state: initial_state,
            current_params: params,
            transition_time: 0.0,
            transition_duration: 5.0,
            rain_emitter: None,
            snow_emitter: None,
            splashes: Vec::new(),
            fog: FogConfig::default(),
            puddle: PuddleState::new(0.05),
            snow: SnowAccumulation::new(0.5),
            lightning: LightningFlash::new(),
            time_since_last_lightning: 0.0,
        }
    }

    /// Start transitioning to a new weather state.
    pub fn transition_to(&mut self, state: WeatherState, duration: f32) {
        self.target_state = state;
        self.transition_time = 0.0;
        self.transition_duration = duration.max(0.1);
    }

    /// Whether a transition is currently in progress.
    pub fn is_transitioning(&self) -> bool {
        self.current_state != self.target_state && self.transition_time < self.transition_duration
    }

    /// Set up emitter bounds for precipitation.
    pub fn setup_emitters(&mut self, bounds_min: Vec3, bounds_max: Vec3, ground_y: f32) {
        self.rain_emitter = Some(ParticleEmitter::new(5000, bounds_min, bounds_max, ground_y));
        self.snow_emitter = Some(ParticleEmitter::new(3000, bounds_min, bounds_max, ground_y));
    }

    /// Advance the weather simulation by dt seconds.
    pub fn update(&mut self, dt: f32) {
        // Transition interpolation
        if self.is_transitioning() {
            self.transition_time += dt;
            let t = (self.transition_time / self.transition_duration).clamp(0.0, 1.0);
            let from = WeatherParams::from_state(self.current_state);
            let to = WeatherParams::from_state(self.target_state);
            self.current_params = from.lerp(&to, t);
            if t >= 1.0 {
                self.current_state = self.target_state;
            }
        }

        let params = self.current_params;
        let wind = Vec3::new(
            params.wind_direction.x * params.wind_strength,
            0.0,
            params.wind_direction.y * params.wind_strength,
        );

        // Update fog
        self.fog.distance_density = params.fog_density;
        self.fog.height_density = params.fog_density * 0.5;

        // Rain particles
        if let Some(emitter) = &mut self.rain_emitter {
            emitter.emit_rate = params.precipitation_intensity * 500.0;
            if params.precipitation_intensity > 0.01 && params.snow_accumulation_rate < 0.01 {
                let rain_vel = Vec3::new(wind.x * 0.5, -8.0, wind.z * 0.5);
                emitter.update(dt, &wind, 9.8, rain_vel, 0.01, 2.0);
            } else {
                emitter.particles.clear();
            }
        }

        // Snow particles
        if let Some(emitter) = &mut self.snow_emitter {
            emitter.emit_rate = params.precipitation_intensity * 200.0;
            if params.snow_accumulation_rate > 0.001 {
                let snow_vel = Vec3::new(wind.x * 0.3, -1.5, wind.z * 0.3);
                emitter.update(dt, &wind, 0.5, snow_vel, 0.03, 5.0);
                self.snow.accumulate(params.snow_accumulation_rate, dt);
            } else {
                emitter.particles.clear();
                self.snow.melt(5.0, dt);
            }
        }

        // Puddle accumulation
        if params.wet_surface_amount > 0.01 {
            self.puddle.accumulate(params.precipitation_intensity, dt);
        } else {
            self.puddle.evaporate(0.01, dt);
        }

        // Splashes
        for s in &mut self.splashes {
            s.update(dt);
        }
        self.splashes.retain(|s| s.alive());

        // Lightning
        self.lightning.update(dt);
        if params.lightning_probability > 0.0 {
            self.time_since_last_lightning += dt;
            let interval = 3.0 / params.lightning_probability.max(0.01);
            if self.time_since_last_lightning > interval {
                self.lightning.trigger(1.5);
                self.time_since_last_lightning = 0.0;
            }
        }
    }

    /// Current precipitation particle count.
    pub fn rain_particle_count(&self) -> usize {
        self.rain_emitter.as_ref().map_or(0, |e| e.active_count())
    }

    pub fn snow_particle_count(&self) -> usize {
        self.snow_emitter.as_ref().map_or(0, |e| e.active_count())
    }

    /// Apply all weather effects to a surface color at given distance/height.
    pub fn apply_effects(&self, surface_color: &Color, distance: f32, height: f32) -> Color {
        let params = &self.current_params;
        // Wet darkening
        let wet = wet_surface_darken(surface_color, params.wet_surface_amount);
        // Snow coverage
        let snowed = self.snow.apply_to_color(&wet);
        // Fog
        let fogged = self.fog.apply(&snowed, distance, height);
        // Lightning flash
        let brightness = self.lightning.apply_to_scene(params.ambient_brightness);
        fogged.scale(brightness.clamp(0.0, 2.0))
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn weather_state_all() {
        assert_eq!(WeatherState::all().len(), 7);
    }

    #[test]
    fn weather_params_from_clear() {
        let p = WeatherParams::from_state(WeatherState::Clear);
        assert!(approx(p.precipitation_intensity, 0.0, 1e-6));
        assert!(approx(p.ambient_brightness, 1.0, 1e-6));
    }

    #[test]
    fn weather_params_from_storm() {
        let p = WeatherParams::from_state(WeatherState::Storm);
        assert!(approx(p.precipitation_intensity, 1.0, 1e-6));
        assert!(p.lightning_probability > 0.0);
    }

    #[test]
    fn weather_params_lerp() {
        let a = WeatherParams::from_state(WeatherState::Clear);
        let b = WeatherParams::from_state(WeatherState::Rain);
        let mid = a.lerp(&b, 0.5);
        assert!(mid.precipitation_intensity > 0.0 && mid.precipitation_intensity < 1.0);
        assert!(mid.cloud_coverage > a.cloud_coverage && mid.cloud_coverage < b.cloud_coverage);
    }

    #[test]
    fn particle_creation_and_update() {
        let mut p = Particle::new(Vec3::new(0.0, 100.0, 0.0), Vec3::new(0.0, -5.0, 0.0), 2.0, 0.01);
        assert!(p.alive());
        p.update(0.5, &Vec3::new(0.0, 0.0, 0.0), 9.8);
        assert!(p.position.y < 100.0);
        assert!(p.alive());
    }

    #[test]
    fn particle_lifetime_expires() {
        let mut p = Particle::new(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 1.0, 0.01);
        p.update(1.5, &Vec3::new(0.0, 0.0, 0.0), 0.0);
        assert!(!p.alive());
    }

    #[test]
    fn particle_alpha() {
        let p = Particle::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 2.0, 0.01);
        assert!(approx(p.alpha(), 1.0, 1e-6));
    }

    #[test]
    fn emitter_creates_particles() {
        let min = Vec3::new(-10.0, 0.0, -10.0);
        let max = Vec3::new(10.0, 50.0, 10.0);
        let mut emitter = ParticleEmitter::new(1000, min, max, 0.0);
        emitter.emit_rate = 200.0;
        let wind = Vec3::new(0.0, 0.0, 0.0);
        let vel = Vec3::new(0.0, -5.0, 0.0);
        emitter.update(0.1, &wind, 9.8, vel, 0.01, 2.0);
        assert!(emitter.active_count() > 0);
    }

    #[test]
    fn emitter_respects_max() {
        let min = Vec3::new(-1.0, 0.0, -1.0);
        let max = Vec3::new(1.0, 10.0, 1.0);
        let mut emitter = ParticleEmitter::new(10, min, max, 0.0);
        emitter.emit_rate = 10000.0;
        let wind = Vec3::new(0.0, 0.0, 0.0);
        let vel = Vec3::new(0.0, -1.0, 0.0);
        emitter.update(1.0, &wind, 0.0, vel, 0.01, 100.0);
        assert!(emitter.particles.len() <= 10);
    }

    #[test]
    fn splash_lifecycle() {
        let mut s = Splash::new(Vec3::new(0.0, 0.0, 0.0), 0.5, 0.1);
        assert!(s.alive());
        assert!(approx(s.alpha(), 1.0, 1e-6));
        s.update(0.25);
        assert!(s.alive());
        assert!(s.alpha() < 1.0 && s.alpha() > 0.0);
        s.update(0.3);
        assert!(!s.alive());
    }

    #[test]
    fn splash_radius_grows() {
        let mut s = Splash::new(Vec3::new(0.0, 0.0, 0.0), 1.0, 0.2);
        let r0 = s.current_radius();
        s.update(0.5);
        let r1 = s.current_radius();
        assert!(r1 > r0);
    }

    #[test]
    fn fog_no_distance() {
        let fog = FogConfig::default();
        let f = fog.fog_factor(0.0, 100.0);
        assert!(f < 0.5, "no fog at zero distance");
    }

    #[test]
    fn fog_large_distance() {
        let fog = FogConfig::default();
        let f = fog.fog_factor(100.0, 0.0);
        assert!(f > 0.5, "heavy fog at large distance");
    }

    #[test]
    fn fog_apply() {
        let fog = FogConfig::default();
        let obj = Color::new(1.0, 0.0, 0.0, 1.0);
        let fogged = fog.apply(&obj, 50.0, 0.0);
        assert!(fogged.r < 1.0, "fogged object should be desaturated");
    }

    #[test]
    fn wet_surface_darker() {
        let c = Color::new(0.5, 0.5, 0.5, 1.0);
        let wet = wet_surface_darken(&c, 1.0);
        assert!(wet.r < c.r);
    }

    #[test]
    fn puddle_accumulation() {
        let mut p = PuddleState::new(0.1);
        p.accumulate(1.0, 1.0);
        assert!(p.depth > 0.0);
        assert!(p.reflectivity > 0.0);
    }

    #[test]
    fn puddle_evaporation() {
        let mut p = PuddleState::new(0.1);
        p.accumulate(1.0, 1.0);
        let before = p.depth;
        p.evaporate(0.1, 0.5);
        assert!(p.depth < before);
    }

    #[test]
    fn snow_accumulation_and_melt() {
        let mut s = SnowAccumulation::new(0.5);
        s.accumulate(0.1, 1.0);
        assert!(s.depth > 0.0);
        s.melt(10.0, 1.0);
        assert!(s.depth < 0.1);
    }

    #[test]
    fn snow_coverage_color() {
        let mut s = SnowAccumulation::new(0.5);
        s.depth = 0.5;
        let green = Color::new(0.2, 0.8, 0.1, 1.0);
        let snowy = s.apply_to_color(&green);
        assert!(snowy.r > green.r, "snow should whiten the surface");
    }

    #[test]
    fn lightning_flash_cycle() {
        let mut l = LightningFlash::new();
        assert!(!l.active);
        l.trigger(2.0);
        assert!(l.active);
        assert!(l.current_brightness() > 0.0);
        l.update(0.3);
        assert!(!l.active);
    }

    #[test]
    fn lightning_default() {
        let l = LightningFlash::default();
        assert!(!l.active);
    }

    #[test]
    fn weather_system_creation() {
        let ws = WeatherSystem::new(WeatherState::Clear);
        assert_eq!(ws.current_state, WeatherState::Clear);
        assert!(!ws.is_transitioning());
    }

    #[test]
    fn weather_system_transition() {
        let mut ws = WeatherSystem::new(WeatherState::Clear);
        ws.transition_to(WeatherState::Rain, 2.0);
        assert!(ws.is_transitioning());
        ws.update(1.0);
        assert!(ws.is_transitioning());
        assert!(ws.current_params.precipitation_intensity > 0.0);
        ws.update(1.5);
        assert!(!ws.is_transitioning());
        assert_eq!(ws.current_state, WeatherState::Rain);
    }

    #[test]
    fn weather_system_apply_effects() {
        let ws = WeatherSystem::new(WeatherState::Clear);
        let surface = Color::new(0.5, 0.8, 0.3, 1.0);
        let result = ws.apply_effects(&surface, 10.0, 5.0);
        assert!(result.r >= 0.0 && result.r <= 2.0);
        assert!(result.g >= 0.0);
    }

    #[test]
    fn weather_system_with_emitters() {
        let mut ws = WeatherSystem::new(WeatherState::Rain);
        ws.setup_emitters(
            Vec3::new(-50.0, 0.0, -50.0),
            Vec3::new(50.0, 100.0, 50.0),
            0.0,
        );
        ws.update(0.1);
        assert!(ws.rain_particle_count() > 0);
    }

    #[test]
    fn weather_system_snow_emitters() {
        let mut ws = WeatherSystem::new(WeatherState::Snow);
        ws.setup_emitters(
            Vec3::new(-50.0, 0.0, -50.0),
            Vec3::new(50.0, 100.0, 50.0),
            0.0,
        );
        ws.update(0.1);
        assert!(ws.snow_particle_count() > 0);
    }
}
