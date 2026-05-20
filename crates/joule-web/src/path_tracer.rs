// Unidirectional path tracer with material interactions and progressive rendering.
// Supports diffuse, specular, glossy (GGX), and refractive surfaces.

use std::fmt;

const PI: f64 = std::f64::consts::PI;
const INV_PI: f64 = 1.0 / PI;
const EPSILON: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn splat(v: f64) -> Self { Self { x: v, y: v, z: v } }

    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-15 { Self::zero() } else { self * (1.0 / l) }
    }
    pub fn reflect(self, n: Self) -> Self {
        self - n * (2.0 * self.dot(n))
    }
    pub fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// RGB color (linear space, can exceed 1.0 for HDR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn black() -> Self { Self { r: 0.0, g: 0.0, b: 0.0 } }
    pub fn white() -> Self { Self { r: 1.0, g: 1.0, b: 1.0 } }
    pub fn luminance(self) -> f64 { 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b }
    pub fn max_channel(self) -> f64 { self.r.max(self.g).max(self.b) }
    pub fn clamp(self) -> Self {
        Self {
            r: self.r.max(0.0).min(1.0),
            g: self.g.max(0.0).min(1.0),
            b: self.b.max(0.0).min(1.0),
        }
    }
    pub fn is_black(self) -> bool {
        self.r <= 0.0 && self.g <= 0.0 && self.b <= 0.0
    }
}

impl std::ops::Add for Color {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { r: self.r + r.r, g: self.g + r.g, b: self.b + r.b } }
}
impl std::ops::Mul<f64> for Color {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}
impl std::ops::Mul for Color {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Self { r: self.r * rhs.r, g: self.g * rhs.g, b: self.b * rhs.b } }
}

/// Simple LCG random number generator (deterministic for reproducibility).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) as f64;
        bits / (1u64 << 53) as f64
    }
}

/// A ray in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalized() }
    }
    pub fn at(self, t: f64) -> Vec3 { self.origin + self.direction * t }
}

/// Hit information from scene intersection.
#[derive(Debug, Clone, Copy)]
pub struct HitInfo {
    pub t: f64,
    pub point: Vec3,
    pub normal: Vec3,
    pub material_id: usize,
}

/// Material types for path tracing.
#[derive(Debug, Clone)]
pub enum Material {
    Diffuse { albedo: Color },
    Specular { albedo: Color },
    Glossy { albedo: Color, roughness: f64 },
    Refractive { albedo: Color, ior: f64 },
    Emissive { emission: Color },
}

/// Camera for generating primary rays.
#[derive(Debug, Clone)]
pub struct Camera {
    pub origin: Vec3,
    pub lower_left: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
}

impl Camera {
    pub fn new(look_from: Vec3, look_at: Vec3, up: Vec3, fov_deg: f64, aspect: f64) -> Self {
        let theta = fov_deg * PI / 180.0;
        let h = (theta * 0.5).tan();
        let viewport_h = 2.0 * h;
        let viewport_w = aspect * viewport_h;

        let w = (look_from - look_at).normalized();
        let u = up.cross(w).normalized();
        let v = w.cross(u);

        let horizontal = u * viewport_w;
        let vertical = v * viewport_h;
        let lower_left = look_from - horizontal * 0.5 - vertical * 0.5 - w;

        Self { origin: look_from, lower_left, horizontal, vertical }
    }

    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        let dir = self.lower_left + self.horizontal * s + self.vertical * t - self.origin;
        Ray::new(self.origin, dir)
    }
}

/// Scene containing spheres for testing.
#[derive(Debug, Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub material_id: usize,
}

/// Simple scene with spheres and materials.
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
}

impl Scene {
    pub fn new() -> Self { Self { spheres: Vec::new(), materials: Vec::new() } }

    pub fn add_material(&mut self, mat: Material) -> usize {
        let id = self.materials.len();
        self.materials.push(mat);
        id
    }

    pub fn add_sphere(&mut self, center: Vec3, radius: f64, material_id: usize) {
        self.spheres.push(Sphere { center, radius, material_id });
    }

    pub fn intersect(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitInfo> {
        let mut closest = t_max;
        let mut hit_info: Option<HitInfo> = None;

        for sphere in &self.spheres {
            let oc = ray.origin - sphere.center;
            let a = ray.direction.dot(ray.direction);
            let half_b = oc.dot(ray.direction);
            let c = oc.dot(oc) - sphere.radius * sphere.radius;
            let disc = half_b * half_b - a * c;
            if disc < 0.0 { continue; }
            let sqrt_d = disc.sqrt();
            let mut t = (-half_b - sqrt_d) / a;
            if t < t_min || t > closest {
                t = (-half_b + sqrt_d) / a;
                if t < t_min || t > closest { continue; }
            }
            closest = t;
            let p = ray.at(t);
            let n = (p - sphere.center) * (1.0 / sphere.radius);
            hit_info = Some(HitInfo { t, point: p, normal: n.normalized(), material_id: sphere.material_id });
        }
        hit_info
    }
}

/// Build an orthonormal basis from a normal vector.
fn onb_from_normal(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let w = n.normalized();
    let a = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let v = w.cross(a).normalized();
    let u = w.cross(v);
    (u, v, w)
}

/// Sample cosine-weighted hemisphere direction.
fn sample_cosine_hemisphere(rng: &mut Rng, normal: Vec3) -> Vec3 {
    let (u, v, w) = onb_from_normal(normal);
    let r1 = rng.next_f64();
    let r2 = rng.next_f64();
    let phi = 2.0 * PI * r1;
    let cos_theta = r2.sqrt();
    let sin_theta = (1.0 - r2).sqrt();
    let dir = u * (phi.cos() * sin_theta) + v * (phi.sin() * sin_theta) + w * cos_theta;
    dir.normalized()
}

/// Schlick Fresnel approximation.
fn fresnel_schlick(cos_theta: f64, f0: f64) -> f64 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
}

/// Refract a vector through a surface. Returns None for total internal reflection.
fn refract(incident: Vec3, normal: Vec3, eta: f64) -> Option<Vec3> {
    let cos_i = (-incident).dot(normal).min(1.0);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t > 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    Some(incident * eta + normal * (eta * cos_i - cos_t))
}

/// GGX microfacet normal sampling.
fn sample_ggx(rng: &mut Rng, roughness: f64, normal: Vec3) -> Vec3 {
    let (u, v, w) = onb_from_normal(normal);
    let a = roughness * roughness;
    let r1 = rng.next_f64();
    let r2 = rng.next_f64();
    let theta = ((a * a * r1) / (1.0 - r1)).sqrt().atan();
    let phi = 2.0 * PI * r2;
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let h = u * (phi.cos() * sin_t) + v * (phi.sin() * sin_t) + w * cos_t;
    h.normalized()
}

/// Path tracing configuration.
#[derive(Debug, Clone)]
pub struct PathTracerConfig {
    pub max_bounces: usize,
    pub samples_per_pixel: usize,
    pub russian_roulette_start: usize,
    pub background: Color,
}

impl PathTracerConfig {
    pub fn default_config() -> Self {
        Self {
            max_bounces: 8,
            samples_per_pixel: 16,
            russian_roulette_start: 3,
            background: Color::new(0.2, 0.3, 0.5),
        }
    }
}

/// Trace a single path and return accumulated radiance.
pub fn trace_path(
    scene: &Scene,
    ray: &Ray,
    config: &PathTracerConfig,
    rng: &mut Rng,
) -> Color {
    let mut throughput = Color::white();
    let mut radiance = Color::black();
    let mut current_ray = *ray;

    for bounce in 0..config.max_bounces {
        let hit = match scene.intersect(&current_ray, EPSILON, f64::MAX) {
            Some(h) => h,
            None => {
                radiance = radiance + throughput * config.background;
                break;
            }
        };

        let material = &scene.materials[hit.material_id];
        let normal = if hit.normal.dot(current_ray.direction) < 0.0 {
            hit.normal
        } else {
            -hit.normal
        };

        match material {
            Material::Emissive { emission } => {
                radiance = radiance + throughput * *emission;
                break;
            }
            Material::Diffuse { albedo } => {
                let wi = sample_cosine_hemisphere(rng, normal);
                throughput = throughput * *albedo;
                current_ray = Ray::new(hit.point + normal * EPSILON, wi);
            }
            Material::Specular { albedo } => {
                let reflected = current_ray.direction.reflect(normal);
                throughput = throughput * *albedo;
                current_ray = Ray::new(hit.point + normal * EPSILON, reflected);
            }
            Material::Glossy { albedo, roughness } => {
                let h = sample_ggx(rng, *roughness, normal);
                let reflected = current_ray.direction.reflect(h);
                if reflected.dot(normal) <= 0.0 {
                    break;
                }
                throughput = throughput * *albedo;
                current_ray = Ray::new(hit.point + normal * EPSILON, reflected);
            }
            Material::Refractive { albedo, ior } => {
                let (out_normal, eta, cos_i) = if current_ray.direction.dot(hit.normal) < 0.0 {
                    (hit.normal, 1.0 / ior, (-current_ray.direction).dot(hit.normal))
                } else {
                    (-hit.normal, *ior, current_ray.direction.dot(hit.normal).abs())
                };

                let f = fresnel_schlick(cos_i, ((1.0 - ior) / (1.0 + ior)).powi(2));
                throughput = throughput * *albedo;

                if rng.next_f64() < f {
                    let reflected = current_ray.direction.reflect(out_normal);
                    current_ray = Ray::new(hit.point + out_normal * EPSILON, reflected);
                } else if let Some(refracted) = refract(current_ray.direction, out_normal, eta) {
                    current_ray = Ray::new(hit.point - out_normal * EPSILON, refracted);
                } else {
                    let reflected = current_ray.direction.reflect(out_normal);
                    current_ray = Ray::new(hit.point + out_normal * EPSILON, reflected);
                }
            }
        }

        // Russian roulette
        if bounce >= config.russian_roulette_start {
            let p = throughput.max_channel().min(0.95);
            if p <= 0.0 || rng.next_f64() > p {
                break;
            }
            throughput = throughput * (1.0 / p);
        }
    }
    radiance
}

/// Render a single pixel with multiple samples.
pub fn render_pixel(
    scene: &Scene,
    camera: &Camera,
    config: &PathTracerConfig,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    rng: &mut Rng,
) -> Color {
    let mut accum = Color::black();
    for _ in 0..config.samples_per_pixel {
        let u = (px as f64 + rng.next_f64()) / width as f64;
        let v = (py as f64 + rng.next_f64()) / height as f64;
        let ray = camera.get_ray(u, v);
        let sample = trace_path(scene, &ray, config, rng);
        accum = accum + sample;
    }
    accum * (1.0 / config.samples_per_pixel as f64)
}

/// Progressive frame buffer for accumulating samples over multiple frames.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    pub accum: Vec<Color>,
    pub frame_count: usize,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            accum: vec![Color::black(); width * height],
            frame_count: 0,
        }
    }

    pub fn add_sample(&mut self, x: usize, y: usize, color: Color) {
        let idx = y * self.width + x;
        if idx < self.accum.len() {
            self.accum[idx] = self.accum[idx] + color;
        }
    }

    pub fn increment_frame(&mut self) {
        self.frame_count += 1;
    }

    pub fn get_averaged(&self, x: usize, y: usize) -> Color {
        if self.frame_count == 0 {
            return Color::black();
        }
        let idx = y * self.width + x;
        if idx >= self.accum.len() {
            return Color::black();
        }
        self.accum[idx] * (1.0 / self.frame_count as f64)
    }

    pub fn reset(&mut self) {
        for c in &mut self.accum {
            *c = Color::black();
        }
        self.frame_count = 0;
    }
}

/// Render one progressive frame into the frame buffer.
pub fn render_progressive_frame(
    scene: &Scene,
    camera: &Camera,
    config: &PathTracerConfig,
    fb: &mut FrameBuffer,
    rng: &mut Rng,
) {
    for y in 0..fb.height {
        for x in 0..fb.width {
            let u = (x as f64 + rng.next_f64()) / fb.width as f64;
            let v = (y as f64 + rng.next_f64()) / fb.height as f64;
            let ray = camera.get_ray(u, v);
            let color = trace_path(scene, &ray, config, rng);
            fb.add_sample(x, y, color);
        }
    }
    fb.increment_frame();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    fn color_approx(a: Color, b: Color, eps: f64) -> bool {
        approx_eq(a.r, b.r, eps) && approx_eq(a.g, b.g, eps) && approx_eq(a.b, b.b, eps)
    }

    #[test]
    fn test_vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalized();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(n);
        assert!(approx_eq(r.x, v.x, 1e-9));
        assert!(approx_eq(r.y, -v.y, 1e-9));
    }

    #[test]
    fn test_color_ops() {
        let a = Color::new(0.5, 0.3, 0.1);
        let b = Color::new(0.1, 0.2, 0.3);
        let sum = a + b;
        assert!(approx_eq(sum.r, 0.6, 1e-9));
        let prod = a * b;
        assert!(approx_eq(prod.r, 0.05, 1e-9));
    }

    #[test]
    fn test_color_luminance() {
        let white = Color::white();
        let lum = white.luminance();
        assert!(approx_eq(lum, 1.0, 1e-6));
    }

    #[test]
    fn test_color_clamp() {
        let c = Color::new(1.5, -0.3, 0.5);
        let cl = c.clamp();
        assert!(approx_eq(cl.r, 1.0, 1e-9));
        assert!(approx_eq(cl.g, 0.0, 1e-9));
        assert!(approx_eq(cl.b, 0.5, 1e-9));
    }

    #[test]
    fn test_camera_center_ray() {
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            90.0,
            1.0,
        );
        let ray = cam.get_ray(0.5, 0.5);
        assert!(approx_eq(ray.direction.z, -1.0, 0.1));
    }

    #[test]
    fn test_scene_sphere_intersection() {
        let mut scene = Scene::new();
        let mat = scene.add_material(Material::Diffuse { albedo: Color::new(0.5, 0.5, 0.5) });
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, mat);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let hit = scene.intersect(&ray, EPSILON, f64::MAX).unwrap();
        assert!(approx_eq(hit.t, 2.0, 1e-6));
    }

    #[test]
    fn test_scene_miss() {
        let mut scene = Scene::new();
        let mat = scene.add_material(Material::Diffuse { albedo: Color::white() });
        scene.add_sphere(Vec3::new(10.0, 10.0, 10.0), 1.0, mat);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        assert!(scene.intersect(&ray, EPSILON, f64::MAX).is_none());
    }

    #[test]
    fn test_trace_emissive() {
        let mut scene = Scene::new();
        let emit = scene.add_material(Material::Emissive { emission: Color::new(1.0, 0.0, 0.0) });
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, emit);
        let config = PathTracerConfig::default_config();
        let mut rng = Rng::new(42);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let c = trace_path(&scene, &ray, &config, &mut rng);
        assert!(approx_eq(c.r, 1.0, 1e-6));
        assert!(approx_eq(c.g, 0.0, 1e-6));
    }

    #[test]
    fn test_trace_background() {
        let scene = Scene::new();
        let config = PathTracerConfig::default_config();
        let mut rng = Rng::new(42);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let c = trace_path(&scene, &ray, &config, &mut rng);
        assert!(color_approx(c, config.background, 1e-6));
    }

    #[test]
    fn test_trace_diffuse_bounded() {
        let mut scene = Scene::new();
        let diff = scene.add_material(Material::Diffuse { albedo: Color::new(0.8, 0.8, 0.8) });
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, diff);
        let config = PathTracerConfig { max_bounces: 4, samples_per_pixel: 1, russian_roulette_start: 10, background: Color::new(1.0, 1.0, 1.0) };
        let mut rng = Rng::new(123);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let c = trace_path(&scene, &ray, &config, &mut rng);
        // Result should be finite and non-negative
        assert!(c.r >= 0.0 && c.r.is_finite());
        assert!(c.g >= 0.0 && c.g.is_finite());
    }

    #[test]
    fn test_trace_specular() {
        let mut scene = Scene::new();
        let spec = scene.add_material(Material::Specular { albedo: Color::white() });
        let emit = scene.add_material(Material::Emissive { emission: Color::new(0.0, 1.0, 0.0) });
        // Mirror sphere reflects to emissive sphere behind origin
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, spec);
        scene.add_sphere(Vec3::new(0.0, 0.0, 3.0), 1.0, emit);
        let config = PathTracerConfig { max_bounces: 4, samples_per_pixel: 1, russian_roulette_start: 10, background: Color::black() };
        let mut rng = Rng::new(42);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let c = trace_path(&scene, &ray, &config, &mut rng);
        // Should pick up green from reflected ray hitting the emissive sphere
        assert!(c.g > 0.0);
    }

    #[test]
    fn test_fresnel_schlick_normal() {
        // At normal incidence with f0=0.04
        let f = fresnel_schlick(1.0, 0.04);
        assert!(approx_eq(f, 0.04, 1e-6));
    }

    #[test]
    fn test_fresnel_schlick_grazing() {
        let f = fresnel_schlick(0.0, 0.04);
        assert!(approx_eq(f, 1.0, 1e-6));
    }

    #[test]
    fn test_refract_normal_incidence() {
        let i = Vec3::new(0.0, 0.0, -1.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let r = refract(i, n, 1.0).unwrap();
        assert!(approx_eq(r.z, -1.0, 1e-6));
    }

    #[test]
    fn test_refract_tir() {
        // Total internal reflection from glass to air at steep angle
        let i = Vec3::new(0.9, 0.0, -0.1).normalized();
        let n = Vec3::new(0.0, 0.0, 1.0);
        // Going from glass (1.5) to air (1.0) — eta = 1.5
        let result = refract(i, n, 1.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_render_pixel_deterministic() {
        let mut scene = Scene::new();
        let emit = scene.add_material(Material::Emissive { emission: Color::new(0.5, 0.5, 0.5) });
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 100.0, emit);
        let cam = Camera::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 1.0);
        let config = PathTracerConfig { max_bounces: 1, samples_per_pixel: 4, russian_roulette_start: 10, background: Color::black() };
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let c1 = render_pixel(&scene, &cam, &config, 5, 5, 10, 10, &mut rng1);
        let c2 = render_pixel(&scene, &cam, &config, 5, 5, 10, 10, &mut rng2);
        assert!(color_approx(c1, c2, 1e-9));
    }

    #[test]
    fn test_frame_buffer_progressive() {
        let mut fb = FrameBuffer::new(2, 2);
        fb.add_sample(0, 0, Color::new(1.0, 0.0, 0.0));
        fb.increment_frame();
        fb.add_sample(0, 0, Color::new(0.0, 1.0, 0.0));
        fb.increment_frame();
        let avg = fb.get_averaged(0, 0);
        assert!(approx_eq(avg.r, 0.5, 1e-9));
        assert!(approx_eq(avg.g, 0.5, 1e-9));
    }

    #[test]
    fn test_frame_buffer_reset() {
        let mut fb = FrameBuffer::new(4, 4);
        fb.add_sample(1, 1, Color::white());
        fb.increment_frame();
        fb.reset();
        assert_eq!(fb.frame_count, 0);
        let c = fb.get_averaged(1, 1);
        assert!(color_approx(c, Color::black(), 1e-9));
    }

    #[test]
    fn test_rng_range() {
        let mut rng = Rng::new(0);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_cosine_hemisphere_on_hemisphere() {
        let mut rng = Rng::new(99);
        let n = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..100 {
            let d = sample_cosine_hemisphere(&mut rng, n);
            assert!(d.dot(n) >= -1e-6, "sample was below hemisphere");
            assert!(approx_eq(d.length(), 1.0, 1e-6));
        }
    }

    #[test]
    fn test_progressive_render() {
        let mut scene = Scene::new();
        let emit = scene.add_material(Material::Emissive { emission: Color::new(0.3, 0.6, 0.9) });
        scene.add_sphere(Vec3::new(0.0, 0.0, -2.0), 100.0, emit);
        let cam = Camera::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 1.0, 0.0), 90.0, 1.0);
        let config = PathTracerConfig { max_bounces: 1, samples_per_pixel: 1, russian_roulette_start: 10, background: Color::black() };
        let mut fb = FrameBuffer::new(4, 4);
        let mut rng = Rng::new(77);
        for _ in 0..10 {
            render_progressive_frame(&scene, &cam, &config, &mut fb, &mut rng);
        }
        assert_eq!(fb.frame_count, 10);
        let c = fb.get_averaged(2, 2);
        // Should be close to the emission color
        assert!(c.r > 0.0 && c.r.is_finite());
    }

    #[test]
    fn test_glossy_material() {
        let mut scene = Scene::new();
        let glossy = scene.add_material(Material::Glossy { albedo: Color::white(), roughness: 0.3 });
        let emit = scene.add_material(Material::Emissive { emission: Color::white() });
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, glossy);
        scene.add_sphere(Vec3::new(0.0, 0.0, -7.0), 1.0, emit);
        let config = PathTracerConfig { max_bounces: 4, samples_per_pixel: 8, russian_roulette_start: 10, background: Color::black() };
        let mut rng = Rng::new(42);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        let c = trace_path(&scene, &ray, &config, &mut rng);
        assert!(c.r.is_finite());
    }

    #[test]
    fn test_russian_roulette_terminates() {
        let mut scene = Scene::new();
        let diff = scene.add_material(Material::Diffuse { albedo: Color::new(0.1, 0.1, 0.1) });
        scene.add_sphere(Vec3::new(0.0, 0.0, 0.0), 1000.0, diff);
        let config = PathTracerConfig { max_bounces: 100, samples_per_pixel: 1, russian_roulette_start: 2, background: Color::black() };
        let mut rng = Rng::new(42);
        let ray = Ray::new(Vec3::new(0.0, 0.0, 500.0), Vec3::new(0.0, 0.0, -1.0));
        // Should terminate due to RR with low albedo
        let c = trace_path(&scene, &ray, &config, &mut rng);
        assert!(c.r.is_finite());
    }
}
