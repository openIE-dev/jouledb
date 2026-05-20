//! 3D Physics — rigid bodies, AABB/sphere collision, gravity integration,
//! impulse-based response, spatial hash broad phase, and raycasting.

use std::collections::HashMap;

use crate::webgl::Vec3;

// ── AABB ──────────────────────────────────────────────────────

/// Axis-Aligned Bounding Box for collision detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsAabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl PhysicsAabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Test overlap with another AABB.
    pub fn intersects(&self, other: &PhysicsAabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Raycast against this AABB. Returns `Some(t)` for the nearest intersection.
    pub fn raycast(&self, origin: &Vec3, dir: &Vec3) -> Option<f64> {
        let inv_d = Vec3::new(
            if dir.x.abs() > 1e-12 { 1.0 / dir.x } else { f64::MAX.copysign(dir.x) },
            if dir.y.abs() > 1e-12 { 1.0 / dir.y } else { f64::MAX.copysign(dir.y) },
            if dir.z.abs() > 1e-12 { 1.0 / dir.z } else { f64::MAX.copysign(dir.z) },
        );

        let t1 = (self.min.x - origin.x) * inv_d.x;
        let t2 = (self.max.x - origin.x) * inv_d.x;
        let t3 = (self.min.y - origin.y) * inv_d.y;
        let t4 = (self.max.y - origin.y) * inv_d.y;
        let t5 = (self.min.z - origin.z) * inv_d.z;
        let t6 = (self.max.z - origin.z) * inv_d.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax < 0.0 || tmin > tmax {
            return None;
        }
        Some(if tmin >= 0.0 { tmin } else { tmax })
    }
}

// ── Sphere collider ───────────────────────────────────────────

/// Bounding sphere for collision detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f64,
}

impl BoundingSphere {
    pub fn new(center: Vec3, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Test overlap with another sphere.
    pub fn intersects(&self, other: &BoundingSphere) -> bool {
        let dist_sq = (self.center - other.center).length_squared();
        let r = self.radius + other.radius;
        dist_sq <= r * r
    }

    /// Test overlap with an AABB.
    pub fn intersects_aabb(&self, aabb: &PhysicsAabb) -> bool {
        let closest = Vec3::new(
            self.center.x.clamp(aabb.min.x, aabb.max.x),
            self.center.y.clamp(aabb.min.y, aabb.max.y),
            self.center.z.clamp(aabb.min.z, aabb.max.z),
        );
        let dist_sq = (self.center - closest).length_squared();
        dist_sq <= self.radius * self.radius
    }

    /// Raycast against this sphere. Returns `Some(t)` for the nearest intersection.
    pub fn raycast(&self, origin: &Vec3, dir: &Vec3) -> Option<f64> {
        let oc = *origin - self.center;
        let a = dir.dot(dir);
        let b = 2.0 * oc.dot(dir);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);
        if t1 >= 0.0 {
            Some(t1)
        } else if t2 >= 0.0 {
            Some(t2)
        } else {
            None
        }
    }
}

// ── Collider ──────────────────────────────────────────────────

/// Collider shape attached to a rigid body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Collider {
    Aabb(PhysicsAabb),
    Sphere(BoundingSphere),
}

impl Collider {
    /// Get the AABB enclosing this collider.
    pub fn enclosing_aabb(&self) -> PhysicsAabb {
        match self {
            Collider::Aabb(a) => *a,
            Collider::Sphere(s) => PhysicsAabb::from_center_half(
                s.center,
                Vec3::new(s.radius, s.radius, s.radius),
            ),
        }
    }
}

// ── RigidBody ─────────────────────────────────────────────────

/// Unique body identifier.
pub type BodyId = u64;

/// A rigid body with mass, velocity, angular velocity, and force accumulator.
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: BodyId,
    pub position: Vec3,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub force: Vec3,
    pub torque: Vec3,
    pub mass: f64,
    pub restitution: f64,
    pub collider: Collider,
    pub is_static: bool,
}

impl RigidBody {
    pub fn new(id: BodyId, mass: f64, collider: Collider) -> Self {
        Self {
            id,
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            angular_velocity: Vec3::zero(),
            force: Vec3::zero(),
            torque: Vec3::zero(),
            mass: mass.max(0.001),
            restitution: 0.5,
            collider,
            is_static: false,
        }
    }

    pub fn new_static(id: BodyId, collider: Collider) -> Self {
        Self {
            id,
            position: Vec3::zero(),
            velocity: Vec3::zero(),
            angular_velocity: Vec3::zero(),
            force: Vec3::zero(),
            torque: Vec3::zero(),
            mass: f64::MAX,
            restitution: 0.5,
            collider,
            is_static: true,
        }
    }

    pub fn inverse_mass(&self) -> f64 {
        if self.is_static { 0.0 } else { 1.0 / self.mass }
    }

    pub fn apply_force(&mut self, f: Vec3) {
        self.force = self.force + f;
    }

    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if !self.is_static {
            self.velocity = self.velocity + impulse * self.inverse_mass();
        }
    }

    /// Get the world-space collider (translated by position).
    pub fn world_collider(&self) -> Collider {
        match self.collider {
            Collider::Aabb(a) => Collider::Aabb(PhysicsAabb {
                min: a.min + self.position,
                max: a.max + self.position,
            }),
            Collider::Sphere(s) => Collider::Sphere(BoundingSphere {
                center: s.center + self.position,
                radius: s.radius,
            }),
        }
    }
}

// ── Contact ───────────────────────────────────────────────────

/// A collision contact between two bodies.
#[derive(Debug, Clone)]
pub struct Contact {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub normal: Vec3,
    pub penetration: f64,
}

// ── Collision detection helpers ───────────────────────────────

/// Test collision between two world-space colliders.
pub fn collide(a: &Collider, b: &Collider) -> Option<(Vec3, f64)> {
    match (a, b) {
        (Collider::Aabb(aa), Collider::Aabb(bb)) => collide_aabb_aabb(aa, bb),
        (Collider::Sphere(sa), Collider::Sphere(sb)) => collide_sphere_sphere(sa, sb),
        (Collider::Aabb(aa), Collider::Sphere(sb)) => collide_aabb_sphere(aa, sb),
        (Collider::Sphere(sa), Collider::Aabb(bb)) => {
            collide_aabb_sphere(bb, sa).map(|(n, p)| (n * -1.0, p))
        }
    }
}

fn collide_aabb_aabb(a: &PhysicsAabb, b: &PhysicsAabb) -> Option<(Vec3, f64)> {
    if !a.intersects(b) {
        return None;
    }
    // Find axis of minimum penetration.
    let overlaps = [
        (a.max.x - b.min.x, Vec3::new(1.0, 0.0, 0.0)),
        (b.max.x - a.min.x, Vec3::new(-1.0, 0.0, 0.0)),
        (a.max.y - b.min.y, Vec3::new(0.0, 1.0, 0.0)),
        (b.max.y - a.min.y, Vec3::new(0.0, -1.0, 0.0)),
        (a.max.z - b.min.z, Vec3::new(0.0, 0.0, 1.0)),
        (b.max.z - a.min.z, Vec3::new(0.0, 0.0, -1.0)),
    ];
    let mut min_pen = f64::MAX;
    let mut normal = Vec3::zero();
    for (pen, n) in &overlaps {
        if *pen < min_pen {
            min_pen = *pen;
            normal = *n;
        }
    }
    Some((normal, min_pen))
}

fn collide_sphere_sphere(a: &BoundingSphere, b: &BoundingSphere) -> Option<(Vec3, f64)> {
    let diff = b.center - a.center;
    let dist = diff.length();
    let sum_r = a.radius + b.radius;
    if dist >= sum_r {
        return None;
    }
    let normal = if dist > 1e-12 { diff * (1.0 / dist) } else { Vec3::up() };
    Some((normal, sum_r - dist))
}

fn collide_aabb_sphere(a: &PhysicsAabb, s: &BoundingSphere) -> Option<(Vec3, f64)> {
    let closest = Vec3::new(
        s.center.x.clamp(a.min.x, a.max.x),
        s.center.y.clamp(a.min.y, a.max.y),
        s.center.z.clamp(a.min.z, a.max.z),
    );
    let diff = s.center - closest;
    let dist = diff.length();
    if dist >= s.radius {
        return None;
    }
    let normal = if dist > 1e-12 { diff * (1.0 / dist) } else { Vec3::up() };
    Some((normal, s.radius - dist))
}

// ── Spatial Hash Grid (Broad Phase) ───────────────────────────

/// Spatial hash grid for broad phase collision detection.
pub struct SpatialHashGrid {
    cell_size: f64,
    cells: HashMap<(i64, i64, i64), Vec<BodyId>>,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size: cell_size.max(0.01),
            cells: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    fn cell_coords(&self, pos: &Vec3) -> (i64, i64, i64) {
        (
            (pos.x / self.cell_size).floor() as i64,
            (pos.y / self.cell_size).floor() as i64,
            (pos.z / self.cell_size).floor() as i64,
        )
    }

    /// Insert a body's AABB into the grid.
    pub fn insert(&mut self, body_id: BodyId, aabb: &PhysicsAabb) {
        let min_cell = self.cell_coords(&aabb.min);
        let max_cell = self.cell_coords(&aabb.max);
        for x in min_cell.0..=max_cell.0 {
            for y in min_cell.1..=max_cell.1 {
                for z in min_cell.2..=max_cell.2 {
                    self.cells.entry((x, y, z)).or_default().push(body_id);
                }
            }
        }
    }

    /// Query potential collision pairs (unordered, may contain duplicates).
    pub fn potential_pairs(&self) -> Vec<(BodyId, BodyId)> {
        let mut pairs = Vec::new();
        for cell in self.cells.values() {
            for i in 0..cell.len() {
                for j in (i + 1)..cell.len() {
                    let a = cell[i].min(cell[j]);
                    let b = cell[i].max(cell[j]);
                    pairs.push((a, b));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }
}

// ── PhysicsWorld ──────────────────────────────────────────────

/// Simple 3D physics world.
pub struct PhysicsWorld {
    pub bodies: Vec<RigidBody>,
    pub gravity: Vec3,
    grid: SpatialHashGrid,
}

impl PhysicsWorld {
    pub fn new(gravity: Vec3, cell_size: f64) -> Self {
        Self {
            bodies: Vec::new(),
            gravity,
            grid: SpatialHashGrid::new(cell_size),
        }
    }

    pub fn add_body(&mut self, body: RigidBody) {
        self.bodies.push(body);
    }

    pub fn body(&self, id: BodyId) -> Option<&RigidBody> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    /// Step the simulation forward by `dt` seconds (Euler integration).
    pub fn step(&mut self, dt: f64) {
        // Apply gravity and integrate.
        for body in &mut self.bodies {
            if body.is_static {
                continue;
            }
            let gravity_force = self.gravity * body.mass;
            body.force = body.force + gravity_force;

            let accel = body.force * body.inverse_mass();
            body.velocity = body.velocity + accel * dt;
            body.position = body.position + body.velocity * dt;

            // Simple angular integration.
            // (No inertia tensor — just direct angular velocity.)

            // Clear accumulators.
            body.force = Vec3::zero();
            body.torque = Vec3::zero();
        }

        // Broad phase.
        self.grid.clear();
        for body in &self.bodies {
            let wc = body.world_collider();
            self.grid.insert(body.id, &wc.enclosing_aabb());
        }
        let pairs = self.grid.potential_pairs();

        // Narrow phase + resolve.
        let mut contacts = Vec::new();
        for (a_id, b_id) in pairs {
            let (ca, cb, rest) = {
                let ba = self.bodies.iter().find(|b| b.id == a_id).unwrap();
                let bb = self.bodies.iter().find(|b| b.id == b_id).unwrap();
                (ba.world_collider(), bb.world_collider(), (ba.restitution + bb.restitution) * 0.5)
            };
            if let Some((normal, penetration)) = collide(&ca, &cb) {
                contacts.push((a_id, b_id, normal, penetration, rest));
            }
        }

        for (a_id, b_id, normal, _penetration, restitution) in contacts {
            // Impulse-based elastic response.
            let (va, vb, inv_ma, inv_mb) = {
                let ba = self.bodies.iter().find(|b| b.id == a_id).unwrap();
                let bb = self.bodies.iter().find(|b| b.id == b_id).unwrap();
                (ba.velocity, bb.velocity, ba.inverse_mass(), bb.inverse_mass())
            };
            let relative_vel = vb - va;
            let vel_along_normal = relative_vel.dot(&normal);
            if vel_along_normal > 0.0 {
                continue; // separating
            }
            let j = -(1.0 + restitution) * vel_along_normal / (inv_ma + inv_mb);
            let impulse = normal * j;

            if let Some(ba) = self.body_mut(a_id) {
                ba.apply_impulse(impulse * -1.0);
            }
            if let Some(bb) = self.body_mut(b_id) {
                bb.apply_impulse(impulse);
            }
        }
    }

    /// Raycast against all bodies. Returns (BodyId, t) of the nearest hit.
    pub fn raycast(&self, origin: &Vec3, direction: &Vec3) -> Option<(BodyId, f64)> {
        let dir = direction.normalize();
        let mut nearest: Option<(BodyId, f64)> = None;
        for body in &self.bodies {
            let wc = body.world_collider();
            let t = match wc {
                Collider::Aabb(a) => a.raycast(origin, &dir),
                Collider::Sphere(s) => s.raycast(origin, &dir),
            };
            if let Some(t) = t {
                if nearest.is_none() || t < nearest.unwrap().1 {
                    nearest = Some((body.id, t));
                }
            }
        }
        nearest
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn aabb_intersection() {
        let a = PhysicsAabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let b = PhysicsAabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.0, 2.0, 2.0));
        assert!(a.intersects(&b));
        let c = PhysicsAabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn sphere_intersection() {
        let a = BoundingSphere::new(Vec3::zero(), 1.0);
        let b = BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(a.intersects(&b));
        let c = BoundingSphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn sphere_aabb_intersection() {
        let s = BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        let a = PhysicsAabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(s.intersects_aabb(&a));
        let s2 = BoundingSphere::new(Vec3::new(5.0, 0.0, 0.0), 0.1);
        assert!(!s2.intersects_aabb(&a));
    }

    #[test]
    fn aabb_raycast_hit() {
        let a = PhysicsAabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let t = a.raycast(&origin, &dir).unwrap();
        assert!((t - 4.0).abs() < EPS);
    }

    #[test]
    fn aabb_raycast_miss() {
        let a = PhysicsAabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(5.0, 5.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        assert!(a.raycast(&origin, &dir).is_none());
    }

    #[test]
    fn sphere_raycast_hit() {
        let s = BoundingSphere::new(Vec3::zero(), 1.0);
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let t = s.raycast(&origin, &dir).unwrap();
        assert!((t - 4.0).abs() < EPS);
    }

    #[test]
    fn gravity_integration() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0), 10.0);
        let body = RigidBody::new(
            1,
            1.0,
            Collider::Sphere(BoundingSphere::new(Vec3::zero(), 0.5)),
        );
        world.add_body(body);
        world.step(1.0);
        let b = world.body(1).unwrap();
        // After 1s of gravity: v = -9.81, y = -9.81
        assert!((b.velocity.y - (-9.81)).abs() < EPS);
        assert!((b.position.y - (-9.81)).abs() < EPS);
    }

    #[test]
    fn static_body_does_not_move() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0), 10.0);
        let body = RigidBody::new_static(
            1,
            Collider::Aabb(PhysicsAabb::new(
                Vec3::new(-10.0, -1.0, -10.0),
                Vec3::new(10.0, 0.0, 10.0),
            )),
        );
        world.add_body(body);
        world.step(1.0);
        let b = world.body(1).unwrap();
        assert!((b.position.y).abs() < EPS);
    }

    #[test]
    fn collision_response_separates_bodies() {
        let mut world = PhysicsWorld::new(Vec3::zero(), 10.0);
        let mut a = RigidBody::new(
            1,
            1.0,
            Collider::Sphere(BoundingSphere::new(Vec3::zero(), 1.0)),
        );
        a.velocity = Vec3::new(1.0, 0.0, 0.0);
        a.position = Vec3::new(-0.5, 0.0, 0.0);

        let mut b = RigidBody::new(
            2,
            1.0,
            Collider::Sphere(BoundingSphere::new(Vec3::zero(), 1.0)),
        );
        b.velocity = Vec3::new(-1.0, 0.0, 0.0);
        b.position = Vec3::new(0.5, 0.0, 0.0);

        world.add_body(a);
        world.add_body(b);
        world.step(0.001); // tiny step to trigger collision

        let ba = world.body(1).unwrap();
        let bb = world.body(2).unwrap();
        // After collision, bodies should be moving apart.
        assert!(ba.velocity.x < 0.0 || bb.velocity.x > 0.0);
    }

    #[test]
    fn spatial_hash_finds_pairs() {
        let mut grid = SpatialHashGrid::new(5.0);
        grid.insert(
            1,
            &PhysicsAabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        );
        grid.insert(
            2,
            &PhysicsAabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
        );
        let pairs = grid.potential_pairs();
        assert!(pairs.contains(&(1, 2)));
    }

    #[test]
    fn world_raycast() {
        let mut world = PhysicsWorld::new(Vec3::zero(), 10.0);
        let mut body = RigidBody::new(
            42,
            1.0,
            Collider::Sphere(BoundingSphere::new(Vec3::zero(), 1.0)),
        );
        body.position = Vec3::new(0.0, 0.0, -5.0);
        world.add_body(body);

        let hit = world.raycast(&Vec3::new(0.0, 0.0, 0.0), &Vec3::new(0.0, 0.0, -1.0));
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().0, 42);
    }

    #[test]
    fn apply_force_then_integrate() {
        let mut world = PhysicsWorld::new(Vec3::zero(), 10.0);
        let body = RigidBody::new(
            1,
            2.0,
            Collider::Sphere(BoundingSphere::new(Vec3::zero(), 0.5)),
        );
        world.add_body(body);
        world.body_mut(1).unwrap().apply_force(Vec3::new(10.0, 0.0, 0.0));
        world.step(1.0);
        let b = world.body(1).unwrap();
        // F=10, m=2, a=5, v=5*1=5, pos=5*1=5
        assert!((b.velocity.x - 5.0).abs() < EPS);
        assert!((b.position.x - 5.0).abs() < EPS);
    }
}
