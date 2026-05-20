//! 2D physics world container — owns bodies, shapes, joints.  Step function:
//! broadphase → narrowphase → solve contacts → integrate → sync transforms.
//! Gravity vector, fixed timestep sub-stepping, ray cast, collision filtering
//! (categories + masks), event callbacks (on_contact_begin, on_contact_end).

use std::collections::{HashMap, HashSet};

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn cross(self, o: Self) -> f64 { self.x * o.y - self.y * o.x }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn negate(self) -> Self { Self { x: -self.x, y: -self.y } }
    pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
    pub fn rotate(self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self { x: self.x * c - self.y * s, y: self.x * s + self.y * c }
    }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── AABB ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min: Vec2, max: Vec2) -> Self { Self { min, max } }
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
    }
}

// ── Body type ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Dynamic,
    Static,
    Kinematic,
}

// ── Collision filter ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionFilter {
    /// Category bits (what this body is).
    pub category: u16,
    /// Mask bits (what this body collides with).
    pub mask: u16,
}

impl CollisionFilter {
    pub fn new(category: u16, mask: u16) -> Self { Self { category, mask } }
    pub fn all() -> Self { Self { category: 0xFFFF, mask: 0xFFFF } }
    pub fn should_collide(&self, other: &CollisionFilter) -> bool {
        (self.category & other.mask) != 0 && (other.category & self.mask) != 0
    }
}

impl Default for CollisionFilter {
    fn default() -> Self { Self::all() }
}

// ── Shape ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Circle { radius: f64 },
    Box { half_width: f64, half_height: f64 },
    Polygon { vertices: Vec<Vec2> },
}

impl Shape {
    pub fn compute_aabb(&self, position: Vec2, rotation: f64) -> AABB {
        match self {
            Shape::Circle { radius } => AABB {
                min: Vec2::new(position.x - radius, position.y - radius),
                max: Vec2::new(position.x + radius, position.y + radius),
            },
            Shape::Box { half_width, half_height } => {
                let cos_a = rotation.cos().abs();
                let sin_a = rotation.sin().abs();
                let hx = half_width * cos_a + half_height * sin_a;
                let hy = half_width * sin_a + half_height * cos_a;
                AABB {
                    min: Vec2::new(position.x - hx, position.y - hy),
                    max: Vec2::new(position.x + hx, position.y + hy),
                }
            }
            Shape::Polygon { vertices } => {
                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;
                for v in vertices {
                    let w = position.add(v.rotate(rotation));
                    min_x = min_x.min(w.x);
                    min_y = min_y.min(w.y);
                    max_x = max_x.max(w.x);
                    max_y = max_y.max(w.y);
                }
                AABB { min: Vec2::new(min_x, min_y), max: Vec2::new(max_x, max_y) }
            }
        }
    }
}

// ── Body ─────────────────────────────────────────────────────

pub type BodyId = u64;

#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub id: BodyId,
    pub body_type: BodyType,
    pub position: Vec2,
    pub rotation: f64,
    pub linear_velocity: Vec2,
    pub angular_velocity: f64,
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia: f64,
    pub inv_inertia: f64,
    pub force: Vec2,
    pub torque: f64,
    pub linear_damping: f64,
    pub angular_damping: f64,
    pub gravity_scale: f64,
    pub shape: Shape,
    pub filter: CollisionFilter,
    pub restitution: f64,
    pub friction: f64,
}

impl Body {
    pub fn new_dynamic(id: BodyId, mass: f64, inertia: f64, shape: Shape) -> Self {
        Self {
            id, body_type: BodyType::Dynamic,
            position: Vec2::zero(), rotation: 0.0,
            linear_velocity: Vec2::zero(), angular_velocity: 0.0,
            mass, inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            inertia, inv_inertia: if inertia > 0.0 { 1.0 / inertia } else { 0.0 },
            force: Vec2::zero(), torque: 0.0,
            linear_damping: 0.0, angular_damping: 0.0,
            gravity_scale: 1.0, shape, filter: CollisionFilter::all(),
            restitution: 0.3, friction: 0.5,
        }
    }

    pub fn new_static(id: BodyId, shape: Shape) -> Self {
        Self {
            id, body_type: BodyType::Static,
            position: Vec2::zero(), rotation: 0.0,
            linear_velocity: Vec2::zero(), angular_velocity: 0.0,
            mass: 0.0, inv_mass: 0.0, inertia: 0.0, inv_inertia: 0.0,
            force: Vec2::zero(), torque: 0.0,
            linear_damping: 0.0, angular_damping: 0.0,
            gravity_scale: 0.0, shape, filter: CollisionFilter::all(),
            restitution: 0.3, friction: 0.5,
        }
    }

    pub fn aabb(&self) -> AABB {
        self.shape.compute_aabb(self.position, self.rotation)
    }

    pub fn integrate(&mut self, dt: f64, gravity: Vec2) {
        if self.body_type == BodyType::Static { return; }
        if self.body_type == BodyType::Kinematic {
            self.position = self.position.add(self.linear_velocity.scale(dt));
            self.rotation += self.angular_velocity * dt;
            return;
        }
        let accel = self.force.scale(self.inv_mass).add(gravity.scale(self.gravity_scale));
        let ang_accel = self.torque * self.inv_inertia;
        self.linear_velocity = self.linear_velocity.add(accel.scale(dt));
        self.angular_velocity += ang_accel * dt;
        self.linear_velocity = self.linear_velocity.scale((1.0 - self.linear_damping).max(0.0));
        self.angular_velocity *= (1.0 - self.angular_damping).max(0.0);
        self.position = self.position.add(self.linear_velocity.scale(dt));
        self.rotation += self.angular_velocity * dt;
        self.force = Vec2::zero();
        self.torque = 0.0;
    }
}

// ── Joint ────────────────────────────────────────────────────

pub type JointId = u64;

#[derive(Debug, Clone, PartialEq)]
pub enum Joint {
    Distance { id: JointId, body_a: BodyId, body_b: BodyId, target_length: f64 },
    Spring { id: JointId, body_a: BodyId, body_b: BodyId, rest_length: f64, stiffness: f64, damping: f64 },
    Revolute { id: JointId, body_a: BodyId, body_b: BodyId },
}

// ── Contact pair ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContactPair(pub BodyId, pub BodyId);

impl ContactPair {
    pub fn new(a: BodyId, b: BodyId) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
}

/// Collision event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactEvent {
    Begin(BodyId, BodyId),
    End(BodyId, BodyId),
}

// ── Ray cast ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCastResult {
    pub body_id: BodyId,
    pub point: Vec2,
    pub normal: Vec2,
    pub distance: f64,
}

// ── World config ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct WorldConfig {
    pub gravity: Vec2,
    pub fixed_dt: f64,
    pub solver_iterations: usize,
    pub broadphase_cell_size: f64,
}

impl WorldConfig {
    pub fn new() -> Self {
        Self {
            gravity: Vec2::new(0.0, -9.81),
            fixed_dt: 1.0 / 60.0,
            solver_iterations: 8,
            broadphase_cell_size: 10.0,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self { Self::new() }
}

// ── Physics World ────────────────────────────────────────────

pub struct PhysicsWorld2D {
    pub config: WorldConfig,
    bodies: HashMap<BodyId, Body>,
    joints: HashMap<JointId, Joint>,
    next_body_id: BodyId,
    next_joint_id: JointId,
    accumulator: f64,
    active_contacts: HashSet<ContactPair>,
    events: Vec<ContactEvent>,
}

impl PhysicsWorld2D {
    pub fn new(config: WorldConfig) -> Self {
        Self {
            config,
            bodies: HashMap::new(),
            joints: HashMap::new(),
            next_body_id: 1,
            next_joint_id: 1,
            accumulator: 0.0,
            active_contacts: HashSet::new(),
            events: Vec::new(),
        }
    }

    // ── Body management ──

    pub fn add_body(&mut self, mut body: Body) -> BodyId {
        let id = self.next_body_id;
        self.next_body_id += 1;
        body.id = id;
        self.bodies.insert(id, body);
        id
    }

    pub fn remove_body(&mut self, id: BodyId) -> Option<Body> {
        // Also remove joints referencing this body.
        let joint_ids: Vec<JointId> = self.joints.iter()
            .filter(|(_, j)| match j {
                Joint::Distance { body_a, body_b, .. } |
                Joint::Spring { body_a, body_b, .. } |
                Joint::Revolute { body_a, body_b, .. } => *body_a == id || *body_b == id,
            })
            .map(|(jid, _)| *jid)
            .collect();
        for jid in joint_ids {
            self.joints.remove(&jid);
        }
        // Remove contacts involving this body.
        self.active_contacts.retain(|cp| cp.0 != id && cp.1 != id);
        self.bodies.remove(&id)
    }

    pub fn body(&self, id: BodyId) -> Option<&Body> { self.bodies.get(&id) }
    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut Body> { self.bodies.get_mut(&id) }
    pub fn body_count(&self) -> usize { self.bodies.len() }

    pub fn body_ids(&self) -> Vec<BodyId> {
        self.bodies.keys().copied().collect()
    }

    // ── Joint management ──

    pub fn add_joint(&mut self, mut joint: Joint) -> JointId {
        let id = self.next_joint_id;
        self.next_joint_id += 1;
        match &mut joint {
            Joint::Distance { id: jid, .. } | Joint::Spring { id: jid, .. } |
            Joint::Revolute { id: jid, .. } => *jid = id,
        }
        self.joints.insert(id, joint);
        id
    }

    pub fn remove_joint(&mut self, id: JointId) -> Option<Joint> {
        self.joints.remove(&id)
    }

    pub fn joint_count(&self) -> usize { self.joints.len() }

    // ── Stepping ──

    /// Fixed-timestep step with accumulator-based sub-stepping.
    pub fn step(&mut self, dt: f64) {
        self.events.clear();
        self.accumulator += dt;
        let fixed_dt = self.config.fixed_dt;
        while self.accumulator >= fixed_dt {
            self.accumulator -= fixed_dt;
            self.fixed_step(fixed_dt);
        }
    }

    /// Single fixed-timestep step.
    fn fixed_step(&mut self, dt: f64) {
        // 1. Broadphase — compute AABBs and find overlapping pairs.
        let pairs = self.broadphase();

        // 2. Narrowphase + contact events.
        let new_contacts: HashSet<ContactPair> = pairs.into_iter().collect();
        // Begin events.
        for cp in &new_contacts {
            if !self.active_contacts.contains(cp) {
                self.events.push(ContactEvent::Begin(cp.0, cp.1));
            }
        }
        // End events.
        let old: Vec<ContactPair> = self.active_contacts.iter().copied().collect();
        for cp in &old {
            if !new_contacts.contains(cp) {
                self.events.push(ContactEvent::End(cp.0, cp.1));
            }
        }
        self.active_contacts = new_contacts;

        // 3. Simple contact resolution (push apart along overlap axis).
        let contact_pairs: Vec<ContactPair> = self.active_contacts.iter().copied().collect();
        for cp in &contact_pairs {
            self.resolve_contact(cp.0, cp.1, dt);
        }

        // 4. Solve joints.
        self.solve_joints(dt);

        // 5. Integrate.
        let gravity = self.config.gravity;
        let ids: Vec<BodyId> = self.bodies.keys().copied().collect();
        for id in ids {
            if let Some(body) = self.bodies.get_mut(&id) {
                body.integrate(dt, gravity);
            }
        }
    }

    fn broadphase(&self) -> Vec<ContactPair> {
        let body_vec: Vec<(BodyId, AABB, &CollisionFilter, BodyType)> = self.bodies.values()
            .map(|b| (b.id, b.aabb(), &b.filter, b.body_type))
            .collect();
        let mut pairs = Vec::new();
        let n = body_vec.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (id_a, ref aabb_a, filter_a, type_a) = body_vec[i];
                let (id_b, ref aabb_b, filter_b, type_b) = body_vec[j];
                // Skip static-static.
                if type_a == BodyType::Static && type_b == BodyType::Static { continue; }
                if !filter_a.should_collide(filter_b) { continue; }
                if aabb_a.overlaps(aabb_b) {
                    pairs.push(ContactPair::new(id_a, id_b));
                }
            }
        }
        pairs
    }

    fn resolve_contact(&mut self, id_a: BodyId, id_b: BodyId, _dt: f64) {
        // Simple circle-circle resolution for demonstration.
        let (pos_a, radius_a, inv_mass_a, type_a) = {
            let b = match self.bodies.get(&id_a) { Some(b) => b, None => return };
            let r = match &b.shape { Shape::Circle { radius } => *radius, Shape::Box { half_width, .. } => *half_width, _ => 1.0 };
            (b.position, r, b.inv_mass, b.body_type)
        };
        let (pos_b, radius_b, inv_mass_b, type_b) = {
            let b = match self.bodies.get(&id_b) { Some(b) => b, None => return };
            let r = match &b.shape { Shape::Circle { radius } => *radius, Shape::Box { half_width, .. } => *half_width, _ => 1.0 };
            (b.position, r, b.inv_mass, b.body_type)
        };

        let d = pos_b.sub(pos_a);
        let dist = d.length();
        let sum_r = radius_a + radius_b;
        if dist >= sum_r || dist < 1e-12 { return; }

        let normal = d.scale(1.0 / dist);
        let overlap = sum_r - dist;
        let inv_sum = inv_mass_a + inv_mass_b;
        if inv_sum < 1e-12 { return; }

        // Position correction.
        let correction = normal.scale(overlap / inv_sum * 0.5);
        if type_a == BodyType::Dynamic {
            if let Some(ba) = self.bodies.get_mut(&id_a) {
                ba.position = ba.position.sub(correction.scale(inv_mass_a));
            }
        }
        if type_b == BodyType::Dynamic {
            if let Some(bb) = self.bodies.get_mut(&id_b) {
                bb.position = bb.position.add(correction.scale(inv_mass_b));
            }
        }

        // Velocity resolution.
        let vel_a = self.bodies.get(&id_a).map(|b| b.linear_velocity).unwrap_or(Vec2::zero());
        let vel_b = self.bodies.get(&id_b).map(|b| b.linear_velocity).unwrap_or(Vec2::zero());
        let rel_vel = vel_b.sub(vel_a);
        let vn = rel_vel.dot(normal);
        if vn > 0.0 { return; } // separating

        let e = {
            let ra = self.bodies.get(&id_a).map(|b| b.restitution).unwrap_or(0.0);
            let rb = self.bodies.get(&id_b).map(|b| b.restitution).unwrap_or(0.0);
            (ra + rb) * 0.5
        };
        let j_val = -(1.0 + e) * vn / inv_sum;
        let impulse = normal.scale(j_val);

        if type_a == BodyType::Dynamic {
            if let Some(ba) = self.bodies.get_mut(&id_a) {
                ba.linear_velocity = ba.linear_velocity.sub(impulse.scale(inv_mass_a));
            }
        }
        if type_b == BodyType::Dynamic {
            if let Some(bb) = self.bodies.get_mut(&id_b) {
                bb.linear_velocity = bb.linear_velocity.add(impulse.scale(inv_mass_b));
            }
        }
    }

    fn solve_joints(&mut self, dt: f64) {
        let joint_list: Vec<Joint> = self.joints.values().cloned().collect();
        for joint in &joint_list {
            match joint {
                Joint::Distance { body_a, body_b, target_length, .. } => {
                    self.solve_distance_joint(*body_a, *body_b, *target_length, dt);
                }
                Joint::Spring { body_a, body_b, rest_length, stiffness, damping, .. } => {
                    self.solve_spring_joint(*body_a, *body_b, *rest_length, *stiffness, *damping, dt);
                }
                Joint::Revolute { body_a, body_b, .. } => {
                    self.solve_revolute_joint(*body_a, *body_b, dt);
                }
            }
        }
    }

    fn solve_distance_joint(&mut self, id_a: BodyId, id_b: BodyId, target: f64, _dt: f64) {
        let (pos_a, inv_mass_a) = match self.bodies.get(&id_a) {
            Some(b) => (b.position, b.inv_mass), None => return,
        };
        let (pos_b, inv_mass_b) = match self.bodies.get(&id_b) {
            Some(b) => (b.position, b.inv_mass), None => return,
        };
        let d = pos_b.sub(pos_a);
        let dist = d.length();
        if dist < 1e-12 { return; }
        let dir = d.scale(1.0 / dist);
        let error = dist - target;
        let inv_sum = inv_mass_a + inv_mass_b;
        if inv_sum < 1e-12 { return; }
        let correction = dir.scale(error / inv_sum);
        if let Some(ba) = self.bodies.get_mut(&id_a) {
            ba.position = ba.position.add(correction.scale(inv_mass_a));
        }
        if let Some(bb) = self.bodies.get_mut(&id_b) {
            bb.position = bb.position.sub(correction.scale(inv_mass_b));
        }
    }

    fn solve_spring_joint(&mut self, id_a: BodyId, id_b: BodyId, rest: f64, stiffness: f64, damp: f64, dt: f64) {
        let (pos_a, vel_a, inv_mass_a) = match self.bodies.get(&id_a) {
            Some(b) => (b.position, b.linear_velocity, b.inv_mass), None => return,
        };
        let (pos_b, vel_b, inv_mass_b) = match self.bodies.get(&id_b) {
            Some(b) => (b.position, b.linear_velocity, b.inv_mass), None => return,
        };
        let d = pos_b.sub(pos_a);
        let dist = d.length();
        if dist < 1e-12 { return; }
        let dir = d.scale(1.0 / dist);
        let extension = dist - rest;
        let spring_force = stiffness * extension;
        let rel_vel = vel_b.sub(vel_a).dot(dir);
        let damp_force = damp * rel_vel;
        let total = (spring_force + damp_force) * dt;
        let impulse = dir.scale(total);
        if let Some(ba) = self.bodies.get_mut(&id_a) {
            ba.linear_velocity = ba.linear_velocity.add(impulse.scale(inv_mass_a));
        }
        if let Some(bb) = self.bodies.get_mut(&id_b) {
            bb.linear_velocity = bb.linear_velocity.sub(impulse.scale(inv_mass_b));
        }
    }

    fn solve_revolute_joint(&mut self, id_a: BodyId, id_b: BodyId, _dt: f64) {
        let (pos_a, inv_mass_a) = match self.bodies.get(&id_a) {
            Some(b) => (b.position, b.inv_mass), None => return,
        };
        let (pos_b, inv_mass_b) = match self.bodies.get(&id_b) {
            Some(b) => (b.position, b.inv_mass), None => return,
        };
        let d = pos_b.sub(pos_a);
        let inv_sum = inv_mass_a + inv_mass_b;
        if inv_sum < 1e-12 { return; }
        let correction = d.scale(1.0 / inv_sum * 0.2);
        if let Some(ba) = self.bodies.get_mut(&id_a) {
            ba.position = ba.position.add(correction.scale(inv_mass_a));
        }
        if let Some(bb) = self.bodies.get_mut(&id_b) {
            bb.position = bb.position.sub(correction.scale(inv_mass_b));
        }
    }

    // ── Ray cast ──

    pub fn ray_cast(&self, origin: Vec2, direction: Vec2, max_dist: f64) -> Vec<RayCastResult> {
        let dir = direction.normalized();
        let mut results = Vec::new();

        for body in self.bodies.values() {
            match &body.shape {
                Shape::Circle { radius } => {
                    let oc = origin.sub(body.position);
                    let a = dir.dot(dir);
                    let b = 2.0 * oc.dot(dir);
                    let c = oc.dot(oc) - radius * radius;
                    let disc = b * b - 4.0 * a * c;
                    if disc < 0.0 { continue; }
                    let sqrt_disc = disc.sqrt();
                    let mut t = (-b - sqrt_disc) / (2.0 * a);
                    if t < 0.0 { t = (-b + sqrt_disc) / (2.0 * a); }
                    if t < 0.0 || t > max_dist { continue; }
                    let point = origin.add(dir.scale(t));
                    let normal = point.sub(body.position).normalized();
                    results.push(RayCastResult { body_id: body.id, point, normal, distance: t });
                }
                Shape::Box { half_width, half_height } => {
                    let bb = body.aabb();
                    let inv_dx = if dir.x.abs() > 1e-12 { 1.0 / dir.x } else { f64::MAX.copysign(dir.x) };
                    let inv_dy = if dir.y.abs() > 1e-12 { 1.0 / dir.y } else { f64::MAX.copysign(dir.y) };
                    let tx1 = (bb.min.x - origin.x) * inv_dx;
                    let tx2 = (bb.max.x - origin.x) * inv_dx;
                    let ty1 = (bb.min.y - origin.y) * inv_dy;
                    let ty2 = (bb.max.y - origin.y) * inv_dy;
                    let tmin = tx1.min(tx2).max(ty1.min(ty2));
                    let tmax = tx1.max(tx2).min(ty1.max(ty2));
                    if tmax < 0.0 || tmin > tmax { continue; }
                    let t = if tmin >= 0.0 { tmin } else { tmax };
                    if t < 0.0 || t > max_dist { continue; }
                    let point = origin.add(dir.scale(t));
                    let normal = if (t - tx1.min(tx2)).abs() < 1e-6 {
                        Vec2::new(if dir.x > 0.0 { -1.0 } else { 1.0 }, 0.0)
                    } else {
                        Vec2::new(0.0, if dir.y > 0.0 { -1.0 } else { 1.0 })
                    };
                    results.push(RayCastResult { body_id: body.id, point, normal, distance: t });
                }
                _ => {}
            }
        }
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    // ── Events ──

    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.events)
    }

    // ── Queries ──

    pub fn bodies_iter(&self) -> impl Iterator<Item = &Body> {
        self.bodies.values()
    }

    pub fn gravity(&self) -> Vec2 { self.config.gravity }
    pub fn set_gravity(&mut self, g: Vec2) { self.config.gravity = g; }

    pub fn active_contact_count(&self) -> usize { self.active_contacts.len() }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v2_approx(a: Vec2, b: Vec2) -> bool { approx(a.x, b.x) && approx(a.y, b.y) }

    fn default_world() -> PhysicsWorld2D {
        PhysicsWorld2D::new(WorldConfig::new())
    }

    fn circle_body(mass: f64, radius: f64) -> Body {
        Body::new_dynamic(0, mass, mass * radius * radius * 0.5, Shape::Circle { radius })
    }

    fn static_floor() -> Body {
        let mut b = Body::new_static(0, Shape::Box { half_width: 100.0, half_height: 1.0 });
        b.position = Vec2::new(0.0, -10.0);
        b
    }

    #[test]
    fn add_remove_body() {
        let mut w = default_world();
        let id = w.add_body(circle_body(1.0, 1.0));
        assert_eq!(w.body_count(), 1);
        w.remove_body(id);
        assert_eq!(w.body_count(), 0);
    }

    #[test]
    fn add_joint() {
        let mut w = default_world();
        let a = w.add_body(circle_body(1.0, 1.0));
        let b = w.add_body(circle_body(1.0, 1.0));
        let jid = w.add_joint(Joint::Distance { id: 0, body_a: a, body_b: b, target_length: 5.0 });
        assert_eq!(w.joint_count(), 1);
        w.remove_joint(jid);
        assert_eq!(w.joint_count(), 0);
    }

    #[test]
    fn remove_body_removes_joints() {
        let mut w = default_world();
        let a = w.add_body(circle_body(1.0, 1.0));
        let b = w.add_body(circle_body(1.0, 1.0));
        w.add_joint(Joint::Distance { id: 0, body_a: a, body_b: b, target_length: 5.0 });
        w.remove_body(a);
        assert_eq!(w.joint_count(), 0);
    }

    #[test]
    fn step_applies_gravity() {
        let mut w = default_world();
        let id = w.add_body(circle_body(1.0, 1.0));
        w.step(1.0 / 60.0);
        let b = w.body(id).unwrap();
        assert!(b.linear_velocity.y < 0.0); // gravity pulls down
    }

    #[test]
    fn static_body_doesnt_move() {
        let mut w = default_world();
        let id = w.add_body(static_floor());
        w.step(1.0 / 60.0);
        let b = w.body(id).unwrap();
        assert!(v2_approx(b.position, Vec2::new(0.0, -10.0)));
    }

    #[test]
    fn collision_filter_works() {
        let cat_a: u16 = 0x0001;
        let cat_b: u16 = 0x0002;
        let fa = CollisionFilter::new(cat_a, cat_a); // only collides with cat_a
        let fb = CollisionFilter::new(cat_b, cat_b); // only collides with cat_b
        assert!(!fa.should_collide(&fb));
        assert!(fa.should_collide(&fa));
    }

    #[test]
    fn collision_filter_symmetric() {
        let fa = CollisionFilter::new(1, 2);
        let fb = CollisionFilter::new(2, 1);
        assert!(fa.should_collide(&fb));
    }

    #[test]
    fn ray_cast_hits_circle() {
        let mut w = default_world();
        let id = w.add_body(circle_body(1.0, 1.0));
        w.body_mut(id).unwrap().position = Vec2::new(5.0, 0.0);
        let hits = w.ray_cast(Vec2::zero(), Vec2::new(1.0, 0.0), 100.0);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].body_id, id);
        assert!(approx(hits[0].distance, 4.0));
    }

    #[test]
    fn ray_cast_misses() {
        let mut w = default_world();
        let id = w.add_body(circle_body(1.0, 1.0));
        w.body_mut(id).unwrap().position = Vec2::new(5.0, 10.0);
        let hits = w.ray_cast(Vec2::zero(), Vec2::new(1.0, 0.0), 100.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn ray_cast_sorted_by_distance() {
        let mut w = default_world();
        let id1 = w.add_body(circle_body(1.0, 0.5));
        let id2 = w.add_body(circle_body(1.0, 0.5));
        w.body_mut(id1).unwrap().position = Vec2::new(10.0, 0.0);
        w.body_mut(id2).unwrap().position = Vec2::new(5.0, 0.0);
        let hits = w.ray_cast(Vec2::zero(), Vec2::new(1.0, 0.0), 100.0);
        assert!(hits.len() >= 2);
        assert!(hits[0].distance < hits[1].distance);
    }

    #[test]
    fn collision_events() {
        let mut w = PhysicsWorld2D::new(WorldConfig {
            gravity: Vec2::zero(), ..WorldConfig::new()
        });
        let a = w.add_body(circle_body(1.0, 1.0));
        let b = w.add_body(circle_body(1.0, 1.0));
        // Place them overlapping.
        w.body_mut(a).unwrap().position = Vec2::new(0.0, 0.0);
        w.body_mut(b).unwrap().position = Vec2::new(1.0, 0.0);
        w.step(1.0 / 60.0);
        let events = w.drain_events();
        assert!(events.iter().any(|e| matches!(e, ContactEvent::Begin(_, _))));
    }

    #[test]
    fn contact_end_event() {
        let mut w = PhysicsWorld2D::new(WorldConfig {
            gravity: Vec2::zero(), ..WorldConfig::new()
        });
        let a = w.add_body(circle_body(1.0, 1.0));
        let b = w.add_body(circle_body(1.0, 1.0));
        w.body_mut(a).unwrap().position = Vec2::new(0.0, 0.0);
        w.body_mut(b).unwrap().position = Vec2::new(1.0, 0.0);
        w.step(1.0 / 60.0);
        let _ = w.drain_events();
        // Move them apart.
        w.body_mut(b).unwrap().position = Vec2::new(100.0, 0.0);
        w.body_mut(b).unwrap().linear_velocity = Vec2::zero();
        w.step(1.0 / 60.0);
        let events = w.drain_events();
        assert!(events.iter().any(|e| matches!(e, ContactEvent::End(_, _))));
    }

    #[test]
    fn set_gravity() {
        let mut w = default_world();
        w.set_gravity(Vec2::new(0.0, -20.0));
        assert!(v2_approx(w.gravity(), Vec2::new(0.0, -20.0)));
    }

    #[test]
    fn body_iteration() {
        let mut w = default_world();
        w.add_body(circle_body(1.0, 1.0));
        w.add_body(circle_body(2.0, 1.0));
        assert_eq!(w.bodies_iter().count(), 2);
    }

    #[test]
    fn sub_stepping() {
        let mut w = PhysicsWorld2D::new(WorldConfig {
            gravity: Vec2::new(0.0, -10.0),
            fixed_dt: 1.0 / 120.0, // 120 Hz
            ..WorldConfig::new()
        });
        let id = w.add_body(circle_body(1.0, 1.0));
        w.step(1.0 / 60.0); // should do 2 sub-steps at 120 Hz
        let b = w.body(id).unwrap();
        assert!(b.linear_velocity.y < 0.0);
    }

    #[test]
    fn kinematic_body_moves() {
        let mut w = default_world();
        let mut b = Body::new_dynamic(0, 0.0, 0.0, Shape::Circle { radius: 1.0 });
        b.body_type = BodyType::Kinematic;
        b.linear_velocity = Vec2::new(10.0, 0.0);
        b.inv_mass = 0.0;
        b.inv_inertia = 0.0;
        let id = w.add_body(b);
        w.step(1.0 / 60.0);
        let b = w.body(id).unwrap();
        assert!(b.position.x > 0.0);
    }

    #[test]
    fn shape_circle_aabb() {
        let s = Shape::Circle { radius: 2.0 };
        let bb = s.compute_aabb(Vec2::new(5.0, 5.0), 0.0);
        assert!(v2_approx(bb.min, Vec2::new(3.0, 3.0)));
        assert!(v2_approx(bb.max, Vec2::new(7.0, 7.0)));
    }

    #[test]
    fn shape_box_aabb_no_rotation() {
        let s = Shape::Box { half_width: 2.0, half_height: 1.0 };
        let bb = s.compute_aabb(Vec2::zero(), 0.0);
        assert!(v2_approx(bb.min, Vec2::new(-2.0, -1.0)));
        assert!(v2_approx(bb.max, Vec2::new(2.0, 1.0)));
    }

    #[test]
    fn contact_pair_canonical() {
        assert_eq!(ContactPair::new(3, 1), ContactPair::new(1, 3));
    }

    #[test]
    fn world_config_defaults() {
        let c = WorldConfig::new();
        assert!(approx(c.fixed_dt, 1.0 / 60.0));
        assert_eq!(c.solver_iterations, 8);
    }

    #[test]
    fn collision_pushes_apart() {
        let mut w = PhysicsWorld2D::new(WorldConfig { gravity: Vec2::zero(), ..WorldConfig::new() });
        let a = w.add_body(circle_body(1.0, 1.0));
        let b = w.add_body(circle_body(1.0, 1.0));
        w.body_mut(a).unwrap().position = Vec2::new(0.0, 0.0);
        w.body_mut(b).unwrap().position = Vec2::new(1.0, 0.0); // overlapping
        w.step(1.0 / 60.0);
        let pa = w.body(a).unwrap().position;
        let pb = w.body(b).unwrap().position;
        let dist = pb.sub(pa).length();
        assert!(dist > 1.0); // pushed apart beyond overlap
    }

    #[test]
    fn spring_joint_pulls_together() {
        let mut w = PhysicsWorld2D::new(WorldConfig { gravity: Vec2::zero(), ..WorldConfig::new() });
        let a = w.add_body(circle_body(1.0, 0.5));
        let b = w.add_body(circle_body(1.0, 0.5));
        w.body_mut(a).unwrap().position = Vec2::new(0.0, 0.0);
        w.body_mut(b).unwrap().position = Vec2::new(10.0, 0.0);
        w.add_joint(Joint::Spring {
            id: 0, body_a: a, body_b: b,
            rest_length: 2.0, stiffness: 100.0, damping: 1.0,
        });
        for _ in 0..60 { w.step(1.0 / 60.0); }
        let pa = w.body(a).unwrap().position;
        let pb = w.body(b).unwrap().position;
        let dist = pb.sub(pa).length();
        assert!(dist < 10.0); // should have been pulled closer
    }

    #[test]
    fn ray_cast_box() {
        let mut w = default_world();
        let mut b = Body::new_static(0, Shape::Box { half_width: 2.0, half_height: 2.0 });
        b.position = Vec2::new(5.0, 0.0);
        let id = w.add_body(b);
        let hits = w.ray_cast(Vec2::zero(), Vec2::new(1.0, 0.0), 100.0);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].body_id, id);
    }

    #[test]
    fn body_ids_list() {
        let mut w = default_world();
        let a = w.add_body(circle_body(1.0, 1.0));
        let b = w.add_body(circle_body(1.0, 1.0));
        let ids = w.body_ids();
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
    }
}
