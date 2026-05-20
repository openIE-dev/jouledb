//! 3D Contact Manifold Management — up to 4 contact points per body pair,
//! contact reduction (maximize contact area), frame-to-frame persistence
//! via closest-point matching, warm-start impulse caching, and age-based
//! stale contact removal.

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f64 { self.x * r.x + self.y * r.y + self.z * r.z }
    pub fn cross(self, r: Self) -> Self {
        Self {
            x: self.y * r.z - self.z * r.y,
            y: self.z * r.x - self.x * r.z,
            z: self.x * r.y - self.y * r.x,
        }
    }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
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

// ── Contact Point ────────────────────────────────────────────

/// Maximum contacts per manifold.
pub const MAX_CONTACTS: usize = 4;

/// Unique identifier for a body pair.
pub type BodyPairId = (u64, u64);

/// Feature ID for contact matching across frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureId {
    pub body_a_feature: u32,
    pub body_b_feature: u32,
}

/// A single contact point in a manifold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    /// World-space contact point on body A surface.
    pub point_on_a: Vec3,
    /// World-space contact point on body B surface.
    pub point_on_b: Vec3,
    /// Contact normal (from A to B).
    pub normal: Vec3,
    /// Penetration depth (positive when overlapping).
    pub depth: f64,
    /// Accumulated normal impulse (for warm-starting).
    pub normal_impulse: f64,
    /// Accumulated tangent impulse 1.
    pub tangent_impulse_1: f64,
    /// Accumulated tangent impulse 2.
    pub tangent_impulse_2: f64,
    /// Optional feature ID for matching.
    pub feature_id: Option<FeatureId>,
    /// Age in frames since this contact was created or refreshed.
    pub age: u32,
}

impl ContactPoint {
    pub fn new(point_on_a: Vec3, point_on_b: Vec3, normal: Vec3, depth: f64) -> Self {
        Self {
            point_on_a,
            point_on_b,
            normal,
            depth,
            normal_impulse: 0.0,
            tangent_impulse_1: 0.0,
            tangent_impulse_2: 0.0,
            feature_id: None,
            age: 0,
        }
    }

    pub fn with_feature_id(mut self, fid: FeatureId) -> Self {
        self.feature_id = Some(fid);
        self
    }

    /// Average contact point.
    pub fn world_point(&self) -> Vec3 {
        (self.point_on_a + self.point_on_b) * 0.5
    }
}

// ── Contact Manifold ─────────────────────────────────────────

/// A contact manifold holding up to MAX_CONTACTS contact points between
/// two bodies. Supports contact persistence, reduction, and warm-starting.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub body_a: u64,
    pub body_b: u64,
    pub contacts: Vec<ContactPoint>,
    /// Number of frames since last refresh.
    pub stale_count: u32,
}

impl ContactManifold {
    pub fn new(body_a: u64, body_b: u64) -> Self {
        Self {
            body_a,
            body_b,
            contacts: Vec::new(),
            stale_count: 0,
        }
    }

    pub fn pair_id(&self) -> BodyPairId {
        if self.body_a <= self.body_b {
            (self.body_a, self.body_b)
        } else {
            (self.body_b, self.body_a)
        }
    }

    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }

    /// Add a new contact. If at capacity, replace the one that least affects
    /// the contact patch area.
    pub fn add_contact(&mut self, contact: ContactPoint) {
        // Try to match existing contact by feature ID
        if let Some(fid) = contact.feature_id {
            if let Some(existing) = self.contacts.iter_mut()
                .find(|c| c.feature_id == Some(fid))
            {
                // Preserve accumulated impulses (warm-starting)
                let ni = existing.normal_impulse;
                let ti1 = existing.tangent_impulse_1;
                let ti2 = existing.tangent_impulse_2;
                *existing = contact;
                existing.normal_impulse = ni;
                existing.tangent_impulse_1 = ti1;
                existing.tangent_impulse_2 = ti2;
                existing.age = 0;
                return;
            }
        }

        // Try to match by closest point
        let match_threshold = 0.02; // 2cm
        if let Some(idx) = self.find_closest_contact(&contact, match_threshold) {
            let ni = self.contacts[idx].normal_impulse;
            let ti1 = self.contacts[idx].tangent_impulse_1;
            let ti2 = self.contacts[idx].tangent_impulse_2;
            self.contacts[idx] = contact;
            self.contacts[idx].normal_impulse = ni;
            self.contacts[idx].tangent_impulse_1 = ti1;
            self.contacts[idx].tangent_impulse_2 = ti2;
            self.contacts[idx].age = 0;
            return;
        }

        if self.contacts.len() < MAX_CONTACTS {
            self.contacts.push(contact);
        } else {
            // Replace the contact that preserves the maximum area
            let replace_idx = self.find_least_significant(&contact);
            self.contacts[replace_idx] = contact;
        }
    }

    /// Find the contact closest to the new one.
    fn find_closest_contact(&self, new: &ContactPoint, threshold: f64) -> Option<usize> {
        let threshold_sq = threshold * threshold;
        let mut best_idx = None;
        let mut best_dist = f64::MAX;

        for (i, c) in self.contacts.iter().enumerate() {
            let d = (c.world_point() - new.world_point()).length_sq();
            if d < best_dist && d < threshold_sq {
                best_dist = d;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    /// Find which contact to replace to maximize the area of the remaining patch.
    fn find_least_significant(&self, new_contact: &ContactPoint) -> usize {
        let mut max_area = 0.0;
        let mut best_replace = 0;
        for skip in 0..self.contacts.len() {
            let mut pts: Vec<Vec3> = Vec::with_capacity(MAX_CONTACTS);
            for (i, c) in self.contacts.iter().enumerate() {
                if i != skip {
                    pts.push(c.world_point());
                }
            }
            pts.push(new_contact.world_point());
            let area = contact_patch_area(&pts);
            if area > max_area {
                max_area = area;
                best_replace = skip;
            }
        }
        best_replace
    }

    /// Remove contacts older than the given age threshold.
    pub fn remove_stale(&mut self, max_age: u32) {
        self.contacts.retain(|c| c.age <= max_age);
    }

    /// Age all contacts by one frame.
    pub fn age_contacts(&mut self) {
        for c in &mut self.contacts {
            c.age += 1;
        }
        self.stale_count += 1;
    }

    /// Mark all contacts as fresh (age=0) and reset stale counter.
    pub fn refresh(&mut self) {
        for c in &mut self.contacts {
            c.age = 0;
        }
        self.stale_count = 0;
    }

    /// Clear all contacts.
    pub fn clear(&mut self) {
        self.contacts.clear();
    }

    /// Get the deepest contact.
    pub fn deepest_contact(&self) -> Option<&ContactPoint> {
        self.contacts.iter().max_by(|a, b|
            a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal)
        )
    }

    /// Total accumulated normal impulse.
    pub fn total_normal_impulse(&self) -> f64 {
        self.contacts.iter().map(|c| c.normal_impulse).sum()
    }

    /// Average contact normal.
    pub fn average_normal(&self) -> Vec3 {
        if self.contacts.is_empty() { return Vec3::ZERO; }
        let mut sum = Vec3::ZERO;
        for c in &self.contacts {
            sum = sum + c.normal;
        }
        sum * (1.0 / self.contacts.len() as f64)
    }
}

/// Compute the area of a contact patch (convex hull of projected points).
fn contact_patch_area(points: &[Vec3]) -> f64 {
    if points.len() < 2 { return 0.0; }
    if points.len() == 2 {
        return (points[1] - points[0]).length();
    }
    // Approximate area using cross products of edges from first point
    let mut area = 0.0;
    let p0 = points[0];
    for i in 1..points.len() - 1 {
        let e1 = points[i] - p0;
        let e2 = points[i + 1] - p0;
        area += e1.cross(e2).length() * 0.5;
    }
    area
}

// ── Manifold Collection ──────────────────────────────────────

/// Manages contact manifolds for all body pairs.
pub struct ManifoldTable {
    manifolds: Vec<ContactManifold>,
    max_stale_frames: u32,
}

impl ManifoldTable {
    pub fn new(max_stale_frames: u32) -> Self {
        Self { manifolds: Vec::new(), max_stale_frames }
    }

    /// Get or create a manifold for the given body pair.
    pub fn get_or_create(&mut self, body_a: u64, body_b: u64) -> &mut ContactManifold {
        let (a, b) = if body_a <= body_b { (body_a, body_b) } else { (body_b, body_a) };

        if let Some(pos) = self.manifolds.iter().position(|m| m.pair_id() == (a, b)) {
            return &mut self.manifolds[pos];
        }

        self.manifolds.push(ContactManifold::new(a, b));
        self.manifolds.last_mut().unwrap()
    }

    /// Find manifold for a pair (if it exists).
    pub fn find(&self, body_a: u64, body_b: u64) -> Option<&ContactManifold> {
        let (a, b) = if body_a <= body_b { (body_a, body_b) } else { (body_b, body_a) };
        self.manifolds.iter().find(|m| m.pair_id() == (a, b))
    }

    /// Find manifold for a pair (mutable, if it exists).
    pub fn find_mut(&mut self, body_a: u64, body_b: u64) -> Option<&mut ContactManifold> {
        let (a, b) = if body_a <= body_b { (body_a, body_b) } else { (body_b, body_a) };
        self.manifolds.iter_mut().find(|m| m.pair_id() == (a, b))
    }

    /// Age all manifolds, remove stale contacts and empty manifolds.
    pub fn update(&mut self) {
        for m in &mut self.manifolds {
            m.age_contacts();
            m.remove_stale(self.max_stale_frames);
        }
        self.manifolds.retain(|m| !m.is_empty());
    }

    /// Remove manifold for a body pair.
    pub fn remove(&mut self, body_a: u64, body_b: u64) {
        let (a, b) = if body_a <= body_b { (body_a, body_b) } else { (body_b, body_a) };
        self.manifolds.retain(|m| m.pair_id() != (a, b));
    }

    /// Remove all manifolds involving a specific body.
    pub fn remove_body(&mut self, body: u64) {
        self.manifolds.retain(|m| m.body_a != body && m.body_b != body);
    }

    /// Number of active manifolds.
    pub fn manifold_count(&self) -> usize {
        self.manifolds.len()
    }

    /// Total contact count across all manifolds.
    pub fn total_contacts(&self) -> usize {
        self.manifolds.iter().map(|m| m.contact_count()).sum()
    }

    /// Iterate over all manifolds.
    pub fn iter(&self) -> impl Iterator<Item = &ContactManifold> {
        self.manifolds.iter()
    }

    /// Clear all manifolds.
    pub fn clear(&mut self) {
        self.manifolds.clear();
    }
}

// ── Contact Reduction ────────────────────────────────────────

/// Reduce a set of contact points to at most MAX_CONTACTS, keeping the most
/// representative points that maximize the contact patch area.
pub fn reduce_contacts(contacts: &[ContactPoint]) -> Vec<ContactPoint> {
    if contacts.len() <= MAX_CONTACTS {
        return contacts.to_vec();
    }

    let mut result: Vec<ContactPoint> = Vec::with_capacity(MAX_CONTACTS);

    // 1. Keep the deepest contact
    let deepest_idx = contacts.iter().enumerate()
        .max_by(|a, b| a.1.depth.partial_cmp(&b.1.depth).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    result.push(contacts[deepest_idx]);

    // 2. Keep the farthest point from the first
    let farthest_idx = contacts.iter().enumerate()
        .max_by(|a, b| {
            let da = (a.1.world_point() - result[0].world_point()).length_sq();
            let db = (b.1.world_point() - result[0].world_point()).length_sq();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    if farthest_idx != deepest_idx {
        result.push(contacts[farthest_idx]);
    }

    // 3. Keep the point that maximizes area with existing selection
    while result.len() < MAX_CONTACTS && result.len() < contacts.len() {
        let mut best_idx = None;
        let mut best_area = -1.0;

        for (i, c) in contacts.iter().enumerate() {
            if result.iter().any(|r| {
                (r.world_point() - c.world_point()).length_sq() < 1e-10
            }) {
                continue;
            }
            let mut pts: Vec<Vec3> = result.iter().map(|r| r.world_point()).collect();
            pts.push(c.world_point());
            let area = contact_patch_area(&pts);
            if area > best_area {
                best_area = area;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) => result.push(contacts[idx]),
            None => break,
        }
    }

    result
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn make_contact(x: f64, y: f64, depth: f64) -> ContactPoint {
        ContactPoint::new(
            Vec3::new(x, y, 0.0),
            Vec3::new(x, y, -depth),
            Vec3::new(0.0, 0.0, 1.0),
            depth,
        )
    }

    #[test]
    fn test_contact_point_creation() {
        let cp = ContactPoint::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, -0.1),
            Vec3::new(0.0, 0.0, 1.0),
            0.1,
        );
        assert!(approx(cp.depth, 0.1));
        assert!(approx(cp.normal_impulse, 0.0));
        assert_eq!(cp.age, 0);
    }

    #[test]
    fn test_contact_world_point() {
        let cp = ContactPoint::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.1,
        );
        let wp = cp.world_point();
        assert!(approx(wp.x, 2.0));
    }

    #[test]
    fn test_feature_id() {
        let fid = FeatureId { body_a_feature: 1, body_b_feature: 2 };
        let cp = ContactPoint::new(
            Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.05,
        ).with_feature_id(fid);
        assert_eq!(cp.feature_id, Some(fid));
    }

    #[test]
    fn test_manifold_add_single() {
        let mut m = ContactManifold::new(1, 2);
        m.add_contact(make_contact(0.0, 0.0, 0.1));
        assert_eq!(m.contact_count(), 1);
    }

    #[test]
    fn test_manifold_add_up_to_max() {
        let mut m = ContactManifold::new(1, 2);
        for i in 0..MAX_CONTACTS {
            m.add_contact(make_contact(i as f64, 0.0, 0.1));
        }
        assert_eq!(m.contact_count(), MAX_CONTACTS);
    }

    #[test]
    fn test_manifold_overflow_replaces() {
        let mut m = ContactManifold::new(1, 2);
        for i in 0..MAX_CONTACTS {
            m.add_contact(make_contact(i as f64, 0.0, 0.1));
        }
        m.add_contact(make_contact(10.0, 10.0, 0.2));
        assert_eq!(m.contact_count(), MAX_CONTACTS);
    }

    #[test]
    fn test_manifold_feature_id_matching() {
        let mut m = ContactManifold::new(1, 2);
        let fid = FeatureId { body_a_feature: 5, body_b_feature: 7 };
        let mut c1 = make_contact(0.0, 0.0, 0.1);
        c1.feature_id = Some(fid);
        c1.normal_impulse = 5.0;
        m.add_contact(c1);

        // Add same feature with different position — should update and preserve impulse
        let mut c2 = make_contact(0.01, 0.0, 0.15);
        c2.feature_id = Some(fid);
        m.add_contact(c2);

        assert_eq!(m.contact_count(), 1);
        assert!(approx(m.contacts[0].normal_impulse, 5.0));
        assert!(approx(m.contacts[0].depth, 0.15));
    }

    #[test]
    fn test_manifold_closest_point_matching() {
        let mut m = ContactManifold::new(1, 2);
        let mut c1 = make_contact(1.0, 0.0, 0.1);
        c1.normal_impulse = 3.0;
        m.add_contact(c1);

        // Very close point should match and preserve impulse
        let c2 = make_contact(1.005, 0.0, 0.12);
        m.add_contact(c2);

        assert_eq!(m.contact_count(), 1);
        assert!(approx(m.contacts[0].normal_impulse, 3.0));
    }

    #[test]
    fn test_manifold_age_and_stale_removal() {
        let mut m = ContactManifold::new(1, 2);
        m.add_contact(make_contact(0.0, 0.0, 0.1));
        m.add_contact(make_contact(1.0, 0.0, 0.1));

        for _ in 0..5 {
            m.age_contacts();
        }
        assert_eq!(m.contacts[0].age, 5);
        m.remove_stale(3);
        assert!(m.is_empty());
    }

    #[test]
    fn test_manifold_refresh() {
        let mut m = ContactManifold::new(1, 2);
        m.add_contact(make_contact(0.0, 0.0, 0.1));
        m.age_contacts();
        m.age_contacts();
        assert_eq!(m.contacts[0].age, 2);
        m.refresh();
        assert_eq!(m.contacts[0].age, 0);
    }

    #[test]
    fn test_deepest_contact() {
        let mut m = ContactManifold::new(1, 2);
        m.add_contact(make_contact(0.0, 0.0, 0.1));
        m.add_contact(make_contact(1.0, 0.0, 0.5));
        m.add_contact(make_contact(2.0, 0.0, 0.3));
        let d = m.deepest_contact().unwrap();
        assert!(approx(d.depth, 0.5));
    }

    #[test]
    fn test_total_normal_impulse() {
        let mut m = ContactManifold::new(1, 2);
        let mut c1 = make_contact(0.0, 0.0, 0.1);
        c1.normal_impulse = 2.0;
        let mut c2 = make_contact(1.0, 0.0, 0.1);
        c2.normal_impulse = 3.0;
        m.add_contact(c1);
        m.add_contact(c2);
        assert!(approx(m.total_normal_impulse(), 5.0));
    }

    #[test]
    fn test_average_normal() {
        let mut m = ContactManifold::new(1, 2);
        m.add_contact(ContactPoint::new(
            Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.1,
        ));
        m.add_contact(ContactPoint::new(
            Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0), 0.1,
        ));
        let n = m.average_normal();
        assert!(approx(n.y, 1.0));
    }

    #[test]
    fn test_manifold_pair_id_canonical() {
        let m1 = ContactManifold::new(5, 3);
        let m2 = ContactManifold::new(3, 5);
        assert_eq!(m1.pair_id(), m2.pair_id());
    }

    #[test]
    fn test_manifold_table_get_or_create() {
        let mut table = ManifoldTable::new(10);
        let m = table.get_or_create(1, 2);
        m.add_contact(make_contact(0.0, 0.0, 0.1));
        assert_eq!(table.manifold_count(), 1);

        // Same pair should return existing
        let m2 = table.get_or_create(2, 1); // reversed
        assert_eq!(m2.contact_count(), 1);
        assert_eq!(table.manifold_count(), 1);
    }

    #[test]
    fn test_manifold_table_remove() {
        let mut table = ManifoldTable::new(10);
        table.get_or_create(1, 2).add_contact(make_contact(0.0, 0.0, 0.1));
        table.get_or_create(3, 4).add_contact(make_contact(0.0, 0.0, 0.1));
        table.remove(1, 2);
        assert_eq!(table.manifold_count(), 1);
    }

    #[test]
    fn test_manifold_table_remove_body() {
        let mut table = ManifoldTable::new(10);
        table.get_or_create(1, 2).add_contact(make_contact(0.0, 0.0, 0.1));
        table.get_or_create(1, 3).add_contact(make_contact(0.0, 0.0, 0.1));
        table.get_or_create(4, 5).add_contact(make_contact(0.0, 0.0, 0.1));
        table.remove_body(1);
        assert_eq!(table.manifold_count(), 1);
    }

    #[test]
    fn test_manifold_table_update_removes_stale() {
        let mut table = ManifoldTable::new(2);
        table.get_or_create(1, 2).add_contact(make_contact(0.0, 0.0, 0.1));
        table.update();
        table.update();
        table.update(); // age is now 3, stale limit is 2
        assert_eq!(table.manifold_count(), 0);
    }

    #[test]
    fn test_manifold_table_total_contacts() {
        let mut table = ManifoldTable::new(10);
        table.get_or_create(1, 2).add_contact(make_contact(0.0, 0.0, 0.1));
        table.get_or_create(1, 2).add_contact(make_contact(1.0, 0.0, 0.1));
        table.get_or_create(3, 4).add_contact(make_contact(0.0, 0.0, 0.1));
        assert_eq!(table.total_contacts(), 3);
    }

    #[test]
    fn test_reduce_contacts_under_limit() {
        let contacts = vec![make_contact(0.0, 0.0, 0.1), make_contact(1.0, 0.0, 0.2)];
        let reduced = reduce_contacts(&contacts);
        assert_eq!(reduced.len(), 2);
    }

    #[test]
    fn test_reduce_contacts_over_limit() {
        let contacts: Vec<ContactPoint> = (0..8)
            .map(|i| {
                let angle = (i as f64) * std::f64::consts::PI * 2.0 / 8.0;
                make_contact(angle.cos(), angle.sin(), 0.1 + i as f64 * 0.01)
            })
            .collect();
        let reduced = reduce_contacts(&contacts);
        assert_eq!(reduced.len(), MAX_CONTACTS);
    }

    #[test]
    fn test_reduce_contacts_keeps_deepest() {
        let mut contacts: Vec<ContactPoint> = (0..6)
            .map(|i| make_contact(i as f64, 0.0, 0.1))
            .collect();
        contacts[3].depth = 5.0; // make one much deeper
        let reduced = reduce_contacts(&contacts);
        assert!(reduced.iter().any(|c| approx(c.depth, 5.0)));
    }

    #[test]
    fn test_contact_patch_area_triangle() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let area = contact_patch_area(&pts);
        assert!(approx(area, 2.0));
    }

    #[test]
    fn test_manifold_clear() {
        let mut m = ContactManifold::new(1, 2);
        m.add_contact(make_contact(0.0, 0.0, 0.1));
        m.add_contact(make_contact(1.0, 0.0, 0.1));
        m.clear();
        assert!(m.is_empty());
    }
}
