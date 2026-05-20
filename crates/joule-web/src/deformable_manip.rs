//! # Deformable Object Manipulation
//!
//! Algorithms for robotic manipulation of deformable objects such as ropes,
//! cables, and cloth. Includes deformation models, shape servoing,
//! grasp planning for flexible materials, and folding strategies.

use std::fmt;

// ── Core Types ──

/// 2D point for deformable object representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl fmt::Display for Point2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3})", self.x, self.y)
    }
}

// ── Mass-Spring Deformation Model ──

/// Node in a mass-spring deformation model.
#[derive(Clone, Debug)]
pub struct MassNode {
    pub position: Point2,
    pub velocity: Point2,
    pub mass: f64,
    pub pinned: bool,
}

impl MassNode {
    pub fn new(x: f64, y: f64, mass: f64) -> Self {
        Self {
            position: Point2::new(x, y),
            velocity: Point2::new(0.0, 0.0),
            mass,
            pinned: false,
        }
    }

    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }
}

/// Spring connecting two mass nodes.
#[derive(Clone, Debug)]
pub struct Spring {
    pub node_a: usize,
    pub node_b: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
}

impl Spring {
    pub fn new(a: usize, b: usize, rest_length: f64, stiffness: f64) -> Self {
        Self { node_a: a, node_b: b, rest_length, stiffness, damping: 0.1 }
    }

    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping = d;
        self
    }
}

/// Mass-spring system for deformable object simulation.
#[derive(Clone, Debug)]
pub struct MassSpringSystem {
    pub nodes: Vec<MassNode>,
    pub springs: Vec<Spring>,
    gravity: f64,
}

impl MassSpringSystem {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), springs: Vec::new(), gravity: -9.81 }
    }

    pub fn with_gravity(mut self, g: f64) -> Self {
        self.gravity = g;
        self
    }

    pub fn add_node(&mut self, node: MassNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub fn add_spring(&mut self, spring: Spring) {
        self.springs.push(spring);
    }

    /// Create a rope (chain of nodes connected by springs).
    pub fn create_rope(start: Point2, end: Point2, num_segments: usize, stiffness: f64, mass_per_node: f64) -> Self {
        let mut sys = Self::new();
        let n = num_segments + 1;
        for i in 0..n {
            let t = i as f64 / num_segments as f64;
            let p = start.lerp(&end, t);
            let mut node = MassNode::new(p.x, p.y, mass_per_node);
            if i == 0 { node.pinned = true; }
            sys.add_node(node);
        }
        for i in 0..num_segments {
            let rest = start.distance(&end) / num_segments as f64;
            sys.add_spring(Spring::new(i, i + 1, rest, stiffness));
        }
        sys
    }

    /// Create a cloth patch (grid of nodes).
    pub fn create_cloth(origin: Point2, width: f64, height: f64, cols: usize, rows: usize, stiffness: f64) -> Self {
        let mut sys = Self::new();
        let dx = width / (cols - 1).max(1) as f64;
        let dy = height / (rows - 1).max(1) as f64;

        for r in 0..rows {
            for c in 0..cols {
                let x = origin.x + c as f64 * dx;
                let y = origin.y + r as f64 * dy;
                let mut node = MassNode::new(x, y, 0.1);
                if r == 0 && (c == 0 || c == cols - 1) { node.pinned = true; }
                sys.add_node(node);
            }
        }

        // Structural springs (horizontal + vertical)
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if c + 1 < cols {
                    sys.add_spring(Spring::new(idx, idx + 1, dx, stiffness));
                }
                if r + 1 < rows {
                    sys.add_spring(Spring::new(idx, idx + cols, dy, stiffness));
                }
            }
        }
        sys
    }

    /// Step the simulation forward by dt using symplectic Euler integration.
    pub fn step(&mut self, dt: f64) {
        let n = self.nodes.len();
        let mut forces = vec![(0.0_f64, 0.0_f64); n];

        // Gravity
        for (i, node) in self.nodes.iter().enumerate() {
            if !node.pinned {
                forces[i].1 += node.mass * self.gravity;
            }
        }

        // Spring forces
        for spring in &self.springs {
            let a = &self.nodes[spring.node_a];
            let b = &self.nodes[spring.node_b];
            let dx = b.position.x - a.position.x;
            let dy = b.position.y - a.position.y;
            let dist = (dx * dx + dy * dy).sqrt().max(1e-10);
            let stretch = dist - spring.rest_length;
            let force_mag = spring.stiffness * stretch;

            let fx = force_mag * dx / dist;
            let fy = force_mag * dy / dist;

            // Damping
            let dvx = b.velocity.x - a.velocity.x;
            let dvy = b.velocity.y - a.velocity.y;
            let damp_x = spring.damping * dvx;
            let damp_y = spring.damping * dvy;

            forces[spring.node_a].0 += fx + damp_x;
            forces[spring.node_a].1 += fy + damp_y;
            forces[spring.node_b].0 -= fx + damp_x;
            forces[spring.node_b].1 -= fy + damp_y;
        }

        // Integrate
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if node.pinned { continue; }
            let ax = forces[i].0 / node.mass;
            let ay = forces[i].1 / node.mass;
            node.velocity.x += ax * dt;
            node.velocity.y += ay * dt;
            node.position.x += node.velocity.x * dt;
            node.position.y += node.velocity.y * dt;
        }
    }

    /// Total elastic energy in the system.
    pub fn elastic_energy(&self) -> f64 {
        self.springs.iter().map(|s| {
            let a = &self.nodes[s.node_a];
            let b = &self.nodes[s.node_b];
            let dist = a.position.distance(&b.position);
            let stretch = dist - s.rest_length;
            0.5 * s.stiffness * stretch * stretch
        }).sum()
    }

    /// Total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        self.nodes.iter()
            .filter(|n| !n.pinned)
            .map(|n| 0.5 * n.mass * (n.velocity.x * n.velocity.x + n.velocity.y * n.velocity.y))
            .sum()
    }
}

impl fmt::Display for MassSpringSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MassSpringSystem({} nodes, {} springs)", self.nodes.len(), self.springs.len())
    }
}

// ── Shape Servoing ──

/// Shape descriptor for a deformable object (ordered point set).
#[derive(Clone, Debug)]
pub struct ShapeDescriptor {
    pub points: Vec<Point2>,
}

impl ShapeDescriptor {
    pub fn new(points: Vec<Point2>) -> Self {
        Self { points }
    }

    pub fn from_system(sys: &MassSpringSystem) -> Self {
        Self { points: sys.nodes.iter().map(|n| n.position).collect() }
    }

    /// Shape error: sum of squared distances between corresponding points.
    pub fn error(&self, target: &Self) -> f64 {
        self.points.iter().zip(target.points.iter())
            .map(|(a, b)| {
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                dx * dx + dy * dy
            })
            .sum()
    }

    /// Centroid of the shape.
    pub fn centroid(&self) -> Point2 {
        if self.points.is_empty() {
            return Point2::new(0.0, 0.0);
        }
        let n = self.points.len() as f64;
        let sx: f64 = self.points.iter().map(|p| p.x).sum();
        let sy: f64 = self.points.iter().map(|p| p.y).sum();
        Point2::new(sx / n, sy / n)
    }

    /// Bounding box dimensions.
    pub fn bounding_box(&self) -> (Point2, Point2) {
        if self.points.is_empty() {
            return (Point2::new(0.0, 0.0), Point2::new(0.0, 0.0));
        }
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for p in &self.points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Total arc length of the shape (sum of segment lengths).
    pub fn arc_length(&self) -> f64 {
        if self.points.len() < 2 { return 0.0; }
        self.points.windows(2).map(|w| w[0].distance(&w[1])).sum()
    }
}

impl fmt::Display for ShapeDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({} points, len={:.3})", self.points.len(), self.arc_length())
    }
}

// ── Fold Planner ──

/// A fold action for cloth manipulation.
#[derive(Clone, Debug)]
pub struct FoldAction {
    pub grasp_point: Point2,
    pub place_point: Point2,
    pub fold_line: (Point2, Point2),
}

impl FoldAction {
    pub fn new(grasp: Point2, place: Point2, line_start: Point2, line_end: Point2) -> Self {
        Self {
            grasp_point: grasp,
            place_point: place,
            fold_line: (line_start, line_end),
        }
    }

    pub fn fold_distance(&self) -> f64 {
        self.grasp_point.distance(&self.place_point)
    }
}

impl fmt::Display for FoldAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fold(grasp={}, place={}, dist={:.3})",
            self.grasp_point, self.place_point, self.fold_distance())
    }
}

/// Plans fold sequences for rectangular cloth.
#[derive(Clone, Debug)]
pub struct FoldPlanner {
    width: f64,
    height: f64,
}

impl FoldPlanner {
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    /// Plan a simple half-fold along the longer axis.
    pub fn plan_half_fold(&self) -> FoldAction {
        if self.width >= self.height {
            // Fold right half onto left
            FoldAction::new(
                Point2::new(self.width, self.height / 2.0),
                Point2::new(0.0, self.height / 2.0),
                Point2::new(self.width / 2.0, 0.0),
                Point2::new(self.width / 2.0, self.height),
            )
        } else {
            // Fold top onto bottom
            FoldAction::new(
                Point2::new(self.width / 2.0, self.height),
                Point2::new(self.width / 2.0, 0.0),
                Point2::new(0.0, self.height / 2.0),
                Point2::new(self.width, self.height / 2.0),
            )
        }
    }

    /// Plan a sequence of folds to reach a target number of layers.
    pub fn plan_multi_fold(&self, num_folds: usize) -> Vec<FoldAction> {
        let mut folds = Vec::with_capacity(num_folds);
        let mut w = self.width;
        let mut h = self.height;

        for _ in 0..num_folds {
            let planner = FoldPlanner::new(w, h);
            folds.push(planner.plan_half_fold());
            if w >= h { w /= 2.0; } else { h /= 2.0; }
        }
        folds
    }

    /// Resulting dimensions after n folds.
    pub fn folded_dimensions(&self, num_folds: usize) -> (f64, f64) {
        let mut w = self.width;
        let mut h = self.height;
        for _ in 0..num_folds {
            if w >= h { w /= 2.0; } else { h /= 2.0; }
        }
        (w, h)
    }
}

impl fmt::Display for FoldPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FoldPlanner({}x{})", self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2_distance() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2_lerp() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(10.0, 10.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mass_node_pinned() {
        let node = MassNode::new(1.0, 2.0, 0.5).pinned();
        assert!(node.pinned);
    }

    #[test]
    fn test_create_rope() {
        let rope = MassSpringSystem::create_rope(
            Point2::new(0.0, 0.0), Point2::new(5.0, 0.0), 10, 100.0, 0.1
        );
        assert_eq!(rope.nodes.len(), 11);
        assert_eq!(rope.springs.len(), 10);
        assert!(rope.nodes[0].pinned);
    }

    #[test]
    fn test_create_cloth() {
        let cloth = MassSpringSystem::create_cloth(
            Point2::new(0.0, 0.0), 1.0, 1.0, 4, 4, 200.0
        );
        assert_eq!(cloth.nodes.len(), 16);
        assert!(cloth.springs.len() > 0);
    }

    #[test]
    fn test_step_simulation() {
        let mut rope = MassSpringSystem::create_rope(
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0), 5, 100.0, 0.1
        );
        let e0 = rope.elastic_energy();
        for _ in 0..10 {
            rope.step(0.001);
        }
        // Energy should change due to gravity
        let e1 = rope.elastic_energy();
        assert!(e0 != e1 || rope.kinetic_energy() > 0.0);
    }

    #[test]
    fn test_elastic_energy_at_rest() {
        let rope = MassSpringSystem::create_rope(
            Point2::new(0.0, 0.0), Point2::new(5.0, 0.0), 5, 100.0, 0.1
        );
        let energy = rope.elastic_energy();
        assert!(energy < 1e-6); // Springs at rest length
    }

    #[test]
    fn test_kinetic_energy_initial() {
        let rope = MassSpringSystem::create_rope(
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0), 3, 50.0, 0.1
        );
        assert!((rope.kinetic_energy() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mass_spring_display() {
        let sys = MassSpringSystem::create_rope(
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0), 3, 50.0, 0.1
        );
        let s = format!("{sys}");
        assert!(s.contains("4 nodes"));
    }

    #[test]
    fn test_shape_descriptor_error() {
        let a = ShapeDescriptor::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)]);
        let b = ShapeDescriptor::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)]);
        assert!((a.error(&b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_descriptor_centroid() {
        let shape = ShapeDescriptor::new(vec![
            Point2::new(0.0, 0.0), Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0), Point2::new(0.0, 2.0),
        ]);
        let c = shape.centroid();
        assert!((c.x - 1.0).abs() < 1e-10);
        assert!((c.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_arc_length() {
        let shape = ShapeDescriptor::new(vec![
            Point2::new(0.0, 0.0), Point2::new(3.0, 0.0), Point2::new(3.0, 4.0),
        ]);
        assert!((shape.arc_length() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_bounding_box() {
        let shape = ShapeDescriptor::new(vec![
            Point2::new(-1.0, -2.0), Point2::new(3.0, 4.0),
        ]);
        let (lo, hi) = shape.bounding_box();
        assert!((lo.x - (-1.0)).abs() < 1e-10);
        assert!((hi.y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fold_action() {
        let fold = FoldAction::new(
            Point2::new(10.0, 5.0), Point2::new(0.0, 5.0),
            Point2::new(5.0, 0.0), Point2::new(5.0, 10.0),
        );
        assert!((fold.fold_distance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fold_planner_half() {
        let planner = FoldPlanner::new(2.0, 1.0);
        let fold = planner.plan_half_fold();
        assert!(fold.fold_distance() > 0.0);
    }

    #[test]
    fn test_fold_planner_multi() {
        let planner = FoldPlanner::new(4.0, 4.0);
        let folds = planner.plan_multi_fold(3);
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn test_folded_dimensions() {
        let planner = FoldPlanner::new(4.0, 2.0);
        let (w, h) = planner.folded_dimensions(2);
        assert!((w - 1.0).abs() < 1e-10);
        assert!((h - 2.0).abs() < 1e-10 || (h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_from_system() {
        let rope = MassSpringSystem::create_rope(
            Point2::new(0.0, 0.0), Point2::new(1.0, 0.0), 3, 50.0, 0.1
        );
        let shape = ShapeDescriptor::from_system(&rope);
        assert_eq!(shape.points.len(), 4);
    }
}
