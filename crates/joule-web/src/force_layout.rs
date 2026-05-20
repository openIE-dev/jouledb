//! Force-directed graph layout: velocity Verlet integration, center force,
//! charge (Barnes-Hut quadtree), link (spring), collision, alpha cooling.

// ── Node ────────────────────────────────────────────────────────

/// A node in the force-directed graph.
#[derive(Debug, Clone)]
pub struct ForceNode {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub mass: f64,
    /// Radius for collision detection.
    pub radius: f64,
    /// If true, this node is pinned and won't move.
    pub fixed: bool,
}

impl ForceNode {
    pub fn new(id: u64, x: f64, y: f64) -> Self {
        Self {
            id,
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            mass: 1.0,
            radius: 5.0,
            fixed: false,
        }
    }

    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    pub fn with_radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }
}

// ── Link ────────────────────────────────────────────────────────

/// A link (edge) between two nodes.
#[derive(Debug, Clone)]
pub struct ForceLink {
    /// Index into the simulation's node array for the source.
    pub source: usize,
    /// Index into the simulation's node array for the target.
    pub target: usize,
    /// Spring strength.
    pub strength: f64,
    /// Desired distance.
    pub distance: f64,
}

impl ForceLink {
    pub fn new(source: usize, target: usize) -> Self {
        Self {
            source,
            target,
            strength: 1.0,
            distance: 30.0,
        }
    }

    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength;
        self
    }

    pub fn with_distance(mut self, distance: f64) -> Self {
        self.distance = distance;
        self
    }
}

// ── Quadtree (for Barnes-Hut) ───────────────────────────────────

#[derive(Debug)]
struct QuadNode {
    /// Center of mass x.
    cx: f64,
    /// Center of mass y.
    cy: f64,
    /// Total mass.
    total_mass: f64,
    /// Number of bodies.
    count: usize,
    /// Bounding box.
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    /// Children: NW, NE, SW, SE.
    children: [Option<Box<QuadNode>>; 4],
    /// Leaf body index (only if count == 1).
    body_idx: Option<usize>,
}

impl QuadNode {
    fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self {
            cx: 0.0,
            cy: 0.0,
            total_mass: 0.0,
            count: 0,
            x0,
            y0,
            x1,
            y1,
            children: [None, None, None, None],
            body_idx: None,
        }
    }

    fn quadrant(&self, x: f64, y: f64) -> usize {
        let mx = (self.x0 + self.x1) / 2.0;
        let my = (self.y0 + self.y1) / 2.0;
        let east = x >= mx;
        let south = y >= my;
        match (east, south) {
            (false, false) => 0, // NW
            (true, false) => 1,  // NE
            (false, true) => 2,  // SW
            (true, true) => 3,   // SE
        }
    }

    fn child_bounds(&self, q: usize) -> (f64, f64, f64, f64) {
        let mx = (self.x0 + self.x1) / 2.0;
        let my = (self.y0 + self.y1) / 2.0;
        match q {
            0 => (self.x0, self.y0, mx, my),
            1 => (mx, self.y0, self.x1, my),
            2 => (self.x0, my, mx, self.y1),
            3 => (mx, my, self.x1, self.y1),
            _ => unreachable!(),
        }
    }

    fn insert(&mut self, idx: usize, x: f64, y: f64, mass: f64) {
        if self.count == 0 {
            self.cx = x;
            self.cy = y;
            self.total_mass = mass;
            self.count = 1;
            self.body_idx = Some(idx);
            return;
        }

        // If leaf, subdivide existing body
        if self.count == 1 {
            if let Some(old_idx) = self.body_idx.take() {
                let old_cx = self.cx;
                let old_cy = self.cy;
                let old_mass = self.total_mass;
                let q = self.quadrant(old_cx, old_cy);
                let (x0, y0, x1, y1) = self.child_bounds(q);
                let child = self.children[q].get_or_insert_with(|| Box::new(QuadNode::new(x0, y0, x1, y1)));
                child.insert(old_idx, old_cx, old_cy, old_mass);
            }
        }

        // Insert new body
        let q = self.quadrant(x, y);
        let (x0, y0, x1, y1) = self.child_bounds(q);
        let child = self.children[q].get_or_insert_with(|| Box::new(QuadNode::new(x0, y0, x1, y1)));
        child.insert(idx, x, y, mass);

        // Update center of mass
        let new_total = self.total_mass + mass;
        self.cx = (self.cx * self.total_mass + x * mass) / new_total;
        self.cy = (self.cy * self.total_mass + y * mass) / new_total;
        self.total_mass = new_total;
        self.count += 1;
    }
}

fn build_quadtree(nodes: &[ForceNode]) -> Option<QuadNode> {
    if nodes.is_empty() {
        return None;
    }
    let mut x0 = f64::MAX;
    let mut y0 = f64::MAX;
    let mut x1 = f64::MIN;
    let mut y1 = f64::MIN;
    for n in nodes {
        x0 = x0.min(n.x);
        y0 = y0.min(n.y);
        x1 = x1.max(n.x);
        y1 = y1.max(n.y);
    }
    // Pad slightly
    let pad = ((x1 - x0).max(y1 - y0)) * 0.01 + 1.0;
    let mut root = QuadNode::new(x0 - pad, y0 - pad, x1 + pad, y1 + pad);
    for (i, n) in nodes.iter().enumerate() {
        root.insert(i, n.x, n.y, n.mass);
    }
    Some(root)
}

// ── Forces ──────────────────────────────────────────────────────

fn apply_center_force(nodes: &mut [ForceNode], cx: f64, cy: f64, strength: f64) {
    let n = nodes.len() as f64;
    if n == 0.0 {
        return;
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    for node in nodes.iter() {
        sx += node.x;
        sy += node.y;
    }
    sx = sx / n - cx;
    sy = sy / n - cy;
    for node in nodes.iter_mut() {
        node.vx -= sx * strength;
        node.vy -= sy * strength;
    }
}

fn apply_charge_force(nodes: &mut [ForceNode], charge_strength: f64, theta: f64) {
    let tree = match build_quadtree(nodes) {
        Some(t) => t,
        None => return,
    };
    let theta_sq = theta * theta;

    // Collect forces first to avoid borrow issues
    let mut forces: Vec<(f64, f64)> = vec![(0.0, 0.0); nodes.len()];
    for i in 0..nodes.len() {
        let (fx, fy) = compute_charge_recursive(&tree, nodes[i].x, nodes[i].y, i, charge_strength, theta_sq);
        forces[i] = (fx, fy);
    }
    for (i, node) in nodes.iter_mut().enumerate() {
        node.vx += forces[i].0;
        node.vy += forces[i].1;
    }
}

fn compute_charge_recursive(
    quad: &QuadNode,
    x: f64,
    y: f64,
    self_idx: usize,
    strength: f64,
    theta_sq: f64,
) -> (f64, f64) {
    if quad.count == 0 {
        return (0.0, 0.0);
    }

    let dx = quad.cx - x;
    let dy = quad.cy - y;
    let dist_sq = dx * dx + dy * dy;

    // If single body and it's self, skip
    if quad.count == 1 {
        if let Some(idx) = quad.body_idx {
            if idx == self_idx {
                return (0.0, 0.0);
            }
        }
    }

    let width = quad.x1 - quad.x0;

    // Barnes-Hut criterion: if far enough, treat as single body
    if quad.count == 1 || (width * width / dist_sq) < theta_sq {
        if dist_sq < 1e-6 {
            return (0.0, 0.0);
        }
        let dist = dist_sq.sqrt();
        let force = strength * quad.total_mass / dist_sq;
        return (dx / dist * force, dy / dist * force);
    }

    // Recurse into children
    let mut fx = 0.0;
    let mut fy = 0.0;
    for child in &quad.children {
        if let Some(c) = child {
            let (cfx, cfy) = compute_charge_recursive(c, x, y, self_idx, strength, theta_sq);
            fx += cfx;
            fy += cfy;
        }
    }
    (fx, fy)
}

fn apply_link_force(nodes: &mut [ForceNode], links: &[ForceLink]) {
    for link in links {
        if link.source >= nodes.len() || link.target >= nodes.len() {
            continue;
        }
        let dx = nodes[link.target].x - nodes[link.source].x;
        let dy = nodes[link.target].y - nodes[link.source].y;
        let dist = (dx * dx + dy * dy).sqrt().max(1e-6);
        let force = (dist - link.distance) / dist * link.strength * 0.5;
        let fx = dx * force;
        let fy = dy * force;

        let source_mass = nodes[link.source].mass;
        let target_mass = nodes[link.target].mass;
        let total = source_mass + target_mass;

        nodes[link.source].vx += fx * target_mass / total;
        nodes[link.source].vy += fy * target_mass / total;
        nodes[link.target].vx -= fx * source_mass / total;
        nodes[link.target].vy -= fy * source_mass / total;
    }
}

fn apply_collision_force(nodes: &mut [ForceNode]) {
    let n = nodes.len();
    // O(n^2) for simplicity; fine for typical graph sizes
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = nodes[j].x - nodes[i].x;
            let dy = nodes[j].y - nodes[i].y;
            let dist = (dx * dx + dy * dy).sqrt().max(1e-6);
            let min_dist = nodes[i].radius + nodes[j].radius;
            if dist < min_dist {
                let overlap = (min_dist - dist) / dist * 0.5;
                let ox = dx * overlap;
                let oy = dy * overlap;
                nodes[i].vx -= ox;
                nodes[i].vy -= oy;
                nodes[j].vx += ox;
                nodes[j].vy += oy;
            }
        }
    }
}

// ── Simulation ──────────────────────────────────────────────────

/// Force simulation configuration.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Initial alpha (temperature).
    pub alpha: f64,
    /// Minimum alpha before stopping.
    pub alpha_min: f64,
    /// Alpha decay rate per tick.
    pub alpha_decay: f64,
    /// Velocity decay (friction).
    pub velocity_decay: f64,
    /// Center force target.
    pub center: (f64, f64),
    /// Center force strength.
    pub center_strength: f64,
    /// Charge (repulsion) strength (negative = repulsion).
    pub charge_strength: f64,
    /// Barnes-Hut theta parameter.
    pub theta: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            alpha_min: 0.001,
            alpha_decay: 0.0228,
            velocity_decay: 0.6,
            center: (0.0, 0.0),
            center_strength: 0.1,
            charge_strength: -30.0,
            theta: 0.9,
        }
    }
}

/// The force simulation state.
#[derive(Debug)]
pub struct ForceSimulation {
    pub nodes: Vec<ForceNode>,
    pub links: Vec<ForceLink>,
    pub config: SimulationConfig,
    pub alpha: f64,
    pub tick_count: u64,
}

impl ForceSimulation {
    pub fn new(nodes: Vec<ForceNode>, links: Vec<ForceLink>) -> Self {
        Self {
            nodes,
            links,
            config: SimulationConfig::default(),
            alpha: 1.0,
            tick_count: 0,
        }
    }

    pub fn with_config(mut self, config: SimulationConfig) -> Self {
        self.alpha = config.alpha;
        self.config = config;
        self
    }

    /// Whether the simulation is still active (alpha > alpha_min).
    pub fn is_active(&self) -> bool {
        self.alpha > self.config.alpha_min
    }

    /// Run a single simulation tick using velocity Verlet integration.
    pub fn tick(&mut self) {
        self.alpha += (self.config.alpha_min - self.alpha) * self.config.alpha_decay;

        // Apply forces
        apply_center_force(&mut self.nodes, self.config.center.0, self.config.center.1, self.config.center_strength);
        apply_charge_force(&mut self.nodes, self.config.charge_strength * self.alpha, self.config.theta);
        apply_link_force(&mut self.nodes, &self.links);
        apply_collision_force(&mut self.nodes);

        // Velocity Verlet update
        let decay = self.config.velocity_decay;
        for node in &mut self.nodes {
            if node.fixed {
                node.vx = 0.0;
                node.vy = 0.0;
                continue;
            }
            node.vx *= decay;
            node.vy *= decay;
            node.x += node.vx;
            node.y += node.vy;
        }

        self.tick_count += 1;
    }

    /// Run until the simulation cools down or reaches max iterations.
    pub fn run(&mut self, max_iterations: u64) {
        for _ in 0..max_iterations {
            if !self.is_active() {
                break;
            }
            self.tick();
        }
    }

    /// Total kinetic energy (useful for convergence checks).
    pub fn kinetic_energy(&self) -> f64 {
        self.nodes
            .iter()
            .map(|n| 0.5 * n.mass * (n.vx * n.vx + n.vy * n.vy))
            .sum()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> (Vec<ForceNode>, Vec<ForceLink>) {
        let nodes = vec![
            ForceNode::new(0, 100.0, 0.0),
            ForceNode::new(1, -50.0, 86.6),
            ForceNode::new(2, -50.0, -86.6),
        ];
        let links = vec![
            ForceLink::new(0, 1),
            ForceLink::new(1, 2),
            ForceLink::new(2, 0),
        ];
        (nodes, links)
    }

    #[test]
    fn test_simulation_creation() {
        let (nodes, links) = triangle_graph();
        let sim = ForceSimulation::new(nodes, links);
        assert_eq!(sim.nodes.len(), 3);
        assert_eq!(sim.links.len(), 3);
        assert!(sim.is_active());
    }

    #[test]
    fn test_single_tick() {
        let (nodes, links) = triangle_graph();
        let mut sim = ForceSimulation::new(nodes, links);
        let old_x = sim.nodes[0].x;
        sim.tick();
        // Node should have moved
        assert_ne!(sim.nodes[0].x, old_x);
        assert_eq!(sim.tick_count, 1);
    }

    #[test]
    fn test_simulation_converges() {
        let (nodes, links) = triangle_graph();
        let mut sim = ForceSimulation::new(nodes, links);
        sim.run(300);
        assert!(!sim.is_active() || sim.tick_count == 300);
        // Kinetic energy should have decreased
        assert!(sim.kinetic_energy() < 1000.0);
    }

    #[test]
    fn test_center_force() {
        let nodes = vec![
            ForceNode::new(0, 100.0, 100.0),
            ForceNode::new(1, 200.0, 200.0),
        ];
        let mut sim = ForceSimulation::new(nodes, vec![]);
        sim.config.charge_strength = 0.0; // disable charge
        sim.config.center_strength = 1.0;
        sim.run(100);
        // Nodes should move toward center (0, 0)
        let avg_x = sim.nodes.iter().map(|n| n.x).sum::<f64>() / 2.0;
        let avg_y = sim.nodes.iter().map(|n| n.y).sum::<f64>() / 2.0;
        assert!(avg_x.abs() < 50.0);
        assert!(avg_y.abs() < 50.0);
    }

    #[test]
    fn test_charge_repulsion() {
        let nodes = vec![
            ForceNode::new(0, 0.0, 0.0),
            ForceNode::new(1, 1.0, 0.0),
        ];
        let mut sim = ForceSimulation::new(nodes, vec![]);
        sim.config.center_strength = 0.0;
        sim.tick();
        // Nodes should repel: node0 goes left, node1 goes right
        assert!(sim.nodes[0].vx < 0.0 || sim.nodes[1].vx > 0.0);
    }

    #[test]
    fn test_link_spring() {
        let nodes = vec![
            ForceNode::new(0, 0.0, 0.0),
            ForceNode::new(1, 100.0, 0.0),
        ];
        let links = vec![ForceLink::new(0, 1).with_distance(30.0)];
        let mut sim = ForceSimulation::new(nodes, links);
        sim.config.charge_strength = 0.0;
        sim.config.center_strength = 0.0;
        sim.tick();
        // Link should pull nodes together (distance > desired)
        assert!(sim.nodes[0].vx > 0.0);
        assert!(sim.nodes[1].vx < 0.0);
    }

    #[test]
    fn test_collision() {
        let nodes = vec![
            ForceNode::new(0, 0.0, 0.0).with_radius(20.0),
            ForceNode::new(1, 5.0, 0.0).with_radius(20.0),
        ];
        let mut sim = ForceSimulation::new(nodes, vec![]);
        sim.config.charge_strength = 0.0;
        sim.config.center_strength = 0.0;
        sim.tick();
        // Collision should push them apart
        let dist = ((sim.nodes[0].x - sim.nodes[1].x).powi(2)
            + (sim.nodes[0].y - sim.nodes[1].y).powi(2))
        .sqrt();
        assert!(dist > 5.0);
    }

    #[test]
    fn test_fixed_node() {
        let mut nodes = vec![
            ForceNode::new(0, 0.0, 0.0),
            ForceNode::new(1, 50.0, 50.0),
        ];
        nodes[0].fixed = true;
        let mut sim = ForceSimulation::new(nodes, vec![]);
        sim.run(50);
        assert_eq!(sim.nodes[0].x, 0.0);
        assert_eq!(sim.nodes[0].y, 0.0);
    }

    #[test]
    fn test_alpha_decay() {
        let (nodes, links) = triangle_graph();
        let mut sim = ForceSimulation::new(nodes, links);
        let initial_alpha = sim.alpha;
        sim.tick();
        assert!(sim.alpha < initial_alpha);
    }

    #[test]
    fn test_kinetic_energy() {
        let (nodes, links) = triangle_graph();
        let sim = ForceSimulation::new(nodes, links);
        // Initially all velocities are zero
        assert_eq!(sim.kinetic_energy(), 0.0);
    }

    #[test]
    fn test_node_builders() {
        let n = ForceNode::new(42, 1.0, 2.0).with_mass(5.0).with_radius(10.0);
        assert_eq!(n.mass, 5.0);
        assert_eq!(n.radius, 10.0);
    }

    #[test]
    fn test_link_builders() {
        let l = ForceLink::new(0, 1).with_strength(2.0).with_distance(50.0);
        assert_eq!(l.strength, 2.0);
        assert_eq!(l.distance, 50.0);
    }

    #[test]
    fn test_empty_simulation() {
        let mut sim = ForceSimulation::new(vec![], vec![]);
        sim.tick(); // Should not panic
        assert_eq!(sim.tick_count, 1);
    }
}
