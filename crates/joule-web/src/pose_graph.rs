//! Pose graph optimization — Gauss-Newton solver with robust Huber kernels.
//!
//! Represents a SLAM trajectory as a graph of pose nodes connected by
//! relative-pose edge constraints. The optimizer iteratively linearizes
//! and solves a sparse normal equation to minimize the total weighted
//! residual error.

use std::fmt;

// ── 2-D pose ──────────────────────────────────────────────────────

/// 2-D pose `(x, y, θ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2D {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2D {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta: wrap_angle(theta) }
    }

    pub fn identity() -> Self { Self { x: 0.0, y: 0.0, theta: 0.0 } }

    /// Compose: self ⊕ other.
    pub fn compose(&self, other: &Pose2D) -> Pose2D {
        let (s, c) = self.theta.sin_cos();
        Pose2D::new(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            self.theta + other.theta,
        )
    }

    /// Inverse pose.
    pub fn inverse(&self) -> Pose2D {
        let (s, c) = self.theta.sin_cos();
        Pose2D::new(
            -(c * self.x + s * self.y),
            -(-s * self.x + c * self.y),
            -self.theta,
        )
    }

    /// Relative transform: self⁻¹ ⊕ other.
    pub fn relative_to(&self, other: &Pose2D) -> Pose2D {
        self.inverse().compose(other)
    }
}

impl fmt::Display for Pose2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pose2D({:.4}, {:.4}, {:.4}rad)", self.x, self.y, self.theta)
    }
}

fn wrap_angle(a: f64) -> f64 {
    let mut v = a % (2.0 * std::f64::consts::PI);
    if v > std::f64::consts::PI { v -= 2.0 * std::f64::consts::PI; }
    if v < -std::f64::consts::PI { v += 2.0 * std::f64::consts::PI; }
    v
}

// ── Edge constraint ───────────────────────────────────────────────

/// An edge in the pose graph: relative-pose measurement between two nodes.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub measurement: Pose2D,
    /// 3×3 information matrix (upper triangle, row-major: [i00, i01, i02, i11, i12, i22]).
    pub information: [f64; 6],
}

impl Edge {
    pub fn new(from: usize, to: usize, measurement: Pose2D) -> Self {
        Self { from, to, measurement, information: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0] }
    }

    pub fn with_information(mut self, info: [f64; 6]) -> Self {
        self.information = info;
        self
    }

    /// Expand upper-triangular information to full 3×3.
    pub fn info_matrix(&self) -> [[f64; 3]; 3] {
        let i = &self.information;
        [
            [i[0], i[1], i[2]],
            [i[1], i[3], i[4]],
            [i[2], i[4], i[5]],
        ]
    }

    /// Compute residual error = measurement⁻¹ ⊕ (from⁻¹ ⊕ to).
    pub fn residual(&self, poses: &[Pose2D]) -> [f64; 3] {
        let predicted = poses[self.from].relative_to(&poses[self.to]);
        let diff = self.measurement.relative_to(&predicted);
        [diff.x, diff.y, wrap_angle(diff.theta)]
    }

    /// Weighted squared error: e^T Ω e.
    pub fn weighted_error(&self, poses: &[Pose2D]) -> f64 {
        let e = self.residual(poses);
        let omega = self.info_matrix();
        let mut result = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                result += e[i] * omega[i][j] * e[j];
            }
        }
        result
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge({} -> {}, meas={})", self.from, self.to, self.measurement)
    }
}

// ── Robust kernel ─────────────────────────────────────────────────

/// Robust cost kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustKernel {
    None,
    Huber(f64),
    Cauchy(f64),
}

impl RobustKernel {
    /// Apply kernel: returns (rho, rho') where rho is the cost and rho' is the weight.
    pub fn apply(&self, sq_error: f64) -> (f64, f64) {
        match self {
            RobustKernel::None => (sq_error, 1.0),
            RobustKernel::Huber(delta) => {
                let d = *delta;
                let s = sq_error.sqrt();
                if s <= d {
                    (sq_error, 1.0)
                } else {
                    (2.0 * d * s - d * d, d / s)
                }
            }
            RobustKernel::Cauchy(c) => {
                let c2 = c * c;
                let ratio = sq_error / c2;
                (c2 * (1.0 + ratio).ln(), 1.0 / (1.0 + ratio))
            }
        }
    }
}

impl fmt::Display for RobustKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RobustKernel::None => write!(f, "None"),
            RobustKernel::Huber(d) => write!(f, "Huber({:.2})", d),
            RobustKernel::Cauchy(c) => write!(f, "Cauchy({:.2})", c),
        }
    }
}

// ── Optimizer configuration ───────────────────────────────────────

/// Pose graph optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub damping: f64,
    pub kernel: RobustKernel,
    pub fixed_node: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            tolerance: 1e-6,
            damping: 1e-3,
            kernel: RobustKernel::None,
            fixed_node: 0,
        }
    }
}

impl OptimizerConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_damping(mut self, d: f64) -> Self { self.damping = d; self }
    pub fn with_kernel(mut self, k: RobustKernel) -> Self { self.kernel = k; self }
    pub fn with_fixed_node(mut self, n: usize) -> Self { self.fixed_node = n; self }
}

// ── Pose graph ────────────────────────────────────────────────────

/// A pose graph with nodes (poses) and edges (constraints).
#[derive(Debug, Clone)]
pub struct PoseGraph {
    pub nodes: Vec<Pose2D>,
    pub edges: Vec<Edge>,
}

impl PoseGraph {
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() } }

    pub fn add_node(&mut self, pose: Pose2D) -> usize {
        let id = self.nodes.len();
        self.nodes.push(pose);
        id
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    pub fn num_nodes(&self) -> usize { self.nodes.len() }
    pub fn num_edges(&self) -> usize { self.edges.len() }

    /// Total weighted error across all edges.
    pub fn total_error(&self) -> f64 {
        self.edges.iter().map(|e| e.weighted_error(&self.nodes)).sum()
    }

    /// Optimize using Gauss-Newton with optional robust kernel.
    pub fn optimize(&mut self, config: &OptimizerConfig) -> OptimizationResult {
        let n = self.nodes.len();
        if n == 0 { return OptimizationResult { iterations: 0, initial_error: 0.0, final_error: 0.0, converged: true }; }

        let initial_error = self.total_error();
        let dim = n * 3; // 3 DOF per node

        for iter in 0..config.max_iterations {
            // Build linear system H dx = -b
            let mut h_diag = vec![[0.0f64; 9]; n]; // Block-diagonal 3×3
            let mut b_vec = vec![[0.0f64; 3]; n];

            // Off-diagonal blocks stored sparsely
            let mut h_off: Vec<(usize, usize, [f64; 9])> = Vec::new();

            for edge in &self.edges {
                let e = edge.residual(&self.nodes);
                let omega = edge.info_matrix();

                // Robust weighting
                let sq_err = edge.weighted_error(&self.nodes);
                let (_, weight) = config.kernel.apply(sq_err);

                // Numerical Jacobians (finite difference)
                let eps = 1e-6;
                let mut ji = [[0.0f64; 3]; 3]; // Jacobian w.r.t. node i
                let mut jj = [[0.0f64; 3]; 3]; // Jacobian w.r.t. node j

                for k in 0..3 {
                    let mut poses_plus = self.nodes.clone();
                    match k {
                        0 => poses_plus[edge.from].x += eps,
                        1 => poses_plus[edge.from].y += eps,
                        _ => poses_plus[edge.from].theta += eps,
                    }
                    let e_plus = edge.residual(&poses_plus);
                    for r in 0..3 { ji[r][k] = (e_plus[r] - e[r]) / eps; }

                    let mut poses_plus2 = self.nodes.clone();
                    match k {
                        0 => poses_plus2[edge.to].x += eps,
                        1 => poses_plus2[edge.to].y += eps,
                        _ => poses_plus2[edge.to].theta += eps,
                    }
                    let e_plus2 = edge.residual(&poses_plus2);
                    for r in 0..3 { jj[r][k] = (e_plus2[r] - e[r]) / eps; }
                }

                // Accumulate: H_ii += Ji^T Ω Ji, H_jj += Jj^T Ω Jj, H_ij += Ji^T Ω Jj
                // b_i += Ji^T Ω e, b_j += Jj^T Ω e
                let w_omega = |r: usize, c: usize| omega[r][c] * weight;

                for r in 0..3 {
                    for c in 0..3 {
                        let mut hii = 0.0;
                        let mut hjj = 0.0;
                        let mut hij = 0.0;
                        for k in 0..3 {
                            for l in 0..3 {
                                hii += ji[k][r] * w_omega(k, l) * ji[l][c];
                                hjj += jj[k][r] * w_omega(k, l) * jj[l][c];
                                hij += ji[k][r] * w_omega(k, l) * jj[l][c];
                            }
                        }
                        h_diag[edge.from][r * 3 + c] += hii;
                        h_diag[edge.to][r * 3 + c] += hjj;
                        h_off.push((edge.from, edge.to, {
                            let mut block = [0.0; 9];
                            block[r * 3 + c] = hij;
                            block
                        }));
                    }
                }

                for r in 0..3 {
                    let mut bi = 0.0;
                    let mut bj = 0.0;
                    for k in 0..3 {
                        bi += ji[k][r] * (w_omega(k, 0) * e[0] + w_omega(k, 1) * e[1] + w_omega(k, 2) * e[2]);
                        bj += jj[k][r] * (w_omega(k, 0) * e[0] + w_omega(k, 1) * e[1] + w_omega(k, 2) * e[2]);
                    }
                    b_vec[edge.from][r] += bi;
                    b_vec[edge.to][r] += bj;
                }
            }

            // Add damping to diagonal
            for i in 0..n {
                for k in 0..3 {
                    h_diag[i][k * 3 + k] += config.damping;
                }
            }

            // Fix node: zero out its row/col in H, zero b
            let fixed = config.fixed_node;
            if fixed < n {
                h_diag[fixed] = [0.0; 9];
                h_diag[fixed][0] = 1.0;
                h_diag[fixed][4] = 1.0;
                h_diag[fixed][8] = 1.0;
                b_vec[fixed] = [0.0; 3];
            }

            // Solve block-diagonal approximation (Jacobi preconditioner)
            // dx_i = -H_ii^{-1} b_i
            let mut dx = vec![[0.0f64; 3]; n];
            for i in 0..n {
                let blk = &h_diag[i];
                // 3×3 inverse via cofactor
                let det = blk[0] * (blk[4] * blk[8] - blk[5] * blk[7])
                    - blk[1] * (blk[3] * blk[8] - blk[5] * blk[6])
                    + blk[2] * (blk[3] * blk[7] - blk[4] * blk[6]);
                if det.abs() < 1e-15 { continue; }
                let inv_det = 1.0 / det;

                let inv = [
                    (blk[4] * blk[8] - blk[5] * blk[7]) * inv_det,
                    (blk[2] * blk[7] - blk[1] * blk[8]) * inv_det,
                    (blk[1] * blk[5] - blk[2] * blk[4]) * inv_det,
                    (blk[5] * blk[6] - blk[3] * blk[8]) * inv_det,
                    (blk[0] * blk[8] - blk[2] * blk[6]) * inv_det,
                    (blk[2] * blk[3] - blk[0] * blk[5]) * inv_det,
                    (blk[3] * blk[7] - blk[4] * blk[6]) * inv_det,
                    (blk[1] * blk[6] - blk[0] * blk[7]) * inv_det,
                    (blk[0] * blk[4] - blk[1] * blk[3]) * inv_det,
                ];

                for r in 0..3 {
                    dx[i][r] = -(inv[r * 3] * b_vec[i][0]
                        + inv[r * 3 + 1] * b_vec[i][1]
                        + inv[r * 3 + 2] * b_vec[i][2]);
                }
            }

            // Apply update
            let mut max_delta = 0.0f64;
            for i in 0..n {
                if i == fixed { continue; }
                self.nodes[i].x += dx[i][0];
                self.nodes[i].y += dx[i][1];
                self.nodes[i].theta = wrap_angle(self.nodes[i].theta + dx[i][2]);
                let delta = dx[i][0].abs() + dx[i][1].abs() + dx[i][2].abs();
                if delta > max_delta { max_delta = delta; }
            }

            if max_delta < config.tolerance {
                return OptimizationResult {
                    iterations: iter + 1,
                    initial_error,
                    final_error: self.total_error(),
                    converged: true,
                };
            }
        }

        OptimizationResult {
            iterations: config.max_iterations,
            initial_error,
            final_error: self.total_error(),
            converged: false,
        }
    }
}

impl fmt::Display for PoseGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PoseGraph(nodes={}, edges={}, error={:.6})", self.num_nodes(), self.num_edges(), self.total_error())
    }
}

// ── Optimization result ───────────────────────────────────────────

/// Result of pose-graph optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub iterations: usize,
    pub initial_error: f64,
    pub final_error: f64,
    pub converged: bool,
}

impl fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OptResult(iters={}, error: {:.6} -> {:.6}, converged={})",
            self.iterations, self.initial_error, self.final_error, self.converged
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_identity() {
        let p = Pose2D::identity();
        assert!((p.x).abs() < 1e-10);
        assert!((p.y).abs() < 1e-10);
        assert!((p.theta).abs() < 1e-10);
    }

    #[test]
    fn test_pose_compose_identity() {
        let a = Pose2D::new(1.0, 2.0, 0.5);
        let id = Pose2D::identity();
        let c = a.compose(&id);
        assert!((c.x - a.x).abs() < 1e-10);
        assert!((c.y - a.y).abs() < 1e-10);
    }

    #[test]
    fn test_pose_inverse() {
        let a = Pose2D::new(1.0, 0.0, 0.0);
        let inv = a.inverse();
        let composed = a.compose(&inv);
        assert!((composed.x).abs() < 1e-10);
        assert!((composed.y).abs() < 1e-10);
    }

    #[test]
    fn test_pose_relative() {
        let a = Pose2D::new(1.0, 0.0, 0.0);
        let b = Pose2D::new(2.0, 0.0, 0.0);
        let rel = a.relative_to(&b);
        assert!((rel.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pose_display() {
        let p = Pose2D::new(1.0, 2.0, 0.5);
        let s = format!("{}", p);
        assert!(s.contains("Pose2D"));
    }

    #[test]
    fn test_edge_creation() {
        let e = Edge::new(0, 1, Pose2D::new(1.0, 0.0, 0.0));
        assert_eq!(e.from, 0);
        assert_eq!(e.to, 1);
    }

    #[test]
    fn test_edge_residual_perfect() {
        let poses = vec![Pose2D::new(0.0, 0.0, 0.0), Pose2D::new(1.0, 0.0, 0.0)];
        let e = Edge::new(0, 1, Pose2D::new(1.0, 0.0, 0.0));
        let r = e.residual(&poses);
        assert!(r[0].abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
    }

    #[test]
    fn test_edge_display() {
        let e = Edge::new(0, 1, Pose2D::new(1.0, 0.0, 0.0));
        let s = format!("{}", e);
        assert!(s.contains("Edge"));
    }

    #[test]
    fn test_huber_kernel_below_threshold() {
        let k = RobustKernel::Huber(1.0);
        let (rho, w) = k.apply(0.5);
        assert!((rho - 0.5).abs() < 1e-10);
        assert!((w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_kernel_above_threshold() {
        let k = RobustKernel::Huber(1.0);
        let (_, w) = k.apply(4.0);
        assert!(w < 1.0);
    }

    #[test]
    fn test_cauchy_kernel() {
        let k = RobustKernel::Cauchy(1.0);
        let (_, w) = k.apply(1.0);
        assert!(w < 1.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_kernel_display() {
        assert_eq!(format!("{}", RobustKernel::None), "None");
        assert!(format!("{}", RobustKernel::Huber(1.5)).contains("Huber"));
    }

    #[test]
    fn test_graph_creation() {
        let g = PoseGraph::new();
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_graph_add_nodes_edges() {
        let mut g = PoseGraph::new();
        let n0 = g.add_node(Pose2D::identity());
        let n1 = g.add_node(Pose2D::new(1.0, 0.0, 0.0));
        g.add_edge(Edge::new(n0, n1, Pose2D::new(1.0, 0.0, 0.0)));
        assert_eq!(g.num_nodes(), 2);
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn test_total_error_perfect() {
        let mut g = PoseGraph::new();
        g.add_node(Pose2D::identity());
        g.add_node(Pose2D::new(1.0, 0.0, 0.0));
        g.add_edge(Edge::new(0, 1, Pose2D::new(1.0, 0.0, 0.0)));
        let err = g.total_error();
        assert!(err < 1e-6, "expected near-zero error, got {}", err);
    }

    #[test]
    fn test_optimize_simple() {
        let mut g = PoseGraph::new();
        g.add_node(Pose2D::identity());
        g.add_node(Pose2D::new(1.1, 0.1, 0.0)); // noisy initial
        g.add_edge(Edge::new(0, 1, Pose2D::new(1.0, 0.0, 0.0)));
        let config = OptimizerConfig::new().with_max_iterations(20);
        let result = g.optimize(&config);
        assert!(result.final_error <= result.initial_error);
    }

    #[test]
    fn test_optimize_with_huber() {
        let mut g = PoseGraph::new();
        g.add_node(Pose2D::identity());
        g.add_node(Pose2D::new(1.0, 0.0, 0.0));
        g.add_edge(Edge::new(0, 1, Pose2D::new(1.0, 0.0, 0.0)));
        let config = OptimizerConfig::new().with_kernel(RobustKernel::Huber(1.0));
        let result = g.optimize(&config);
        assert!(result.final_error < 1e-3);
    }

    #[test]
    fn test_config_builder() {
        let cfg = OptimizerConfig::new()
            .with_max_iterations(50)
            .with_tolerance(1e-8)
            .with_damping(0.01)
            .with_fixed_node(2);
        assert_eq!(cfg.max_iterations, 50);
        assert_eq!(cfg.fixed_node, 2);
    }

    #[test]
    fn test_graph_display() {
        let g = PoseGraph::new();
        let s = format!("{}", g);
        assert!(s.contains("PoseGraph"));
    }

    #[test]
    fn test_optimization_result_display() {
        let r = OptimizationResult { iterations: 5, initial_error: 1.0, final_error: 0.01, converged: true };
        let s = format!("{}", r);
        assert!(s.contains("converged=true"));
    }

    #[test]
    fn test_edge_information_matrix() {
        let e = Edge::new(0, 1, Pose2D::identity()).with_information([2.0, 0.0, 0.0, 2.0, 0.0, 2.0]);
        let m = e.info_matrix();
        assert!((m[0][0] - 2.0).abs() < 1e-10);
        assert!((m[1][1] - 2.0).abs() < 1e-10);
    }
}
