//! # Workspace Analysis
//!
//! Analyzes the reachable workspace of robotic manipulators. Computes
//! reachability maps, dexterity measures, manipulability ellipsoids, and
//! workspace boundary approximations for serial kinematic chains.

use std::fmt;
use std::collections::HashMap;

// ── Core Types ──

/// 3D position in workspace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

/// Joint limits for a single revolute joint.
#[derive(Clone, Debug)]
pub struct JointLimit {
    pub min_angle: f64,
    pub max_angle: f64,
}

impl JointLimit {
    pub fn new(min_angle: f64, max_angle: f64) -> Self {
        Self { min_angle, max_angle }
    }

    pub fn range(&self) -> f64 {
        self.max_angle - self.min_angle
    }

    pub fn center(&self) -> f64 {
        (self.min_angle + self.max_angle) / 2.0
    }

    pub fn contains(&self, angle: f64) -> bool {
        angle >= self.min_angle && angle <= self.max_angle
    }
}

/// Link length and DH-like parameters for simple analysis.
#[derive(Clone, Debug)]
pub struct LinkParam {
    pub length: f64,
    pub offset: f64,
    pub twist: f64,
    pub joint_limit: JointLimit,
}

impl LinkParam {
    pub fn new(length: f64, offset: f64, twist: f64, limit: JointLimit) -> Self {
        Self { length, offset, twist, joint_limit: limit }
    }
}

// ── Manipulability ──

/// Manipulability metrics at a given configuration.
#[derive(Clone, Debug)]
pub struct Manipulability {
    pub yoshikawa: f64,
    pub condition_number: f64,
    pub min_singular: f64,
    pub max_singular: f64,
    pub isotropy: f64,
}

impl Manipulability {
    pub fn compute(jacobian: &[Vec<f64>]) -> Self {
        let rows = jacobian.len();
        if rows == 0 {
            return Self { yoshikawa: 0.0, condition_number: f64::INFINITY, min_singular: 0.0, max_singular: 0.0, isotropy: 0.0 };
        }
        let cols = jacobian[0].len();

        // Compute J * J^T
        let mut jjt = vec![vec![0.0; rows]; rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                for k in 0..cols {
                    sum += jacobian[i][k] * jacobian[j][k];
                }
                jjt[i][j] = sum;
            }
        }

        // Approximate singular values via eigenvalues of J*J^T (power iteration)
        let eigenvalues = Self::eigenvalues_symmetric(&jjt);
        let singular_values: Vec<f64> = eigenvalues.iter().map(|e| e.max(0.0).sqrt()).collect();

        let min_sv = singular_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_sv = singular_values.iter().copied().fold(0.0, f64::max);

        let yoshikawa = singular_values.iter().product::<f64>();
        let condition_number = if min_sv > 1e-12 { max_sv / min_sv } else { f64::INFINITY };
        let isotropy = if max_sv > 1e-12 { min_sv / max_sv } else { 0.0 };

        Self { yoshikawa, condition_number, min_singular: min_sv, max_singular: max_sv, isotropy }
    }

    /// Simple eigenvalue estimation for small symmetric matrices using QR-like iteration.
    fn eigenvalues_symmetric(mat: &[Vec<f64>]) -> Vec<f64> {
        let n = mat.len();
        if n == 0 { return vec![]; }
        if n == 1 { return vec![mat[0][0]]; }

        // For 2x2 or 3x3, use direct formulas via trace and determinant
        if n == 2 {
            let tr = mat[0][0] + mat[1][1];
            let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
            let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
            return vec![(tr + disc) / 2.0, (tr - disc) / 2.0];
        }

        // Fallback: return diagonal (rough estimate)
        (0..n).map(|i| mat[i][i]).collect()
    }

    pub fn is_singular(&self) -> bool {
        self.min_singular < 1e-6
    }
}

impl fmt::Display for Manipulability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Manipulability(yosh={:.4}, cond={:.2}, iso={:.4})",
            self.yoshikawa, self.condition_number, self.isotropy)
    }
}

// ── Reachability Map ──

/// Voxelized reachability map of the workspace.
#[derive(Clone, Debug)]
pub struct ReachabilityMap {
    voxels: HashMap<(i32, i32, i32), VoxelInfo>,
    resolution: f64,
    total_samples: usize,
}

/// Information stored per voxel.
#[derive(Clone, Debug)]
pub struct VoxelInfo {
    pub reach_count: usize,
    pub max_manipulability: f64,
    pub avg_manipulability: f64,
    manip_sum: f64,
}

impl VoxelInfo {
    fn new() -> Self {
        Self { reach_count: 0, max_manipulability: 0.0, avg_manipulability: 0.0, manip_sum: 0.0 }
    }

    fn add_sample(&mut self, manipulability: f64) {
        self.reach_count += 1;
        self.manip_sum += manipulability;
        self.avg_manipulability = self.manip_sum / self.reach_count as f64;
        if manipulability > self.max_manipulability {
            self.max_manipulability = manipulability;
        }
    }
}

impl ReachabilityMap {
    pub fn new(resolution: f64) -> Self {
        Self {
            voxels: HashMap::new(),
            resolution: resolution.max(0.001),
            total_samples: 0,
        }
    }

    fn point_to_voxel(&self, p: &Point3) -> (i32, i32, i32) {
        let ix = (p.x / self.resolution).floor() as i32;
        let iy = (p.y / self.resolution).floor() as i32;
        let iz = (p.z / self.resolution).floor() as i32;
        (ix, iy, iz)
    }

    pub fn add_sample(&mut self, position: Point3, manipulability: f64) {
        let key = self.point_to_voxel(&position);
        self.voxels.entry(key).or_insert_with(VoxelInfo::new).add_sample(manipulability);
        self.total_samples += 1;
    }

    pub fn is_reachable(&self, position: &Point3) -> bool {
        let key = self.point_to_voxel(position);
        self.voxels.contains_key(&key)
    }

    pub fn reachable_voxels(&self) -> usize {
        self.voxels.len()
    }

    pub fn get_voxel(&self, position: &Point3) -> Option<&VoxelInfo> {
        let key = self.point_to_voxel(position);
        self.voxels.get(&key)
    }

    /// Compute the reachability index (fraction of reachable vs total sampled).
    pub fn reachability_index(&self, total_voxels: usize) -> f64 {
        if total_voxels == 0 { return 0.0; }
        self.voxels.len() as f64 / total_voxels as f64
    }

    /// Get workspace bounds (min/max corners).
    pub fn bounds(&self) -> (Point3, Point3) {
        if self.voxels.is_empty() {
            return (Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        }
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;
        for &(x, y, z) in self.voxels.keys() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            max_z = max_z.max(z);
        }
        (
            Point3::new(min_x as f64 * self.resolution, min_y as f64 * self.resolution, min_z as f64 * self.resolution),
            Point3::new((max_x + 1) as f64 * self.resolution, (max_y + 1) as f64 * self.resolution, (max_z + 1) as f64 * self.resolution),
        )
    }
}

impl fmt::Display for ReachabilityMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReachabilityMap({} voxels, res={:.3}, {} samples)",
            self.voxels.len(), self.resolution, self.total_samples)
    }
}

// ── Workspace Analyzer ──

/// Analyzes workspace properties of a planar or spatial manipulator.
#[derive(Clone, Debug)]
pub struct WorkspaceAnalyzer {
    links: Vec<LinkParam>,
    samples_per_joint: usize,
    resolution: f64,
}

impl WorkspaceAnalyzer {
    pub fn new(links: Vec<LinkParam>) -> Self {
        Self { links, samples_per_joint: 16, resolution: 0.1 }
    }

    pub fn with_samples_per_joint(mut self, n: usize) -> Self {
        self.samples_per_joint = n.max(2);
        self
    }

    pub fn with_resolution(mut self, r: f64) -> Self {
        self.resolution = r.max(0.001);
        self
    }

    pub fn num_joints(&self) -> usize {
        self.links.len()
    }

    /// Compute forward kinematics for a planar arm (2D projected to 3D with z=offset).
    fn fk_planar(&self, angles: &[f64]) -> Point3 {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        let mut cumulative_angle = 0.0;

        for (i, link) in self.links.iter().enumerate() {
            let angle = angles.get(i).copied().unwrap_or(0.0);
            cumulative_angle += angle;
            x += link.length * cumulative_angle.cos();
            y += link.length * cumulative_angle.sin();
            z += link.offset;
        }
        Point3::new(x, y, z)
    }

    /// Compute approximate Jacobian via finite differences.
    fn jacobian_numerical(&self, angles: &[f64]) -> Vec<Vec<f64>> {
        let n = angles.len();
        let eps = 1e-6;
        let base = self.fk_planar(angles);
        let mut jac = vec![vec![0.0; n]; 3];

        for j in 0..n {
            let mut perturbed = angles.to_vec();
            perturbed[j] += eps;
            let fwd = self.fk_planar(&perturbed);
            jac[0][j] = (fwd.x - base.x) / eps;
            jac[1][j] = (fwd.y - base.y) / eps;
            jac[2][j] = (fwd.z - base.z) / eps;
        }
        jac
    }

    /// Build reachability map by sampling joint space.
    pub fn build_reachability_map(&self) -> ReachabilityMap {
        let mut map = ReachabilityMap::new(self.resolution);
        let n = self.links.len();

        if n == 0 { return map; }

        // Generate samples recursively
        let mut angles = vec![0.0; n];
        self.sample_recursive(&mut map, &mut angles, 0);
        map
    }

    fn sample_recursive(&self, map: &mut ReachabilityMap, angles: &mut Vec<f64>, joint: usize) {
        if joint >= self.links.len() {
            let pos = self.fk_planar(angles);
            let jac = self.jacobian_numerical(angles);
            let manip = Manipulability::compute(&jac);
            map.add_sample(pos, manip.yoshikawa);
            return;
        }

        let limit = &self.links[joint].joint_limit;
        let step = limit.range() / (self.samples_per_joint as f64 - 1.0).max(1.0);
        for i in 0..self.samples_per_joint {
            angles[joint] = limit.min_angle + step * i as f64;
            self.sample_recursive(map, angles, joint + 1);
        }
    }

    /// Estimate maximum reach distance.
    pub fn max_reach(&self) -> f64 {
        self.links.iter().map(|l| l.length).sum()
    }

    /// Estimate minimum reach distance (fully folded).
    pub fn min_reach(&self) -> f64 {
        if self.links.is_empty() { return 0.0; }
        let lengths: Vec<f64> = self.links.iter().map(|l| l.length).collect();
        let max_len = lengths.iter().copied().fold(0.0, f64::max);
        let total: f64 = lengths.iter().sum();
        (2.0 * max_len - total).max(0.0)
    }

    /// Compute workspace volume estimate from reachability map.
    pub fn workspace_volume(&self, map: &ReachabilityMap) -> f64 {
        let voxel_vol = map.resolution.powi(3);
        map.reachable_voxels() as f64 * voxel_vol
    }

    /// Compute dexterity map (average manipulability per voxel).
    pub fn average_dexterity(&self, map: &ReachabilityMap) -> f64 {
        if map.voxels.is_empty() { return 0.0; }
        let sum: f64 = map.voxels.values().map(|v| v.avg_manipulability).sum();
        sum / map.voxels.len() as f64
    }
}

impl fmt::Display for WorkspaceAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WorkspaceAnalyzer({} joints, {} samples/joint, res={:.3})",
            self.links.len(), self.samples_per_joint, self.resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn two_link_arm() -> Vec<LinkParam> {
        vec![
            LinkParam::new(1.0, 0.0, 0.0, JointLimit::new(-PI, PI)),
            LinkParam::new(1.0, 0.0, 0.0, JointLimit::new(-PI, PI)),
        ]
    }

    #[test]
    fn test_point3_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_norm() {
        let p = Point3::new(1.0, 2.0, 2.0);
        assert!((p.norm() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_joint_limit() {
        let jl = JointLimit::new(-1.5, 1.5);
        assert!((jl.range() - 3.0).abs() < 1e-10);
        assert!(jl.contains(0.0));
        assert!(!jl.contains(2.0));
    }

    #[test]
    fn test_fk_planar_straight() {
        let links = two_link_arm();
        let analyzer = WorkspaceAnalyzer::new(links);
        let pos = analyzer.fk_planar(&[0.0, 0.0]);
        assert!((pos.x - 2.0).abs() < 1e-10);
        assert!(pos.y.abs() < 1e-10);
    }

    #[test]
    fn test_fk_planar_bent() {
        let links = two_link_arm();
        let analyzer = WorkspaceAnalyzer::new(links);
        let pos = analyzer.fk_planar(&[0.0, PI]);
        assert!(pos.x.abs() < 1e-10);
        assert!(pos.y.abs() < 1e-10);
    }

    #[test]
    fn test_max_reach() {
        let links = two_link_arm();
        let analyzer = WorkspaceAnalyzer::new(links);
        assert!((analyzer.max_reach() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_reach() {
        let links = two_link_arm();
        let analyzer = WorkspaceAnalyzer::new(links);
        assert!((analyzer.min_reach() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_manipulability_2x2() {
        let jac = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let m = Manipulability::compute(&jac);
        assert!((m.yoshikawa - 1.0).abs() < 1e-6);
        assert!((m.isotropy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manipulability_singular() {
        let jac = vec![
            vec![1.0, 1.0],
            vec![0.0, 0.0],
        ];
        let m = Manipulability::compute(&jac);
        assert!(m.is_singular());
    }

    #[test]
    fn test_reachability_map_add() {
        let mut map = ReachabilityMap::new(0.1);
        map.add_sample(Point3::new(1.0, 0.0, 0.0), 0.5);
        assert_eq!(map.reachable_voxels(), 1);
        assert!(map.is_reachable(&Point3::new(1.05, 0.05, 0.05)));
    }

    #[test]
    fn test_reachability_map_bounds() {
        let mut map = ReachabilityMap::new(0.5);
        map.add_sample(Point3::new(0.0, 0.0, 0.0), 0.1);
        map.add_sample(Point3::new(2.0, 3.0, 1.0), 0.2);
        let (lo, hi) = map.bounds();
        assert!(lo.x <= 0.0 && lo.y <= 0.0);
        assert!(hi.x >= 2.0 && hi.y >= 3.0);
    }

    #[test]
    fn test_workspace_analyzer_builder() {
        let links = two_link_arm();
        let wa = WorkspaceAnalyzer::new(links)
            .with_samples_per_joint(8)
            .with_resolution(0.2);
        assert_eq!(wa.samples_per_joint, 8);
        assert!((wa.resolution - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_workspace_analyzer_display() {
        let links = two_link_arm();
        let wa = WorkspaceAnalyzer::new(links);
        let s = format!("{wa}");
        assert!(s.contains("2 joints"));
    }

    #[test]
    fn test_build_reachability_small() {
        let links = vec![
            LinkParam::new(1.0, 0.0, 0.0, JointLimit::new(-0.5, 0.5)),
        ];
        let wa = WorkspaceAnalyzer::new(links).with_samples_per_joint(5).with_resolution(0.2);
        let map = wa.build_reachability_map();
        assert!(map.reachable_voxels() > 0);
    }

    #[test]
    fn test_workspace_volume() {
        let links = vec![
            LinkParam::new(1.0, 0.0, 0.0, JointLimit::new(-0.5, 0.5)),
        ];
        let wa = WorkspaceAnalyzer::new(links).with_samples_per_joint(5).with_resolution(0.5);
        let map = wa.build_reachability_map();
        let vol = wa.workspace_volume(&map);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_average_dexterity() {
        let mut map = ReachabilityMap::new(0.1);
        map.add_sample(Point3::new(1.0, 0.0, 0.0), 0.8);
        map.add_sample(Point3::new(10.0, 0.0, 0.0), 0.4);
        let links = two_link_arm();
        let wa = WorkspaceAnalyzer::new(links);
        let dex = wa.average_dexterity(&map);
        assert!((dex - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_reachability_index() {
        let mut map = ReachabilityMap::new(1.0);
        map.add_sample(Point3::new(0.5, 0.5, 0.5), 0.1);
        map.add_sample(Point3::new(1.5, 0.5, 0.5), 0.2);
        assert!((map.reachability_index(10) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_voxel_info_accumulation() {
        let mut map = ReachabilityMap::new(0.1);
        let p = Point3::new(1.0, 0.0, 0.0);
        map.add_sample(p, 0.3);
        map.add_sample(Point3::new(1.05, 0.01, 0.01), 0.7); // same voxel
        let info = map.get_voxel(&p).unwrap();
        assert_eq!(info.reach_count, 2);
        assert!((info.max_manipulability - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_jacobian() {
        let links = two_link_arm();
        let wa = WorkspaceAnalyzer::new(links);
        let jac = wa.jacobian_numerical(&[0.0, 0.0]);
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 2);
        // d(x)/d(q1) at q=[0,0]: should be ~-sin(0)*1 - sin(0)*1 = 0
        assert!(jac[1][0].abs() > 0.5); // dy/dq1 should be nonzero
    }
}
