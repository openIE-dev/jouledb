//! Denavit-Hartenberg parameters — joint definitions, link frames,
//! transform matrices, and parameter tables.
//!
//! Provides a structured representation of DH parameter tables for serial
//! manipulators, including standard and modified (Craig) DH conventions,
//! link-frame computation, and common robot configurations.

use std::f64::consts::PI;

// ── Errors ──────────────────────────────────────────────────────

/// Errors for DH parameter operations.
#[derive(Debug, Clone, PartialEq)]
pub enum DhError {
    /// Empty parameter table.
    EmptyTable,
    /// Joint index out of range.
    IndexOutOfRange { index: usize, count: usize },
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Convention mismatch when composing chains.
    ConventionMismatch,
}

impl std::fmt::Display for DhError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => write!(f, "DH parameter table is empty"),
            Self::IndexOutOfRange { index, count } => {
                write!(f, "DH index {index} out of range ({count} rows)")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid DH parameter: {msg}"),
            Self::ConventionMismatch => write!(f, "DH convention mismatch"),
        }
    }
}

impl std::error::Error for DhError {}

// ── DH Convention ──────────────────────────────────────────────

/// DH convention variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DhConvention {
    /// Standard (classic) DH convention.
    Standard,
    /// Modified (Craig / proximal) DH convention.
    Modified,
}

impl std::fmt::Display for DhConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard DH"),
            Self::Modified => write!(f, "Modified DH (Craig)"),
        }
    }
}

// ── Joint Type ─────────────────────────────────────────────────

/// Type of joint actuated by a DH row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DhJointKind {
    /// Revolute — theta is the variable.
    Revolute,
    /// Prismatic — d is the variable.
    Prismatic,
    /// Fixed (no variable DOF).
    Fixed,
}

impl std::fmt::Display for DhJointKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Revolute => write!(f, "R"),
            Self::Prismatic => write!(f, "P"),
            Self::Fixed => write!(f, "F"),
        }
    }
}

// ── DH Row ─────────────────────────────────────────────────────

/// A single row of a DH parameter table.
#[derive(Debug, Clone, PartialEq)]
pub struct DhRow {
    /// Joint kind.
    pub kind: DhJointKind,
    /// Theta offset (radians) — angle about z.
    pub theta: f64,
    /// d — offset along z.
    pub d: f64,
    /// a — link length (offset along x).
    pub a: f64,
    /// Alpha (radians) — link twist about x.
    pub alpha: f64,
    /// Lower joint limit.
    pub q_min: f64,
    /// Upper joint limit.
    pub q_max: f64,
    /// Optional human-readable label.
    pub label: String,
}

impl DhRow {
    /// Create a revolute DH row.
    pub fn revolute(theta_offset: f64, d: f64, a: f64, alpha: f64) -> Self {
        Self {
            kind: DhJointKind::Revolute,
            theta: theta_offset,
            d,
            a,
            alpha,
            q_min: -PI,
            q_max: PI,
            label: String::new(),
        }
    }

    /// Create a prismatic DH row.
    pub fn prismatic(theta: f64, d_offset: f64, a: f64, alpha: f64) -> Self {
        Self {
            kind: DhJointKind::Prismatic,
            theta,
            d: d_offset,
            a,
            alpha,
            q_min: 0.0,
            q_max: 1.0,
            label: String::new(),
        }
    }

    /// Create a fixed DH row (no variable).
    pub fn fixed(theta: f64, d: f64, a: f64, alpha: f64) -> Self {
        Self {
            kind: DhJointKind::Fixed,
            theta,
            d,
            a,
            alpha,
            q_min: 0.0,
            q_max: 0.0,
            label: String::new(),
        }
    }

    /// Set joint limits.
    pub fn with_limits(mut self, q_min: f64, q_max: f64) -> Self {
        self.q_min = q_min;
        self.q_max = q_max;
        self
    }

    /// Set label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    /// Whether the joint variable is within limits.
    pub fn in_limits(&self, q: f64) -> bool {
        q >= self.q_min && q <= self.q_max
    }

    /// Effective theta for a given joint variable.
    pub fn effective_theta(&self, q: f64) -> f64 {
        match self.kind {
            DhJointKind::Revolute => self.theta + q,
            _ => self.theta,
        }
    }

    /// Effective d for a given joint variable.
    pub fn effective_d(&self, q: f64) -> f64 {
        match self.kind {
            DhJointKind::Prismatic => self.d + q,
            _ => self.d,
        }
    }
}

impl std::fmt::Display for DhRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lbl = if self.label.is_empty() { "-" } else { &self.label };
        write!(
            f,
            "{} θ={:.4} d={:.4} a={:.4} α={:.4} [{:.3},{:.3}] {}",
            self.kind, self.theta, self.d, self.a, self.alpha,
            self.q_min, self.q_max, lbl,
        )
    }
}

// ── 4x4 Transform ──────────────────────────────────────────────

/// Row-major 4x4 homogeneous transform.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat4 {
    pub data: [f64; 16],
}

impl Mat4 {
    /// Identity.
    pub fn identity() -> Self {
        let mut data = [0.0; 16];
        data[0] = 1.0;
        data[5] = 1.0;
        data[10] = 1.0;
        data[15] = 1.0;
        Self { data }
    }

    /// Get element at (row, col).
    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * 4 + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * 4 + col] = val;
    }

    /// Multiply two 4x4 matrices.
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += self.data[r * 4 + k] * rhs.data[k * 4 + c];
                }
                out[r * 4 + c] = s;
            }
        }
        Mat4 { data: out }
    }

    /// Extract translation.
    pub fn translation(&self) -> [f64; 3] {
        [self.data[3], self.data[7], self.data[11]]
    }

    /// Extract the z-axis of the rotation (third column).
    pub fn z_axis(&self) -> [f64; 3] {
        [self.data[2], self.data[6], self.data[10]]
    }

    /// Standard DH transform: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha).
    pub fn standard_dh(theta: f64, d: f64, a: f64, alpha: f64) -> Self {
        let (st, ct) = theta.sin_cos();
        let (sa, ca) = alpha.sin_cos();
        let mut m = Self::identity();
        m.data[0] = ct;       m.data[1] = -st * ca;  m.data[2] = st * sa;   m.data[3] = a * ct;
        m.data[4] = st;       m.data[5] = ct * ca;    m.data[6] = -ct * sa;  m.data[7] = a * st;
        m.data[8] = 0.0;      m.data[9] = sa;         m.data[10] = ca;       m.data[11] = d;
        m
    }

    /// Modified DH transform: Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i).
    pub fn modified_dh(theta: f64, d: f64, a_prev: f64, alpha_prev: f64) -> Self {
        let (st, ct) = theta.sin_cos();
        let (sa, ca) = alpha_prev.sin_cos();
        let mut m = Self::identity();
        m.data[0] = ct;         m.data[1] = -st;        m.data[2] = 0.0;   m.data[3] = a_prev;
        m.data[4] = st * ca;    m.data[5] = ct * ca;     m.data[6] = -sa;   m.data[7] = -sa * d;
        m.data[8] = st * sa;    m.data[9] = ct * sa;     m.data[10] = ca;   m.data[11] = ca * d;
        m
    }

    /// Determinant of the upper-left 3x3.
    pub fn rotation_determinant(&self) -> f64 {
        let m = &self.data;
        m[0] * (m[5] * m[10] - m[6] * m[9])
            - m[1] * (m[4] * m[10] - m[6] * m[8])
            + m[2] * (m[4] * m[9] - m[5] * m[8])
    }
}

impl std::fmt::Display for Mat4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in 0..4 {
            let i = r * 4;
            writeln!(
                f,
                "[{:8.4} {:8.4} {:8.4} {:8.4}]",
                self.data[i], self.data[i + 1], self.data[i + 2], self.data[i + 3],
            )?;
        }
        Ok(())
    }
}

// ── DH Table ───────────────────────────────────────────────────

/// A complete DH parameter table for a serial manipulator.
#[derive(Debug, Clone)]
pub struct DhTable {
    /// Convention in use.
    pub convention: DhConvention,
    /// Rows (one per joint/link).
    rows: Vec<DhRow>,
    /// Optional robot name.
    name: String,
}

impl DhTable {
    /// Create a new DH table.
    pub fn new(convention: DhConvention, rows: Vec<DhRow>) -> Result<Self, DhError> {
        if rows.is_empty() {
            return Err(DhError::EmptyTable);
        }
        Ok(Self { convention, rows, name: String::new() })
    }

    /// Set a name for this robot.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of actuated DOFs (non-fixed rows).
    pub fn num_dof(&self) -> usize {
        self.rows.iter().filter(|r| r.kind != DhJointKind::Fixed).count()
    }

    /// Get a row by index.
    pub fn row(&self, index: usize) -> Result<&DhRow, DhError> {
        self.rows.get(index).ok_or(DhError::IndexOutOfRange {
            index,
            count: self.rows.len(),
        })
    }

    /// Iterate over rows.
    pub fn rows(&self) -> &[DhRow] {
        &self.rows
    }

    /// Compute the transform for a single row at joint value `q`.
    pub fn row_transform(&self, index: usize, q: f64) -> Result<Mat4, DhError> {
        let row = self.row(index)?;
        let theta = row.effective_theta(q);
        let d = row.effective_d(q);
        let t = match self.convention {
            DhConvention::Standard => Mat4::standard_dh(theta, d, row.a, row.alpha),
            DhConvention::Modified => Mat4::modified_dh(theta, d, row.a, row.alpha),
        };
        Ok(t)
    }

    /// Compute the cumulative transform through all rows.
    ///
    /// `q` must have one entry per actuated DOF (skipping fixed rows).
    pub fn forward_transform(&self, q: &[f64]) -> Result<Mat4, DhError> {
        let n_dof = self.num_dof();
        if q.len() != n_dof {
            return Err(DhError::InvalidParameter(format!(
                "expected {n_dof} joint values, got {}",
                q.len(),
            )));
        }
        let mut t = Mat4::identity();
        let mut qi = 0;
        for (i, row) in self.rows.iter().enumerate() {
            let qv = if row.kind == DhJointKind::Fixed {
                0.0
            } else {
                let v = q[qi];
                qi += 1;
                v
            };
            let ti = self.row_transform(i, qv)?;
            t = t.mul(&ti);
        }
        Ok(t)
    }

    /// Compute all intermediate transforms (n+1: identity + each cumulative product).
    pub fn all_transforms(&self, q: &[f64]) -> Result<Vec<Mat4>, DhError> {
        let n_dof = self.num_dof();
        if q.len() != n_dof {
            return Err(DhError::InvalidParameter(format!(
                "expected {n_dof} joint values, got {}",
                q.len(),
            )));
        }
        let mut frames = Vec::with_capacity(self.rows.len() + 1);
        let mut t = Mat4::identity();
        frames.push(t.clone());
        let mut qi = 0;
        for (i, row) in self.rows.iter().enumerate() {
            let qv = if row.kind == DhJointKind::Fixed {
                0.0
            } else {
                let v = q[qi];
                qi += 1;
                v
            };
            let ti = self.row_transform(i, qv)?;
            t = t.mul(&ti);
            frames.push(t.clone());
        }
        Ok(frames)
    }

    /// Validate that all joint values are within limits.
    pub fn validate_joints(&self, q: &[f64]) -> Result<(), DhError> {
        let mut qi = 0;
        for (i, row) in self.rows.iter().enumerate() {
            if row.kind == DhJointKind::Fixed {
                continue;
            }
            if qi >= q.len() {
                return Err(DhError::InvalidParameter(format!(
                    "not enough joint values (row {i})",
                )));
            }
            if !row.in_limits(q[qi]) {
                return Err(DhError::InvalidParameter(format!(
                    "joint {i} value {:.4} outside [{:.4}, {:.4}]",
                    q[qi], row.q_min, row.q_max,
                )));
            }
            qi += 1;
        }
        Ok(())
    }

    /// Append another table (must share convention).
    pub fn append(&mut self, other: &DhTable) -> Result<(), DhError> {
        if self.convention != other.convention {
            return Err(DhError::ConventionMismatch);
        }
        self.rows.extend_from_slice(&other.rows);
        Ok(())
    }
}

impl std::fmt::Display for DhTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = if self.name.is_empty() { "unnamed" } else { &self.name };
        writeln!(f, "DhTable \"{name}\" ({}, {} rows, {} DOF):", self.convention, self.rows.len(), self.num_dof())?;
        writeln!(f, "  {:>3}  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {}", "#", "Kind", "theta", "d", "a", "alpha", "limits", "label")?;
        for (i, row) in self.rows.iter().enumerate() {
            let lbl = if row.label.is_empty() { "-" } else { &row.label };
            writeln!(
                f,
                "  {:>3}  {:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  [{:>4.2},{:>4.2}]  {}",
                i, row.kind, row.theta, row.d, row.a, row.alpha,
                row.q_min, row.q_max, lbl,
            )?;
        }
        Ok(())
    }
}

// ── Common Robot Configurations ────────────────────────────────

/// Create DH table for a planar RR arm.
pub fn planar_rr(l1: f64, l2: f64) -> Result<DhTable, DhError> {
    DhTable::new(
        DhConvention::Standard,
        vec![
            DhRow::revolute(0.0, 0.0, l1, 0.0).with_label("shoulder"),
            DhRow::revolute(0.0, 0.0, l2, 0.0).with_label("elbow"),
        ],
    )
}

/// Create DH table for a 3-DOF anthropomorphic arm.
pub fn anthropomorphic_3dof(l1: f64, l2: f64, l3: f64) -> Result<DhTable, DhError> {
    DhTable::new(
        DhConvention::Standard,
        vec![
            DhRow::revolute(0.0, l1, 0.0, PI / 2.0).with_label("waist"),
            DhRow::revolute(0.0, 0.0, l2, 0.0).with_label("shoulder"),
            DhRow::revolute(0.0, 0.0, l3, 0.0).with_label("elbow"),
        ],
    )
}

/// Create DH table for a Stanford manipulator (RRP spherical wrist).
pub fn stanford_arm(d2: f64) -> Result<DhTable, DhError> {
    DhTable::new(
        DhConvention::Standard,
        vec![
            DhRow::revolute(0.0, 0.4120, 0.0, -PI / 2.0).with_label("J1"),
            DhRow::revolute(0.0, 0.15005, 0.0, PI / 2.0).with_label("J2"),
            DhRow::prismatic(0.0, d2, 0.0, 0.0)
                .with_limits(0.0, 1.0)
                .with_label("J3"),
            DhRow::revolute(0.0, 0.0, 0.0, -PI / 2.0).with_label("J4"),
            DhRow::revolute(0.0, 0.0, 0.0, PI / 2.0).with_label("J5"),
            DhRow::revolute(0.0, 0.0, 0.0, 0.0).with_label("J6"),
        ],
    )
}

/// Create DH table for a SCARA robot.
pub fn scara(l1: f64, l2: f64, d_max: f64) -> Result<DhTable, DhError> {
    DhTable::new(
        DhConvention::Standard,
        vec![
            DhRow::revolute(0.0, 0.0, l1, 0.0).with_label("J1"),
            DhRow::revolute(0.0, 0.0, l2, PI).with_label("J2"),
            DhRow::prismatic(0.0, 0.0, 0.0, 0.0)
                .with_limits(0.0, d_max)
                .with_label("J3"),
            DhRow::revolute(0.0, 0.0, 0.0, 0.0).with_label("J4"),
        ],
    )
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_empty_table_error() {
        let r = DhTable::new(DhConvention::Standard, vec![]);
        assert!(matches!(r, Err(DhError::EmptyTable)));
    }

    #[test]
    fn test_planar_rr_dof() {
        let table = planar_rr(1.0, 1.0).unwrap();
        assert_eq!(table.num_dof(), 2);
        assert_eq!(table.num_rows(), 2);
    }

    #[test]
    fn test_planar_rr_zero_config() {
        let table = planar_rr(1.0, 1.0).unwrap();
        let t = table.forward_transform(&[0.0, 0.0]).unwrap();
        let p = t.translation();
        assert!(approx_eq(p[0], 2.0));
        assert!(approx_eq(p[1], 0.0));
    }

    #[test]
    fn test_planar_rr_90_degrees() {
        let table = planar_rr(1.0, 1.0).unwrap();
        let t = table.forward_transform(&[PI / 2.0, 0.0]).unwrap();
        let p = t.translation();
        assert!(approx_eq(p[0], 0.0));
        assert!(approx_eq(p[1], 2.0));
    }

    #[test]
    fn test_standard_dh_identity_params() {
        let t = Mat4::standard_dh(0.0, 0.0, 0.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(t.at(i, j), expected));
            }
        }
    }

    #[test]
    fn test_modified_dh_identity_params() {
        let t = Mat4::modified_dh(0.0, 0.0, 0.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(t.at(i, j), expected));
            }
        }
    }

    #[test]
    fn test_rotation_determinant() {
        let t = Mat4::standard_dh(0.5, 0.3, 1.0, -0.7);
        let det = t.rotation_determinant();
        assert!(approx_eq(det, 1.0));
    }

    #[test]
    fn test_all_transforms_count() {
        let table = planar_rr(1.0, 1.0).unwrap();
        let frames = table.all_transforms(&[0.0, 0.0]).unwrap();
        assert_eq!(frames.len(), 3); // identity + 2 rows
    }

    #[test]
    fn test_validate_joints_ok() {
        let table = planar_rr(1.0, 1.0).unwrap();
        assert!(table.validate_joints(&[0.0, 0.0]).is_ok());
    }

    #[test]
    fn test_validate_joints_violation() {
        let table = planar_rr(1.0, 1.0).unwrap();
        assert!(table.validate_joints(&[10.0, 0.0]).is_err());
    }

    #[test]
    fn test_wrong_dof_count() {
        let table = planar_rr(1.0, 1.0).unwrap();
        assert!(table.forward_transform(&[0.0]).is_err());
    }

    #[test]
    fn test_scara_dof() {
        let table = scara(0.4, 0.3, 0.2).unwrap();
        assert_eq!(table.num_dof(), 4);
    }

    #[test]
    fn test_stanford_arm_dof() {
        let table = stanford_arm(0.2).unwrap();
        assert_eq!(table.num_dof(), 6);
    }

    #[test]
    fn test_anthropomorphic_3dof() {
        let table = anthropomorphic_3dof(0.5, 0.4, 0.3).unwrap();
        assert_eq!(table.num_dof(), 3);
    }

    #[test]
    fn test_fixed_row_not_counted() {
        let rows = vec![
            DhRow::revolute(0.0, 0.0, 1.0, 0.0),
            DhRow::fixed(0.0, 0.5, 0.0, 0.0),
            DhRow::revolute(0.0, 0.0, 1.0, 0.0),
        ];
        let table = DhTable::new(DhConvention::Standard, rows).unwrap();
        assert_eq!(table.num_dof(), 2);
        assert_eq!(table.num_rows(), 3);
    }

    #[test]
    fn test_append_same_convention() {
        let mut t1 = planar_rr(1.0, 1.0).unwrap();
        let t2 = planar_rr(0.5, 0.5).unwrap();
        assert!(t1.append(&t2).is_ok());
        assert_eq!(t1.num_rows(), 4);
    }

    #[test]
    fn test_append_different_convention() {
        let mut t1 = DhTable::new(DhConvention::Standard, vec![DhRow::revolute(0.0, 0.0, 1.0, 0.0)]).unwrap();
        let t2 = DhTable::new(DhConvention::Modified, vec![DhRow::revolute(0.0, 0.0, 1.0, 0.0)]).unwrap();
        assert!(matches!(t1.append(&t2), Err(DhError::ConventionMismatch)));
    }

    #[test]
    fn test_display_table() {
        let table = planar_rr(1.0, 1.0).unwrap().with_name("TestBot");
        let s = format!("{table}");
        assert!(s.contains("TestBot"));
        assert!(s.contains("Standard DH"));
    }

    #[test]
    fn test_display_error() {
        let e = DhError::EmptyTable;
        let s = format!("{e}");
        assert!(s.contains("empty"));
    }

    #[test]
    fn test_row_display() {
        let row = DhRow::revolute(0.0, 0.0, 1.0, 0.5).with_label("elbow");
        let s = format!("{row}");
        assert!(s.contains("R"));
        assert!(s.contains("elbow"));
    }
}
