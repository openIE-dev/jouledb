//! Chord diagram: circular layout with arcs for groups and chords for
//! relationships.  Matrix-based input, arc angles proportional to values,
//! group labels, SVG output.  Pure Rust — no browser dependency.

use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A group (arc segment) in the chord diagram.
#[derive(Debug, Clone)]
pub struct ChordGroup {
    pub label: String,
    pub color: String,
    /// Computed: start angle in radians.
    pub start_angle: f64,
    /// Computed: end angle in radians.
    pub end_angle: f64,
    /// Computed: total value for this group.
    pub value: f64,
}

impl ChordGroup {
    pub fn new(label: impl Into<String>, color: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            color: color.into(),
            start_angle: 0.0,
            end_angle: 0.0,
            value: 0.0,
        }
    }

    /// Arc span in radians.
    pub fn arc_span(&self) -> f64 {
        (self.end_angle - self.start_angle).abs()
    }

    /// Midpoint angle for label placement.
    pub fn mid_angle(&self) -> f64 {
        (self.start_angle + self.end_angle) / 2.0
    }
}

/// A chord connecting two groups.
#[derive(Debug, Clone)]
pub struct Chord {
    pub source_index: usize,
    pub target_index: usize,
    pub source_start: f64,
    pub source_end: f64,
    pub target_start: f64,
    pub target_end: f64,
    pub value: f64,
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for chord diagram layout and rendering.
#[derive(Debug, Clone)]
pub struct ChordConfig {
    pub width: f64,
    pub height: f64,
    /// Outer radius of the arc ring.
    pub outer_radius: f64,
    /// Inner radius of the arc ring.
    pub inner_radius: f64,
    /// Gap between group arcs in radians.
    pub pad_angle: f64,
    pub font_size: f64,
    pub chord_opacity: f64,
}

impl Default for ChordConfig {
    fn default() -> Self {
        Self {
            width: 500.0,
            height: 500.0,
            outer_radius: 220.0,
            inner_radius: 200.0,
            pad_angle: 0.04,
            font_size: 12.0,
            chord_opacity: 0.5,
        }
    }
}

impl ChordConfig {
    pub fn center(&self) -> (f64, f64) {
        (self.width / 2.0, self.height / 2.0)
    }
}

// ── Layout ──────────────────────────────────────────────────────

/// Compute groups and chords from a square matrix.
///
/// `matrix[i][j]` is the flow from group `i` to group `j`.
/// `groups` must have the same length as the matrix dimension.
pub fn compute_layout(
    matrix: &[Vec<f64>],
    groups: &mut [ChordGroup],
    cfg: &ChordConfig,
) -> Vec<Chord> {
    let n = groups.len();
    assert!(
        matrix.len() == n && matrix.iter().all(|row| row.len() == n),
        "matrix must be {n}x{n}"
    );

    // Compute row sums (total per group)
    let row_sums: Vec<f64> = matrix
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect();
    let grand_total: f64 = row_sums.iter().sum();

    // Assign group values
    for (i, g) in groups.iter_mut().enumerate() {
        g.value = row_sums[i];
    }

    // Distribute angles
    let total_pad = cfg.pad_angle * n as f64;
    let available = (2.0 * PI - total_pad).max(0.0);
    let scale = if grand_total > 0.0 {
        available / grand_total
    } else {
        0.0
    };

    let mut angle = 0.0_f64;
    for (i, g) in groups.iter_mut().enumerate() {
        g.start_angle = angle;
        g.end_angle = angle + row_sums[i] * scale;
        angle = g.end_angle + cfg.pad_angle;
    }

    // Compute chords
    let mut chords = Vec::new();
    // Track sub-arc offsets within each group
    let mut group_offsets: Vec<f64> = groups.iter().map(|g| g.start_angle).collect();

    for i in 0..n {
        for j in 0..n {
            let val = matrix[i][j];
            if val <= 0.0 {
                continue;
            }
            let src_span = if row_sums[i] > 0.0 {
                val / row_sums[i] * groups[i].arc_span()
            } else {
                0.0
            };
            let tgt_span = if row_sums[j] > 0.0 {
                val / row_sums[j] * groups[j].arc_span()
            } else {
                0.0
            };

            let src_start = group_offsets[i];
            let src_end = src_start + src_span;
            group_offsets[i] = src_end;

            let tgt_start = group_offsets[j];
            let tgt_end = tgt_start + tgt_span;
            // Only advance target offset when i != j to avoid double-counting self-loops
            if i != j {
                group_offsets[j] = tgt_end;
            }

            chords.push(Chord {
                source_index: i,
                target_index: j,
                source_start: src_start,
                source_end: src_end,
                target_start: tgt_start,
                target_end: tgt_end,
                value: val,
            });
        }
    }

    chords
}

// ── SVG helpers ─────────────────────────────────────────────────

/// SVG arc path for a group ring segment.
fn arc_path(
    start: f64,
    end: f64,
    inner_r: f64,
    outer_r: f64,
    cx: f64,
    cy: f64,
) -> String {
    let x0_outer = cx + outer_r * start.cos();
    let y0_outer = cy + outer_r * start.sin();
    let x1_outer = cx + outer_r * end.cos();
    let y1_outer = cy + outer_r * end.sin();
    let x0_inner = cx + inner_r * end.cos();
    let y0_inner = cy + inner_r * end.sin();
    let x1_inner = cx + inner_r * start.cos();
    let y1_inner = cy + inner_r * start.sin();

    let large = if (end - start).abs() > PI { 1 } else { 0 };

    format!(
        "M{x0_outer},{y0_outer} \
         A{outer_r},{outer_r} 0 {large},1 {x1_outer},{y1_outer} \
         L{x0_inner},{y0_inner} \
         A{inner_r},{inner_r} 0 {large},0 {x1_inner},{y1_inner} Z"
    )
}

/// SVG path for a chord (two arcs connected by curves).
fn chord_path(chord: &Chord, radius: f64, cx: f64, cy: f64) -> String {
    let src_s = chord.source_start;
    let src_e = chord.source_end;
    let tgt_s = chord.target_start;
    let tgt_e = chord.target_end;

    let x_ss = cx + radius * src_s.cos();
    let y_ss = cy + radius * src_s.sin();
    let x_se = cx + radius * src_e.cos();
    let y_se = cy + radius * src_e.sin();
    let x_ts = cx + radius * tgt_s.cos();
    let y_ts = cy + radius * tgt_s.sin();
    let x_te = cx + radius * tgt_e.cos();
    let y_te = cy + radius * tgt_e.sin();

    let large_src = if (src_e - src_s).abs() > PI { 1 } else { 0 };
    let large_tgt = if (tgt_e - tgt_s).abs() > PI { 1 } else { 0 };

    format!(
        "M{x_ss},{y_ss} \
         A{radius},{radius} 0 {large_src},1 {x_se},{y_se} \
         Q{cx},{cy} {x_ts},{y_ts} \
         A{radius},{radius} 0 {large_tgt},1 {x_te},{y_te} \
         Q{cx},{cy} {x_ss},{y_ss} Z"
    )
}

// ── Rendering ───────────────────────────────────────────────────

/// Render the chord diagram as an SVG string.
pub fn render_chord_diagram(
    groups: &[ChordGroup],
    chords: &[Chord],
    cfg: &ChordConfig,
) -> String {
    let (cx, cy) = cfg.center();
    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    // Chords
    svg.push_str("<g class=\"chords\">");
    for chord in chords {
        let color = &groups[chord.source_index].color;
        let opacity = cfg.chord_opacity;
        let path = chord_path(chord, cfg.inner_radius, cx, cy);
        let _ = write!(
            svg,
            "<path d=\"{path}\" fill=\"{color}\" fill-opacity=\"{opacity}\" stroke=\"none\" />"
        );
    }
    svg.push_str("</g>");

    // Group arcs
    svg.push_str("<g class=\"groups\">");
    for g in groups {
        let path = arc_path(g.start_angle, g.end_angle, cfg.inner_radius, cfg.outer_radius, cx, cy);
        let _ = write!(
            svg,
            "<path d=\"{path}\" fill=\"{}\" stroke=\"white\" stroke-width=\"1\" />",
            g.color
        );

        // Label
        let mid = g.mid_angle();
        let label_r = cfg.outer_radius + 14.0;
        let lx = cx + label_r * mid.cos();
        let ly = cy + label_r * mid.sin();
        let anchor = if mid.cos() < -0.01 {
            "end"
        } else if mid.cos() > 0.01 {
            "start"
        } else {
            "middle"
        };
        let fs = cfg.font_size;
        let _ = write!(
            svg,
            "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" \
             text-anchor=\"{anchor}\" dominant-baseline=\"middle\">{}</text>",
            g.label
        );
    }
    svg.push_str("</g>");

    svg.push_str("</svg>");
    svg
}

/// Convenience: compute layout + render.
pub fn chord_diagram(
    matrix: &[Vec<f64>],
    groups: &mut [ChordGroup],
    cfg: &ChordConfig,
) -> String {
    let chords = compute_layout(matrix, groups, cfg);
    render_chord_diagram(groups, &chords, cfg)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_groups() -> Vec<ChordGroup> {
        vec![
            ChordGroup::new("Alpha", "steelblue"),
            ChordGroup::new("Beta", "coral"),
            ChordGroup::new("Gamma", "mediumseagreen"),
        ]
    }

    fn sample_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 10.0, 5.0],
            vec![10.0, 0.0, 15.0],
            vec![5.0, 15.0, 0.0],
        ]
    }

    #[test]
    fn group_new() {
        let g = ChordGroup::new("Test", "red");
        assert_eq!(g.label, "Test");
        assert_eq!(g.color, "red");
        assert_eq!(g.value, 0.0);
    }

    #[test]
    fn group_arc_span() {
        let mut g = ChordGroup::new("X", "blue");
        g.start_angle = 0.5;
        g.end_angle = 1.5;
        assert!((g.arc_span() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn group_mid_angle() {
        let mut g = ChordGroup::new("X", "blue");
        g.start_angle = 1.0;
        g.end_angle = 3.0;
        assert!((g.mid_angle() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn config_default() {
        let cfg = ChordConfig::default();
        assert!(cfg.width > 0.0);
        assert!(cfg.outer_radius > cfg.inner_radius);
        assert!(cfg.pad_angle > 0.0);
    }

    #[test]
    fn config_center() {
        let cfg = ChordConfig::default();
        let (cx, cy) = cfg.center();
        assert!((cx - 250.0).abs() < 1e-9);
        assert!((cy - 250.0).abs() < 1e-9);
    }

    #[test]
    fn compute_layout_groups_have_values() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let _ = compute_layout(&matrix, &mut groups, &cfg);
        assert!((groups[0].value - 15.0).abs() < 1e-9); // 0+10+5
        assert!((groups[1].value - 25.0).abs() < 1e-9); // 10+0+15
        assert!((groups[2].value - 20.0).abs() < 1e-9); // 5+15+0
    }

    #[test]
    fn compute_layout_angles_span_circle() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let _ = compute_layout(&matrix, &mut groups, &cfg);
        let last = groups.last().unwrap();
        // Last group end + accumulated padding should be close to 2*PI
        let total_span: f64 = groups.iter().map(|g| g.arc_span()).sum();
        let total_pad = cfg.pad_angle * groups.len() as f64;
        assert!((total_span + total_pad - 2.0 * PI).abs() < 0.01);
    }

    #[test]
    fn compute_layout_produces_chords() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let chords = compute_layout(&matrix, &mut groups, &cfg);
        // 6 non-zero entries in the matrix
        assert_eq!(chords.len(), 6);
    }

    #[test]
    fn chord_has_valid_angles() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let chords = compute_layout(&matrix, &mut groups, &cfg);
        for c in &chords {
            assert!(c.source_end >= c.source_start);
            assert!(c.target_end >= c.target_start);
        }
    }

    #[test]
    fn render_produces_svg() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let svg = chord_diagram(&matrix, &mut groups, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_contains_group_labels() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let svg = chord_diagram(&matrix, &mut groups, &cfg);
        assert!(svg.contains("Alpha"));
        assert!(svg.contains("Beta"));
        assert!(svg.contains("Gamma"));
    }

    #[test]
    fn render_contains_paths() {
        let mut groups = sample_groups();
        let matrix = sample_matrix();
        let cfg = ChordConfig::default();
        let svg = chord_diagram(&matrix, &mut groups, &cfg);
        assert!(svg.contains("<path"));
    }

    #[test]
    fn identity_matrix_no_chords() {
        let mut groups = vec![
            ChordGroup::new("A", "red"),
            ChordGroup::new("B", "blue"),
        ];
        let matrix = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let cfg = ChordConfig::default();
        let chords = compute_layout(&matrix, &mut groups, &cfg);
        assert!(chords.is_empty());
    }

    #[test]
    fn single_group_self_loop() {
        let mut groups = vec![ChordGroup::new("Self", "green")];
        let matrix = vec![vec![100.0]];
        let cfg = ChordConfig::default();
        let chords = compute_layout(&matrix, &mut groups, &cfg);
        assert_eq!(chords.len(), 1);
        assert_eq!(chords[0].source_index, 0);
        assert_eq!(chords[0].target_index, 0);
    }

    #[test]
    fn arc_path_short() {
        let path = arc_path(0.0, 0.5, 100.0, 120.0, 250.0, 250.0);
        assert!(path.contains("M"));
        assert!(path.contains("A"));
    }

    #[test]
    fn arc_path_large_arc() {
        // Span > PI triggers large-arc flag
        let path = arc_path(0.0, 4.0, 100.0, 120.0, 250.0, 250.0);
        assert!(path.contains(" 1,1 ")); // large-arc-flag = 1
    }
}
