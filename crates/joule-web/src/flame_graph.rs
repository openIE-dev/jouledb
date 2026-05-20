//! Flame graph data model — stack frame recording, frame collapsing,
//! flame graph builder, SVG output concepts, folded stack format,
//! and top functions by self time.
//!
//! Replaces `flamegraph` / `inferno` with a pure-Rust implementation that tracks
//! sample stacks, merges identical prefixes into a tree, computes self/total time,
//! exports folded-stack format, and generates proportional SVG rectangles.

use std::collections::HashMap;

// ── Stack Frame ──────────────────────────────────────────────────

/// A single frame in a call stack.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackFrame {
    pub name: String,
    pub file: String,
    pub line: u32,
}

impl StackFrame {
    pub fn new(name: &str, file: &str, line: u32) -> Self {
        Self {
            name: name.to_string(),
            file: file.to_string(),
            line,
        }
    }

    /// Short display: "name (file:line)".
    pub fn display(&self) -> String {
        format!("{} ({}:{})", self.name, self.file, self.line)
    }
}

// ── Sample ───────────────────────────────────────────────────────

/// A profiling sample: a stack trace with a hit count.
#[derive(Debug, Clone)]
pub struct Sample {
    pub stack: Vec<StackFrame>,
    pub count: u64,
}

impl Sample {
    pub fn new(stack: Vec<StackFrame>, count: u64) -> Self {
        Self { stack, count }
    }

    /// Return the leaf (top-of-stack) frame, if any.
    pub fn leaf(&self) -> Option<&StackFrame> {
        self.stack.last()
    }

    /// Convert to folded-stack format: "frame1;frame2;frame3 count".
    pub fn to_folded(&self) -> String {
        let path: Vec<&str> = self.stack.iter().map(|f| f.name.as_str()).collect();
        format!("{} {}", path.join(";"), self.count)
    }
}

// ── Flame Node ───────────────────────────────────────────────────

/// A node in the flame tree.
#[derive(Debug, Clone)]
pub struct FlameNode {
    pub name: String,
    pub self_count: u64,
    pub total_count: u64,
    pub children: Vec<FlameNode>,
    pub depth: usize,
}

impl FlameNode {
    fn new(name: &str, depth: usize) -> Self {
        Self {
            name: name.to_string(),
            self_count: 0,
            total_count: 0,
            children: Vec::new(),
            depth,
        }
    }

    /// Find or create a child with the given name.
    fn get_or_insert_child(&mut self, name: &str, depth: usize) -> &mut FlameNode {
        let pos = self.children.iter().position(|c| c.name == name);
        match pos {
            Some(i) => &mut self.children[i],
            None => {
                self.children.push(FlameNode::new(name, depth));
                self.children.last_mut().unwrap()
            }
        }
    }

    /// Maximum depth in the subtree.
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth)
        }
    }

    /// Search for nodes matching a name substring.
    pub fn search(&self, query: &str) -> Vec<&FlameNode> {
        let mut results = Vec::new();
        if self.name.contains(query) {
            results.push(self);
        }
        for child in &self.children {
            results.extend(child.search(query));
        }
        results
    }

    /// Filter: return a new tree keeping only nodes matching the query
    /// (and their ancestors).
    pub fn filter(&self, query: &str) -> Option<FlameNode> {
        let child_matches: Vec<FlameNode> =
            self.children.iter().filter_map(|c| c.filter(query)).collect();

        if self.name.contains(query) || !child_matches.is_empty() {
            let mut node = FlameNode::new(&self.name, self.depth);
            node.self_count = self.self_count;
            node.total_count = self.total_count;
            node.children = child_matches;
            Some(node)
        } else {
            None
        }
    }

    /// Count total nodes in the subtree (including self).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Collect all leaf nodes (no children).
    pub fn leaves(&self) -> Vec<&FlameNode> {
        if self.children.is_empty() {
            vec![self]
        } else {
            self.children.iter().flat_map(|c| c.leaves()).collect()
        }
    }

    /// Self-time percentage relative to root total.
    pub fn self_percentage(&self, root_total: u64) -> f64 {
        if root_total == 0 {
            return 0.0;
        }
        (self.self_count as f64 / root_total as f64) * 100.0
    }

    /// Total-time percentage relative to root total.
    pub fn total_percentage(&self, root_total: u64) -> f64 {
        if root_total == 0 {
            return 0.0;
        }
        (self.total_count as f64 / root_total as f64) * 100.0
    }
}

// ── Top Function ─────────────────────────────────────────────────

/// A function's aggregated self-time across the entire flame tree.
#[derive(Debug, Clone)]
pub struct TopFunction {
    pub name: String,
    pub self_count: u64,
    pub total_count: u64,
    pub percentage: f64,
}

// ── Collapsed Stack ──────────────────────────────────────────────

/// A collapsed stack entry: semicolon-separated frame names with a count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollapsedStack {
    pub path: String,
    pub count: u64,
}

// ── Build Flame Tree ─────────────────────────────────────────────

/// Build a flame tree from a list of samples by merging matching stack prefixes.
pub fn build_flame_tree(samples: &[Sample]) -> FlameNode {
    let mut root = FlameNode::new("(root)", 0);

    for sample in samples {
        let mut current = &mut root;
        current.total_count += sample.count;

        for (i, frame) in sample.stack.iter().enumerate() {
            let depth = i + 1;
            current = current.get_or_insert_child(&frame.name, depth);
            current.total_count += sample.count;
        }
        // The leaf frame gets the self count
        current.self_count += sample.count;
    }

    root
}

/// Build an inverted (icicle) flame tree: stacks are reversed so that
/// leaf functions appear at the top.
pub fn build_inverted_tree(samples: &[Sample]) -> FlameNode {
    let inverted: Vec<Sample> = samples
        .iter()
        .map(|s| {
            let mut reversed = s.stack.clone();
            reversed.reverse();
            Sample::new(reversed, s.count)
        })
        .collect();
    build_flame_tree(&inverted)
}

// ── Frame Collapsing ─────────────────────────────────────────────

/// Collapse samples into folded stack format (as used by FlameGraph tools).
/// Merges samples with identical stacks and sums their counts.
pub fn collapse_stacks(samples: &[Sample]) -> Vec<CollapsedStack> {
    let mut map: HashMap<String, u64> = HashMap::new();
    for sample in samples {
        let path: Vec<&str> = sample.stack.iter().map(|f| f.name.as_str()).collect();
        let key = path.join(";");
        *map.entry(key).or_insert(0) += sample.count;
    }

    let mut result: Vec<CollapsedStack> = map
        .into_iter()
        .map(|(path, count)| CollapsedStack { path, count })
        .collect();
    result.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.path.cmp(&b.path)));
    result
}

/// Parse folded stack format lines into samples.
pub fn parse_folded(text: &str) -> Vec<Sample> {
    let mut samples = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(space_pos) = line.rfind(' ') {
            let path = &line[..space_pos];
            if let Ok(count) = line[space_pos + 1..].parse::<u64>() {
                let frames: Vec<StackFrame> = path
                    .split(';')
                    .map(|name| StackFrame::new(name, "", 0))
                    .collect();
                samples.push(Sample::new(frames, count));
            }
        }
    }
    samples
}

// ── Top Functions ────────────────────────────────────────────────

/// Collect top functions by self time from the flame tree.
fn collect_self_times(node: &FlameNode, map: &mut HashMap<String, (u64, u64)>) {
    if node.name != "(root)" {
        let entry = map.entry(node.name.clone()).or_insert((0, 0));
        entry.0 += node.self_count;
        entry.1 += node.total_count;
    }
    for child in &node.children {
        collect_self_times(child, map);
    }
}

/// Return the top N functions by self time.
pub fn top_functions(root: &FlameNode, n: usize) -> Vec<TopFunction> {
    let mut map: HashMap<String, (u64, u64)> = HashMap::new();
    collect_self_times(root, &mut map);

    let root_total = root.total_count;
    let mut funcs: Vec<TopFunction> = map
        .into_iter()
        .map(|(name, (self_count, total_count))| {
            let percentage = if root_total > 0 {
                (self_count as f64 / root_total as f64) * 100.0
            } else {
                0.0
            };
            TopFunction {
                name,
                self_count,
                total_count,
                percentage,
            }
        })
        .collect();

    funcs.sort_by(|a, b| b.self_count.cmp(&a.self_count).then_with(|| a.name.cmp(&b.name)));
    funcs.truncate(n);
    funcs
}

// ── SVG Generation ───────────────────────────────────────────────

/// A single SVG rectangle for rendering.
#[derive(Debug, Clone)]
pub struct SvgRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub name: String,
    pub count: u64,
    pub depth: usize,
}

/// Configuration for SVG rendering.
#[derive(Debug, Clone)]
pub struct SvgConfig {
    pub total_width: f64,
    pub row_height: f64,
    pub min_width: f64,
}

impl Default for SvgConfig {
    fn default() -> Self {
        Self {
            total_width: 1200.0,
            row_height: 20.0,
            min_width: 1.0,
        }
    }
}

/// Generate SVG rects from a flame tree.
pub fn generate_svg_rects(root: &FlameNode, config: &SvgConfig) -> Vec<SvgRect> {
    let mut rects = Vec::new();
    if root.total_count == 0 {
        return rects;
    }
    emit_rects(root, 0.0, config.total_width, config, &mut rects);
    rects
}

fn emit_rects(
    node: &FlameNode,
    x: f64,
    width: f64,
    config: &SvgConfig,
    rects: &mut Vec<SvgRect>,
) {
    if width < config.min_width {
        return;
    }

    let y = node.depth as f64 * config.row_height;

    if node.name != "(root)" {
        rects.push(SvgRect {
            x,
            y,
            width,
            height: config.row_height,
            name: node.name.clone(),
            count: node.total_count,
            depth: node.depth,
        });
    }

    let parent_count = node.total_count as f64;
    if parent_count == 0.0 {
        return;
    }

    let mut child_x = x;
    for child in &node.children {
        let child_width = (child.total_count as f64 / parent_count) * width;
        emit_rects(child, child_x, child_width, config, rects);
        child_x += child_width;
    }
}

/// Render SVG rects to an SVG string.
pub fn render_svg(rects: &[SvgRect], config: &SvgConfig, max_depth: usize) -> String {
    let svg_height = (max_depth + 1) as f64 * config.row_height;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">",
        config.total_width, svg_height
    );
    svg.push('\n');

    for rect in rects {
        let hue = (rect.name.bytes().fold(0u32, |a, b| a.wrapping_add(b as u32)) % 60) + 10;
        svg.push_str(&format!(
            "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"hsl({}, 70%, 55%)\" />",
            rect.x, rect.y, rect.width, rect.height, hue
        ));
        svg.push('\n');
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\">{} ({})</text>",
            rect.x + 2.0,
            rect.y + 14.0,
            rect.name,
            rect.count
        ));
        svg.push('\n');
    }

    svg.push_str("</svg>");
    svg
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples() -> Vec<Sample> {
        vec![
            Sample::new(
                vec![
                    StackFrame::new("main", "main.rs", 1),
                    StackFrame::new("process", "proc.rs", 10),
                    StackFrame::new("compute", "math.rs", 20),
                ],
                5,
            ),
            Sample::new(
                vec![
                    StackFrame::new("main", "main.rs", 1),
                    StackFrame::new("process", "proc.rs", 10),
                    StackFrame::new("allocate", "alloc.rs", 5),
                ],
                3,
            ),
            Sample::new(
                vec![
                    StackFrame::new("main", "main.rs", 1),
                    StackFrame::new("io_wait", "io.rs", 50),
                ],
                2,
            ),
        ]
    }

    #[test]
    fn test_build_flame_tree_root() {
        let tree = build_flame_tree(&make_samples());
        assert_eq!(tree.name, "(root)");
        assert_eq!(tree.total_count, 10);
    }

    #[test]
    fn test_flame_tree_children() {
        let tree = build_flame_tree(&make_samples());
        assert_eq!(tree.children.len(), 1);
        let main_node = &tree.children[0];
        assert_eq!(main_node.name, "main");
        assert_eq!(main_node.total_count, 10);
        assert_eq!(main_node.children.len(), 2);
    }

    #[test]
    fn test_flame_tree_merge() {
        let tree = build_flame_tree(&make_samples());
        let main_node = &tree.children[0];
        let process = main_node
            .children
            .iter()
            .find(|c| c.name == "process")
            .unwrap();
        assert_eq!(process.total_count, 8);
        assert_eq!(process.children.len(), 2);
    }

    #[test]
    fn test_self_count() {
        let tree = build_flame_tree(&make_samples());
        let main_node = &tree.children[0];
        let process = main_node
            .children
            .iter()
            .find(|c| c.name == "process")
            .unwrap();
        let compute = process
            .children
            .iter()
            .find(|c| c.name == "compute")
            .unwrap();
        assert_eq!(compute.self_count, 5);
        assert_eq!(compute.total_count, 5);
    }

    #[test]
    fn test_max_depth() {
        let tree = build_flame_tree(&make_samples());
        assert_eq!(tree.max_depth(), 3);
    }

    #[test]
    fn test_search() {
        let tree = build_flame_tree(&make_samples());
        let results = tree.search("compute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "compute");
    }

    #[test]
    fn test_search_no_match() {
        let tree = build_flame_tree(&make_samples());
        let results = tree.search("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter() {
        let tree = build_flame_tree(&make_samples());
        let filtered = tree.filter("compute").unwrap();
        assert_eq!(filtered.name, "(root)");
        assert_eq!(filtered.children.len(), 1);
        let main_f = &filtered.children[0];
        assert_eq!(main_f.children.len(), 1);
    }

    #[test]
    fn test_inverted_tree() {
        let tree = build_inverted_tree(&make_samples());
        assert_eq!(tree.name, "(root)");
        let child_names: Vec<&str> = tree.children.iter().map(|c| c.name.as_str()).collect();
        assert!(child_names.contains(&"compute"));
        assert!(child_names.contains(&"allocate"));
        assert!(child_names.contains(&"io_wait"));
    }

    #[test]
    fn test_svg_rect_generation() {
        let tree = build_flame_tree(&make_samples());
        let config = SvgConfig::default();
        let rects = generate_svg_rects(&tree, &config);
        assert_eq!(rects.len(), 5);
        let main_rect = rects.iter().find(|r| r.name == "main").unwrap();
        assert!((main_rect.width - config.total_width).abs() < 0.01);
    }

    #[test]
    fn test_svg_rect_proportional_width() {
        let tree = build_flame_tree(&make_samples());
        let config = SvgConfig {
            total_width: 1000.0,
            ..Default::default()
        };
        let rects = generate_svg_rects(&tree, &config);

        let process_rect = rects.iter().find(|r| r.name == "process").unwrap();
        let io_rect = rects.iter().find(|r| r.name == "io_wait").unwrap();
        assert!((process_rect.width - 800.0).abs() < 0.01);
        assert!((io_rect.width - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_svg_depth_positions() {
        let tree = build_flame_tree(&make_samples());
        let config = SvgConfig {
            row_height: 20.0,
            ..Default::default()
        };
        let rects = generate_svg_rects(&tree, &config);

        let main_rect = rects.iter().find(|r| r.name == "main").unwrap();
        assert_eq!(main_rect.depth, 1);
        assert!((main_rect.y - 20.0).abs() < 0.01);

        let compute_rect = rects.iter().find(|r| r.name == "compute").unwrap();
        assert_eq!(compute_rect.depth, 3);
        assert!((compute_rect.y - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_render_svg_string() {
        let tree = build_flame_tree(&make_samples());
        let config = SvgConfig::default();
        let rects = generate_svg_rects(&tree, &config);
        let svg = render_svg(&rects, &config, tree.max_depth());
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("rect"));
        assert!(svg.contains("text"));
    }

    #[test]
    fn test_empty_samples() {
        let tree = build_flame_tree(&[]);
        assert_eq!(tree.total_count, 0);
        assert!(tree.children.is_empty());
        let rects = generate_svg_rects(&tree, &SvgConfig::default());
        assert!(rects.is_empty());
    }

    #[test]
    fn test_collapse_stacks() {
        let collapsed = collapse_stacks(&make_samples());
        assert_eq!(collapsed.len(), 3);
        // Sorted by count descending
        assert_eq!(collapsed[0].count, 5);
        assert!(collapsed[0].path.contains("compute"));
    }

    #[test]
    fn test_collapse_merges_identical() {
        let samples = vec![
            Sample::new(
                vec![StackFrame::new("a", "", 0), StackFrame::new("b", "", 0)],
                3,
            ),
            Sample::new(
                vec![StackFrame::new("a", "", 0), StackFrame::new("b", "", 0)],
                7,
            ),
        ];
        let collapsed = collapse_stacks(&samples);
        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].count, 10);
        assert_eq!(collapsed[0].path, "a;b");
    }

    #[test]
    fn test_parse_folded() {
        let text = "main;process;compute 5\nmain;io_wait 2\n";
        let samples = parse_folded(text);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].stack.len(), 3);
        assert_eq!(samples[0].count, 5);
        assert_eq!(samples[1].stack[1].name, "io_wait");
    }

    #[test]
    fn test_folded_roundtrip() {
        let samples = make_samples();
        let collapsed = collapse_stacks(&samples);
        let text: String = collapsed
            .iter()
            .map(|c| format!("{} {}", c.path, c.count))
            .collect::<Vec<_>>()
            .join("\n");
        let parsed = parse_folded(&text);
        assert_eq!(parsed.len(), collapsed.len());
    }

    #[test]
    fn test_top_functions() {
        let tree = build_flame_tree(&make_samples());
        let top = top_functions(&tree, 3);
        assert_eq!(top[0].name, "compute");
        assert_eq!(top[0].self_count, 5);
        assert!((top[0].percentage - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_top_functions_limit() {
        let tree = build_flame_tree(&make_samples());
        let top = top_functions(&tree, 1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_node_count() {
        let tree = build_flame_tree(&make_samples());
        // root -> main -> {process -> {compute, allocate}, io_wait} = 6
        assert_eq!(tree.node_count(), 6);
    }

    #[test]
    fn test_leaves() {
        let tree = build_flame_tree(&make_samples());
        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 3);
        let leaf_names: Vec<&str> = leaves.iter().map(|l| l.name.as_str()).collect();
        assert!(leaf_names.contains(&"compute"));
        assert!(leaf_names.contains(&"allocate"));
        assert!(leaf_names.contains(&"io_wait"));
    }

    #[test]
    fn test_self_percentage() {
        let tree = build_flame_tree(&make_samples());
        let main_node = &tree.children[0];
        let process = main_node
            .children
            .iter()
            .find(|c| c.name == "process")
            .unwrap();
        let compute = process
            .children
            .iter()
            .find(|c| c.name == "compute")
            .unwrap();
        assert!((compute.self_percentage(10) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_total_percentage() {
        let tree = build_flame_tree(&make_samples());
        let main_node = &tree.children[0];
        assert!((main_node.total_percentage(10) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_sample_leaf() {
        let s = Sample::new(
            vec![
                StackFrame::new("a", "", 0),
                StackFrame::new("b", "", 0),
            ],
            1,
        );
        assert_eq!(s.leaf().unwrap().name, "b");
    }

    #[test]
    fn test_sample_to_folded() {
        let s = Sample::new(
            vec![
                StackFrame::new("main", "m.rs", 1),
                StackFrame::new("run", "r.rs", 5),
            ],
            42,
        );
        assert_eq!(s.to_folded(), "main;run 42");
    }

    #[test]
    fn test_stack_frame_display() {
        let f = StackFrame::new("compute", "math.rs", 20);
        assert_eq!(f.display(), "compute (math.rs:20)");
    }

    #[test]
    fn test_parse_folded_ignores_blanks() {
        let text = "\n  \nmain;run 10\n\n";
        let samples = parse_folded(text);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn test_percentage_zero_root() {
        let node = FlameNode::new("test", 0);
        assert!((node.self_percentage(0) - 0.0).abs() < 0.001);
        assert!((node.total_percentage(0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_single_frame_sample() {
        let samples = vec![Sample::new(vec![StackFrame::new("only", "", 0)], 1)];
        let tree = build_flame_tree(&samples);
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].self_count, 1);
        assert_eq!(tree.children[0].total_count, 1);
    }
}
