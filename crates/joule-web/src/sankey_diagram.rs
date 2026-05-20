//! Sankey flow diagram with node/link layout, flow-proportional widths, and SVG
//! output.  Replaces d3-sankey with a pure-Rust implementation suitable for WASM
//! and server-side rendering.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A node in the Sankey diagram.
#[derive(Debug, Clone)]
pub struct SankeyNode {
    pub id: String,
    pub label: String,
    pub color: String,
    /// Computed: total value flowing through this node.
    pub value: f64,
    /// Computed: column assignment (0 = leftmost).
    pub column: usize,
    /// Computed: y-position of top edge in pixels.
    pub y: f64,
    /// Computed: height in pixels (proportional to value).
    pub height: f64,
}

impl SankeyNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>, color: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            color: color.into(),
            value: 0.0,
            column: 0,
            y: 0.0,
            height: 0.0,
        }
    }

    /// The x-position of the left edge for a given column width and gap.
    pub fn x(&self, node_width: f64, col_gap: f64) -> f64 {
        self.column as f64 * (node_width + col_gap)
    }

    /// Centre y of the node.
    pub fn center_y(&self) -> f64 {
        self.y + self.height / 2.0
    }
}

/// A directed link (flow) between two nodes.
#[derive(Debug, Clone)]
pub struct SankeyLink {
    pub source_id: String,
    pub target_id: String,
    pub value: f64,
    /// Computed: y-position at source node (top of band).
    pub source_y: f64,
    /// Computed: y-position at target node (top of band).
    pub target_y: f64,
    /// Computed: band width in pixels.
    pub width: f64,
}

impl SankeyLink {
    pub fn new(source: impl Into<String>, target: impl Into<String>, value: f64) -> Self {
        Self {
            source_id: source.into(),
            target_id: target.into(),
            value: value.max(0.0),
            source_y: 0.0,
            target_y: 0.0,
            width: 0.0,
        }
    }
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the Sankey layout and rendering.
#[derive(Debug, Clone)]
pub struct SankeyConfig {
    pub width: f64,
    pub height: f64,
    pub node_width: f64,
    pub node_padding: f64,
    pub padding_top: f64,
    pub padding_left: f64,
    pub font_size: f64,
    pub link_opacity: f64,
}

impl Default for SankeyConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 400.0,
            node_width: 20.0,
            node_padding: 10.0,
            padding_top: 20.0,
            padding_left: 20.0,
            font_size: 11.0,
            link_opacity: 0.4,
        }
    }
}

// ── Layout engine ───────────────────────────────────────────────

/// Compute column assignments via topological depth from sources.
fn assign_columns(nodes: &mut [SankeyNode], links: &[SankeyLink]) {
    // Build adjacency using owned Strings so we don't borrow `nodes`
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    let mut has_incoming: HashMap<String, bool> = HashMap::new();
    for n in nodes.iter() {
        adj.entry(n.id.clone()).or_default();
        has_incoming.entry(n.id.clone()).or_insert(false);
    }
    for l in links {
        adj.entry(l.source_id.clone())
            .or_default()
            .push(l.target_id.clone());
        has_incoming.insert(l.target_id.clone(), true);
    }

    // BFS from sources
    let mut depth: HashMap<String, usize> = HashMap::new();
    let mut queue: Vec<String> = Vec::new();
    for n in nodes.iter() {
        if !has_incoming[n.id.as_str()] {
            depth.insert(n.id.clone(), 0);
            queue.push(n.id.clone());
        }
    }
    // Nodes with no edges at all get column 0
    for n in nodes.iter() {
        depth.entry(n.id.clone()).or_insert(0);
    }

    let mut idx = 0;
    while idx < queue.len() {
        let cur = queue[idx].clone();
        idx += 1;
        let cur_depth = depth[cur.as_str()];
        if let Some(targets) = adj.get(cur.as_str()) {
            let targets_owned: Vec<String> = targets.clone();
            for t in targets_owned {
                let new_d = cur_depth + 1;
                let entry = depth.entry(t.clone()).or_insert(0);
                if new_d > *entry {
                    *entry = new_d;
                }
                if !queue.contains(&t) {
                    queue.push(t);
                }
            }
        }
    }

    for n in nodes.iter_mut() {
        n.column = depth.get(n.id.as_str()).copied().unwrap_or(0);
    }
}

/// Compute node values from link sums (max of incoming / outgoing).
fn compute_node_values(nodes: &mut [SankeyNode], links: &[SankeyLink]) {
    let mut outgoing: HashMap<&str, f64> = HashMap::new();
    let mut incoming: HashMap<&str, f64> = HashMap::new();
    for l in links {
        *outgoing.entry(l.source_id.as_str()).or_insert(0.0) += l.value;
        *incoming.entry(l.target_id.as_str()).or_insert(0.0) += l.value;
    }
    for n in nodes.iter_mut() {
        let out_v = outgoing.get(n.id.as_str()).copied().unwrap_or(0.0);
        let in_v = incoming.get(n.id.as_str()).copied().unwrap_or(0.0);
        n.value = out_v.max(in_v).max(0.0);
    }
}

/// Position nodes vertically within their columns.
fn position_nodes(nodes: &mut [SankeyNode], cfg: &SankeyConfig) {
    let max_col = nodes.iter().map(|n| n.column).max().unwrap_or(0);
    let usable_height = cfg.height - 2.0 * cfg.padding_top;

    for col in 0..=max_col {
        let mut col_nodes: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.column == col)
            .map(|(i, _)| i)
            .collect();

        let total_value: f64 = col_nodes.iter().map(|i| nodes[*i].value).sum();
        let total_padding = cfg.node_padding * (col_nodes.len().saturating_sub(1)) as f64;
        let available = (usable_height - total_padding).max(1.0);

        // Sort by value descending for stable layout
        col_nodes.sort_by(|&a, &b| {
            nodes[b]
                .value
                .partial_cmp(&nodes[a].value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut y = cfg.padding_top;
        let scale = if total_value > 0.0 {
            available / total_value
        } else {
            0.0
        };
        for &i in &col_nodes {
            nodes[i].height = (nodes[i].value * scale).max(2.0);
            nodes[i].y = y;
            y += nodes[i].height + cfg.node_padding;
        }
    }
}

/// Route links: assign source_y, target_y, and width.
fn route_links(nodes: &[SankeyNode], links: &mut [SankeyLink], cfg: &SankeyConfig) {
    let usable_height = cfg.height - 2.0 * cfg.padding_top;
    let node_map: HashMap<&str, &SankeyNode> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Track running y offsets per node for stacking links
    let mut source_offsets: HashMap<String, f64> = HashMap::new();
    let mut target_offsets: HashMap<String, f64> = HashMap::new();

    // Sort links by value descending for consistent layout
    let mut indices: Vec<usize> = (0..links.len()).collect();
    indices.sort_by(|&a, &b| {
        links[b]
            .value
            .partial_cmp(&links[a].value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_value: f64 = nodes.iter().map(|n| n.value).fold(0.0_f64, f64::max);
    let scale = if max_value > 0.0 {
        usable_height / max_value
    } else {
        0.0
    };

    for &i in &indices {
        // Extract all needed data from the link into locals to avoid borrow conflicts
        let link_src = links[i].source_id.clone();
        let link_tgt = links[i].target_id.clone();
        let link_value = links[i].value;

        let src = node_map.get(link_src.as_str());
        let tgt = node_map.get(link_tgt.as_str());

        if let (Some(src_node), Some(tgt_node)) = (src, tgt) {
            let src_value = src_node.value;
            let src_y = src_node.y;
            let src_height = src_node.height;
            let tgt_y = tgt_node.y;

            let band_width = if src_value > 0.0 {
                (link_value / src_value) * src_height
            } else {
                2.0
            };

            let src_off = source_offsets.entry(link_src).or_insert(0.0);
            let sy = src_y + *src_off;
            *src_off += band_width;

            let tgt_off = target_offsets.entry(link_tgt).or_insert(0.0);
            let ty = tgt_y + *tgt_off;
            *tgt_off += band_width;

            links[i].source_y = sy;
            links[i].target_y = ty;
            links[i].width = band_width.max(1.0);
        }
    }
    let _ = (cfg, scale); // suppress unused
}

/// Run the full layout algorithm.
pub fn layout(nodes: &mut [SankeyNode], links: &mut [SankeyLink], cfg: &SankeyConfig) {
    compute_node_values(nodes, links);
    assign_columns(nodes, links);
    position_nodes(nodes, cfg);
    route_links(nodes, links, cfg);
}

// ── SVG rendering ───────────────────────────────────────────────

/// Generate a cubic-Bezier SVG path for a link band.
fn link_path(link: &SankeyLink, nodes: &[SankeyNode], cfg: &SankeyConfig) -> String {
    let node_map: HashMap<&str, &SankeyNode> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let src = node_map.get(link.source_id.as_str());
    let tgt = node_map.get(link.target_id.as_str());
    if src.is_none() || tgt.is_none() {
        return String::new();
    }
    let src = src.unwrap();
    let tgt = tgt.unwrap();

    let x0 = cfg.padding_left + src.x(cfg.node_width, col_gap(cfg)) + cfg.node_width;
    let x1 = cfg.padding_left + tgt.x(cfg.node_width, col_gap(cfg));
    let y0_top = link.source_y;
    let y0_bot = link.source_y + link.width;
    let y1_top = link.target_y;
    let y1_bot = link.target_y + link.width;

    let mx = (x0 + x1) / 2.0;

    format!(
        "M{x0},{y0_top} C{mx},{y0_top} {mx},{y1_top} {x1},{y1_top} \
         L{x1},{y1_bot} C{mx},{y1_bot} {mx},{y0_bot} {x0},{y0_bot} Z"
    )
}

/// Compute the gap between columns.
fn col_gap(cfg: &SankeyConfig) -> f64 {
    let max_cols = 10; // will be adjusted dynamically
    let usable_w = cfg.width - 2.0 * cfg.padding_left;
    (usable_w - cfg.node_width) / max_cols as f64
}

/// Compute the actual column gap based on data.
fn actual_col_gap(nodes: &[SankeyNode], cfg: &SankeyConfig) -> f64 {
    let max_col = nodes.iter().map(|n| n.column).max().unwrap_or(0);
    if max_col == 0 {
        return 0.0;
    }
    let usable_w = cfg.width - 2.0 * cfg.padding_left - cfg.node_width;
    usable_w / max_col as f64
}

/// Render the complete Sankey diagram as an SVG string.
pub fn render_sankey(
    nodes: &[SankeyNode],
    links: &[SankeyLink],
    cfg: &SankeyConfig,
) -> String {
    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    let gap = actual_col_gap(nodes, cfg);
    let node_map: HashMap<&str, &SankeyNode> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Links
    svg.push_str("<g class=\"sankey-links\">");
    for link in links {
        let src = node_map.get(link.source_id.as_str());
        let tgt = node_map.get(link.target_id.as_str());
        if src.is_none() || tgt.is_none() {
            continue;
        }
        let src = src.unwrap();
        let tgt = tgt.unwrap();

        let x0 = cfg.padding_left + src.x(cfg.node_width, gap) + cfg.node_width;
        let x1 = cfg.padding_left + tgt.x(cfg.node_width, gap);
        let y0_top = link.source_y;
        let y0_bot = link.source_y + link.width;
        let y1_top = link.target_y;
        let y1_bot = link.target_y + link.width;
        let mx = (x0 + x1) / 2.0;
        let opacity = cfg.link_opacity;

        let _ = write!(
            svg,
            "<path d=\"M{x0},{y0_top} C{mx},{y0_top} {mx},{y1_top} {x1},{y1_top} \
             L{x1},{y1_bot} C{mx},{y1_bot} {mx},{y0_bot} {x0},{y0_bot} Z\" \
             fill=\"{}\" fill-opacity=\"{opacity}\" stroke=\"none\" />",
            src.color
        );
    }
    svg.push_str("</g>");

    // Nodes
    svg.push_str("<g class=\"sankey-nodes\">");
    for node in nodes {
        let x = cfg.padding_left + node.x(cfg.node_width, gap);
        let y = node.y;
        let w = cfg.node_width;
        let h = node.height;
        let fs = cfg.font_size;
        let lx = x + w + 4.0;
        let ly = node.center_y();

        let _ = write!(
            svg,
            "<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"{}\" />",
            node.color
        );
        let _ = write!(
            svg,
            "<text x=\"{lx}\" y=\"{ly}\" font-size=\"{fs}\" \
             dominant-baseline=\"middle\">{} ({:.0})</text>",
            node.label, node.value
        );
    }
    svg.push_str("</g>");

    svg.push_str("</svg>");
    svg
}

/// Convenience: layout + render in one call.
pub fn sankey_diagram(
    nodes: &mut [SankeyNode],
    links: &mut [SankeyLink],
    cfg: &SankeyConfig,
) -> String {
    layout(nodes, links, cfg);
    render_sankey(nodes, links, cfg)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nodes() -> Vec<SankeyNode> {
        vec![
            SankeyNode::new("a", "Source A", "steelblue"),
            SankeyNode::new("b", "Source B", "coral"),
            SankeyNode::new("c", "Middle", "mediumseagreen"),
            SankeyNode::new("d", "Sink", "goldenrod"),
        ]
    }

    fn sample_links() -> Vec<SankeyLink> {
        vec![
            SankeyLink::new("a", "c", 30.0),
            SankeyLink::new("b", "c", 20.0),
            SankeyLink::new("c", "d", 50.0),
        ]
    }

    #[test]
    fn node_new() {
        let n = SankeyNode::new("x", "X", "red");
        assert_eq!(n.id, "x");
        assert_eq!(n.label, "X");
        assert_eq!(n.color, "red");
        assert_eq!(n.value, 0.0);
    }

    #[test]
    fn link_new_clamps_negative() {
        let l = SankeyLink::new("a", "b", -10.0);
        assert_eq!(l.value, 0.0);
    }

    #[test]
    fn compute_values() {
        let mut nodes = sample_nodes();
        let links = sample_links();
        compute_node_values(&mut nodes, &links);
        let map: HashMap<&str, f64> = nodes.iter().map(|n| (n.id.as_str(), n.value)).collect();
        assert!((map["a"] - 30.0).abs() < 1e-9);
        assert!((map["b"] - 20.0).abs() < 1e-9);
        assert!((map["c"] - 50.0).abs() < 1e-9);
        assert!((map["d"] - 50.0).abs() < 1e-9);
    }

    #[test]
    fn columns_assigned() {
        let mut nodes = sample_nodes();
        let links = sample_links();
        compute_node_values(&mut nodes, &links);
        assign_columns(&mut nodes, &links);
        let map: HashMap<&str, usize> = nodes.iter().map(|n| (n.id.as_str(), n.column)).collect();
        assert_eq!(map["a"], 0);
        assert_eq!(map["b"], 0);
        assert_eq!(map["c"], 1);
        assert_eq!(map["d"], 2);
    }

    #[test]
    fn layout_positions_nodes() {
        let mut nodes = sample_nodes();
        let mut links = sample_links();
        let cfg = SankeyConfig::default();
        layout(&mut nodes, &mut links, &cfg);
        for n in &nodes {
            assert!(n.height > 0.0, "node {} should have height", n.id);
            assert!(n.y >= 0.0, "node {} should have y >= 0", n.id);
        }
    }

    #[test]
    fn link_routing_assigns_widths() {
        let mut nodes = sample_nodes();
        let mut links = sample_links();
        let cfg = SankeyConfig::default();
        layout(&mut nodes, &mut links, &cfg);
        for l in &links {
            assert!(l.width > 0.0, "link {}→{} width > 0", l.source_id, l.target_id);
        }
    }

    #[test]
    fn render_produces_svg() {
        let mut nodes = sample_nodes();
        let mut links = sample_links();
        let cfg = SankeyConfig::default();
        let svg = sankey_diagram(&mut nodes, &mut links, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_contains_nodes() {
        let mut nodes = sample_nodes();
        let mut links = sample_links();
        let cfg = SankeyConfig::default();
        let svg = sankey_diagram(&mut nodes, &mut links, &cfg);
        assert!(svg.contains("Source A"));
        assert!(svg.contains("Source B"));
        assert!(svg.contains("Sink"));
    }

    #[test]
    fn render_contains_paths() {
        let mut nodes = sample_nodes();
        let mut links = sample_links();
        let cfg = SankeyConfig::default();
        let svg = sankey_diagram(&mut nodes, &mut links, &cfg);
        assert!(svg.contains("<path"));
    }

    #[test]
    fn render_contains_rects() {
        let mut nodes = sample_nodes();
        let mut links = sample_links();
        let cfg = SankeyConfig::default();
        let svg = sankey_diagram(&mut nodes, &mut links, &cfg);
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn single_node_no_links() {
        let mut nodes = vec![SankeyNode::new("x", "Only", "red")];
        let mut links: Vec<SankeyLink> = vec![];
        let cfg = SankeyConfig::default();
        let svg = sankey_diagram(&mut nodes, &mut links, &cfg);
        assert!(svg.contains("Only"));
    }

    #[test]
    fn node_center_y() {
        let mut n = SankeyNode::new("a", "A", "blue");
        n.y = 10.0;
        n.height = 40.0;
        assert!((n.center_y() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn node_x_position() {
        let mut n = SankeyNode::new("a", "A", "blue");
        n.column = 2;
        let x = n.x(20.0, 50.0);
        assert!((x - 140.0).abs() < 1e-9); // 2 * (20 + 50)
    }

    #[test]
    fn config_default() {
        let cfg = SankeyConfig::default();
        assert!(cfg.width > 0.0);
        assert!(cfg.height > 0.0);
        assert!(cfg.node_width > 0.0);
        assert!(cfg.link_opacity > 0.0);
    }

    #[test]
    fn actual_col_gap_single_col() {
        let nodes = vec![SankeyNode::new("a", "A", "red")];
        let cfg = SankeyConfig::default();
        assert_eq!(actual_col_gap(&nodes, &cfg), 0.0);
    }

    #[test]
    fn large_flow() {
        let mut nodes = vec![
            SankeyNode::new("s", "Source", "blue"),
            SankeyNode::new("t", "Target", "red"),
        ];
        let mut links = vec![SankeyLink::new("s", "t", 1_000_000.0)];
        let cfg = SankeyConfig::default();
        let svg = sankey_diagram(&mut nodes, &mut links, &cfg);
        assert!(svg.contains("1000000"));
    }

    #[test]
    fn diamond_topology() {
        // A -> B, A -> C, B -> D, C -> D
        let mut nodes = vec![
            SankeyNode::new("a", "A", "red"),
            SankeyNode::new("b", "B", "green"),
            SankeyNode::new("c", "C", "blue"),
            SankeyNode::new("d", "D", "orange"),
        ];
        let mut links = vec![
            SankeyLink::new("a", "b", 10.0),
            SankeyLink::new("a", "c", 10.0),
            SankeyLink::new("b", "d", 10.0),
            SankeyLink::new("c", "d", 10.0),
        ];
        let cfg = SankeyConfig::default();
        layout(&mut nodes, &mut links, &cfg);
        let map: HashMap<&str, usize> = nodes.iter().map(|n| (n.id.as_str(), n.column)).collect();
        assert_eq!(map["a"], 0);
        assert!(map["b"] == 1);
        assert!(map["c"] == 1);
        assert_eq!(map["d"], 2);
    }
}
