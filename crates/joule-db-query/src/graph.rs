//! Graph algorithms for SQL graph functions.
//!
//! Provides BFS, shortest path, connected components, PageRank, and
//! traversal algorithms that operate on edge lists extracted from SQL tables.

use std::collections::{HashMap, HashSet, VecDeque};

/// Build an adjacency list from edge pairs. Returns (outgoing, incoming, all_nodes).
fn build_adjacency(
    edges: &[(String, String)],
) -> (
    HashMap<String, Vec<String>>,
    HashMap<String, Vec<String>>,
    HashSet<String>,
) {
    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    let mut incoming: HashMap<String, Vec<String>> = HashMap::new();
    let mut nodes: HashSet<String> = HashSet::new();
    for (src, dst) in edges {
        nodes.insert(src.clone());
        nodes.insert(dst.clone());
        outgoing.entry(src.clone()).or_default().push(dst.clone());
        incoming.entry(dst.clone()).or_default().push(src.clone());
    }
    (outgoing, incoming, nodes)
}

/// Extract edge pairs from table data.
/// columns: column names, rows: row values, src_col/dst_col: column names for source/dest.
pub fn extract_edges(
    columns: &[String],
    rows: &[Vec<serde_json::Value>],
    src_col: &str,
    dst_col: &str,
) -> Vec<(String, String)> {
    let src_idx = columns.iter().position(|c| c.eq_ignore_ascii_case(src_col));
    let dst_idx = columns.iter().position(|c| c.eq_ignore_ascii_case(dst_col));
    match (src_idx, dst_idx) {
        (Some(si), Some(di)) => rows
            .iter()
            .filter_map(|row| {
                let src = value_to_string(row.get(si)?)?;
                let dst = value_to_string(row.get(di)?)?;
                Some((src, dst))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn value_to_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Convert a JSON value to a string node identifier (public, for use by SQL dispatch).
pub fn value_to_node_id(v: &serde_json::Value) -> Option<String> {
    value_to_string(v)
}

/// BFS shortest path length from start to end. Returns None if unreachable.
pub fn shortest_path_length(edges: &[(String, String)], start: &str, end: &str) -> Option<i64> {
    if start == end {
        return Some(0);
    }
    let (outgoing, _, _) = build_adjacency(edges);
    let mut visited: HashSet<&str> = HashSet::new();
    let mut queue: VecDeque<(&str, i64)> = VecDeque::new();
    visited.insert(start);
    queue.push_back((start, 0));
    while let Some((node, dist)) = queue.pop_front() {
        if let Some(neighbors) = outgoing.get(node) {
            for neighbor in neighbors {
                if neighbor.as_str() == end {
                    return Some(dist + 1);
                }
                if visited.insert(neighbor.as_str()) {
                    queue.push_back((neighbor.as_str(), dist + 1));
                }
            }
        }
    }
    None
}

/// BFS shortest path as list of node IDs from start to end. Returns None if unreachable.
pub fn shortest_path(edges: &[(String, String)], start: &str, end: &str) -> Option<Vec<String>> {
    if start == end {
        return Some(vec![start.to_string()]);
    }
    let (outgoing, _, _) = build_adjacency(edges);
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<Vec<String>> = VecDeque::new();
    visited.insert(start.to_string());
    queue.push_back(vec![start.to_string()]);
    while let Some(path) = queue.pop_front() {
        let node = path.last().unwrap();
        if let Some(neighbors) = outgoing.get(node.as_str()) {
            for neighbor in neighbors {
                if neighbor.as_str() == end {
                    let mut result = path.clone();
                    result.push(end.to_string());
                    return Some(result);
                }
                if visited.insert(neighbor.clone()) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    queue.push_back(new_path);
                }
            }
        }
    }
    None
}

/// Get neighbors of a node up to max_depth hops. Returns set of node IDs (excluding start).
pub fn neighbors(edges: &[(String, String)], start: &str, max_depth: i64) -> Vec<String> {
    let (outgoing, _, _) = build_adjacency(edges);
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, i64)> = VecDeque::new();
    visited.insert(start.to_string());
    queue.push_back((start.to_string(), 0));
    let mut result = Vec::new();
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        if let Some(nbrs) = outgoing.get(node.as_str()) {
            for nbr in nbrs {
                if visited.insert(nbr.clone()) {
                    result.push(nbr.clone());
                    queue.push_back((nbr.clone(), depth + 1));
                }
            }
        }
    }
    result
}

/// Connected components using union-find. Returns map of node_id → component_id.
pub fn connected_components(edges: &[(String, String)]) -> HashMap<String, i64> {
    let (_, _, all_nodes) = build_adjacency(edges);
    let nodes: Vec<String> = all_nodes.into_iter().collect();
    let node_to_idx: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // Union-Find
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    let mut rank: Vec<usize> = vec![0; nodes.len()];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return;
        }
        if rank[rx] < rank[ry] {
            parent[rx] = ry;
        } else if rank[rx] > rank[ry] {
            parent[ry] = rx;
        } else {
            parent[ry] = rx;
            rank[rx] += 1;
        }
    }

    for (src, dst) in edges {
        if let (Some(&si), Some(&di)) =
            (node_to_idx.get(src.as_str()), node_to_idx.get(dst.as_str()))
        {
            union(&mut parent, &mut rank, si, di);
        }
    }

    // Map roots to sequential component IDs
    let mut root_to_component: HashMap<usize, i64> = HashMap::new();
    let mut next_id: i64 = 0;
    let mut result = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let root = find(&mut parent, i);
        let comp_id = *root_to_component.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        result.insert(node.clone(), comp_id);
    }
    result
}

/// PageRank algorithm. Returns map of node_id → score.
pub fn pagerank(
    edges: &[(String, String)],
    iterations: usize,
    damping: f64,
) -> HashMap<String, f64> {
    let (outgoing, _, all_nodes) = build_adjacency(edges);
    let n = all_nodes.len();
    if n == 0 {
        return HashMap::new();
    }
    let nodes: Vec<String> = all_nodes.into_iter().collect();
    let node_to_idx: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let initial = 1.0 / n as f64;
    let mut scores: Vec<f64> = vec![initial; n];

    // Pre-compute out-degree
    let out_degree: Vec<usize> = nodes
        .iter()
        .map(|node| outgoing.get(node.as_str()).map_or(0, |v| v.len()))
        .collect();

    for _ in 0..iterations {
        let mut new_scores = vec![(1.0 - damping) / n as f64; n];
        for (i, node) in nodes.iter().enumerate() {
            if out_degree[i] > 0 {
                let share = scores[i] / out_degree[i] as f64;
                if let Some(nbrs) = outgoing.get(node.as_str()) {
                    for nbr in nbrs {
                        if let Some(&j) = node_to_idx.get(nbr.as_str()) {
                            new_scores[j] += damping * share;
                        }
                    }
                }
            } else {
                // Dangling node: distribute evenly
                let share = scores[i] / n as f64;
                for s in new_scores.iter_mut() {
                    *s += damping * share;
                }
            }
        }
        scores = new_scores;
    }

    nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node, scores[i]))
        .collect()
}

/// Check if end is reachable from start via BFS.
pub fn graph_reach(edges: &[(String, String)], start: &str, end: &str) -> bool {
    shortest_path_length(edges, start, end).is_some()
}

/// BFS traversal from start up to max_depth. Returns nodes in BFS order.
pub fn bfs_traverse(
    edges: &[(String, String)],
    start: &str,
    max_depth: Option<i64>,
) -> Vec<String> {
    let (outgoing, _, _) = build_adjacency(edges);
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, i64)> = VecDeque::new();
    let mut result = Vec::new();
    visited.insert(start.to_string());
    result.push(start.to_string());
    queue.push_back((start.to_string(), 0));
    while let Some((node, depth)) = queue.pop_front() {
        if let Some(max) = max_depth {
            if depth >= max {
                continue;
            }
        }
        if let Some(nbrs) = outgoing.get(node.as_str()) {
            for nbr in nbrs {
                if visited.insert(nbr.clone()) {
                    result.push(nbr.clone());
                    queue.push_back((nbr.clone(), depth + 1));
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_edges() -> Vec<(String, String)> {
        vec![
            ("1".into(), "2".into()),
            ("1".into(), "3".into()),
            ("2".into(), "4".into()),
            ("3".into(), "4".into()),
            ("4".into(), "5".into()),
        ]
    }

    #[test]
    fn test_shortest_path_length() {
        let edges = sample_edges();
        assert_eq!(shortest_path_length(&edges, "1", "5"), Some(3));
        assert_eq!(shortest_path_length(&edges, "1", "4"), Some(2));
        assert_eq!(shortest_path_length(&edges, "1", "1"), Some(0));
        assert_eq!(shortest_path_length(&edges, "5", "1"), None); // no reverse path
    }

    #[test]
    fn test_shortest_path() {
        let edges = sample_edges();
        let path = shortest_path(&edges, "1", "5").unwrap();
        assert_eq!(path.len(), 4); // 1 → 2/3 → 4 → 5
        assert_eq!(path.first().unwrap(), "1");
        assert_eq!(path.last().unwrap(), "5");
    }

    #[test]
    fn test_neighbors() {
        let edges = sample_edges();
        let nbrs = neighbors(&edges, "1", 1);
        assert_eq!(nbrs.len(), 2); // 2, 3
        assert!(nbrs.contains(&"2".to_string()));
        assert!(nbrs.contains(&"3".to_string()));

        let nbrs2 = neighbors(&edges, "1", 2);
        assert_eq!(nbrs2.len(), 3); // 2, 3, 4
    }

    #[test]
    fn test_connected_components() {
        let mut edges = sample_edges();
        // Add disconnected component
        edges.push(("10".into(), "11".into()));
        let comps = connected_components(&edges);
        // 1,2,3,4,5 in one component; 10,11 in another
        assert_eq!(comps.get("1"), comps.get("2"));
        assert_eq!(comps.get("1"), comps.get("5"));
        assert_eq!(comps.get("10"), comps.get("11"));
        assert_ne!(comps.get("1"), comps.get("10"));
    }

    #[test]
    fn test_pagerank() {
        let edges = sample_edges();
        let pr = pagerank(&edges, 20, 0.85);
        assert_eq!(pr.len(), 5);
        // Node 4 and 5 should have higher PageRank (they receive more links)
        assert!(pr["4"] > pr["1"]);
        assert!(pr["5"] > pr["1"]);
    }

    #[test]
    fn test_graph_reach() {
        let edges = sample_edges();
        assert!(graph_reach(&edges, "1", "5"));
        assert!(!graph_reach(&edges, "5", "1"));
    }

    #[test]
    fn test_bfs_traverse() {
        let edges = sample_edges();
        let traversal = bfs_traverse(&edges, "1", None);
        assert_eq!(traversal[0], "1");
        assert_eq!(traversal.len(), 5);

        let limited = bfs_traverse(&edges, "1", Some(1));
        assert_eq!(limited.len(), 3); // 1, 2, 3
    }

    #[test]
    fn test_extract_edges() {
        let columns = vec!["src".to_string(), "dst".to_string(), "weight".to_string()];
        let rows = vec![
            vec![
                serde_json::json!(1),
                serde_json::json!(2),
                serde_json::json!(1.0),
            ],
            vec![
                serde_json::json!(2),
                serde_json::json!(3),
                serde_json::json!(2.0),
            ],
        ];
        let edges = extract_edges(&columns, &rows, "src", "dst");
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0], ("1".to_string(), "2".to_string()));
        assert_eq!(edges[1], ("2".to_string(), "3".to_string()));
    }
}
