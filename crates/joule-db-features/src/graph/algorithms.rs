//! Advanced Graph Algorithms
//!
//! Implements comprehensive graph algorithms including:
//! - PageRank
//! - Community Detection (Louvain)
//! - Centrality Measures (Betweenness, Closeness, Eigenvector)
//! - Graph Clustering
//! - Minimum Spanning Tree

use super::{GraphStore, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};

/// PageRank algorithm result
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// Node ID -> PageRank score
    pub scores: HashMap<NodeId, f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence delta
    pub delta: f64,
}

/// Community detection result
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Node ID -> Community ID
    pub communities: HashMap<NodeId, usize>,
    /// Number of communities
    pub num_communities: usize,
    /// Modularity score
    pub modularity: f64,
}

/// Centrality measures
#[derive(Debug, Clone)]
pub struct CentralityMeasures {
    /// Betweenness centrality
    pub betweenness: HashMap<NodeId, f64>,
    /// Closeness centrality
    pub closeness: HashMap<NodeId, f64>,
    /// Eigenvector centrality
    pub eigenvector: HashMap<NodeId, f64>,
}

/// Graph algorithms implementation
impl GraphStore {
    /// Compute PageRank for all nodes
    ///
    /// PageRank measures the importance of nodes based on the structure of the graph.
    /// Returns a map of node IDs to their PageRank scores.
    pub fn pagerank(
        &self,
        damping_factor: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> PageRankResult {
        let nodes = self.nodes.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let incoming = self.incoming.read().unwrap();

        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return PageRankResult {
                scores: HashMap::new(),
                iterations: 0,
                delta: 0.0,
            };
        }

        let initial_score = 1.0 / num_nodes as f64;
        let mut scores: HashMap<NodeId, f64> =
            nodes.keys().map(|&id| (id, initial_score)).collect();
        let mut new_scores = scores.clone();

        // Calculate out-degrees
        let out_degrees: HashMap<NodeId, usize> = outgoing
            .iter()
            .map(|(id, edges)| (*id, edges.len()))
            .collect();

        for iteration in 0..max_iterations {
            let mut max_delta: f64 = 0.0;

            // Compute new scores
            for node_id in nodes.keys() {
                let mut rank = (1.0 - damping_factor) / num_nodes as f64;

                // Sum contributions from incoming edges
                if let Some(incoming_edges) = incoming.get(node_id) {
                    for edge_id in incoming_edges {
                        if let Some(edge) = self.edges.read().unwrap().get(edge_id) {
                            let source = edge.from;
                            if let Some(&source_score) = scores.get(&source) {
                                let out_degree = out_degrees.get(&source).copied().unwrap_or(1);
                                if out_degree > 0 {
                                    rank += damping_factor * source_score / out_degree as f64;
                                }
                            }
                        }
                    }
                }

                let old_score = scores.get(node_id).copied().unwrap_or(0.0);
                let delta = (rank - old_score).abs();
                max_delta = max_delta.max(delta);
                new_scores.insert(*node_id, rank);
            }

            // Check convergence
            if max_delta < tolerance {
                return PageRankResult {
                    scores: new_scores,
                    iterations: iteration + 1,
                    delta: max_delta,
                };
            }

            scores = new_scores.clone();
        }

        PageRankResult {
            scores,
            iterations: max_iterations,
            delta: 0.0,
        }
    }

    /// Detect communities using Louvain algorithm
    ///
    /// Returns a map of node IDs to their community assignments.
    pub fn detect_communities(&self, resolution: f64) -> CommunityResult {
        let nodes = self.nodes.read().unwrap();
        let edges = self.edges.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let incoming = self.incoming.read().unwrap();

        if nodes.is_empty() {
            return CommunityResult {
                communities: HashMap::new(),
                num_communities: 0,
                modularity: 0.0,
            };
        }

        // Initialize: each node in its own community
        let mut communities: HashMap<NodeId, usize> =
            nodes.keys().enumerate().map(|(i, &id)| (id, i)).collect();

        let _community_id_counter = nodes.len();

        // Calculate total edge weight (assuming unweighted = 1.0)
        let total_weight = edges.len() as f64;

        // Iterative optimization
        let mut improved = true;
        let mut iterations = 0;
        let max_iterations = 100;

        while improved && iterations < max_iterations {
            improved = false;
            iterations += 1;

            // Try moving each node to neighboring communities
            for node_id in nodes.keys() {
                let current_community = communities.get(node_id).copied().unwrap();

                // Get neighboring communities
                let mut neighbor_communities: HashMap<usize, usize> = HashMap::new();

                // Check outgoing edges
                if let Some(edge_ids) = outgoing.get(node_id) {
                    for edge_id in edge_ids {
                        if let Some(edge) = edges.get(edge_id) {
                            let neighbor_community = communities.get(&edge.to).copied().unwrap();
                            *neighbor_communities.entry(neighbor_community).or_insert(0) += 1;
                        }
                    }
                }

                // Check incoming edges
                if let Some(edge_ids) = incoming.get(node_id) {
                    for edge_id in edge_ids {
                        if let Some(edge) = edges.get(edge_id) {
                            let neighbor_community = communities.get(&edge.from).copied().unwrap();
                            *neighbor_communities.entry(neighbor_community).or_insert(0) += 1;
                        }
                    }
                }

                // Find best community (simplified Louvain)
                let mut best_community = current_community;
                let mut best_gain = 0.0;

                for (&community, &connections) in &neighbor_communities {
                    if community != current_community {
                        // Simplified modularity gain calculation
                        let gain = connections as f64 * resolution;
                        if gain > best_gain {
                            best_gain = gain;
                            best_community = community;
                        }
                    }
                }

                // Move node if improvement found
                if best_community != current_community {
                    communities.insert(*node_id, best_community);
                    improved = true;
                }
            }
        }

        // Calculate modularity
        let modularity = self.calculate_modularity(&communities, total_weight);

        // Count unique communities
        let unique_communities: HashSet<usize> = communities.values().copied().collect();
        let num_communities = unique_communities.len();

        CommunityResult {
            communities,
            num_communities,
            modularity,
        }
    }

    /// Calculate modularity of community assignment
    fn calculate_modularity(&self, communities: &HashMap<NodeId, usize>, total_weight: f64) -> f64 {
        if total_weight == 0.0 {
            return 0.0;
        }

        let edges = self.edges.read().unwrap();
        let mut modularity = 0.0;

        for edge in edges.values() {
            let from_community = communities.get(&edge.from).copied().unwrap_or(0);
            let to_community = communities.get(&edge.to).copied().unwrap_or(0);

            let weight = 1.0; // Assuming unweighted

            if from_community == to_community {
                // Calculate degree of from and to nodes
                let from_degree = self.get_outgoing_edges(edge.from).len()
                    + self.get_incoming_edges(edge.from).len();
                let to_degree =
                    self.get_outgoing_edges(edge.to).len() + self.get_incoming_edges(edge.to).len();

                modularity +=
                    weight - (from_degree as f64 * to_degree as f64) / (2.0 * total_weight);
            }
        }

        modularity / (2.0 * total_weight)
    }

    /// Compute betweenness centrality for all nodes
    ///
    /// Betweenness centrality measures how often a node appears on shortest paths
    /// between other nodes.
    pub fn betweenness_centrality(&self) -> HashMap<NodeId, f64> {
        let nodes = self.nodes.read().unwrap();
        let mut centrality: HashMap<NodeId, f64> = nodes.keys().map(|&id| (id, 0.0)).collect();

        // For each node, compute shortest paths through it
        for source in nodes.keys() {
            // BFS from source
            let mut distances: HashMap<NodeId, usize> = HashMap::new();
            let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
            let mut queue = VecDeque::new();

            distances.insert(*source, 0);
            queue.push_back(*source);

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];

                for edge in self.get_outgoing_edges(current) {
                    let neighbor = edge.to;

                    if !distances.contains_key(&neighbor) {
                        distances.insert(neighbor, current_dist + 1);
                        queue.push_back(neighbor);
                    }

                    if distances.get(&neighbor) == Some(&(current_dist + 1)) {
                        predecessors
                            .entry(neighbor)
                            .or_insert_with(Vec::new)
                            .push(current);
                    }
                }
            }

            // Count shortest paths through each node
            let mut path_counts: HashMap<NodeId, usize> = HashMap::new();
            for target in nodes.keys() {
                if *target != *source {
                    self.count_paths(*source, *target, &predecessors, &mut path_counts);
                }
            }

            // Update centrality
            for (node, count) in path_counts {
                *centrality.entry(node).or_insert(0.0) += count as f64;
            }
        }

        // Normalize
        let n = nodes.len() as f64;
        let normalization = (n - 1.0) * (n - 2.0);
        if normalization > 0.0 {
            for value in centrality.values_mut() {
                *value /= normalization;
            }
        }

        centrality
    }

    /// Compute closeness centrality for all nodes
    ///
    /// Closeness centrality measures how close a node is to all other nodes.
    pub fn closeness_centrality(&self) -> HashMap<NodeId, f64> {
        let nodes = self.nodes.read().unwrap();
        let mut centrality: HashMap<NodeId, f64> = HashMap::new();

        for node_id in nodes.keys() {
            // BFS to compute distances to all nodes
            let mut distances: HashMap<NodeId, usize> = HashMap::new();
            let mut queue = VecDeque::new();

            distances.insert(*node_id, 0);
            queue.push_back(*node_id);

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[&current];

                for edge in self.get_outgoing_edges(current) {
                    if !distances.contains_key(&edge.to) {
                        distances.insert(edge.to, current_dist + 1);
                        queue.push_back(edge.to);
                    }
                }
            }

            // Calculate sum of distances
            let sum_distances: usize = distances.values().sum();
            if sum_distances > 0 {
                centrality.insert(*node_id, (nodes.len() - 1) as f64 / sum_distances as f64);
            } else {
                centrality.insert(*node_id, 0.0);
            }
        }

        centrality
    }

    /// Compute all centrality measures
    pub fn compute_centrality_measures(&self) -> CentralityMeasures {
        CentralityMeasures {
            betweenness: self.betweenness_centrality(),
            closeness: self.closeness_centrality(),
            eigenvector: self.eigenvector_centrality(),
        }
    }

    /// Compute eigenvector centrality (simplified power iteration)
    fn eigenvector_centrality(&self) -> HashMap<NodeId, f64> {
        let nodes = self.nodes.read().unwrap();
        let _outgoing = self.outgoing.read().unwrap();

        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return HashMap::new();
        }

        let mut scores: HashMap<NodeId, f64> = nodes
            .keys()
            .map(|&id| (id, 1.0 / num_nodes as f64))
            .collect();

        // Power iteration
        for _ in 0..50 {
            let mut new_scores = HashMap::new();
            let mut total = 0.0;

            for node_id in nodes.keys() {
                let mut score = 0.0;

                let incoming = self.incoming.read().unwrap();
                if let Some(edge_ids) = incoming.get(node_id) {
                    let edges = self.edges.read().unwrap();
                    for edge_id in edge_ids {
                        if let Some(edge) = edges.get(edge_id) {
                            score += scores.get(&edge.from).copied().unwrap_or(0.0);
                        }
                    }
                }

                new_scores.insert(*node_id, score);
                total += score * score;
            }

            // Normalize
            let norm = total.sqrt();
            if norm > 0.0 {
                for value in new_scores.values_mut() {
                    *value /= norm;
                }
            }

            scores = new_scores;
        }

        scores
    }

    /// Helper: count shortest paths
    fn count_paths(
        &self,
        source: NodeId,
        target: NodeId,
        predecessors: &HashMap<NodeId, Vec<NodeId>>,
        path_counts: &mut HashMap<NodeId, usize>,
    ) {
        if source == target {
            *path_counts.entry(target).or_insert(0) += 1;
            return;
        }

        if let Some(preds) = predecessors.get(&target) {
            for pred in preds {
                self.count_paths(source, *pred, predecessors, path_counts);
                *path_counts.entry(*pred).or_insert(0) += 1;
            }
        }
    }
}
