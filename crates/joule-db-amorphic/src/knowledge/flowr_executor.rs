//! flowR Executor Interface: trait-based integration with codegraph's reasoning engine.
//!
//! inv-ai-codegraph cannot be a direct dependency (cyclic dependency chain).
//! Instead, this module defines the trait that the real executor implements.
//! The application layer wires the real `ReasoningExecutor<EchoBackend>` at startup.
//!
//! This gives JouleDB AI the same API without the cyclic dependency:
//! - `FlowRExecutor` trait matches codegraph's `ReasoningExecutor.execute_graph()`
//! - `ReasoningOutput` matches codegraph's `ReasoningTrace`
//! - The materializer calls through the trait

use std::collections::HashMap;

/// A node in the reasoning graph (mirrors codegraph's ReasoningNode).
#[derive(Clone, Debug)]
pub struct RNode {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub initial_value: Option<String>,
    pub prompt_template: Option<String>,
}

/// A wire in the reasoning graph (mirrors codegraph's ReasoningWire).
#[derive(Clone, Debug)]
pub struct RWire {
    pub from_id: String,
    pub to_id: String,
    pub wire_kind: String,
}

/// Output from a reasoning execution.
#[derive(Clone, Debug)]
pub struct ReasoningOutput {
    pub conclusion: String,
    pub confidence: f64,
    pub steps: usize,
    pub total_tokens: u32,
    pub total_joules: f64,
    pub trace: Vec<(String, String, f64)>, // (node_id, output, confidence)
}

impl ReasoningOutput {
    pub fn render(&self) -> String {
        let mut out = format!(
            "Conclusion: {}\nConfidence: {:.2}\nSteps: {}\nEnergy: {:.6} J\n",
            self.conclusion, self.confidence, self.steps, self.total_joules
        );
        for (id, output, conf) in &self.trace {
            out.push_str(&format!("  [{}] {} (conf={:.2})\n", id, output, conf));
        }
        out
    }
}

/// Trait for any reasoning executor.
/// Implement with codegraph's ReasoningExecutor at the app layer.
pub trait FlowRExecutor: Send + Sync {
    /// Execute a reasoning graph.
    fn execute(
        &self,
        nodes: &HashMap<String, RNode>,
        wires: &[RWire],
        entry: &str,
    ) -> Result<ReasoningOutput, String>;
}

/// Built-in deterministic executor: walks the graph topologically,
/// concatenates outputs. No LLM. No neural. Pure graph traversal.
pub struct DeterministicExecutor;

impl FlowRExecutor for DeterministicExecutor {
    fn execute(
        &self,
        nodes: &HashMap<String, RNode>,
        wires: &[RWire],
        entry: &str,
    ) -> Result<ReasoningOutput, String> {
        // Simple topological walk from entry
        let mut outputs: HashMap<String, String> = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = vec![entry.to_string()];
        let mut trace = Vec::new();

        while let Some(node_id) = queue.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());

            if let Some(node) = nodes.get(&node_id) {
                // Collect inputs from predecessors
                let inputs: Vec<String> = wires
                    .iter()
                    .filter(|w| w.to_id == node_id)
                    .filter_map(|w| outputs.get(&w.from_id).cloned())
                    .collect();

                let input = if inputs.is_empty() {
                    node.initial_value.clone().unwrap_or_default()
                } else {
                    inputs.join(" | ")
                };

                // Execute: apply template if present, otherwise pass through
                let output = if let Some(ref template) = node.prompt_template {
                    template.replace("{input}", &input)
                } else {
                    input
                };

                outputs.insert(node_id.clone(), output.clone());
                trace.push((node_id.clone(), output, 0.8));

                // Queue successors
                for wire in wires {
                    if wire.from_id == node_id {
                        queue.push(wire.to_id.clone());
                    }
                }
            }
        }

        let conclusion = trace
            .last()
            .map(|(_, o, _)| o.clone())
            .unwrap_or_default();

        Ok(ReasoningOutput {
            conclusion,
            confidence: 0.8,
            steps: trace.len(),
            total_tokens: 0,
            total_joules: 0.001 * trace.len() as f64,
            trace,
        })
    }
}

/// Build a simple reasoning graph for a query.
pub fn build_query_graph(
    query: &str,
    domain: &str,
) -> (HashMap<String, RNode>, Vec<RWire>, String) {
    let mut nodes = HashMap::new();
    let mut wires = Vec::new();

    let bind_id = "n0".to_string();
    nodes.insert(
        bind_id.clone(),
        RNode {
            id: bind_id.clone(),
            kind: "Bind".into(),
            name: "bind_query".into(),
            initial_value: Some(query.to_string()),
            prompt_template: None,
        },
    );

    let analyze_id = "n1".to_string();
    nodes.insert(
        analyze_id.clone(),
        RNode {
            id: analyze_id.clone(),
            kind: "Apply".into(),
            name: "analyze".into(),
            initial_value: None,
            prompt_template: Some(format!("Using {} knowledge, analyze: {{input}}", domain)),
        },
    );
    wires.push(RWire {
        from_id: bind_id.clone(),
        to_id: analyze_id.clone(),
        wire_kind: "Copy".into(),
    });

    let observe_id = "n2".to_string();
    nodes.insert(
        observe_id.clone(),
        RNode {
            id: observe_id.clone(),
            kind: "Observe".into(),
            name: "synthesize".into(),
            initial_value: None,
            prompt_template: Some("Result: {input}".into()),
        },
    );
    wires.push(RWire {
        from_id: analyze_id,
        to_id: observe_id,
        wire_kind: "Move".into(),
    });

    (nodes, wires, bind_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_executor() {
        let executor = DeterministicExecutor;
        let (nodes, wires, entry) = build_query_graph("Why is the sky blue?", "physics");
        let result = executor.execute(&nodes, &wires, &entry).unwrap();

        assert!(!result.conclusion.is_empty());
        assert!(result.steps > 0);
        assert!(result.total_joules > 0.0);
        eprintln!("{}", result.render());
    }

    #[test]
    fn test_custom_graph() {
        let executor = DeterministicExecutor;
        let mut nodes = HashMap::new();
        let mut wires = Vec::new();

        nodes.insert("start".into(), RNode {
            id: "start".into(),
            kind: "Bind".into(),
            name: "premise".into(),
            initial_value: Some("Cancer exhibits replication and feedback".into()),
            prompt_template: None,
        });

        nodes.insert("reason".into(), RNode {
            id: "reason".into(),
            kind: "Apply".into(),
            name: "reason_about".into(),
            initial_value: None,
            prompt_template: Some("Given that {input}, what structural patterns are present?".into()),
        });

        wires.push(RWire {
            from_id: "start".into(),
            to_id: "reason".into(),
            wire_kind: "Copy".into(),
        });

        let result = executor.execute(&nodes, &wires, "start").unwrap();
        assert!(result.conclusion.contains("replication"));
        assert!(result.conclusion.contains("feedback"));
    }
}
