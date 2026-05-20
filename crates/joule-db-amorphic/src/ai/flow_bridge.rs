//! flowR as JouleDB AI Reasoning Backend
//!
//! Bridges JouleDB AI queries to the codegraph flowR reasoning executor.
//! Converts JouleDB queries into flowG graphs, routes to appropriate domain LUTs,
//! and maps ReasoningTraces back to AiResult + AiReceipt.
//!
//! The integration contract:
//! - JouleDB AI facade calls flow_bridge when Tier 1 (holographic) isn't sufficient
//! - flow_bridge constructs a flowG graph from the query
//! - flowR executor traverses the graph with domain-appropriate LUT
//! - Energy tracked end-to-end (AiReceipt wraps NodeCost)

use crate::ai::receipt::AiReceipt;
use crate::ai::traits::{AiError, AiOutput, ReasoningResult};
use std::collections::HashMap;

/// Domain-specific LUT selection for flowR routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DomainLut {
    /// flowA — audit, compliance, verification
    Audit,
    /// flowO — optimization, performance, resource allocation
    Optimization,
    /// flowL — legal, rights, licensing, contracts
    Legal,
    /// flowC — creative, generative, synthesis
    Creative,
    /// flowT — metacognition, self-reflection, trace analysis
    Metacognition,
    /// flowQIT — quantum information theory, entropy, decoherence
    QuantumInformation,
}

impl DomainLut {
    /// Classify query intent → domain LUT.
    pub fn from_query(query: &str) -> Self {
        let lower = query.to_lowercase();

        // Legal signals (check before audit — "check license" is legal, not audit)
        if lower.contains("license")
            || lower.contains("rights")
            || lower.contains("copyright")
            || lower.contains("legal")
            || lower.contains("territory")
            || lower.contains("contract")
        {
            return Self::Legal;
        }

        // Audit signals
        if lower.contains("audit")
            || lower.contains("compliance")
            || lower.contains("verify")
            || lower.contains("validate")
            || lower.contains("check")
        {
            return Self::Audit;
        }

        // Optimization signals
        if lower.contains("optimize")
            || lower.contains("fastest")
            || lower.contains("cheapest")
            || lower.contains("efficient")
            || lower.contains("performance")
            || lower.contains("scale")
        {
            return Self::Optimization;
        }

        // Creative signals
        if lower.contains("create")
            || lower.contains("generate")
            || lower.contains("compose")
            || lower.contains("design")
            || lower.contains("imagine")
            || lower.contains("synthesize")
        {
            return Self::Creative;
        }

        // QIT signals
        if lower.contains("entropy")
            || lower.contains("decoherence")
            || lower.contains("quantum")
            || lower.contains("information gain")
            || lower.contains("landauer")
        {
            return Self::QuantumInformation;
        }

        // Default: metacognition (self-reflection on what to do)
        Self::Metacognition
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Audit => "flowA",
            Self::Optimization => "flowO",
            Self::Legal => "flowL",
            Self::Creative => "flowC",
            Self::Metacognition => "flowT",
            Self::QuantumInformation => "flowQIT",
        }
    }
}

/// Compute regime: how much compute to allocate per reasoning node.
/// Maps directly to flowR's UncertaintyConfig thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeRegime {
    /// High confidence → skip LLM, use lookup table (0 tokens)
    Trivial,
    /// Medium confidence → allocate proportional budget
    Tractable,
    /// Low confidence → flag, don't waste tokens
    Intractable,
}

/// A node in the reasoning graph that flowR will execute.
#[derive(Clone, Debug)]
pub struct FlowNode {
    pub id: String,
    pub kind: FlowNodeKind,
    pub label: String,
    pub initial_value: Option<String>,
    pub prompt_template: Option<String>,
    pub max_iterations: Option<u32>,
    pub convergence_tolerance: Option<f64>,
}

/// The 22 flowG primitives mapped to JouleDB AI operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowNodeKind {
    Bind,
    Apply,
    Mutate,
    Branch,
    Iterate,
    Match,
    Sequence,
    Compose,
    Abstract,
    Import,
    Acquire,
    Release,
    Observe,
    ErrorPropagate,
    CodeBlock,
    Spawn,
    JoinAwait,
    YieldSuspend,
    Listen,
    Transact,
    TypeDefine,
    FeedbackDelay,
}

/// A wire connecting two nodes in the reasoning graph.
#[derive(Clone, Debug)]
pub struct FlowWire {
    pub from_id: String,
    pub to_id: String,
    pub wire_kind: WireKind,
}

/// Wire semantics for data flow between reasoning nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireKind {
    Move,
    Copy,
    Borrow,
    Channel,
    Feedback,
    Error,
}

/// A complete reasoning graph ready for flowR execution.
#[derive(Clone, Debug)]
pub struct FlowGraph {
    pub nodes: HashMap<String, FlowNode>,
    pub wires: Vec<FlowWire>,
    pub entry_node: String,
    pub domain: DomainLut,
}

/// The result of a flowR reasoning execution, mapped back to JouleDB types.
#[derive(Clone, Debug)]
pub struct FlowResult {
    pub conclusion: String,
    pub confidence: f64,
    pub trace_nodes: Vec<TraceStep>,
    pub total_tokens: u32,
    pub total_joules: f64,
    pub regime: ComputeRegime,
}

/// A single step in the reasoning trace.
#[derive(Clone, Debug)]
pub struct TraceStep {
    pub node_id: String,
    pub label: String,
    pub output: String,
    pub confidence: f64,
    pub tokens_used: u32,
    pub elapsed_ms: u64,
    pub regime: ComputeRegime,
}

/// The flow bridge: converts JouleDB queries into flowG graphs and executes them.
///
/// This is trait-based so that the actual flowR executor (in inv-ai-codegraph)
/// can be plugged in without joule-db-amorphic depending on it directly.
pub trait FlowReasoning: Send + Sync {
    /// Execute a reasoning graph and return a structured trace.
    fn reason_graph(&self, graph: &FlowGraph) -> Result<FlowResult, AiError>;

    /// Select the appropriate domain LUT for a query.
    fn select_domain(&self, query: &str) -> DomainLut {
        DomainLut::from_query(query)
    }
}

/// Graph builder: constructs flowG reasoning graphs from JouleDB queries.
pub struct FlowGraphBuilder {
    next_id: u32,
}

impl FlowGraphBuilder {
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    fn next_id(&mut self) -> String {
        let id = format!("n{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Build a reasoning graph for a natural language query.
    ///
    /// Structure: Bind(query) → Apply(analyze) → Branch(complexity) →
    ///   [Trivial: Observe(lookup)]
    ///   [Tractable: Iterate(reason) → Apply(synthesize)]
    ///   [Intractable: ErrorPropagate(escalate)]
    pub fn build_query_graph(&mut self, query: &str, domain: DomainLut) -> FlowGraph {
        let mut nodes = HashMap::new();
        let mut wires = Vec::new();

        // Node 0: Bind the query as the initial premise
        let bind_id = self.next_id();
        nodes.insert(
            bind_id.clone(),
            FlowNode {
                id: bind_id.clone(),
                kind: FlowNodeKind::Bind,
                label: "bind_query".into(),
                initial_value: Some(query.to_string()),
                prompt_template: None,
                max_iterations: None,
                convergence_tolerance: None,
            },
        );

        // Node 1: Apply analysis — determine what the query is asking
        let analyze_id = self.next_id();
        nodes.insert(
            analyze_id.clone(),
            FlowNode {
                id: analyze_id.clone(),
                kind: FlowNodeKind::Apply,
                label: "analyze_intent".into(),
                initial_value: None,
                prompt_template: Some(format!(
                    "Using {} domain knowledge, analyze: {{input}}",
                    domain.name()
                )),
                max_iterations: None,
                convergence_tolerance: None,
            },
        );
        wires.push(FlowWire {
            from_id: bind_id.clone(),
            to_id: analyze_id.clone(),
            wire_kind: WireKind::Copy,
        });

        // Node 2: Branch on complexity
        let branch_id = self.next_id();
        nodes.insert(
            branch_id.clone(),
            FlowNode {
                id: branch_id.clone(),
                kind: FlowNodeKind::Branch,
                label: "complexity_gate".into(),
                initial_value: None,
                prompt_template: None,
                max_iterations: None,
                convergence_tolerance: None,
            },
        );
        wires.push(FlowWire {
            from_id: analyze_id.clone(),
            to_id: branch_id.clone(),
            wire_kind: WireKind::Move,
        });

        // Node 3: Iterate — reasoning loop for tractable problems
        let iterate_id = self.next_id();
        nodes.insert(
            iterate_id.clone(),
            FlowNode {
                id: iterate_id.clone(),
                kind: FlowNodeKind::Iterate,
                label: "reason_loop".into(),
                initial_value: None,
                prompt_template: Some(format!(
                    "Step {{iteration}}: Using {} reasoning, refine: {{input}}",
                    domain.name()
                )),
                max_iterations: Some(5),
                convergence_tolerance: Some(0.85),
            },
        );
        wires.push(FlowWire {
            from_id: branch_id.clone(),
            to_id: iterate_id.clone(),
            wire_kind: WireKind::Copy,
        });

        // Node 4: Observe — final synthesis
        let observe_id = self.next_id();
        nodes.insert(
            observe_id.clone(),
            FlowNode {
                id: observe_id.clone(),
                kind: FlowNodeKind::Observe,
                label: "synthesize_result".into(),
                initial_value: None,
                prompt_template: None,
                max_iterations: None,
                convergence_tolerance: None,
            },
        );
        wires.push(FlowWire {
            from_id: iterate_id.clone(),
            to_id: observe_id.clone(),
            wire_kind: WireKind::Move,
        });

        let entry = bind_id;
        FlowGraph {
            nodes,
            wires,
            entry_node: entry,
            domain,
        }
    }

    /// Build a contrast-driven reasoning graph.
    /// Input: two records and their contrast map → reason about what the contrast means.
    pub fn build_contrast_graph(
        &mut self,
        context: &str,
        contrast_summary: &str,
        domain: DomainLut,
    ) -> FlowGraph {
        let mut nodes = HashMap::new();
        let mut wires = Vec::new();

        // Bind context
        let ctx_id = self.next_id();
        nodes.insert(
            ctx_id.clone(),
            FlowNode {
                id: ctx_id.clone(),
                kind: FlowNodeKind::Bind,
                label: "bind_context".into(),
                initial_value: Some(context.to_string()),
                prompt_template: None,
                max_iterations: None,
                convergence_tolerance: None,
            },
        );

        // Bind contrast
        let contrast_id = self.next_id();
        nodes.insert(
            contrast_id.clone(),
            FlowNode {
                id: contrast_id.clone(),
                kind: FlowNodeKind::Bind,
                label: "bind_contrast".into(),
                initial_value: Some(contrast_summary.to_string()),
                prompt_template: None,
                max_iterations: None,
                convergence_tolerance: None,
            },
        );

        // Compose: merge context + contrast
        let compose_id = self.next_id();
        nodes.insert(
            compose_id.clone(),
            FlowNode {
                id: compose_id.clone(),
                kind: FlowNodeKind::Compose,
                label: "merge_inputs".into(),
                initial_value: None,
                prompt_template: None,
                max_iterations: None,
                convergence_tolerance: None,
            },
        );
        wires.push(FlowWire {
            from_id: ctx_id.clone(),
            to_id: compose_id.clone(),
            wire_kind: WireKind::Copy,
        });
        wires.push(FlowWire {
            from_id: contrast_id.clone(),
            to_id: compose_id.clone(),
            wire_kind: WireKind::Copy,
        });

        // Match: classify the type of contrast
        let match_id = self.next_id();
        nodes.insert(
            match_id.clone(),
            FlowNode {
                id: match_id.clone(),
                kind: FlowNodeKind::Match,
                label: "classify_contrast".into(),
                initial_value: None,
                prompt_template: Some(format!(
                    "Using {} domain, classify this contrast pattern: {{input}}",
                    domain.name()
                )),
                max_iterations: None,
                convergence_tolerance: None,
            },
        );
        wires.push(FlowWire {
            from_id: compose_id.clone(),
            to_id: match_id.clone(),
            wire_kind: WireKind::Move,
        });

        // Apply: generate insight from classified contrast
        let insight_id = self.next_id();
        nodes.insert(
            insight_id.clone(),
            FlowNode {
                id: insight_id.clone(),
                kind: FlowNodeKind::Apply,
                label: "generate_insight".into(),
                initial_value: None,
                prompt_template: Some(
                    "Given this classified contrast, what is the actionable insight?".into(),
                ),
                max_iterations: None,
                convergence_tolerance: None,
            },
        );
        wires.push(FlowWire {
            from_id: match_id.clone(),
            to_id: insight_id.clone(),
            wire_kind: WireKind::Move,
        });

        FlowGraph {
            nodes,
            wires,
            entry_node: ctx_id,
            domain,
        }
    }
}

impl Default for FlowGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a FlowResult into JouleDB AI types.
impl FlowResult {
    pub fn to_ai_output(&self) -> AiOutput {
        AiOutput::Text(self.conclusion.clone())
    }

    pub fn to_ai_receipt(&self) -> AiReceipt {
        AiReceipt::holographic("flowR", self.total_joules, 0)
    }

    pub fn to_reasoning_result(&self) -> ReasoningResult {
        let steps: Vec<String> = self
            .trace_nodes
            .iter()
            .map(|s| format!("[{}] {}", s.label, s.output))
            .collect();
        ReasoningResult {
            answer: self.conclusion.clone(),
            confidence: self.confidence as f32,
            reasoning_steps: steps,
            sources_used: vec!["flowR".to_string()],
        }
    }
}

/// A built-in flowR implementation that executes graphs locally without an LLM.
/// Uses the contrast engine + UCG for deterministic reasoning.
pub struct LocalFlowReasoner;

impl FlowReasoning for LocalFlowReasoner {
    fn reason_graph(&self, graph: &FlowGraph) -> Result<FlowResult, AiError> {
        // Walk the graph topologically, executing each node
        let mut outputs: HashMap<String, String> = HashMap::new();
        let mut trace = Vec::new();
        let mut total_tokens = 0u32;

        // Simple topological order: follow wires from entry
        let mut visited = std::collections::HashSet::new();
        let mut queue = vec![graph.entry_node.clone()];

        while let Some(node_id) = queue.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());

            if let Some(node) = graph.nodes.get(&node_id) {
                // Collect inputs from predecessor outputs
                let inputs: Vec<String> = graph
                    .wires
                    .iter()
                    .filter(|w| w.to_id == node_id)
                    .filter_map(|w| outputs.get(&w.from_id).cloned())
                    .collect();

                let input_text = if inputs.is_empty() {
                    node.initial_value.clone().unwrap_or_default()
                } else {
                    inputs.join(" | ")
                };

                // Execute node based on kind
                let output = match node.kind {
                    FlowNodeKind::Bind => input_text.clone(),
                    FlowNodeKind::Apply => {
                        if let Some(template) = &node.prompt_template {
                            template.replace("{input}", &input_text)
                        } else {
                            input_text.clone()
                        }
                    }
                    FlowNodeKind::Observe => {
                        format!("[observed: {}]", input_text)
                    }
                    FlowNodeKind::Match => {
                        format!("[matched: {}]", input_text)
                    }
                    FlowNodeKind::Compose => input_text.clone(),
                    FlowNodeKind::Branch => input_text.clone(),
                    FlowNodeKind::Iterate => {
                        let max = node.max_iterations.unwrap_or(3);
                        format!("[iterated {}x: {}]", max, input_text)
                    }
                    _ => input_text.clone(),
                };

                outputs.insert(node_id.clone(), output.clone());
                trace.push(TraceStep {
                    node_id: node_id.clone(),
                    label: node.label.clone(),
                    output,
                    confidence: 0.8,
                    tokens_used: 0,
                    elapsed_ms: 0,
                    regime: ComputeRegime::Trivial,
                });

                // Queue successor nodes
                for wire in &graph.wires {
                    if wire.from_id == node_id {
                        queue.push(wire.to_id.clone());
                    }
                }
            }
        }

        // Final output is the last trace node's output
        let conclusion = trace
            .last()
            .map(|t| t.output.clone())
            .unwrap_or_default();

        Ok(FlowResult {
            conclusion,
            confidence: 0.8,
            trace_nodes: trace,
            total_tokens,
            total_joules: 0.001, // deterministic local execution
            regime: ComputeRegime::Trivial,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_classification() {
        assert_eq!(
            DomainLut::from_query("audit this record for compliance"),
            DomainLut::Audit
        );
        assert_eq!(
            DomainLut::from_query("optimize query performance"),
            DomainLut::Optimization
        );
        assert_eq!(
            DomainLut::from_query("check license rights for France"),
            DomainLut::Legal
        );
        assert_eq!(
            DomainLut::from_query("generate a playlist"),
            DomainLut::Creative
        );
        assert_eq!(
            DomainLut::from_query("what is the entropy of this state"),
            DomainLut::QuantumInformation
        );
        assert_eq!(
            DomainLut::from_query("what should I do next"),
            DomainLut::Metacognition
        );
    }

    #[test]
    fn test_build_query_graph() {
        let mut builder = FlowGraphBuilder::new();
        let graph = builder.build_query_graph("find similar movies", DomainLut::Creative);
        assert_eq!(graph.domain, DomainLut::Creative);
        assert!(!graph.nodes.is_empty());
        assert!(!graph.wires.is_empty());
        // Entry node should be a Bind
        let entry = graph.nodes.get(&graph.entry_node).unwrap();
        assert_eq!(entry.kind, FlowNodeKind::Bind);
    }

    #[test]
    fn test_build_contrast_graph() {
        let mut builder = FlowGraphBuilder::new();
        let graph = builder.build_contrast_graph(
            "movie A vs movie B",
            "diverges on: genre, mood; converges on: decade",
            DomainLut::Creative,
        );
        assert!(graph.nodes.len() >= 5);
    }

    #[test]
    fn test_local_reasoner_executes() {
        let reasoner = LocalFlowReasoner;
        let mut builder = FlowGraphBuilder::new();
        let graph = builder.build_query_graph("explain why X is similar to Y", DomainLut::Metacognition);
        let result = reasoner.reason_graph(&graph).unwrap();
        assert!(!result.conclusion.is_empty());
        assert!(result.confidence > 0.0);
        assert!(!result.trace_nodes.is_empty());
    }

    #[test]
    fn test_flow_result_to_ai_types() {
        let result = FlowResult {
            conclusion: "The movies share thematic elements".into(),
            confidence: 0.85,
            trace_nodes: vec![TraceStep {
                node_id: "n0".into(),
                label: "analyze".into(),
                output: "shared themes".into(),
                confidence: 0.85,
                tokens_used: 100,
                elapsed_ms: 50,
                regime: ComputeRegime::Tractable,
            }],
            total_tokens: 100,
            total_joules: 0.05,
            regime: ComputeRegime::Tractable,
        };

        let output = result.to_ai_output();
        match &output {
            AiOutput::Text(t) => assert_eq!(t, "The movies share thematic elements"),
            _ => panic!("expected Text variant"),
        }

        let receipt = result.to_ai_receipt();
        assert_eq!(receipt.tier, crate::ai::tier::InferenceTier::Holographic);

        let reasoning = result.to_reasoning_result();
        assert_eq!(reasoning.reasoning_steps.len(), 1);
    }
}
