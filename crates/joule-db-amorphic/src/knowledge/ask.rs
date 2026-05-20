//! Ask: the end-to-end pipeline.
//!
//! Type a question. Get an answer. See where it breaks.
//!
//! Full pipeline:
//! 1. Parse the question into intent + subject + context
//! 2. Contrast gate: is this novel relative to the core?
//! 3. If novel → Oracle lookup → expand core
//! 4. Traverse the knowledge manifold for structural answer
//! 5. Generate text from traversal path + sequence memory
//! 6. Return answer + trace of every step

use super::concept::ConceptEncoder;
use super::context::ContextWindow;
use super::core::KnowledgeCore;
use super::generate::{Generator, GenerationResult};
use super::oracle::{Oracle, OracleSource};
use super::relation::RelationType;
use super::traverse::{Traverser, TraversalResult};

/// The full pipeline: question → answer.
pub struct Ask {
    pub core: KnowledgeCore,
    pub oracle: Oracle,
    pub generator: Generator,
    pub encoder: ConceptEncoder,
    pub traverser: Traverser,
    /// SDM-backed conversation context (replaces fixed window).
    pub context: ContextWindow,
    /// Novelty threshold for triggering oracle expansion.
    pub novelty_threshold: f64,
}

impl Ask {
    /// Create with bootstrap knowledge + optional oracle backends.
    pub fn new() -> Self {
        let mut core = KnowledgeCore::new();
        let bootstrap = super::ingest::ConceptNetParser::bootstrap_core();
        core.ingest_batch(&bootstrap);

        let mut encoder = super::concept::ConceptEncoder::with_default_dim();

        // Train generator on bootstrap sentence patterns
        let mut generator = Generator::new(3);
        generator.train(&SEED_CORPUS, &mut encoder);

        Self {
            core,
            oracle: Oracle::default(),
            generator,
            encoder,
            traverser: Traverser::new().with_max_steps(5),
            context: ContextWindow::new(500, super::concept::KNOWLEDGE_DIM),
            novelty_threshold: 0.55,
        }
    }

    /// Ask a question. Get an answer with full trace.
    pub fn ask(&mut self, question: &str) -> Answer {
        let mut trace = Vec::new();

        // Step 1: Parse the question
        let parsed = self.parse_question(question);
        trace.push(format!(
            "PARSE: intent={:?}, subject='{}', keywords={:?}",
            parsed.intent, parsed.subject, parsed.keywords
        ));

        // Step 2: Contrast gate — is the subject known?
        let novelty = self.core.novelty(&parsed.subject);
        let subject_known = self.core.query_concept(&parsed.subject).is_some();
        trace.push(format!(
            "CONTRAST: novelty={:.3}, subject_known={}",
            novelty, subject_known
        ));

        // Step 3: Oracle expansion if novel
        let mut oracle_used = false;
        if novelty > self.novelty_threshold || !subject_known {
            let (result, added) = self.oracle.query_and_ingest(&parsed.subject, &mut self.core);
            oracle_used = true;
            trace.push(format!(
                "ORACLE: source={:?}, triples_added={}, related={}",
                result.source,
                added,
                result.related.len()
            ));

            // Also look up keywords
            for kw in &parsed.keywords {
                if self.core.query_concept(kw).is_none() {
                    let (r, a) = self.oracle.query_and_ingest(kw, &mut self.core);
                    if a > 0 {
                        trace.push(format!("ORACLE: expanded '{}' +{} triples", kw, a));
                    }
                }
            }
        }

        // Write question into SDM context for conversational memory
        let question_hv = self.encoder.encode(question).vector;
        self.context.write(question_hv, question);

        // Step 4: Answer based on intent
        let (text, method) = match parsed.intent {
            Intent::WhatIs => self.answer_what_is(&parsed, &mut trace),
            Intent::HowRelated => self.answer_how_related(&parsed, &mut trace),
            Intent::CanIt => self.answer_can_it(&parsed, &mut trace),
            Intent::WhereIs => self.answer_where_is(&parsed, &mut trace),
            Intent::Generate => self.answer_generate(&parsed, &mut trace),
            Intent::Unknown => self.answer_fallback(&parsed, &mut trace),
        };

        // Write answer into SDM context
        let answer_hv = self.encoder.encode(&text).vector;
        self.context.write(answer_hv, &text);

        // Context stats
        trace.push(format!(
            "CONTEXT: {} items, cleanup success rate={:.0}%",
            self.context.len(),
            self.core.cleanup.success_rate() * 100.0,
        ));

        Answer {
            question: question.to_string(),
            answer: text,
            method,
            oracle_used,
            novelty,
            trace,
        }
    }

    // ================================================================
    // Question parsing
    // ================================================================

    fn parse_question(&mut self, question: &str) -> ParsedQuestion {
        let lower = question.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Intent detection
        let intent = if lower.starts_with("what is") || lower.starts_with("what's") {
            Intent::WhatIs
        } else if lower.contains("related") || lower.contains("connection") || lower.contains("between") {
            Intent::HowRelated
        } else if lower.starts_with("can") || lower.contains("capable") || lower.contains("able to") {
            Intent::CanIt
        } else if lower.starts_with("where") || lower.contains("location") || lower.contains("found") {
            Intent::WhereIs
        } else if lower.starts_with("tell me about") || lower.starts_with("describe") || lower.starts_with("explain") {
            Intent::Generate
        } else {
            Intent::Unknown
        };

        // Extract subject: the most content-bearing word(s)
        let stop_words = [
            "what", "is", "a", "an", "the", "are", "how", "can", "does", "do",
            "where", "why", "tell", "me", "about", "describe", "explain", "and",
            "or", "of", "to", "in", "at", "on", "for", "with", "between",
            "related", "connection", "capable", "able", "it",
        ];

        let keywords: Vec<String> = words
            .iter()
            .filter(|w| w.len() > 2 && !stop_words.contains(&w.as_ref()))
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let subject = keywords.first().cloned().unwrap_or_default();

        ParsedQuestion {
            intent,
            subject,
            keywords,
            raw: question.to_string(),
        }
    }

    // ================================================================
    // Answer strategies
    // ================================================================

    fn answer_what_is(&mut self, parsed: &ParsedQuestion, trace: &mut Vec<String>) -> (String, String) {
        // Query with cleanup: "subject IsA ?" → denoise → clean concept
        let is_a_results = self.core.query_object_clean(&parsed.subject, RelationType::IsA, 3);
        trace.push(format!("WHAT_IS: cleaned IsA results: {:?}", is_a_results));

        if let Some((concept, sim)) = is_a_results.first() {
            if *sim > 0.50 {
                // Also get cleaned properties
                let properties = self.get_properties_clean(&parsed.subject);
                let props_str = if properties.is_empty() {
                    String::new()
                } else {
                    format!(". It is {}", properties.join(", "))
                };

                return (
                    format!("{} is a {}{}", parsed.subject, concept, props_str),
                    "what_is_clean".to_string(),
                );
            }
        }

        // Fallback: raw unbind if cleanup didn't find a match
        if let Some(recovered) = self.core.query_object(&parsed.subject, RelationType::IsA) {
            let nearest = self.core.nearest_concepts(&recovered, 3);
            trace.push(format!("WHAT_IS: raw nearest to IsA: {:?}", nearest));

            if let Some((concept, sim)) = nearest.first() {
                if *sim > 0.45 {
                    let properties = self.get_properties(&parsed.subject);
                    let props_str = if properties.is_empty() {
                        String::new()
                    } else {
                        format!(". It is {}", properties.join(", "))
                    };

                    return (
                        format!("{} is a {}{}", parsed.subject, concept, props_str),
                        "what_is_unbind".to_string(),
                    );
                }
            }
        }

        // Fallback: traverse from subject
        let traversal = self.traverser.traverse(&mut self.core, &parsed.subject);
        trace.push(format!("WHAT_IS: traversal = {}", traversal.render()));

        if traversal.path.len() > 1 {
            let connections: Vec<String> = traversal.path[1..]
                .iter()
                .map(|s| {
                    if let Some(rel) = &s.via_relation {
                        format!("{:?} {}", rel, s.concept)
                    } else {
                        s.concept.clone()
                    }
                })
                .collect();
            (
                format!("{} is related to: {}", parsed.subject, connections.join(", ")),
                "what_is_traversal".to_string(),
            )
        } else {
            (
                format!("I don't have enough information about '{}'", parsed.subject),
                "what_is_unknown".to_string(),
            )
        }
    }

    fn answer_how_related(&mut self, parsed: &ParsedQuestion, trace: &mut Vec<String>) -> (String, String) {
        if parsed.keywords.len() < 2 {
            return (
                "I need two concepts to compare.".to_string(),
                "how_related_insufficient".to_string(),
            );
        }

        let a = &parsed.keywords[0];
        let b = &parsed.keywords[1];

        let relatedness = self.core.relatedness(a, b);
        trace.push(format!("HOW_RELATED: {}~{} = {:.3}", a, b, relatedness));

        // Try directed traversal
        let traversal = self.traverser.traverse_toward(&mut self.core, a, b);
        trace.push(format!("HOW_RELATED: path = {}", traversal.render()));

        if traversal.path.len() > 1 {
            let path_str: Vec<String> = traversal.path.iter().map(|s| s.concept.clone()).collect();
            (
                format!(
                    "{} and {} are related (similarity: {:.0}%). Path: {}",
                    a,
                    b,
                    relatedness * 100.0,
                    path_str.join(" → ")
                ),
                "how_related_traversal".to_string(),
            )
        } else {
            (
                format!(
                    "{} and {} have a similarity of {:.0}%",
                    a,
                    b,
                    relatedness * 100.0
                ),
                "how_related_score".to_string(),
            )
        }
    }

    fn answer_can_it(&mut self, parsed: &ParsedQuestion, trace: &mut Vec<String>) -> (String, String) {
        // Try cleanup first
        let clean_results = self.core.query_object_clean(&parsed.subject, RelationType::CapableOf, 5);
        trace.push(format!("CAN_IT: cleaned CapableOf: {:?}", clean_results));

        if !clean_results.is_empty() {
            let capabilities: Vec<String> = clean_results.into_iter().map(|(c, _)| c).collect();
            return (
                format!("{} can: {}", parsed.subject, capabilities.join(", ")),
                "can_it_clean".to_string(),
            );
        }

        // Fallback: raw unbind
        if let Some(recovered) = self.core.query_object(&parsed.subject, RelationType::CapableOf) {
            let nearest = self.core.nearest_concepts(&recovered, 5);
            let capabilities: Vec<String> = nearest
                .iter()
                .filter(|(_, sim)| *sim > 0.45)
                .map(|(c, _)| c.clone())
                .collect();
            if !capabilities.is_empty() {
                return (
                    format!("{} can: {}", parsed.subject, capabilities.join(", ")),
                    "can_it_unbind".to_string(),
                );
            }
        }

        (
            format!("I'm not sure what {} can do", parsed.subject),
            "can_it_unknown".to_string(),
        )
    }

    fn answer_where_is(&mut self, parsed: &ParsedQuestion, trace: &mut Vec<String>) -> (String, String) {
        // Try cleanup first
        let clean_results = self.core.query_object_clean(&parsed.subject, RelationType::AtLocation, 3);
        trace.push(format!("WHERE_IS: cleaned AtLocation: {:?}", clean_results));

        if let Some((location, sim)) = clean_results.first() {
            return (
                format!("{} can be found at/in {}", parsed.subject, location),
                "where_is_clean".to_string(),
            );
        }

        // Fallback: raw unbind
        if let Some(recovered) = self.core.query_object(&parsed.subject, RelationType::AtLocation) {
            let nearest = self.core.nearest_concepts(&recovered, 3);
            if let Some((location, sim)) = nearest.first() {
                if *sim > 0.45 {
                    return (
                        format!("{} can be found at/in {}", parsed.subject, location),
                        "where_is_unbind".to_string(),
                    );
                }
            }
        }

        (
            format!("I don't know where {} is found", parsed.subject),
            "where_is_unknown".to_string(),
        )
    }

    fn answer_generate(&mut self, parsed: &ParsedQuestion, trace: &mut Vec<String>) -> (String, String) {
        // Use the full generator with knowledge core fallback
        let prompt = if parsed.keywords.is_empty() {
            parsed.raw.clone()
        } else {
            parsed.keywords.join(" ")
        };

        let result = self.generator.generate(&prompt, &mut self.encoder, Some(&mut self.core));
        trace.push(format!(
            "GENERATE: {} words, confidence={:.3}",
            result.length(),
            result.total_confidence
        ));

        for step in &result.steps {
            trace.push(format!(
                "  GEN_STEP: '{}' via {} (conf={:.2})",
                step.word, step.source, step.confidence
            ));
        }

        (result.render(), "generate".to_string())
    }

    fn answer_fallback(&mut self, parsed: &ParsedQuestion, trace: &mut Vec<String>) -> (String, String) {
        // Try traversal from subject, then generation
        if !parsed.subject.is_empty() {
            let traversal = self.traverser.traverse(&mut self.core, &parsed.subject);
            trace.push(format!("FALLBACK: traversal = {}", traversal.render()));

            if traversal.path.len() > 1 {
                let concepts: Vec<String> =
                    traversal.path.iter().map(|s| s.concept.clone()).collect();
                return (
                    format!(
                        "Here's what I know about {}: {}",
                        parsed.subject,
                        concepts.join(" → ")
                    ),
                    "fallback_traversal".to_string(),
                );
            }
        }

        (
            format!(
                "I don't have enough information to answer '{}'. My knowledge core is small — ask me about animals, objects, or basic concepts.",
                parsed.raw
            ),
            "fallback_unknown".to_string(),
        )
    }

    // ================================================================
    // Helpers
    // ================================================================

    fn get_properties(&mut self, concept: &str) -> Vec<String> {
        if let Some(recovered) = self.core.query_object(concept, RelationType::HasProperty) {
            let nearest = self.core.nearest_concepts(&recovered, 5);
            nearest
                .iter()
                .filter(|(_, sim)| *sim > 0.45)
                .map(|(c, _)| c.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get properties with cleanup denoising — should produce much cleaner results.
    fn get_properties_clean(&mut self, concept: &str) -> Vec<String> {
        self.core
            .query_object_clean(concept, RelationType::HasProperty, 3)
            .into_iter()
            .map(|(label, _)| label)
            .collect()
    }
}

impl Default for Ask {
    fn default() -> Self {
        Self::new()
    }
}

/// A parsed question.
#[derive(Debug)]
struct ParsedQuestion {
    intent: Intent,
    subject: String,
    keywords: Vec<String>,
    raw: String,
}

/// Question intent categories.
#[derive(Debug, PartialEq, Eq)]
enum Intent {
    WhatIs,
    HowRelated,
    CanIt,
    WhereIs,
    Generate,
    Unknown,
}

/// The answer + full diagnostic trace.
#[derive(Clone, Debug)]
pub struct Answer {
    /// The original question.
    pub question: String,
    /// The generated answer text.
    pub answer: String,
    /// Which method produced the answer.
    pub method: String,
    /// Whether the oracle was consulted.
    pub oracle_used: bool,
    /// Novelty of the question subject.
    pub novelty: f64,
    /// Step-by-step trace of the pipeline.
    pub trace: Vec<String>,
}

impl Answer {
    /// Pretty-print the answer + trace.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Q: {}\n", self.question));
        out.push_str(&format!("A: {}\n", self.answer));
        out.push_str(&format!(
            "   [method={}, oracle={}, novelty={:.2}]\n",
            self.method, self.oracle_used, self.novelty
        ));
        out.push_str("   Trace:\n");
        for step in &self.trace {
            out.push_str(&format!("     {}\n", step));
        }
        out
    }
}

/// Seed corpus for the generator — basic sentence patterns.
const SEED_CORPUS: [&str; 30] = [
    "a dog is a loyal animal",
    "a cat is an independent animal",
    "a bird is an animal that can fly",
    "a fish is an animal that can swim",
    "a dog can bark and run",
    "a cat can purr and climb",
    "a car is a vehicle used for transportation",
    "water is a substance that is wet",
    "fire is hot and causes heat",
    "ice is cold and made of water",
    "a human is an animal that can think",
    "a human can speak and learn",
    "the brain is part of a human",
    "the heart is part of an animal",
    "a wheel is part of a car",
    "a leaf is part of a plant",
    "food is used for eating and energy",
    "a tool is used for building things",
    "a house is a place where humans live",
    "a city is a place with many houses",
    "rain causes things to be wet",
    "learning causes knowledge to grow",
    "a dog lives in a house with humans",
    "a fish lives in water like the ocean",
    "a bird lives in the sky and trees",
    "an engine is part of a car",
    "metal is a substance used for building",
    "wood is a substance used for houses",
    "language is used for communication",
    "the animal kingdom is diverse and large",
];

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::oracle::InMemoryBackend;

    fn build_ask_with_oracle() -> Ask {
        let mut ask = Ask::new();

        // Add oracle backend with some extra knowledge
        let mut backend = InMemoryBackend::new("test_ucg");
        backend.add_concept(
            "whale",
            Some(vec![0.1; 479]),
            None,
            vec![
                ("mammal".to_string(), RelationType::IsA, 0.95),
                ("ocean".to_string(), RelationType::AtLocation, 0.8),
                ("swim".to_string(), RelationType::CapableOf, 0.9),
                ("large".to_string(), RelationType::HasProperty, 0.85),
            ],
        );
        backend.add_concept(
            "dolphin",
            Some(vec![0.15; 479]),
            None,
            vec![
                ("mammal".to_string(), RelationType::IsA, 0.95),
                ("ocean".to_string(), RelationType::AtLocation, 0.85),
                ("echolocation".to_string(), RelationType::CapableOf, 0.7),
            ],
        );
        backend.add_concept(
            "python",
            None,
            None,
            vec![
                ("programming_language".to_string(), RelationType::IsA, 0.9),
                ("snake".to_string(), RelationType::IsA, 0.85),
            ],
        );

        ask.oracle.register_backend(Box::new(backend));
        ask
    }

    #[test]
    fn test_what_is_known_concept() {
        let mut ask = Ask::new();
        let answer = ask.ask("What is a dog?");
        assert!(!answer.answer.is_empty());
        assert!(!answer.trace.is_empty());
        // Dog is in bootstrap — should not need oracle
        assert!(!answer.oracle_used);
    }

    #[test]
    fn test_what_is_unknown_triggers_oracle() {
        let mut ask = build_ask_with_oracle();
        let answer = ask.ask("What is a whale?");
        assert!(answer.oracle_used, "whale should trigger oracle");
        assert!(
            answer.answer.contains("whale") || answer.answer.contains("mammal"),
            "answer should mention whale or mammal: {}",
            answer.answer
        );
    }

    #[test]
    fn test_how_related() {
        let mut ask = Ask::new();
        let answer = ask.ask("How are dog and cat related?");
        assert!(!answer.answer.is_empty());
        assert!(
            answer.answer.contains("dog") && answer.answer.contains("cat"),
            "should mention both concepts: {}",
            answer.answer
        );
    }

    #[test]
    fn test_can_it() {
        let mut ask = Ask::new();
        let answer = ask.ask("Can a bird fly?");
        assert!(!answer.answer.is_empty());
    }

    #[test]
    fn test_where_is() {
        let mut ask = Ask::new();
        let answer = ask.ask("Where is a fish?");
        assert!(!answer.answer.is_empty());
    }

    #[test]
    fn test_generate() {
        let mut ask = Ask::new();
        let answer = ask.ask("Tell me about dogs");
        assert!(!answer.answer.is_empty());
    }

    #[test]
    fn test_unknown_question() {
        let mut ask = Ask::new();
        let answer = ask.ask("flurble garbonzo");
        assert!(!answer.answer.is_empty());
        // Should gracefully handle nonsense
    }

    #[test]
    fn test_answer_trace() {
        let mut ask = Ask::new();
        let answer = ask.ask("What is a dog?");
        // Trace should have at least PARSE and CONTRAST steps
        assert!(
            answer.trace.len() >= 2,
            "trace should have multiple steps: {:?}",
            answer.trace
        );
        assert!(answer.trace[0].starts_with("PARSE:"));
        assert!(answer.trace[1].starts_with("CONTRAST:"));
    }

    #[test]
    fn test_render() {
        let mut ask = Ask::new();
        let answer = ask.ask("What is a cat?");
        let rendered = answer.render();
        assert!(rendered.contains("Q:"));
        assert!(rendered.contains("A:"));
        assert!(rendered.contains("Trace:"));
    }

    #[test]
    fn test_oracle_expansion_improves_answers() {
        let mut ask = build_ask_with_oracle();

        // First ask about whale without oracle
        let novelty_before = ask.core.novelty("whale");

        // Ask triggers oracle
        let answer = ask.ask("What is a whale?");
        assert!(answer.oracle_used);

        // Now whale should be in the core
        let whale_known = ask.core.query_concept("whale");
        assert!(whale_known.is_some(), "whale should be known after oracle");

        // Second ask should be faster (cached)
        let answer2 = ask.ask("What is a whale?");
        // Oracle still "used" because novelty check happens before cache
        // but the oracle itself should hit cache
    }

    #[test]
    fn test_full_conversation() {
        let mut ask = build_ask_with_oracle();

        let a1 = ask.ask("What is a dog?");
        let a2 = ask.ask("What is a whale?");
        let a3 = ask.ask("How are whale and dolphin related?");
        let a4 = ask.ask("Can a fish swim?");
        let a5 = ask.ask("Where is a bird?");

        // All should produce non-empty answers
        for (i, a) in [&a1, &a2, &a3, &a4, &a5].iter().enumerate() {
            assert!(
                !a.answer.is_empty(),
                "question {} produced empty answer: {}",
                i + 1,
                a.question
            );
        }

        // Print the conversation for human inspection
        // (visible with --nocapture)
        for a in [&a1, &a2, &a3, &a4, &a5] {
            eprintln!("{}", a.render());
        }
    }
}
