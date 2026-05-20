//! Primitive 8: Reflect — apply the system to its own output.
//!
//! Metacognition as a primitive. The system doesn't just detect contrast —
//! it detects *that it detected contrast*. This is the bridge between
//! measurement and understanding.
//!
//! Reflect takes a vector and produces a second-order representation:
//! what this vector means *relative to the system's current state*.

use crate::turbo_holographic::{BinaryHV, BundleAccumulator};

/// The result of reflecting on a vector.
#[derive(Clone, Debug)]
pub struct Reflection {
    /// The original vector being reflected on.
    pub subject: BinaryHV,
    /// The centroid of the system (what "normal" looks like).
    pub context: BinaryHV,
    /// The residual: subject XOR context — what's *different* about this input.
    pub residual: BinaryHV,
    /// The self-model: subject bound with residual — "what I notice about this."
    pub self_model: BinaryHV,
    /// Novelty: how different is this from context (0.0 = identical, 1.0 = opposite).
    pub novelty: f32,
    /// Self-similarity: how much does the self-model resemble the subject?
    /// High = the reflection didn't add much. Low = the context changed the picture.
    pub self_similarity: f32,
}

/// Trait for anything that can reflect on its own representations.
pub trait Reflectable {
    /// Reflect: produce a second-order representation of a vector
    /// relative to a context (the system's current centroid/model).
    fn reflect(&self, context: &BinaryHV) -> Reflection;

    /// Iterate reflection: reflect on the reflection itself.
    /// Each iteration deepens the metacognitive stack.
    /// Returns the chain of reflections from outermost to innermost.
    fn reflect_n(&self, context: &BinaryHV, depth: usize) -> Vec<Reflection>;
}

impl Reflectable for BinaryHV {
    fn reflect(&self, context: &BinaryHV) -> Reflection {
        // Residual: what's different between subject and context
        let residual = self.bind(context); // XOR = difference

        // Self-model: bind the subject with its own residual
        // This creates a representation of "what I am, given what's normal"
        let self_model = self.bind(&residual);

        let novelty = 1.0 - self.similarity(context);
        let self_similarity = self.similarity(&self_model);

        Reflection {
            subject: self.clone(),
            context: context.clone(),
            residual,
            self_model,
            novelty,
            self_similarity,
        }
    }

    fn reflect_n(&self, context: &BinaryHV, depth: usize) -> Vec<Reflection> {
        let mut chain = Vec::with_capacity(depth);
        let mut current = self.clone();

        for _ in 0..depth {
            let reflection = current.reflect(context);
            current = reflection.self_model.clone();
            chain.push(reflection);
        }

        chain
    }
}

impl Reflection {
    /// Bundle all reflections in a chain into a single "understanding" vector.
    /// Deeper reflections contribute equally — understanding is the superposition
    /// of all metacognitive levels.
    pub fn bundle_chain(chain: &[Reflection]) -> BinaryHV {
        if chain.is_empty() {
            return BinaryHV::zeros(10000);
        }
        let dim = chain[0].subject.dimension();
        let mut acc = BundleAccumulator::new(dim);
        for r in chain {
            acc.add(&r.self_model);
        }
        acc.threshold()
    }

    /// The reflection's information content: how much did reflecting change
    /// the representation? If novelty is high but self_similarity is also high,
    /// the subject carries its own explanation. If novelty is high and
    /// self_similarity is low, the context was essential to understanding.
    pub fn information_gain(&self) -> f32 {
        self.novelty * (1.0 - self.self_similarity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflect_identical_to_context() {
        let hv = BinaryHV::random(10000, 42);
        let reflection = hv.reflect(&hv);
        assert!(reflection.novelty < 0.01);
    }

    #[test]
    fn test_reflect_orthogonal_to_context() {
        let hv = BinaryHV::random(10000, 1);
        let ctx = BinaryHV::random(10000, 2);
        let reflection = hv.reflect(&ctx);
        assert!(reflection.novelty > 0.3);
    }

    #[test]
    fn test_reflect_n_depth() {
        let hv = BinaryHV::random(10000, 42);
        let ctx = BinaryHV::random(10000, 99);
        let chain = hv.reflect_n(&ctx, 5);
        assert_eq!(chain.len(), 5);
    }

    #[test]
    fn test_bundle_chain() {
        let hv = BinaryHV::random(10000, 42);
        let ctx = BinaryHV::random(10000, 99);
        let chain = hv.reflect_n(&ctx, 3);
        let understanding = Reflection::bundle_chain(&chain);
        assert_eq!(understanding.dimension(), 10000);
        // Understanding should be related to the original
        let sim = understanding.similarity(&hv);
        assert!(sim > 0.3); // Not identical but related
    }

    #[test]
    fn test_information_gain() {
        let hv = BinaryHV::random(10000, 1);
        let ctx = BinaryHV::random(10000, 2);
        let reflection = hv.reflect(&ctx);
        // Random vectors should have meaningful information gain
        assert!(reflection.information_gain() > 0.0);
    }
}
