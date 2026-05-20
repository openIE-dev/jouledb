//! Symbolic Layer
//!
//! Rule-based reasoning with forward chaining inference.
//! Supports first-order logic with variables.

use super::{NeurosymbolicError, NeurosymbolicResult};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// A logical rule: condition -> conclusion
#[derive(Debug, Clone)]
pub struct Rule {
    /// Rule name/id
    pub name: String,
    /// Condition predicates (premises)
    pub conditions: Vec<Predicate>,
    /// Conclusion predicate
    pub conclusion: Predicate,
}

impl Rule {
    /// Create a new rule
    pub fn new(name: &str, conditions: Vec<Predicate>, conclusion: Predicate) -> Self {
        Self {
            name: name.to_string(),
            conditions,
            conclusion,
        }
    }

    /// Parse a simple rule string: "condition => conclusion"
    pub fn parse(name: &str, rule_str: &str) -> NeurosymbolicResult<Self> {
        let parts: Vec<&str> = rule_str.split("=>").collect();
        if parts.len() != 2 {
            return Err(NeurosymbolicError::InvalidRule(
                "rule must have format 'condition => conclusion'".to_string(),
            ));
        }

        let condition_str = parts[0].trim();
        let conclusion_str = parts[1].trim();

        let conditions = if condition_str.is_empty() {
            Vec::new()
        } else {
            condition_str
                .split(',')
                .map(|s| Predicate::parse(s.trim()))
                .collect::<NeurosymbolicResult<Vec<_>>>()?
        };

        let conclusion = Predicate::parse(conclusion_str)?;

        Ok(Self::new(name, conditions, conclusion))
    }
}

/// A logical predicate: name(arg1, arg2, ...)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Predicate {
    /// Predicate name
    pub name: String,
    /// Arguments (constants or variables)
    pub args: Vec<Term>,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(name: &str, args: Vec<Term>) -> Self {
        Self {
            name: name.to_string(),
            args,
        }
    }

    /// Parse a predicate string: "name(arg1, arg2)"
    pub fn parse(s: &str) -> NeurosymbolicResult<Self> {
        let s = s.trim();

        // Find opening paren
        let paren_pos = s.find('(').ok_or_else(|| {
            NeurosymbolicError::InvalidFact(format!("missing '(' in predicate: {}", s))
        })?;

        let name = s[..paren_pos].trim();
        if name.is_empty() {
            return Err(NeurosymbolicError::InvalidFact(
                "empty predicate name".to_string(),
            ));
        }

        // Find closing paren
        if !s.ends_with(')') {
            return Err(NeurosymbolicError::InvalidFact(format!(
                "missing ')' in predicate: {}",
                s
            )));
        }

        let args_str = &s[paren_pos + 1..s.len() - 1];
        let args = if args_str.trim().is_empty() {
            Vec::new()
        } else {
            args_str.split(',').map(|a| Term::parse(a.trim())).collect()
        };

        Ok(Self::new(name, args))
    }

    /// Apply bindings to get a ground predicate
    pub fn apply_bindings(&self, bindings: &Binding) -> Self {
        Self {
            name: self.name.clone(),
            args: self
                .args
                .iter()
                .map(|t| t.apply_bindings(bindings))
                .collect(),
        }
    }

    /// Check if predicate is ground (no variables)
    pub fn is_ground(&self) -> bool {
        self.args.iter().all(|t| matches!(t, Term::Constant(_)))
    }
}

/// A term: either a constant or a variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// A constant value
    Constant(String),
    /// A variable (starts with uppercase or ?)
    Variable(String),
}

impl Term {
    /// Parse a term
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        // Variables start with uppercase letter or ?
        if s.starts_with('?') || s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            Term::Variable(s.to_string())
        } else {
            Term::Constant(s.to_string())
        }
    }

    /// Apply bindings
    pub fn apply_bindings(&self, bindings: &Binding) -> Self {
        match self {
            Term::Variable(name) => bindings.get(name).cloned().unwrap_or_else(|| self.clone()),
            Term::Constant(_) => self.clone(),
        }
    }
}

/// A fact: a ground predicate that is known to be true
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    /// The predicate
    pub predicate: Predicate,
}

impl Fact {
    /// Create a new fact
    pub fn new(name: &str, args: Vec<&str>) -> Self {
        Self {
            predicate: Predicate::new(
                name,
                args.into_iter()
                    .map(|a| Term::Constant(a.to_string()))
                    .collect(),
            ),
        }
    }

    /// Parse a fact string
    pub fn parse(s: &str) -> NeurosymbolicResult<Self> {
        let predicate = Predicate::parse(s)?;
        if !predicate.is_ground() {
            return Err(NeurosymbolicError::InvalidFact(
                "fact must not contain variables".to_string(),
            ));
        }
        Ok(Self { predicate })
    }
}

/// Variable bindings
pub type Binding = HashMap<String, Term>;

/// Symbolic Reasoner with forward chaining
pub struct SymbolicReasoner {
    /// Known rules
    rules: Arc<RwLock<Vec<Rule>>>,
    /// Known facts
    facts: Arc<RwLock<HashSet<Fact>>>,
}

impl SymbolicReasoner {
    /// Create new symbolic reasoner
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            facts: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Add a rule
    pub fn add_rule(&self, rule: Rule) {
        self.rules.write().unwrap().push(rule);
    }

    /// Add a rule from string
    pub fn add_rule_str(&self, name: &str, rule_str: &str) -> NeurosymbolicResult<()> {
        let rule = Rule::parse(name, rule_str)?;
        self.add_rule(rule);
        Ok(())
    }

    /// Add a fact
    pub fn add_fact(&self, fact: Fact) {
        self.facts.write().unwrap().insert(fact);
    }

    /// Add a fact from string
    pub fn add_fact_str(&self, fact_str: &str) -> NeurosymbolicResult<()> {
        let fact = Fact::parse(fact_str)?;
        self.add_fact(fact);
        Ok(())
    }

    /// Add a simple fact
    pub fn add_simple_fact(&self, predicate: &str, args: Vec<&str>) {
        self.add_fact(Fact::new(predicate, args));
    }

    /// Get number of facts
    pub fn fact_count(&self) -> usize {
        self.facts.read().unwrap().len()
    }

    /// Get number of rules
    pub fn rule_count(&self) -> usize {
        self.rules.read().unwrap().len()
    }

    /// Forward chaining inference
    ///
    /// Derives new facts by applying rules until fixed point.
    pub fn forward_chain(&self) -> usize {
        let mut new_facts_count = 0;
        let mut changed = true;

        while changed {
            changed = false;
            let rules = self.rules.read().unwrap().clone();

            for rule in &rules {
                let new_facts = self.apply_rule(rule);
                for fact in new_facts {
                    let mut facts = self.facts.write().unwrap();
                    if !facts.contains(&fact) {
                        facts.insert(fact);
                        new_facts_count += 1;
                        changed = true;
                    }
                }
            }
        }

        new_facts_count
    }

    /// Apply a rule to derive new facts
    fn apply_rule(&self, rule: &Rule) -> Vec<Fact> {
        let facts = self.facts.read().unwrap();

        if rule.conditions.is_empty() {
            // Rule with no conditions - always applies
            if rule.conclusion.is_ground() {
                return vec![Fact {
                    predicate: rule.conclusion.clone(),
                }];
            }
            return Vec::new();
        }

        // Find all bindings that satisfy the conditions
        let bindings = self.find_bindings(&rule.conditions, &facts);

        // For each binding, generate the conclusion
        bindings
            .into_iter()
            .filter_map(|binding| {
                let conclusion = rule.conclusion.apply_bindings(&binding);
                if conclusion.is_ground() {
                    Some(Fact {
                        predicate: conclusion,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find all bindings that satisfy a set of conditions
    fn find_bindings(&self, conditions: &[Predicate], facts: &HashSet<Fact>) -> Vec<Binding> {
        if conditions.is_empty() {
            return vec![HashMap::new()];
        }

        let first = &conditions[0];
        let rest = &conditions[1..];

        let mut results = Vec::new();

        // Try to match first condition against each fact
        for fact in facts {
            if fact.predicate.name == first.name && fact.predicate.args.len() == first.args.len() {
                if let Some(binding) = Self::unify(first, &fact.predicate) {
                    // Recursively find bindings for rest
                    let extended_conditions: Vec<Predicate> =
                        rest.iter().map(|c| c.apply_bindings(&binding)).collect();

                    let sub_bindings = self.find_bindings(&extended_conditions, facts);

                    for sub_binding in sub_bindings {
                        let mut combined = binding.clone();
                        combined.extend(sub_binding);
                        results.push(combined);
                    }
                }
            }
        }

        results
    }

    /// Unify a pattern with a ground predicate
    fn unify(pattern: &Predicate, ground: &Predicate) -> Option<Binding> {
        if pattern.name != ground.name || pattern.args.len() != ground.args.len() {
            return None;
        }

        let mut binding = HashMap::new();

        for (pat_term, ground_term) in pattern.args.iter().zip(ground.args.iter()) {
            match (pat_term, ground_term) {
                (Term::Constant(c1), Term::Constant(c2)) => {
                    if c1 != c2 {
                        return None;
                    }
                }
                (Term::Variable(var), Term::Constant(_val)) => {
                    if let Some(existing) = binding.get(var) {
                        if existing != ground_term {
                            return None;
                        }
                    } else {
                        binding.insert(var.clone(), ground_term.clone());
                    }
                }
                _ => return None, // Can't unify variable with variable
            }
        }

        Some(binding)
    }

    /// Query for facts matching a pattern
    pub fn query(&self, pattern: &Predicate) -> Vec<(Fact, Binding)> {
        let facts = self.facts.read().unwrap();
        let mut results = Vec::new();

        for fact in facts.iter() {
            if let Some(binding) = Self::unify(pattern, &fact.predicate) {
                results.push((fact.clone(), binding));
            }
        }

        results
    }

    /// Query from string
    pub fn query_str(&self, query_str: &str) -> NeurosymbolicResult<Vec<(Fact, Binding)>> {
        let pattern = Predicate::parse(query_str)?;
        Ok(self.query(&pattern))
    }
}

impl Default for SymbolicReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SymbolicReasoner {
    fn clone(&self) -> Self {
        Self {
            rules: Arc::new(RwLock::new(self.rules.read().unwrap().clone())),
            facts: Arc::new(RwLock::new(self.facts.read().unwrap().clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_parse() {
        let p = Predicate::parse("human(socrates)").unwrap();
        assert_eq!(p.name, "human");
        assert_eq!(p.args.len(), 1);
        assert!(matches!(&p.args[0], Term::Constant(s) if s == "socrates"));
    }

    #[test]
    fn test_predicate_with_variable() {
        let p = Predicate::parse("human(X)").unwrap();
        assert_eq!(p.name, "human");
        assert!(matches!(&p.args[0], Term::Variable(s) if s == "X"));
    }

    #[test]
    fn test_rule_parse() {
        let rule = Rule::parse("r1", "human(X) => mortal(X)").unwrap();
        assert_eq!(rule.name, "r1");
        assert_eq!(rule.conditions.len(), 1);
        assert_eq!(rule.conditions[0].name, "human");
        assert_eq!(rule.conclusion.name, "mortal");
    }

    #[test]
    fn test_forward_chaining() {
        let reasoner = SymbolicReasoner::new();

        // Add rule: human(X) => mortal(X)
        reasoner
            .add_rule_str("r1", "human(X) => mortal(X)")
            .unwrap();

        // Add fact: human(socrates)
        reasoner.add_fact_str("human(socrates)").unwrap();

        // Run inference
        let new_facts = reasoner.forward_chain();
        assert_eq!(new_facts, 1);

        // Query for mortal(socrates)
        let results = reasoner.query_str("mortal(socrates)").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_transitive_reasoning() {
        let reasoner = SymbolicReasoner::new();

        // parent(X, Y), parent(Y, Z) => grandparent(X, Z)
        // This is more complex - let's do simpler chain

        // a(X) => b(X)
        // b(X) => c(X)
        reasoner.add_rule_str("r1", "a(X) => b(X)").unwrap();
        reasoner.add_rule_str("r2", "b(X) => c(X)").unwrap();

        reasoner.add_fact_str("a(foo)").unwrap();

        let new_facts = reasoner.forward_chain();
        assert_eq!(new_facts, 2); // b(foo) and c(foo)

        let results = reasoner.query_str("c(foo)").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_with_variable() {
        let reasoner = SymbolicReasoner::new();

        reasoner.add_fact_str("likes(alice, bob)").unwrap();
        reasoner.add_fact_str("likes(alice, charlie)").unwrap();
        reasoner.add_fact_str("likes(bob, alice)").unwrap();

        // Query: who does alice like?
        let results = reasoner.query_str("likes(alice, X)").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_multiple_conditions() {
        let reasoner = SymbolicReasoner::new();

        // parent(X, Y), parent(Y, Z) => grandparent(X, Z)
        let rule = Rule::new(
            "grandparent_rule",
            vec![
                Predicate::parse("parent(X, Y)").unwrap(),
                Predicate::parse("parent(Y, Z)").unwrap(),
            ],
            Predicate::parse("grandparent(X, Z)").unwrap(),
        );
        reasoner.add_rule(rule);

        reasoner.add_fact_str("parent(alice, bob)").unwrap();
        reasoner.add_fact_str("parent(bob, charlie)").unwrap();

        reasoner.forward_chain();

        let results = reasoner.query_str("grandparent(alice, charlie)").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_empty_reasoner() {
        let reasoner = SymbolicReasoner::new();
        assert_eq!(reasoner.fact_count(), 0);
        assert_eq!(reasoner.rule_count(), 0);

        let new_facts = reasoner.forward_chain();
        assert_eq!(new_facts, 0);
    }
}
