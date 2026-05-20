//! ConceptNet parser: TSV → knowledge triples.
//!
//! ConceptNet data format (assertions.csv):
//! ```text
//! /a/[assertion_id] /r/RelationType /c/en/subject /c/en/object {"weight": 1.0, ...}
//! ```
//!
//! We parse this into Triple structs and feed them into the KnowledgeCore.
//! Only English concepts are ingested (filtering on /c/en/).

use super::relation::RelationType;
use super::triple::Triple;

/// Parser for ConceptNet TSV/CSV format.
pub struct ConceptNetParser {
    /// Minimum edge weight to include (filters low-confidence edges).
    pub min_weight: f64,
    /// Only include English concepts.
    pub english_only: bool,
    /// Maximum concept label length (filters overly specific phrases).
    pub max_label_len: usize,
    /// Statistics
    pub lines_parsed: u64,
    pub triples_accepted: u64,
    pub triples_rejected: u64,
}

impl ConceptNetParser {
    pub fn new() -> Self {
        Self {
            min_weight: 1.0,
            english_only: true,
            max_label_len: 50,
            lines_parsed: 0,
            triples_accepted: 0,
            triples_rejected: 0,
        }
    }

    /// Set minimum weight filter.
    pub fn with_min_weight(mut self, w: f64) -> Self {
        self.min_weight = w;
        self
    }

    /// Parse a single ConceptNet TSV line.
    /// Format: assertion_uri \t relation \t subject \t object \t json_metadata
    pub fn parse_line(&mut self, line: &str) -> Option<Triple> {
        self.lines_parsed += 1;

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            self.triples_rejected += 1;
            return None;
        }

        let relation_str = fields[1];
        let subject_uri = fields[2];
        let object_uri = fields[3];
        let metadata = fields[4];

        // Filter: English only
        if self.english_only {
            if !subject_uri.starts_with("/c/en/") || !object_uri.starts_with("/c/en/") {
                self.triples_rejected += 1;
                return None;
            }
        }

        // Extract concept labels from URIs: /c/en/dog → dog
        let subject = self.extract_label(subject_uri)?;
        let object = self.extract_label(object_uri)?;

        // Filter: label length
        if subject.len() > self.max_label_len || object.len() > self.max_label_len {
            self.triples_rejected += 1;
            return None;
        }

        // Parse relation
        let relation = RelationType::from_conceptnet(relation_str);

        // Parse weight from JSON metadata
        let weight = self.extract_weight(metadata);
        if weight < self.min_weight {
            self.triples_rejected += 1;
            return None;
        }

        self.triples_accepted += 1;
        Some(Triple::new(&subject, relation, &object).with_weight(weight))
    }

    /// Parse multiple lines (e.g., from a file reader).
    pub fn parse_lines<'a>(&mut self, lines: impl Iterator<Item = &'a str>) -> Vec<Triple> {
        lines.filter_map(|line| self.parse_line(line)).collect()
    }

    /// Build triples from simple (subject, relation, object) tuples.
    /// For bootstrapping without ConceptNet files.
    pub fn from_tuples(tuples: &[(&str, &str, &str)]) -> Vec<Triple> {
        tuples
            .iter()
            .map(|(s, r, o)| {
                let relation = RelationType::from_conceptnet(r);
                Triple::new(s, relation, o)
            })
            .collect()
    }

    /// Build a minimal bootstrap knowledge set — the structural skeleton.
    /// These are the most fundamental category relationships that form
    /// the backbone of common-sense knowledge.
    pub fn bootstrap_core() -> Vec<Triple> {
        Self::from_tuples(&[
            // Taxonomy backbone
            ("entity", "IsA", "thing"),
            ("physical_entity", "IsA", "entity"),
            ("abstract_entity", "IsA", "entity"),
            ("living_thing", "IsA", "physical_entity"),
            ("non_living_thing", "IsA", "physical_entity"),
            ("animal", "IsA", "living_thing"),
            ("plant", "IsA", "living_thing"),
            ("human", "IsA", "animal"),
            ("object", "IsA", "non_living_thing"),
            ("substance", "IsA", "non_living_thing"),
            ("place", "IsA", "physical_entity"),
            ("event", "IsA", "abstract_entity"),
            ("action", "IsA", "event"),
            ("state", "IsA", "abstract_entity"),
            ("property", "IsA", "abstract_entity"),
            ("relation", "IsA", "abstract_entity"),
            ("quantity", "IsA", "abstract_entity"),
            ("time", "IsA", "abstract_entity"),
            // Common animals
            ("dog", "IsA", "animal"),
            ("cat", "IsA", "animal"),
            ("bird", "IsA", "animal"),
            ("fish", "IsA", "animal"),
            ("horse", "IsA", "animal"),
            ("cow", "IsA", "animal"),
            // Common objects
            ("car", "IsA", "vehicle"),
            ("vehicle", "IsA", "object"),
            ("tool", "IsA", "object"),
            ("food", "IsA", "substance"),
            ("water", "IsA", "substance"),
            ("house", "IsA", "place"),
            ("city", "IsA", "place"),
            ("country", "IsA", "place"),
            // Properties
            ("dog", "HasProperty", "loyal"),
            ("cat", "HasProperty", "independent"),
            ("water", "HasProperty", "wet"),
            ("fire", "HasProperty", "hot"),
            ("ice", "HasProperty", "cold"),
            // Capabilities
            ("human", "CapableOf", "think"),
            ("human", "CapableOf", "speak"),
            ("bird", "CapableOf", "fly"),
            ("fish", "CapableOf", "swim"),
            ("dog", "CapableOf", "bark"),
            // Causality
            ("fire", "Causes", "heat"),
            ("rain", "Causes", "wet"),
            ("learning", "Causes", "knowledge"),
            ("eating", "Causes", "energy"),
            // Part-whole
            ("wheel", "PartOf", "car"),
            ("engine", "PartOf", "car"),
            ("leaf", "PartOf", "plant"),
            ("brain", "PartOf", "human"),
            ("heart", "PartOf", "animal"),
            // Locations
            ("human", "AtLocation", "city"),
            ("fish", "AtLocation", "water"),
            ("bird", "AtLocation", "sky"),
            // Used for
            ("car", "UsedFor", "transportation"),
            ("food", "UsedFor", "eating"),
            ("tool", "UsedFor", "building"),
            ("language", "UsedFor", "communication"),
            // Made of
            ("house", "MadeOf", "wood"),
            ("car", "MadeOf", "metal"),
            ("ice", "MadeOf", "water"),
        ])
    }

    // Internal helpers

    fn extract_label(&self, uri: &str) -> Option<String> {
        // /c/en/dog/n/wn/animal → dog
        // /c/en/run_away → run_away
        let parts: Vec<&str> = uri.split('/').collect();
        if parts.len() >= 4 {
            Some(parts[3].to_string())
        } else {
            None
        }
    }

    fn extract_weight(&self, metadata: &str) -> f64 {
        // Simple JSON parsing for "weight" field
        if let Some(pos) = metadata.find("\"weight\"") {
            let after = &metadata[pos + 8..];
            if let Some(colon) = after.find(':') {
                let value_str = &after[colon + 1..];
                let end = value_str
                    .find(|c: char| c == ',' || c == '}')
                    .unwrap_or(value_str.len());
                let num_str = value_str[..end].trim();
                return num_str.parse::<f64>().unwrap_or(1.0);
            }
        }
        1.0
    }
}

impl Default for ConceptNetParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_conceptnet_line() {
        let mut parser = ConceptNetParser::new().with_min_weight(0.5);
        let line = "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]\t/r/IsA\t/c/en/dog\t/c/en/animal\t{\"weight\": 4.0}";
        let triple = parser.parse_line(line);
        assert!(triple.is_some());
        let t = triple.unwrap();
        assert_eq!(t.subject, "dog");
        assert_eq!(t.relation, RelationType::IsA);
        assert_eq!(t.object, "animal");
        assert!((t.weight - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_reject_non_english() {
        let mut parser = ConceptNetParser::new();
        let line = "/a/[id]\t/r/IsA\t/c/fr/chien\t/c/fr/animal\t{\"weight\": 1.0}";
        let triple = parser.parse_line(line);
        assert!(triple.is_none());
        assert_eq!(parser.triples_rejected, 1);
    }

    #[test]
    fn test_reject_low_weight() {
        let mut parser = ConceptNetParser::new().with_min_weight(2.0);
        let line = "/a/[id]\t/r/IsA\t/c/en/dog\t/c/en/animal\t{\"weight\": 1.0}";
        let triple = parser.parse_line(line);
        assert!(triple.is_none());
    }

    #[test]
    fn test_bootstrap_core() {
        let triples = ConceptNetParser::bootstrap_core();
        assert!(triples.len() > 50);
        // Should contain fundamental taxonomy
        let has_is_a = triples
            .iter()
            .any(|t| t.subject == "dog" && t.relation == RelationType::IsA);
        assert!(has_is_a);
    }

    #[test]
    fn test_from_tuples() {
        let triples = ConceptNetParser::from_tuples(&[
            ("sun", "IsA", "star"),
            ("earth", "IsA", "planet"),
        ]);
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].relation, RelationType::IsA);
    }

    #[test]
    fn test_bootstrap_into_core() {
        use super::super::core::KnowledgeCore;

        let mut core = KnowledgeCore::new();
        let triples = ConceptNetParser::bootstrap_core();
        core.ingest_batch(&triples);

        assert!(core.triple_count > 50);
        assert!(core.concept_count > 30);

        // Should be able to query
        let result = core.query_concept("dog");
        assert!(result.is_some());

        // Concepts with shared structure should show relatedness
        let dog_cat = core.relatedness("dog", "cat");
        assert!(
            dog_cat > 0.45,
            "dog~cat should show relatedness in bootstrap core: {dog_cat}"
        );
    }
}
