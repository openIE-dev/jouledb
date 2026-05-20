//! Faceted search and aggregations.
//!
//! Facet counting, hierarchical facets, range facets (price buckets),
//! multi-select facets, facet intersection, drill-down, and result filtering.

use std::collections::{BTreeMap, HashMap, HashSet};

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, thiserror::Error)]
pub enum FacetError {
    #[error("facet field not found: {0}")]
    FieldNotFound(String),
    #[error("document not found: {0}")]
    DocumentNotFound(String),
    #[error("invalid range: lower {lower} >= upper {upper}")]
    InvalidRange { lower: f64, upper: f64 },
}

// ── FacetValue ──────────────────────────────────────────────────

/// A single facet value with its count.
#[derive(Debug, Clone)]
pub struct FacetValue {
    pub value: String,
    pub count: usize,
    /// Child facet values (for hierarchical facets).
    pub children: Vec<FacetValue>,
}

// ── RangeBucket ─────────────────────────────────────────────────

/// A range bucket for numeric facets.
#[derive(Debug, Clone)]
pub struct RangeBucket {
    pub label: String,
    pub lower: f64,
    pub upper: f64,
    /// Whether lower bound is inclusive.
    pub lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    pub upper_inclusive: bool,
    pub count: usize,
}

// ── FacetSpec ───────────────────────────────────────────────────

/// Specification for a range facet.
#[derive(Debug, Clone)]
pub struct RangeSpec {
    pub label: String,
    pub lower: f64,
    pub upper: f64,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

// ── FacetResult ─────────────────────────────────────────────────

/// Aggregated facet result for a single field.
#[derive(Debug, Clone)]
pub struct FacetResult {
    pub field: String,
    pub values: Vec<FacetValue>,
    pub total_count: usize,
}

// ── Document ────────────────────────────────────────────────────

/// Internal document representation.
#[derive(Debug, Clone)]
struct FacetDocument {
    /// field_name -> list of values (supports multi-valued fields).
    fields: HashMap<String, Vec<String>>,
    /// field_name -> numeric value (for range facets).
    numeric_fields: HashMap<String, f64>,
}

// ── FilterClause ────────────────────────────────────────────────

/// A filter to apply during faceted search.
#[derive(Debug, Clone)]
pub enum FilterClause {
    /// Exact value match on a field.
    Exact { field: String, value: String },
    /// Match any of the values (multi-select).
    AnyOf { field: String, values: Vec<String> },
    /// Numeric range filter.
    Range {
        field: String,
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// Hierarchical prefix match (e.g., "Electronics > Phones").
    HierarchyPrefix { field: String, prefix: String },
}

// ── FacetEngine ─────────────────────────────────────────────────

/// Faceted search engine.
#[derive(Debug, Clone)]
pub struct FacetEngine {
    documents: HashMap<String, FacetDocument>,
    /// Hierarchical facet separator.
    hierarchy_separator: String,
}

impl FacetEngine {
    /// Create a new facet engine.
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            hierarchy_separator: " > ".to_string(),
        }
    }

    /// Set the hierarchy separator (default " > ").
    pub fn with_separator(mut self, sep: &str) -> Self {
        self.hierarchy_separator = sep.to_string();
        self
    }

    /// Number of documents.
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Add a document with string facet fields.
    pub fn add_document(
        &mut self,
        doc_id: &str,
        fields: &HashMap<String, Vec<String>>,
    ) {
        let doc = FacetDocument {
            fields: fields.clone(),
            numeric_fields: HashMap::new(),
        };
        self.documents.insert(doc_id.to_string(), doc);
    }

    /// Add a document with both string and numeric fields.
    pub fn add_document_full(
        &mut self,
        doc_id: &str,
        fields: &HashMap<String, Vec<String>>,
        numeric: &HashMap<String, f64>,
    ) {
        let doc = FacetDocument {
            fields: fields.clone(),
            numeric_fields: numeric.clone(),
        };
        self.documents.insert(doc_id.to_string(), doc);
    }

    /// Remove a document.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        self.documents.remove(doc_id).is_some()
    }

    /// Get all unique values for a field.
    pub fn field_values(&self, field: &str) -> Vec<String> {
        let mut values: HashSet<String> = HashSet::new();
        for doc in self.documents.values() {
            if let Some(vals) = doc.fields.get(field) {
                for v in vals {
                    values.insert(v.clone());
                }
            }
        }
        let mut sorted: Vec<String> = values.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Count facet values for a field across all documents.
    pub fn facet_count(&self, field: &str) -> FacetResult {
        self.facet_count_filtered(field, &[])
    }

    /// Count facet values with filters applied.
    /// Documents matching ALL filters are counted.
    /// For multi-select: the filter for `field` itself is excluded from
    /// filtering (so you see what other options are available).
    pub fn facet_count_filtered(&self, field: &str, filters: &[FilterClause]) -> FacetResult {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();

        for doc in self.documents.values() {
            // Check all filters EXCEPT the one on this field.
            if !self.matches_filters_excluding(doc, filters, field) {
                continue;
            }

            if let Some(vals) = doc.fields.get(field) {
                for v in vals {
                    *counts.entry(v.clone()).or_insert(0) += 1;
                }
            }
        }

        let total: usize = counts.values().sum();
        let values: Vec<FacetValue> = counts
            .into_iter()
            .map(|(value, count)| FacetValue {
                value,
                count,
                children: Vec::new(),
            })
            .collect();

        FacetResult {
            field: field.to_string(),
            values,
            total_count: total,
        }
    }

    /// Compute hierarchical facet counts.
    /// Values use the hierarchy separator (e.g., "Electronics > Phones > Android").
    pub fn hierarchical_facet_count(&self, field: &str) -> FacetResult {
        let mut flat_counts: BTreeMap<String, usize> = BTreeMap::new();

        for doc in self.documents.values() {
            if let Some(vals) = doc.fields.get(field) {
                for v in vals {
                    // Count each level of the hierarchy.
                    let parts: Vec<&str> = v.split(&self.hierarchy_separator).collect();
                    let mut prefix = String::new();
                    for (i, part) in parts.iter().enumerate() {
                        if i > 0 {
                            prefix.push_str(&self.hierarchy_separator);
                        }
                        prefix.push_str(part);
                        *flat_counts.entry(prefix.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Build tree structure.
        let values = self.build_hierarchy_tree(&flat_counts);
        let total: usize = flat_counts.values().sum();

        FacetResult {
            field: field.to_string(),
            values,
            total_count: total,
        }
    }

    /// Build hierarchical tree from flat counts.
    fn build_hierarchy_tree(&self, counts: &BTreeMap<String, usize>) -> Vec<FacetValue> {
        // Top-level: values without a separator.
        let mut roots: Vec<FacetValue> = Vec::new();

        for (path, &count) in counts {
            if !path.contains(&self.hierarchy_separator) {
                let children = self.collect_children(path, counts);
                roots.push(FacetValue {
                    value: path.clone(),
                    count,
                    children,
                });
            }
        }

        roots
    }

    /// Collect child facet values.
    fn collect_children(
        &self,
        parent_path: &str,
        counts: &BTreeMap<String, usize>,
    ) -> Vec<FacetValue> {
        let prefix = format!("{}{}", parent_path, self.hierarchy_separator);
        let mut children = Vec::new();

        for (path, &count) in counts {
            if let Some(rest) = path.strip_prefix(&prefix) {
                // Only direct children (no further separator).
                if !rest.contains(&self.hierarchy_separator) {
                    let sub_children = self.collect_children(path, counts);
                    children.push(FacetValue {
                        value: path.clone(),
                        count,
                        children: sub_children,
                    });
                }
            }
        }

        children
    }

    /// Range facet: bucket numeric values into ranges.
    pub fn range_facet(
        &self,
        field: &str,
        ranges: &[RangeSpec],
    ) -> Result<Vec<RangeBucket>, FacetError> {
        let mut buckets: Vec<RangeBucket> = ranges
            .iter()
            .map(|r| {
                if r.lower >= r.upper {
                    return Err(FacetError::InvalidRange {
                        lower: r.lower,
                        upper: r.upper,
                    });
                }
                Ok(RangeBucket {
                    label: r.label.clone(),
                    lower: r.lower,
                    upper: r.upper,
                    lower_inclusive: r.lower_inclusive,
                    upper_inclusive: r.upper_inclusive,
                    count: 0,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for doc in self.documents.values() {
            if let Some(&value) = doc.numeric_fields.get(field) {
                for bucket in &mut buckets {
                    let above_lower = if bucket.lower_inclusive {
                        value >= bucket.lower
                    } else {
                        value > bucket.lower
                    };
                    let below_upper = if bucket.upper_inclusive {
                        value <= bucket.upper
                    } else {
                        value < bucket.upper
                    };
                    if above_lower && below_upper {
                        bucket.count += 1;
                    }
                }
            }
        }

        Ok(buckets)
    }

    /// Filter documents by a set of filter clauses (AND logic).
    pub fn filter(&self, filters: &[FilterClause]) -> Vec<String> {
        let mut result: Vec<String> = self
            .documents
            .iter()
            .filter(|(_, doc)| self.matches_all_filters(doc, filters))
            .map(|(id, _)| id.clone())
            .collect();
        result.sort();
        result
    }

    /// Drill-down: apply a facet selection and return matching doc IDs.
    pub fn drill_down(&self, field: &str, value: &str) -> Vec<String> {
        let filter = FilterClause::Exact {
            field: field.to_string(),
            value: value.to_string(),
        };
        self.filter(&[filter])
    }

    /// Multi-select drill-down: select multiple values for a field.
    pub fn drill_down_multi(&self, field: &str, values: &[&str]) -> Vec<String> {
        let filter = FilterClause::AnyOf {
            field: field.to_string(),
            values: values.iter().map(|v| v.to_string()).collect(),
        };
        self.filter(&[filter])
    }

    /// Facet intersection: compute facet counts for field2 given a selection on field1.
    pub fn facet_intersection(
        &self,
        count_field: &str,
        filter_field: &str,
        filter_value: &str,
    ) -> FacetResult {
        let filters = vec![FilterClause::Exact {
            field: filter_field.to_string(),
            value: filter_value.to_string(),
        }];
        self.facet_count_filtered(count_field, &filters)
    }

    // ── Internal filter helpers ─────────────────────────────────

    fn matches_all_filters(&self, doc: &FacetDocument, filters: &[FilterClause]) -> bool {
        filters.iter().all(|f| self.matches_filter(doc, f))
    }

    fn matches_filters_excluding(
        &self,
        doc: &FacetDocument,
        filters: &[FilterClause],
        exclude_field: &str,
    ) -> bool {
        filters.iter().all(|f| {
            let filter_field = match f {
                FilterClause::Exact { field, .. } => field.as_str(),
                FilterClause::AnyOf { field, .. } => field.as_str(),
                FilterClause::Range { field, .. } => field.as_str(),
                FilterClause::HierarchyPrefix { field, .. } => field.as_str(),
            };
            if filter_field == exclude_field {
                return true; // skip this filter
            }
            self.matches_filter(doc, f)
        })
    }

    fn matches_filter(&self, doc: &FacetDocument, filter: &FilterClause) -> bool {
        match filter {
            FilterClause::Exact { field, value } => doc
                .fields
                .get(field)
                .map_or(false, |vals| vals.contains(value)),
            FilterClause::AnyOf { field, values } => doc
                .fields
                .get(field)
                .map_or(false, |doc_vals| {
                    values.iter().any(|v| doc_vals.contains(v))
                }),
            FilterClause::Range {
                field,
                lower,
                upper,
            } => {
                if let Some(&val) = doc.numeric_fields.get(field) {
                    let above = lower.map_or(true, |l| val >= l);
                    let below = upper.map_or(true, |u| val <= u);
                    above && below
                } else {
                    false
                }
            }
            FilterClause::HierarchyPrefix { field, prefix } => doc
                .fields
                .get(field)
                .map_or(false, |vals| {
                    vals.iter().any(|v| v.starts_with(prefix))
                }),
        }
    }
}

impl Default for FacetEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_engine() -> FacetEngine {
        let mut engine = FacetEngine::new();

        let mut f1 = HashMap::new();
        f1.insert("color".to_string(), vec!["red".to_string()]);
        f1.insert("size".to_string(), vec!["small".to_string()]);
        f1.insert("category".to_string(), vec!["Electronics > Phones > Android".to_string()]);
        let mut n1 = HashMap::new();
        n1.insert("price".to_string(), 29.99);
        engine.add_document_full("d1", &f1, &n1);

        let mut f2 = HashMap::new();
        f2.insert("color".to_string(), vec!["blue".to_string()]);
        f2.insert("size".to_string(), vec!["medium".to_string()]);
        f2.insert("category".to_string(), vec!["Electronics > Phones > iOS".to_string()]);
        let mut n2 = HashMap::new();
        n2.insert("price".to_string(), 99.99);
        engine.add_document_full("d2", &f2, &n2);

        let mut f3 = HashMap::new();
        f3.insert("color".to_string(), vec!["red".to_string()]);
        f3.insert("size".to_string(), vec!["large".to_string()]);
        f3.insert("category".to_string(), vec!["Electronics > Laptops".to_string()]);
        let mut n3 = HashMap::new();
        n3.insert("price".to_string(), 499.99);
        engine.add_document_full("d3", &f3, &n3);

        let mut f4 = HashMap::new();
        f4.insert("color".to_string(), vec!["green".to_string(), "blue".to_string()]);
        f4.insert("size".to_string(), vec!["small".to_string()]);
        f4.insert("category".to_string(), vec!["Clothing > Shirts".to_string()]);
        let mut n4 = HashMap::new();
        n4.insert("price".to_string(), 19.99);
        engine.add_document_full("d4", &f4, &n4);

        engine
    }

    #[test]
    fn test_num_documents() {
        let engine = build_engine();
        assert_eq!(engine.num_documents(), 4);
    }

    #[test]
    fn test_add_remove() {
        let mut engine = FacetEngine::new();
        let f = HashMap::new();
        engine.add_document("d1", &f);
        assert_eq!(engine.num_documents(), 1);
        assert!(engine.remove_document("d1"));
        assert_eq!(engine.num_documents(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut engine = FacetEngine::new();
        assert!(!engine.remove_document("nope"));
    }

    #[test]
    fn test_field_values() {
        let engine = build_engine();
        let colors = engine.field_values("color");
        assert!(colors.contains(&"red".to_string()));
        assert!(colors.contains(&"blue".to_string()));
        assert!(colors.contains(&"green".to_string()));
    }

    #[test]
    fn test_facet_count() {
        let engine = build_engine();
        let result = engine.facet_count("color");
        assert_eq!(result.field, "color");
        let red = result.values.iter().find(|v| v.value == "red").unwrap();
        assert_eq!(red.count, 2); // d1, d3
        let blue = result.values.iter().find(|v| v.value == "blue").unwrap();
        assert_eq!(blue.count, 2); // d2, d4
    }

    #[test]
    fn test_facet_count_size() {
        let engine = build_engine();
        let result = engine.facet_count("size");
        let small = result.values.iter().find(|v| v.value == "small").unwrap();
        assert_eq!(small.count, 2); // d1, d4
    }

    #[test]
    fn test_drill_down() {
        let engine = build_engine();
        let docs = engine.drill_down("color", "red");
        assert_eq!(docs, vec!["d1", "d3"]);
    }

    #[test]
    fn test_drill_down_multi() {
        let engine = build_engine();
        let docs = engine.drill_down_multi("color", &["red", "blue"]);
        // d1 (red), d2 (blue), d3 (red), d4 (green+blue)
        assert!(docs.contains(&"d1".to_string()));
        assert!(docs.contains(&"d2".to_string()));
        assert!(docs.contains(&"d3".to_string()));
        assert!(docs.contains(&"d4".to_string()));
    }

    #[test]
    fn test_range_facet() {
        let engine = build_engine();
        let ranges = vec![
            RangeSpec {
                label: "cheap".to_string(),
                lower: 0.0,
                upper: 50.0,
                lower_inclusive: true,
                upper_inclusive: true,
            },
            RangeSpec {
                label: "moderate".to_string(),
                lower: 50.0,
                upper: 200.0,
                lower_inclusive: false,
                upper_inclusive: true,
            },
            RangeSpec {
                label: "expensive".to_string(),
                lower: 200.0,
                upper: 1000.0,
                lower_inclusive: false,
                upper_inclusive: true,
            },
        ];
        let buckets = engine.range_facet("price", &ranges).unwrap();
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].count, 2); // 29.99, 19.99
        assert_eq!(buckets[1].count, 1); // 99.99
        assert_eq!(buckets[2].count, 1); // 499.99
    }

    #[test]
    fn test_range_facet_invalid() {
        let engine = build_engine();
        let ranges = vec![RangeSpec {
            label: "bad".to_string(),
            lower: 100.0,
            upper: 50.0,
            lower_inclusive: true,
            upper_inclusive: true,
        }];
        let err = engine.range_facet("price", &ranges).unwrap_err();
        assert!(matches!(err, FacetError::InvalidRange { .. }));
    }

    #[test]
    fn test_filter_exact() {
        let engine = build_engine();
        let docs = engine.filter(&[FilterClause::Exact {
            field: "color".to_string(),
            value: "red".to_string(),
        }]);
        assert_eq!(docs, vec!["d1", "d3"]);
    }

    #[test]
    fn test_filter_range() {
        let engine = build_engine();
        let docs = engine.filter(&[FilterClause::Range {
            field: "price".to_string(),
            lower: Some(50.0),
            upper: Some(500.0),
        }]);
        // 99.99 (d2), 499.99 (d3)
        assert!(docs.contains(&"d2".to_string()));
        assert!(docs.contains(&"d3".to_string()));
        assert!(!docs.contains(&"d1".to_string()));
    }

    #[test]
    fn test_filter_combined() {
        let engine = build_engine();
        let docs = engine.filter(&[
            FilterClause::Exact {
                field: "color".to_string(),
                value: "red".to_string(),
            },
            FilterClause::Range {
                field: "price".to_string(),
                lower: None,
                upper: Some(100.0),
            },
        ]);
        // Only d1: red AND price <= 100
        assert_eq!(docs, vec!["d1"]);
    }

    #[test]
    fn test_hierarchical_facet() {
        let engine = build_engine();
        let result = engine.hierarchical_facet_count("category");
        assert!(!result.values.is_empty());
        let electronics = result.values.iter().find(|v| v.value == "Electronics").unwrap();
        assert_eq!(electronics.count, 3); // d1, d2, d3
        assert!(!electronics.children.is_empty());
    }

    #[test]
    fn test_hierarchy_prefix_filter() {
        let engine = build_engine();
        let docs = engine.filter(&[FilterClause::HierarchyPrefix {
            field: "category".to_string(),
            prefix: "Electronics > Phones".to_string(),
        }]);
        assert!(docs.contains(&"d1".to_string())); // Phones > Android
        assert!(docs.contains(&"d2".to_string())); // Phones > iOS
        assert!(!docs.contains(&"d3".to_string())); // Laptops
    }

    #[test]
    fn test_facet_intersection() {
        let engine = build_engine();
        let result = engine.facet_intersection("size", "color", "red");
        // d1 (small), d3 (large)
        let small = result.values.iter().find(|v| v.value == "small");
        let large = result.values.iter().find(|v| v.value == "large");
        assert!(small.is_some());
        assert!(large.is_some());
        assert_eq!(small.unwrap().count, 1);
        assert_eq!(large.unwrap().count, 1);
    }

    #[test]
    fn test_multi_select_facet_count() {
        let engine = build_engine();
        // When filtering by color=red, the facet count for color should still show all
        // colors (multi-select behavior).
        let filters = vec![FilterClause::Exact {
            field: "color".to_string(),
            value: "red".to_string(),
        }];
        let result = engine.facet_count_filtered("color", &filters);
        // Since we exclude the "color" filter for color facet counting,
        // all colors should appear.
        assert!(result.values.len() >= 3);
    }

    #[test]
    fn test_multi_valued_field() {
        let engine = build_engine();
        // d4 has both "green" and "blue" for color
        let docs_green = engine.drill_down("color", "green");
        assert!(docs_green.contains(&"d4".to_string()));
        let docs_blue = engine.drill_down("color", "blue");
        assert!(docs_blue.contains(&"d4".to_string()));
    }

    #[test]
    fn test_default_trait() {
        let engine = FacetEngine::default();
        assert_eq!(engine.num_documents(), 0);
    }
}
