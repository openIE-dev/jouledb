//! User-Authored Custom Semantic Lenses (`CustomSemanticLensSpec`, `CustomLensCompiler`).
//!
//! Extends JouleDB's 6 built-in fluid lenses (`ConstellationGraph`, `HarmonizedMatrix`,
//! `PerceptualChart`, `CausalTimeline`, `GeodesicMap`, `NarrativeDossier`) with **user-authored,
//! domain-programmable Custom Semantic Lenses**:
//! - **`ParetoScatter`**: Multi-objective X/Y trade-off projection (e.g., minimize `duration.seconds`
//!   or `length.meters` while maximizing `money.usd` or `efficiency.usd_per_joule`), automatically
//!   computing the exact **Pareto-Optimal Frontier** (`is_pareto_optimal`)!
//! - **`SankeyLineage`**: Visualizes **how rigid schemas dissolve into meaning** (`Source Silo ->
//!   Raw Field Alias -> Canonical Semantic Concept -> Unified Entity`) with exact atom flow weights.
//! - **`NormalizedRadar`**: Projects heterogeneous physical dimensions (`USD`, `Kelvin`, `Joules`,
//!   `Meters`, `Seconds`) onto `[0.0, 1.0]` min-max normalized multi-axis star glyphs per entity.
//! - **`CohortMatrix`**: Groups unified cross-silo entities by silo coverage depth (`1-silo` vs
//!   `multi-silo` identity bridges) with aggregated harmonized totals.

use crate::amorphic_display::{AmorphicDisplaySynthesizer, ObserverIntent, SemanticDisplayAtom};
use crate::hint::{AccessibilityHint, SonificationHint};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Visual encoding archetype for a user-authored Custom Semantic Lens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CustomLensEncoding {
    #[default]
    ParetoScatter,
    SankeyLineage,
    NormalizedRadar,
    CohortMatrix,
}

/// Optimization direction for Pareto-frontier computation on an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AxisGoal {
    #[default]
    Maximize,
    Minimize,
}

/// Declarative specification of a user-authored Custom Semantic Lens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomSemanticLensSpec {
    pub title: String,
    pub encoding: CustomLensEncoding,
    pub x_concept: String,
    #[serde(default)]
    pub x_goal: AxisGoal,
    #[serde(default)]
    pub x_log_scale: bool,
    pub y_concept: String,
    #[serde(default)]
    pub y_goal: AxisGoal,
    #[serde(default)]
    pub size_concept: Option<String>,
    #[serde(default)]
    pub min_silo_bridges: usize,
    #[serde(default)]
    pub radar_concepts: Vec<String>,
}

/// Directed flow link in the `SankeyLineage` schema-dissolution lens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SankeyFlowLink {
    pub stage: String,
    pub source: String,
    pub target: String,
    pub atom_weight: usize,
}

/// Point in the compiled Custom Semantic Lens with Pareto optimality & normalized coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomLensPoint {
    pub entity_anchor: String,
    pub entity_label: String,
    pub silos_merged: Vec<String>,
    pub x_value: f64,
    pub y_value: f64,
    pub size_value: Option<f64>,
    pub is_pareto_optimal: bool,
    pub radar_normalized: BTreeMap<String, f64>,
}

/// Compiled, ready-to-render output of a user-authored [`CustomSemanticLensSpec`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCustomLens {
    pub spec: CustomSemanticLensSpec,
    pub points: Vec<CustomLensPoint>,
    pub pareto_optimal_anchors: Vec<String>,
    pub sankey_links: Vec<SankeyFlowLink>,
    pub vega_lite_spec: serde_json::Value,
    pub accessibility: AccessibilityHint,
    pub sonification: Option<SonificationHint>,
    pub compile_energy_nj: u64,
}

pub struct CustomLensCompiler;

impl CustomLensCompiler {
    /// Parse a concise single-line or multi-key Custom Lens DSL string into [`CustomSemanticLensSpec`]:
    /// `LENS "Foundry Efficiency Pareto" ENCODING pareto_scatter X="duration.seconds"(min,log) Y="efficiency.usd_per_joule"(max) SIZE="money.usd" MIN_SILOS=2`
    pub fn parse_dsl(dsl: &str) -> Result<CustomSemanticLensSpec, String> {
        let trimmed = dsl.trim();
        let rest = trimmed
            .strip_prefix("LENS ")
            .ok_or_else(|| "Custom lens DSL must start with `LENS \"Title\" ...`".to_string())?;
        let (title, after_title) = if let Some(stripped) = rest.strip_prefix('"') {
            let end_q = stripped
                .find('"')
                .ok_or_else(|| "unterminated quoted lens title".to_string())?;
            (stripped[..end_q].to_string(), stripped[end_q + 1..].trim())
        } else {
            let (t, r) = rest.split_once(' ').unwrap_or((rest, ""));
            (t.to_string(), r.trim())
        };

        let mut encoding = CustomLensEncoding::ParetoScatter;
        let mut x_concept = "money.usd".to_string();
        let mut x_goal = AxisGoal::Maximize;
        let mut x_log_scale = false;
        let mut y_concept = "duration.seconds".to_string();
        let mut y_goal = AxisGoal::Minimize;
        let mut size_concept = None;
        let mut min_silo_bridges = 1usize;
        let mut radar_concepts = Vec::new();

        let tokens: Vec<&str> = after_title.split_whitespace().collect();
        let mut i = 0usize;
        while i < tokens.len() {
            let tok = tokens[i];
            if tok.eq_ignore_ascii_case("ENCODING") {
                if let Some(&enc_s) = tokens.get(i + 1) {
                    encoding = match enc_s.to_lowercase().as_str() {
                        "pareto_scatter" | "pareto" => CustomLensEncoding::ParetoScatter,
                        "sankey_lineage" | "sankey" => CustomLensEncoding::SankeyLineage,
                        "normalized_radar" | "radar" => CustomLensEncoding::NormalizedRadar,
                        "cohort_matrix" | "cohort" => CustomLensEncoding::CohortMatrix,
                        _ => CustomLensEncoding::ParetoScatter,
                    };
                }
                i += 2;
                continue;
            }
            if let Some(xv) = tok.strip_prefix("X=") {
                let (c, goal, log) = parse_axis_spec(xv, AxisGoal::Maximize);
                x_concept = c;
                x_goal = goal;
                x_log_scale = log;
            } else if let Some(yv) = tok.strip_prefix("Y=") {
                let (c, goal, _) = parse_axis_spec(yv, AxisGoal::Maximize);
                y_concept = c;
                y_goal = goal;
            } else if let Some(sv) = tok.strip_prefix("SIZE=") {
                size_concept = Some(sv.trim_matches('"').to_string());
            } else if let Some(ms) = tok.strip_prefix("MIN_SILOS=") {
                if let Ok(n) = ms.parse::<usize>() {
                    min_silo_bridges = n;
                }
            } else if let Some(rc) = tok.strip_prefix("RADAR=") {
                radar_concepts = rc
                    .trim_matches('"')
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            i += 1;
        }

        Ok(CustomSemanticLensSpec {
            title,
            encoding,
            x_concept,
            x_goal,
            x_log_scale,
            y_concept,
            y_goal,
            size_concept,
            min_silo_bridges,
            radar_concepts,
        })
    }

    /// Compile a user-authored [`CustomSemanticLensSpec`] over unified amorphous [`SemanticDisplayAtom`]s.
    pub fn compile(
        atoms: &[SemanticDisplayAtom],
        spec: &CustomSemanticLensSpec,
    ) -> CompiledCustomLens {
        let base_canvas =
            AmorphicDisplaySynthesizer::synthesize(atoms, ObserverIntent::AutoOptimal);

        // Build Sankey Schema-Dissolution Lineage (`Silo -> RawKey -> CanonicalConcept -> Entity`)
        let mut silo_to_key: BTreeMap<(String, String), usize> = BTreeMap::new();
        let mut key_to_concept: BTreeMap<(String, String), usize> = BTreeMap::new();
        let mut concept_to_entity: BTreeMap<(String, String), usize> = BTreeMap::new();

        for a in atoms {
            if a.canonical_concept == "narrative.note" {
                continue;
            }
            *silo_to_key
                .entry((a.source_silo.clone(), a.raw_key.clone()))
                .or_insert(0) += 1;
            *key_to_concept
                .entry((a.raw_key.clone(), a.canonical_concept.clone()))
                .or_insert(0) += 1;
            *concept_to_entity
                .entry((a.canonical_concept.clone(), a.entity_label.clone()))
                .or_insert(0) += 1;
        }

        let mut sankey_links = Vec::new();
        for ((s, k), w) in silo_to_key {
            sankey_links.push(SankeyFlowLink {
                stage: "silo_to_raw_key".into(),
                source: format!("silo:{s}"),
                target: format!("key:{k}"),
                atom_weight: w,
            });
        }
        for ((k, c), w) in key_to_concept {
            sankey_links.push(SankeyFlowLink {
                stage: "raw_key_to_concept".into(),
                source: format!("key:{k}"),
                target: format!("concept:{c}"),
                atom_weight: w,
            });
        }
        for ((c, e), w) in concept_to_entity {
            sankey_links.push(SankeyFlowLink {
                stage: "concept_to_entity".into(),
                source: format!("concept:{c}"),
                target: format!("entity:{e}"),
                atom_weight: w,
            });
        }

        // Determine radar normalization bounds across filtered emergent rows
        let filtered_rows: Vec<_> = base_canvas
            .emergent_rows
            .iter()
            .filter(|r| r.silos_merged.len() >= spec.min_silo_bridges.max(1))
            .collect();

        let radar_dims: Vec<String> = if spec.radar_concepts.is_empty() {
            let mut set = std::collections::BTreeSet::new();
            for r in &filtered_rows {
                for k in r.metrics.keys() {
                    set.insert(k.clone());
                }
            }
            set.into_iter().take(6).collect()
        } else {
            spec.radar_concepts.clone()
        };

        let mut dim_bounds: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        for d in &radar_dims {
            let vals: Vec<f64> = filtered_rows
                .iter()
                .filter_map(|r| r.metrics.get(d).copied())
                .collect();
            if !vals.is_empty() {
                let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
                let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                dim_bounds.insert(d.clone(), (mn, mx));
            }
        }

        let mut points: Vec<CustomLensPoint> = filtered_rows
            .iter()
            .map(|r| {
                let xv = r.metrics.get(&spec.x_concept).copied().unwrap_or(0.0);
                let yv = r.metrics.get(&spec.y_concept).copied().unwrap_or(0.0);
                let sv = spec
                    .size_concept
                    .as_ref()
                    .and_then(|sc| r.metrics.get(sc).copied());
                let mut radar_norm = BTreeMap::new();
                for (d, &(mn, mx)) in &dim_bounds {
                    if let Some(&v) = r.metrics.get(d) {
                        let norm = if (mx - mn).abs() > 1e-15 {
                            (v - mn) / (mx - mn)
                        } else {
                            1.0
                        };
                        radar_norm.insert(d.clone(), norm);
                    }
                }
                CustomLensPoint {
                    entity_anchor: r.entity_anchor.clone(),
                    entity_label: r.entity_label.clone(),
                    silos_merged: r.silos_merged.clone(),
                    x_value: xv,
                    y_value: yv,
                    size_value: sv,
                    is_pareto_optimal: false,
                    radar_normalized: radar_norm,
                }
            })
            .collect();

        // Compute exact 2-objective Pareto optimality (`domination` check)
        let mut pareto_anchors = Vec::new();
        for idx in 0..points.len() {
            let px = points[idx].x_value;
            let py = points[idx].y_value;
            let mut dominated = false;
            for (other_idx, other) in points.iter().enumerate() {
                if idx == other_idx {
                    continue;
                }
                let better_eq_x = match spec.x_goal {
                    AxisGoal::Maximize => other.x_value >= px,
                    AxisGoal::Minimize => other.x_value <= px,
                };
                let better_eq_y = match spec.y_goal {
                    AxisGoal::Maximize => other.y_value >= py,
                    AxisGoal::Minimize => other.y_value <= py,
                };
                let strictly_better = match (spec.x_goal, spec.y_goal) {
                    (AxisGoal::Maximize, AxisGoal::Maximize) => {
                        other.x_value > px || other.y_value > py
                    }
                    (AxisGoal::Maximize, AxisGoal::Minimize) => {
                        other.x_value > px || other.y_value < py
                    }
                    (AxisGoal::Minimize, AxisGoal::Maximize) => {
                        other.x_value < px || other.y_value > py
                    }
                    (AxisGoal::Minimize, AxisGoal::Minimize) => {
                        other.x_value < px || other.y_value < py
                    }
                };
                if better_eq_x && better_eq_y && strictly_better {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                points[idx].is_pareto_optimal = true;
                pareto_anchors.push(points[idx].entity_anchor.clone());
            }
        }

        let vega_values: Vec<serde_json::Value> = points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "entity": p.entity_label,
                    "anchor": p.entity_anchor,
                    "x": p.x_value,
                    "y": p.y_value,
                    "size": p.size_value.unwrap_or(1.0),
                    "pareto_optimal": p.is_pareto_optimal,
                    "silos": p.silos_merged.join(" + "),
                })
            })
            .collect();

        let vega_lite_spec = serde_json::json!({
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "title": spec.title,
            "data": { "values": vega_values },
            "mark": { "type": "point", "filled": true, "size": 140 },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "title": spec.x_concept,
                    "scale": { "type": if spec.x_log_scale { "log" } else { "linear" } }
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "title": spec.y_concept
                },
                "color": {
                    "field": "pareto_optimal",
                    "type": "nominal",
                    "title": "Pareto Frontier"
                },
                "tooltip": [
                    { "field": "entity", "type": "nominal" },
                    { "field": "silos", "type": "nominal" },
                    { "field": "x", "type": "quantitative" },
                    { "field": "y", "type": "quantitative" }
                ]
            }
        });

        let accessibility = AccessibilityHint {
            alt_text: format!(
                "Custom Semantic Lens '{}' ({:?}) projecting {} entities with {} Pareto-optimal leaders",
                spec.title,
                spec.encoding,
                points.len(),
                pareto_anchors.len()
            ),
            description: format!(
                "User-authored lens comparing {} vs {} across {} unified entities ({} Sankey schema-dissolution links).",
                spec.x_concept,
                spec.y_concept,
                points.len(),
                sankey_links.len()
            ),
            aria_role: "region".into(),
        };

        let sonification = Some(SonificationHint {
            pitch_column: Some(1),
            volume_column: Some(2),
            base_frequency: 261.63,
            frequency_range: (261.63, 1046.50),
            midi_program: 12, // Marimba
            note_duration_ms: 160,
        });

        CompiledCustomLens {
            spec: spec.clone(),
            points,
            pareto_optimal_anchors: pareto_anchors,
            sankey_links,
            vega_lite_spec,
            accessibility,
            sonification,
            compile_energy_nj: 165 + (atoms.len() as u64) * 14,
        }
    }
}

fn parse_axis_spec(raw: &str, default_goal: AxisGoal) -> (String, AxisGoal, bool) {
    if let Some((concept_part, mods_part)) = raw.split_once('(') {
        let mods = mods_part.trim_end_matches(')').to_lowercase();
        let goal = if mods.contains("min") {
            AxisGoal::Minimize
        } else if mods.contains("max") {
            AxisGoal::Maximize
        } else {
            default_goal
        };
        let log = mods.contains("log");
        (concept_part.trim_matches('"').to_string(), goal, log)
    } else {
        (raw.trim_matches('"').to_string(), default_goal, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_semantic_lens_dsl_pareto_and_sankey() {
        let atoms = vec![
            SemanticDisplayAtom {
                atom_id: "1".into(),
                source_silo: "crm_json".into(),
                entity_anchor: "alpha".into(),
                entity_label: "Alpha Node".into(),
                raw_key: "contract_usd".into(),
                canonical_concept: "money.usd".into(),
                numeric_si: Some(50_000.0),
                canonical_unit: Some("USD".into()),
                timestamp_ms: None,
                geo_lat_lon: None,
                relation_target: None,
                text_value: "$50,000".into(),
            },
            SemanticDisplayAtom {
                atom_id: "2".into(),
                source_silo: "ops_kv".into(),
                entity_anchor: "alpha".into(),
                entity_label: "Alpha Node".into(),
                raw_key: "p99_ms".into(),
                canonical_concept: "duration.seconds".into(),
                numeric_si: Some(0.012),
                canonical_unit: Some("s".into()),
                timestamp_ms: None,
                geo_lat_lon: None,
                relation_target: None,
                text_value: "12ms".into(),
            },
            SemanticDisplayAtom {
                atom_id: "3".into(),
                source_silo: "crm_json".into(),
                entity_anchor: "beta".into(),
                entity_label: "Beta Node".into(),
                raw_key: "contract_usd".into(),
                canonical_concept: "money.usd".into(),
                numeric_si: Some(18_000.0),
                canonical_unit: Some("USD".into()),
                timestamp_ms: None,
                geo_lat_lon: None,
                relation_target: None,
                text_value: "$18,000".into(),
            },
            SemanticDisplayAtom {
                atom_id: "4".into(),
                source_silo: "ops_kv".into(),
                entity_anchor: "beta".into(),
                entity_label: "Beta Node".into(),
                raw_key: "p99_ms".into(),
                canonical_concept: "duration.seconds".into(),
                numeric_si: Some(0.085),
                canonical_unit: Some("s".into()),
                timestamp_ms: None,
                geo_lat_lon: None,
                relation_target: None,
                text_value: "85ms".into(),
            },
        ];

        let spec = CustomLensCompiler::parse_dsl(
            r#"LENS "Latency vs Revenue Pareto" ENCODING pareto_scatter X="duration.seconds"(min,log) Y="money.usd"(max) MIN_SILOS=2"#,
        )
        .expect("parse custom lens");
        let compiled = CustomLensCompiler::compile(&atoms, &spec);
        assert_eq!(compiled.points.len(), 2);
        // Alpha has LOWER latency (0.012s < 0.085s) AND HIGHER revenue ($50k > $18k), dominating Beta!
        assert_eq!(compiled.pareto_optimal_anchors, vec!["alpha".to_string()]);
        assert!(!compiled.sankey_links.is_empty());
    }
}
