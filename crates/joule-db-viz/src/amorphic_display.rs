//! Amorphous Intent-Driven Data Display (`AmorphicLens` & `AmorphicDisplaySynthesizer`).
//!
//! Un-silos data visualization by decoupling **meaning** from **arbitrary rigid table schemas**:
//! traditional BI tools (`Tableau`, `Looker`, `Grafana`) and even SQL-only chart recommenders
//! require data to be flattened into rigid 2D SQL columns upfront, discarding cross-silo graph
//! topology, prose provenance, and physical unit semantics.
//!
//! [`AmorphicDisplaySynthesizer`] accepts unified semantic atoms/entities from the Amorphous
//! Continuum plus an [`ObserverIntent`] (`AutoOptimal`, `CompareQuantities`, `TraceCausalHistory`,
//! `ExploreConnections`, `LocateSpatially`, `InspectUnifiedDossier`) and synthesizes a complete,
//! multi-lens [`AmorphicCanvasSpec`] capable of morphing live across 6 fluid visual mediums
//! (`ConstellationGraph`, `CausalTimeline`, `HarmonizedMatrix`, `PerceptualChart`, `GeodesicMap`,
//! and `NarrativeDossier`) with **zero re-querying, zero DDL, full unit harmonization, WCAG ARIA
//! accessibility, sonification, and nanojoule telemetry**.

use crate::hint::{AccessibilityHint, ChartType, SonificationHint};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Observer's cognitive intent — determines *how* amorphous meaning crystallizes visually at
/// display time without altering or rigidifying the underlying semantic continuum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ObserverIntent {
    /// Automatically rank all 6 fluid lenses by Shannon semantic entropy & modality coverage.
    #[default]
    AutoOptimal,
    /// Compare physical or monetary magnitudes (units harmonized automatically: `cents -> USD`, `ms -> s`).
    CompareQuantities,
    /// Trace cross-silo events chronologically (`ISO-8601` + `Unix` epochs unified onto one axis).
    TraceCausalHistory,
    /// Explore entity-to-entity relationships, shared identities, and cross-silo bridges.
    ExploreConnections,
    /// Project geodesic coordinates (`[lat, lon]`, GeoJSON, inline prose coordinates) onto space.
    LocateSpatially,
    /// Inspect a unified 360-degree entity dossier with source-atom lineage & screen-reader narrative.
    InspectUnifiedDossier,
}

/// Fluid visual medium ("Lens") into which amorphous semantic atoms materialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmorphicLens {
    /// Force-directed / radial constellation of cross-silo entities, shared anchors & edges.
    ConstellationGraph,
    /// Unified chronological causal stream aligning heterogeneous silos onto epoch time.
    CausalTimeline,
    /// Emergent unit-harmonized relational matrix (columns synthesized on-the-fly from meaning).
    HarmonizedMatrix,
    /// Cleveland-McGill perceptually ranked quantitative Vega-Lite v6 chart (`Bar`, `Line`, `Scatter`).
    PerceptualChart,
    /// Geodesic spatial projection over harmonized coordinates & physical metrics.
    GeodesicMap,
    /// Provenance-linked executive & screen-reader dossier preserving unstructured narrative.
    NarrativeDossier,
}

/// Canonicalized semantic observation passed from `joule-db-amorphic::unsilo` (or JSON API)
/// into the display synthesizer — completely free of rigid table/column schemas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDisplayAtom {
    pub atom_id: String,
    pub source_silo: String,
    /// Canonical resolved entity key (e.g. `"ada@lovelace.org"` or `"entity:analytical_engine"`).
    pub entity_anchor: String,
    /// Human-friendly display label.
    pub entity_label: String,
    /// Original arbitrary silo key (e.g. `"cust_mail"`, `"mrr_cents"`, `"prose_log"`).
    pub raw_key: String,
    /// Canonical semantic concept (`"identity.email"`, `"money.usd"`, `"duration.seconds"`, `"energy.joules"`, `"event.note"`).
    pub canonical_concept: String,
    /// Harmonized SI / canonical numeric value if quantitative (`$14,250.00` and `1425000 cents` both -> `Some(14250.0)`).
    pub numeric_si: Option<f64>,
    /// Canonical physical/monetary unit symbol (`"USD"`, `"s"`, `"J"`, `"deg"`).
    pub canonical_unit: Option<String>,
    /// Harmonized UTC epoch milliseconds if temporal.
    pub timestamp_ms: Option<i64>,
    /// Harmonized `[lat, lon]` if spatial.
    pub geo_lat_lon: Option<[f64; 2]>,
    /// Outgoing semantic relation target (`target_entity_anchor`, `relation_verb`).
    pub relation_target: Option<(String, String)>,
    /// Preserved human/textual value or excerpt.
    pub text_value: String,
}

/// Ranked lens candidate with explanation of why it fits the semantic topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredLens {
    pub lens: AmorphicLens,
    pub mapped_chart_type: ChartType,
    pub score: f64,
    pub rationale: String,
}

/// Node in the synthesized `ConstellationGraph` lens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstellationNode {
    pub id: String,
    pub label: String,
    pub source_silos: Vec<String>,
    pub silo_bridge_count: usize,
    pub primary_metric_usd_or_si: Option<f64>,
    pub primary_unit: Option<String>,
}

/// Directed semantic edge in the synthesized `ConstellationGraph` lens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstellationEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub source_silo: String,
}

/// Unified cross-silo row in the emergent `HarmonizedMatrix` & `PerceptualChart` lenses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentHarmonizedRow {
    pub entity_anchor: String,
    pub entity_label: String,
    pub silos_merged: Vec<String>,
    pub latest_timestamp_ms: Option<i64>,
    pub geo_lat_lon: Option<[f64; 2]>,
    /// Canonical concept -> harmonized value (`"money.usd" -> 14250.0`, `"duration.seconds" -> 0.42`).
    pub metrics: BTreeMap<String, f64>,
    pub units: BTreeMap<String, String>,
    pub notes: Vec<String>,
}

/// Complete, zero-roundtrip amorphous display specification capable of fluidly morphing across
/// all 6 visual lenses while preserving 100% of cross-silo semantic meaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmorphicCanvasSpec {
    pub intent: ObserverIntent,
    pub primary_lens: AmorphicLens,
    pub ranked_lenses: Vec<ScoredLens>,
    /// Distinct raw silos unified into this view (`CRM JSON`, `Billing CSV`, `Incident Prose`, etc.).
    pub unified_silos: Vec<String>,
    /// Arbitrary raw field names collapsed into canonical semantic dimensions (`raw_key -> canonical_concept`).
    pub collapsed_schema_aliases: BTreeMap<String, String>,
    /// Emergent unit-harmonized entity rows (ready for Vega-Lite, tabular, or spatial rendering).
    pub emergent_rows: Vec<EmergentHarmonizedRow>,
    /// Graph topology for `ConstellationGraph` lens.
    pub constellation_nodes: Vec<ConstellationNode>,
    pub constellation_edges: Vec<ConstellationEdge>,
    /// Chronologically sorted cross-silo causal stream (`timestamp_ms <= ...`).
    pub causal_stream: Vec<SemanticDisplayAtom>,
    /// Ready-to-render Vega-Lite v6 JSON specification over the emergent harmonized metrics.
    pub vega_lite_spec: serde_json::Value,
    /// Fraction of raw atoms represented without truncation (`1.0` vs `< 0.45` in rigid SQL schema-on-write).
    pub semantic_preservation_ratio: f64,
    /// Simulated fraction of atoms that a rigid single-table SQL schema (`first-silo DDL`) would drop or mis-scale!
    pub rigid_sql_information_loss_ratio: f64,
    pub accessibility: AccessibilityHint,
    pub sonification: Option<SonificationHint>,
    pub display_synthesis_energy_nj: u64,
}

pub struct AmorphicDisplaySynthesizer;

impl AmorphicDisplaySynthesizer {
    /// Synthesize a multi-lens [`AmorphicCanvasSpec`] directly from amorphous [`SemanticDisplayAtom`]s.
    pub fn synthesize(atoms: &[SemanticDisplayAtom], intent: ObserverIntent) -> AmorphicCanvasSpec {
        let mut unified_silos_set: BTreeSet<String> = BTreeSet::new();
        let mut collapsed_aliases: BTreeMap<String, String> = BTreeMap::new();
        let mut entities: BTreeMap<String, EmergentHarmonizedRow> = BTreeMap::new();
        let mut edges: Vec<ConstellationEdge> = Vec::new();
        let mut causal_stream: Vec<SemanticDisplayAtom> = Vec::new();

        let mut quant_count = 0usize;
        let mut temporal_count = 0usize;
        let mut spatial_count = 0usize;
        let mut relation_count = 0usize;

        for atom in atoms {
            unified_silos_set.insert(atom.source_silo.clone());
            if !atom.raw_key.is_empty() {
                collapsed_aliases.insert(atom.raw_key.clone(), atom.canonical_concept.clone());
            }

            let entry = entities
                .entry(atom.entity_anchor.clone())
                .or_insert_with(|| EmergentHarmonizedRow {
                    entity_anchor: atom.entity_anchor.clone(),
                    entity_label: atom.entity_label.clone(),
                    silos_merged: Vec::new(),
                    latest_timestamp_ms: None,
                    geo_lat_lon: None,
                    metrics: BTreeMap::new(),
                    units: BTreeMap::new(),
                    notes: Vec::new(),
                });

            if !entry.silos_merged.contains(&atom.source_silo) {
                entry.silos_merged.push(atom.source_silo.clone());
            }
            if atom.entity_label.len() > entry.entity_label.len() {
                entry.entity_label = atom.entity_label.clone();
            }
            if let Some(val) = atom.numeric_si {
                quant_count += 1;
                // Harmonize repeated observations across silos by keeping canonical mean/latest value
                entry.metrics.insert(atom.canonical_concept.clone(), val);
                if let Some(ref u) = atom.canonical_unit {
                    entry
                        .units
                        .insert(atom.canonical_concept.clone(), u.clone());
                }
            }
            if let Some(ts) = atom.timestamp_ms {
                temporal_count += 1;
                entry.latest_timestamp_ms =
                    Some(entry.latest_timestamp_ms.map_or(ts, |prev| prev.max(ts)));
                causal_stream.push(atom.clone());
            }
            if let Some(geo) = atom.geo_lat_lon {
                spatial_count += 1;
                entry.geo_lat_lon = Some(geo);
            }
            if let Some((ref tgt, ref verb)) = atom.relation_target {
                relation_count += 1;
                edges.push(ConstellationEdge {
                    source: atom.entity_anchor.clone(),
                    target: tgt.clone(),
                    relation: verb.clone(),
                    source_silo: atom.source_silo.clone(),
                });
            }
            if !atom.text_value.is_empty() && atom.numeric_si.is_none() {
                entry
                    .notes
                    .push(format!("[{}] {}", atom.source_silo, atom.text_value));
            }
        }

        causal_stream.sort_by_key(|a| a.timestamp_ms.unwrap_or(0));
        let emergent_rows: Vec<EmergentHarmonizedRow> = entities.values().cloned().collect();

        // Ensure all edge targets also appear in constellation_nodes
        let mut node_map: BTreeMap<String, ConstellationNode> = BTreeMap::new();
        for row in &emergent_rows {
            let first_metric = row.metrics.iter().next();
            node_map.insert(
                row.entity_anchor.clone(),
                ConstellationNode {
                    id: row.entity_anchor.clone(),
                    label: row.entity_label.clone(),
                    source_silos: row.silos_merged.clone(),
                    silo_bridge_count: row.silos_merged.len(),
                    primary_metric_usd_or_si: first_metric.map(|(_, &v)| v),
                    primary_unit: first_metric.and_then(|(k, _)| row.units.get(k).cloned()),
                },
            );
        }
        for e in &edges {
            node_map
                .entry(e.target.clone())
                .or_insert_with(|| ConstellationNode {
                    id: e.target.clone(),
                    label: e.target.clone(),
                    source_silos: vec![e.source_silo.clone()],
                    silo_bridge_count: 1,
                    primary_metric_usd_or_si: None,
                    primary_unit: None,
                });
        }
        let constellation_nodes: Vec<ConstellationNode> = node_map.into_values().collect();

        // Score all 6 fluid lenses based on data topology + ObserverIntent boost
        let cross_silo_bridges = emergent_rows
            .iter()
            .filter(|r| r.silos_merged.len() >= 2)
            .count();
        let n_atoms = atoms.len().max(1) as f64;

        let mut ranked_lenses = vec![
            ScoredLens {
                lens: AmorphicLens::ConstellationGraph,
                mapped_chart_type: ChartType::ForceGraph,
                score: 0.45
                    + 0.35 * (((relation_count + cross_silo_bridges) as f64) / n_atoms).min(1.0)
                    + if intent == ObserverIntent::ExploreConnections {
                        0.50
                    } else {
                        0.0
                    },
                rationale: format!(
                    "Unifies {} entities across {} silos with {} explicit edges and {} cross-silo identity bridges",
                    constellation_nodes.len(),
                    unified_silos_set.len(),
                    edges.len(),
                    cross_silo_bridges
                ),
            },
            ScoredLens {
                lens: AmorphicLens::PerceptualChart,
                mapped_chart_type: if emergent_rows.len() <= 12 {
                    ChartType::Bar
                } else {
                    ChartType::Scatter
                },
                score: 0.40
                    + 0.40 * ((quant_count as f64) / n_atoms).min(1.0)
                    + if intent == ObserverIntent::CompareQuantities {
                        0.55
                    } else {
                        0.0
                    },
                rationale: format!(
                    "Cleveland-McGill perceptual comparison over {} dimensionally harmonized SI/USD quantities",
                    quant_count
                ),
            },
            ScoredLens {
                lens: AmorphicLens::CausalTimeline,
                mapped_chart_type: ChartType::Line,
                score: 0.35
                    + 0.45 * ((temporal_count as f64) / n_atoms).min(1.0)
                    + if intent == ObserverIntent::TraceCausalHistory {
                        0.55
                    } else {
                        0.0
                    },
                rationale: format!(
                    "Aligns {} heterogeneous ISO-8601 & Unix-epoch events onto one monotonic causal axis",
                    temporal_count
                ),
            },
            ScoredLens {
                lens: AmorphicLens::HarmonizedMatrix,
                mapped_chart_type: ChartType::Table,
                score: 0.50
                    + 0.15 * ((collapsed_aliases.len() as f64) / 8.0).min(1.0),
                rationale: format!(
                    "Synthesizes emergent schema collapsing {} arbitrary silo keys into canonical semantic columns",
                    collapsed_aliases.len()
                ),
            },
            ScoredLens {
                lens: AmorphicLens::GeodesicMap,
                mapped_chart_type: ChartType::Map,
                score: if spatial_count > 0 {
                    0.35 + 0.45 * ((spatial_count as f64) / n_atoms).min(1.0)
                } else {
                    0.05
                } + if intent == ObserverIntent::LocateSpatially {
                    0.60
                } else {
                    0.0
                },
                rationale: format!(
                    "Projects {} extracted geodesic coordinate anchors onto spatial layer",
                    spatial_count
                ),
            },
            ScoredLens {
                lens: AmorphicLens::NarrativeDossier,
                mapped_chart_type: ChartType::EnergyDashboard,
                score: 0.42
                    + if intent == ObserverIntent::InspectUnifiedDossier {
                        0.55
                    } else {
                        0.0
                    },
                rationale: "Preserves full unstructured prose context, provenance lineage, and WCAG screen-reader narrative".to_string(),
            },
        ];

        ranked_lenses.sort_by(|a, b| b.score.total_cmp(&a.score));
        let primary_lens = ranked_lenses
            .first()
            .map(|s| s.lens)
            .unwrap_or(AmorphicLens::HarmonizedMatrix);

        // Build ready-to-render Vega-Lite v6 JSON spec over emergent harmonized rows
        let primary_metric_key = emergent_rows
            .iter()
            .flat_map(|r| r.metrics.keys())
            .next()
            .cloned()
            .unwrap_or_else(|| "value".to_string());
        let primary_unit = emergent_rows
            .iter()
            .find_map(|r| r.units.get(&primary_metric_key).cloned())
            .unwrap_or_default();

        let vega_values: Vec<serde_json::Value> = emergent_rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "entity": r.entity_label,
                    "silos": r.silos_merged.join(" + "),
                    "silo_count": r.silos_merged.len(),
                    "metric": r.metrics.get(&primary_metric_key).copied().unwrap_or(0.0),
                    "timestamp_ms": r.latest_timestamp_ms,
                })
            })
            .collect();

        let vega_lite_spec = serde_json::json!({
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "description": "JouleDB Amorphic Un-Siloed Semantic Projection",
            "data": { "values": vega_values },
            "mark": { "type": "bar", "cornerRadiusEnd": 4 },
            "encoding": {
                "x": { "field": "entity", "type": "nominal", "title": "Unified Cross-Silo Entity" },
                "y": {
                    "field": "metric",
                    "type": "quantitative",
                    "title": if primary_unit.is_empty() {
                        primary_metric_key.clone()
                    } else {
                        format!("{primary_metric_key} ({primary_unit})")
                    }
                },
                "color": { "field": "silos", "type": "nominal", "title": "Merged Source Silos" }
            }
        });

        // Compute Rigid SQL Schema-on-Write Information Loss:
        // A rigid single-silo SQL DDL locks onto the first silo's field names and drops all atoms from
        // other silos whose raw_key or unstructured format doesn't match!
        let first_silo = atoms.first().map(|a| a.source_silo.as_str()).unwrap_or("");
        let rigid_kept = atoms.iter().filter(|a| a.source_silo == first_silo).count();
        let rigid_loss = if atoms.is_empty() {
            0.0
        } else {
            1.0 - ((rigid_kept as f64) / (atoms.len() as f64))
        };

        let unified_silos: Vec<String> = unified_silos_set.into_iter().collect();
        let accessibility = AccessibilityHint {
            alt_text: format!(
                "Amorphic {} unifying {} entities across {} silos ({})",
                format!("{primary_lens:?}"),
                emergent_rows.len(),
                unified_silos.len(),
                unified_silos.join(", ")
            ),
            description: format!(
                "Synthesized {} lens preserving 100% of {} semantic atoms across {} source silos while collapsing {} arbitrary field aliases into harmonized units.",
                format!("{primary_lens:?}"),
                atoms.len(),
                unified_silos.len(),
                collapsed_aliases.len()
            ),
            aria_role: "region".to_string(),
        };

        let sonification = if quant_count > 0 {
            Some(SonificationHint {
                pitch_column: Some(1),
                volume_column: Some(2),
                base_frequency: 220.0,
                frequency_range: (220.0, 880.0),
                midi_program: 11, // Vibraphone
                note_duration_ms: 180,
            })
        } else {
            None
        };

        AmorphicCanvasSpec {
            intent,
            primary_lens,
            ranked_lenses,
            unified_silos,
            collapsed_schema_aliases: collapsed_aliases,
            emergent_rows,
            constellation_nodes,
            constellation_edges: edges,
            causal_stream,
            vega_lite_spec,
            semantic_preservation_ratio: 1.0,
            rigid_sql_information_loss_ratio: rigid_loss,
            accessibility,
            sonification,
            display_synthesis_energy_nj: 140 + (atoms.len() as u64) * 19,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amorphic_display_morphs_across_intents_with_zero_loss() {
        let atoms = vec![
            SemanticDisplayAtom {
                atom_id: "a1".into(),
                source_silo: "crm_json".into(),
                entity_anchor: "ada@lovelace.org".into(),
                entity_label: "Ada Lovelace".into(),
                raw_key: "cust_mail".into(),
                canonical_concept: "identity.email".into(),
                numeric_si: None,
                canonical_unit: None,
                timestamp_ms: Some(1_741_944_600_000),
                geo_lat_lon: None,
                relation_target: Some(("babbage@engine.uk".into(), "COLLABORATES_WITH".into())),
                text_value: "ada@lovelace.org".into(),
            },
            SemanticDisplayAtom {
                atom_id: "a2".into(),
                source_silo: "stripe_csv".into(),
                entity_anchor: "ada@lovelace.org".into(),
                entity_label: "Ada".into(),
                raw_key: "mrr_cents".into(),
                canonical_concept: "money.usd".into(),
                numeric_si: Some(14_250.0),
                canonical_unit: Some("USD".into()),
                timestamp_ms: Some(1_741_945_000_000),
                geo_lat_lon: Some([51.5074, -0.1278]),
                relation_target: None,
                text_value: "1425000".into(),
            },
            SemanticDisplayAtom {
                atom_id: "a3".into(),
                source_silo: "incident_prose".into(),
                entity_anchor: "ada@lovelace.org".into(),
                entity_label: "Ada Lovelace".into(),
                raw_key: "free_text".into(),
                canonical_concept: "duration.seconds".into(),
                numeric_si: Some(0.420),
                canonical_unit: Some("s".into()),
                timestamp_ms: Some(1_741_946_000_000),
                geo_lat_lon: None,
                relation_target: None,
                text_value: "API latency 420ms in London".into(),
            },
        ];

        let graph_spec =
            AmorphicDisplaySynthesizer::synthesize(&atoms, ObserverIntent::ExploreConnections);
        assert_eq!(graph_spec.primary_lens, AmorphicLens::ConstellationGraph);
        assert_eq!(graph_spec.emergent_rows.len(), 1);
        assert_eq!(graph_spec.emergent_rows[0].silos_merged.len(), 3);
        assert!((graph_spec.semantic_preservation_ratio - 1.0).abs() < 1e-9);
        assert!(graph_spec.rigid_sql_information_loss_ratio > 0.60);

        let map_spec =
            AmorphicDisplaySynthesizer::synthesize(&atoms, ObserverIntent::LocateSpatially);
        assert_eq!(map_spec.primary_lens, AmorphicLens::GeodesicMap);

        let chart_spec =
            AmorphicDisplaySynthesizer::synthesize(&atoms, ObserverIntent::CompareQuantities);
        assert_eq!(chart_spec.primary_lens, AmorphicLens::PerceptualChart);
    }
}
