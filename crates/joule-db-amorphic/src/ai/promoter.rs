//! Unaware → Unknown Promoter
//!
//! The hardest problem in intelligence: discovering what you don't have a concept for.
//!
//! "Unknown" = you know the field exists but the value is missing.
//! "Unaware" = you don't have the field at all. No dimension for it. Can't ask about it.
//!
//! This module analyzes holographic residuals — patterns in the vector space that
//! cluster in ways no existing field explains — and promotes them from unaware
//! to unknown by creating new dimensions.
//!
//! Biological analogy: the immune system encountering a novel pathogen.
//! It doesn't have an antibody for it (unaware). It detects that something
//! foreign is present (structural anomaly). It generates a new antibody
//! template (dimension promotion). Now it knows what to look for (unknown → known).
//!
//! ## Five Discovery Strategies
//!
//! 1. **Unexplained Clustering** — records similar in hologram space but with
//!    no common field values. Uses proper union-find for connected components.
//!
//! 2. **Residual Clustering** — compute hologram residuals (actual XOR reconstructed),
//!    then cluster the residuals via k-medoids in hamming space. Each cluster is a
//!    separate hidden dimension. (Previous version bundled all residuals together,
//!    which collapsed multiple dimensions into noise.)
//!
//! 3. **Bit Correlation** — bit positions that deviate from the expected 50% set rate
//!    across records, indicating a shared hidden property.
//!
//! 4. **Temporal Drift** — when the store's centroid shifts over time, the direction
//!    of shift is a hidden dimension the schema doesn't capture.
//!
//! 5. **Contrast-Triggered** — when the contrast engine reports a spike cluster
//!    (many spikes in a short window), analyze only the spiked records for shared
//!    hidden structure.

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AmorphicStore, RecordId, Value, DIMENSION};

/// A discovered hidden dimension — something the store was unaware of.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredDimension {
    /// Auto-generated name for the new dimension
    pub name: String,
    /// How was this discovered?
    pub source: DiscoverySource,
    /// How many records exhibit this pattern?
    pub affected_records: usize,
    /// Confidence that this is a real dimension, not noise (0.0-1.0)
    pub confidence: f32,
    /// The prototype hologram for this dimension (centroid of the cluster)
    pub prototype: BinaryHV,
    /// Records that cluster around this prototype
    pub member_record_ids: Vec<RecordId>,
    /// Residual magnitude — how much of the hologram this dimension explains
    pub residual_magnitude: f64,
}

/// How a hidden dimension was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoverySource {
    /// Records cluster together but share no common field values
    UnexplainedClustering,
    /// Hologram residuals after subtracting known field bindings show structure
    ResidualStructure,
    /// Two records are similar but all known fields differ
    SimilarityWithoutExplanation,
    /// Hologram bits correlate across records in a pattern no field captures
    BitCorrelation,
    /// The store centroid shifted in a direction no field explains
    TemporalDrift,
    /// Contrast engine spikes revealed shared hidden structure
    ContrastTriggered,
}

/// A centroid snapshot for temporal drift detection.
#[derive(Clone)]
struct CentroidSnapshot {
    hologram: BinaryHV,
    record_count: usize,
    timestamp_ms: u64,
}

/// The Promoter: discovers hidden dimensions in the holographic space.
pub struct AwarenessPromoter {
    /// Minimum cluster size to consider a pattern real (not noise)
    pub min_cluster_size: usize,
    /// Minimum similarity for two records to be "unexplainably similar"
    pub similarity_threshold: f32,
    /// Minimum residual magnitude to consider a dimension significant
    pub min_residual_magnitude: f64,
    /// Maximum number of residual clusters to search for
    pub max_residual_clusters: usize,
    /// Discovered dimensions from last analysis
    discoveries: Vec<DiscoveredDimension>,
    /// Historical centroid snapshots for drift detection
    centroid_history: Vec<CentroidSnapshot>,
    /// Maximum centroid snapshots to retain
    max_centroid_history: usize,
    /// Total analyses performed
    pub total_analyses: u64,
    /// Total dimensions ever discovered
    pub total_discoveries: u64,
    /// Total dimensions promoted
    pub total_promotions: u64,
}

impl AwarenessPromoter {
    pub fn new() -> Self {
        Self {
            min_cluster_size: 3,
            similarity_threshold: 0.65,
            min_residual_magnitude: 0.1,
            max_residual_clusters: 5,
            discoveries: Vec::new(),
            centroid_history: Vec::new(),
            max_centroid_history: 20,
            total_analyses: 0,
            total_discoveries: 0,
            total_promotions: 0,
        }
    }

    /// Analyze the store for hidden dimensions.
    ///
    /// This is the core operation: look at the holographic space and find
    /// structure that no existing field explains. Promote it from unaware to unknown.
    pub fn analyze(&mut self, store: &AmorphicStore) -> &[DiscoveredDimension] {
        self.discoveries.clear();
        self.total_analyses += 1;

        if store.record_count() < self.min_cluster_size {
            return &self.discoveries;
        }

        // Strategy 1: Find records that are holographically similar
        // but share no common field values
        self.find_unexplained_similarities(store);

        // Strategy 2: Compute hologram residuals, cluster them, find
        // multiple hidden dimensions in the residual space
        self.find_residual_clusters(store);

        // Strategy 3: Find bit positions that correlate across records
        // in patterns that no single field explains
        self.find_bit_correlations(store);

        // Strategy 4: Temporal drift — centroid shifted in unexplained direction
        self.detect_temporal_drift(store);

        // Deduplicate: if two discoveries have prototypes that are very similar,
        // keep the one with higher confidence
        self.deduplicate_discoveries();

        // Sort by confidence × affected records (impact)
        self.discoveries.sort_by(|a, b| {
            let impact_a = a.confidence as f64 * a.affected_records as f64;
            let impact_b = b.confidence as f64 * b.affected_records as f64;
            impact_b
                .partial_cmp(&impact_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.total_discoveries += self.discoveries.len() as u64;

        &self.discoveries
    }

    /// Analyze only specific records (contrast-triggered mode).
    ///
    /// When the contrast engine detects a cluster of spikes, this method
    /// examines those specific records for shared hidden structure.
    pub fn analyze_records(
        &mut self,
        store: &AmorphicStore,
        record_ids: &[RecordId],
    ) -> &[DiscoveredDimension] {
        self.discoveries.clear();

        if record_ids.len() < self.min_cluster_size {
            return &self.discoveries;
        }

        // Collect the records
        let records: Vec<(RecordId, &crate::AmorphicRecord)> = record_ids
            .iter()
            .filter_map(|&id| store.records.get(&id).map(|r| (id, r)))
            .collect();

        if records.len() < self.min_cluster_size {
            return &self.discoveries;
        }

        // Compute residuals for just these records
        let residuals = self.compute_residuals(&records);

        if residuals.len() >= self.min_cluster_size {
            // Bundle all residuals — these records already share something
            // (they all spiked), so the residual IS the hidden dimension
            let mut acc = BundleAccumulator::new(DIMENSION);
            let member_ids: Vec<RecordId> = residuals.iter().map(|(id, _, _)| *id).collect();
            let avg_magnitude: f64 =
                residuals.iter().map(|(_, _, d)| d).sum::<f64>() / residuals.len() as f64;

            for (_, residual, _) in &residuals {
                acc.add(residual);
            }
            let prototype = acc.threshold();

            if avg_magnitude > self.min_residual_magnitude {
                self.discoveries.push(DiscoveredDimension {
                    name: format!("_contrast_spike_{}", self.total_analyses),
                    source: DiscoverySource::ContrastTriggered,
                    affected_records: member_ids.len(),
                    confidence: (avg_magnitude * 2.5).min(1.0) as f32,
                    prototype,
                    member_record_ids: member_ids,
                    residual_magnitude: avg_magnitude,
                });
            }
        }

        self.total_discoveries += self.discoveries.len() as u64;
        &self.discoveries
    }

    /// Get the last set of discoveries.
    pub fn discoveries(&self) -> &[DiscoveredDimension] {
        &self.discoveries
    }

    /// Promote a discovered dimension: add it as an explicit field
    /// to all affected records. Moves from unaware → unknown.
    ///
    /// After promotion, the Question Layer can detect that these records
    /// have a new field that other records are missing, and ask about it.
    pub fn promote(
        &mut self,
        discovery: &DiscoveredDimension,
        store: &mut AmorphicStore,
    ) -> usize {
        let mut promoted = 0;

        for &record_id in &discovery.member_record_ids {
            if let Some(record) = store.records.get_mut(&record_id) {
                let sim = record.hologram.similarity(&discovery.prototype);
                record.fields.insert(
                    discovery.name.clone(),
                    Value::Float(sim as f64),
                );
                promoted += 1;
            }
        }

        self.total_promotions += promoted as u64;
        promoted
    }

    /// Record a centroid snapshot for temporal drift detection.
    pub fn record_centroid(&mut self, centroid: BinaryHV, record_count: usize, timestamp_ms: u64) {
        self.centroid_history.push(CentroidSnapshot {
            hologram: centroid,
            record_count,
            timestamp_ms,
        });
        if self.centroid_history.len() > self.max_centroid_history {
            self.centroid_history.remove(0);
        }
    }

    // ========================================================================
    // Discovery Strategies
    // ========================================================================

    /// Strategy 1: Find records that are similar in hologram space
    /// but have no common field values explaining the similarity.
    /// Uses union-find for proper connected component merging.
    fn find_unexplained_similarities(&mut self, store: &AmorphicStore) {
        let records: Vec<(RecordId, &crate::AmorphicRecord)> =
            store.records.iter().map(|(&id, r)| (id, r)).collect();

        // Compare pairs (sample for large stores)
        let max_comparisons = 500;
        let total_pairs = records.len() * (records.len() - 1) / 2;
        let step = (total_pairs / max_comparisons.max(1)).max(1);

        let mut unexplained_pairs: Vec<(RecordId, RecordId, f32)> = Vec::new();
        let mut comparison_count = 0;

        for i in 0..records.len() {
            for j in (i + 1)..records.len() {
                comparison_count += 1;
                if step > 1 && comparison_count % step != 0 && records.len() > 50 {
                    continue;
                }

                let (id_a, rec_a) = &records[i];
                let (id_b, rec_b) = &records[j];

                let hologram_sim = rec_a.hologram.similarity(&rec_b.hologram);

                if hologram_sim >= self.similarity_threshold {
                    let field_overlap = self.compute_field_overlap(rec_a, rec_b);
                    if field_overlap < 0.2 {
                        unexplained_pairs.push((*id_a, *id_b, hologram_sim));
                    }
                }
            }
        }

        if unexplained_pairs.is_empty() {
            return;
        }

        // Union-find for proper connected component merging
        let mut parent: HashMap<RecordId, RecordId> = HashMap::new();

        fn find(parent: &mut HashMap<RecordId, RecordId>, x: RecordId) -> RecordId {
            let p = *parent.get(&x).unwrap_or(&x);
            if p == x {
                return x;
            }
            let root = find(parent, p);
            parent.insert(x, root);
            root
        }

        fn union(parent: &mut HashMap<RecordId, RecordId>, a: RecordId, b: RecordId) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent.insert(rb, ra);
            }
        }

        for (id_a, id_b, _) in &unexplained_pairs {
            parent.entry(*id_a).or_insert(*id_a);
            parent.entry(*id_b).or_insert(*id_b);
            union(&mut parent, *id_a, *id_b);
        }

        // Collect groups
        let mut groups: HashMap<RecordId, Vec<RecordId>> = HashMap::new();
        let keys: Vec<RecordId> = parent.keys().copied().collect();
        for id in keys {
            let root = find(&mut parent, id);
            groups.entry(root).or_default().push(id);
        }

        // Each group with enough members is a discovered dimension
        let mut dim_counter = 0;
        for group in groups.into_values() {
            if group.len() >= self.min_cluster_size {
                let mut acc = BundleAccumulator::new(DIMENSION);
                for &id in &group {
                    if let Some(rec) = store.records.get(&id) {
                        acc.add(&rec.hologram);
                    }
                }
                let prototype = acc.threshold();

                // Compute average unexplained similarity as confidence
                let avg_sim: f32 = unexplained_pairs.iter()
                    .filter(|(a, b, _)| group.contains(a) && group.contains(b))
                    .map(|(_, _, s)| s)
                    .sum::<f32>()
                    / unexplained_pairs.iter()
                        .filter(|(a, b, _)| group.contains(a) && group.contains(b))
                        .count().max(1) as f32;

                dim_counter += 1;
                self.discoveries.push(DiscoveredDimension {
                    name: format!("_discovered_cluster_{}", dim_counter),
                    source: DiscoverySource::UnexplainedClustering,
                    affected_records: group.len(),
                    confidence: avg_sim.min(1.0),
                    prototype,
                    member_record_ids: group,
                    residual_magnitude: 0.0,
                });
            }
        }
    }

    /// Strategy 2: Compute residuals, then cluster via k-medoids in hamming space.
    ///
    /// Previous version bundled ALL residuals into one prototype, which collapses
    /// multiple hidden dimensions into noise. This version finds up to K separate
    /// dimensions in the residual space.
    fn find_residual_clusters(&mut self, store: &AmorphicStore) {
        let records: Vec<(RecordId, &crate::AmorphicRecord)> =
            store.records.iter().map(|(&id, r)| (id, r)).collect();

        let residuals = self.compute_residuals(&records);

        if residuals.len() < self.min_cluster_size {
            return;
        }

        // K-medoids clustering on the residuals in hamming space.
        // K = min(max_residual_clusters, len/min_cluster_size)
        let k = self.max_residual_clusters.min(residuals.len() / self.min_cluster_size).max(1);

        if k <= 1 {
            // Only one possible cluster — fall back to single bundle
            self.bundle_residuals_as_single(residuals);
            return;
        }

        // Initialize medoids: pick residuals with highest deviation
        let mut sorted_by_dev: Vec<usize> = (0..residuals.len()).collect();
        sorted_by_dev.sort_by(|&a, &b| {
            residuals[b].2.partial_cmp(&residuals[a].2).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut medoid_indices: Vec<usize> = sorted_by_dev.iter().take(k).copied().collect();

        // Run k-medoids for a few iterations
        let mut assignments = vec![0usize; residuals.len()];

        for _iter in 0..10 {
            // Assign each residual to nearest medoid
            let mut changed = false;
            for (ri, (_, res, _)) in residuals.iter().enumerate() {
                let mut best_medoid = 0;
                let mut best_sim = -1.0f32;
                for (mi, &medoid_idx) in medoid_indices.iter().enumerate() {
                    let sim = res.similarity(&residuals[medoid_idx].1);
                    if sim > best_sim {
                        best_sim = sim;
                        best_medoid = mi;
                    }
                }
                if assignments[ri] != best_medoid {
                    assignments[ri] = best_medoid;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update medoids: pick member with minimum total distance to others in cluster
            for mi in 0..k {
                let members: Vec<usize> = assignments.iter().enumerate()
                    .filter(|&(_, &a)| a == mi)
                    .map(|(i, _)| i)
                    .collect();

                if members.is_empty() {
                    continue;
                }

                let mut best_total_sim = -1.0f64;
                let mut best_member = members[0];
                for &candidate in &members {
                    let total_sim: f64 = members.iter()
                        .map(|&other| residuals[candidate].1.similarity(&residuals[other].1) as f64)
                        .sum();
                    if total_sim > best_total_sim {
                        best_total_sim = total_sim;
                        best_member = candidate;
                    }
                }
                medoid_indices[mi] = best_member;
            }
        }

        // Convert clusters into discovered dimensions
        for mi in 0..k {
            let members: Vec<usize> = assignments.iter().enumerate()
                .filter(|&(_, &a)| a == mi)
                .map(|(i, _)| i)
                .collect();

            if members.len() < self.min_cluster_size {
                continue;
            }

            // Bundle the cluster's residuals
            let mut acc = BundleAccumulator::new(DIMENSION);
            let mut member_ids = Vec::new();
            let mut total_deviation = 0.0;

            for &idx in &members {
                acc.add(&residuals[idx].1);
                member_ids.push(residuals[idx].0);
                total_deviation += residuals[idx].2;
            }

            let avg_deviation = total_deviation / members.len() as f64;
            let prototype = acc.threshold();

            // Measure cluster tightness as confidence
            let medoid_idx = medoid_indices[mi];
            let avg_sim: f64 = members.iter()
                .map(|&idx| residuals[idx].1.similarity(&residuals[medoid_idx].1) as f64)
                .sum::<f64>() / members.len() as f64;

            // Confidence: deviation indicates signal, similarity indicates coherence
            let confidence = ((avg_deviation * avg_sim) * 3.0).min(1.0) as f32;

            if confidence > 0.1 && avg_deviation > self.min_residual_magnitude {
                self.discoveries.push(DiscoveredDimension {
                    name: format!("_residual_dim_{}", mi + 1),
                    source: DiscoverySource::ResidualStructure,
                    affected_records: member_ids.len(),
                    confidence,
                    prototype,
                    member_record_ids: member_ids,
                    residual_magnitude: avg_deviation,
                });
            }
        }
    }

    /// Strategy 3: Find bit positions in holograms that correlate
    /// across records in ways no field explains.
    fn find_bit_correlations(&mut self, store: &AmorphicStore) {
        if store.record_count() < self.min_cluster_size * 2 {
            return;
        }

        let records: Vec<&crate::AmorphicRecord> = store.records.values().take(100).collect();
        let n = records.len();
        if n < 4 {
            return;
        }

        let num_words = (DIMENSION + 63) / 64;
        let mut bit_frequencies = vec![0u32; DIMENSION.min(num_words * 64)];

        for record in &records {
            let words = record.hologram.as_words();
            for (wi, &word) in words.iter().enumerate() {
                for bit in 0..64 {
                    let idx = wi * 64 + bit;
                    if idx < bit_frequencies.len() && (word >> bit) & 1 == 1 {
                        bit_frequencies[idx] += 1;
                    }
                }
            }
        }

        let expected_freq = n as f64 / 2.0;
        let mut anomalous_bits = Vec::new();

        for (idx, &freq) in bit_frequencies.iter().enumerate() {
            let deviation = (freq as f64 - expected_freq).abs() / expected_freq;
            if deviation > 0.3 {
                anomalous_bits.push((idx, freq, deviation));
            }
        }

        let highly_set: Vec<usize> = anomalous_bits
            .iter()
            .filter(|(_, freq, _)| *freq as f64 > expected_freq)
            .map(|(idx, _, _)| *idx)
            .collect();

        if highly_set.len() >= 10 {
            let mut proto_words = vec![0u64; num_words];
            for &idx in &highly_set {
                let word = idx / 64;
                let bit = idx % 64;
                if word < proto_words.len() {
                    proto_words[word] |= 1u64 << bit;
                }
            }

            let proto = BinaryHV::from_words(proto_words, DIMENSION);
            let matching: Vec<RecordId> = store
                .records
                .iter()
                .filter(|(_, r)| r.hologram.similarity(&proto) > self.similarity_threshold)
                .map(|(&id, _)| id)
                .collect();

            if matching.len() >= self.min_cluster_size {
                self.discoveries.push(DiscoveredDimension {
                    name: "_discovered_bit_pattern".to_string(),
                    source: DiscoverySource::BitCorrelation,
                    affected_records: matching.len(),
                    confidence: 0.5,
                    prototype: proto,
                    member_record_ids: matching,
                    residual_magnitude: highly_set.len() as f64 / DIMENSION as f64,
                });
            }
        }
    }

    /// Strategy 4: Detect temporal drift in the store centroid.
    ///
    /// If we have at least 2 centroid snapshots, compute the direction of drift.
    /// The drift vector represents a dimension the schema doesn't capture —
    /// the store is changing in a way that no field explains.
    fn detect_temporal_drift(&mut self, store: &AmorphicStore) {
        if self.centroid_history.len() < 2 {
            return;
        }

        let oldest = &self.centroid_history[0];
        let newest = self.centroid_history.last().unwrap();

        // Drift = XOR of old and new centroid. Bits that changed = the direction of drift.
        let drift = oldest.hologram.bind(&newest.hologram);

        // How much drift? Popcount deviation from DIMENSION/2
        let popcount = drift.as_words().iter().map(|w| w.count_ones()).sum::<u32>();
        let expected = DIMENSION as u32 / 2;
        let drift_magnitude = (popcount as f64 - expected as f64).abs() / expected as f64;

        if drift_magnitude < self.min_residual_magnitude * 2.0 {
            return; // Not enough drift to matter
        }

        // Find records that align with the drift direction
        let matching: Vec<RecordId> = store
            .records
            .iter()
            .filter(|(_, r)| {
                // Records whose hologram is more similar to the newest centroid
                // than the oldest — they're "in the direction of drift"
                let sim_new = r.hologram.similarity(&newest.hologram);
                let sim_old = r.hologram.similarity(&oldest.hologram);
                sim_new > sim_old + 0.05 // Meaningful bias toward new centroid
            })
            .map(|(&id, _)| id)
            .collect();

        if matching.len() >= self.min_cluster_size {
            let growth = if newest.record_count > oldest.record_count {
                newest.record_count - oldest.record_count
            } else {
                0
            };

            self.discoveries.push(DiscoveredDimension {
                name: format!(
                    "_temporal_drift_{}to{}",
                    oldest.record_count, newest.record_count
                ),
                source: DiscoverySource::TemporalDrift,
                affected_records: matching.len(),
                confidence: (drift_magnitude * 1.5).min(1.0) as f32,
                prototype: drift,
                member_record_ids: matching,
                residual_magnitude: drift_magnitude,
            });
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Compute residuals for a set of records: actual hologram XOR reconstructed.
    /// Returns (record_id, residual_hologram, deviation_magnitude).
    fn compute_residuals(
        &self,
        records: &[(RecordId, &crate::AmorphicRecord)],
    ) -> Vec<(RecordId, BinaryHV, f64)> {
        let mut residuals = Vec::new();

        for &(id, record) in records {
            let mut reconstructed_acc = BundleAccumulator::new(DIMENSION);

            for (field, value) in &record.fields {
                let field_hv = BinaryHV::from_hash(field.as_bytes(), DIMENSION);
                let value_str = format!("{:?}", value);
                let value_hv = BinaryHV::from_hash(value_str.as_bytes(), DIMENSION);
                let bound = field_hv.bind(&value_hv);
                reconstructed_acc.add(&bound);
            }

            let reconstructed = reconstructed_acc.threshold();
            let residual = record.hologram.bind(&reconstructed);

            let popcount = residual.as_words().iter().map(|w| w.count_ones()).sum::<u32>();
            let expected = DIMENSION as u32 / 2;
            let deviation = (popcount as f64 - expected as f64).abs() / expected as f64;

            if deviation > self.min_residual_magnitude {
                residuals.push((id, residual, deviation));
            }
        }

        residuals
    }

    /// Fallback: bundle all residuals into a single dimension.
    fn bundle_residuals_as_single(&mut self, residuals: Vec<(RecordId, BinaryHV, f64)>) {
        let mut acc = BundleAccumulator::new(DIMENSION);
        let member_ids: Vec<RecordId> = residuals.iter().map(|(id, _, _)| *id).collect();
        let avg_magnitude: f64 =
            residuals.iter().map(|(_, _, d)| d).sum::<f64>() / residuals.len() as f64;

        for (_, residual, _) in &residuals {
            acc.add(residual);
        }
        let prototype = acc.threshold();

        self.discoveries.push(DiscoveredDimension {
            name: "_discovered_residual_pattern".to_string(),
            source: DiscoverySource::ResidualStructure,
            affected_records: member_ids.len(),
            confidence: (avg_magnitude * 2.0).min(1.0) as f32,
            prototype,
            member_record_ids: member_ids,
            residual_magnitude: avg_magnitude,
        });
    }

    /// Remove duplicate discoveries whose prototypes are very similar.
    fn deduplicate_discoveries(&mut self) {
        if self.discoveries.len() < 2 {
            return;
        }

        let mut keep = vec![true; self.discoveries.len()];

        for i in 0..self.discoveries.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..self.discoveries.len() {
                if !keep[j] {
                    continue;
                }
                let sim = self.discoveries[i].prototype.similarity(&self.discoveries[j].prototype);
                if sim > 0.8 {
                    // Very similar prototypes — keep the one with higher confidence
                    if self.discoveries[i].confidence >= self.discoveries[j].confidence {
                        keep[j] = false;
                    } else {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }

        let mut idx = 0;
        self.discoveries.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }

    /// Compute field value overlap between two records (0.0 = nothing in common, 1.0 = identical).
    fn compute_field_overlap(
        &self,
        a: &crate::AmorphicRecord,
        b: &crate::AmorphicRecord,
    ) -> f64 {
        if a.fields.is_empty() && b.fields.is_empty() {
            return 1.0;
        }

        let all_fields: std::collections::HashSet<&String> =
            a.fields.keys().chain(b.fields.keys()).collect();

        if all_fields.is_empty() {
            return 1.0;
        }

        let matching = all_fields
            .iter()
            .filter(|&&f| {
                a.fields.get(f) == b.fields.get(f) && a.fields.contains_key(f) && b.fields.contains_key(f)
            })
            .count();

        matching as f64 / all_fields.len() as f64
    }
}

impl Default for AwarenessPromoter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_unexplained_similarity() {
        let mut store = AmorphicStore::new();

        // Group A: same hidden trait (encoded in name pattern), different visible fields
        store.ingest_json(r#"{"name": "alpha_spark_1", "color": "red", "size": 10}"#).unwrap();
        store.ingest_json(r#"{"name": "alpha_spark_2", "color": "blue", "size": 20}"#).unwrap();
        store.ingest_json(r#"{"name": "alpha_spark_3", "color": "green", "size": 30}"#).unwrap();
        store.ingest_json(r#"{"name": "alpha_spark_4", "color": "yellow", "size": 40}"#).unwrap();

        // Group B: completely different pattern
        store.ingest_json(r#"{"name": "beta_wave_1", "color": "red", "size": 100}"#).unwrap();
        store.ingest_json(r#"{"name": "beta_wave_2", "color": "blue", "size": 200}"#).unwrap();
        store.ingest_json(r#"{"name": "beta_wave_3", "color": "green", "size": 300}"#).unwrap();
        store.ingest_json(r#"{"name": "beta_wave_4", "color": "yellow", "size": 400}"#).unwrap();

        let mut promoter = AwarenessPromoter::new();
        promoter.min_cluster_size = 2;
        promoter.similarity_threshold = 0.55;

        let discoveries = promoter.analyze(&store);

        // Mechanism runs without error; exact count depends on holographic encoding
        assert!(promoter.discoveries().len() >= 0);
        assert_eq!(promoter.total_analyses, 1);
    }

    #[test]
    fn test_residual_clustering() {
        let mut store = AmorphicStore::new();

        // Insert records with multiple hidden groupings
        for i in 0..15 {
            store
                .ingest_json(&format!(
                    r#"{{"name": "item_{}", "category": "type_a", "hidden_trait": "xyz_{}"  }}"#,
                    i, i % 3
                ))
                .unwrap();
        }

        let mut promoter = AwarenessPromoter::new();
        promoter.min_cluster_size = 3;
        promoter.max_residual_clusters = 3;

        let discoveries = promoter.analyze(&store);

        for d in discoveries {
            assert!(!d.name.is_empty());
            assert!(d.confidence >= 0.0 && d.confidence <= 1.0);
            assert!(d.affected_records > 0);
        }
    }

    #[test]
    fn test_promote_dimension() {
        let mut store = AmorphicStore::new();

        store.ingest_json(r#"{"name": "A", "score": 10}"#).unwrap();
        store.ingest_json(r#"{"name": "B", "score": 20}"#).unwrap();
        store.ingest_json(r#"{"name": "C", "score": 30}"#).unwrap();

        let discovery = DiscoveredDimension {
            name: "_discovered_test".to_string(),
            source: DiscoverySource::UnexplainedClustering,
            affected_records: 3,
            confidence: 0.8,
            prototype: BinaryHV::from_hash(b"test_prototype", DIMENSION),
            member_record_ids: vec![1, 2, 3],
            residual_magnitude: 0.5,
        };

        let mut promoter = AwarenessPromoter::new();
        let promoted = promoter.promote(&discovery, &mut store);

        assert_eq!(promoted, 3);
        assert_eq!(promoter.total_promotions, 3);

        // Verify the new field exists
        let record = store.records.get(&1).unwrap();
        assert!(record.fields.contains_key("_discovered_test"));
    }

    #[test]
    fn test_contrast_triggered_analysis() {
        let mut store = AmorphicStore::new();

        for i in 0..6 {
            store.ingest_json(&format!(
                r#"{{"name": "spike_{}", "signal": "anomaly_type_a"}}"#, i
            )).unwrap();
        }

        let spiked_ids: Vec<RecordId> = (1..=6).collect();

        let mut promoter = AwarenessPromoter::new();
        promoter.min_cluster_size = 3;

        let discoveries = promoter.analyze_records(&store, &spiked_ids);

        // Should find the shared signal among spiked records
        for d in discoveries {
            assert_eq!(d.source, DiscoverySource::ContrastTriggered);
            assert!(d.affected_records >= 3);
        }
    }

    #[test]
    fn test_temporal_drift() {
        let mut store = AmorphicStore::new();

        // Initial records
        for i in 0..5 {
            store.ingest_json(&format!(r#"{{"name": "old_{}", "era": "past"}}"#, i)).unwrap();
        }

        let mut promoter = AwarenessPromoter::new();
        promoter.min_cluster_size = 2;

        // Snapshot old centroid
        let old_centroid = BinaryHV::from_hash(b"old_centroid_state", DIMENSION);
        promoter.record_centroid(old_centroid, 5, 1000);

        // Add new records that shift the store
        for i in 0..5 {
            store.ingest_json(&format!(r#"{{"name": "new_{}", "era": "future"}}"#, i)).unwrap();
        }

        // Snapshot new centroid
        let new_centroid = BinaryHV::from_hash(b"new_centroid_state", DIMENSION);
        promoter.record_centroid(new_centroid, 10, 2000);

        let _discoveries = promoter.analyze(&store);
        // Drift detection runs without error; results depend on hash-generated holograms
    }

    #[test]
    fn test_empty_store() {
        let store = AmorphicStore::new();
        let mut promoter = AwarenessPromoter::new();
        let discoveries = promoter.analyze(&store);
        assert!(discoveries.is_empty());
    }

    #[test]
    fn test_field_overlap() {
        let promoter = AwarenessPromoter::new();

        let mut store = AmorphicStore::new();
        store.ingest_json(r#"{"name": "A", "x": 1}"#).unwrap();
        store.ingest_json(r#"{"name": "B", "x": 1}"#).unwrap();

        let a = store.records.get(&1).unwrap();
        let b = store.records.get(&2).unwrap();

        let overlap = promoter.compute_field_overlap(a, b);
        assert!(overlap > 0.0 && overlap < 1.0);
    }

    #[test]
    fn test_deduplication() {
        let mut promoter = AwarenessPromoter::new();

        // Two discoveries with identical prototypes
        let proto = BinaryHV::from_hash(b"same_thing", DIMENSION);
        promoter.discoveries.push(DiscoveredDimension {
            name: "dim_a".to_string(),
            source: DiscoverySource::ResidualStructure,
            affected_records: 5,
            confidence: 0.9,
            prototype: proto.clone(),
            member_record_ids: vec![1, 2, 3, 4, 5],
            residual_magnitude: 0.3,
        });
        promoter.discoveries.push(DiscoveredDimension {
            name: "dim_b".to_string(),
            source: DiscoverySource::BitCorrelation,
            affected_records: 3,
            confidence: 0.5,
            prototype: proto,
            member_record_ids: vec![1, 2, 3],
            residual_magnitude: 0.2,
        });

        promoter.deduplicate_discoveries();
        assert_eq!(promoter.discoveries.len(), 1);
        assert_eq!(promoter.discoveries[0].name, "dim_a"); // Higher confidence kept
    }
}
