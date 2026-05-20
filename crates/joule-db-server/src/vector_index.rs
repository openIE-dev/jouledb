//! Vector Index Manager — bridges HNSW/LSH/IVF/SQ/PQ indexes with SQL execution

use joule_db_core::index::hnsw::{DistanceMetric as CoreDistanceMetric, HnswConfig, HnswIndex};
use joule_db_hdc::manifold::{
    DistanceMetric, HNSWIndex, IVFIndex, LSHIndex, ProductQuantizer, ScalarQuantizer,
};
use joule_db_query::executor::TableStorage;
use std::collections::HashMap;
use std::sync::RwLock;

/// A live vector index (HNSW, LSH, IVF, HNSW+SQ, IVF+PQ, or HNSW-Core)
pub enum VectorIndex {
    Hnsw(HNSWIndex),
    Lsh(LSHIndex),
    Ivf(IVFIndex),
    HnswSq(HNSWIndex, ScalarQuantizer),
    IvfPq(IVFIndex, ProductQuantizer),
    /// Pure-Rust HNSW from joule-db-core (no HDC dependency, u64 row IDs)
    HnswCore(HnswIndex, HashMap<u64, String>, HashMap<String, u64>, u64),
}

/// Metadata + live index for a single vector index
pub struct VectorIndexInfo {
    pub index: VectorIndex,
    pub table: String,
    pub column: String,
    pub metric: DistanceMetric,
}

/// Manages all live vector indexes in the server
pub struct VectorIndexManager {
    /// index_name -> VectorIndexInfo
    indexes: RwLock<HashMap<String, VectorIndexInfo>>,
}

impl VectorIndexManager {
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Build an index from existing table rows.
    ///
    /// `rows` is a list of (row_id, vector_f32) tuples.
    pub fn build_index(
        &self,
        name: &str,
        table: &str,
        column: &str,
        method: &str,
        metric: DistanceMetric,
        options: &HashMap<String, String>,
        rows: Vec<(String, Vec<f32>)>,
    ) -> Result<(), String> {
        let dimension = if let Some((_, first_vec)) = rows.first() {
            first_vec.len()
        } else {
            options
                .get("dimension")
                .and_then(|d| d.parse::<usize>().ok())
                .unwrap_or(0)
        };

        // Use dimension=1 as placeholder if no rows yet; index will work once vectors are inserted
        let dim = if dimension == 0 { 1 } else { dimension };

        let mut vi = match method.to_uppercase().as_str() {
            "LSH" => {
                let num_tables = options
                    .get("num_tables")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8);
                let num_bits = options
                    .get("num_bits")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                VectorIndex::Lsh(LSHIndex::with_metric(dim, num_tables, num_bits, metric))
            }
            "IVF" => {
                let n_clusters = options
                    .get("n_clusters")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                let n_probe = options
                    .get("n_probe")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4);
                let mut ivf = IVFIndex::with_metric(dim, n_clusters, n_probe, metric);
                ivf.train(&rows)
                    .map_err(|e| format!("IVF train error: {}", e))?;
                // rows are already inserted during train, return early
                let info = VectorIndexInfo {
                    index: VectorIndex::Ivf(ivf),
                    table: table.to_string(),
                    column: column.to_string(),
                    metric,
                };
                let mut indexes = self
                    .indexes
                    .write()
                    .map_err(|e| format!("lock error: {}", e))?;
                indexes.insert(name.to_string(), info);
                return Ok(());
            }
            "HNSW_SQ" | "HNSW+SQ" => {
                let max_connections = options
                    .get("m")
                    .or_else(|| options.get("max_connections"))
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                let ef_construction = options
                    .get("ef_construction")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(200);
                let mut hnsw =
                    HNSWIndex::with_metric(dim, max_connections, ef_construction, metric);
                let mut sq = ScalarQuantizer::with_metric(dim, metric);
                sq.train(&rows)
                    .map_err(|e| format!("SQ train error: {}", e))?;
                // Insert into HNSW (it also uses the original vectors for graph construction)
                for (row_id, vector) in &rows {
                    let _ = hnsw.insert(row_id.clone(), vector.clone());
                }
                let info = VectorIndexInfo {
                    index: VectorIndex::HnswSq(hnsw, sq),
                    table: table.to_string(),
                    column: column.to_string(),
                    metric,
                };
                let mut indexes = self
                    .indexes
                    .write()
                    .map_err(|e| format!("lock error: {}", e))?;
                indexes.insert(name.to_string(), info);
                return Ok(());
            }
            "IVF_PQ" | "IVF+PQ" => {
                let n_clusters = options
                    .get("n_clusters")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                let n_probe = options
                    .get("n_probe")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4);
                let n_subquantizers = options
                    .get("n_subquantizers")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8.min(dim));
                let mut ivf = IVFIndex::with_metric(dim, n_clusters, n_probe, metric);
                let mut pq = ProductQuantizer::with_metric(dim, n_subquantizers, metric);
                ivf.train(&rows)
                    .map_err(|e| format!("IVF train error: {}", e))?;
                pq.train(&rows)
                    .map_err(|e| format!("PQ train error: {}", e))?;
                let info = VectorIndexInfo {
                    index: VectorIndex::IvfPq(ivf, pq),
                    table: table.to_string(),
                    column: column.to_string(),
                    metric,
                };
                let mut indexes = self
                    .indexes
                    .write()
                    .map_err(|e| format!("lock error: {}", e))?;
                indexes.insert(name.to_string(), info);
                return Ok(());
            }
            "HNSW_CORE" => {
                let m = options
                    .get("m")
                    .or_else(|| options.get("max_connections"))
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                let ef_construction = options
                    .get("ef_construction")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(200);
                let ef_search = options
                    .get("ef_search")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(50);
                let core_metric = hdc_to_core_metric(metric);
                let config = HnswConfig {
                    m,
                    m0: m * 2,
                    ef_construction,
                    ef_search,
                    metric: core_metric,
                    dimensions: dim,
                    ml: 1.0 / (m as f64).ln(),
                };
                let mut hnsw = HnswIndex::new(config);
                let mut id_to_str: HashMap<u64, String> = HashMap::new();
                let mut str_to_id: HashMap<String, u64> = HashMap::new();
                let mut next_id: u64 = 0;
                for (row_id, vector) in rows {
                    hnsw.insert(next_id, &vector);
                    id_to_str.insert(next_id, row_id.clone());
                    str_to_id.insert(row_id, next_id);
                    next_id += 1;
                }
                let info = VectorIndexInfo {
                    index: VectorIndex::HnswCore(hnsw, id_to_str, str_to_id, next_id),
                    table: table.to_string(),
                    column: column.to_string(),
                    metric,
                };
                let mut indexes = self
                    .indexes
                    .write()
                    .map_err(|e| format!("lock error: {}", e))?;
                indexes.insert(name.to_string(), info);
                return Ok(());
            }
            _ => {
                let max_connections = options
                    .get("m")
                    .or_else(|| options.get("max_connections"))
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                let ef_construction = options
                    .get("ef_construction")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(200);
                VectorIndex::Hnsw(HNSWIndex::with_metric(
                    dim,
                    max_connections,
                    ef_construction,
                    metric,
                ))
            }
        };

        for (row_id, vector) in rows {
            match &mut vi {
                VectorIndex::Hnsw(hnsw) => {
                    let _ = hnsw.insert(row_id, vector);
                }
                VectorIndex::Lsh(lsh) => {
                    let _ = lsh.insert(row_id, vector);
                }
                _ => {} // IVF/HnswSq/IvfPq/HnswCore handled above with early return
            }
        }

        let info = VectorIndexInfo {
            index: vi,
            table: table.to_string(),
            column: column.to_string(),
            metric,
        };

        let mut indexes = self
            .indexes
            .write()
            .map_err(|e| format!("lock error: {}", e))?;
        indexes.insert(name.to_string(), info);
        Ok(())
    }

    /// Insert a single vector into an existing index (incremental update on INSERT).
    pub fn insert_into_index(
        &self,
        index_name: &str,
        row_id: String,
        vector: Vec<f32>,
    ) -> Result<(), String> {
        let mut indexes = self
            .indexes
            .write()
            .map_err(|e| format!("lock error: {}", e))?;
        let info = indexes
            .get_mut(index_name)
            .ok_or_else(|| format!("Vector index '{}' not found", index_name))?;

        match &mut info.index {
            VectorIndex::Hnsw(hnsw) => {
                hnsw.insert(row_id, vector)
                    .map_err(|e| format!("HNSW insert error: {}", e))?;
            }
            VectorIndex::Lsh(lsh) => {
                lsh.insert(row_id, vector)
                    .map_err(|e| format!("LSH insert error: {}", e))?;
            }
            VectorIndex::Ivf(ivf) => {
                ivf.insert(row_id, vector)
                    .map_err(|e| format!("IVF insert error: {}", e))?;
            }
            VectorIndex::HnswSq(hnsw, sq) => {
                hnsw.insert(row_id.clone(), vector.clone())
                    .map_err(|e| format!("HNSW insert error: {}", e))?;
                sq.insert(row_id, vector)
                    .map_err(|e| format!("SQ insert error: {}", e))?;
            }
            VectorIndex::IvfPq(ivf, pq) => {
                ivf.insert(row_id.clone(), vector.clone())
                    .map_err(|e| format!("IVF insert error: {}", e))?;
                pq.insert(row_id, vector)
                    .map_err(|e| format!("PQ insert error: {}", e))?;
            }
            VectorIndex::HnswCore(hnsw, id_to_str, str_to_id, next_id) => {
                let id = *next_id;
                *next_id += 1;
                hnsw.insert(id, &vector);
                id_to_str.insert(id, row_id.clone());
                str_to_id.insert(row_id, id);
            }
        }
        Ok(())
    }

    /// K-nearest-neighbor search on a named index.
    /// Returns (row_id, distance) pairs sorted by distance ascending.
    pub fn knn_search(
        &self,
        index_name: &str,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        self.knn_search_with_ef(index_name, query_vector, k, None)
    }

    /// K-nearest-neighbor search with optional ef_search parameter.
    /// `ef_search` only applies to HNSW-based indexes. For others, it's ignored.
    pub fn knn_search_with_ef(
        &self,
        index_name: &str,
        query_vector: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<(String, f32)>, String> {
        let indexes = self
            .indexes
            .read()
            .map_err(|e| format!("lock error: {}", e))?;
        let info = indexes
            .get(index_name)
            .ok_or_else(|| format!("Vector index '{}' not found", index_name))?;

        let results: Vec<(String, f32)> = match &info.index {
            VectorIndex::Hnsw(hnsw) => match ef_search {
                Some(ef) => hnsw.query_with_ef(query_vector, k, ef),
                None => hnsw.query(query_vector, k),
            }
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect(),
            VectorIndex::Lsh(lsh) => lsh
                .query(query_vector, k)
                .into_iter()
                .map(|r| (r.id, r.distance))
                .collect(),
            VectorIndex::Ivf(ivf) => ivf
                .query(query_vector, k)
                .into_iter()
                .map(|r| (r.id, r.distance))
                .collect(),
            VectorIndex::HnswSq(hnsw, _sq) => {
                // Use HNSW graph for neighbor traversal (SQ is for storage compression)
                match ef_search {
                    Some(ef) => hnsw.query_with_ef(query_vector, k, ef),
                    None => hnsw.query(query_vector, k),
                }
                .into_iter()
                .map(|r| (r.id, r.distance))
                .collect()
            }
            VectorIndex::IvfPq(ivf, _pq) => {
                // Use IVF for coarse search (PQ is for compressed distance)
                ivf.query(query_vector, k)
                    .into_iter()
                    .map(|r| (r.id, r.distance))
                    .collect()
            }
            VectorIndex::HnswCore(hnsw, id_to_str, _, _) => {
                hnsw.search(query_vector, k)
                    .into_iter()
                    .filter_map(|(id, dist)| {
                        id_to_str.get(&id).map(|s| (s.clone(), dist))
                    })
                    .collect()
            }
        };

        Ok(results)
    }

    /// Check if any vector index covers the given table + column.
    /// Returns the index name if found.
    pub fn find_index_for(&self, table: &str, column: &str) -> Option<String> {
        let indexes = self.indexes.read().ok()?;
        for (name, info) in indexes.iter() {
            if info.table.eq_ignore_ascii_case(table) && info.column.eq_ignore_ascii_case(column) {
                return Some(name.clone());
            }
        }
        None
    }

    /// Check if any vector index exists for the given table (any column).
    /// Returns (index_name, column_name) pairs.
    pub fn indexes_for_table(&self, table: &str) -> Vec<(String, String)> {
        let indexes = match self.indexes.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        indexes
            .iter()
            .filter(|(_, info)| info.table.eq_ignore_ascii_case(table))
            .map(|(name, info)| (name.clone(), info.column.clone()))
            .collect()
    }

    /// Drop a named index.
    pub fn drop_index(&self, name: &str) -> Result<(), String> {
        let mut indexes = self
            .indexes
            .write()
            .map_err(|e| format!("lock error: {}", e))?;
        indexes.remove(name);
        Ok(())
    }

    /// Rebuild all vector indexes from metadata + table data.
    /// Called on server startup.
    pub fn rebuild_from_metadata(
        &self,
        amorphic: &crate::amorphic_adapter::AmorphicTableStorage,
    ) -> Result<usize, String> {
        let index_records = amorphic.scan("__indexes__").unwrap_or_default();

        let mut count = 0;
        for row in &index_records {
            let idx_type_pos = row
                .columns
                .iter()
                .position(|c: &String| c == "__index_type__");
            let idx_type = idx_type_pos.and_then(|i| {
                row.values.get(i).and_then(|v| match v {
                    joule_db_query::ast::Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
            });

            if idx_type != Some("vector") {
                continue;
            }

            let get_str = |col: &str| -> Option<String> {
                let pos = row.columns.iter().position(|c: &String| c == col)?;
                match row.values.get(pos)? {
                    joule_db_query::ast::Value::String(s) => Some(s.clone()),
                    _ => None,
                }
            };

            let name = match get_str("__index_name__") {
                Some(n) => n,
                None => continue,
            };
            let table = match get_str("__index_table_ref__") {
                Some(t) => t,
                None => continue,
            };
            let column_str = get_str("__index_columns__").unwrap_or_default();
            let column = column_str
                .trim_matches(|c: char| c == '[' || c == ']' || c == '"')
                .to_string();
            let method = get_str("__vector_method__").unwrap_or_else(|| "HNSW".to_string());
            let options_str = get_str("__vector_options__").unwrap_or_else(|| "{}".to_string());
            let options: HashMap<String, String> =
                serde_json::from_str(&options_str).unwrap_or_default();

            let metric = parse_metric(&options, DistanceMetric::Euclidean);

            // Use scan_with_record_ids so the index stores real amorphic record IDs
            // (consistent with INSERT updates and query-time pre-filtering).
            let table_rows_with_ids = amorphic.scan_with_record_ids(&table).unwrap_or_default();
            let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();

            for (record_id, trow) in &table_rows_with_ids {
                let col_pos = trow
                    .columns
                    .iter()
                    .position(|c: &String| c.eq_ignore_ascii_case(&column));
                if let Some(pos) = col_pos {
                    if let Some(joule_db_query::ast::Value::String(s)) = trow.values.get(pos) {
                        if let Some(v64) = joule_db_query::vector::parse_vector(s.as_str()) {
                            let v32: Vec<f32> = v64.iter().map(|x| *x as f32).collect();
                            vectors.push((record_id.clone(), v32));
                        }
                    }
                }
            }

            if let Err(e) =
                self.build_index(&name, &table, &column, &method, metric, &options, vectors)
            {
                eprintln!("Warning: failed to rebuild vector index '{}': {}", name, e);
                continue;
            }
            count += 1;
        }

        Ok(count)
    }
}

/// Convert HDC DistanceMetric to core DistanceMetric
fn hdc_to_core_metric(m: DistanceMetric) -> CoreDistanceMetric {
    match m {
        DistanceMetric::Euclidean => CoreDistanceMetric::Euclidean,
        DistanceMetric::Cosine => CoreDistanceMetric::Cosine,
        DistanceMetric::InnerProduct => CoreDistanceMetric::InnerProduct,
        DistanceMetric::Hamming => CoreDistanceMetric::Euclidean, // Hamming maps to L2 on binary vectors
    }
}

/// Parse a metric string from SQL options to DistanceMetric
pub fn parse_metric(options: &HashMap<String, String>, default: DistanceMetric) -> DistanceMetric {
    match options.get("metric").map(|s| s.to_uppercase()).as_deref() {
        Some("COSINE") => DistanceMetric::Cosine,
        Some("INNER_PRODUCT") | Some("DOT") | Some("IP") => DistanceMetric::InnerProduct,
        Some("EUCLIDEAN") | Some("L2") => DistanceMetric::Euclidean,
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_manager_new() {
        let mgr = VectorIndexManager::new();
        assert!(mgr.find_index_for("test", "vec").is_none());
    }

    #[test]
    fn test_build_hnsw_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.0, 1.0, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
        ];
        mgr.build_index(
            "idx1",
            "test_table",
            "embedding",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();
        assert_eq!(
            mgr.find_index_for("test_table", "embedding"),
            Some("idx1".to_string())
        );
    }

    #[test]
    fn test_knn_search_hnsw() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
            ("3".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        mgr.build_index(
            "idx1",
            "t",
            "v",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();

        let results = mgr.knn_search("idx1", &[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "0");
        assert_eq!(results[1].0, "1");
    }

    #[test]
    fn test_insert_into_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![("0".to_string(), vec![1.0f32, 0.0, 0.0])];
        mgr.build_index(
            "idx1",
            "t",
            "v",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();

        mgr.insert_into_index("idx1", "1".to_string(), vec![0.5, 0.5, 0.0])
            .unwrap();

        let results = mgr.knn_search("idx1", &[0.5, 0.5, 0.0], 1).unwrap();
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn test_build_lsh_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.0, 1.0, 0.0]),
        ];
        let mut opts = HashMap::new();
        opts.insert("num_tables".to_string(), "4".to_string());
        opts.insert("num_bits".to_string(), "8".to_string());
        mgr.build_index(
            "lsh1",
            "t",
            "v",
            "LSH",
            DistanceMetric::Euclidean,
            &opts,
            rows,
        )
        .unwrap();
        assert_eq!(mgr.find_index_for("t", "v"), Some("lsh1".to_string()));
    }

    #[test]
    fn test_drop_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![("0".to_string(), vec![1.0f32, 0.0, 0.0])];
        mgr.build_index(
            "idx1",
            "t",
            "v",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();
        assert!(mgr.find_index_for("t", "v").is_some());
        mgr.drop_index("idx1").unwrap();
        assert!(mgr.find_index_for("t", "v").is_none());
    }

    #[test]
    fn test_indexes_for_table() {
        let mgr = VectorIndexManager::new();
        mgr.build_index(
            "idx1",
            "t",
            "v1",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            vec![],
        )
        .unwrap();
        mgr.build_index(
            "idx2",
            "t",
            "v2",
            "HNSW",
            DistanceMetric::Cosine,
            &HashMap::new(),
            vec![],
        )
        .unwrap();
        mgr.build_index(
            "idx3",
            "other",
            "v1",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            vec![],
        )
        .unwrap();
        let idxs = mgr.indexes_for_table("t");
        assert_eq!(idxs.len(), 2);
    }

    #[test]
    fn test_parse_metric() {
        let mut opts = HashMap::new();
        assert_eq!(
            parse_metric(&opts, DistanceMetric::Euclidean),
            DistanceMetric::Euclidean
        );
        opts.insert("metric".to_string(), "cosine".to_string());
        assert_eq!(
            parse_metric(&opts, DistanceMetric::Euclidean),
            DistanceMetric::Cosine
        );
        opts.insert("metric".to_string(), "inner_product".to_string());
        assert_eq!(
            parse_metric(&opts, DistanceMetric::Euclidean),
            DistanceMetric::InnerProduct
        );
    }

    // ---- IVF Index Tests ----

    #[test]
    fn test_build_ivf_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
            ("3".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        let mut opts = HashMap::new();
        opts.insert("n_clusters".to_string(), "2".to_string());
        opts.insert("n_probe".to_string(), "2".to_string());
        mgr.build_index(
            "ivf1",
            "t",
            "v",
            "IVF",
            DistanceMetric::Euclidean,
            &opts,
            rows,
        )
        .unwrap();
        assert_eq!(mgr.find_index_for("t", "v"), Some("ivf1".to_string()));
    }

    #[test]
    fn test_knn_search_ivf() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
            ("3".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        let mut opts = HashMap::new();
        opts.insert("n_clusters".to_string(), "2".to_string());
        opts.insert("n_probe".to_string(), "2".to_string());
        mgr.build_index(
            "ivf1",
            "t",
            "v",
            "IVF",
            DistanceMetric::Euclidean,
            &opts,
            rows,
        )
        .unwrap();

        let results = mgr.knn_search("ivf1", &[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "0");
    }

    #[test]
    fn test_insert_into_ivf_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![("0".to_string(), vec![1.0f32, 0.0, 0.0])];
        let mut opts = HashMap::new();
        opts.insert("n_clusters".to_string(), "1".to_string());
        opts.insert("n_probe".to_string(), "1".to_string());
        mgr.build_index(
            "ivf1",
            "t",
            "v",
            "IVF",
            DistanceMetric::Euclidean,
            &opts,
            rows,
        )
        .unwrap();
        mgr.insert_into_index("ivf1", "1".to_string(), vec![0.5, 0.5, 0.0])
            .unwrap();

        let results = mgr.knn_search("ivf1", &[0.5, 0.5, 0.0], 1).unwrap();
        assert_eq!(results[0].0, "1");
    }

    // ---- HNSW+SQ Tests ----

    #[test]
    fn test_build_hnsw_sq_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.0, 1.0, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
        ];
        mgr.build_index(
            "sq1",
            "t",
            "v",
            "HNSW_SQ",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();
        assert_eq!(mgr.find_index_for("t", "v"), Some("sq1".to_string()));
    }

    #[test]
    fn test_knn_search_hnsw_sq() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0]),
            ("2".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        mgr.build_index(
            "sq1",
            "t",
            "v",
            "HNSW_SQ",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();

        let results = mgr.knn_search("sq1", &[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "0");
    }

    // ---- IVF+PQ Tests ----

    #[test]
    fn test_build_ivf_pq_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0, 0.0]),
            ("1".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0, 0.0]),
            ("3".to_string(), vec![0.0, 0.0, 0.0, 1.0]),
        ];
        let mut opts = HashMap::new();
        opts.insert("n_clusters".to_string(), "2".to_string());
        opts.insert("n_subquantizers".to_string(), "2".to_string());
        mgr.build_index(
            "pq1",
            "t",
            "v",
            "IVF_PQ",
            DistanceMetric::Euclidean,
            &opts,
            rows,
        )
        .unwrap();
        assert_eq!(mgr.find_index_for("t", "v"), Some("pq1".to_string()));
    }

    #[test]
    fn test_knn_search_ivf_pq() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0, 0.0]),
            ("2".to_string(), vec![-1.0, 0.0, 0.0, 0.0]),
        ];
        let mut opts = HashMap::new();
        opts.insert("n_clusters".to_string(), "2".to_string());
        opts.insert("n_probe".to_string(), "2".to_string());
        opts.insert("n_subquantizers".to_string(), "2".to_string());
        mgr.build_index(
            "pq1",
            "t",
            "v",
            "IVF_PQ",
            DistanceMetric::Euclidean,
            &opts,
            rows,
        )
        .unwrap();

        let results = mgr.knn_search("pq1", &[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "0");
    }

    // ---- HNSW Core (joule-db-core) Tests ----

    #[test]
    fn test_build_hnsw_core_index() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.0, 1.0, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
        ];
        mgr.build_index(
            "core1",
            "t",
            "v",
            "HNSW_CORE",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();
        assert_eq!(mgr.find_index_for("t", "v"), Some("core1".to_string()));
    }

    #[test]
    fn test_knn_search_hnsw_core() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
            ("3".to_string(), vec![-1.0, 0.0, 0.0]),
        ];
        mgr.build_index(
            "core1",
            "t",
            "v",
            "HNSW_CORE",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();

        let results = mgr.knn_search("core1", &[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "0");
        assert_eq!(results[1].0, "1");
    }

    #[test]
    fn test_insert_into_hnsw_core() {
        let mgr = VectorIndexManager::new();
        let rows = vec![("0".to_string(), vec![1.0f32, 0.0, 0.0])];
        mgr.build_index(
            "core1",
            "t",
            "v",
            "HNSW_CORE",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();

        mgr.insert_into_index("core1", "1".to_string(), vec![0.5, 0.5, 0.0])
            .unwrap();

        let results = mgr.knn_search("core1", &[0.5, 0.5, 0.0], 1).unwrap();
        assert_eq!(results[0].0, "1");
    }

    // ---- ef_search Tests ----

    #[test]
    fn test_knn_search_with_ef() {
        let mgr = VectorIndexManager::new();
        let rows = vec![
            ("0".to_string(), vec![1.0f32, 0.0, 0.0]),
            ("1".to_string(), vec![0.9, 0.1, 0.0]),
            ("2".to_string(), vec![0.0, 0.0, 1.0]),
        ];
        mgr.build_index(
            "hnsw1",
            "t",
            "v",
            "HNSW",
            DistanceMetric::Euclidean,
            &HashMap::new(),
            rows,
        )
        .unwrap();

        // ef_search=100 (high accuracy)
        let results = mgr
            .knn_search_with_ef("hnsw1", &[1.0, 0.0, 0.0], 2, Some(100))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "0");

        // ef_search=2 (minimal)
        let results2 = mgr
            .knn_search_with_ef("hnsw1", &[1.0, 0.0, 0.0], 2, Some(2))
            .unwrap();
        assert_eq!(results2.len(), 2);
    }
}
