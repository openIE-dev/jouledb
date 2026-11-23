//! Weight query helpers for retrieval during inference.

use crate::WeightDB;
use joule_db_core::error::Error;

/// Retrieve all tensors for a given layer.
pub fn get_layer_tensors(
    db: &WeightDB,
    model_id: &str,
    layer_idx: usize,
) -> Result<Vec<(String, Vec<u8>)>, Error> {
    // Get the model index to find tensor names for this layer
    let index_json = db.get_model_index(model_id)?.ok_or_else(|| {
        Error::Storage(joule_db_core::error::StorageError::Io(format!(
            "model index not found for '{}'",
            model_id
        )))
    })?;

    // Parse tensor names from index
    // Simple parsing — extract tensor_names array from JSON
    let prefix = format!("blk.{}.", layer_idx);
    let mut results = Vec::new();

    // Quick parse: find tensor names containing the layer prefix
    if let Some(names_start) = index_json.find("\"tensor_names\":[") {
        let after = &index_json[names_start + 16..];
        if let Some(end) = after.find(']') {
            let names_str = &after[..end];
            for name in names_str.split(',') {
                let name = name.trim().trim_matches('"');
                if name.starts_with(&prefix) {
                    if let Some(data) = db.get_tensor(model_id, name)? {
                        results.push((name.to_string(), data));
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Get a single tensor by name. Convenience wrapper.
pub fn get_tensor(
    db: &WeightDB,
    model_id: &str,
    tensor_name: &str,
) -> Result<Option<Vec<u8>>, Error> {
    db.get_tensor(model_id, tensor_name)
}
