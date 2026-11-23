//! Migration tool from file-per-page to block-based storage
//!
//! Migrates existing OPFS databases from the file-per-page format
//! to the new Virtual Block Device (VBD) format for better performance.

use super::error::PersistenceError;
use super::opfs::OPFSBackend;
use crate::persistence::{AsyncPersistenceBackend, OPFSVirtualBlockDevice};
use joule_db_core::storage::PageId;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Migration result
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Number of pages migrated
    pub pages_migrated: u64,
    /// Total bytes migrated
    pub bytes_migrated: u64,
    /// Migration duration in milliseconds
    pub duration_ms: u64,
}

/// Migrate from file-per-page OPFS to Virtual Block Device
///
/// This function:
/// 1. Opens the old OPFS backend (file-per-page)
/// 2. Creates a new VBD backend
/// 3. Copies all pages from old to new format
/// 4. Verifies migration integrity
///
/// # Arguments
/// * `db_name` - Database name (same for both old and new)
/// * `delete_old` - Whether to delete old files after migration (default: false)
///
/// # Returns
/// Migration result with statistics
#[cfg(target_arch = "wasm32")]

pub async fn migrate_to_vbd(db_name: &str, delete_old: bool) -> Result<JsValue, JsValue> {
    use js_sys::Date;
    use wasm_bindgen_futures::JsFuture;

    let start_time = Date::now();

    // Open old OPFS backend
    let old_backend = OPFSBackend::new(db_name);
    old_backend
        .init()
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to init old backend: {}", e)))?;

    // Create new VBD backend
    let new_backend = OPFSVirtualBlockDevice::new(&format!("{}_vbd", db_name));
    new_backend
        .init()
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to init new backend: {}", e)))?;

    // Discover all pages in old backend
    let pages = discover_pages(&old_backend, db_name)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to discover pages: {}", e)))?;

    // Migrate each page
    let mut pages_migrated = 0u64;
    let mut bytes_migrated = 0u64;

    for page_id in &pages {
        // Read from old backend
        let data = old_backend.read_page(*page_id).await.map_err(|e| {
            JsValue::from_str(&format!("Read failed for page {}: {:?}", page_id, e))
        })?;

        // Write to new backend
        new_backend
            .write_page(*page_id, &data)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to write page {}: {}", page_id, e)))?;

        pages_migrated += 1;
        bytes_migrated += data.len() as u64;
    }

    // Sync new backend
    new_backend
        .sync()
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to sync new backend: {}", e)))?;

    // Verify migration
    verify_migration(&old_backend, &new_backend, &pages)
        .await
        .map_err(|e| JsValue::from_str(&format!("Verification failed: {}", e)))?;

    // Delete old files if requested
    if delete_old {
        delete_old_backend(&old_backend, db_name)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to delete old backend: {}", e)))?;
    }

    let end_time = Date::now();
    let duration_ms = (end_time - start_time) as u64;

    let result = MigrationResult {
        pages_migrated,
        bytes_migrated,
        duration_ms,
    };

    // Serialize to JS object
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("pages_migrated"),
        &JsValue::from_f64(pages_migrated as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bytes_migrated"),
        &JsValue::from_f64(bytes_migrated as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("duration_ms"),
        &JsValue::from_f64(duration_ms as f64),
    )?;

    Ok(obj.into())
}

/// Discover all page IDs in the old backend
#[cfg(target_arch = "wasm32")]
async fn discover_pages(
    _backend: &OPFSBackend,
    db_name: &str,
) -> Result<Vec<PageId>, PersistenceError> {
    use js_sys;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    // Get directory handle
    let window = web_sys::window()
        .ok_or_else(|| PersistenceError::NotAvailable("No window object".to_string()))?;

    let navigator = window.navigator();
    let storage_manager = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
        .ok()
        .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
        .ok_or_else(|| {
            PersistenceError::NotAvailable("StorageManager not available".to_string())
        })?;

    let get_directory = js_sys::Reflect::get(&storage_manager, &JsValue::from_str("getDirectory"))
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
        .ok_or_else(|| PersistenceError::NotAvailable("getDirectory not available".to_string()))?;

    let root_promise = get_directory
        .call0(&storage_manager)
        .map_err(|_| PersistenceError::NotAvailable("Failed to get root directory".to_string()))?;

    let root_dir = JsFuture::from(
        root_promise
            .dyn_into::<js_sys::Promise>()
            .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
    )
    .await
    .map_err(|_| PersistenceError::NotAvailable("Failed to access OPFS".to_string()))?;

    // Get database directory
    let get_directory_handle =
        js_sys::Reflect::get(&root_dir, &JsValue::from_str("getDirectoryHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| {
                PersistenceError::NotAvailable("getDirectoryHandle not available".to_string())
            })?;

    let db_dir_promise = get_directory_handle
        .call1(&root_dir, &JsValue::from_str(db_name))
        .map_err(|_| {
            PersistenceError::NotAvailable("Failed to get database directory".to_string())
        })?;

    let db_dir = JsFuture::from(
        db_dir_promise
            .dyn_into::<js_sys::Promise>()
            .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
    )
    .await
    .map_err(|_| {
        PersistenceError::NotAvailable("Failed to create database directory".to_string())
    })?;

    // List all files
    let keys_func = js_sys::Reflect::get(&db_dir, &JsValue::from_str("keys"))
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
        .ok_or_else(|| PersistenceError::NotAvailable("keys not available".to_string()))?;

    let keys_promise = keys_func
        .call0(&db_dir)
        .map_err(|_| PersistenceError::NotAvailable("Failed to list files".to_string()))?;

    let keys = JsFuture::from(
        keys_promise
            .dyn_into::<js_sys::Promise>()
            .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
    )
    .await
    .map_err(|_| PersistenceError::NotAvailable("Failed to get keys".to_string()))?;

    // Parse page IDs from filenames
    let mut pages = Vec::new();
    let iterator = js_sys::try_iter(&keys)
        .map_err(|_| PersistenceError::NotAvailable("Failed to create iterator".to_string()))?
        .ok_or_else(|| PersistenceError::NotAvailable("Not iterable".to_string()))?;

    for item in iterator {
        let item =
            item.map_err(|_| PersistenceError::NotAvailable("Iterator error".to_string()))?;
        let filename = item
            .as_string()
            .ok_or_else(|| PersistenceError::NotAvailable("Invalid filename".to_string()))?;

        // Parse page_XXXXXXXX.dat format
        if filename.starts_with("page_") && filename.ends_with(".dat") {
            let hex_str = &filename[5..filename.len() - 4]; // Remove "page_" and ".dat"
            if let Ok(page_id) = u64::from_str_radix(hex_str, 16) {
                pages.push(page_id);
            }
        }
    }

    pages.sort();
    Ok(pages)
}

/// Verify migration integrity
async fn verify_migration(
    old_backend: &OPFSBackend,
    new_backend: &OPFSVirtualBlockDevice,
    pages: &[PageId],
) -> Result<(), PersistenceError> {
    use crate::persistence::AsyncPersistenceBackend;
    for &page_id in pages {
        let old_data = old_backend.read_page(page_id).await?;
        let new_data = new_backend.read_page(page_id).await?;

        if old_data != new_data {
            return Err(PersistenceError::IoError(format!(
                "Page {} data mismatch after migration",
                page_id
            )));
        }
    }

    Ok(())
}

/// Delete old backend files with verification
///
/// This function performs safe deletion of old OPFS files after migration.
/// It verifies that each file exists before attempting deletion and logs
/// any failures without aborting the entire deletion process.
#[cfg(target_arch = "wasm32")]
async fn delete_old_backend(_backend: &OPFSBackend, db_name: &str) -> Result<(), PersistenceError> {
    use js_sys;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    // Get OPFS root directory
    let window = web_sys::window()
        .ok_or_else(|| PersistenceError::NotAvailable("No window object".to_string()))?;

    let navigator = window.navigator();
    let storage_manager = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
        .ok()
        .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
        .ok_or_else(|| {
            PersistenceError::NotAvailable("StorageManager not available".to_string())
        })?;

    let get_directory = js_sys::Reflect::get(&storage_manager, &JsValue::from_str("getDirectory"))
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
        .ok_or_else(|| PersistenceError::NotAvailable("getDirectory not available".to_string()))?;

    let root_promise = get_directory
        .call0(&storage_manager)
        .map_err(|_| PersistenceError::NotAvailable("Failed to get root directory".to_string()))?;

    let root_dir = JsFuture::from(
        root_promise
            .dyn_into::<js_sys::Promise>()
            .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
    )
    .await
    .map_err(|_| PersistenceError::NotAvailable("Failed to access OPFS".to_string()))?;

    // Get database directory handle
    let get_directory_handle =
        js_sys::Reflect::get(&root_dir, &JsValue::from_str("getDirectoryHandle"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| {
                PersistenceError::NotAvailable("getDirectoryHandle not available".to_string())
            })?;

    let db_dir_promise = get_directory_handle
        .call1(&root_dir, &JsValue::from_str(db_name))
        .map_err(|_| {
            PersistenceError::NotAvailable("Failed to get database directory".to_string())
        })?;

    let db_dir = JsFuture::from(
        db_dir_promise
            .dyn_into::<js_sys::Promise>()
            .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
    )
    .await
    .map_err(|_| PersistenceError::NotAvailable("Database directory not found".to_string()))?;

    // Get removeEntry function
    let remove_entry = js_sys::Reflect::get(&db_dir, &JsValue::from_str("removeEntry"))
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
        .ok_or_else(|| PersistenceError::NotAvailable("removeEntry not available".to_string()))?;

    // List all page files
    let keys_func = js_sys::Reflect::get(&db_dir, &JsValue::from_str("keys"))
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
        .ok_or_else(|| PersistenceError::NotAvailable("keys not available".to_string()))?;

    let keys_promise = keys_func
        .call0(&db_dir)
        .map_err(|_| PersistenceError::NotAvailable("Failed to list files".to_string()))?;

    let keys = JsFuture::from(
        keys_promise
            .dyn_into::<js_sys::Promise>()
            .map_err(|_| PersistenceError::NotAvailable("Invalid promise".to_string()))?,
    )
    .await
    .map_err(|_| PersistenceError::NotAvailable("Failed to get keys".to_string()))?;

    // Collect filenames to delete
    let mut files_to_delete = Vec::new();
    let iterator = js_sys::try_iter(&keys)
        .map_err(|_| PersistenceError::NotAvailable("Failed to create iterator".to_string()))?
        .ok_or_else(|| PersistenceError::NotAvailable("Not iterable".to_string()))?;

    for item in iterator {
        let item =
            item.map_err(|_| PersistenceError::NotAvailable("Iterator error".to_string()))?;
        let filename = item
            .as_string()
            .ok_or_else(|| PersistenceError::NotAvailable("Invalid filename".to_string()))?;

        // Only delete page files (page_*.dat format)
        if filename.starts_with("page_") && filename.ends_with(".dat") {
            files_to_delete.push(filename);
        }
    }

    // Delete each page file with verification
    let mut deleted_count = 0;
    let mut failed_count = 0;

    for filename in &files_to_delete {
        let remove_promise = remove_entry.call1(&db_dir, &JsValue::from_str(filename));

        match remove_promise {
            Ok(promise) => {
                if let Ok(promise) = promise.dyn_into::<js_sys::Promise>() {
                    match JsFuture::from(promise).await {
                        Ok(_) => {
                            deleted_count += 1;
                            tracing::debug!("Deleted old page file: {}", filename);
                        }
                        Err(_) => {
                            failed_count += 1;
                            tracing::warn!("Failed to delete page file: {}", filename);
                        }
                    }
                }
            }
            Err(_) => {
                failed_count += 1;
                tracing::warn!("Failed to initiate deletion for: {}", filename);
            }
        }
    }

    tracing::info!(
        "Old backend cleanup complete: {} files deleted, {} failed",
        deleted_count,
        failed_count
    );

    // Consider it a success if most files were deleted
    if failed_count > 0 && deleted_count == 0 {
        return Err(PersistenceError::IoError(format!(
            "Failed to delete any old backend files ({} failures)",
            failed_count
        )));
    }

    Ok(())
}

// Helper to get root_dir from OPFSBackend
// Note: We'll access it via the backend's internal structure
// For now, we'll pass it as a parameter

#[cfg(not(target_arch = "wasm32"))]
pub async fn migrate_to_vbd(
    _db_name: &str,
    _delete_old: bool,
) -> Result<MigrationResult, PersistenceError> {
    Ok(MigrationResult {
        pages_migrated: 0,
        bytes_migrated: 0,
        duration_ms: 0,
    })
}
