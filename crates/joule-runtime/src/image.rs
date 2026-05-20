//! OCI Image Management — pull, cache, list, remove, inspect.
//!
//! Manages container images used by the `ContainerBackend`. Images are
//! cached locally under `~/.jouledb/images/` with content-addressable layers.
//!
//! Currently delegates to external tools (`docker`, `podman`, or `invisible-vm`)
//! for the actual OCI registry protocol. A future iteration will use
//! `oci-distribution` for native Rust registry access.

use crate::RuntimeError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;

/// Information about a cached container image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    /// Full image reference (e.g. `"nginx:latest"`, `"ghcr.io/user/app:v1"`).
    pub reference: String,
    /// Image ID (content-addressed digest).
    pub id: String,
    /// Uncompressed image size in bytes.
    pub size_bytes: u64,
    /// When the image was pulled.
    pub pulled_at: chrono::DateTime<chrono::Utc>,
    /// Exposed ports from the image manifest.
    pub exposed_ports: Vec<u16>,
    /// Default entrypoint from the image.
    pub entrypoint: Option<Vec<String>>,
    /// Default command from the image.
    pub cmd: Option<Vec<String>>,
    /// Environment variables baked into the image.
    pub env: Vec<String>,
    /// Working directory from the image.
    pub working_dir: Option<String>,
}

/// Registry authentication credentials.
#[derive(Debug, Clone)]
pub struct RegistryAuth {
    pub username: String,
    pub password: String,
}

/// Manages locally cached container images.
pub struct ImageStore {
    cache_dir: PathBuf,
    images: RwLock<HashMap<String, ImageInfo>>,
    registry_auth: RwLock<HashMap<String, RegistryAuth>>,
}

impl ImageStore {
    /// Create a new image store with the given cache directory.
    pub fn new(cache_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&cache_dir).ok();

        // Load existing image catalog
        let catalog_path = cache_dir.join("catalog.json");
        let images = if catalog_path.exists() {
            std::fs::read_to_string(&catalog_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        Self {
            cache_dir,
            images: RwLock::new(images),
            registry_auth: RwLock::new(HashMap::new()),
        }
    }

    /// Pull an image from a registry.
    ///
    /// Tries `docker pull`, `podman pull`, or `invisible-vm image pull` in order.
    /// Returns the image info on success.
    pub async fn pull(&self, image_ref: &str) -> Result<ImageInfo, RuntimeError> {
        log::info!("Pulling image: {}", image_ref);

        // Try docker first, then podman
        let tool = find_container_tool()?;

        let output = tokio::process::Command::new(&tool)
            .args(["pull", image_ref])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| RuntimeError::ProcessError(format!("failed to run {}: {}", tool, e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RuntimeError::ProcessError(format!(
                "failed to pull {}: {}",
                image_ref, stderr
            )));
        }

        // Inspect the pulled image to get metadata
        let info = self.inspect_with_tool(&tool, image_ref).await?;

        // Cache the image info
        let mut images = self.images.write().await;
        images.insert(image_ref.to_string(), info.clone());
        drop(images);
        self.persist().await;

        log::info!("Image pulled: {} ({})", image_ref, info.id);
        Ok(info)
    }

    /// List all cached images.
    pub async fn list(&self) -> Vec<ImageInfo> {
        self.images.read().await.values().cloned().collect()
    }

    /// Check if an image is already cached.
    pub async fn has(&self, image_ref: &str) -> bool {
        self.images.read().await.contains_key(image_ref)
    }

    /// Get info about a specific image.
    pub async fn get(&self, image_ref: &str) -> Option<ImageInfo> {
        self.images.read().await.get(image_ref).cloned()
    }

    /// Remove a cached image.
    pub async fn remove(&self, image_ref: &str) -> Result<(), RuntimeError> {
        let tool = find_container_tool()?;

        let output = tokio::process::Command::new(&tool)
            .args(["rmi", image_ref])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| RuntimeError::ProcessError(format!("failed to run {}: {}", tool, e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RuntimeError::ProcessError(format!(
                "failed to remove image {}: {}",
                image_ref, stderr
            )));
        }

        let mut images = self.images.write().await;
        images.remove(image_ref);
        drop(images);
        self.persist().await;

        Ok(())
    }

    /// Inspect an image and return its metadata.
    pub async fn inspect(&self, image_ref: &str) -> Result<ImageInfo, RuntimeError> {
        // Check cache first
        if let Some(info) = self.get(image_ref).await {
            return Ok(info);
        }

        let tool = find_container_tool()?;
        self.inspect_with_tool(&tool, image_ref).await
    }

    /// Add registry credentials for a specific registry host.
    pub async fn add_auth(&self, registry: String, auth: RegistryAuth) {
        self.registry_auth.write().await.insert(registry, auth);
    }

    /// Get the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Inspect an image using a specific container tool.
    async fn inspect_with_tool(
        &self,
        tool: &str,
        image_ref: &str,
    ) -> Result<ImageInfo, RuntimeError> {
        let output = tokio::process::Command::new(tool)
            .args(["inspect", "--format", "{{json .}}", image_ref])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                RuntimeError::ProcessError(format!("failed to inspect {}: {}", image_ref, e))
            })?;

        if !output.status.success() {
            // Image may not exist locally — return a minimal info
            return Ok(ImageInfo {
                reference: image_ref.to_string(),
                id: String::new(),
                size_bytes: 0,
                pulled_at: chrono::Utc::now(),
                exposed_ports: vec![],
                entrypoint: None,
                cmd: None,
                env: vec![],
                working_dir: None,
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse the JSON output — docker inspect returns an array
        let json_str = stdout.trim();
        let json_str = if json_str.starts_with('[') {
            // Docker returns an array, take the first element
            &json_str[1..json_str.len().saturating_sub(1)]
        } else {
            json_str
        };

        parse_inspect_json(json_str, image_ref)
    }

    /// Persist image catalog to disk.
    async fn persist(&self) {
        let catalog_path = self.cache_dir.join("catalog.json");
        let images = self.images.read().await;
        let json = serde_json::to_string_pretty(&*images).unwrap_or_default();
        drop(images);

        let tmp = catalog_path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp, &json) {
            log::error!("Failed to write image catalog: {}", e);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &catalog_path) {
            log::error!("Failed to rename image catalog: {}", e);
        }
    }
}

/// Find an available container tool on PATH.
fn find_container_tool() -> Result<String, RuntimeError> {
    for tool in ["docker", "podman", "nerdctl"] {
        if crate::native::which_exists(tool) {
            return Ok(tool.to_string());
        }
    }
    Err(RuntimeError::ProcessError(
        "no container tool found. Install docker, podman, or nerdctl.".into(),
    ))
}

/// Parse `docker inspect` / `podman inspect` JSON output into ImageInfo.
fn parse_inspect_json(json_str: &str, image_ref: &str) -> Result<ImageInfo, RuntimeError> {
    let val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| RuntimeError::ProcessError(format!("failed to parse inspect JSON: {}", e)))?;

    let id = val["Id"]
        .as_str()
        .or_else(|| val["ID"].as_str())
        .unwrap_or("")
        .to_string();

    let size_bytes = val["Size"].as_u64().unwrap_or(0);

    let config = &val["Config"];
    let exposed_ports: Vec<u16> = config["ExposedPorts"]
        .as_object()
        .map(|ports| {
            ports
                .keys()
                .filter_map(|k| k.split('/').next()?.parse().ok())
                .collect()
        })
        .unwrap_or_default();

    let entrypoint = config["Entrypoint"].as_array().map(|a| {
        a.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    });

    let cmd = config["Cmd"].as_array().map(|a| {
        a.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    });

    let env = config["Env"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let working_dir = config["WorkingDir"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(String::from);

    Ok(ImageInfo {
        reference: image_ref.to_string(),
        id,
        size_bytes,
        pulled_at: chrono::Utc::now(),
        exposed_ports,
        entrypoint,
        cmd,
        env,
        working_dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_info_serde() {
        let info = ImageInfo {
            reference: "nginx:latest".into(),
            id: "sha256:abc123".into(),
            size_bytes: 142_000_000,
            pulled_at: chrono::Utc::now(),
            exposed_ports: vec![80, 443],
            entrypoint: Some(vec!["/docker-entrypoint.sh".into()]),
            cmd: Some(vec!["nginx".into(), "-g".into(), "daemon off;".into()]),
            env: vec!["PATH=/usr/local/sbin:/usr/local/bin".into()],
            working_dir: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: ImageInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.reference, "nginx:latest");
        assert_eq!(parsed.exposed_ports, vec![80, 443]);
        assert_eq!(parsed.size_bytes, 142_000_000);
    }

    #[test]
    fn test_image_store_new() {
        let dir = tempfile::tempdir().unwrap();
        let store = ImageStore::new(dir.path().join("images"));
        assert!(store.cache_dir().exists());
    }

    #[tokio::test]
    async fn test_image_store_has_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = ImageStore::new(dir.path().join("images"));
        assert!(!store.has("nginx:latest").await);
    }

    #[tokio::test]
    async fn test_image_store_list_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = ImageStore::new(dir.path().join("images"));
        assert!(store.list().await.is_empty());
    }

    #[test]
    fn test_parse_inspect_json_basic() {
        let json = r#"{
            "Id": "sha256:abc123",
            "Size": 100000,
            "Config": {
                "ExposedPorts": {"80/tcp": {}, "443/tcp": {}},
                "Entrypoint": ["/entrypoint.sh"],
                "Cmd": ["nginx"],
                "Env": ["PATH=/usr/bin"],
                "WorkingDir": "/app"
            }
        }"#;
        let info = parse_inspect_json(json, "nginx:latest").unwrap();
        assert_eq!(info.reference, "nginx:latest");
        assert_eq!(info.id, "sha256:abc123");
        assert_eq!(info.size_bytes, 100000);
        assert!(info.exposed_ports.contains(&80));
        assert!(info.exposed_ports.contains(&443));
        assert_eq!(info.entrypoint, Some(vec!["/entrypoint.sh".into()]));
        assert_eq!(info.cmd, Some(vec!["nginx".into()]));
        assert_eq!(info.working_dir, Some("/app".into()));
    }

    #[test]
    fn test_parse_inspect_json_minimal() {
        let json = r#"{"Id": "sha256:def456"}"#;
        let info = parse_inspect_json(json, "alpine:3").unwrap();
        assert_eq!(info.reference, "alpine:3");
        assert_eq!(info.id, "sha256:def456");
        assert!(info.exposed_ports.is_empty());
        assert!(info.entrypoint.is_none());
    }

    #[tokio::test]
    async fn test_image_store_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let store = ImageStore::new(dir.path().join("images"));

        // Manually insert an image
        {
            let mut images = store.images.write().await;
            images.insert(
                "test:latest".into(),
                ImageInfo {
                    reference: "test:latest".into(),
                    id: "sha256:test".into(),
                    size_bytes: 1000,
                    pulled_at: chrono::Utc::now(),
                    exposed_ports: vec![],
                    entrypoint: None,
                    cmd: None,
                    env: vec![],
                    working_dir: None,
                },
            );
        }
        store.persist().await;

        // Reload from disk
        let store2 = ImageStore::new(dir.path().join("images"));
        assert!(store2.has("test:latest").await);
    }
}
