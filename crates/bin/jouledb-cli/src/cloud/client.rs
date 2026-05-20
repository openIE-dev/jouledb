//! JouleDB Cloud API Client

use crate::{Config, Result, config::Credentials, error::CliError};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Serialize, de::DeserializeOwned};

/// Cloud API Client
pub struct CloudClient {
    client: reqwest::Client,
    base_url: String,
}

impl CloudClient {
    /// Create new cloud client
    pub fn new(config: &Config) -> Result<Self> {
        let creds = Credentials::load()?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(token) = &creds.access_token {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", token))
                    .map_err(|_| CliError::Auth("Invalid token".into()))?,
            );
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            base_url: config.cloud.api_url.clone(),
        })
    }

    /// Check if authenticated
    pub fn is_authenticated() -> Result<bool> {
        let creds = Credentials::load()?;
        Ok(creds.is_authenticated())
    }

    /// Make GET request
    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let data = response.json().await?;
            Ok(data)
        } else {
            let status = response.status();
            let message = response.text().await.unwrap_or_default();
            Err(CliError::CloudApi {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Make POST request
    pub async fn post<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: &B) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);

        let response = self.client.post(&url).json(body).send().await?;

        if response.status().is_success() {
            let data = response.json().await?;
            Ok(data)
        } else {
            let status = response.status();
            let message = response.text().await.unwrap_or_default();
            Err(CliError::CloudApi {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Make PATCH request
    pub async fn patch<T: DeserializeOwned, B: Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);

        let response = self.client.patch(&url).json(body).send().await?;

        if response.status().is_success() {
            let data = response.json().await?;
            Ok(data)
        } else {
            let status = response.status();
            let message = response.text().await.unwrap_or_default();
            Err(CliError::CloudApi {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Make DELETE request
    pub async fn delete(&self, path: &str) -> Result<()> {
        let url = format!("{}{}", self.base_url, path);

        let response = self.client.delete(&url).send().await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let message = response.text().await.unwrap_or_default();
            Err(CliError::CloudApi {
                status: status.as_u16(),
                message,
            })
        }
    }

    // API Methods

    /// List projects
    pub async fn list_projects(&self) -> Result<Vec<Project>> {
        self.get("/v1/projects").await
    }

    /// Create project
    pub async fn create_project(&self, name: &str, org: Option<&str>) -> Result<Project> {
        let body = serde_json::json!({
            "name": name,
            "organization": org,
        });
        self.post("/v1/projects", &body).await
    }

    /// Delete project
    pub async fn delete_project(&self, id: &str) -> Result<()> {
        self.delete(&format!("/v1/projects/{}", id)).await
    }

    /// List clusters
    pub async fn list_clusters(&self, project_id: &str) -> Result<Vec<Cluster>> {
        self.get(&format!("/v1/projects/{}/clusters", project_id))
            .await
    }

    /// Create cluster
    pub async fn create_cluster(
        &self,
        project_id: &str,
        name: &str,
        tier: &str,
        region: &str,
    ) -> Result<Cluster> {
        let body = serde_json::json!({
            "name": name,
            "tier": tier,
            "region": region,
        });
        self.post(&format!("/v1/projects/{}/clusters", project_id), &body)
            .await
    }

    /// Get cluster
    pub async fn get_cluster(&self, cluster_id: &str) -> Result<Cluster> {
        self.get(&format!("/v1/clusters/{}", cluster_id)).await
    }

    /// Delete cluster
    pub async fn delete_cluster(&self, cluster_id: &str) -> Result<()> {
        self.delete(&format!("/v1/clusters/{}", cluster_id)).await
    }

    /// Get usage
    pub async fn get_usage(&self) -> Result<Usage> {
        self.get("/v1/usage").await
    }
}

/// Project response
#[derive(Debug, serde::Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub organization: Option<String>,
    pub created_at: String,
}

/// Cluster response
#[derive(Debug, serde::Deserialize)]
pub struct Cluster {
    pub id: String,
    pub name: String,
    pub project_id: String,
    pub tier: String,
    pub region: String,
    pub status: String,
    pub connection_string: String,
    pub created_at: String,
}

/// Usage response
#[derive(Debug, serde::Deserialize)]
pub struct Usage {
    pub queries: i64,
    pub queries_limit: i64,
    pub storage_bytes: i64,
    pub storage_limit_bytes: i64,
    pub bandwidth_bytes: i64,
    pub estimated_cost: f64,
}
