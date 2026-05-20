//! Status command

use crate::{Config, Result, config::Credentials, output::Output};

pub async fn execute(config: &Config, output: &Output) -> Result<()> {
    output.section("JouleDB CLI Status");

    // Local connection status
    output.info("Local Server:");
    let url = format!(
        "http://{}:{}/health",
        config.connection.host, config.connection.port
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let local_status = match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => "Connected".to_string(),
        Ok(response) => {
            format!("Unhealthy ({})", response.status())
        }
        Err(_) => "Not available".to_string(),
    };

    output.key_value(vec![
        ("Host", config.connection.host.clone()),
        ("Port", config.connection.port.to_string()),
        ("Status", local_status),
        (
            "Database",
            config
                .connection
                .database
                .clone()
                .unwrap_or_else(|| "(default)".to_string()),
        ),
    ]);

    output.raw("");

    // Cloud status
    output.info("JouleDB Cloud:");
    let creds = Credentials::load().unwrap_or_default();

    if creds.is_authenticated() {
        output.key_value(vec![
            ("Status", "Authenticated".to_string()),
            ("Email", creds.email.unwrap_or_else(|| "-".to_string())),
            (
                "Project",
                config
                    .cloud
                    .project_id
                    .clone()
                    .unwrap_or_else(|| "(none)".to_string()),
            ),
            (
                "Cluster",
                config
                    .cloud
                    .cluster_id
                    .clone()
                    .unwrap_or_else(|| "(none)".to_string()),
            ),
        ]);
    } else {
        output.key_value(vec![("Status", "Not authenticated".to_string())]);
        output.info("  Run 'jouledb cloud login' to authenticate");
    }

    output.raw("");

    // Configuration
    output.info("Configuration:");
    let config_path = Config::default_config_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(unknown)".to_string());

    output.key_value(vec![
        ("Config file", config_path),
        ("Output format", config.output.format.clone()),
    ]);

    Ok(())
}
