//! Database-specific energy tuning.
//!
//! Generates and executes engine-specific configuration commands to reduce
//! energy consumption under thermal pressure. Each database has different
//! knobs that can be adjusted to trade throughput for lower power draw.

use crate::DatabaseEngine;

/// Severity of thermal pressure for tuning decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalSeverity {
    /// Fair/Serious — reduce non-essential background work.
    Moderate,
    /// Critical — minimize everything possible.
    Critical,
}

/// A tuning action to send to a specific database engine.
#[derive(Debug, Clone)]
pub struct TuningAction {
    /// Engine display name (for logging).
    pub engine: String,
    /// The CLI tool to use inside the container (e.g., "psql", "redis-cli").
    pub tool: String,
    /// Arguments to pass to the tool.
    pub args: Vec<String>,
    /// Human-readable description of what this tuning does.
    pub description: String,
}

/// Database-specific energy tuner.
///
/// Stateless: generates tuning commands based on engine type and severity.
pub struct DatabaseTuner;

impl DatabaseTuner {
    /// Generate tuning commands to reduce energy consumption.
    pub fn tune_for_thermal(
        engine: &DatabaseEngine,
        severity: ThermalSeverity,
    ) -> Vec<TuningAction> {
        match engine {
            DatabaseEngine::Postgres => Self::postgres_tune(severity),
            DatabaseEngine::Redis => Self::redis_tune(severity),
            DatabaseEngine::MongoDB => Self::mongodb_tune(severity),
            DatabaseEngine::MySQL => Self::mysql_tune(severity),
            _ => Vec::new(), // No known tuning for other engines
        }
    }

    /// Generate commands to restore default performance settings.
    pub fn restore_defaults(engine: &DatabaseEngine) -> Vec<TuningAction> {
        match engine {
            DatabaseEngine::Postgres => Self::postgres_restore(),
            DatabaseEngine::Redis => Self::redis_restore(),
            DatabaseEngine::MongoDB => Self::mongodb_restore(),
            DatabaseEngine::MySQL => Self::mysql_restore(),
            _ => Vec::new(),
        }
    }

    /// Execute a tuning command inside a running container.
    pub async fn execute(
        container_name: &str,
        action: &TuningAction,
    ) -> Result<(), crate::RuntimeError> {
        let tool = crate::governor::find_container_tool()?;

        let mut cmd_args = vec![
            "exec".to_string(),
            container_name.to_string(),
            action.tool.clone(),
        ];
        cmd_args.extend(action.args.iter().cloned());

        log::info!(
            "db_tuner: {} → {} {}",
            action.engine,
            action.description,
            container_name
        );

        let output = tokio::process::Command::new(&tool)
            .args(&cmd_args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                crate::RuntimeError::ProcessError(format!("db_tuner exec failed: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::warn!("db_tuner: command failed: {}", stderr.trim());
        }

        Ok(())
    }

    // --- PostgreSQL ---

    fn postgres_tune(severity: ThermalSeverity) -> Vec<TuningAction> {
        let engine = "PostgreSQL".to_string();
        let tool = "psql".to_string();

        match severity {
            ThermalSeverity::Moderate => vec![
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["-c".into(), "SET work_mem = '2MB';".into()],
                    description: "reduce work_mem (less sorting memory)".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["-c".into(), "SET maintenance_work_mem = '32MB';".into()],
                    description: "reduce maintenance_work_mem (slower VACUUM)".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "-c".into(),
                        "SET checkpoint_completion_target = 0.9;".into(),
                    ],
                    description: "spread checkpoints (less I/O burst)".into(),
                },
            ],
            ThermalSeverity::Critical => vec![
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["-c".into(), "SET work_mem = '1MB';".into()],
                    description: "minimize work_mem".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "-c".into(),
                        "SET max_parallel_workers_per_gather = 0;".into(),
                    ],
                    description: "disable parallel queries".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["-c".into(), "SET wal_writer_delay = '1s';".into()],
                    description: "batch WAL writes (less fsync)".into(),
                },
            ],
        }
    }

    fn postgres_restore() -> Vec<TuningAction> {
        let engine = "PostgreSQL".to_string();
        let tool = "psql".to_string();
        vec![
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec!["-c".into(), "SET work_mem = '4MB';".into()],
                description: "restore work_mem to default".into(),
            },
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec!["-c".into(), "SET maintenance_work_mem = '64MB';".into()],
                description: "restore maintenance_work_mem".into(),
            },
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec![
                    "-c".into(),
                    "SET max_parallel_workers_per_gather = 2;".into(),
                ],
                description: "restore parallel queries".into(),
            },
        ]
    }

    // --- Redis ---

    fn redis_tune(severity: ThermalSeverity) -> Vec<TuningAction> {
        let engine = "Redis".to_string();
        let tool = "redis-cli".to_string();

        match severity {
            ThermalSeverity::Moderate => vec![
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["CONFIG".into(), "SET".into(), "hz".into(), "5".into()],
                    description: "reduce background task frequency".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "CONFIG".into(),
                        "SET".into(),
                        "lazyfree-lazy-eviction".into(),
                        "yes".into(),
                    ],
                    description: "enable async eviction".into(),
                },
            ],
            ThermalSeverity::Critical => vec![
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["CONFIG".into(), "SET".into(), "hz".into(), "1".into()],
                    description: "minimize background tasks".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["CONFIG".into(), "SET".into(), "save".into(), "\"\"".into()],
                    description: "disable RDB snapshots".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "CONFIG".into(),
                        "SET".into(),
                        "appendfsync".into(),
                        "no".into(),
                    ],
                    description: "disable AOF fsync".into(),
                },
            ],
        }
    }

    fn redis_restore() -> Vec<TuningAction> {
        let engine = "Redis".to_string();
        let tool = "redis-cli".to_string();
        vec![
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec!["CONFIG".into(), "SET".into(), "hz".into(), "10".into()],
                description: "restore default hz".into(),
            },
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec![
                    "CONFIG".into(),
                    "SET".into(),
                    "appendfsync".into(),
                    "everysec".into(),
                ],
                description: "restore AOF fsync".into(),
            },
        ]
    }

    // --- MongoDB ---

    fn mongodb_tune(severity: ThermalSeverity) -> Vec<TuningAction> {
        let engine = "MongoDB".to_string();
        let tool = "mongosh".to_string();

        match severity {
            ThermalSeverity::Moderate => vec![TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec![
                    "--eval".into(),
                    "db.adminCommand({setParameter: 1, wiredTigerCacheSizeGB: 0.5})".into(),
                ],
                description: "reduce WiredTiger cache to 512MB".into(),
            }],
            ThermalSeverity::Critical => vec![
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "--eval".into(),
                        "db.adminCommand({setParameter: 1, wiredTigerCacheSizeGB: 0.25})".into(),
                    ],
                    description: "minimize WiredTiger cache to 256MB".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "--eval".into(),
                        "db.adminCommand({setParameter: 1, journalCommitInterval: 500})".into(),
                    ],
                    description: "batch journal commits (500ms)".into(),
                },
            ],
        }
    }

    fn mongodb_restore() -> Vec<TuningAction> {
        let engine = "MongoDB".to_string();
        let tool = "mongosh".to_string();
        vec![TuningAction {
            engine,
            tool,
            args: vec![
                "--eval".into(),
                "db.adminCommand({setParameter: 1, wiredTigerCacheSizeGB: 1.0})".into(),
            ],
            description: "restore WiredTiger cache to 1GB".into(),
        }]
    }

    // --- MySQL ---

    fn mysql_tune(severity: ThermalSeverity) -> Vec<TuningAction> {
        let engine = "MySQL".to_string();
        let tool = "mysql".to_string();

        match severity {
            ThermalSeverity::Moderate => vec![TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec![
                    "-e".into(),
                    "SET GLOBAL innodb_buffer_pool_size = 134217728;".into(), // 128MB
                ],
                description: "reduce InnoDB buffer pool to 128MB".into(),
            }],
            ThermalSeverity::Critical => vec![
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec![
                        "-e".into(),
                        "SET GLOBAL innodb_buffer_pool_size = 67108864;".into(), // 64MB
                    ],
                    description: "minimize InnoDB buffer pool to 64MB".into(),
                },
                TuningAction {
                    engine: engine.clone(),
                    tool: tool.clone(),
                    args: vec!["-e".into(), "SET GLOBAL innodb_io_capacity = 100;".into()],
                    description: "reduce I/O capacity".into(),
                },
            ],
        }
    }

    fn mysql_restore() -> Vec<TuningAction> {
        let engine = "MySQL".to_string();
        let tool = "mysql".to_string();
        vec![
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec![
                    "-e".into(),
                    "SET GLOBAL innodb_buffer_pool_size = 268435456;".into(), // 256MB
                ],
                description: "restore InnoDB buffer pool to 256MB".into(),
            },
            TuningAction {
                engine: engine.clone(),
                tool: tool.clone(),
                args: vec!["-e".into(), "SET GLOBAL innodb_io_capacity = 200;".into()],
                description: "restore I/O capacity".into(),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postgres_moderate_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::Postgres, ThermalSeverity::Moderate);
        assert_eq!(actions.len(), 3);
        assert!(actions[0].args.iter().any(|a| a.contains("work_mem")));
        assert!(
            actions[1]
                .args
                .iter()
                .any(|a| a.contains("maintenance_work_mem"))
        );
        assert!(
            actions[2]
                .args
                .iter()
                .any(|a| a.contains("checkpoint_completion_target"))
        );
    }

    #[test]
    fn test_postgres_critical_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::Postgres, ThermalSeverity::Critical);
        assert_eq!(actions.len(), 3);
        assert!(
            actions[1]
                .args
                .iter()
                .any(|a| a.contains("max_parallel_workers_per_gather"))
        );
    }

    #[test]
    fn test_redis_moderate_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::Redis, ThermalSeverity::Moderate);
        assert_eq!(actions.len(), 2);
        assert!(actions[0].args.contains(&"hz".to_string()));
        assert!(actions[0].args.contains(&"5".to_string()));
    }

    #[test]
    fn test_redis_critical_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::Redis, ThermalSeverity::Critical);
        assert_eq!(actions.len(), 3);
        // Should disable RDB saves and AOF fsync
        assert!(actions[1].description.contains("RDB"));
        assert!(actions[2].description.contains("AOF"));
    }

    #[test]
    fn test_mongodb_moderate_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::MongoDB, ThermalSeverity::Moderate);
        assert_eq!(actions.len(), 1);
        assert!(actions[0].description.contains("WiredTiger"));
    }

    #[test]
    fn test_mongodb_critical_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::MongoDB, ThermalSeverity::Critical);
        assert_eq!(actions.len(), 2);
        assert!(actions[1].description.contains("journal"));
    }

    #[test]
    fn test_mysql_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::MySQL, ThermalSeverity::Moderate);
        assert_eq!(actions.len(), 1);
        assert!(actions[0].description.contains("InnoDB"));
    }

    #[test]
    fn test_unknown_engine_no_tuning() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::SQLite, ThermalSeverity::Moderate);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_postgres_restore() {
        let actions = DatabaseTuner::restore_defaults(&DatabaseEngine::Postgres);
        assert_eq!(actions.len(), 3);
        assert!(actions[0].args.iter().any(|a| a.contains("4MB")));
    }

    #[test]
    fn test_redis_restore() {
        let actions = DatabaseTuner::restore_defaults(&DatabaseEngine::Redis);
        assert_eq!(actions.len(), 2);
        assert!(actions[0].args.contains(&"10".to_string()));
    }

    #[test]
    fn test_mongodb_restore() {
        let actions = DatabaseTuner::restore_defaults(&DatabaseEngine::MongoDB);
        assert_eq!(actions.len(), 1);
        assert!(actions[0].args.iter().any(|a| a.contains("1.0")));
    }

    #[test]
    fn test_mysql_restore() {
        let actions = DatabaseTuner::restore_defaults(&DatabaseEngine::MySQL);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_tuning_action_fields() {
        let actions =
            DatabaseTuner::tune_for_thermal(&DatabaseEngine::Postgres, ThermalSeverity::Moderate);
        for action in &actions {
            assert_eq!(action.engine, "PostgreSQL");
            assert_eq!(action.tool, "psql");
            assert!(!action.description.is_empty());
            assert!(!action.args.is_empty());
        }
    }
}
