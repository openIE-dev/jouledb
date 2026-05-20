//! Database engine catalog — built-in specs for supported database engines.
//!
//! Each `DatabaseSpec` describes how to find, launch, configure, and health-check
//! a database engine. The Joule runtime uses these specs to generically manage
//! any supported database while wrapping it with energy telemetry.

use crate::{DatabaseEngine, WorkloadKind};

/// Specification for a database engine — how to find, launch, and health-check it.
#[derive(Debug, Clone)]
pub struct DatabaseSpec {
    /// Human-readable engine name.
    pub display_name: &'static str,
    /// Binary candidates to search for (in order of preference).
    /// On Windows, `.exe` is appended automatically.
    pub binary_candidates: &'static [&'static str],
    /// CLI flag for the data directory (e.g., `--data`, `-D`).
    pub data_dir_flag: Option<&'static str>,
    /// CLI flag for the port (e.g., `--port`, `-p`).
    pub port_flag: Option<&'static str>,
    /// Default port the engine listens on.
    pub default_port: u16,
    /// Default arguments to always pass when starting the engine.
    pub default_args: &'static [&'static str],
    /// Health check command template. `{port}` is substituted with the actual port.
    /// Each element is a separate argument (first element is the binary).
    pub health_check_cmd: Option<&'static [&'static str]>,
    /// Protocol name for port mapping (e.g., "http", "postgresql", "redis").
    pub protocol_name: &'static str,
    /// Whether this engine supports WASM isolation mode.
    pub supports_wasm: bool,
    /// Whether this engine needs init before first start (e.g., `initdb` for Postgres).
    pub needs_init: bool,
    /// Init command template (if needs_init is true). `{data_dir}` is substituted.
    pub init_cmd: Option<&'static [&'static str]>,
}

/// Returns the built-in spec for a database engine.
pub fn get_spec(engine: &DatabaseEngine) -> DatabaseSpec {
    match engine {
        DatabaseEngine::JouleDB => jouledb_spec(),
        DatabaseEngine::Postgres => postgres_spec(),
        DatabaseEngine::MySQL => mysql_spec(),
        DatabaseEngine::Redis => redis_spec(),
        DatabaseEngine::MongoDB => mongodb_spec(),
        DatabaseEngine::SQLite => sqlite_spec(),
        DatabaseEngine::Custom(name) => custom_spec(name),
    }
}

/// Returns a spec for any workload kind.
///
/// - `Database` → delegates to `get_spec()` with the engine
/// - `Process` → returns a minimal spec with the user-supplied binary
/// - `Container` → returns a container-oriented spec
pub fn get_workload_spec(workload: &WorkloadKind) -> DatabaseSpec {
    match workload {
        WorkloadKind::Database { engine } => get_spec(engine),
        WorkloadKind::Process { binary, .. } => process_spec(binary),
        WorkloadKind::Container { image } => container_spec(image),
    }
}

fn process_spec(binary: &str) -> DatabaseSpec {
    let leaked: &'static str = Box::leak(binary.to_string().into_boxed_str());
    let display = std::path::Path::new(binary)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| binary.to_string());
    let display_leaked: &'static str = Box::leak(display.into_boxed_str());
    DatabaseSpec {
        display_name: display_leaked,
        binary_candidates: &[],
        data_dir_flag: None,
        port_flag: None,
        default_port: 0,
        default_args: &[],
        health_check_cmd: None,
        protocol_name: "tcp",
        supports_wasm: false,
        needs_init: false,
        init_cmd: None,
    }
}

fn container_spec(image: &str) -> DatabaseSpec {
    let leaked: &'static str = Box::leak(image.to_string().into_boxed_str());
    DatabaseSpec {
        display_name: leaked,
        binary_candidates: &[],
        data_dir_flag: None,
        port_flag: None,
        default_port: 0,
        default_args: &[],
        health_check_cmd: None,
        protocol_name: "http",
        supports_wasm: false,
        needs_init: false,
        init_cmd: None,
    }
}

fn jouledb_spec() -> DatabaseSpec {
    DatabaseSpec {
        display_name: "JouleDB",
        binary_candidates: &[
            "joule-db-server",
            "./target/release/joule-db-server",
            "./target/debug/joule-db-server",
        ],
        data_dir_flag: Some("--data"),
        port_flag: Some("--port"),
        default_port: 8080,
        default_args: &[],
        health_check_cmd: None, // JouleDB has built-in /health endpoint
        protocol_name: "http",
        supports_wasm: true,
        needs_init: false,
        init_cmd: None,
    }
}

fn postgres_spec() -> DatabaseSpec {
    DatabaseSpec {
        display_name: "PostgreSQL",
        binary_candidates: &[
            "postgres",
            "/usr/lib/postgresql/16/bin/postgres",
            "/usr/lib/postgresql/15/bin/postgres",
            "/usr/lib/postgresql/14/bin/postgres",
            "/opt/homebrew/opt/postgresql@16/bin/postgres",
            "/opt/homebrew/opt/postgresql@15/bin/postgres",
            "/usr/local/opt/postgresql@16/bin/postgres",
        ],
        data_dir_flag: Some("-D"),
        port_flag: Some("-p"),
        default_port: 5432,
        default_args: &[],
        health_check_cmd: Some(&["pg_isready", "-h", "localhost", "-p", "{port}"]),
        protocol_name: "postgresql",
        supports_wasm: false,
        needs_init: true,
        init_cmd: Some(&["initdb", "-D", "{data_dir}", "--no-locale", "-E", "UTF8"]),
    }
}

fn mysql_spec() -> DatabaseSpec {
    DatabaseSpec {
        display_name: "MySQL",
        binary_candidates: &[
            "mysqld",
            "/usr/sbin/mysqld",
            "/usr/local/mysql/bin/mysqld",
            "/opt/homebrew/opt/mysql/bin/mysqld",
        ],
        data_dir_flag: Some("--datadir"),
        port_flag: Some("--port"),
        default_port: 3306,
        default_args: &["--skip-grant-tables"],
        health_check_cmd: Some(&["mysqladmin", "ping", "-h", "localhost", "-P", "{port}"]),
        protocol_name: "mysql",
        supports_wasm: false,
        needs_init: true,
        init_cmd: Some(&["mysqld", "--initialize-insecure", "--datadir={data_dir}"]),
    }
}

fn redis_spec() -> DatabaseSpec {
    DatabaseSpec {
        display_name: "Redis",
        binary_candidates: &[
            "redis-server",
            "/opt/homebrew/opt/redis/bin/redis-server",
            "/usr/local/bin/redis-server",
        ],
        data_dir_flag: Some("--dir"),
        port_flag: Some("--port"),
        default_port: 6379,
        default_args: &["--save", "", "--appendonly", "no"],
        health_check_cmd: Some(&["redis-cli", "-p", "{port}", "ping"]),
        protocol_name: "redis",
        supports_wasm: false,
        needs_init: false,
        init_cmd: None,
    }
}

fn mongodb_spec() -> DatabaseSpec {
    DatabaseSpec {
        display_name: "MongoDB",
        binary_candidates: &[
            "mongod",
            "/opt/homebrew/opt/mongodb-community/bin/mongod",
            "/usr/local/bin/mongod",
        ],
        data_dir_flag: Some("--dbpath"),
        port_flag: Some("--port"),
        default_port: 27017,
        default_args: &["--noauth"],
        health_check_cmd: Some(&[
            "mongosh",
            "--port",
            "{port}",
            "--eval",
            "db.runCommand({ping:1})",
            "--quiet",
        ]),
        protocol_name: "mongodb",
        supports_wasm: false,
        needs_init: false,
        init_cmd: None,
    }
}

fn sqlite_spec() -> DatabaseSpec {
    DatabaseSpec {
        display_name: "SQLite",
        binary_candidates: &["sqlite3"],
        data_dir_flag: None,
        port_flag: None,
        default_port: 0,
        default_args: &[],
        health_check_cmd: None, // Embedded — no server process
        protocol_name: "file",
        supports_wasm: true,
        needs_init: false,
        init_cmd: None,
    }
}

fn custom_spec(name: &str) -> DatabaseSpec {
    // Leak the string to get a 'static lifetime — acceptable for a small number
    // of custom engines created at startup.
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    DatabaseSpec {
        display_name: leaked,
        binary_candidates: &[],
        data_dir_flag: None,
        port_flag: None,
        default_port: 0,
        default_args: &[],
        health_check_cmd: None,
        protocol_name: "tcp",
        supports_wasm: false,
        needs_init: false,
        init_cmd: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jouledb_spec() {
        let spec = get_spec(&DatabaseEngine::JouleDB);
        assert_eq!(spec.display_name, "JouleDB");
        assert_eq!(spec.default_port, 8080);
        assert_eq!(spec.data_dir_flag, Some("--data"));
        assert_eq!(spec.port_flag, Some("--port"));
        assert!(spec.health_check_cmd.is_none()); // built-in
        assert!(spec.supports_wasm);
        assert!(!spec.needs_init);
        assert!(spec.binary_candidates.contains(&"joule-db-server"));
    }

    #[test]
    fn test_postgres_spec() {
        let spec = get_spec(&DatabaseEngine::Postgres);
        assert_eq!(spec.display_name, "PostgreSQL");
        assert_eq!(spec.default_port, 5432);
        assert_eq!(spec.data_dir_flag, Some("-D"));
        assert_eq!(spec.port_flag, Some("-p"));
        assert_eq!(spec.protocol_name, "postgresql");
        assert!(spec.health_check_cmd.is_some());
        assert!(spec.needs_init);
        assert!(spec.init_cmd.is_some());
        assert!(!spec.supports_wasm);
        assert!(spec.binary_candidates.contains(&"postgres"));
    }

    #[test]
    fn test_mysql_spec() {
        let spec = get_spec(&DatabaseEngine::MySQL);
        assert_eq!(spec.display_name, "MySQL");
        assert_eq!(spec.default_port, 3306);
        assert_eq!(spec.data_dir_flag, Some("--datadir"));
        assert_eq!(spec.port_flag, Some("--port"));
        assert!(spec.health_check_cmd.is_some());
        assert!(spec.needs_init);
        assert!(spec.binary_candidates.contains(&"mysqld"));
    }

    #[test]
    fn test_redis_spec() {
        let spec = get_spec(&DatabaseEngine::Redis);
        assert_eq!(spec.display_name, "Redis");
        assert_eq!(spec.default_port, 6379);
        assert_eq!(spec.data_dir_flag, Some("--dir"));
        assert_eq!(spec.port_flag, Some("--port"));
        assert!(!spec.needs_init);
        assert!(spec.binary_candidates.contains(&"redis-server"));
        // Default args include save and appendonly settings
        assert!(spec.default_args.contains(&"--save"));
    }

    #[test]
    fn test_mongodb_spec() {
        let spec = get_spec(&DatabaseEngine::MongoDB);
        assert_eq!(spec.display_name, "MongoDB");
        assert_eq!(spec.default_port, 27017);
        assert_eq!(spec.data_dir_flag, Some("--dbpath"));
        assert!(spec.health_check_cmd.is_some());
        assert!(spec.binary_candidates.contains(&"mongod"));
    }

    #[test]
    fn test_sqlite_spec() {
        let spec = get_spec(&DatabaseEngine::SQLite);
        assert_eq!(spec.display_name, "SQLite");
        assert_eq!(spec.default_port, 0);
        assert!(spec.data_dir_flag.is_none());
        assert!(spec.port_flag.is_none());
        assert!(spec.health_check_cmd.is_none());
        assert!(spec.supports_wasm);
    }

    #[test]
    fn test_custom_spec() {
        let spec = get_spec(&DatabaseEngine::Custom("cockroachdb".into()));
        assert_eq!(spec.display_name, "cockroachdb");
        assert_eq!(spec.default_port, 0);
        assert!(spec.binary_candidates.is_empty());
        assert!(!spec.supports_wasm);
    }

    #[test]
    fn test_all_engines_have_specs() {
        let engines = vec![
            DatabaseEngine::JouleDB,
            DatabaseEngine::Postgres,
            DatabaseEngine::MySQL,
            DatabaseEngine::Redis,
            DatabaseEngine::MongoDB,
            DatabaseEngine::SQLite,
            DatabaseEngine::Custom("test".into()),
        ];
        for engine in engines {
            let spec = get_spec(&engine);
            assert!(!spec.display_name.is_empty());
            assert!(!spec.protocol_name.is_empty());
        }
    }

    #[test]
    fn test_workload_spec_database() {
        let wk = crate::WorkloadKind::database(DatabaseEngine::Postgres);
        let spec = get_workload_spec(&wk);
        assert_eq!(spec.display_name, "PostgreSQL");
        assert_eq!(spec.default_port, 5432);
    }

    #[test]
    fn test_workload_spec_process() {
        let wk = crate::WorkloadKind::process("/usr/bin/nginx", vec![]);
        let spec = get_workload_spec(&wk);
        assert_eq!(spec.display_name, "nginx");
        assert_eq!(spec.default_port, 0);
        assert_eq!(spec.protocol_name, "tcp");
    }

    #[test]
    fn test_workload_spec_container() {
        let wk = crate::WorkloadKind::container("redis:7-alpine");
        let spec = get_workload_spec(&wk);
        assert_eq!(spec.display_name, "redis:7-alpine");
        assert_eq!(spec.default_port, 0);
        assert_eq!(spec.protocol_name, "http");
    }

    #[test]
    fn test_health_check_has_port_placeholder() {
        for engine in [
            DatabaseEngine::Postgres,
            DatabaseEngine::MySQL,
            DatabaseEngine::Redis,
            DatabaseEngine::MongoDB,
        ] {
            let spec = get_spec(&engine);
            let cmd = spec
                .health_check_cmd
                .expect(&format!("{:?} should have health check", engine));
            let joined: String = cmd.iter().copied().collect::<Vec<_>>().join(" ");
            assert!(
                joined.contains("{port}"),
                "{:?} health check missing {{port}} placeholder: {}",
                engine,
                joined
            );
        }
    }
}
