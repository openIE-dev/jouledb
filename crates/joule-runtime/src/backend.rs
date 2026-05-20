use crate::{InstanceInfo, InstanceState, RuntimeConfig, RuntimeError, ServerOverrides};

/// Output from executing a command inside a running instance.
#[derive(Debug, Clone)]
pub struct ExecOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Trait for runtime isolation backends.
///
/// Each backend (native, VM, WASM, container) implements this trait to provide
/// a uniform interface for starting, stopping, and monitoring instances.
pub trait RuntimeBackend: Send + Sync {
    /// Start a new instance with the given configuration.
    fn start(
        &self,
        config: &RuntimeConfig,
        instance: &InstanceInfo,
        overrides: &ServerOverrides,
    ) -> impl std::future::Future<Output = Result<(), RuntimeError>> + Send;

    /// Stop a running instance by ID.
    fn stop(
        &self,
        instance_id: &str,
    ) -> impl std::future::Future<Output = Result<(), RuntimeError>> + Send;

    /// Get the current state of an instance.
    fn status(
        &self,
        instance_id: &str,
    ) -> impl std::future::Future<Output = Result<InstanceState, RuntimeError>> + Send;

    /// Check if an instance is healthy and responding.
    fn health_check(
        &self,
        instance_id: &str,
    ) -> impl std::future::Future<Output = Result<bool, RuntimeError>> + Send;

    /// Execute a command inside a running instance.
    ///
    /// Default implementation returns an error (not supported by this backend).
    fn exec(
        &self,
        _instance_id: &str,
        _command: &[String],
    ) -> impl std::future::Future<Output = Result<ExecOutput, RuntimeError>> + Send {
        async {
            Err(RuntimeError::ProcessError(
                "exec not supported by this backend".into(),
            ))
        }
    }

    /// Retrieve logs from an instance.
    ///
    /// Returns up to `tail` lines (None = all). Default implementation returns empty.
    fn logs(
        &self,
        _instance_id: &str,
        _tail: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Vec<String>, RuntimeError>> + Send {
        async { Ok(vec![]) }
    }
}
