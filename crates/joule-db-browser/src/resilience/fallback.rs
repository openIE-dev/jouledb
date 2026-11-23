//! Graceful degradation patterns

use std::sync::Arc;

/// Graceful degradation handler
pub struct GracefulDegradation {
    primary_available: Arc<std::sync::atomic::AtomicBool>,
    fallback_available: Arc<std::sync::atomic::AtomicBool>,
}

impl GracefulDegradation {
    /// Create new degradation handler
    pub fn new() -> Self {
        Self {
            primary_available: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            fallback_available: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Check if primary is available
    pub fn is_primary_available(&self) -> bool {
        self.primary_available
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Mark primary as unavailable
    pub fn mark_primary_unavailable(&self) {
        self.primary_available
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Mark primary as available
    pub fn mark_primary_available(&self) {
        self.primary_available
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Execute with fallback
    pub async fn execute_with_fallback<F, G, T>(&self, primary: F, fallback: G) -> Result<T, String>
    where
        F: std::future::Future<Output = Result<T, String>>,
        G: std::future::Future<Output = Result<T, String>>,
    {
        if self.is_primary_available() {
            match primary.await {
                Ok(result) => {
                    self.mark_primary_available();
                    Ok(result)
                }
                Err(_e) => {
                    self.mark_primary_unavailable();
                    // Fall back to secondary
                    fallback.await
                }
            }
        } else {
            fallback.await
        }
    }
}

impl Default for GracefulDegradation {
    fn default() -> Self {
        Self::new()
    }
}
