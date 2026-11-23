//! Retry logic for transient failures

use std::future::Future;
use std::time::Duration;

/// Retry policy configuration
pub struct RetryPolicy {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Create new retry policy
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }

    /// Execute with retry
    pub async fn execute<F, Fut, T, E>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut delay = self.initial_delay;
        let mut last_error: Option<E> = None;

        for attempt in 0..self.max_attempts {
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::debug!("Retry attempt {} failed: {:?}", attempt + 1, e);
                    last_error = Some(e);

                    if attempt == self.max_attempts - 1 {
                        // Last attempt failed, will return error below
                        break;
                    }

                    // Wait before retry
                    #[cfg(target_arch = "wasm32")]
                    {
                        use js_sys::Promise;
                        use wasm_bindgen_futures::JsFuture;

                        let delay_ms = delay.as_millis() as i32;
                        let _ = JsFuture::from(Promise::new(&mut |resolve, _| {
                            if let Some(window) = web_sys::window() {
                                let _ = window
                                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                                        &resolve, delay_ms,
                                    );
                            }
                        }))
                        .await;
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        tokio::time::sleep(delay).await;
                    }

                    let delay_secs = delay.as_secs_f64() * self.backoff_multiplier;
                    let max_delay_secs = self.max_delay.as_secs_f64();
                    if delay_secs > max_delay_secs {
                        delay = self.max_delay;
                    } else {
                        delay = Duration::from_secs_f64(delay_secs);
                    }
                }
            }
        }

        // All attempts exhausted (or max_attempts was 0)
        Err(last_error.expect("At least one attempt should have been made"))
    }
}
