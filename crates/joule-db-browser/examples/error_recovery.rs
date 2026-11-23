//! Error recovery example
//!
//! Demonstrates retry logic, circuit breakers, and graceful degradation

use joule_db_browser::resilience::{CircuitBreaker, GracefulDegradation, RetryPolicy};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Retry policy example
    let retry = RetryPolicy::new(3);

    let result: Result<&str, String> = retry
        .execute(|| async {
            // Simulate operation that might fail
            Ok::<&str, String>("Success")
        })
        .await;

    println!("Retry result: {:?}", result);

    // Circuit breaker example
    let breaker = CircuitBreaker::new(3, std::time::Duration::from_secs(1));

    if breaker.allow_request() {
        // Perform operation
        breaker.record_success();
    }

    // Graceful degradation example
    let degradation = GracefulDegradation::new();

    let result: Result<&str, String> = degradation
        .execute_with_fallback(
            async {
                // Primary operation (e.g., GPU)
                Ok::<&str, String>("Primary result")
            },
            async {
                // Fallback operation (e.g., CPU)
                Ok::<&str, String>("Fallback result")
            },
        )
        .await;

    println!("Degradation result: {:?}", result);
    Ok(())
}
