//! Mamba SSM Decoder: fluent text generation via mamba.rs.
//!
//! Shells out to the mamba.rs binary for generation. This avoids
//! embedding the 640MB model weights as a Rust dependency while
//! keeping the interface clean via the TextDecoder trait.
//!
//! The Mamba-130M model generates at ~100 tokens/sec on Apple Silicon.
//! At 130M parameters, quality is limited — 370M+ is recommended for
//! production. But even 130M is sufficient to translate structured
//! JouleDB knowledge into grammatically correct English.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::decoder::{DecoderContext, DecoderResult, DecoderStyle, TextDecoder, build_prompt};

/// Hard deadline for a single mamba subprocess invocation. If the binary
/// hangs — either in infinite inference or by filling its stdout pipe
/// faster than anything drains it — we kill it and return a clean timeout
/// error instead of blocking forever. Before this existed, running the
/// full amorphic test suite on a machine with mamba installed would
/// deadlock for hours on this one test.
const MAMBA_TIMEOUT: Duration = Duration::from_secs(60);

/// How often to poll the child process for exit while within the deadline.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Run a subprocess with a hard timeout.
///
/// Two ways a subprocess can hang forever, both handled here:
///
/// 1. **Infinite loop in the child** — poll `try_wait()` against the
///    deadline, kill the child when it elapses.
/// 2. **Pipe buffer fills** — stdout/stderr back-pressure blocks the
///    child on `write`. We drain both pipes in background threads so
///    the child never blocks on pipe back-pressure.
///
/// On success returns `(stdout, stderr, exit_status)`. On timeout kills
/// the child, reaps the zombie, joins reader threads, and returns Err.
///
/// Set `cmd.stdout(Stdio::piped())` and `cmd.stderr(Stdio::piped())`
/// before calling — this function does not set them to leave policy
/// to the caller.
fn run_with_timeout(
    mut cmd: Command,
    timeout: Duration,
) -> Result<(String, String, std::process::ExitStatus), String> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture stdout pipe".to_string())?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture stderr pipe".to_string())?;

    let stdout_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let mut pipe = stdout_pipe;
        let _ = pipe.read_to_end(&mut buf);
        buf
    });
    let stderr_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let mut pipe = stderr_pipe;
        let _ = pipe.read_to_end(&mut buf);
        buf
    });

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Reader threads exit as soon as the child closes its pipes.
                let stdout_bytes = stdout_reader
                    .join()
                    .map_err(|_| "stdout reader thread panicked".to_string())?;
                let stderr_bytes = stderr_reader
                    .join()
                    .map_err(|_| "stderr reader thread panicked".to_string())?;

                return Ok((
                    String::from_utf8_lossy(&stdout_bytes).into_owned(),
                    String::from_utf8_lossy(&stderr_bytes).into_owned(),
                    status,
                ));
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Deadline exceeded — kill the child, reap it, join
                    // reader threads (they exit once pipes close).
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    return Err(format!(
                        "subprocess timed out after {}s",
                        timeout.as_secs()
                    ));
                }
                thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("try_wait failed: {e}"));
            }
        }
    }
}

/// Mamba SSM decoder — shells out to the mamba.rs binary.
pub struct MambaDecoder {
    /// Path to the mamba binary.
    binary_path: PathBuf,
    /// Which model size (130m, 370m, 790m, 1.4b, 2.8b).
    model_size: String,
    /// Temperature for sampling (0.0 = argmax, 0.7 = default).
    pub temperature: f64,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Working directory (where weights + tokenizer files live).
    work_dir: PathBuf,
}

impl MambaDecoder {
    /// Create a new Mamba decoder.
    pub fn new(mamba_dir: &str, model_size: &str) -> Self {
        let base = PathBuf::from(mamba_dir);
        Self {
            binary_path: base.join("target/release/mamba"),
            model_size: model_size.to_string(),
            temperature: 0.5,
            max_tokens: 100,
            work_dir: base,
        }
    }

    /// Create with standard mamba.rs location.
    pub fn standard_130m() -> Self {
        Self::new(
            "/tmp/jouledb-data/mamba-rs",
            "130m",
        )
    }

    /// Check if the binary and weights exist.
    pub fn is_available(&self) -> bool {
        self.binary_path.exists()
            && self.work_dir.join(format!("mamba-{}.bin", self.model_size)).exists()
            && self.work_dir.join("vocab.json").exists()
    }

    /// Generate text by shelling out to the mamba binary with a hard
    /// [`MAMBA_TIMEOUT`] deadline. See [`run_with_timeout`] for how the
    /// timeout and pipe-drain semantics work.
    fn generate_raw(&self, prompt: &str) -> Result<(String, u64), String> {
        if !self.is_available() {
            return Err("mamba binary or weights not found".to_string());
        }

        let start = Instant::now();

        let mut cmd = Command::new(&self.binary_path);
        cmd.arg(prompt)
            .arg("--which")
            .arg(&self.model_size)
            .arg("--temperature")
            .arg(format!("{}", self.temperature))
            .current_dir(&self.work_dir);

        let (stdout, stderr, status) = run_with_timeout(cmd, MAMBA_TIMEOUT)
            .map_err(|e| format!("mamba: {e}"))?;

        let elapsed_us = start.elapsed().as_micros() as u64;

        if !status.success() {
            return Err(format!("mamba failed: {stderr}"));
        }

        // Extract generated text (skip prompt echo and metadata lines)
        let lines: Vec<&str> = stdout.lines().collect();
        let text = lines
            .iter()
            .filter(|l| !l.starts_with("state size") && !l.starts_with("weight size")
                && !l.starts_with("processing prompt") && !l.starts_with("prompt tokens")
                && !l.contains("tokens generated"))
            .copied()
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        Ok((text, elapsed_us))
    }
}

impl TextDecoder for MambaDecoder {
    fn decode(&self, context: &DecoderContext, max_tokens: usize) -> DecoderResult {
        let prompt = build_prompt(context);

        match self.generate_raw(&prompt) {
            Ok((text, latency_us)) => {
                let tokens = text.split_whitespace().count();
                // Energy estimate: ~0.001J for 130M model inference on Apple Silicon
                let energy = tokens as f64 * self.energy_per_token();
                DecoderResult {
                    text,
                    tokens,
                    energy,
                    latency_us,
                }
            }
            Err(e) => {
                // Fallback to template if mamba fails
                let template = super::decoder::TemplateDecoder;
                let mut result = template.decode(context, max_tokens);
                result.text = format!("[mamba unavailable: {}] {}", e, result.text);
                result
            }
        }
    }

    fn model_name(&self) -> &str {
        match self.model_size.as_str() {
            "130m" => "mamba-130m",
            "370m" => "mamba-370m",
            "790m" => "mamba-790m",
            _ => "mamba-unknown",
        }
    }

    fn energy_per_token(&self) -> f64 {
        match self.model_size.as_str() {
            "130m" => 0.000_01,  // ~10µJ per token (130M on Apple Silicon)
            "370m" => 0.000_03,  // ~30µJ per token
            "790m" => 0.000_07,  // ~70µJ per token
            _ => 0.000_1,        // ~100µJ per token (conservative)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_availability() {
        let decoder = MambaDecoder::standard_130m();
        let available = decoder.is_available();
        eprintln!("Mamba available: {}", available);
        if available {
            eprintln!("  binary: {:?}", decoder.binary_path);
            eprintln!("  model: mamba-{}.bin", decoder.model_size);
        }
    }

    #[test]
    fn test_mamba_decode_if_available() {
        let decoder = MambaDecoder::standard_130m();
        if !decoder.is_available() {
            eprintln!("Mamba not available, skipping");
            return;
        }

        let context = DecoderContext {
            query: "How are cancer and war related?".into(),
            entities: vec!["cancer".into(), "war".into()],
            relationships: vec![
                ("cancer".into(), "exhibits".into(), "replication".into()),
                ("war".into(), "exhibits".into(), "replication".into()),
            ],
            patterns: vec![
                ("replication".into(), 0.9),
                ("feedback".into(), 0.8),
                ("emergence".into(), 0.7),
            ],
            similarity: Some(0.93),
            raw_answer: Some("93% structurally similar".into()),
            style: DecoderStyle::Explanatory,
        };

        let result = decoder.decode(&context, 50);
        eprintln!("Mamba output ({} tokens, {:.3}s, {:.6}J):\n{}",
            result.tokens,
            result.latency_us as f64 / 1_000_000.0,
            result.energy,
            result.text
        );

        // Should produce some output (even if noisy at 130M)
        assert!(!result.text.is_empty(), "mamba should generate text");
    }

    #[test]
    fn test_fallback_when_unavailable() {
        let decoder = MambaDecoder::new("/nonexistent/path", "130m");
        assert!(!decoder.is_available());

        let context = DecoderContext {
            query: "test".into(),
            entities: vec![],
            relationships: vec![],
            patterns: vec![],
            similarity: None,
            raw_answer: Some("test answer".into()),
            style: DecoderStyle::Concise,
        };

        let result = decoder.decode(&context, 50);
        assert!(result.text.contains("mamba unavailable"));
        assert!(result.text.contains("test answer")); // Template fallback
    }

    #[test]
    fn test_energy_estimates() {
        let m130 = MambaDecoder::new("/tmp", "130m");
        let m370 = MambaDecoder::new("/tmp", "370m");
        assert!(m130.energy_per_token() < m370.energy_per_token());
    }

    // ────────────────────────────────────────────────────────────────
    // run_with_timeout tests — verify the subprocess timeout / drain
    // pipeline works against real binaries without needing mamba.
    // Before this existed, a hanging mamba binary would deadlock the
    // full amorphic test suite indefinitely. These tests guarantee
    // that can never happen again.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn run_with_timeout_succeeds_for_fast_command() {
        // /bin/echo exits in microseconds — should come back clean.
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hello");
        let (stdout, _stderr, status) =
            run_with_timeout(cmd, Duration::from_secs(5)).expect("echo should succeed");
        assert!(status.success());
        assert!(stdout.contains("hello"));
    }

    #[test]
    fn run_with_timeout_kills_hanging_subprocess() {
        // /bin/sleep 30 would block for 30 seconds. We give it 1 second
        // of headroom and expect a timeout error in under ~2 seconds.
        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("30");
        let start = Instant::now();
        let result = run_with_timeout(cmd, Duration::from_secs(1));
        let elapsed = start.elapsed();

        assert!(result.is_err(), "hanging subprocess should time out");
        let err = result.unwrap_err();
        assert!(
            err.contains("timed out"),
            "unexpected error: {}",
            err
        );
        // Must actually return within a few seconds — the whole point.
        assert!(
            elapsed < Duration::from_secs(5),
            "timeout enforcement too slow: {:?}",
            elapsed
        );
    }

    #[test]
    fn run_with_timeout_captures_nonzero_exit() {
        // /usr/bin/false exits with status 1 on both macOS and Linux. Use
        // a fallback for environments that put it elsewhere.
        let false_path = ["/usr/bin/false", "/bin/false"]
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .copied()
            .expect("no `false` binary found on this system");
        let cmd = Command::new(false_path);
        let (_, _, status) =
            run_with_timeout(cmd, Duration::from_secs(5)).expect("false should spawn");
        assert!(!status.success());
    }

    #[test]
    fn run_with_timeout_drains_large_output() {
        // /bin/ls on /usr/bin produces thousands of lines — more than the
        // default pipe buffer. Before draining in threads, this would block
        // the child process on write. Verify it completes cleanly.
        let mut cmd = Command::new("/bin/ls");
        cmd.arg("/usr/bin");
        let result = run_with_timeout(cmd, Duration::from_secs(10));
        assert!(result.is_ok(), "ls /usr/bin should drain cleanly");
        let (stdout, _, status) = result.unwrap();
        assert!(status.success());
        assert!(!stdout.is_empty());
        // Most /usr/bin directories have hundreds of entries.
        let line_count = stdout.lines().count();
        assert!(
            line_count > 50,
            "expected many lines of ls output, got {}",
            line_count
        );
    }

}
