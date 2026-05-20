//! Output formatting and display

use clap::ValueEnum;
use colored::Colorize;
use comfy_table::{ContentArrangement, Table, presets};
use serde::Serialize;

/// Output format
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON format
    Json,
    /// Table format
    Table,
    /// CSV format
    Csv,
}

/// Output handler
pub struct Output {
    format: OutputFormat,
    verbose: bool,
    quiet: bool,
}

impl Output {
    /// Create new output handler
    pub fn new(format: OutputFormat, verbose: bool, quiet: bool) -> Self {
        Self {
            format,
            verbose,
            quiet,
        }
    }

    /// Print success message
    pub fn success(&self, message: &str) {
        if !self.quiet {
            println!("{} {}", "✓".green(), message);
        }
    }

    /// Print error message
    pub fn error(&self, message: &str) {
        eprintln!("{} {}", "✗".red(), message.red());
    }

    /// Print warning message
    pub fn warning(&self, message: &str) {
        if !self.quiet {
            eprintln!("{} {}", "⚠".yellow(), message.yellow());
        }
    }

    /// Print info message
    pub fn info(&self, message: &str) {
        if !self.quiet {
            println!("{} {}", "ℹ".blue(), message);
        }
    }

    /// Print verbose message
    pub fn verbose(&self, message: &str) {
        if self.verbose && !self.quiet {
            println!("{} {}", "·".dimmed(), message.dimmed());
        }
    }

    /// Print data in configured format
    pub fn data<T: Serialize>(&self, data: &T) -> crate::Result<()> {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(data)?;
                println!("{}", json);
            }
            _ => {
                let json = serde_json::to_string_pretty(data)?;
                println!("{}", json);
            }
        }
        Ok(())
    }

    /// Print a table
    pub fn table(&self, headers: Vec<&str>, rows: Vec<Vec<String>>) {
        match self.format {
            OutputFormat::Json => {
                let data: Vec<_> = rows
                    .iter()
                    .map(|row| {
                        headers
                            .iter()
                            .zip(row.iter())
                            .map(|(h, v)| (h.to_string(), v.clone()))
                            .collect::<std::collections::HashMap<_, _>>()
                    })
                    .collect();
                if let Ok(json) = serde_json::to_string_pretty(&data) {
                    println!("{}", json);
                }
            }
            OutputFormat::Csv => {
                println!("{}", headers.join(","));
                for row in rows {
                    println!("{}", row.join(","));
                }
            }
            _ => {
                let mut table = Table::new();
                table.load_preset(presets::UTF8_FULL_CONDENSED);
                table.set_content_arrangement(ContentArrangement::Dynamic);

                table.set_header(headers);
                for row in rows {
                    table.add_row(row);
                }

                println!("{}", table);
            }
        }
    }

    /// Print key-value pairs
    pub fn key_value(&self, pairs: Vec<(&str, String)>) {
        match self.format {
            OutputFormat::Json => {
                let data: std::collections::HashMap<_, _> = pairs.into_iter().collect();
                if let Ok(json) = serde_json::to_string_pretty(&data) {
                    println!("{}", json);
                }
            }
            _ => {
                let max_key_len = pairs.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
                for (key, value) in pairs {
                    println!(
                        "{:width$}  {}",
                        format!("{}:", key).cyan(),
                        value,
                        width = max_key_len + 1
                    );
                }
            }
        }
    }

    /// Print raw text
    pub fn raw(&self, text: &str) {
        println!("{}", text);
    }

    /// Print a section header
    pub fn section(&self, title: &str) {
        if !self.quiet && !matches!(self.format, OutputFormat::Json) {
            println!();
            println!("{}", title.bold().underline());
            println!();
        }
    }

    /// Get output format
    pub fn format(&self) -> OutputFormat {
        self.format
    }

    /// Is quiet mode
    pub fn is_quiet(&self) -> bool {
        self.quiet
    }
}

/// Progress indicator
pub struct Progress {
    bar: indicatif::ProgressBar,
}

impl Progress {
    /// Create new progress bar
    pub fn new(total: u64, message: &str) -> Self {
        let bar = indicatif::ProgressBar::new(total);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        bar.set_message(message.to_string());
        Self { bar }
    }

    /// Create spinner
    pub fn spinner(message: &str) -> Self {
        let bar = indicatif::ProgressBar::new_spinner();
        bar.set_style(
            indicatif::ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        bar.set_message(message.to_string());
        bar.enable_steady_tick(std::time::Duration::from_millis(100));
        Self { bar }
    }

    /// Increment progress
    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    /// Set message
    pub fn set_message(&self, message: &str) {
        self.bar.set_message(message.to_string());
    }

    /// Finish with message
    pub fn finish(&self, message: &str) {
        self.bar.finish_with_message(message.to_string());
    }

    /// Finish and clear
    pub fn finish_and_clear(&self) {
        self.bar.finish_and_clear();
    }
}
