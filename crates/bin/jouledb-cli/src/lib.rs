//! JouleDB CLI Library
//!
//! This crate provides the command-line interface for JouleDB.

pub mod cloud;
pub mod commands;
pub mod config;
pub mod error;
pub mod output;

pub use config::Config;
pub use error::{CliError, Result};
pub use output::{Output, OutputFormat, Progress};
