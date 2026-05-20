//! Local storage backends

pub mod file;
pub mod single_file;
pub mod wal;

pub use wal::{RecoveryManager, RecoveryResult, WalEntry, WalEntryType, WalManager};
