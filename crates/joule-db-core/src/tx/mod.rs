//! Transaction management for JouleDB
//!
//! Provides ACID transaction support with multiple isolation levels.

mod simple;
mod traits;

pub use simple::{SimpleReadTransaction, SimpleTransaction, SimpleTransactionManager};
pub use traits::{IsolationLevel, ReadTransaction, Transaction, TransactionManager, TxId, TxState};
