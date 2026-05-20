use joule_db_local::Database;
use std::sync::{Arc, Barrier, RwLock};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

const NUM_ACCOUNTS: usize = 10;
const INITIAL_BALANCE: u64 = 1000;
const NUM_THREADS: usize = 8;
const NUM_TRANSFERS: usize = 500;

#[test]
fn test_acid_bank_transfer() {
    let temp_dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(temp_dir.path()).unwrap());

    // 1. Initialize Accounts
    for i in 0..NUM_ACCOUNTS {
        let key = format!("acc_{}", i);
        let val = INITIAL_BALANCE.to_le_bytes();
        db.put(key.as_bytes(), &val).unwrap();
    }
    db.sync().unwrap();

    let initial_total = check_total_balance(&db);
    assert_eq!(initial_total, (NUM_ACCOUNTS as u64) * INITIAL_BALANCE);
    println!("Initial Total Balance: {}", initial_total);

    // 2. Spawn concurrent transfer threads
    let start_barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();

    for t_id in 0..NUM_THREADS {
        let db_clone = db.clone();
        let barrier = start_barrier.clone();

        handles.push(thread::spawn(move || {
            barrier.wait();

            // Random number generation (simple LCG to avoid dep)
            let mut seed = (t_id as u64 + 1) * 123456789;
            let mut rand = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as usize
            };

            for _ in 0..NUM_TRANSFERS {
                // Select two distinct accounts
                let from_idx = rand() % NUM_ACCOUNTS;
                let mut to_idx = rand() % NUM_ACCOUNTS;
                while to_idx == from_idx {
                    to_idx = rand() % NUM_ACCOUNTS;
                }

                let from_key = format!("acc_{}", from_idx);
                let to_key = format!("acc_{}", to_idx);
                let amount = (rand() as u64 % 50) + 1; // Transfer 1-50

                // TRANSACTION: Transfer money
                // Note: Database::put/get are typically atomic per-call.
                // But a transfer requires Read-Modify-Write on TWO keys.
                // Without explicit multi-key transaction API (begin/commit), this will RACE.
                // WE EXPECT THIS TO FAIL without proper Tx support.
                // If JouleDB has Tx support, we should use it.
                // Based on lib.rs, we don't have explicit TX exposed yet in Database struct efficiently.
                // However, we can simulate strict serializability by locking the DB externally
                // IF the DB itself doesn't provide it.
                // BUT the goal is to test the DB's capabilities.

                // Let's try to do it "optimistically" or see if we can use a lock if exposed.
                // The current API only has put/get.
                // Doing get(A) -> put(A-x) -> get(B) -> put(B+x) is NOT atomic.

                // If the test intention is to verify consistent state *at rest* or *with* TX support,
                // we need TX support.
                // Does Database expose `begin()`? No, it was commented out in docs.

                // So this test PROVES we need Transactions.
                // I will write it to fail (race), then we "fix" it by implementing a basic Tx mechanism or using a lock.
                // Wait, if I just want to verify thread-safety of `put` and `get`, that's one thing.
                // But Bank Transfer requires Multi-Key Atomicity.

                // Implementation attempting to be safe without TX (impossible):
                move_money(&db_clone, &from_key, &to_key, amount);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // 3. Verify Total Balance
    let final_total = check_total_balance(&db);
    println!("Final Total Balance: {}", final_total);

    assert_eq!(
        final_total, initial_total,
        "Money was created or destroyed due to race conditions!"
    );
}

fn check_total_balance(db: &Database) -> u64 {
    let mut total = 0;
    for i in 0..NUM_ACCOUNTS {
        let key = format!("acc_{}", i);
        if let Some(val) = db.get(key.as_bytes()).unwrap() {
            let bal = u64::from_le_bytes(val.try_into().unwrap());
            total += bal;
        }
    }
    total
}

fn move_money(db: &Database, from: &str, to: &str, amount: u64) {
    // This is the CRITICAL SECTION that needs Atomicity
    // We use transactional_update to ensure atomic execution.
    // The previous implementation WAS racy; this one is protected by the engine's write lock.

    let _ = db.transactional_update(|tx| {
        if let Some(from_bytes) = tx.get(from.as_bytes())? {
            let from_bal = u64::from_le_bytes(from_bytes.try_into().unwrap());
            if from_bal >= amount {
                if let Some(to_bytes) = tx.get(to.as_bytes())? {
                    let to_bal = u64::from_le_bytes(to_bytes.try_into().unwrap());

                    // Write back
                    let new_from = from_bal - amount;
                    let new_to = to_bal + amount;

                    tx.put(from.as_bytes(), &new_from.to_le_bytes())?;
                    tx.put(to.as_bytes(), &new_to.to_le_bytes())?;
                }
            }
        }
        Ok(())
    });
}
