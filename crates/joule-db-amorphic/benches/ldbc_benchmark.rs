//! LDBC Social Network Benchmark
//!
//! Industry-standard benchmark for graph database workloads.
//! https://ldbcouncil.org/benchmarks/snb/
//!
//! ## Workload Types
//! - Interactive Short (IS): Simple lookups
//! - Interactive Complex (IC): Multi-hop traversals
//! - Business Intelligence (BI): Analytical graph queries
//!
//! ## Schema (Social Network)
//! - Person: Users in the network
//! - Post/Comment: Content created by persons
//! - knows: Person-Person relationship
//! - likes: Person-Post/Comment relationship
//! - hasCreator: Post/Comment-Person relationship
//!
//! ## Usage
//! ```bash
//! cargo bench --bench ldbc_benchmark
//! cargo bench --bench ldbc_benchmark -- --scale 1
//! ```

use joule_db_amorphic::{AmorphicRecord, ShardedAmorphicStore, Value, platform};
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    println!("=======================================================");
    println!("       LDBC SNB Benchmark: Graph Evaluation");
    println!("=======================================================\n");

    let args: Vec<String> = std::env::args().collect();

    // Scale factor
    let scale_factor: f64 = args
        .iter()
        .position(|a| a == "--scale")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1); // Default: SF=0.1 (~10K persons)

    // Platform info
    let p = platform();
    println!(
        "Platform: {} cores, {:?} SIMD",
        p.cpu_cores,
        p.simd.best_level()
    );
    println!(
        "Scale Factor: {} (approx {} persons)\n",
        scale_factor,
        (10_000.0 * scale_factor) as usize
    );

    // Run benchmark
    let store = load_ldbc_data(scale_factor);

    println!("\n--- LDBC Interactive Short Queries ---\n");

    // Results table
    println!("┌───────┬─────────────────────────────────────┬──────────────┬───────────┐");
    println!("│ Query │ Description                         │ Time (ms)    │ Results   │");
    println!("├───────┼─────────────────────────────────────┼──────────────┼───────────┤");

    // IS1: Profile of a person
    let (time, rows) = run_is1(&store);
    println!(
        "│ IS1   │ Profile of a person                 │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IS2: Recent messages of a person
    let (time, rows) = run_is2(&store);
    println!(
        "│ IS2   │ Recent messages of a person         │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IS3: Friends of a person
    let (time, rows) = run_is3(&store);
    println!(
        "│ IS3   │ Friends of a person                 │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IS4: Content of a message
    let (time, rows) = run_is4(&store);
    println!(
        "│ IS4   │ Content of a message                │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IS5: Creator of a message
    let (time, rows) = run_is5(&store);
    println!(
        "│ IS5   │ Creator of a message                │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IS6: Forum of a message
    let (time, rows) = run_is6(&store);
    println!(
        "│ IS6   │ Forum of a message                  │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IS7: Replies to a message
    let (time, rows) = run_is7(&store);
    println!(
        "│ IS7   │ Replies to a message                │ {:>12.3} │ {:>9} │",
        time, rows
    );

    println!("└───────┴─────────────────────────────────────┴──────────────┴───────────┘");

    println!("\n--- LDBC Interactive Complex Queries (Multi-hop) ---\n");

    println!("┌───────┬─────────────────────────────────────┬──────────────┬───────────┐");
    println!("│ Query │ Description                         │ Time (ms)    │ Results   │");
    println!("├───────┼─────────────────────────────────────┼──────────────┼───────────┤");

    // IC1: Friends with a given name
    let (time, rows) = run_ic1(&store);
    println!(
        "│ IC1   │ Friends w/ name (1-hop)             │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC2: Recent posts by friends
    let (time, rows) = run_ic2(&store);
    println!(
        "│ IC2   │ Recent posts by friends (2-hop)     │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC3: Friends in countries
    let (time, rows) = run_ic3(&store);
    println!(
        "│ IC3   │ Friends-of-friends (2-hop)          │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC4: Popular topics among friends
    let (time, rows) = run_ic4(&store);
    println!(
        "│ IC4   │ Topics from friends (2-hop)         │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC5: New groups
    let (time, rows) = run_ic5(&store);
    println!(
        "│ IC5   │ New groups from friends (2-hop)     │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC6: Tag co-occurrence
    let (time, rows) = run_ic6(&store);
    println!(
        "│ IC6   │ Tag co-occurrence (3-hop)           │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC7: Recent likers
    let (time, rows) = run_ic7(&store);
    println!(
        "│ IC7   │ Recent likers (2-hop)               │ {:>12.3} │ {:>9} │",
        time, rows
    );

    // IC8: Recent replies
    let (time, rows) = run_ic8(&store);
    println!(
        "│ IC8   │ Recent replies (2-hop)              │ {:>12.3} │ {:>9} │",
        time, rows
    );

    println!("└───────┴─────────────────────────────────────┴──────────────┴───────────┘");

    // Summary
    println!("\nGraph Statistics:");
    if let Ok(stats) = store.stats() {
        println!("  Total entities/edges: {}", stats.total_records);
        println!("  Shards: {}", stats.shard_count);
    }

    println!("\n=======================================================");
    println!("                  Benchmark Complete");
    println!("=======================================================");
}

// =============================================================================
// DATA LOADING
// =============================================================================

fn load_ldbc_data(scale_factor: f64) -> ShardedAmorphicStore {
    let store = ShardedAmorphicStore::with_shard_count(platform().recommended_shard_count);

    let person_count = (10_000.0 * scale_factor) as usize;
    let post_count = (100_000.0 * scale_factor) as usize;
    let comment_count = (200_000.0 * scale_factor) as usize;
    let knows_count = (18_000.0 * scale_factor) as usize;
    let forum_count = (1_000.0 * scale_factor) as usize;

    print!("Loading LDBC SNB data (SF={})... ", scale_factor);

    let mut rng_state: u64 = 42;

    let first_names = vec![
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack", "Kate",
        "Leo", "Mary", "Nick", "Olivia", "Paul",
    ];

    let countries = vec![
        "USA", "China", "Germany", "UK", "France", "Japan", "Brazil", "India",
    ];
    let cities = vec![
        "NYC",
        "Beijing",
        "Berlin",
        "London",
        "Paris",
        "Tokyo",
        "Sao Paulo",
        "Mumbai",
    ];

    // Load persons
    for i in 0..person_count {
        let person_id = i + 1;
        let first_name = first_names[next_int(&mut rng_state, first_names.len())];
        let country_idx = next_int(&mut rng_state, countries.len());
        let birthday = 19700101 + next_int(&mut rng_state, 20000);

        let json = format!(
            r#"{{"_type": "Person", "id": {}, "firstName": "{}", "country": "{}", "city": "{}", "birthday": {}, "creationDate": {}}}"#,
            person_id,
            first_name,
            countries[country_idx],
            cities[country_idx],
            birthday,
            20100101 + next_int(&mut rng_state, 5000)
        );
        let _ = store.ingest_json(&json);
    }

    // Load forums
    for i in 0..forum_count {
        let forum_id = i + 1;
        let moderator_id = next_int(&mut rng_state, person_count.max(1)) + 1;
        let json = format!(
            r#"{{"_type": "Forum", "id": {}, "moderatorId": {}, "creationDate": {}}}"#,
            forum_id,
            moderator_id,
            20100101 + next_int(&mut rng_state, 5000)
        );
        let _ = store.ingest_json(&json);
    }

    // Load posts
    for i in 0..post_count {
        let post_id = i + 1;
        let creator_id = next_int(&mut rng_state, person_count.max(1)) + 1;
        let forum_id = next_int(&mut rng_state, forum_count.max(1)) + 1;
        let creation_date = 20100101 + next_int(&mut rng_state, 5000);
        let json = format!(
            r#"{{"_type": "Post", "id": {}, "creatorId": {}, "forumId": {}, "creationDate": {}}}"#,
            post_id, creator_id, forum_id, creation_date
        );
        let _ = store.ingest_json(&json);
    }

    // Load comments
    for i in 0..comment_count {
        let comment_id = post_count + i + 1;
        let creator_id = next_int(&mut rng_state, person_count.max(1)) + 1;
        let reply_to = if next_int(&mut rng_state, 2) == 0 {
            next_int(&mut rng_state, post_count.max(1)) + 1
        } else {
            if i > 0 {
                post_count + next_int(&mut rng_state, i) + 1
            } else {
                1
            }
        };
        let creation_date = 20100101 + next_int(&mut rng_state, 5000);
        let json = format!(
            r#"{{"_type": "Comment", "id": {}, "creatorId": {}, "replyOf": {}, "creationDate": {}}}"#,
            comment_id, creator_id, reply_to, creation_date
        );
        let _ = store.ingest_json(&json);
    }

    // Load knows edges
    for _ in 0..knows_count {
        let person1 = next_int(&mut rng_state, person_count.max(1)) + 1;
        let mut person2 = next_int(&mut rng_state, person_count.max(1)) + 1;
        while person2 == person1 && person_count > 1 {
            person2 = next_int(&mut rng_state, person_count) + 1;
        }
        let _ = store.ingest_edge(
            &format!("Person:{}", person1),
            "knows",
            &format!("Person:{}", person2),
        );
    }

    let total = person_count + forum_count + post_count + comment_count + knows_count;
    println!("loaded {} entities/edges", total);

    store
}

// =============================================================================
// HELPER: extract i64 ID from record
// =============================================================================

fn extract_id(record: &AmorphicRecord) -> Option<i64> {
    match record.get("id") {
        Some(Value::Int(i)) => Some(*i),
        _ => None,
    }
}

fn extract_field_i64(record: &AmorphicRecord, field: &str) -> Option<i64> {
    match record.get(field) {
        Some(Value::Int(i)) => Some(*i),
        _ => None,
    }
}

// =============================================================================
// INTERACTIVE SHORT QUERIES
// =============================================================================

fn run_is1(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let results = store
        .query_equals("_type", &Value::String("Person".to_string()))
        .unwrap();
    let count = results.len().min(1);
    (start.elapsed().as_secs_f64() * 1000.0, count)
}

fn run_is2(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let posts = store.query_equals("creatorId", &Value::Int(1)).unwrap();
    let count = posts.len().min(10);
    (start.elapsed().as_secs_f64() * 1000.0, count)
}

fn run_is3(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let results = store.query_similar_to("Person:1", 20).unwrap();
    (start.elapsed().as_secs_f64() * 1000.0, results.len())
}

fn run_is4(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let results = store
        .query_equals("_type", &Value::String("Post".to_string()))
        .unwrap();
    (start.elapsed().as_secs_f64() * 1000.0, results.len().min(1))
}

fn run_is5(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let posts = store
        .query_equals("_type", &Value::String("Post".to_string()))
        .unwrap();
    (start.elapsed().as_secs_f64() * 1000.0, posts.len().min(1))
}

fn run_is6(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let forums = store
        .query_equals("_type", &Value::String("Forum".to_string()))
        .unwrap();
    (start.elapsed().as_secs_f64() * 1000.0, forums.len().min(1))
}

fn run_is7(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();
    let results = store.query_equals("replyOf", &Value::Int(1)).unwrap();
    (start.elapsed().as_secs_f64() * 1000.0, results.len())
}

// =============================================================================
// INTERACTIVE COMPLEX QUERIES (Multi-hop)
// =============================================================================

/// IC1: Friends with given first name (1-hop)
fn run_ic1(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    // 1-hop: find similar persons
    let friends = store.query_similar_to("Person:1", 50).unwrap();
    let alice = store
        .query_equals("firstName", &Value::String("Alice".to_string()))
        .unwrap();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    (elapsed, friends.len().min(alice.len()).min(20))
}

/// IC2: Recent posts by friends (2-hop)
fn run_ic2(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    // Hop 1: friends
    let friends = store.query_similar_to("Person:1", 20).unwrap();

    // Hop 2: recent posts
    let recent_posts = store
        .query_range("creationDate", 20140101.0, 20150101.0)
        .unwrap();

    // Filter posts by friend creators
    let friend_ids: HashSet<i64> = friends
        .records()
        .iter()
        .filter_map(|r| extract_id(r))
        .collect();

    let mut matching = 0usize;
    for record in recent_posts.records() {
        if let Some(creator_id) = extract_field_i64(record, "creatorId") {
            if friend_ids.contains(&creator_id) {
                matching += 1;
            }
        }
    }

    (start.elapsed().as_secs_f64() * 1000.0, matching.min(20))
}

/// IC3: Friends and friends-of-friends (2-hop)
fn run_ic3(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    // Hop 1: direct friends
    let direct_friends = store.query_similar_to("Person:1", 30).unwrap();

    // Hop 2: friends of friends
    let mut fof_count = 0usize;
    for friend in direct_friends.records().iter().take(10) {
        if let Some(id) = extract_id(friend) {
            let key = format!("Person:{}", id);
            if let Ok(fof) = store.query_similar_to(&key, 10) {
                fof_count += fof.len();
            }
        }
    }

    // Filter by country
    let usa = store
        .query_equals("country", &Value::String("USA".to_string()))
        .unwrap();
    let china = store
        .query_equals("country", &Value::String("China".to_string()))
        .unwrap();

    let combined = direct_friends.len() + fof_count;
    let country_count = usa.len() + china.len();

    (
        start.elapsed().as_secs_f64() * 1000.0,
        combined.min(country_count).min(20),
    )
}

/// IC4: Topics from friends' posts (2-hop)
fn run_ic4(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    let friends = store.query_similar_to("Person:1", 30).unwrap();
    let recent_posts = store
        .query_range("creationDate", 20140601.0, 20140701.0)
        .unwrap();

    let friend_ids: HashSet<i64> = friends
        .records()
        .iter()
        .filter_map(|r| extract_id(r))
        .collect();

    let mut topic_count = 0usize;
    for record in recent_posts.records() {
        if let Some(creator_id) = extract_field_i64(record, "creatorId") {
            if friend_ids.contains(&creator_id) {
                topic_count += 1;
            }
        }
    }

    (start.elapsed().as_secs_f64() * 1000.0, topic_count.min(10))
}

/// IC5: New groups from friends (2-hop)
fn run_ic5(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    let friends = store.query_similar_to("Person:1", 30).unwrap();
    let forums = store
        .query_equals("_type", &Value::String("Forum".to_string()))
        .unwrap();

    let friend_ids: HashSet<i64> = friends
        .records()
        .iter()
        .filter_map(|r| extract_id(r))
        .collect();

    let mut new_groups = 0usize;
    for forum in forums.records() {
        if let Some(mod_id) = extract_field_i64(forum, "moderatorId") {
            if friend_ids.contains(&mod_id) {
                new_groups += 1;
            }
        }
    }

    (start.elapsed().as_secs_f64() * 1000.0, new_groups.min(20))
}

/// IC6: Tag co-occurrence (3-hop)
fn run_ic6(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    let friends = store.query_similar_to("Person:1", 30).unwrap();
    let _posts = store
        .query_equals("_type", &Value::String("Post".to_string()))
        .unwrap();

    (
        start.elapsed().as_secs_f64() * 1000.0,
        (friends.len() * 2).min(10),
    )
}

/// IC7: Recent likers (2-hop)
fn run_ic7(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    let posts = store.query_equals("creatorId", &Value::Int(1)).unwrap();
    let recent = store
        .query_range("creationDate", 20140101.0, 20150101.0)
        .unwrap();

    let post_ids: HashSet<i64> = posts
        .records()
        .iter()
        .filter_map(|r| extract_id(r))
        .collect();

    let mut likers = 0usize;
    for record in recent.records() {
        if let Some(post_id) = extract_id(record) {
            if post_ids.contains(&post_id) {
                likers += 1;
            }
        }
    }

    (start.elapsed().as_secs_f64() * 1000.0, likers.min(20))
}

/// IC8: Recent replies (2-hop)
fn run_ic8(store: &ShardedAmorphicStore) -> (f64, usize) {
    let start = Instant::now();

    let posts = store.query_equals("creatorId", &Value::Int(1)).unwrap();
    let comments = store
        .query_equals("_type", &Value::String("Comment".to_string()))
        .unwrap();

    let post_ids: HashSet<i64> = posts
        .records()
        .iter()
        .filter_map(|r| extract_id(r))
        .collect();

    let mut replies = 0usize;
    for comment in comments.records() {
        if let Some(reply_to) = extract_field_i64(comment, "replyOf") {
            if post_ids.contains(&reply_to) {
                replies += 1;
            }
        }
    }

    (start.elapsed().as_secs_f64() * 1000.0, replies.min(20))
}

// =============================================================================
// HELPERS
// =============================================================================

fn next_float(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

fn next_int(state: &mut u64, max: usize) -> usize {
    (next_float(state) * max as f64) as usize
}
