//! `joule-db-recover` — forensic page-walk recovery tool.
//!
//! Walks a JouleDB `data.wdb` directly (bypassing Engine), classifies
//! pages by type, builds an adjacency graph from internal nodes, and
//! locates **candidate root pages** — pages NOT referenced as a child
//! by any other internal node. For each candidate, BFS measures the
//! reachable subtree size (internal + leaf pages).
//!
//! Use case: a corrupted database whose `meta.wdb` points at a tiny
//! tree, but whose 594GB data file still contains the orphaned
//! original tree's pages. The original tree's root is the candidate
//! with the largest reachable subtree.
//!
//! ## Modes
//!
//! - **`scan` (default)** — classify pages, find candidate roots,
//!   print a report. Read-only; safe to run on a live database (the
//!   server's writes are CoW-additive so the scan sees a consistent
//!   point-in-time view).
//! - **`scan --write-recovery-meta`** — also rewrite `meta.wdb`
//!   atomically (tmp + rename) to point at the largest candidate
//!   root. Subsequent `Database::open` calls see this root.
//!   **DESTRUCTIVE** — the existing `meta.wdb` is preserved as
//!   `meta.wdb.pre-recover-<timestamp>`.
//!
//! ## Memory + time
//!
//! ~50 bytes per scanned page metadata + ~12 bytes per internal-node
//! child pointer. For 594GB / 64KB pages = ~9M pages with ~1%
//! internal × 256 children = ~30M ptrs × 12 bytes ≈ 360MB peak.
//! Sequential read at ~1GB/s = ~10 minutes scan time.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PAGE_MAGIC: u32 = 0x57444250; // "WDBP"
const PAGE_HEADER_SIZE: usize = 32;
const PAGE_SIZE: usize = 65536;
const META_V1_MAGIC: [u8; 4] = *b"JDBM";
const META_V1_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum PageType {
    Free = 0,
    BTreeInternal = 1,
    BTreeLeaf = 2,
    Overflow = 3,
    Other = 255,
}

impl PageType {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => PageType::Free,
            1 => PageType::BTreeInternal,
            2 => PageType::BTreeLeaf,
            3 => PageType::Overflow,
            _ => PageType::Other,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PageHeader {
    page_type: PageType,
    page_id: u64,
    data_len: u32,
}

fn parse_header(buf: &[u8]) -> Option<PageHeader> {
    if buf.len() < PAGE_HEADER_SIZE {
        return None;
    }
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != PAGE_MAGIC {
        return None;
    }
    let page_type = PageType::from_byte(buf[5]);
    let page_id =
        u64::from_le_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]);
    let data_len = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    Some(PageHeader {
        page_type,
        page_id,
        data_len,
    })
}

/// Parse a `BTreeInternal` page body.
///
/// Body format (from `BTreeNode::serialize` in joule-db-core):
///   is_leaf (1 byte) = 0
///   num_keys (4 bytes LE)
///   keys: [{key_len(4 LE), key bytes}]*
///   num_children (4 bytes LE)
///   children: [u64 LE]*  (num_children elements)
///
/// Returns (first_key, children) on success. first_key is empty for
/// nodes with no separators (rare but possible at root with 1 child).
fn parse_internal(body: &[u8]) -> Option<(Vec<u8>, Vec<u64>)> {
    if body.is_empty() || body[0] != 0 {
        return None;
    }
    let mut cursor = 1;
    if cursor + 4 > body.len() {
        return None;
    }
    let num_keys = u32::from_le_bytes([
        body[cursor],
        body[cursor + 1],
        body[cursor + 2],
        body[cursor + 3],
    ]) as usize;
    cursor += 4;

    let mut first_key: Vec<u8> = Vec::new();
    for i in 0..num_keys {
        if cursor + 4 > body.len() {
            return None;
        }
        let key_len = u32::from_le_bytes([
            body[cursor],
            body[cursor + 1],
            body[cursor + 2],
            body[cursor + 3],
        ]) as usize;
        cursor += 4;
        if cursor + key_len > body.len() {
            return None;
        }
        if i == 0 {
            // Capture first key (truncated to 64 bytes for memory bounds).
            let take = key_len.min(64);
            first_key.extend_from_slice(&body[cursor..cursor + take]);
        }
        cursor += key_len;
    }

    if cursor + 4 > body.len() {
        return None;
    }
    let num_children = u32::from_le_bytes([
        body[cursor],
        body[cursor + 1],
        body[cursor + 2],
        body[cursor + 3],
    ]) as usize;
    cursor += 4;

    if num_children > 4096 {
        return None;
    }
    if cursor + num_children * 8 > body.len() {
        return None;
    }
    let mut children = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        let id = u64::from_le_bytes([
            body[cursor],
            body[cursor + 1],
            body[cursor + 2],
            body[cursor + 3],
            body[cursor + 4],
            body[cursor + 5],
            body[cursor + 6],
            body[cursor + 7],
        ]);
        children.push(id);
        cursor += 8;
    }

    Some((first_key, children))
}

/// Parse the FIRST and LAST keys of a `BTreeLeaf` page body.
///
/// Leaf body format:
///   is_leaf (1 byte) = 1
///   num_keys (4 bytes LE)
///   keys: [{key_len(4 LE), key bytes}]*
///   values: [{value_len(4 LE), value bytes}]*
///
/// Returns (first_key, last_key) each truncated to 64 bytes, or None
/// if not a leaf or empty. Both keys come from the same scan pass to
/// avoid double-parsing.
fn parse_leaf_first_last(body: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    if body.is_empty() || body[0] != 1 {
        return None;
    }
    let mut cursor = 1;
    if cursor + 4 > body.len() {
        return None;
    }
    let num_keys = u32::from_le_bytes([
        body[cursor],
        body[cursor + 1],
        body[cursor + 2],
        body[cursor + 3],
    ]) as usize;
    cursor += 4;
    if num_keys == 0 || num_keys > 100_000 {
        return None;
    }

    let mut first: Option<Vec<u8>> = None;
    let mut last: Option<Vec<u8>> = None;
    for i in 0..num_keys {
        if cursor + 4 > body.len() {
            return None;
        }
        let key_len = u32::from_le_bytes([
            body[cursor],
            body[cursor + 1],
            body[cursor + 2],
            body[cursor + 3],
        ]) as usize;
        cursor += 4;
        if cursor + key_len > body.len() {
            return None;
        }
        if i == 0 {
            let take = key_len.min(64);
            first = Some(body[cursor..cursor + take].to_vec());
        }
        if i == num_keys - 1 {
            let take = key_len.min(64);
            last = Some(body[cursor..cursor + take].to_vec());
        }
        cursor += key_len;
    }
    match (first, last) {
        (Some(f), Some(l)) => Some((f, l)),
        _ => None,
    }
}

#[derive(Default)]
struct ScanResult {
    total_pages: u64,
    valid_pages: u64,
    page_id_mismatches: u64,
    page_type_counts: HashMap<u8, u64>,
    /// internal_id -> [children]
    internal_children: HashMap<u64, Vec<u64>>,
    /// internal_id -> first separator key (= first entry of `keys`),
    /// used to detect what table-prefix range this internal covers.
    /// Truncated to ~64 bytes to bound memory.
    internal_first_key: HashMap<u64, Vec<u8>>,
    /// All valid page ids we encountered (so BFS can verify children
    /// actually correspond to existing pages).
    all_pages: HashSet<u64>,
    /// Leaf id -> first key (for finding subtree boundaries).
    /// Populated for every leaf; bounded to ~64 bytes.
    leaf_first_key: HashMap<u64, Vec<u8>>,
    /// Leaf id -> last key (for determining subtree range coverage).
    leaf_last_key: HashMap<u64, Vec<u8>>,
    max_page_id: u64,
}

fn scan_data_file(path: &Path) -> std::io::Result<ScanResult> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let total_pages = file_size / PAGE_SIZE as u64;

    eprintln!(
        "scanning {} ({:.2} GB, {} pages of {} bytes)",
        path.display(),
        file_size as f64 / 1024.0 / 1024.0 / 1024.0,
        total_pages,
        PAGE_SIZE
    );

    let mut result = ScanResult::default();
    result.total_pages = total_pages;

    let mut reader = std::io::BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buf = vec![0u8; PAGE_SIZE];
    let start = Instant::now();
    let mut last_progress_log = Instant::now();

    for page_offset in 0..total_pages {
        let read = reader.read(&mut buf)?;
        if read < PAGE_HEADER_SIZE {
            break;
        }
        // Skip remainder of buffer if file ends mid-page.
        if read < PAGE_SIZE {
            // Pad with zeros; header parse may still succeed but body reads
            // need to be careful.
            buf[read..].fill(0);
        }

        let header = match parse_header(&buf[..PAGE_HEADER_SIZE]) {
            Some(h) => h,
            None => continue, // bad magic / unwritten page
        };
        let expected_page_id = page_offset + 1;
        if header.page_id != expected_page_id {
            // header is internally consistent magic-wise but offset
            // mismatches. Treat as junk (corrupt or relocated page).
            result.page_id_mismatches += 1;
            continue;
        }
        result.valid_pages += 1;
        result.all_pages.insert(header.page_id);
        result.max_page_id = result.max_page_id.max(header.page_id);
        *result
            .page_type_counts
            .entry(header.page_type as u8)
            .or_insert(0) += 1;

        let body_end = PAGE_HEADER_SIZE
            + (header.data_len as usize).min(PAGE_SIZE - PAGE_HEADER_SIZE);
        if header.page_type == PageType::BTreeInternal {
            if let Some((first_key, children)) =
                parse_internal(&buf[PAGE_HEADER_SIZE..body_end])
            {
                result.internal_children.insert(header.page_id, children);
                if !first_key.is_empty() {
                    result.internal_first_key.insert(header.page_id, first_key);
                }
            }
        } else if header.page_type == PageType::BTreeLeaf {
            if let Some((first_key, last_key)) = parse_leaf_first_last(&buf[PAGE_HEADER_SIZE..body_end]) {
                result.leaf_first_key.insert(header.page_id, first_key);
                result.leaf_last_key.insert(header.page_id, last_key);
            }
        }

        if last_progress_log.elapsed().as_secs() >= 10 {
            let pct = (page_offset as f64 / total_pages as f64) * 100.0;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = page_offset as f64 / elapsed.max(0.001);
            eprintln!(
                "  scanned {} / {} pages ({:.1}%, {:.0} pages/s, {:.1}s elapsed)",
                page_offset, total_pages, pct, rate, elapsed
            );
            last_progress_log = Instant::now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "scan complete: {} valid / {} total pages in {:.1}s",
        result.valid_pages, total_pages, elapsed
    );

    Ok(result)
}

/// Find pages that are NOT referenced as a child by any internal
/// node. These are root candidates. There's typically only one true
/// root, but a corrupted database may have multiple orphaned trees;
/// each shows up as a candidate.
fn find_root_candidates(scan: &ScanResult) -> Vec<u64> {
    let mut referenced: HashSet<u64> = HashSet::new();
    for (_parent, children) in &scan.internal_children {
        for c in children {
            referenced.insert(*c);
        }
    }
    let mut roots: Vec<u64> = scan
        .internal_children
        .keys()
        .copied()
        .filter(|id| !referenced.contains(id))
        .collect();
    roots.sort();
    roots
}

#[derive(Debug, Default)]
struct RootStats {
    page_id: u64,
    reachable_internal: u64,
    reachable_leaf: u64,
    /// Pages referenced as children but not seen in scan (dangling).
    dangling_children: u64,
    /// Estimated key count: sum across reachable leaves (approximated
    /// as `leaf_count * 256` per typical BTreeNode capacity, since we
    /// don't parse leaf bodies during scan to save time).
    estimated_keys: u64,
}

/// **Shared-visited BFS.** With a corrupted database that has 1M+
/// orphaned candidate roots, measuring each root's full reachable
/// set independently is O(roots × tree_size) ≈ 10^13 operations.
/// We instead share a global visited set across all measurements
/// and process candidates in fanout-descending order — the original
/// tree's true root will be processed early (because it has the
/// widest fanout) and claim its full subtree. Subsequent candidates
/// that overlap get small "newly-reached" counts, which correctly
/// reflects them as inner pages of an already-claimed tree.
///
/// Total work: each page visited at most once. O(total_pages).
fn measure_all_shared(
    scan: &ScanResult,
    candidates: &[u64],
) -> Vec<RootStats> {
    // Sort by direct fanout descending. Heuristic: the original tree's
    // root has the widest fanout (root pulls together the largest
    // number of subtree branches). Processing it first ensures its
    // claim isn't stolen by an inner internal page that happens to
    // also be a "candidate" by virtue of orphaning.
    let mut sorted: Vec<u64> = candidates.to_vec();
    sorted.sort_by_key(|id| {
        std::cmp::Reverse(
            scan.internal_children
                .get(id)
                .map(|c| c.len())
                .unwrap_or(0),
        )
    });

    let mut visited: HashSet<u64> = HashSet::with_capacity(scan.valid_pages as usize);
    let mut stats: Vec<RootStats> = Vec::with_capacity(sorted.len());

    for root in sorted {
        let mut s = RootStats {
            page_id: root,
            ..Default::default()
        };
        let mut frontier: Vec<u64> = vec![root];
        while let Some(id) = frontier.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(children) = scan.internal_children.get(&id) {
                s.reachable_internal += 1;
                for c in children {
                    if !visited.contains(c) {
                        frontier.push(*c);
                    }
                }
            } else if scan.all_pages.contains(&id) {
                s.reachable_leaf += 1;
            } else {
                s.dangling_children += 1;
            }
        }
        s.estimated_keys = s.reachable_leaf * 256;
        // Skip candidates that ended up reaching nothing — they
        // were inside a previously-claimed tree.
        if s.reachable_internal + s.reachable_leaf > 0 {
            stats.push(s);
        }
    }

    stats.sort_by(|a, b| b.reachable_leaf.cmp(&a.reachable_leaf));
    stats
}

/// Find the first leaf reachable from a root and return its first key.
/// Used to identify which table prefix a subtree covers.
fn find_subtree_first_key(scan: &ScanResult, root: u64) -> Option<Vec<u8>> {
    let mut current = root;
    let mut visited: HashSet<u64> = HashSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        if let Some(k) = scan.leaf_first_key.get(&current) {
            return Some(k.clone());
        }
        if let Some(children) = scan.internal_children.get(&current) {
            if let Some(first) = children.first() {
                current = *first;
                continue;
            }
        }
        return None;
    }
}

/// Symmetric to find_subtree_first_key: rightmost descent + last key
/// of the rightmost leaf. Tells us where the subtree's range ends.
fn find_subtree_last_key(scan: &ScanResult, root: u64) -> Option<Vec<u8>> {
    let mut current = root;
    let mut visited: HashSet<u64> = HashSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        if let Some(k) = scan.leaf_last_key.get(&current) {
            return Some(k.clone());
        }
        if let Some(children) = scan.internal_children.get(&current) {
            if let Some(last) = children.last() {
                current = *last;
                continue;
            }
        }
        return None;
    }
}

/// Walk a candidate subtree and bucket leaf-first-keys by table prefix.
/// Returns (HashMap<prefix, leaf_count>, leaf_total). Fresh DFS per
/// candidate (NOT shared visited) — gives accurate coverage. Bounded
/// at `max_pages` to prevent runaway on cycles.
fn coverage_per_root(
    scan: &ScanResult,
    root: u64,
    max_pages: usize,
) -> (HashMap<Vec<u8>, u64>, u64) {
    let mut frontier: Vec<u64> = vec![root];
    let mut visited: HashSet<u64> = HashSet::new();
    let mut by_prefix: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut leaf_total: u64 = 0;
    while let Some(id) = frontier.pop() {
        if visited.len() >= max_pages {
            break;
        }
        if !visited.insert(id) {
            continue;
        }
        if let Some(children) = scan.internal_children.get(&id) {
            for c in children {
                if !visited.contains(c) {
                    frontier.push(*c);
                }
            }
        } else if let Some(first_key) = scan.leaf_first_key.get(&id) {
            leaf_total += 1;
            // Bucket by table prefix; non-row keys go to "<other>".
            let bucket = extract_table_prefix(first_key)
                .unwrap_or_else(|| b"<other>".to_vec());
            *by_prefix.entry(bucket).or_insert(0) += 1;
        }
    }
    (by_prefix, leaf_total)
}

/// Extract the table prefix (e.g. `b"row::scholar_works\x00"`) from a
/// row key. Returns None if the key isn't in the `row::<table>\x00<pk>`
/// shape.
fn extract_table_prefix(key: &[u8]) -> Option<Vec<u8>> {
    if !key.starts_with(b"row::") {
        return None;
    }
    let after = &key[5..];
    let nul = after.iter().position(|&b| b == 0)?;
    let mut prefix = Vec::with_capacity(5 + nul + 1);
    prefix.extend_from_slice(b"row::");
    prefix.extend_from_slice(&after[..nul]);
    prefix.push(0);
    Some(prefix)
}

#[derive(Debug, Clone)]
struct PerTablePick {
    prefix: Vec<u8>,
    root: u64,
    first_key: Vec<u8>,
    reachable_internal: u64,
    reachable_leaf: u64,
}

/// Group candidate roots by the table prefix of their first leaf's
/// first key. For each table, pick the candidate with the most
/// reachable leaves.
fn pick_per_table(scan: &ScanResult, root_stats: &[RootStats]) -> Vec<PerTablePick> {
    let mut by_prefix: HashMap<Vec<u8>, PerTablePick> = HashMap::new();
    for s in root_stats {
        let first_key = match find_subtree_first_key(scan, s.page_id) {
            Some(k) => k,
            None => continue,
        };
        let prefix = match extract_table_prefix(&first_key) {
            Some(p) => p,
            None => continue,
        };
        let candidate = PerTablePick {
            prefix: prefix.clone(),
            root: s.page_id,
            first_key,
            reachable_internal: s.reachable_internal,
            reachable_leaf: s.reachable_leaf,
        };
        match by_prefix.get(&prefix) {
            Some(existing) if existing.reachable_leaf >= candidate.reachable_leaf => {
                // existing is bigger; keep it
            }
            _ => {
                by_prefix.insert(prefix, candidate);
            }
        }
    }
    let mut picks: Vec<PerTablePick> = by_prefix.into_values().collect();
    // Sort by table prefix bytes — gives natural alphabetic order
    // (scholar_authorities < scholar_authorship < ...) which is the
    // order the synthetic root needs its children in.
    picks.sort_by(|a, b| a.prefix.cmp(&b.prefix));
    picks
}

/// Encode a synthetic B-tree internal node body that points to one
/// child per per-table pick. Separator keys are the FIRST KEY of each
/// subsequent child — the engine's `find_key_index` will route lookups
/// for prefix P into the correct child as long as P sorts within that
/// child's covered range.
///
/// Body format matches `BTreeNode::serialize` for internal nodes:
///   is_leaf (1 byte) = 0
///   num_keys (u32 LE)
///   keys: [{key_len (u32 LE), key bytes}]*
///   num_children (u32 LE)
///   children: [u64 LE]*
fn encode_synthetic_internal(picks: &[PerTablePick]) -> Vec<u8> {
    let mut body = Vec::with_capacity(64 * picks.len());
    body.push(0u8); // is_leaf = false
    // num_keys = num_children - 1 separators
    let num_keys = (picks.len() - 1) as u32;
    body.extend_from_slice(&num_keys.to_le_bytes());

    // Separators: pick i is preceded by separator = picks[i].first_key
    // for i >= 1. So num_keys separators total.
    for i in 1..picks.len() {
        let sep = &picks[i].first_key;
        body.extend_from_slice(&(sep.len() as u32).to_le_bytes());
        body.extend_from_slice(sep);
    }

    body.extend_from_slice(&(picks.len() as u32).to_le_bytes());
    for p in picks {
        body.extend_from_slice(&p.root.to_le_bytes());
    }
    body
}

/// Wrap the synthetic body with a valid PageHeader and CRC32. Output
/// is exactly PAGE_SIZE bytes, ready to write to data.wdb at offset
/// `(page_id - 1) * PAGE_SIZE`.
fn encode_synthetic_page(page_id: u64, body: &[u8]) -> Vec<u8> {
    let mut page = vec![0u8; PAGE_SIZE];
    // Header: magic (4) + version (1) + page_type (1=BTreeInternal) +
    //         flags (2 zeros) + page_id (8) + data_len (4) + checksum (4)
    page[0..4].copy_from_slice(&PAGE_MAGIC.to_le_bytes());
    page[4] = 1; // version
    page[5] = PageType::BTreeInternal as u8;
    page[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags
    page[8..16].copy_from_slice(&page_id.to_le_bytes());
    page[16..20].copy_from_slice(&(body.len() as u32).to_le_bytes());
    let checksum = crc32_meta(body);
    page[20..24].copy_from_slice(&checksum.to_le_bytes());
    // Body
    page[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + body.len()].copy_from_slice(body);
    page
}

fn write_synthetic_recovery(
    db_dir: &Path,
    scan: &ScanResult,
    picks: &[PerTablePick],
) -> std::io::Result<()> {
    if picks.is_empty() {
        eprintln!("no per-table picks — aborting");
        std::process::exit(1);
    }

    // Allocate a fresh page id beyond max_page_id seen.
    let synthetic_root_id = scan.max_page_id + 1;

    // Encode + sanity check.
    let body = encode_synthetic_internal(picks);
    let body_capacity = PAGE_SIZE - PAGE_HEADER_SIZE;
    if body.len() > body_capacity {
        eprintln!(
            "synthetic body ({} bytes) exceeds page body capacity ({} bytes); aborting",
            body.len(),
            body_capacity
        );
        std::process::exit(1);
    }
    let page_bytes = encode_synthetic_page(synthetic_root_id, &body);

    eprintln!(
        "writing synthetic root at page {} pointing at {} subtrees",
        synthetic_root_id,
        picks.len()
    );
    for (i, p) in picks.iter().enumerate() {
        let prefix_str = String::from_utf8_lossy(&p.prefix);
        eprintln!(
            "  child[{}] root={} leaf-count={} prefix={}",
            i, p.root, p.reachable_leaf, prefix_str
        );
    }

    // Write the page to data.wdb.
    let data_path = db_dir.join("data.wdb");
    let mut data_file = OpenOptions::new().write(true).open(&data_path)?;
    let offset = (synthetic_root_id - 1) * PAGE_SIZE as u64;
    data_file.seek(SeekFrom::Start(offset))?;
    data_file.write_all(&page_bytes)?;
    data_file.sync_all()?;
    eprintln!("wrote synthetic page bytes to data.wdb at offset {}", offset);

    // Write meta.wdb pointing at the synthetic root.
    write_recovery_meta(db_dir, synthetic_root_id, synthetic_root_id + 1)?;

    Ok(())
}

fn write_recovery_meta(
    db_dir: &Path,
    committed_root: u64,
    next_page_id: u64,
) -> std::io::Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let meta_path = db_dir.join("meta.wdb");
    let backup_name = format!(
        "meta.wdb.pre-recover-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    let backup_path = db_dir.join(&backup_name);
    let tmp_path = db_dir.join(format!(
        "meta.wdb.recover.tmp.{}",
        std::process::id()
    ));

    // Backup existing meta first.
    if meta_path.exists() {
        std::fs::copy(&meta_path, &backup_path)?;
        eprintln!("backed up existing meta.wdb to {}", backup_path.display());
    }

    // Build the v1 record.
    // Layout: magic(4) + format_version(4) + committed_version(8) +
    //         committed_root(8) + next_page_id(8) + free_pages_count(4) +
    //         crc32(4)
    // We pick a high committed_version so any future writer's
    // monotonic-version check accepts new commits without complaining.
    // The existing meta's version + 1000 is a safe pad.
    let existing_version = read_existing_committed_version(&meta_path).unwrap_or(0);
    let new_version = existing_version.saturating_add(1000);

    let mut buf = Vec::with_capacity(40);
    buf.extend_from_slice(&META_V1_MAGIC);
    buf.extend_from_slice(&META_V1_FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&new_version.to_le_bytes());
    buf.extend_from_slice(&committed_root.to_le_bytes());
    buf.extend_from_slice(&next_page_id.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // free_pages_count = 0
    let crc = crc32_meta(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    // Atomic write: tmp + fsync + rename.
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp_path, &meta_path)?;
    eprintln!(
        "wrote new meta.wdb: committed_version={} committed_root={} next_page_id={}",
        new_version, committed_root, next_page_id
    );
    Ok(())
}

/// Walk the global leaf set and produce a non-overlapping, sorted
/// list of leaf page ids that, taken together, form a consistent
/// B-tree leaf level (no two leaves cover the same key).
///
/// Strategy: sort by (first_key asc, last_key desc, page_id desc).
/// Greedy include: skip any leaf whose first_key falls within the
/// already-covered range. Among ties on first_key, the one with the
/// largest last_key (widest range) wins; if same last_key, highest
/// page_id wins (most recent CoW snapshot).
///
/// Filtering: we only include leaves whose first_key matches
/// `__catalog__::` or `row::scholar_*` prefixes; pages whose first key
/// no longer parses as a known prefix (likely overwritten) are
/// dropped.
fn build_consistent_leaf_order(scan: &ScanResult) -> Vec<(u64, Vec<u8>, Vec<u8>)> {
    let mut leaves: Vec<(u64, Vec<u8>, Vec<u8>)> = scan
        .leaf_first_key
        .iter()
        .filter_map(|(id, fk)| {
            scan.leaf_last_key.get(id).map(|lk| (*id, fk.clone(), lk.clone()))
        })
        .filter(|(_, fk, _)| {
            fk.starts_with(b"__catalog__::") || fk.starts_with(b"row::")
        })
        // Reject leaves whose [first..last] range crosses table or
        // catalog/row boundaries. Such leaves are CoW fragments where
        // the on-disk page was partially overwritten with a different
        // tree's data — including them in dedup would gobble up
        // adjacent tables. A real leaf has first and last in the same
        // logical bucket.
        .filter(|(_, fk, lk)| {
            fn bucket(k: &[u8]) -> Vec<u8> {
                if k.starts_with(b"__catalog__::") {
                    b"__catalog__".to_vec()
                } else if let Some(p) = extract_table_prefix(k) {
                    p
                } else {
                    k.to_vec()
                }
            }
            bucket(fk) == bucket(lk)
        })
        .collect();

    leaves.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(b.2.cmp(&a.2))
            .then(b.0.cmp(&a.0))
    });

    let mut out: Vec<(u64, Vec<u8>, Vec<u8>)> = Vec::with_capacity(leaves.len());
    let mut prev_last: Option<Vec<u8>> = None;
    for (id, fk, lk) in leaves {
        if let Some(p) = &prev_last {
            if fk.as_slice() <= p.as_slice() {
                continue;
            }
        }
        prev_last = Some(lk.clone());
        out.push((id, fk, lk));
    }
    out
}

const REBUILD_FANOUT: usize = 200;
const REBUILD_BODY_BUDGET: usize = PAGE_SIZE - PAGE_HEADER_SIZE - 256;

/// Encode an internal-node body referencing the given children with
/// separator keys. Returns the body bytes if the encoding fits within
/// `REBUILD_BODY_BUDGET`, or None if it overflows.
///
/// `children` is a slice of (first_key, page_id) pairs, sorted by
/// first_key. Separators between adjacent children = first_key of the
/// right neighbor.
fn try_encode_internal_body(children: &[(Vec<u8>, u64)]) -> Option<Vec<u8>> {
    if children.is_empty() {
        return None;
    }
    let n = children.len();
    let num_keys = (n - 1) as u32;
    let mut body = Vec::with_capacity(64 * n);
    body.push(0u8);
    body.extend_from_slice(&num_keys.to_le_bytes());
    for i in 1..n {
        let sep = &children[i].0;
        body.extend_from_slice(&(sep.len() as u32).to_le_bytes());
        body.extend_from_slice(sep);
    }
    body.extend_from_slice(&(n as u32).to_le_bytes());
    for (_, id) in children {
        body.extend_from_slice(&id.to_le_bytes());
    }
    if body.len() > REBUILD_BODY_BUDGET {
        return None;
    }
    Some(body)
}

/// Bulk-build a B-tree on top of an ordered, non-overlapping leaf set.
/// Returns (new_root_id, new_pages_to_write).
///
/// `leaves` is (page_id, first_key) sorted by first_key. `next_page_id`
/// is the first fresh page id we may allocate; all generated internal
/// pages take ids >= next_page_id.
///
/// We pack greedily: walk forward consuming children until adding the
/// next would overflow the page body budget, emit an internal page,
/// then continue. This adapts to varying key sizes.
fn build_btree_bulk(
    leaves: Vec<(Vec<u8>, u64)>,
    next_page_id: u64,
) -> (u64, Vec<(u64, Vec<u8>)>) {
    let mut next_id = next_page_id;
    let mut new_pages: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut current = leaves;

    while current.len() > 1 {
        let mut next_level: Vec<(Vec<u8>, u64)> = Vec::new();
        let mut i = 0;
        while i < current.len() {
            let mut chunk_end = (i + REBUILD_FANOUT).min(current.len());
            // Try the planned chunk; if it overflows, shrink.
            loop {
                let chunk: Vec<(Vec<u8>, u64)> = current[i..chunk_end].to_vec();
                if let Some(body) = try_encode_internal_body(&chunk) {
                    let pid = next_id;
                    next_id += 1;
                    let first_key = chunk[0].0.clone();
                    new_pages.push((pid, body));
                    next_level.push((first_key, pid));
                    i = chunk_end;
                    break;
                }
                if chunk_end - i <= 2 {
                    panic!(
                        "rebuild: cannot fit even 2 children into a page; \
                         keys must be smaller than {} bytes total",
                        REBUILD_BODY_BUDGET
                    );
                }
                chunk_end = i + (chunk_end - i) / 2;
            }
        }
        current = next_level;
    }

    let root_id = current[0].1;
    (root_id, new_pages)
}

fn encode_internal_page(page_id: u64, body: &[u8]) -> Vec<u8> {
    encode_synthetic_page(page_id, body)
}

/// Write the rebuilt B-tree to data.wdb and update meta.wdb.
fn write_rebuild(
    db_dir: &Path,
    scan: &ScanResult,
) -> std::io::Result<()> {
    eprintln!("==> bulk B-tree rebuild from leaves");
    eprintln!("    scanned leaves total: {}", scan.leaf_first_key.len());

    let included = build_consistent_leaf_order(scan);
    eprintln!(
        "    after dedup-overlap: {} leaves included (skipped {})",
        included.len(),
        scan.leaf_first_key.len() as i64 - included.len() as i64
    );

    // Histogram included leaves by table prefix.
    let mut by_prefix: HashMap<Vec<u8>, u64> = HashMap::new();
    for (_, fk, _) in &included {
        let bucket = extract_table_prefix(fk).unwrap_or_else(|| b"<other>".to_vec());
        *by_prefix.entry(bucket).or_insert(0) += 1;
    }
    let mut entries: Vec<(Vec<u8>, u64)> = by_prefix.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("    included leaf-prefix histogram:");
    for (prefix, count) in &entries {
        eprintln!("      {:>10}  {}", count, String::from_utf8_lossy(prefix));
    }

    if included.is_empty() {
        eprintln!("    nothing to rebuild — aborting");
        std::process::exit(1);
    }

    let leaf_pairs: Vec<(Vec<u8>, u64)> = included
        .into_iter()
        .map(|(id, fk, _lk)| (fk, id))
        .collect();

    let next_page_id = scan.max_page_id + 1;
    eprintln!("    allocating new internal pages starting at {}", next_page_id);
    let (root_id, new_pages) = build_btree_bulk(leaf_pairs, next_page_id);
    eprintln!(
        "    rebuild produced {} new internal pages; new root = {}",
        new_pages.len(),
        root_id
    );

    // Write the new pages.
    let data_path = db_dir.join("data.wdb");
    let mut data_file = OpenOptions::new().write(true).open(&data_path)?;
    let mut written = 0u64;
    for (pid, body) in &new_pages {
        let page_bytes = encode_internal_page(*pid, body);
        let offset = (*pid - 1) * PAGE_SIZE as u64;
        data_file.seek(SeekFrom::Start(offset))?;
        data_file.write_all(&page_bytes)?;
        written += 1;
        if written % 1024 == 0 {
            eprintln!("      wrote {} / {} pages", written, new_pages.len());
        }
    }
    data_file.sync_all()?;
    eprintln!("    wrote {} pages, fsynced", new_pages.len());

    // Update meta to point at the new root.
    let new_next_page_id = next_page_id + new_pages.len() as u64;
    write_recovery_meta(db_dir, root_id, new_next_page_id)?;

    Ok(())
}

fn read_existing_committed_version(meta_path: &Path) -> Option<u64> {
    let mut f = File::open(meta_path).ok()?;
    let mut buf = [0u8; 16];
    f.seek(SeekFrom::Start(0)).ok()?;
    f.read_exact(&mut buf).ok()?;
    if buf[0..4] != META_V1_MAGIC {
        return None;
    }
    let version = u64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);
    Some(version)
}

fn crc32_meta(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn print_report(scan: &ScanResult, root_stats: &[RootStats]) {
    println!();
    println!("=== joule-db-recover scan report ===");
    println!();
    println!("file pages:            {:>12}", scan.total_pages);
    println!("valid pages:           {:>12}", scan.valid_pages);
    println!("invalid (bad magic):   {:>12}", scan.total_pages - scan.valid_pages - scan.page_id_mismatches);
    println!("offset/id mismatches:  {:>12}", scan.page_id_mismatches);
    println!("max page id seen:      {:>12}", scan.max_page_id);
    println!();
    println!("page type breakdown:");
    let pt_names: HashMap<u8, &str> = [
        (0, "Free"),
        (1, "BTreeInternal"),
        (2, "BTreeLeaf"),
        (3, "Overflow"),
        (4, "Metadata"),
        (5, "Free-list"),
        (6, "ExtentHeader"),
        (7, "ExtentData"),
    ]
    .iter()
    .copied()
    .collect();
    for (pt, count) in &scan.page_type_counts {
        let name = pt_names.get(pt).copied().unwrap_or("Unknown");
        println!("  {:<16} {:>12}", name, count);
    }
    println!();
    println!("internal nodes parsed: {:>12}", scan.internal_children.len());
    println!();
    println!("=== root candidates (top 20 by reachable subtree size) ===");
    println!(
        "{:>12}  {:>14}  {:>14}  {:>14}  {:>14}",
        "page_id", "internal", "leaf", "est_keys", "dangling"
    );
    for s in root_stats.iter().take(20) {
        println!(
            "{:>12}  {:>14}  {:>14}  {:>14}  {:>14}",
            s.page_id,
            s.reachable_internal,
            s.reachable_leaf,
            s.estimated_keys,
            s.dangling_children
        );
    }
}

fn printable(k: &[u8], take: usize) -> String {
    let take = k.len().min(take);
    let mut out = String::new();
    for &b in &k[..take] {
        if b.is_ascii_graphic() || b == b' ' {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\x{:02x}", b));
        }
    }
    out
}

fn print_top_candidates_with_range(scan: &ScanResult, root_stats: &[RootStats], n: usize) {
    println!();
    println!("=== top {} candidates with [first..last] leaf range ===", n);
    println!(
        "{:>12}  {:>14}  {:>14}  {}",
        "root", "internal", "leaf", "[first_leaf_key  ..  last_leaf_key]"
    );
    for s in root_stats.iter().take(n) {
        let first = find_subtree_first_key(scan, s.page_id);
        let last = find_subtree_last_key(scan, s.page_id);
        let f = first.as_ref().map(|k| printable(k, 64)).unwrap_or_else(|| "<none>".into());
        let l = last.as_ref().map(|k| printable(k, 64)).unwrap_or_else(|| "<none>".into());
        println!(
            "{:>12}  {:>14}  {:>14}  [{}  ..  {}]",
            s.page_id, s.reachable_internal, s.reachable_leaf, f, l
        );
    }
}

fn print_global_leaf_prefix_histogram(scan: &ScanResult) {
    println!();
    println!("=== global leaf-prefix histogram (count of distinct leaf pages per prefix) ===");
    let mut by_prefix: HashMap<Vec<u8>, u64> = HashMap::new();
    for k in scan.leaf_first_key.values() {
        let bucket = extract_table_prefix(k).unwrap_or_else(|| b"<other>".to_vec());
        *by_prefix.entry(bucket).or_insert(0) += 1;
    }
    let mut entries: Vec<(Vec<u8>, u64)> = by_prefix.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    for (prefix, count) in entries {
        let prefix_str = String::from_utf8_lossy(&prefix);
        println!("  {:>12}  {}", count, prefix_str);
    }
    println!();
    println!("(NB: these are RAW leaf-page counts. CoW snapshots can produce many copies of");
    println!(" the same logical row across distinct leaf pages, so this is an upper bound.)");
}

fn print_coverage_for_top_candidates(
    scan: &ScanResult,
    root_stats: &[RootStats],
    n: usize,
) {
    println!();
    println!(
        "=== leaf coverage by table-prefix for top {} candidates ===",
        n
    );
    for s in root_stats.iter().take(n) {
        let (by_prefix, leaf_total) =
            coverage_per_root(scan, s.page_id, scan.valid_pages as usize);
        if leaf_total == 0 {
            continue;
        }
        println!(
            "  root={} (internal={}, leaf={}):",
            s.page_id, s.reachable_internal, s.reachable_leaf
        );
        let mut entries: Vec<(Vec<u8>, u64)> = by_prefix.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        for (prefix, count) in entries.iter().take(15) {
            let prefix_str = String::from_utf8_lossy(prefix);
            println!("    {:>10}  {}", count, prefix_str);
        }
    }
}

fn print_per_table_picks(picks: &[PerTablePick]) {
    println!();
    println!("=== per-table subtree picks ===");
    println!(
        "{:>12}  {:>14}  {:>14}  {}",
        "root", "internal", "leaf", "table_prefix"
    );
    for p in picks {
        let prefix_str = String::from_utf8_lossy(&p.prefix);
        println!(
            "{:>12}  {:>14}  {:>14}  {}",
            p.root, p.reachable_internal, p.reachable_leaf, prefix_str
        );
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut db_dir: Option<PathBuf> = None;
    let mut write_meta = false;
    let mut write_synthetic = false;
    let mut rebuild_tree = false;
    let mut explicit_root: Option<u64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--write-recovery-meta" => write_meta = true,
            "--write-synthetic-recovery" => write_synthetic = true,
            "--rebuild-tree" => rebuild_tree = true,
            "--root" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--root requires a page id argument");
                    std::process::exit(2);
                }
                match args[i].parse::<u64>() {
                    Ok(v) => explicit_root = Some(v),
                    Err(e) => {
                        eprintln!("invalid --root page id '{}': {}", args[i], e);
                        std::process::exit(2);
                    }
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "joule-db-recover <db_dir> \\\n  \
                       [--write-recovery-meta [--root <id>] | \\\n   \
                        --write-synthetic-recovery | --rebuild-tree]\n\n\
                     Walks data.wdb directly to find candidate root pages by reachable\n\
                     subtree size. Prints a report.\n\n\
                     --write-recovery-meta [--root <id>]:\n\
                       atomically rewrites meta.wdb to point at the SINGLE largest\n\
                       candidate root, or the explicit page id passed via --root.\n\n\
                     --write-synthetic-recovery:\n\
                       allocates a fresh page id and writes a synthetic B-tree internal\n\
                       page that points at the largest reachable subtree per table\n\
                       prefix; rewrites meta.wdb to point at the synthetic page.\n\n\
                     --rebuild-tree:\n\
                       most invasive recovery. Sorts ALL leaf pages by first key,\n\
                       deduplicates overlapping ranges (skip-on-overlap, recent-wins\n\
                       tie-break), then bulk-builds a fresh internal-node tree on\n\
                       top of the consistent leaf set. Allocates fresh page ids for\n\
                       all new internals (~L/200 pages where L = surviving leaves)\n\
                       and points meta.wdb at the new root. Use when no single\n\
                       existing root reaches the desired set of tables.\n\n\
                     The previous meta.wdb is preserved as meta.wdb.pre-recover-<unix_ts>\n\
                     and the new tmp goes through tmp + fsync + rename for atomicity."
                );
                std::process::exit(0);
            }
            arg if !arg.starts_with("--") && db_dir.is_none() => {
                db_dir = Some(PathBuf::from(arg));
            }
            other => {
                eprintln!("unknown argument: {}", other);
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let mode_count = [write_meta, write_synthetic, rebuild_tree]
        .iter()
        .filter(|b| **b)
        .count();
    if mode_count > 1 {
        eprintln!(
            "error: --write-recovery-meta, --write-synthetic-recovery, and --rebuild-tree are mutually exclusive"
        );
        std::process::exit(2);
    }

    let db_dir = db_dir.unwrap_or_else(|| {
        eprintln!("usage: joule-db-recover <db_dir> [--write-recovery-meta | --write-synthetic-recovery]");
        std::process::exit(2);
    });
    let data_path = db_dir.join("data.wdb");
    if !data_path.exists() {
        eprintln!("error: {} does not exist", data_path.display());
        std::process::exit(1);
    }

    let scan = scan_data_file(&data_path)?;
    let candidate_roots = find_root_candidates(&scan);
    eprintln!("found {} candidate root pages", candidate_roots.len());

    let measure_start = Instant::now();
    let root_stats: Vec<RootStats> = measure_all_shared(&scan, &candidate_roots);
    eprintln!(
        "measure complete: {} non-empty candidate trees in {:.1}s",
        root_stats.len(),
        measure_start.elapsed().as_secs_f64()
    );

    print_report(&scan, &root_stats);
    print_global_leaf_prefix_histogram(&scan);
    print_top_candidates_with_range(&scan, &root_stats, 30);
    print_coverage_for_top_candidates(&scan, &root_stats, 12);

    let picks = pick_per_table(&scan, &root_stats);
    print_per_table_picks(&picks);

    if rebuild_tree {
        write_rebuild(&db_dir, &scan)?;
    } else if write_synthetic {
        if picks.is_empty() {
            eprintln!("no per-table picks found — aborting");
            std::process::exit(1);
        }
        eprintln!();
        eprintln!(
            "==> writing synthetic recovery root with {} per-table subtrees",
            picks.len()
        );
        write_synthetic_recovery(&db_dir, &scan, &picks)?;
    } else if write_meta {
        let chosen_root = if let Some(r) = explicit_root {
            r
        } else {
            match root_stats.first() {
                Some(b) => b.page_id,
                None => {
                    eprintln!("no root candidates found — aborting");
                    std::process::exit(1);
                }
            }
        };
        let leaves = root_stats
            .iter()
            .find(|s| s.page_id == chosen_root)
            .map(|s| s.reachable_leaf)
            .unwrap_or(0);
        eprintln!();
        eprintln!(
            "==> writing recovery meta pointing at root {} ({} reachable leaves)",
            chosen_root, leaves
        );
        write_recovery_meta(&db_dir, chosen_root, scan.max_page_id + 1)?;
    } else {
        println!();
        println!(
            "(scan-only. Re-run with --write-recovery-meta for single-root recovery,\n\
             or --write-synthetic-recovery to stitch all per-table subtrees together.)"
        );
    }

    Ok(())
}
