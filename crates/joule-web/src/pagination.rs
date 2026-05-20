// pagination.rs — Pagination: offset/limit with bounds, cursor-based
// with encode/decode, keyset pagination, page metadata, Link header generation.

/// Offset/limit pagination with bounds enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetLimit {
    pub offset: u64,
    pub limit: u64,
}

impl OffsetLimit {
    /// Create with clamped limit (1..=max_limit).
    pub fn new(offset: u64, limit: u64, max_limit: u64) -> Self {
        let clamped = if limit == 0 { 1 } else { limit.min(max_limit) };
        Self {
            offset,
            limit: clamped,
        }
    }

    /// The page number (1-indexed) implied by offset/limit.
    pub fn page_number(&self) -> u64 {
        (self.offset / self.limit) + 1
    }

    /// Compute the offset for the next page.
    pub fn next_offset(&self) -> u64 {
        self.offset + self.limit
    }

    /// Compute the offset for the previous page (saturating at 0).
    pub fn prev_offset(&self) -> u64 {
        self.offset.saturating_sub(self.limit)
    }
}

// ---------------------------------------------------------------------------
// Cursor-based pagination
// ---------------------------------------------------------------------------

/// Encode a cursor value to a URL-safe base64-like string.
/// Uses a simple reversible encoding: hex of bytes.
pub fn encode_cursor(value: &str) -> String {
    let mut out = String::with_capacity(value.len() * 2);
    for &b in value.as_bytes() {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Decode a cursor string back to the original value.
pub fn decode_cursor(encoded: &str) -> Option<String> {
    let bytes = encoded.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks(2) {
        let hex_str = std::str::from_utf8(chunk).ok()?;
        let b = u8::from_str_radix(hex_str, 16).ok()?;
        out.push(b);
    }
    String::from_utf8(out).ok()
}

/// Cursor-based page request.
#[derive(Debug, Clone)]
pub struct CursorPage {
    /// Opaque cursor (None = first page).
    pub after: Option<String>,
    /// Number of items per page.
    pub first: u64,
}

impl CursorPage {
    pub fn first_page(size: u64) -> Self {
        Self {
            after: None,
            first: size,
        }
    }

    pub fn after(cursor: &str, size: u64) -> Self {
        Self {
            after: Some(cursor.to_string()),
            first: size,
        }
    }
}

/// Edge in a cursor-paginated result.
#[derive(Debug, Clone)]
pub struct Edge<T> {
    pub node: T,
    pub cursor: String,
}

/// Cursor-paginated result.
#[derive(Debug, Clone)]
pub struct CursorResult<T> {
    pub edges: Vec<Edge<T>>,
    pub has_next_page: bool,
    pub has_previous_page: bool,
}

impl<T> CursorResult<T> {
    pub fn start_cursor(&self) -> Option<&str> {
        self.edges.first().map(|e| e.cursor.as_str())
    }

    pub fn end_cursor(&self) -> Option<&str> {
        self.edges.last().map(|e| e.cursor.as_str())
    }

    pub fn node_count(&self) -> usize {
        self.edges.len()
    }
}

/// Build a CursorResult from a slice of items and a cursor-maker function.
pub fn paginate_cursor<T: Clone, F>(
    items: &[T],
    page: &CursorPage,
    make_cursor: F,
) -> CursorResult<T>
where
    F: Fn(&T, usize) -> String,
{
    let start = match &page.after {
        None => 0,
        Some(cursor) => {
            // Find the item matching the cursor, then start after it.
            items
                .iter()
                .enumerate()
                .find(|(i, item)| make_cursor(item, *i) == *cursor)
                .map(|(i, _)| i + 1)
                .unwrap_or(0)
        }
    };

    let limit = page.first as usize;
    let end = (start + limit).min(items.len());
    let has_next = end < items.len();
    let has_prev = start > 0;

    let edges: Vec<Edge<T>> = items[start..end]
        .iter()
        .enumerate()
        .map(|(rel_idx, item)| Edge {
            node: item.clone(),
            cursor: make_cursor(item, start + rel_idx),
        })
        .collect();

    CursorResult {
        edges,
        has_next_page: has_next,
        has_previous_page: has_prev,
    }
}

// ---------------------------------------------------------------------------
// Keyset pagination
// ---------------------------------------------------------------------------

/// Keyset pagination: resume after a specific key value.
#[derive(Debug, Clone)]
pub struct KeysetPage {
    /// Resume after this key (None = start from beginning).
    pub after_key: Option<String>,
    pub limit: u64,
}

impl KeysetPage {
    pub fn first(limit: u64) -> Self {
        Self {
            after_key: None,
            limit,
        }
    }

    pub fn after(key: &str, limit: u64) -> Self {
        Self {
            after_key: Some(key.to_string()),
            limit,
        }
    }
}

/// Apply keyset pagination to a sorted slice using a key extractor.
pub fn paginate_keyset<T: Clone, F>(
    sorted_items: &[T],
    page: &KeysetPage,
    extract_key: F,
) -> (Vec<T>, Option<String>)
where
    F: Fn(&T) -> String,
{
    let start = match &page.after_key {
        None => 0,
        Some(key) => sorted_items
            .iter()
            .position(|item| extract_key(item) == *key)
            .map(|i| i + 1)
            .unwrap_or(0),
    };

    let limit = page.limit as usize;
    let end = (start + limit).min(sorted_items.len());
    let items: Vec<T> = sorted_items[start..end].to_vec();
    let next_key = items.last().map(|item| extract_key(item));

    (items, next_key)
}

// ---------------------------------------------------------------------------
// Page metadata
// ---------------------------------------------------------------------------

/// Metadata about a paginated result set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageMeta {
    pub page: u64,
    pub per_page: u64,
    pub total_items: u64,
    pub total_pages: u64,
    pub has_next: bool,
    pub has_prev: bool,
}

impl PageMeta {
    pub fn compute(page: u64, per_page: u64, total_items: u64) -> Self {
        let per = if per_page == 0 { 1 } else { per_page };
        let total_pages = (total_items + per - 1) / per;
        let p = page.max(1).min(total_pages.max(1));
        Self {
            page: p,
            per_page: per,
            total_items,
            total_pages,
            has_next: p < total_pages,
            has_prev: p > 1,
        }
    }

    pub fn offset(&self) -> u64 {
        (self.page - 1) * self.per_page
    }

    pub fn is_first_page(&self) -> bool {
        self.page == 1
    }

    pub fn is_last_page(&self) -> bool {
        self.page == self.total_pages || self.total_pages == 0
    }
}

// ---------------------------------------------------------------------------
// Link header generation (RFC 8288)
// ---------------------------------------------------------------------------

/// Generate a Link header value for pagination.
pub fn generate_link_header(base_url: &str, meta: &PageMeta) -> String {
    let mut links = Vec::new();

    // first
    links.push(format!(
        "<{base_url}?page=1&per_page={}>; rel=\"first\"",
        meta.per_page
    ));

    // last
    if meta.total_pages > 0 {
        links.push(format!(
            "<{base_url}?page={}&per_page={}>; rel=\"last\"",
            meta.total_pages, meta.per_page
        ));
    }

    // prev
    if meta.has_prev {
        links.push(format!(
            "<{base_url}?page={}&per_page={}>; rel=\"prev\"",
            meta.page - 1,
            meta.per_page
        ));
    }

    // next
    if meta.has_next {
        links.push(format!(
            "<{base_url}?page={}&per_page={}>; rel=\"next\"",
            meta.page + 1,
            meta.per_page
        ));
    }

    links.join(", ")
}

// ---------------------------------------------------------------------------
// Range helpers
// ---------------------------------------------------------------------------

/// Calculate which items to return from a slice, given page/per_page.
pub fn page_slice_bounds(page: u64, per_page: u64, total: usize) -> (usize, usize) {
    let p = page.max(1) as usize;
    let pp = per_page.max(1) as usize;
    let start = (p - 1) * pp;
    let end = (start + pp).min(total);
    if start >= total {
        (total, total)
    } else {
        (start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- OffsetLimit ----

    #[test]
    fn test_offset_limit_clamp() {
        let ol = OffsetLimit::new(0, 500, 100);
        assert_eq!(ol.limit, 100);
    }

    #[test]
    fn test_offset_limit_zero_becomes_one() {
        let ol = OffsetLimit::new(0, 0, 100);
        assert_eq!(ol.limit, 1);
    }

    #[test]
    fn test_offset_limit_page_number() {
        let ol = OffsetLimit::new(0, 10, 100);
        assert_eq!(ol.page_number(), 1);
        let ol2 = OffsetLimit::new(20, 10, 100);
        assert_eq!(ol2.page_number(), 3);
    }

    #[test]
    fn test_offset_limit_next_prev() {
        let ol = OffsetLimit::new(10, 10, 100);
        assert_eq!(ol.next_offset(), 20);
        assert_eq!(ol.prev_offset(), 0);
    }

    #[test]
    fn test_offset_limit_prev_saturate() {
        let ol = OffsetLimit::new(0, 10, 100);
        assert_eq!(ol.prev_offset(), 0);
    }

    // ---- Cursor encode/decode ----

    #[test]
    fn test_cursor_roundtrip() {
        let orig = "user:42:abc";
        let encoded = encode_cursor(orig);
        let decoded = decode_cursor(&encoded).unwrap();
        assert_eq!(decoded, orig);
    }

    #[test]
    fn test_cursor_empty() {
        assert_eq!(encode_cursor(""), "");
        assert_eq!(decode_cursor(""), Some(String::new()));
    }

    #[test]
    fn test_cursor_decode_invalid() {
        assert!(decode_cursor("g").is_none()); // odd length
        assert!(decode_cursor("zz").is_none()); // not valid hex
    }

    // ---- CursorPage / CursorResult ----

    #[test]
    fn test_cursor_first_page() {
        let items: Vec<i32> = (1..=10).collect();
        let page = CursorPage::first_page(3);
        let result = paginate_cursor(&items, &page, |_, idx| encode_cursor(&idx.to_string()));
        assert_eq!(result.node_count(), 3);
        assert_eq!(result.edges[0].node, 1);
        assert!(result.has_next_page);
        assert!(!result.has_previous_page);
    }

    #[test]
    fn test_cursor_after() {
        let items: Vec<i32> = (1..=10).collect();
        let cursor = encode_cursor("2"); // index 2
        let page = CursorPage::after(&cursor, 3);
        let result = paginate_cursor(&items, &page, |_, idx| encode_cursor(&idx.to_string()));
        assert_eq!(result.node_count(), 3);
        assert_eq!(result.edges[0].node, 4); // index 3 -> value 4
        assert!(result.has_next_page);
        assert!(result.has_previous_page);
    }

    #[test]
    fn test_cursor_last_page() {
        let items: Vec<i32> = (1..=5).collect();
        let cursor = encode_cursor("2");
        let page = CursorPage::after(&cursor, 10);
        let result = paginate_cursor(&items, &page, |_, idx| encode_cursor(&idx.to_string()));
        assert_eq!(result.node_count(), 2); // items 4, 5
        assert!(!result.has_next_page);
        assert!(result.has_previous_page);
    }

    #[test]
    fn test_cursor_result_cursors() {
        let items = vec!["a", "b", "c"];
        let page = CursorPage::first_page(3);
        let result = paginate_cursor(&items, &page, |_, idx| encode_cursor(&idx.to_string()));
        assert!(result.start_cursor().is_some());
        assert!(result.end_cursor().is_some());
    }

    // ---- Keyset ----

    #[test]
    fn test_keyset_first() {
        let items: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();
        let page = KeysetPage::first(3);
        let (result, next_key) = paginate_keyset(&items, &page, |s| s.clone());
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "a");
        assert_eq!(next_key, Some("c".to_string()));
    }

    #[test]
    fn test_keyset_after() {
        let items: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();
        let page = KeysetPage::after("c", 2);
        let (result, next_key) = paginate_keyset(&items, &page, |s| s.clone());
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "d");
        assert_eq!(next_key, Some("e".to_string()));
    }

    #[test]
    fn test_keyset_past_end() {
        let items: Vec<String> = vec!["a", "b"].into_iter().map(String::from).collect();
        let page = KeysetPage::after("b", 5);
        let (result, next_key) = paginate_keyset(&items, &page, |s| s.clone());
        assert!(result.is_empty());
        assert!(next_key.is_none());
    }

    // ---- PageMeta ----

    #[test]
    fn test_page_meta_basic() {
        let meta = PageMeta::compute(1, 10, 95);
        assert_eq!(meta.total_pages, 10);
        assert!(meta.has_next);
        assert!(!meta.has_prev);
        assert_eq!(meta.offset(), 0);
    }

    #[test]
    fn test_page_meta_last_page() {
        let meta = PageMeta::compute(10, 10, 95);
        assert!(!meta.has_next);
        assert!(meta.has_prev);
        assert!(meta.is_last_page());
    }

    #[test]
    fn test_page_meta_middle() {
        let meta = PageMeta::compute(5, 10, 100);
        assert!(meta.has_next);
        assert!(meta.has_prev);
        assert_eq!(meta.offset(), 40);
    }

    #[test]
    fn test_page_meta_zero_per_page() {
        let meta = PageMeta::compute(1, 0, 10);
        assert_eq!(meta.per_page, 1);
        assert_eq!(meta.total_pages, 10);
    }

    #[test]
    fn test_page_meta_empty_dataset() {
        let meta = PageMeta::compute(1, 10, 0);
        assert_eq!(meta.total_pages, 0);
        assert!(!meta.has_next);
        assert!(!meta.has_prev);
    }

    #[test]
    fn test_page_meta_clamp_beyond_total() {
        let meta = PageMeta::compute(999, 10, 50);
        assert_eq!(meta.page, 5); // clamped to last
        assert!(meta.is_last_page());
    }

    // ---- Link header ----

    #[test]
    fn test_link_header_first_page() {
        let meta = PageMeta::compute(1, 10, 50);
        let link = generate_link_header("https://api.example.com/items", &meta);
        assert!(link.contains("rel=\"first\""));
        assert!(link.contains("rel=\"last\""));
        assert!(link.contains("rel=\"next\""));
        assert!(!link.contains("rel=\"prev\""));
    }

    #[test]
    fn test_link_header_middle_page() {
        let meta = PageMeta::compute(3, 10, 50);
        let link = generate_link_header("https://api.example.com/items", &meta);
        assert!(link.contains("rel=\"prev\""));
        assert!(link.contains("rel=\"next\""));
        assert!(link.contains("page=2")); // prev
        assert!(link.contains("page=4")); // next
    }

    #[test]
    fn test_link_header_last_page() {
        let meta = PageMeta::compute(5, 10, 50);
        let link = generate_link_header("https://api.example.com/items", &meta);
        assert!(link.contains("rel=\"prev\""));
        assert!(!link.contains("rel=\"next\""));
    }

    // ---- page_slice_bounds ----

    #[test]
    fn test_page_slice_bounds_first() {
        let (start, end) = page_slice_bounds(1, 10, 35);
        assert_eq!(start, 0);
        assert_eq!(end, 10);
    }

    #[test]
    fn test_page_slice_bounds_last() {
        let (start, end) = page_slice_bounds(4, 10, 35);
        assert_eq!(start, 30);
        assert_eq!(end, 35);
    }

    #[test]
    fn test_page_slice_bounds_beyond() {
        let (start, end) = page_slice_bounds(100, 10, 35);
        assert_eq!(start, 35);
        assert_eq!(end, 35);
    }

    #[test]
    fn test_page_slice_bounds_zero_page() {
        let (start, end) = page_slice_bounds(0, 10, 35);
        assert_eq!(start, 0);
        assert_eq!(end, 10);
    }
}
