//! API pagination strategies — offset/limit, cursor-based, keyset pagination,
//! page metadata (total, has_next, has_prev), link headers (RFC 8288),
//! pagination from query params, and stable sort requirement.
//!
//! Replaces `koa-paginate`, `express-paginate`, and similar JS pagination
//! libraries with a pure-Rust pagination engine.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Pagination error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaginationError {
    /// Invalid page number.
    InvalidPage(String),
    /// Invalid page size.
    InvalidPageSize { requested: usize, max: usize },
    /// Invalid cursor.
    InvalidCursor(String),
    /// Invalid offset.
    InvalidOffset(String),
    /// Missing sort field for keyset pagination.
    MissingSortField(String),
    /// Page out of range.
    PageOutOfRange { page: usize, total_pages: usize },
}

impl fmt::Display for PaginationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPage(msg) => write!(f, "invalid page: {msg}"),
            Self::InvalidPageSize { requested, max } => {
                write!(f, "page size {requested} exceeds max {max}")
            }
            Self::InvalidCursor(c) => write!(f, "invalid cursor: {c}"),
            Self::InvalidOffset(msg) => write!(f, "invalid offset: {msg}"),
            Self::MissingSortField(field) => {
                write!(f, "missing sort field for keyset pagination: {field}")
            }
            Self::PageOutOfRange { page, total_pages } => {
                write!(f, "page {page} out of range (total: {total_pages})")
            }
        }
    }
}

impl std::error::Error for PaginationError {}

// ── Page Metadata ────────────────────────────────────────────────

/// Metadata about a paginated response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageMeta {
    /// Current page number (1-based).
    pub page: usize,
    /// Items per page.
    pub per_page: usize,
    /// Total number of items.
    pub total_items: usize,
    /// Total number of pages.
    pub total_pages: usize,
    /// Whether there is a next page.
    pub has_next: bool,
    /// Whether there is a previous page.
    pub has_prev: bool,
}

impl PageMeta {
    /// Compute page metadata from total items, page number, and per-page size.
    pub fn compute(total_items: usize, page: usize, per_page: usize) -> Self {
        let per_page = per_page.max(1);
        let total_pages = if total_items == 0 {
            1
        } else {
            (total_items + per_page - 1) / per_page
        };
        let page = page.max(1).min(total_pages);

        Self {
            page,
            per_page,
            total_items,
            total_pages,
            has_next: page < total_pages,
            has_prev: page > 1,
        }
    }

    /// The starting offset for the current page.
    pub fn offset(&self) -> usize {
        (self.page - 1) * self.per_page
    }
}

// ── Offset/Limit Pagination ─────────────────────────────────────

/// Offset/limit pagination request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OffsetRequest {
    /// Zero-based offset.
    pub offset: usize,
    /// Number of items to return.
    pub limit: usize,
}

/// Offset/limit pagination response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OffsetResponse<T> {
    /// The items for this page.
    pub items: Vec<T>,
    /// The offset used.
    pub offset: usize,
    /// The limit used.
    pub limit: usize,
    /// Total number of items.
    pub total: usize,
    /// Whether there are more items.
    pub has_more: bool,
}

/// Paginate a slice with offset/limit.
pub fn paginate_offset<T: Clone>(
    data: &[T],
    offset: usize,
    limit: usize,
    max_limit: usize,
) -> Result<OffsetResponse<T>, PaginationError> {
    if limit > max_limit {
        return Err(PaginationError::InvalidPageSize {
            requested: limit,
            max: max_limit,
        });
    }
    let limit = limit.max(1);
    let total = data.len();
    let start = offset.min(total);
    let end = (start + limit).min(total);
    let items = data[start..end].to_vec();
    let has_more = end < total;

    Ok(OffsetResponse {
        items,
        offset: start,
        limit,
        total,
        has_more,
    })
}

// ── Page Number Pagination ───────────────────────────────────────

/// Page-number based pagination request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageRequest {
    /// Page number (1-based).
    pub page: usize,
    /// Items per page.
    pub per_page: usize,
}

/// Page-number based pagination response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageResponse<T> {
    /// The items for this page.
    pub items: Vec<T>,
    /// Page metadata.
    pub meta: PageMeta,
}

/// Paginate a slice with page numbers.
pub fn paginate_page<T: Clone>(
    data: &[T],
    page: usize,
    per_page: usize,
    max_per_page: usize,
) -> Result<PageResponse<T>, PaginationError> {
    if per_page > max_per_page {
        return Err(PaginationError::InvalidPageSize {
            requested: per_page,
            max: max_per_page,
        });
    }
    if page == 0 {
        return Err(PaginationError::InvalidPage("page must be >= 1".to_string()));
    }

    let meta = PageMeta::compute(data.len(), page, per_page);

    if page > meta.total_pages {
        return Err(PaginationError::PageOutOfRange {
            page,
            total_pages: meta.total_pages,
        });
    }

    let start = meta.offset();
    let end = (start + per_page).min(data.len());
    let items = data[start..end].to_vec();

    Ok(PageResponse { items, meta })
}

// ── Cursor-Based Pagination ─────────────────────────────────────

/// Cursor value (opaque string).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cursor(pub String);

impl Cursor {
    /// Create a cursor from a string.
    pub fn new(value: &str) -> Self {
        Self(value.to_string())
    }

    /// Encode an offset as a cursor (base64-like encoding).
    pub fn from_offset(offset: usize) -> Self {
        Self(format!("off:{offset}"))
    }

    /// Decode an offset from a cursor.
    pub fn to_offset(&self) -> Result<usize, PaginationError> {
        if let Some(offset_str) = self.0.strip_prefix("off:") {
            offset_str
                .parse::<usize>()
                .map_err(|_| PaginationError::InvalidCursor(self.0.clone()))
        } else {
            Err(PaginationError::InvalidCursor(self.0.clone()))
        }
    }

    /// Encode a key as a cursor.
    pub fn from_key(key: &str) -> Self {
        Self(format!("key:{key}"))
    }

    /// Decode a key from a cursor.
    pub fn to_key(&self) -> Result<String, PaginationError> {
        if let Some(key) = self.0.strip_prefix("key:") {
            Ok(key.to_string())
        } else {
            Err(PaginationError::InvalidCursor(self.0.clone()))
        }
    }
}

impl fmt::Display for Cursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Cursor-based pagination request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorRequest {
    /// Cursor for the starting position (None = from beginning).
    pub after: Option<Cursor>,
    /// Number of items to return.
    pub first: usize,
}

/// Cursor-based pagination response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorResponse<T> {
    /// The items (edges) with cursors.
    pub edges: Vec<Edge<T>>,
    /// Page info.
    pub page_info: CursorPageInfo,
}

/// An edge in cursor pagination (item + cursor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge<T> {
    /// The item.
    pub node: T,
    /// Cursor for this item.
    pub cursor: Cursor,
}

/// Page info for cursor pagination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CursorPageInfo {
    /// Whether there is a next page.
    pub has_next_page: bool,
    /// Whether there is a previous page.
    pub has_previous_page: bool,
    /// Cursor of the first item.
    pub start_cursor: Option<Cursor>,
    /// Cursor of the last item.
    pub end_cursor: Option<Cursor>,
    /// Total count (optional).
    pub total_count: Option<usize>,
}

/// Paginate a slice with cursor-based pagination.
pub fn paginate_cursor<T: Clone>(
    data: &[T],
    after: Option<&Cursor>,
    first: usize,
    max_first: usize,
) -> Result<CursorResponse<T>, PaginationError> {
    if first > max_first {
        return Err(PaginationError::InvalidPageSize {
            requested: first,
            max: max_first,
        });
    }

    let start = match after {
        Some(cursor) => {
            let offset = cursor.to_offset()?;
            offset + 1 // Start after the cursor position.
        }
        None => 0,
    };

    let total = data.len();
    let end = (start + first).min(total);
    let has_next = end < total;
    let has_prev = start > 0;

    let mut edges = Vec::new();
    for i in start..end {
        edges.push(Edge {
            node: data[i].clone(),
            cursor: Cursor::from_offset(i),
        });
    }

    let start_cursor = edges.first().map(|e| e.cursor.clone());
    let end_cursor = edges.last().map(|e| e.cursor.clone());

    Ok(CursorResponse {
        edges,
        page_info: CursorPageInfo {
            has_next_page: has_next,
            has_previous_page: has_prev,
            start_cursor,
            end_cursor,
            total_count: Some(total),
        },
    })
}

// ── Keyset Pagination ────────────────────────────────────────────

/// Keyset pagination — uses the last value of a sort column for efficient
/// seek-based pagination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeysetRequest {
    /// The last value seen (None = start from beginning).
    pub after_key: Option<String>,
    /// Number of items to return.
    pub limit: usize,
    /// Sort field name.
    pub sort_field: String,
    /// Sort direction.
    pub ascending: bool,
}

/// Keyset pagination response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeysetResponse<T> {
    /// The items.
    pub items: Vec<T>,
    /// The last key in this page (for next request).
    pub last_key: Option<String>,
    /// Whether there are more items.
    pub has_more: bool,
}

/// Paginate string-keyed items using keyset pagination.
pub fn paginate_keyset(
    keys: &[String],
    data: &[String],
    after_key: Option<&str>,
    limit: usize,
    ascending: bool,
) -> KeysetResponse<(String, String)> {
    assert_eq!(keys.len(), data.len());

    let mut indices: Vec<usize> = (0..keys.len()).collect();
    if ascending {
        indices.sort_by(|a, b| keys[*a].cmp(&keys[*b]));
    } else {
        indices.sort_by(|a, b| keys[*b].cmp(&keys[*a]));
    }

    let start_idx = match after_key {
        Some(ak) => {
            let pos = indices.iter().position(|i| {
                if ascending {
                    keys[*i].as_str() > ak
                } else {
                    keys[*i].as_str() < ak
                }
            });
            pos.unwrap_or(indices.len())
        }
        None => 0,
    };

    let end_idx = (start_idx + limit).min(indices.len());
    let items: Vec<(String, String)> = indices[start_idx..end_idx]
        .iter()
        .map(|i| (keys[*i].clone(), data[*i].clone()))
        .collect();

    let last_key = items.last().map(|(k, _)| k.clone());
    let has_more = end_idx < indices.len();

    KeysetResponse { items, last_key, has_more }
}

// ── Link Headers (RFC 8288) ─────────────────────────────────────

/// Generate Link headers for pagination (RFC 8288).
#[derive(Debug, Clone)]
pub struct LinkHeaders {
    links: Vec<(String, String)>,
}

impl LinkHeaders {
    /// Create link headers for page-number pagination.
    pub fn for_page(base_url: &str, meta: &PageMeta) -> Self {
        let mut links = Vec::new();

        // First page.
        links.push((
            format!("{base_url}?page=1&per_page={}", meta.per_page),
            "first".to_string(),
        ));

        // Last page.
        links.push((
            format!("{base_url}?page={}&per_page={}", meta.total_pages, meta.per_page),
            "last".to_string(),
        ));

        // Next page.
        if meta.has_next {
            links.push((
                format!("{base_url}?page={}&per_page={}", meta.page + 1, meta.per_page),
                "next".to_string(),
            ));
        }

        // Previous page.
        if meta.has_prev {
            links.push((
                format!("{base_url}?page={}&per_page={}", meta.page - 1, meta.per_page),
                "prev".to_string(),
            ));
        }

        Self { links }
    }

    /// Create link headers for cursor-based pagination.
    pub fn for_cursor(
        base_url: &str,
        page_info: &CursorPageInfo,
        first: usize,
    ) -> Self {
        let mut links = Vec::new();

        if page_info.has_next_page {
            if let Some(end_cursor) = &page_info.end_cursor {
                links.push((
                    format!("{base_url}?after={}&first={first}", end_cursor.0),
                    "next".to_string(),
                ));
            }
        }

        Self { links }
    }

    /// Format as a Link header value.
    pub fn to_header(&self) -> String {
        self.links
            .iter()
            .map(|(url, rel)| format!("<{url}>; rel=\"{rel}\""))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Get links as pairs.
    pub fn links(&self) -> &[(String, String)] {
        &self.links
    }
}

impl fmt::Display for LinkHeaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_header())
    }
}

// ── Query Param Parsing ──────────────────────────────────────────

/// Parse pagination parameters from a query string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationParams {
    pub page: Option<usize>,
    pub per_page: Option<usize>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
    pub cursor: Option<String>,
    pub sort: Option<String>,
    pub order: Option<String>,
}

impl PaginationParams {
    /// Parse from a query string like "page=2&per_page=20".
    pub fn from_query(query: &str) -> Self {
        let mut params = PaginationParams {
            page: None,
            per_page: None,
            offset: None,
            limit: None,
            cursor: None,
            sort: None,
            order: None,
        };

        for pair in query.split('&') {
            if let Some((key, value)) = pair.split_once('=') {
                match key {
                    "page" => params.page = value.parse().ok(),
                    "per_page" | "perPage" | "page_size" | "pageSize" => {
                        params.per_page = value.parse().ok();
                    }
                    "offset" | "skip" => params.offset = value.parse().ok(),
                    "limit" | "count" | "first" => params.limit = value.parse().ok(),
                    "cursor" | "after" => params.cursor = Some(value.to_string()),
                    "sort" | "sort_by" | "sortBy" => params.sort = Some(value.to_string()),
                    "order" | "direction" | "dir" => params.order = Some(value.to_string()),
                    _ => {}
                }
            }
        }

        params
    }

    /// Determine the pagination strategy from the parameters.
    pub fn strategy(&self) -> PaginationStrategy {
        if self.cursor.is_some() {
            PaginationStrategy::Cursor
        } else if self.offset.is_some() {
            PaginationStrategy::Offset
        } else if self.sort.is_some() && self.cursor.is_some() {
            PaginationStrategy::Keyset
        } else {
            PaginationStrategy::Page
        }
    }

    /// Get sort direction.
    pub fn is_ascending(&self) -> bool {
        match self.order.as_deref() {
            Some("desc") | Some("DESC") | Some("descending") => false,
            _ => true,
        }
    }
}

/// Pagination strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaginationStrategy {
    Page,
    Offset,
    Cursor,
    Keyset,
}

// ── Sort Stability ───────────────────────────────────────────────

/// Check if a sort specification includes a unique tiebreaker
/// (required for stable cursor/keyset pagination).
#[derive(Debug, Clone)]
pub struct SortSpec {
    pub fields: Vec<SortField>,
}

/// A single sort field.
#[derive(Debug, Clone)]
pub struct SortField {
    pub name: String,
    pub ascending: bool,
    pub unique: bool,
}

impl SortSpec {
    /// Create a new sort spec.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a sort field.
    pub fn add(mut self, name: &str, ascending: bool, unique: bool) -> Self {
        self.fields.push(SortField {
            name: name.to_string(),
            ascending,
            unique,
        });
        self
    }

    /// Check if this sort is stable (has at least one unique field).
    pub fn is_stable(&self) -> bool {
        self.fields.iter().any(|f| f.unique)
    }

    /// Ensure stability by adding an ID tiebreaker if needed.
    pub fn ensure_stable(mut self, id_field: &str) -> Self {
        if !self.is_stable() {
            self.fields.push(SortField {
                name: id_field.to_string(),
                ascending: true,
                unique: true,
            });
        }
        self
    }
}

impl Default for SortSpec {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_meta_basic() {
        let meta = PageMeta::compute(100, 1, 10);
        assert_eq!(meta.total_pages, 10);
        assert!(meta.has_next);
        assert!(!meta.has_prev);
        assert_eq!(meta.offset(), 0);
    }

    #[test]
    fn test_page_meta_middle() {
        let meta = PageMeta::compute(100, 5, 10);
        assert_eq!(meta.page, 5);
        assert!(meta.has_next);
        assert!(meta.has_prev);
        assert_eq!(meta.offset(), 40);
    }

    #[test]
    fn test_page_meta_last_page() {
        let meta = PageMeta::compute(100, 10, 10);
        assert!(!meta.has_next);
        assert!(meta.has_prev);
    }

    #[test]
    fn test_page_meta_empty() {
        let meta = PageMeta::compute(0, 1, 10);
        assert_eq!(meta.total_pages, 1);
        assert!(!meta.has_next);
        assert!(!meta.has_prev);
    }

    #[test]
    fn test_page_meta_partial_page() {
        let meta = PageMeta::compute(25, 1, 10);
        assert_eq!(meta.total_pages, 3);
    }

    #[test]
    fn test_offset_pagination() {
        let data: Vec<i32> = (0..50).collect();
        let resp = paginate_offset(&data, 10, 5, 100).unwrap();
        assert_eq!(resp.items, vec![10, 11, 12, 13, 14]);
        assert_eq!(resp.total, 50);
        assert!(resp.has_more);
    }

    #[test]
    fn test_offset_pagination_end() {
        let data: Vec<i32> = (0..50).collect();
        let resp = paginate_offset(&data, 48, 5, 100).unwrap();
        assert_eq!(resp.items, vec![48, 49]);
        assert!(!resp.has_more);
    }

    #[test]
    fn test_offset_pagination_exceeds_max() {
        let data: Vec<i32> = (0..50).collect();
        let err = paginate_offset(&data, 0, 200, 100).unwrap_err();
        assert!(matches!(err, PaginationError::InvalidPageSize { .. }));
    }

    #[test]
    fn test_page_pagination() {
        let data: Vec<i32> = (0..25).collect();
        let resp = paginate_page(&data, 2, 10, 100).unwrap();
        assert_eq!(resp.items, vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(resp.meta.page, 2);
        assert!(resp.meta.has_next);
        assert!(resp.meta.has_prev);
    }

    #[test]
    fn test_page_pagination_page_zero() {
        let data: Vec<i32> = (0..10).collect();
        let err = paginate_page(&data, 0, 10, 100).unwrap_err();
        assert!(matches!(err, PaginationError::InvalidPage(_)));
    }

    #[test]
    fn test_page_pagination_out_of_range() {
        let data: Vec<i32> = (0..10).collect();
        let err = paginate_page(&data, 5, 10, 100).unwrap_err();
        assert!(matches!(err, PaginationError::PageOutOfRange { .. }));
    }

    #[test]
    fn test_cursor_pagination_first_page() {
        let data: Vec<i32> = (0..20).collect();
        let resp = paginate_cursor(&data, None, 5, 100).unwrap();
        assert_eq!(resp.edges.len(), 5);
        assert_eq!(resp.edges[0].node, 0);
        assert_eq!(resp.edges[4].node, 4);
        assert!(resp.page_info.has_next_page);
        assert!(!resp.page_info.has_previous_page);
    }

    #[test]
    fn test_cursor_pagination_next_page() {
        let data: Vec<i32> = (0..20).collect();
        let resp1 = paginate_cursor(&data, None, 5, 100).unwrap();
        let end_cursor = resp1.page_info.end_cursor.as_ref().unwrap();

        let resp2 = paginate_cursor(&data, Some(end_cursor), 5, 100).unwrap();
        assert_eq!(resp2.edges[0].node, 5);
        assert!(resp2.page_info.has_previous_page);
    }

    #[test]
    fn test_cursor_pagination_last_page() {
        let data: Vec<i32> = (0..12).collect();
        let cursor = Cursor::from_offset(9);
        let resp = paginate_cursor(&data, Some(&cursor), 5, 100).unwrap();
        assert_eq!(resp.edges.len(), 2);
        assert!(!resp.page_info.has_next_page);
    }

    #[test]
    fn test_cursor_encode_decode() {
        let cursor = Cursor::from_offset(42);
        assert_eq!(cursor.to_offset().unwrap(), 42);

        let key_cursor = Cursor::from_key("user_abc");
        assert_eq!(key_cursor.to_key().unwrap(), "user_abc");
    }

    #[test]
    fn test_keyset_pagination_ascending() {
        let keys: Vec<String> = vec!["c", "a", "d", "b", "e"]
            .iter().map(|s| s.to_string()).collect();
        let data: Vec<String> = vec!["C", "A", "D", "B", "E"]
            .iter().map(|s| s.to_string()).collect();

        let resp = paginate_keyset(&keys, &data, None, 3, true);
        assert_eq!(resp.items.len(), 3);
        assert_eq!(resp.items[0].0, "a");
        assert_eq!(resp.items[2].0, "c");
        assert!(resp.has_more);
    }

    #[test]
    fn test_keyset_pagination_with_after() {
        let keys: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .iter().map(|s| s.to_string()).collect();
        let data: Vec<String> = keys.clone();

        let resp = paginate_keyset(&keys, &data, Some("c"), 2, true);
        assert_eq!(resp.items.len(), 2);
        assert_eq!(resp.items[0].0, "d");
        assert_eq!(resp.items[1].0, "e");
        assert!(!resp.has_more);
    }

    #[test]
    fn test_link_headers_page() {
        let meta = PageMeta::compute(100, 2, 10);
        let links = LinkHeaders::for_page("/api/items", &meta);
        let header = links.to_header();
        assert!(header.contains("rel=\"first\""));
        assert!(header.contains("rel=\"last\""));
        assert!(header.contains("rel=\"next\""));
        assert!(header.contains("rel=\"prev\""));
        assert!(header.contains("page=3"));
        assert!(header.contains("page=1"));
    }

    #[test]
    fn test_link_headers_no_prev_first_page() {
        let meta = PageMeta::compute(100, 1, 10);
        let links = LinkHeaders::for_page("/api/items", &meta);
        let header = links.to_header();
        assert!(header.contains("rel=\"next\""));
        assert!(!header.contains("rel=\"prev\""));
    }

    #[test]
    fn test_query_params_page() {
        let params = PaginationParams::from_query("page=3&per_page=25");
        assert_eq!(params.page, Some(3));
        assert_eq!(params.per_page, Some(25));
        assert_eq!(params.strategy(), PaginationStrategy::Page);
    }

    #[test]
    fn test_query_params_offset() {
        let params = PaginationParams::from_query("offset=100&limit=20");
        assert_eq!(params.offset, Some(100));
        assert_eq!(params.limit, Some(20));
        assert_eq!(params.strategy(), PaginationStrategy::Offset);
    }

    #[test]
    fn test_query_params_cursor() {
        let params = PaginationParams::from_query("cursor=abc123&limit=10");
        assert_eq!(params.cursor, Some("abc123".to_string()));
        assert_eq!(params.strategy(), PaginationStrategy::Cursor);
    }

    #[test]
    fn test_query_params_sort_direction() {
        let asc = PaginationParams::from_query("sort=name&order=asc");
        assert!(asc.is_ascending());

        let desc = PaginationParams::from_query("sort=name&order=desc");
        assert!(!desc.is_ascending());
    }

    #[test]
    fn test_sort_stability() {
        let sort = SortSpec::new()
            .add("name", true, false);
        assert!(!sort.is_stable());

        let stable = sort.ensure_stable("id");
        assert!(stable.is_stable());
        assert_eq!(stable.fields.len(), 2);
    }

    #[test]
    fn test_sort_already_stable() {
        let sort = SortSpec::new()
            .add("id", true, true);
        assert!(sort.is_stable());

        let still = sort.ensure_stable("id");
        // Should not add a duplicate.
        assert_eq!(still.fields.len(), 1);
    }

    #[test]
    fn test_error_display() {
        let err = PaginationError::InvalidPageSize { requested: 500, max: 100 };
        assert!(err.to_string().contains("500"));
        assert!(err.to_string().contains("100"));
    }
}
