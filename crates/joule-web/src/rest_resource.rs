//! REST resource modeling — resource definition with CRUD operations, nested
//! resources, link relations (HATEOAS), pagination (cursor/offset), filtering,
//! sorting, bulk operations, and ETag generation.
//!
//! Replaces `json-api-serializer`, `hal`, `express-resource`, and similar JS
//! REST libraries with a pure-Rust resource modeling layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// REST resource error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceError {
    /// Resource not found.
    NotFound { resource: String, id: String },
    /// Duplicate resource ID.
    Duplicate { resource: String, id: String },
    /// Invalid field for sorting/filtering.
    InvalidField(String),
    /// Invalid pagination parameters.
    InvalidPagination(String),
    /// Validation error.
    Validation(String),
    /// Bulk operation partial failure.
    BulkPartialFailure { succeeded: usize, failed: usize },
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { resource, id } => write!(f, "{resource} not found: {id}"),
            Self::Duplicate { resource, id } => write!(f, "{resource} already exists: {id}"),
            Self::InvalidField(name) => write!(f, "invalid field: {name}"),
            Self::InvalidPagination(msg) => write!(f, "invalid pagination: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::BulkPartialFailure { succeeded, failed } => {
                write!(f, "bulk operation: {succeeded} succeeded, {failed} failed")
            }
        }
    }
}

impl std::error::Error for ResourceError {}

// ── Types ────────────────────────────────────────────────────────

/// HTTP method for CRUD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Patch => write!(f, "PATCH"),
            Self::Delete => write!(f, "DELETE"),
        }
    }
}

/// A single link relation (HATEOAS).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub rel: String,
    pub href: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<HttpMethod>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Link {
    pub fn new(rel: &str, href: &str) -> Self {
        Self {
            rel: rel.to_string(),
            href: href.to_string(),
            method: None,
            title: None,
        }
    }

    pub fn with_method(mut self, method: HttpMethod) -> Self {
        self.method = Some(method);
        self
    }

    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }
}

/// A resource instance with data and links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource type name (e.g., "user", "post").
    pub resource_type: String,
    /// The resource data.
    pub data: serde_json::Value,
    /// HATEOAS links.
    #[serde(rename = "_links")]
    pub links: Vec<Link>,
    /// Embedded sub-resources.
    #[serde(rename = "_embedded", skip_serializing_if = "HashMap::is_empty")]
    pub embedded: HashMap<String, Vec<Resource>>,
    /// ETag for caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

impl Resource {
    /// Create a new resource.
    pub fn new(resource_type: &str, data: serde_json::Value) -> Self {
        Self {
            resource_type: resource_type.to_string(),
            data,
            links: Vec::new(),
            embedded: HashMap::new(),
            etag: None,
        }
    }

    /// Add a self link.
    pub fn with_self_link(mut self, href: &str) -> Self {
        self.links.push(Link::new("self", href));
        self
    }

    /// Add a generic link.
    pub fn with_link(mut self, link: Link) -> Self {
        self.links.push(link);
        self
    }

    /// Add embedded resources.
    pub fn with_embedded(mut self, rel: &str, resources: Vec<Resource>) -> Self {
        self.embedded.insert(rel.to_string(), resources);
        self
    }

    /// Compute and set the ETag from data.
    pub fn with_etag(mut self) -> Self {
        self.etag = Some(compute_etag(&self.data));
        self
    }

    /// Get the resource ID from data (assumes "id" field).
    pub fn id(&self) -> Option<&str> {
        self.data.get("id").and_then(|v| v.as_str())
    }

    /// Get a specific link by relation.
    pub fn link(&self, rel: &str) -> Option<&Link> {
        self.links.iter().find(|l| l.rel == rel)
    }
}

// ── Pagination ───────────────────────────────────────────────────

/// Pagination strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaginationKind {
    Offset,
    Cursor,
}

/// Pagination parameters.
#[derive(Debug, Clone)]
pub struct PaginationParams {
    pub kind: PaginationKind,
    /// For offset pagination: page number (1-based).
    pub page: Option<u64>,
    /// Items per page / limit.
    pub per_page: u64,
    /// For cursor pagination: cursor value.
    pub cursor: Option<String>,
}

impl PaginationParams {
    /// Create offset pagination params.
    pub fn offset(page: u64, per_page: u64) -> Self {
        Self {
            kind: PaginationKind::Offset,
            page: Some(page),
            per_page,
            cursor: None,
        }
    }

    /// Create cursor pagination params.
    pub fn cursor(cursor: Option<&str>, per_page: u64) -> Self {
        Self {
            kind: PaginationKind::Cursor,
            page: None,
            per_page,
            cursor: cursor.map(|s| s.to_string()),
        }
    }

    /// Compute the offset for database queries.
    pub fn offset_value(&self) -> u64 {
        match self.kind {
            PaginationKind::Offset => {
                let page = self.page.unwrap_or(1).max(1);
                (page - 1) * self.per_page
            }
            PaginationKind::Cursor => 0,
        }
    }
}

/// Paginated result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResult {
    pub items: Vec<Resource>,
    pub total: u64,
    pub page_info: PageInfo,
    #[serde(rename = "_links")]
    pub links: Vec<Link>,
}

/// Page metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageInfo {
    pub has_next_page: bool,
    pub has_previous_page: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_pages: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_page: Option<u64>,
}

/// Build a paginated result from items.
pub fn paginate(
    items: Vec<Resource>,
    total: u64,
    params: &PaginationParams,
    base_url: &str,
) -> PaginatedResult {
    let mut links = vec![Link::new("self", base_url)];

    match params.kind {
        PaginationKind::Offset => {
            let page = params.page.unwrap_or(1).max(1);
            let total_pages = (total + params.per_page - 1) / params.per_page.max(1);
            let has_next = page < total_pages;
            let has_prev = page > 1;

            if has_next {
                links.push(Link::new(
                    "next",
                    &format!("{base_url}?page={}&per_page={}", page + 1, params.per_page),
                ));
            }
            if has_prev {
                links.push(Link::new(
                    "prev",
                    &format!("{base_url}?page={}&per_page={}", page - 1, params.per_page),
                ));
            }
            links.push(Link::new(
                "first",
                &format!("{base_url}?page=1&per_page={}", params.per_page),
            ));
            links.push(Link::new(
                "last",
                &format!("{base_url}?page={total_pages}&per_page={}", params.per_page),
            ));

            PaginatedResult {
                items,
                total,
                page_info: PageInfo {
                    has_next_page: has_next,
                    has_previous_page: has_prev,
                    next_cursor: None,
                    total_pages: Some(total_pages),
                    current_page: Some(page),
                },
                links,
            }
        }
        PaginationKind::Cursor => {
            let has_next = items.len() as u64 >= params.per_page;
            let next_cursor = if has_next {
                items.last().and_then(|r| r.id()).map(|id| id.to_string())
            } else {
                None
            };

            if let Some(ref cursor) = next_cursor {
                links.push(Link::new(
                    "next",
                    &format!("{base_url}?cursor={cursor}&per_page={}", params.per_page),
                ));
            }

            PaginatedResult {
                items,
                total,
                page_info: PageInfo {
                    has_next_page: has_next,
                    has_previous_page: params.cursor.is_some(),
                    next_cursor,
                    total_pages: None,
                    current_page: None,
                },
                links,
            }
        }
    }
}

// ── Sorting & Filtering ─────────────────────────────────────────

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Sort specification.
#[derive(Debug, Clone)]
pub struct SortSpec {
    pub field: String,
    pub direction: SortDirection,
}

/// Parse a sort string like "-created_at,name" into SortSpecs.
pub fn parse_sort(input: &str) -> Vec<SortSpec> {
    input
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let s = s.trim();
            if let Some(field) = s.strip_prefix('-') {
                SortSpec {
                    field: field.to_string(),
                    direction: SortDirection::Descending,
                }
            } else {
                let field = s.strip_prefix('+').unwrap_or(s);
                SortSpec {
                    field: field.to_string(),
                    direction: SortDirection::Ascending,
                }
            }
        })
        .collect()
}

/// Filter operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
    StartsWith,
    In,
}

/// A single filter condition.
#[derive(Debug, Clone)]
pub struct FilterSpec {
    pub field: String,
    pub op: FilterOp,
    pub value: serde_json::Value,
}

/// Parse simple filter params like "status=active" or "age[gte]=18".
pub fn parse_filters(params: &HashMap<String, String>) -> Vec<FilterSpec> {
    let mut filters = Vec::new();
    for (key, value) in params {
        // Check for bracket notation: field[op]=value
        if let Some(bracket_pos) = key.find('[') {
            if let Some(end) = key.find(']') {
                let field = &key[..bracket_pos];
                let op_str = &key[bracket_pos + 1..end];
                let op = match op_str {
                    "eq" => FilterOp::Eq,
                    "neq" | "ne" => FilterOp::Neq,
                    "gt" => FilterOp::Gt,
                    "gte" | "ge" => FilterOp::Gte,
                    "lt" => FilterOp::Lt,
                    "lte" | "le" => FilterOp::Lte,
                    "contains" | "like" => FilterOp::Contains,
                    "starts_with" | "prefix" => FilterOp::StartsWith,
                    "in" => FilterOp::In,
                    _ => continue,
                };
                filters.push(FilterSpec {
                    field: field.to_string(),
                    op,
                    value: serde_json::Value::String(value.clone()),
                });
            }
        } else {
            // Simple equality: field=value
            filters.push(FilterSpec {
                field: key.clone(),
                op: FilterOp::Eq,
                value: serde_json::Value::String(value.clone()),
            });
        }
    }
    filters
}

/// Apply filters to a list of JSON objects (in-memory filtering).
pub fn apply_filters(
    items: &[serde_json::Value],
    filters: &[FilterSpec],
) -> Vec<serde_json::Value> {
    items
        .iter()
        .filter(|item| {
            filters.iter().all(|filter| {
                let field_val = item.get(&filter.field);
                match &filter.op {
                    FilterOp::Eq => field_val.map(|v| matches_value(v, &filter.value)).unwrap_or(false),
                    FilterOp::Neq => field_val.map(|v| !matches_value(v, &filter.value)).unwrap_or(true),
                    FilterOp::Contains => {
                        field_val
                            .and_then(|v| v.as_str())
                            .map(|s| {
                                let pattern = filter.value.as_str().unwrap_or("");
                                s.contains(pattern)
                            })
                            .unwrap_or(false)
                    }
                    FilterOp::StartsWith => {
                        field_val
                            .and_then(|v| v.as_str())
                            .map(|s| {
                                let pattern = filter.value.as_str().unwrap_or("");
                                s.starts_with(pattern)
                            })
                            .unwrap_or(false)
                    }
                    FilterOp::Gt | FilterOp::Gte | FilterOp::Lt | FilterOp::Lte => {
                        compare_values(field_val, &filter.value, &filter.op)
                    }
                    FilterOp::In => {
                        let csv = filter.value.as_str().unwrap_or("");
                        let vals: Vec<&str> = csv.split(',').collect();
                        field_val
                            .and_then(|v| v.as_str())
                            .map(|s| vals.contains(&s))
                            .unwrap_or(false)
                    }
                }
            })
        })
        .cloned()
        .collect()
}

fn matches_value(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::String(sa), serde_json::Value::String(sb)) => sa == sb,
        (serde_json::Value::Number(na), serde_json::Value::String(sb)) => {
            na.to_string() == *sb
        }
        (serde_json::Value::Bool(ba), serde_json::Value::String(sb)) => {
            ba.to_string() == *sb
        }
        _ => a == b,
    }
}

fn compare_values(
    field_val: Option<&serde_json::Value>,
    filter_val: &serde_json::Value,
    op: &FilterOp,
) -> bool {
    let fv = match field_val.and_then(|v| to_f64(v)) {
        Some(v) => v,
        None => return false,
    };
    let tv = match to_f64(filter_val) {
        Some(v) => v,
        None => return false,
    };
    match op {
        FilterOp::Gt => fv > tv,
        FilterOp::Gte => fv >= tv,
        FilterOp::Lt => fv < tv,
        FilterOp::Lte => fv <= tv,
        _ => false,
    }
}

fn to_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Apply sort to a list of JSON objects.
pub fn apply_sort(items: &mut [serde_json::Value], specs: &[SortSpec]) {
    items.sort_by(|a, b| {
        for spec in specs {
            let va = a.get(&spec.field);
            let vb = b.get(&spec.field);
            let cmp = compare_json_values(va, vb);
            let cmp = match spec.direction {
                SortDirection::Ascending => cmp,
                SortDirection::Descending => cmp.reverse(),
            };
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
}

fn compare_json_values(
    a: Option<&serde_json::Value>,
    b: Option<&serde_json::Value>,
) -> std::cmp::Ordering {
    match (a, b) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(va), Some(vb)) => {
            if let (Some(sa), Some(sb)) = (va.as_str(), vb.as_str()) {
                return sa.cmp(sb);
            }
            if let (Some(na), Some(nb)) = (va.as_f64(), vb.as_f64()) {
                return na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal);
            }
            std::cmp::Ordering::Equal
        }
    }
}

// ── ETag Generation ──────────────────────────────────────────────

/// Compute a simple ETag from a JSON value using DJB2 hash.
pub fn compute_etag(data: &serde_json::Value) -> String {
    let serialized = serde_json::to_string(data).unwrap_or_default();
    let mut hash: u64 = 5381;
    for byte in serialized.as_bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(*byte as u64);
    }
    format!("\"{hash:016x}\"")
}

/// Check if an ETag matches (for conditional requests).
pub fn etag_matches(etag: &str, if_none_match: &str) -> bool {
    if if_none_match == "*" {
        return true;
    }
    // Parse comma-separated ETags
    if_none_match
        .split(',')
        .any(|tag| tag.trim() == etag)
}

// ── Bulk Operations ──────────────────────────────────────────────

/// Bulk operation kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BulkAction {
    Create,
    Update,
    Delete,
}

/// A single bulk operation entry.
#[derive(Debug, Clone)]
pub struct BulkEntry {
    pub action: BulkAction,
    pub id: Option<String>,
    pub data: Option<serde_json::Value>,
}

/// Result of a single bulk operation.
#[derive(Debug, Clone)]
pub struct BulkResult {
    pub index: usize,
    pub action: BulkAction,
    pub success: bool,
    pub id: Option<String>,
    pub error: Option<String>,
}

/// Process bulk operations and return individual results.
pub fn process_bulk(
    entries: &[BulkEntry],
    mut handler: impl FnMut(&BulkEntry) -> Result<String, String>,
) -> Vec<BulkResult> {
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| match handler(entry) {
            Ok(id) => BulkResult {
                index: i,
                action: entry.action.clone(),
                success: true,
                id: Some(id),
                error: None,
            },
            Err(msg) => BulkResult {
                index: i,
                action: entry.action.clone(),
                success: false,
                id: entry.id.clone(),
                error: Some(msg),
            },
        })
        .collect()
}

/// Resource definition — describes a REST resource and its capabilities.
#[derive(Debug, Clone)]
pub struct ResourceDefinition {
    /// Resource type name (e.g., "users").
    pub name: String,
    /// Base path (e.g., "/users").
    pub base_path: String,
    /// Allowed operations.
    pub allowed_methods: Vec<HttpMethod>,
    /// Sortable fields.
    pub sortable_fields: Vec<String>,
    /// Filterable fields.
    pub filterable_fields: Vec<String>,
    /// Nested resource names.
    pub nested_resources: Vec<String>,
    /// Default items per page.
    pub default_per_page: u64,
    /// Max items per page.
    pub max_per_page: u64,
}

impl ResourceDefinition {
    pub fn new(name: &str, base_path: &str) -> Self {
        Self {
            name: name.to_string(),
            base_path: base_path.to_string(),
            allowed_methods: vec![
                HttpMethod::Get,
                HttpMethod::Post,
                HttpMethod::Put,
                HttpMethod::Patch,
                HttpMethod::Delete,
            ],
            sortable_fields: Vec::new(),
            filterable_fields: Vec::new(),
            nested_resources: Vec::new(),
            default_per_page: 20,
            max_per_page: 100,
        }
    }

    /// Generate standard HATEOAS links for a resource instance.
    pub fn instance_links(&self, id: &str) -> Vec<Link> {
        let href = format!("{}/{id}", self.base_path);
        let mut links = vec![Link::new("self", &href)];

        links.push(Link::new("collection", &self.base_path));

        if self.allowed_methods.contains(&HttpMethod::Put) {
            links.push(Link::new("update", &href).with_method(HttpMethod::Put));
        }
        if self.allowed_methods.contains(&HttpMethod::Delete) {
            links.push(Link::new("delete", &href).with_method(HttpMethod::Delete));
        }

        for nested in &self.nested_resources {
            links.push(Link::new(nested, &format!("{href}/{nested}")));
        }

        links
    }

    /// Generate collection-level links.
    pub fn collection_links(&self) -> Vec<Link> {
        let mut links = vec![Link::new("self", &self.base_path)];
        if self.allowed_methods.contains(&HttpMethod::Post) {
            links.push(
                Link::new("create", &self.base_path)
                    .with_method(HttpMethod::Post)
                    .with_title(&format!("Create {}", self.name)),
            );
        }
        links
    }

    /// Validate that a sort field is allowed.
    pub fn validate_sort_field(&self, field: &str) -> Result<(), ResourceError> {
        if self.sortable_fields.is_empty() || self.sortable_fields.contains(&field.to_string()) {
            Ok(())
        } else {
            Err(ResourceError::InvalidField(format!(
                "field '{field}' is not sortable"
            )))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_resource() {
        let r = Resource::new("user", serde_json::json!({"id": "1", "name": "Alice"}));
        assert_eq!(r.resource_type, "user");
        assert_eq!(r.id(), Some("1"));
    }

    #[test]
    fn resource_with_self_link() {
        let r = Resource::new("user", serde_json::json!({"id": "1"}))
            .with_self_link("/users/1");
        let link = r.link("self").unwrap();
        assert_eq!(link.href, "/users/1");
    }

    #[test]
    fn resource_with_etag() {
        let r = Resource::new("user", serde_json::json!({"id": "1"})).with_etag();
        assert!(r.etag.is_some());
        assert!(r.etag.as_ref().unwrap().starts_with('"'));
    }

    #[test]
    fn resource_with_embedded() {
        let post = Resource::new("post", serde_json::json!({"id": "p1", "title": "Hello"}));
        let r = Resource::new("user", serde_json::json!({"id": "1"}))
            .with_embedded("posts", vec![post]);
        assert_eq!(r.embedded["posts"].len(), 1);
    }

    #[test]
    fn offset_pagination() {
        let items = vec![
            Resource::new("item", serde_json::json!({"id": "1"})),
            Resource::new("item", serde_json::json!({"id": "2"})),
        ];
        let params = PaginationParams::offset(1, 10);
        let result = paginate(items, 25, &params, "/items");
        assert_eq!(result.total, 25);
        assert_eq!(result.page_info.total_pages, Some(3));
        assert!(result.page_info.has_next_page);
        assert!(!result.page_info.has_previous_page);
        assert!(result.links.iter().any(|l| l.rel == "next"));
    }

    #[test]
    fn cursor_pagination() {
        let items = vec![
            Resource::new("item", serde_json::json!({"id": "abc"})),
            Resource::new("item", serde_json::json!({"id": "def"})),
        ];
        let params = PaginationParams::cursor(None, 2);
        let result = paginate(items, 10, &params, "/items");
        assert!(result.page_info.has_next_page);
        assert!(!result.page_info.has_previous_page);
        assert_eq!(result.page_info.next_cursor.as_deref(), Some("def"));
    }

    #[test]
    fn offset_calculation() {
        let p = PaginationParams::offset(3, 20);
        assert_eq!(p.offset_value(), 40);
    }

    #[test]
    fn parse_sort_ascending() {
        let specs = parse_sort("name");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].field, "name");
        assert_eq!(specs[0].direction, SortDirection::Ascending);
    }

    #[test]
    fn parse_sort_descending() {
        let specs = parse_sort("-created_at");
        assert_eq!(specs[0].direction, SortDirection::Descending);
    }

    #[test]
    fn parse_sort_multiple() {
        let specs = parse_sort("-date,name");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].direction, SortDirection::Descending);
        assert_eq!(specs[1].direction, SortDirection::Ascending);
    }

    #[test]
    fn parse_simple_filter() {
        let mut params = HashMap::new();
        params.insert("status".to_string(), "active".to_string());
        let filters = parse_filters(&params);
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].op, FilterOp::Eq);
    }

    #[test]
    fn parse_bracket_filter() {
        let mut params = HashMap::new();
        params.insert("age[gte]".to_string(), "18".to_string());
        let filters = parse_filters(&params);
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].field, "age");
        assert_eq!(filters[0].op, FilterOp::Gte);
    }

    #[test]
    fn apply_eq_filter() {
        let items = vec![
            serde_json::json!({"name": "Alice", "role": "admin"}),
            serde_json::json!({"name": "Bob", "role": "user"}),
        ];
        let filters = vec![FilterSpec {
            field: "role".to_string(),
            op: FilterOp::Eq,
            value: serde_json::json!("admin"),
        }];
        let filtered = apply_filters(&items, &filters);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["name"], "Alice");
    }

    #[test]
    fn apply_contains_filter() {
        let items = vec![
            serde_json::json!({"name": "Alice Cooper"}),
            serde_json::json!({"name": "Bob Smith"}),
        ];
        let filters = vec![FilterSpec {
            field: "name".to_string(),
            op: FilterOp::Contains,
            value: serde_json::json!("Cooper"),
        }];
        let filtered = apply_filters(&items, &filters);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn apply_sort_ascending() {
        let mut items = vec![
            serde_json::json!({"name": "Charlie"}),
            serde_json::json!({"name": "Alice"}),
            serde_json::json!({"name": "Bob"}),
        ];
        apply_sort(
            &mut items,
            &[SortSpec {
                field: "name".to_string(),
                direction: SortDirection::Ascending,
            }],
        );
        assert_eq!(items[0]["name"], "Alice");
        assert_eq!(items[1]["name"], "Bob");
        assert_eq!(items[2]["name"], "Charlie");
    }

    #[test]
    fn etag_consistency() {
        let data = serde_json::json!({"id": "1", "name": "test"});
        let etag1 = compute_etag(&data);
        let etag2 = compute_etag(&data);
        assert_eq!(etag1, etag2);
    }

    #[test]
    fn etag_matches_exact() {
        let etag = "\"abc123\"";
        assert!(etag_matches(etag, "\"abc123\""));
        assert!(!etag_matches(etag, "\"xyz789\""));
    }

    #[test]
    fn etag_matches_wildcard() {
        assert!(etag_matches("\"anything\"", "*"));
    }

    #[test]
    fn bulk_operations() {
        let entries = vec![
            BulkEntry { action: BulkAction::Create, id: None, data: Some(serde_json::json!({"name": "A"})) },
            BulkEntry { action: BulkAction::Delete, id: Some("bad".to_string()), data: None },
        ];
        let results = process_bulk(&entries, |entry| {
            match entry.action {
                BulkAction::Create => Ok("new-1".to_string()),
                BulkAction::Delete => Err("not found".to_string()),
                _ => Ok("ok".to_string()),
            }
        });
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert!(!results[1].success);
        assert_eq!(results[1].error.as_deref(), Some("not found"));
    }

    #[test]
    fn resource_definition_links() {
        let def = ResourceDefinition::new("users", "/api/users");
        let links = def.instance_links("123");
        assert!(links.iter().any(|l| l.rel == "self" && l.href == "/api/users/123"));
        assert!(links.iter().any(|l| l.rel == "collection"));
    }

    #[test]
    fn resource_definition_collection_links() {
        let def = ResourceDefinition::new("users", "/api/users");
        let links = def.collection_links();
        assert!(links.iter().any(|l| l.rel == "create"));
    }

    #[test]
    fn resource_definition_nested() {
        let mut def = ResourceDefinition::new("users", "/api/users");
        def.nested_resources.push("posts".to_string());
        let links = def.instance_links("1");
        assert!(links.iter().any(|l| l.rel == "posts"));
    }

    #[test]
    fn validate_sort_field() {
        let mut def = ResourceDefinition::new("users", "/users");
        def.sortable_fields = vec!["name".to_string(), "created_at".to_string()];
        assert!(def.validate_sort_field("name").is_ok());
        assert!(def.validate_sort_field("password").is_err());
    }

    #[test]
    fn link_builder() {
        let link = Link::new("edit", "/users/1")
            .with_method(HttpMethod::Put)
            .with_title("Edit user");
        assert_eq!(link.method, Some(HttpMethod::Put));
        assert_eq!(link.title.as_deref(), Some("Edit user"));
    }

    #[test]
    fn resource_error_display() {
        let err = ResourceError::NotFound {
            resource: "User".to_string(),
            id: "42".to_string(),
        };
        assert!(err.to_string().contains("User"));
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "GET");
        assert_eq!(HttpMethod::Post.to_string(), "POST");
    }
}
