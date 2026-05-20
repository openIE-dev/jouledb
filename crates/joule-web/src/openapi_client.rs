// openapi_client.rs — OpenAPI client builder: operation definitions,
// parameter types (path/query/header/body), request construction,
// type mapping, response type handling, client code generation plan.

use std::collections::HashMap;

/// Where a parameter lives in the HTTP request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterLocation {
    Path,
    Query,
    Header,
    Body,
}

impl std::fmt::Display for ParameterLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path => write!(f, "path"),
            Self::Query => write!(f, "query"),
            Self::Header => write!(f, "header"),
            Self::Body => write!(f, "body"),
        }
    }
}

/// OpenAPI-style types that parameters / responses can have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiType {
    String,
    Integer,
    Number,
    Boolean,
    Array(Box<ApiType>),
    Object(Vec<(String, ApiType)>),
}

impl ApiType {
    /// Map to a Rust-like type name (for code generation).
    pub fn rust_type_name(&self) -> String {
        match self {
            Self::String => "String".to_string(),
            Self::Integer => "i64".to_string(),
            Self::Number => "f64".to_string(),
            Self::Boolean => "bool".to_string(),
            Self::Array(inner) => format!("Vec<{}>", inner.rust_type_name()),
            Self::Object(fields) => {
                let mut s = "struct { ".to_string();
                for (name, ty) in fields {
                    s.push_str(&format!("{name}: {}, ", ty.rust_type_name()));
                }
                s.push('}');
                s
            }
        }
    }
}

/// A single parameter definition.
#[derive(Debug, Clone)]
pub struct ParameterDef {
    pub name: String,
    pub location: ParameterLocation,
    pub required: bool,
    pub api_type: ApiType,
    pub description: Option<String>,
}

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        };
        write!(f, "{s}")
    }
}

/// Response type for an operation.
#[derive(Debug, Clone)]
pub struct ResponseDef {
    pub status_code: u16,
    pub description: String,
    pub content_type: Option<String>,
    pub body_type: Option<ApiType>,
}

/// An API operation (one endpoint).
#[derive(Debug, Clone)]
pub struct OperationDef {
    pub operation_id: String,
    pub method: HttpMethod,
    pub path_template: String,
    pub summary: Option<String>,
    pub parameters: Vec<ParameterDef>,
    pub responses: Vec<ResponseDef>,
    pub tags: Vec<String>,
}

impl OperationDef {
    pub fn new(id: &str, method: HttpMethod, path: &str) -> Self {
        Self {
            operation_id: id.to_string(),
            method,
            path_template: path.to_string(),
            summary: None,
            parameters: Vec::new(),
            responses: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_summary(mut self, s: &str) -> Self {
        self.summary = Some(s.to_string());
        self
    }

    pub fn add_param(&mut self, p: ParameterDef) {
        self.parameters.push(p);
    }

    pub fn add_response(&mut self, r: ResponseDef) {
        self.responses.push(r);
    }

    pub fn path_params(&self) -> Vec<&ParameterDef> {
        self.parameters
            .iter()
            .filter(|p| p.location == ParameterLocation::Path)
            .collect()
    }

    pub fn query_params(&self) -> Vec<&ParameterDef> {
        self.parameters
            .iter()
            .filter(|p| p.location == ParameterLocation::Query)
            .collect()
    }

    pub fn header_params(&self) -> Vec<&ParameterDef> {
        self.parameters
            .iter()
            .filter(|p| p.location == ParameterLocation::Header)
            .collect()
    }

    pub fn body_param(&self) -> Option<&ParameterDef> {
        self.parameters
            .iter()
            .find(|p| p.location == ParameterLocation::Body)
    }

    pub fn required_params(&self) -> Vec<&ParameterDef> {
        self.parameters.iter().filter(|p| p.required).collect()
    }
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

/// A concrete request built from an operation + parameter values.
#[derive(Debug, Clone)]
pub struct BuiltRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub query_pairs: Vec<(String, String)>,
    pub body: Option<String>,
}

impl BuiltRequest {
    /// Full URL with query string appended.
    pub fn full_url(&self) -> String {
        if self.query_pairs.is_empty() {
            return self.url.clone();
        }
        let qs: Vec<String> = self
            .query_pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{}?{}", self.url, qs.join("&"))
    }
}

/// Build a request from an operation definition + supplied parameter values.
pub fn build_request(
    base_url: &str,
    op: &OperationDef,
    values: &HashMap<String, String>,
) -> Result<BuiltRequest, BuildError> {
    // Check required params.
    for p in &op.parameters {
        if p.required && !values.contains_key(&p.name) {
            return Err(BuildError::MissingRequired(p.name.clone()));
        }
    }

    // Expand path template.
    let mut path = op.path_template.clone();
    for p in op.path_params() {
        if let Some(val) = values.get(&p.name) {
            let placeholder = format!("{{{}}}", p.name);
            path = path.replace(&placeholder, val);
        }
    }

    let url = format!(
        "{}{}",
        base_url.trim_end_matches('/'),
        path
    );

    let mut headers = HashMap::new();
    for p in op.header_params() {
        if let Some(val) = values.get(&p.name) {
            headers.insert(p.name.clone(), val.clone());
        }
    }

    let query_pairs: Vec<(String, String)> = op
        .query_params()
        .iter()
        .filter_map(|p| values.get(&p.name).map(|v| (p.name.clone(), v.clone())))
        .collect();

    let body = op
        .body_param()
        .and_then(|p| values.get(&p.name).cloned());

    Ok(BuiltRequest {
        method: op.method,
        url,
        headers,
        query_pairs,
        body,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    MissingRequired(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRequired(name) => write!(f, "missing required parameter: {name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Client code generation plan
// ---------------------------------------------------------------------------

/// A plan for generating a typed client from a set of operations.
#[derive(Debug, Clone)]
pub struct ClientCodePlan {
    pub client_name: String,
    pub base_url: String,
    pub operations: Vec<OperationDef>,
}

impl ClientCodePlan {
    pub fn new(name: &str, base_url: &str) -> Self {
        Self {
            client_name: name.to_string(),
            base_url: base_url.to_string(),
            operations: Vec::new(),
        }
    }

    pub fn add_operation(&mut self, op: OperationDef) {
        self.operations.push(op);
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn operations_by_tag(&self, tag: &str) -> Vec<&OperationDef> {
        self.operations
            .iter()
            .filter(|op| op.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Generate method signatures (as strings) for all operations.
    pub fn method_signatures(&self) -> Vec<String> {
        self.operations
            .iter()
            .map(|op| {
                let params: Vec<String> = op
                    .parameters
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.api_type.rust_type_name()))
                    .collect();
                let ret = op
                    .responses
                    .first()
                    .and_then(|r| r.body_type.as_ref())
                    .map(|t| t.rust_type_name())
                    .unwrap_or_else(|| "()".to_string());
                format!(
                    "fn {}({}) -> Result<{}, Error>",
                    op.operation_id,
                    params.join(", "),
                    ret
                )
            })
            .collect()
    }

    /// Collect all unique tags across operations (sorted).
    pub fn all_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = self
            .operations
            .iter()
            .flat_map(|op| op.tags.iter().cloned())
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }
}

// ---------------------------------------------------------------------------
// Type mapping helpers
// ---------------------------------------------------------------------------

/// Map an OpenAPI type string to our ApiType.
pub fn map_openapi_type(type_str: &str, format: Option<&str>) -> ApiType {
    match type_str {
        "string" => ApiType::String,
        "integer" => match format {
            Some("int32") => ApiType::Integer,
            _ => ApiType::Integer,
        },
        "number" => ApiType::Number,
        "boolean" => ApiType::Boolean,
        "array" => ApiType::Array(Box::new(ApiType::String)), // default item type
        _ => ApiType::String,
    }
}

pub fn map_openapi_type_with_items(type_str: &str, item_type: ApiType) -> ApiType {
    if type_str == "array" {
        ApiType::Array(Box::new(item_type))
    } else {
        map_openapi_type(type_str, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_op() -> OperationDef {
        let mut op = OperationDef::new("getUser", HttpMethod::Get, "/users/{user_id}");
        op.add_param(ParameterDef {
            name: "user_id".into(),
            location: ParameterLocation::Path,
            required: true,
            api_type: ApiType::String,
            description: Some("The user ID".into()),
        });
        op.add_param(ParameterDef {
            name: "fields".into(),
            location: ParameterLocation::Query,
            required: false,
            api_type: ApiType::String,
            description: None,
        });
        op.add_param(ParameterDef {
            name: "Authorization".into(),
            location: ParameterLocation::Header,
            required: true,
            api_type: ApiType::String,
            description: None,
        });
        op.add_response(ResponseDef {
            status_code: 200,
            description: "OK".into(),
            content_type: Some("application/json".into()),
            body_type: Some(ApiType::Object(vec![
                ("id".into(), ApiType::String),
                ("name".into(), ApiType::String),
            ])),
        });
        op.tags.push("users".into());
        op
    }

    #[test]
    fn test_operation_param_filters() {
        let op = sample_op();
        assert_eq!(op.path_params().len(), 1);
        assert_eq!(op.query_params().len(), 1);
        assert_eq!(op.header_params().len(), 1);
        assert!(op.body_param().is_none());
        assert_eq!(op.required_params().len(), 2);
    }

    #[test]
    fn test_build_request_ok() {
        let op = sample_op();
        let mut vals = HashMap::new();
        vals.insert("user_id".into(), "42".into());
        vals.insert("Authorization".into(), "Bearer tok".into());
        vals.insert("fields".into(), "name,email".into());

        let req = build_request("https://api.example.com", &op, &vals).unwrap();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.url, "https://api.example.com/users/42");
        assert_eq!(req.query_pairs.len(), 1);
        assert_eq!(req.query_pairs[0].0, "fields");
        assert_eq!(req.headers.get("Authorization").unwrap(), "Bearer tok");
        assert!(req.body.is_none());
    }

    #[test]
    fn test_build_request_missing_required() {
        let op = sample_op();
        let vals = HashMap::new();
        let err = build_request("https://api.example.com", &op, &vals).unwrap_err();
        assert_eq!(err, BuildError::MissingRequired("user_id".into()));
    }

    #[test]
    fn test_full_url_with_query() {
        let req = BuiltRequest {
            method: HttpMethod::Get,
            url: "https://api.example.com/items".into(),
            headers: HashMap::new(),
            query_pairs: vec![
                ("page".into(), "1".into()),
                ("limit".into(), "10".into()),
            ],
            body: None,
        };
        assert_eq!(req.full_url(), "https://api.example.com/items?page=1&limit=10");
    }

    #[test]
    fn test_full_url_without_query() {
        let req = BuiltRequest {
            method: HttpMethod::Get,
            url: "https://api.example.com/items".into(),
            headers: HashMap::new(),
            query_pairs: vec![],
            body: None,
        };
        assert_eq!(req.full_url(), "https://api.example.com/items");
    }

    #[test]
    fn test_api_type_rust_names() {
        assert_eq!(ApiType::String.rust_type_name(), "String");
        assert_eq!(ApiType::Integer.rust_type_name(), "i64");
        assert_eq!(ApiType::Number.rust_type_name(), "f64");
        assert_eq!(ApiType::Boolean.rust_type_name(), "bool");
        assert_eq!(
            ApiType::Array(Box::new(ApiType::String)).rust_type_name(),
            "Vec<String>"
        );
    }

    #[test]
    fn test_map_openapi_type() {
        assert_eq!(map_openapi_type("string", None), ApiType::String);
        assert_eq!(map_openapi_type("integer", Some("int32")), ApiType::Integer);
        assert_eq!(map_openapi_type("number", None), ApiType::Number);
        assert_eq!(map_openapi_type("boolean", None), ApiType::Boolean);
    }

    #[test]
    fn test_map_openapi_type_with_items() {
        let t = map_openapi_type_with_items("array", ApiType::Integer);
        assert_eq!(t, ApiType::Array(Box::new(ApiType::Integer)));
    }

    #[test]
    fn test_client_code_plan() {
        let mut plan = ClientCodePlan::new("UserClient", "https://api.example.com");
        plan.add_operation(sample_op());
        assert_eq!(plan.operation_count(), 1);
        assert_eq!(plan.all_tags(), vec!["users".to_string()]);
        assert_eq!(plan.operations_by_tag("users").len(), 1);
        assert_eq!(plan.operations_by_tag("other").len(), 0);
    }

    #[test]
    fn test_method_signatures() {
        let mut plan = ClientCodePlan::new("C", "https://x");
        plan.add_operation(sample_op());
        let sigs = plan.method_signatures();
        assert_eq!(sigs.len(), 1);
        assert!(sigs[0].contains("getUser"));
        assert!(sigs[0].contains("user_id: String"));
    }

    #[test]
    fn test_parameter_location_display() {
        assert_eq!(format!("{}", ParameterLocation::Path), "path");
        assert_eq!(format!("{}", ParameterLocation::Query), "query");
        assert_eq!(format!("{}", ParameterLocation::Header), "header");
        assert_eq!(format!("{}", ParameterLocation::Body), "body");
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(format!("{}", HttpMethod::Get), "GET");
        assert_eq!(format!("{}", HttpMethod::Post), "POST");
        assert_eq!(format!("{}", HttpMethod::Delete), "DELETE");
    }

    #[test]
    fn test_build_error_display() {
        let e = BuildError::MissingRequired("id".into());
        assert_eq!(e.to_string(), "missing required parameter: id");
    }

    #[test]
    fn test_operation_with_body() {
        let mut op = OperationDef::new("createUser", HttpMethod::Post, "/users");
        op.add_param(ParameterDef {
            name: "body".into(),
            location: ParameterLocation::Body,
            required: true,
            api_type: ApiType::Object(vec![("name".into(), ApiType::String)]),
            description: None,
        });
        assert!(op.body_param().is_some());

        let mut vals = HashMap::new();
        vals.insert("body".into(), r#"{"name":"Alice"}"#.into());
        let req = build_request("https://api.example.com", &op, &vals).unwrap();
        assert_eq!(req.body.as_deref(), Some(r#"{"name":"Alice"}"#));
    }

    #[test]
    fn test_operation_with_summary() {
        let op = OperationDef::new("listItems", HttpMethod::Get, "/items")
            .with_summary("List all items");
        assert_eq!(op.summary.as_deref(), Some("List all items"));
    }

    #[test]
    fn test_base_url_trailing_slash() {
        let op = OperationDef::new("root", HttpMethod::Get, "/health");
        let req = build_request("https://api.example.com/", &op, &HashMap::new()).unwrap();
        assert_eq!(req.url, "https://api.example.com/health");
    }

    #[test]
    fn test_object_type_rust_name() {
        let t = ApiType::Object(vec![
            ("id".into(), ApiType::Integer),
            ("name".into(), ApiType::String),
        ]);
        let name = t.rust_type_name();
        assert!(name.contains("id: i64"));
        assert!(name.contains("name: String"));
    }

    #[test]
    fn test_nested_array_type() {
        let t = ApiType::Array(Box::new(ApiType::Array(Box::new(ApiType::Integer))));
        assert_eq!(t.rust_type_name(), "Vec<Vec<i64>>");
    }
}
