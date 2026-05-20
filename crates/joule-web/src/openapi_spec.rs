//! OpenAPI 3.1 specification builder — paths, operations, schemas, parameters,
//! request bodies, responses, components, JSON Schema `$ref`, spec serialization.
//!
//! Pure-Rust replacement for swagger-codegen, openapi-generator, redocly, etc.

use std::collections::BTreeMap;
use std::fmt;

// ── Schema types ──────────────────────────────────────────────────

/// JSON Schema types used inside OpenAPI component schemas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
}

impl fmt::Display for SchemaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Integer => write!(f, "integer"),
            Self::Number => write!(f, "number"),
            Self::Boolean => write!(f, "boolean"),
            Self::Array => write!(f, "array"),
            Self::Object => write!(f, "object"),
        }
    }
}

/// A JSON Schema definition (subset used by OpenAPI 3.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub schema_type: Option<SchemaType>,
    pub format: Option<String>,
    pub description: Option<String>,
    pub required: Vec<String>,
    pub properties: BTreeMap<String, Schema>,
    pub items: Option<Box<Schema>>,
    pub ref_path: Option<String>,
    pub enum_values: Vec<String>,
    pub nullable: bool,
    pub example: Option<String>,
    pub min_length: Option<u64>,
    pub max_length: Option<u64>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub pattern: Option<String>,
    pub default_value: Option<String>,
    pub one_of: Vec<Schema>,
    pub any_of: Vec<Schema>,
    pub all_of: Vec<Schema>,
}

impl Default for Schema {
    fn default() -> Self {
        Self {
            schema_type: None,
            format: None,
            description: None,
            required: Vec::new(),
            properties: BTreeMap::new(),
            items: None,
            ref_path: None,
            enum_values: Vec::new(),
            nullable: false,
            example: None,
            min_length: None,
            max_length: None,
            minimum: None,
            maximum: None,
            pattern: None,
            default_value: None,
            one_of: Vec::new(),
            any_of: Vec::new(),
            all_of: Vec::new(),
        }
    }
}

impl Schema {
    /// Create a `$ref` pointer to a component schema.
    pub fn reference(path: &str) -> Self {
        Self { ref_path: Some(path.to_string()), ..Default::default() }
    }

    /// Shorthand for a typed schema.
    pub fn typed(st: SchemaType) -> Self {
        Self { schema_type: Some(st), ..Default::default() }
    }

    /// Shorthand for an array of items.
    pub fn array_of(item: Schema) -> Self {
        Self {
            schema_type: Some(SchemaType::Array),
            items: Some(Box::new(item)),
            ..Default::default()
        }
    }

    /// Serialize to a JSON value (BTreeMap-based for deterministic order).
    pub fn to_json(&self) -> serde_json::Value {
        if let Some(ref r) = self.ref_path {
            let mut m = serde_json::Map::new();
            m.insert("$ref".into(), serde_json::Value::String(r.clone()));
            return serde_json::Value::Object(m);
        }
        let mut m = serde_json::Map::new();
        if let Some(ref st) = self.schema_type {
            m.insert("type".into(), serde_json::Value::String(st.to_string()));
        }
        if let Some(ref fmt) = self.format {
            m.insert("format".into(), serde_json::Value::String(fmt.clone()));
        }
        if let Some(ref desc) = self.description {
            m.insert("description".into(), serde_json::Value::String(desc.clone()));
        }
        if !self.required.is_empty() {
            let arr: Vec<serde_json::Value> = self.required.iter()
                .map(|s| serde_json::Value::String(s.clone())).collect();
            m.insert("required".into(), serde_json::Value::Array(arr));
        }
        if !self.properties.is_empty() {
            let mut props = serde_json::Map::new();
            for (k, v) in &self.properties {
                props.insert(k.clone(), v.to_json());
            }
            m.insert("properties".into(), serde_json::Value::Object(props));
        }
        if let Some(ref items) = self.items {
            m.insert("items".into(), items.to_json());
        }
        if !self.enum_values.is_empty() {
            let arr: Vec<serde_json::Value> = self.enum_values.iter()
                .map(|s| serde_json::Value::String(s.clone())).collect();
            m.insert("enum".into(), serde_json::Value::Array(arr));
        }
        if self.nullable {
            m.insert("nullable".into(), serde_json::Value::Bool(true));
        }
        if let Some(ref ex) = self.example {
            m.insert("example".into(), serde_json::Value::String(ex.clone()));
        }
        if let Some(v) = self.min_length {
            m.insert("minLength".into(), serde_json::Value::Number(v.into()));
        }
        if let Some(v) = self.max_length {
            m.insert("maxLength".into(), serde_json::Value::Number(v.into()));
        }
        if let Some(v) = self.minimum {
            if let Some(n) = serde_json::Number::from_f64(v) {
                m.insert("minimum".into(), serde_json::Value::Number(n));
            }
        }
        if let Some(v) = self.maximum {
            if let Some(n) = serde_json::Number::from_f64(v) {
                m.insert("maximum".into(), serde_json::Value::Number(n));
            }
        }
        if let Some(ref p) = self.pattern {
            m.insert("pattern".into(), serde_json::Value::String(p.clone()));
        }
        if let Some(ref d) = self.default_value {
            m.insert("default".into(), serde_json::Value::String(d.clone()));
        }
        if !self.one_of.is_empty() {
            let arr: Vec<serde_json::Value> = self.one_of.iter().map(|s| s.to_json()).collect();
            m.insert("oneOf".into(), serde_json::Value::Array(arr));
        }
        if !self.any_of.is_empty() {
            let arr: Vec<serde_json::Value> = self.any_of.iter().map(|s| s.to_json()).collect();
            m.insert("anyOf".into(), serde_json::Value::Array(arr));
        }
        if !self.all_of.is_empty() {
            let arr: Vec<serde_json::Value> = self.all_of.iter().map(|s| s.to_json()).collect();
            m.insert("allOf".into(), serde_json::Value::Array(arr));
        }
        serde_json::Value::Object(m)
    }
}

// ── Parameter ─────────────────────────────────────────────────────

/// Where a parameter lives in the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterIn {
    Query,
    Header,
    Path,
    Cookie,
}

impl fmt::Display for ParameterIn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query => write!(f, "query"),
            Self::Header => write!(f, "header"),
            Self::Path => write!(f, "path"),
            Self::Cookie => write!(f, "cookie"),
        }
    }
}

/// An operation parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub location: ParameterIn,
    pub description: Option<String>,
    pub required: bool,
    pub schema: Schema,
    pub deprecated: bool,
}

impl Parameter {
    pub fn new(name: &str, location: ParameterIn, schema: Schema) -> Self {
        let required = matches!(location, ParameterIn::Path);
        Self { name: name.into(), location, description: None, required, schema, deprecated: false }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("name".into(), serde_json::Value::String(self.name.clone()));
        m.insert("in".into(), serde_json::Value::String(self.location.to_string()));
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        m.insert("required".into(), serde_json::Value::Bool(self.required));
        m.insert("schema".into(), self.schema.to_json());
        if self.deprecated {
            m.insert("deprecated".into(), serde_json::Value::Bool(true));
        }
        serde_json::Value::Object(m)
    }
}

// ── Request Body ──────────────────────────────────────────────────

/// Media type content.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaType {
    pub schema: Schema,
}

/// A request body definition.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestBody {
    pub description: Option<String>,
    pub required: bool,
    pub content: BTreeMap<String, MediaType>,
}

impl RequestBody {
    pub fn json(schema: Schema) -> Self {
        let mut content = BTreeMap::new();
        content.insert("application/json".into(), MediaType { schema });
        Self { description: None, required: true, content }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        m.insert("required".into(), serde_json::Value::Bool(self.required));
        let mut ct = serde_json::Map::new();
        for (mime, mt) in &self.content {
            let mut mt_obj = serde_json::Map::new();
            mt_obj.insert("schema".into(), mt.schema.to_json());
            ct.insert(mime.clone(), serde_json::Value::Object(mt_obj));
        }
        m.insert("content".into(), serde_json::Value::Object(ct));
        serde_json::Value::Object(m)
    }
}

// ── Response ──────────────────────────────────────────────────────

/// An HTTP response definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub description: String,
    pub content: BTreeMap<String, MediaType>,
    pub headers: BTreeMap<String, Schema>,
}

impl Response {
    pub fn new(description: &str) -> Self {
        Self { description: description.into(), content: BTreeMap::new(), headers: BTreeMap::new() }
    }

    pub fn with_json(description: &str, schema: Schema) -> Self {
        let mut content = BTreeMap::new();
        content.insert("application/json".into(), MediaType { schema });
        Self { description: description.into(), content, headers: BTreeMap::new() }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("description".into(), serde_json::Value::String(self.description.clone()));
        if !self.content.is_empty() {
            let mut ct = serde_json::Map::new();
            for (mime, mt) in &self.content {
                let mut mt_obj = serde_json::Map::new();
                mt_obj.insert("schema".into(), mt.schema.to_json());
                ct.insert(mime.clone(), serde_json::Value::Object(mt_obj));
            }
            m.insert("content".into(), serde_json::Value::Object(ct));
        }
        if !self.headers.is_empty() {
            let mut hdrs = serde_json::Map::new();
            for (k, v) in &self.headers {
                let mut hdr = serde_json::Map::new();
                hdr.insert("schema".into(), v.to_json());
                hdrs.insert(k.clone(), serde_json::Value::Object(hdr));
            }
            m.insert("headers".into(), serde_json::Value::Object(hdrs));
        }
        serde_json::Value::Object(m)
    }
}

// ── Operation ─────────────────────────────────────────────────────

/// An HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get, Post, Put, Delete, Patch, Head, Options, Trace,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get => write!(f, "get"),
            Self::Post => write!(f, "post"),
            Self::Put => write!(f, "put"),
            Self::Delete => write!(f, "delete"),
            Self::Patch => write!(f, "patch"),
            Self::Head => write!(f, "head"),
            Self::Options => write!(f, "options"),
            Self::Trace => write!(f, "trace"),
        }
    }
}

/// A single API operation (e.g., GET /pets).
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub operation_id: Option<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub parameters: Vec<Parameter>,
    pub request_body: Option<RequestBody>,
    pub responses: BTreeMap<String, Response>,
    pub deprecated: bool,
    pub security: Vec<BTreeMap<String, Vec<String>>>,
}

impl Default for Operation {
    fn default() -> Self {
        Self {
            operation_id: None, summary: None, description: None,
            tags: Vec::new(), parameters: Vec::new(), request_body: None,
            responses: BTreeMap::new(), deprecated: false, security: Vec::new(),
        }
    }
}

impl Operation {
    pub fn new() -> Self { Self::default() }

    pub fn with_id(mut self, id: &str) -> Self { self.operation_id = Some(id.into()); self }
    pub fn with_summary(mut self, s: &str) -> Self { self.summary = Some(s.into()); self }
    pub fn with_tag(mut self, t: &str) -> Self { self.tags.push(t.into()); self }

    pub fn with_parameter(mut self, p: Parameter) -> Self {
        self.parameters.push(p); self
    }

    pub fn with_body(mut self, b: RequestBody) -> Self {
        self.request_body = Some(b); self
    }

    pub fn with_response(mut self, status: &str, r: Response) -> Self {
        self.responses.insert(status.into(), r); self
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if let Some(ref id) = self.operation_id {
            m.insert("operationId".into(), serde_json::Value::String(id.clone()));
        }
        if let Some(ref s) = self.summary {
            m.insert("summary".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        if !self.tags.is_empty() {
            let arr: Vec<serde_json::Value> = self.tags.iter()
                .map(|t| serde_json::Value::String(t.clone())).collect();
            m.insert("tags".into(), serde_json::Value::Array(arr));
        }
        if !self.parameters.is_empty() {
            let arr: Vec<serde_json::Value> = self.parameters.iter()
                .map(|p| p.to_json()).collect();
            m.insert("parameters".into(), serde_json::Value::Array(arr));
        }
        if let Some(ref rb) = self.request_body {
            m.insert("requestBody".into(), rb.to_json());
        }
        if !self.responses.is_empty() {
            let mut resp = serde_json::Map::new();
            for (status, r) in &self.responses {
                resp.insert(status.clone(), r.to_json());
            }
            m.insert("responses".into(), serde_json::Value::Object(resp));
        }
        if self.deprecated {
            m.insert("deprecated".into(), serde_json::Value::Bool(true));
        }
        if !self.security.is_empty() {
            let arr: Vec<serde_json::Value> = self.security.iter().map(|sec| {
                let mut sm = serde_json::Map::new();
                for (k, v) in sec {
                    let scopes: Vec<serde_json::Value> = v.iter()
                        .map(|s| serde_json::Value::String(s.clone())).collect();
                    sm.insert(k.clone(), serde_json::Value::Array(scopes));
                }
                serde_json::Value::Object(sm)
            }).collect();
            m.insert("security".into(), serde_json::Value::Array(arr));
        }
        serde_json::Value::Object(m)
    }
}

// ── Path Item ─────────────────────────────────────────────────────

/// A path item containing operations for each HTTP method.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathItem {
    pub operations: BTreeMap<String, Operation>,
    pub summary: Option<String>,
    pub description: Option<String>,
}

impl PathItem {
    pub fn new() -> Self { Self::default() }

    pub fn operation(mut self, method: HttpMethod, op: Operation) -> Self {
        self.operations.insert(method.to_string(), op);
        self
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if let Some(ref s) = self.summary {
            m.insert("summary".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        for (method, op) in &self.operations {
            m.insert(method.clone(), op.to_json());
        }
        serde_json::Value::Object(m)
    }
}

// ── Security Scheme ───────────────────────────────────────────────

/// Security scheme type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecuritySchemeType {
    ApiKey { name: String, location: ParameterIn },
    Http { scheme: String, bearer_format: Option<String> },
    OAuth2,
    OpenIdConnect { url: String },
}

/// A security scheme definition.
#[derive(Debug, Clone, PartialEq)]
pub struct SecurityScheme {
    pub scheme_type: SecuritySchemeType,
    pub description: Option<String>,
}

impl SecurityScheme {
    pub fn bearer_jwt() -> Self {
        Self {
            scheme_type: SecuritySchemeType::Http {
                scheme: "bearer".into(),
                bearer_format: Some("JWT".into()),
            },
            description: Some("JWT Bearer token".into()),
        }
    }

    pub fn api_key(name: &str, location: ParameterIn) -> Self {
        Self {
            scheme_type: SecuritySchemeType::ApiKey {
                name: name.into(), location,
            },
            description: None,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        match &self.scheme_type {
            SecuritySchemeType::ApiKey { name, location } => {
                m.insert("type".into(), serde_json::Value::String("apiKey".into()));
                m.insert("name".into(), serde_json::Value::String(name.clone()));
                m.insert("in".into(), serde_json::Value::String(location.to_string()));
            }
            SecuritySchemeType::Http { scheme, bearer_format } => {
                m.insert("type".into(), serde_json::Value::String("http".into()));
                m.insert("scheme".into(), serde_json::Value::String(scheme.clone()));
                if let Some(bf) = bearer_format {
                    m.insert("bearerFormat".into(), serde_json::Value::String(bf.clone()));
                }
            }
            SecuritySchemeType::OAuth2 => {
                m.insert("type".into(), serde_json::Value::String("oauth2".into()));
            }
            SecuritySchemeType::OpenIdConnect { url } => {
                m.insert("type".into(), serde_json::Value::String("openIdConnect".into()));
                m.insert("openIdConnectUrl".into(), serde_json::Value::String(url.clone()));
            }
        }
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        serde_json::Value::Object(m)
    }
}

// ── Info / Server / Tag ───────────────────────────────────────────

/// API info block.
#[derive(Debug, Clone, PartialEq)]
pub struct Info {
    pub title: String,
    pub version: String,
    pub description: Option<String>,
    pub terms_of_service: Option<String>,
    pub license_name: Option<String>,
    pub license_url: Option<String>,
}

impl Info {
    pub fn new(title: &str, version: &str) -> Self {
        Self {
            title: title.into(), version: version.into(),
            description: None, terms_of_service: None,
            license_name: None, license_url: None,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("title".into(), serde_json::Value::String(self.title.clone()));
        m.insert("version".into(), serde_json::Value::String(self.version.clone()));
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        if let Some(ref t) = self.terms_of_service {
            m.insert("termsOfService".into(), serde_json::Value::String(t.clone()));
        }
        if self.license_name.is_some() || self.license_url.is_some() {
            let mut lic = serde_json::Map::new();
            if let Some(ref n) = self.license_name {
                lic.insert("name".into(), serde_json::Value::String(n.clone()));
            }
            if let Some(ref u) = self.license_url {
                lic.insert("url".into(), serde_json::Value::String(u.clone()));
            }
            m.insert("license".into(), serde_json::Value::Object(lic));
        }
        serde_json::Value::Object(m)
    }
}

/// A server entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Server {
    pub url: String,
    pub description: Option<String>,
}

impl Server {
    pub fn new(url: &str) -> Self { Self { url: url.into(), description: None } }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("url".into(), serde_json::Value::String(self.url.clone()));
        if let Some(ref d) = self.description {
            m.insert("description".into(), serde_json::Value::String(d.clone()));
        }
        serde_json::Value::Object(m)
    }
}

/// A tag.
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    pub name: String,
    pub description: Option<String>,
}

impl Tag {
    pub fn new(name: &str) -> Self { Self { name: name.into(), description: None } }
}

// ── OpenAPI Spec ──────────────────────────────────────────────────

/// The top-level OpenAPI 3.1 specification.
#[derive(Debug, Clone)]
pub struct OpenApiSpec {
    pub openapi: String,
    pub info: Info,
    pub servers: Vec<Server>,
    pub paths: BTreeMap<String, PathItem>,
    pub components_schemas: BTreeMap<String, Schema>,
    pub components_security_schemes: BTreeMap<String, SecurityScheme>,
    pub tags: Vec<Tag>,
}

impl OpenApiSpec {
    pub fn new(info: Info) -> Self {
        Self {
            openapi: "3.1.0".into(),
            info,
            servers: Vec::new(),
            paths: BTreeMap::new(),
            components_schemas: BTreeMap::new(),
            components_security_schemes: BTreeMap::new(),
            tags: Vec::new(),
        }
    }

    pub fn add_server(&mut self, server: Server) { self.servers.push(server); }

    pub fn add_path(&mut self, path: &str, item: PathItem) {
        self.paths.insert(path.into(), item);
    }

    pub fn add_schema(&mut self, name: &str, schema: Schema) {
        self.components_schemas.insert(name.into(), schema);
    }

    pub fn add_security_scheme(&mut self, name: &str, scheme: SecurityScheme) {
        self.components_security_schemes.insert(name.into(), scheme);
    }

    pub fn add_tag(&mut self, tag: Tag) { self.tags.push(tag); }

    /// Validate the spec: checks refs resolve, path params exist, etc.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        self.validate_schema_refs(&mut errors);
        self.validate_path_params(&mut errors);
        self.validate_operation_ids(&mut errors);
        errors
    }

    fn validate_schema_refs(&self, errors: &mut Vec<String>) {
        for (path, item) in &self.paths {
            for (method, op) in &item.operations {
                for param in &op.parameters {
                    self.check_ref(&param.schema, &format!("{path} {method} param {}", param.name), errors);
                }
                if let Some(ref rb) = op.request_body {
                    for (mime, mt) in &rb.content {
                        self.check_ref(&mt.schema, &format!("{path} {method} body {mime}"), errors);
                    }
                }
                for (status, resp) in &op.responses {
                    for (mime, mt) in &resp.content {
                        self.check_ref(&mt.schema, &format!("{path} {method} {status} {mime}"), errors);
                    }
                }
            }
        }
    }

    fn check_ref(&self, schema: &Schema, context: &str, errors: &mut Vec<String>) {
        if let Some(ref r) = schema.ref_path {
            let prefix = "#/components/schemas/";
            if let Some(name) = r.strip_prefix(prefix) {
                if !self.components_schemas.contains_key(name) {
                    errors.push(format!("{context}: unresolved $ref '{r}'"));
                }
            }
        }
        for (_, prop) in &schema.properties {
            self.check_ref(prop, context, errors);
        }
        if let Some(ref items) = schema.items {
            self.check_ref(items, context, errors);
        }
        for s in &schema.one_of { self.check_ref(s, context, errors); }
        for s in &schema.any_of { self.check_ref(s, context, errors); }
        for s in &schema.all_of { self.check_ref(s, context, errors); }
    }

    fn validate_path_params(&self, errors: &mut Vec<String>) {
        for (path, item) in &self.paths {
            let template_params: Vec<&str> = path.split('{')
                .skip(1)
                .filter_map(|s| s.split('}').next())
                .collect();
            for (method, op) in &item.operations {
                for tp in &template_params {
                    let found = op.parameters.iter()
                        .any(|p| p.name == *tp && p.location == ParameterIn::Path);
                    if !found {
                        errors.push(format!("{path} {method}: path param '{tp}' not declared"));
                    }
                }
            }
        }
    }

    fn validate_operation_ids(&self, errors: &mut Vec<String>) {
        let mut seen = std::collections::HashSet::new();
        for (path, item) in &self.paths {
            for (method, op) in &item.operations {
                if let Some(ref id) = op.operation_id {
                    if !seen.insert(id.clone()) {
                        errors.push(format!("{path} {method}: duplicate operationId '{id}'"));
                    }
                }
            }
        }
    }

    /// Serialize the entire spec to a JSON string.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_json()).unwrap_or_default()
    }

    /// Serialize the entire spec to a `serde_json::Value`.
    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("openapi".into(), serde_json::Value::String(self.openapi.clone()));
        m.insert("info".into(), self.info.to_json());
        if !self.servers.is_empty() {
            let arr: Vec<serde_json::Value> = self.servers.iter().map(|s| s.to_json()).collect();
            m.insert("servers".into(), serde_json::Value::Array(arr));
        }
        if !self.paths.is_empty() {
            let mut paths = serde_json::Map::new();
            for (p, item) in &self.paths {
                paths.insert(p.clone(), item.to_json());
            }
            m.insert("paths".into(), serde_json::Value::Object(paths));
        }
        let mut components = serde_json::Map::new();
        if !self.components_schemas.is_empty() {
            let mut schemas = serde_json::Map::new();
            for (k, v) in &self.components_schemas {
                schemas.insert(k.clone(), v.to_json());
            }
            components.insert("schemas".into(), serde_json::Value::Object(schemas));
        }
        if !self.components_security_schemes.is_empty() {
            let mut sec = serde_json::Map::new();
            for (k, v) in &self.components_security_schemes {
                sec.insert(k.clone(), v.to_json());
            }
            components.insert("securitySchemes".into(), serde_json::Value::Object(sec));
        }
        if !components.is_empty() {
            m.insert("components".into(), serde_json::Value::Object(components));
        }
        if !self.tags.is_empty() {
            let arr: Vec<serde_json::Value> = self.tags.iter().map(|t| {
                let mut tm = serde_json::Map::new();
                tm.insert("name".into(), serde_json::Value::String(t.name.clone()));
                if let Some(ref d) = t.description {
                    tm.insert("description".into(), serde_json::Value::String(d.clone()));
                }
                serde_json::Value::Object(tm)
            }).collect();
            m.insert("tags".into(), serde_json::Value::Array(arr));
        }
        serde_json::Value::Object(m)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_string_type() {
        let s = Schema::typed(SchemaType::String);
        let j = s.to_json();
        assert_eq!(j["type"], "string");
    }

    #[test]
    fn schema_ref() {
        let s = Schema::reference("#/components/schemas/Pet");
        let j = s.to_json();
        assert_eq!(j["$ref"], "#/components/schemas/Pet");
        assert!(j.get("type").is_none());
    }

    #[test]
    fn schema_array_of_ref() {
        let s = Schema::array_of(Schema::reference("#/components/schemas/Pet"));
        let j = s.to_json();
        assert_eq!(j["type"], "array");
        assert_eq!(j["items"]["$ref"], "#/components/schemas/Pet");
    }

    #[test]
    fn schema_object_with_properties() {
        let mut s = Schema::typed(SchemaType::Object);
        s.required = vec!["name".into()];
        s.properties.insert("name".into(), Schema::typed(SchemaType::String));
        s.properties.insert("age".into(), Schema::typed(SchemaType::Integer));
        let j = s.to_json();
        assert_eq!(j["type"], "object");
        assert_eq!(j["properties"]["name"]["type"], "string");
        assert_eq!(j["properties"]["age"]["type"], "integer");
        assert_eq!(j["required"][0], "name");
    }

    #[test]
    fn schema_enum_values() {
        let mut s = Schema::typed(SchemaType::String);
        s.enum_values = vec!["active".into(), "inactive".into()];
        let j = s.to_json();
        assert_eq!(j["enum"][0], "active");
        assert_eq!(j["enum"][1], "inactive");
    }

    #[test]
    fn schema_nullable() {
        let mut s = Schema::typed(SchemaType::String);
        s.nullable = true;
        let j = s.to_json();
        assert_eq!(j["nullable"], true);
    }

    #[test]
    fn schema_constraints() {
        let mut s = Schema::typed(SchemaType::String);
        s.min_length = Some(1);
        s.max_length = Some(255);
        s.pattern = Some("^[a-z]+$".into());
        let j = s.to_json();
        assert_eq!(j["minLength"], 1);
        assert_eq!(j["maxLength"], 255);
        assert_eq!(j["pattern"], "^[a-z]+$");
    }

    #[test]
    fn schema_numeric_constraints() {
        let mut s = Schema::typed(SchemaType::Number);
        s.minimum = Some(0.0);
        s.maximum = Some(100.0);
        let j = s.to_json();
        assert_eq!(j["minimum"], 0.0);
        assert_eq!(j["maximum"], 100.0);
    }

    #[test]
    fn schema_one_of() {
        let mut s = Schema::default();
        s.one_of = vec![
            Schema::reference("#/components/schemas/Cat"),
            Schema::reference("#/components/schemas/Dog"),
        ];
        let j = s.to_json();
        assert_eq!(j["oneOf"][0]["$ref"], "#/components/schemas/Cat");
        assert_eq!(j["oneOf"][1]["$ref"], "#/components/schemas/Dog");
    }

    #[test]
    fn parameter_path() {
        let p = Parameter::new("petId", ParameterIn::Path, Schema::typed(SchemaType::String));
        assert!(p.required);
        let j = p.to_json();
        assert_eq!(j["in"], "path");
        assert_eq!(j["name"], "petId");
    }

    #[test]
    fn parameter_query() {
        let p = Parameter::new("limit", ParameterIn::Query, Schema::typed(SchemaType::Integer));
        assert!(!p.required);
        let j = p.to_json();
        assert_eq!(j["in"], "query");
    }

    #[test]
    fn request_body_json() {
        let rb = RequestBody::json(Schema::reference("#/components/schemas/Pet"));
        let j = rb.to_json();
        assert_eq!(j["required"], true);
        assert_eq!(j["content"]["application/json"]["schema"]["$ref"], "#/components/schemas/Pet");
    }

    #[test]
    fn response_empty() {
        let r = Response::new("No Content");
        let j = r.to_json();
        assert_eq!(j["description"], "No Content");
        assert!(j.get("content").is_none());
    }

    #[test]
    fn response_with_json() {
        let r = Response::with_json("OK", Schema::typed(SchemaType::Object));
        let j = r.to_json();
        assert_eq!(j["content"]["application/json"]["schema"]["type"], "object");
    }

    #[test]
    fn operation_builder() {
        let op = Operation::new()
            .with_id("listPets")
            .with_summary("List all pets")
            .with_tag("pets")
            .with_parameter(Parameter::new("limit", ParameterIn::Query, Schema::typed(SchemaType::Integer)))
            .with_response("200", Response::with_json("OK", Schema::array_of(Schema::reference("#/components/schemas/Pet"))));
        let j = op.to_json();
        assert_eq!(j["operationId"], "listPets");
        assert_eq!(j["tags"][0], "pets");
        assert_eq!(j["parameters"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn path_item_multi_methods() {
        let item = PathItem::new()
            .operation(HttpMethod::Get, Operation::new().with_id("list"))
            .operation(HttpMethod::Post, Operation::new().with_id("create"));
        let j = item.to_json();
        assert_eq!(j["get"]["operationId"], "list");
        assert_eq!(j["post"]["operationId"], "create");
    }

    #[test]
    fn security_scheme_bearer() {
        let s = SecurityScheme::bearer_jwt();
        let j = s.to_json();
        assert_eq!(j["type"], "http");
        assert_eq!(j["scheme"], "bearer");
        assert_eq!(j["bearerFormat"], "JWT");
    }

    #[test]
    fn security_scheme_api_key() {
        let s = SecurityScheme::api_key("X-API-Key", ParameterIn::Header);
        let j = s.to_json();
        assert_eq!(j["type"], "apiKey");
        assert_eq!(j["name"], "X-API-Key");
        assert_eq!(j["in"], "header");
    }

    #[test]
    fn full_spec_petstore() {
        let mut spec = OpenApiSpec::new(Info::new("Petstore", "1.0.0"));
        spec.add_server(Server::new("https://api.example.com"));
        spec.add_tag(Tag::new("pets"));

        let mut pet = Schema::typed(SchemaType::Object);
        pet.required = vec!["id".into(), "name".into()];
        pet.properties.insert("id".into(), Schema::typed(SchemaType::Integer));
        pet.properties.insert("name".into(), Schema::typed(SchemaType::String));
        spec.add_schema("Pet", pet);

        let list_op = Operation::new()
            .with_id("listPets")
            .with_tag("pets")
            .with_response("200", Response::with_json(
                "OK", Schema::array_of(Schema::reference("#/components/schemas/Pet")),
            ));
        let create_op = Operation::new()
            .with_id("createPet")
            .with_tag("pets")
            .with_body(RequestBody::json(Schema::reference("#/components/schemas/Pet")))
            .with_response("201", Response::new("Created"));
        spec.add_path("/pets", PathItem::new()
            .operation(HttpMethod::Get, list_op)
            .operation(HttpMethod::Post, create_op));

        let get_op = Operation::new()
            .with_id("getPet")
            .with_tag("pets")
            .with_parameter(Parameter::new("petId", ParameterIn::Path, Schema::typed(SchemaType::String)))
            .with_response("200", Response::with_json(
                "OK", Schema::reference("#/components/schemas/Pet"),
            ));
        spec.add_path("/pets/{petId}", PathItem::new()
            .operation(HttpMethod::Get, get_op));

        spec.add_security_scheme("bearerAuth", SecurityScheme::bearer_jwt());

        let json = spec.to_json();
        assert_eq!(json["openapi"], "3.1.0");
        assert_eq!(json["info"]["title"], "Petstore");
        assert!(json["paths"]["/pets"]["get"].is_object());
        assert!(json["components"]["schemas"]["Pet"].is_object());

        let errors = spec.validate();
        assert!(errors.is_empty(), "validation errors: {errors:?}");
    }

    #[test]
    fn validate_unresolved_ref() {
        let mut spec = OpenApiSpec::new(Info::new("Test", "1.0"));
        let op = Operation::new()
            .with_response("200", Response::with_json(
                "OK", Schema::reference("#/components/schemas/Missing"),
            ));
        spec.add_path("/test", PathItem::new().operation(HttpMethod::Get, op));
        let errors = spec.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("unresolved"));
    }

    #[test]
    fn validate_missing_path_param() {
        let mut spec = OpenApiSpec::new(Info::new("Test", "1.0"));
        let op = Operation::new().with_id("get_item");
        spec.add_path("/items/{itemId}", PathItem::new().operation(HttpMethod::Get, op));
        let errors = spec.validate();
        assert!(errors.iter().any(|e| e.contains("itemId")));
    }

    #[test]
    fn validate_duplicate_operation_id() {
        let mut spec = OpenApiSpec::new(Info::new("Test", "1.0"));
        spec.add_path("/a", PathItem::new()
            .operation(HttpMethod::Get, Operation::new().with_id("dup")));
        spec.add_path("/b", PathItem::new()
            .operation(HttpMethod::Get, Operation::new().with_id("dup")));
        let errors = spec.validate();
        assert!(errors.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn spec_to_json_string() {
        let spec = OpenApiSpec::new(Info::new("API", "0.1.0"));
        let json_str = spec.to_json_string();
        assert!(json_str.contains("3.1.0"));
        assert!(json_str.contains("API"));
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["openapi"], "3.1.0");
    }

    #[test]
    fn info_with_license() {
        let mut info = Info::new("My API", "2.0");
        info.license_name = Some("MIT".into());
        info.license_url = Some("https://opensource.org/licenses/MIT".into());
        let j = info.to_json();
        assert_eq!(j["license"]["name"], "MIT");
        assert_eq!(j["license"]["url"], "https://opensource.org/licenses/MIT");
    }

    #[test]
    fn schema_format_and_example() {
        let mut s = Schema::typed(SchemaType::String);
        s.format = Some("date-time".into());
        s.example = Some("2024-01-01T00:00:00Z".into());
        let j = s.to_json();
        assert_eq!(j["format"], "date-time");
        assert_eq!(j["example"], "2024-01-01T00:00:00Z");
    }

    #[test]
    fn schema_default_value() {
        let mut s = Schema::typed(SchemaType::String);
        s.default_value = Some("active".into());
        let j = s.to_json();
        assert_eq!(j["default"], "active");
    }

    #[test]
    fn server_with_description() {
        let mut s = Server::new("https://staging.example.com");
        s.description = Some("Staging".into());
        let j = s.to_json();
        assert_eq!(j["url"], "https://staging.example.com");
        assert_eq!(j["description"], "Staging");
    }

    #[test]
    fn http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "get");
        assert_eq!(HttpMethod::Post.to_string(), "post");
        assert_eq!(HttpMethod::Delete.to_string(), "delete");
        assert_eq!(HttpMethod::Patch.to_string(), "patch");
    }

    #[test]
    fn deprecated_parameter() {
        let mut p = Parameter::new("old", ParameterIn::Query, Schema::typed(SchemaType::String));
        p.deprecated = true;
        let j = p.to_json();
        assert_eq!(j["deprecated"], true);
    }

    #[test]
    fn operation_security() {
        let mut op = Operation::new().with_id("secure");
        let mut sec = BTreeMap::new();
        sec.insert("bearerAuth".into(), vec!["read".into(), "write".into()]);
        op.security.push(sec);
        let j = op.to_json();
        assert_eq!(j["security"][0]["bearerAuth"][0], "read");
    }

    #[test]
    fn response_headers() {
        let mut r = Response::new("OK");
        r.headers.insert("X-Rate-Limit".into(), Schema::typed(SchemaType::Integer));
        let j = r.to_json();
        assert_eq!(j["headers"]["X-Rate-Limit"]["schema"]["type"], "integer");
    }

    #[test]
    fn schema_all_of_composition() {
        let mut s = Schema::default();
        s.all_of = vec![
            Schema::reference("#/components/schemas/Base"),
            Schema::typed(SchemaType::Object),
        ];
        let j = s.to_json();
        assert_eq!(j["allOf"][0]["$ref"], "#/components/schemas/Base");
        assert_eq!(j["allOf"][1]["type"], "object");
    }

    #[test]
    fn schema_any_of() {
        let mut s = Schema::default();
        s.any_of = vec![
            Schema::typed(SchemaType::String),
            Schema::typed(SchemaType::Integer),
        ];
        let j = s.to_json();
        assert_eq!(j["anyOf"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn validate_path_param_declared_correctly() {
        let mut spec = OpenApiSpec::new(Info::new("T", "1.0"));
        let op = Operation::new()
            .with_id("getItem")
            .with_parameter(Parameter::new("itemId", ParameterIn::Path, Schema::typed(SchemaType::String)));
        spec.add_path("/items/{itemId}", PathItem::new().operation(HttpMethod::Get, op));
        let errors = spec.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn openid_connect_scheme() {
        let s = SecurityScheme {
            scheme_type: SecuritySchemeType::OpenIdConnect {
                url: "https://example.com/.well-known/openid".into(),
            },
            description: None,
        };
        let j = s.to_json();
        assert_eq!(j["type"], "openIdConnect");
        assert_eq!(j["openIdConnectUrl"], "https://example.com/.well-known/openid");
    }
}
