//! Service stub/proxy generation — definitions, code generation, registry.
//!
//! Provides [`ServiceDefinition`] for declaring RPC services with typed methods,
//! [`StubGenerator`] for producing client stub code (as Rust source strings),
//! skeleton generation for server implementations, service versioning, method
//! metadata (timeout, idempotency), a [`ServiceRegistry`] for discovery, and
//! automatic health check method generation.

use std::collections::HashMap;
use std::fmt;

// ── Method Kind ────────────────────────────────────────────────

/// The streaming pattern of an RPC method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodKind {
    Unary,
    ServerStream,
    ClientStream,
    BidiStream,
}

impl fmt::Display for MethodKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unary => write!(f, "unary"),
            Self::ServerStream => write!(f, "server_stream"),
            Self::ClientStream => write!(f, "client_stream"),
            Self::BidiStream => write!(f, "bidi_stream"),
        }
    }
}

// ── Method Metadata ────────────────────────────────────────────

/// Additional metadata for an RPC method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodMetadata {
    pub timeout_ms: Option<u64>,
    pub idempotent: bool,
    pub deprecated: bool,
    pub description: String,
}

impl MethodMetadata {
    pub fn new() -> Self {
        Self { timeout_ms: None, idempotent: false, deprecated: false, description: String::new() }
    }

    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms); self
    }

    pub fn with_idempotent(mut self, v: bool) -> Self {
        self.idempotent = v; self
    }

    pub fn with_deprecated(mut self, v: bool) -> Self {
        self.deprecated = v; self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into(); self
    }
}

impl Default for MethodMetadata {
    fn default() -> Self { Self::new() }
}

// ── Method Definition ──────────────────────────────────────────

/// Defines a single RPC method within a service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDef {
    pub name: String,
    pub request_type: String,
    pub response_type: String,
    pub kind: MethodKind,
    pub metadata: MethodMetadata,
}

impl MethodDef {
    pub fn new(name: impl Into<String>, req: impl Into<String>, resp: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            request_type: req.into(),
            response_type: resp.into(),
            kind: MethodKind::Unary,
            metadata: MethodMetadata::new(),
        }
    }

    pub fn with_kind(mut self, kind: MethodKind) -> Self {
        self.kind = kind; self
    }

    pub fn with_metadata(mut self, meta: MethodMetadata) -> Self {
        self.metadata = meta; self
    }

    /// Whether this is a streaming method.
    pub fn is_streaming(&self) -> bool {
        self.kind != MethodKind::Unary
    }
}

impl fmt::Display for MethodDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}) -> {} [{}]", self.name, self.request_type, self.response_type, self.kind)
    }
}

// ── Service Version ────────────────────────────────────────────

/// Semantic version for a service.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServiceVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ServiceVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Whether two versions are compatible (same major version).
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for ServiceVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ── Service Definition ─────────────────────────────────────────

/// Full definition of an RPC service.
#[derive(Debug, Clone)]
pub struct ServiceDefinition {
    pub name: String,
    pub version: ServiceVersion,
    pub methods: Vec<MethodDef>,
    pub description: String,
}

impl ServiceDefinition {
    pub fn new(name: impl Into<String>, version: ServiceVersion) -> Self {
        Self {
            name: name.into(),
            version,
            methods: Vec::new(),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into(); self
    }

    pub fn add_method(&mut self, method: MethodDef) {
        self.methods.push(method);
    }

    /// Add a standard health check method.
    pub fn add_health_check(&mut self) {
        self.methods.push(MethodDef {
            name: "health_check".to_string(),
            request_type: "HealthCheckRequest".to_string(),
            response_type: "HealthCheckResponse".to_string(),
            kind: MethodKind::Unary,
            metadata: MethodMetadata::new()
                .with_idempotent(true)
                .with_description("Standard health check endpoint"),
        });
    }

    pub fn method_count(&self) -> usize { self.methods.len() }

    pub fn find_method(&self, name: &str) -> Option<&MethodDef> {
        self.methods.iter().find(|m| m.name == name)
    }

    pub fn unary_methods(&self) -> Vec<&MethodDef> {
        self.methods.iter().filter(|m| m.kind == MethodKind::Unary).collect()
    }

    pub fn streaming_methods(&self) -> Vec<&MethodDef> {
        self.methods.iter().filter(|m| m.is_streaming()).collect()
    }
}

impl fmt::Display for ServiceDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} v{} ({} methods)", self.name, self.version, self.methods.len())
    }
}

// ── Stub Generator ─────────────────────────────────────────────

/// Generates client stub source code from a service definition.
#[derive(Debug)]
pub struct StubGenerator {
    indent: String,
    include_docs: bool,
}

impl StubGenerator {
    pub fn new() -> Self {
        Self { indent: "    ".to_string(), include_docs: true }
    }

    pub fn with_indent(mut self, indent: impl Into<String>) -> Self {
        self.indent = indent.into(); self
    }

    pub fn with_docs(mut self, include: bool) -> Self {
        self.include_docs = include; self
    }

    /// Generate client stub code as a string.
    pub fn generate_client_stub(&self, service: &ServiceDefinition) -> String {
        let mut out = String::new();
        let ind = &self.indent;
        if self.include_docs {
            out.push_str(&format!("/// Client stub for {} v{}.\n", service.name, service.version));
        }
        out.push_str(&format!("pub struct {}Client {{\n", service.name));
        out.push_str(&format!("{ind}// connection handle\n"));
        out.push_str("}\n\n");
        out.push_str(&format!("impl {}Client {{\n", service.name));
        for method in &service.methods {
            if self.include_docs {
                if !method.metadata.description.is_empty() {
                    out.push_str(&format!("{ind}/// {}\n", method.metadata.description));
                }
                if method.metadata.deprecated {
                    out.push_str(&format!("{ind}#[deprecated]\n"));
                }
            }
            match method.kind {
                MethodKind::Unary => {
                    out.push_str(&format!(
                        "{ind}pub fn {}(&self, req: {}) -> Result<{}, Error> {{\n",
                        method.name, method.request_type, method.response_type
                    ));
                    out.push_str(&format!("{ind}{ind}todo!()\n"));
                    out.push_str(&format!("{ind}}}\n\n"));
                }
                _ => {
                    out.push_str(&format!(
                        "{ind}pub fn {}(&self, req: {}) -> Result<Stream<{}>, Error> {{\n",
                        method.name, method.request_type, method.response_type
                    ));
                    out.push_str(&format!("{ind}{ind}todo!()\n"));
                    out.push_str(&format!("{ind}}}\n\n"));
                }
            }
        }
        out.push_str("}\n");
        out
    }

    /// Generate server skeleton code as a string.
    pub fn generate_server_skeleton(&self, service: &ServiceDefinition) -> String {
        let mut out = String::new();
        let ind = &self.indent;
        if self.include_docs {
            out.push_str(&format!("/// Server trait for {} v{}.\n", service.name, service.version));
        }
        out.push_str(&format!("pub trait {}Server {{\n", service.name));
        for method in &service.methods {
            if self.include_docs && !method.metadata.description.is_empty() {
                out.push_str(&format!("{ind}/// {}\n", method.metadata.description));
            }
            out.push_str(&format!(
                "{ind}fn {}(&self, req: {}) -> Result<{}, Error>;\n",
                method.name, method.request_type, method.response_type
            ));
        }
        out.push_str("}\n");
        out
    }
}

impl Default for StubGenerator {
    fn default() -> Self { Self::new() }
}

// ── Service Registry ───────────────────────────────────────────

/// Registry for discovering available services.
#[derive(Debug, Clone)]
pub struct ServiceRegistry {
    services: HashMap<String, Vec<ServiceDefinition>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self { services: HashMap::new() }
    }

    /// Register a service definition.
    pub fn register(&mut self, service: ServiceDefinition) {
        self.services.entry(service.name.clone())
            .or_default()
            .push(service);
    }

    /// Look up a service by name (latest version).
    pub fn lookup(&self, name: &str) -> Option<&ServiceDefinition> {
        self.services.get(name)
            .and_then(|versions| versions.last())
    }

    /// Look up a specific version of a service.
    pub fn lookup_version(&self, name: &str, version: &ServiceVersion) -> Option<&ServiceDefinition> {
        self.services.get(name)
            .and_then(|versions| versions.iter().find(|s| &s.version == version))
    }

    /// List all registered service names.
    pub fn service_names(&self) -> Vec<&str> {
        self.services.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered services (counting all versions).
    pub fn total_count(&self) -> usize {
        self.services.values().map(|v| v.len()).sum()
    }

    /// Check if a service is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.services.contains_key(name)
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self { Self::new() }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_service() -> ServiceDefinition {
        let mut svc = ServiceDefinition::new("Greeter", ServiceVersion::new(1, 0, 0));
        svc.add_method(MethodDef::new("say_hello", "HelloRequest", "HelloResponse"));
        svc.add_method(
            MethodDef::new("stream_greetings", "HelloRequest", "HelloResponse")
                .with_kind(MethodKind::ServerStream)
        );
        svc
    }

    #[test]
    fn service_definition_basics() {
        let svc = sample_service();
        assert_eq!(svc.name, "Greeter");
        assert_eq!(svc.method_count(), 2);
        assert_eq!(svc.version, ServiceVersion::new(1, 0, 0));
    }

    #[test]
    fn find_method_by_name() {
        let svc = sample_service();
        let m = svc.find_method("say_hello").unwrap();
        assert_eq!(m.request_type, "HelloRequest");
        assert!(svc.find_method("nonexistent").is_none());
    }

    #[test]
    fn unary_and_streaming_split() {
        let svc = sample_service();
        assert_eq!(svc.unary_methods().len(), 1);
        assert_eq!(svc.streaming_methods().len(), 1);
    }

    #[test]
    fn health_check_auto_generated() {
        let mut svc = sample_service();
        svc.add_health_check();
        assert_eq!(svc.method_count(), 3);
        let hc = svc.find_method("health_check").unwrap();
        assert!(hc.metadata.idempotent);
        assert_eq!(hc.kind, MethodKind::Unary);
    }

    #[test]
    fn method_is_streaming() {
        let unary = MethodDef::new("m", "Req", "Resp");
        assert!(!unary.is_streaming());
        let stream = MethodDef::new("m", "Req", "Resp").with_kind(MethodKind::BidiStream);
        assert!(stream.is_streaming());
    }

    #[test]
    fn method_metadata_builder() {
        let meta = MethodMetadata::new()
            .with_timeout_ms(5000)
            .with_idempotent(true)
            .with_deprecated(true)
            .with_description("test method");
        assert_eq!(meta.timeout_ms, Some(5000));
        assert!(meta.idempotent);
        assert!(meta.deprecated);
        assert_eq!(meta.description, "test method");
    }

    #[test]
    fn version_compatibility() {
        let v1 = ServiceVersion::new(1, 0, 0);
        let v1_1 = ServiceVersion::new(1, 1, 0);
        let v2 = ServiceVersion::new(2, 0, 0);
        assert!(v1.is_compatible(&v1_1));
        assert!(!v1.is_compatible(&v2));
    }

    #[test]
    fn version_display() {
        assert_eq!(format!("{}", ServiceVersion::new(1, 2, 3)), "1.2.3");
    }

    #[test]
    fn generate_client_stub_contains_struct() {
        let svc = sample_service();
        let generator = StubGenerator::new();
        let code = generator.generate_client_stub(&svc);
        assert!(code.contains("pub struct GreeterClient"));
        assert!(code.contains("fn say_hello"));
        assert!(code.contains("fn stream_greetings"));
        assert!(code.contains("Result<"));
    }

    #[test]
    fn generate_server_skeleton_contains_trait() {
        let svc = sample_service();
        let generator = StubGenerator::new();
        let code = generator.generate_server_skeleton(&svc);
        assert!(code.contains("pub trait GreeterServer"));
        assert!(code.contains("fn say_hello"));
    }

    #[test]
    fn stub_without_docs() {
        let svc = sample_service();
        let generator = StubGenerator::new().with_docs(false);
        let code = generator.generate_client_stub(&svc);
        assert!(!code.contains("///"));
    }

    #[test]
    fn service_registry_register_and_lookup() {
        let mut reg = ServiceRegistry::new();
        reg.register(sample_service());
        assert!(reg.contains("Greeter"));
        let svc = reg.lookup("Greeter").unwrap();
        assert_eq!(svc.name, "Greeter");
    }

    #[test]
    fn registry_lookup_version() {
        let mut reg = ServiceRegistry::new();
        let mut v1 = sample_service();
        v1.version = ServiceVersion::new(1, 0, 0);
        let mut v2 = sample_service();
        v2.version = ServiceVersion::new(2, 0, 0);
        reg.register(v1);
        reg.register(v2);
        let found = reg.lookup_version("Greeter", &ServiceVersion::new(1, 0, 0)).unwrap();
        assert_eq!(found.version.major, 1);
    }

    #[test]
    fn registry_service_names() {
        let mut reg = ServiceRegistry::new();
        reg.register(sample_service());
        let names = reg.service_names();
        assert!(names.contains(&"Greeter"));
    }

    #[test]
    fn registry_total_count() {
        let mut reg = ServiceRegistry::new();
        reg.register(sample_service());
        let mut v2 = sample_service();
        v2.version = ServiceVersion::new(2, 0, 0);
        reg.register(v2);
        assert_eq!(reg.total_count(), 2);
    }

    #[test]
    fn registry_lookup_missing() {
        let reg = ServiceRegistry::new();
        assert!(reg.lookup("Nonexistent").is_none());
    }

    #[test]
    fn method_display() {
        let m = MethodDef::new("get_user", "GetUserReq", "GetUserResp");
        let s = format!("{m}");
        assert!(s.contains("get_user"));
        assert!(s.contains("unary"));
    }

    #[test]
    fn service_display() {
        let svc = sample_service();
        let s = format!("{svc}");
        assert!(s.contains("Greeter"));
        assert!(s.contains("1.0.0"));
        assert!(s.contains("2 methods"));
    }

    #[test]
    fn method_with_deprecated_in_stub() {
        let mut svc = ServiceDefinition::new("Svc", ServiceVersion::new(1, 0, 0));
        svc.add_method(
            MethodDef::new("old_method", "Req", "Resp")
                .with_metadata(MethodMetadata::new().with_deprecated(true))
        );
        let generator = StubGenerator::new();
        let code = generator.generate_client_stub(&svc);
        assert!(code.contains("#[deprecated]"));
    }
}
