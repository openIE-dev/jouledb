//! GraphQL execution engine — schema definition, field resolvers, query
//! execution against a schema, argument passing, nested object resolution,
//! list resolution, error collection, and data + errors response format.
//!
//! Replaces server-side JS GraphQL runtimes (`graphql-js`, `apollo-server`)
//! with a pure-Rust execution engine using closure-based resolvers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Execution error.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionError {
    pub message: String,
    pub path: Vec<PathSegment>,
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if !self.path.is_empty() {
            write!(f, " at ")?;
            for (i, seg) in self.path.iter().enumerate() {
                if i > 0 {
                    write!(f, ".")?;
                }
                match seg {
                    PathSegment::Field(name) => write!(f, "{name}")?,
                    PathSegment::Index(idx) => write!(f, "[{idx}]")?,
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for ExecutionError {}

/// Path segment within a response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PathSegment {
    Field(String),
    Index(usize),
}

// ── Schema Types ─────────────────────────────────────────────────

/// A GraphQL type within the schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GqlType {
    Scalar(ScalarType),
    Object(String),
    List(Box<GqlType>),
    NonNull(Box<GqlType>),
    Enum(String),
}

/// Built-in scalar types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    String,
    Int,
    Float,
    Boolean,
    Id,
}

impl ScalarType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::String => "String",
            Self::Int => "Int",
            Self::Float => "Float",
            Self::Boolean => "Boolean",
            Self::Id => "ID",
        }
    }
}

/// Schema-level field definition with type and optional argument defs.
#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub field_type: GqlType,
    pub arguments: Vec<ArgumentDef>,
}

/// Schema-level argument definition.
#[derive(Debug, Clone)]
pub struct ArgumentDef {
    pub name: String,
    pub arg_type: GqlType,
    pub default_value: Option<serde_json::Value>,
}

/// Object type definition in the schema.
#[derive(Debug, Clone)]
pub struct ObjectTypeDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
}

/// Enum type definition.
#[derive(Debug, Clone)]
pub struct EnumTypeDef {
    pub name: String,
    pub values: Vec<String>,
}

// ── Resolver ─────────────────────────────────────────────────────

/// Context passed to resolvers.
#[derive(Debug, Clone)]
pub struct ResolverContext {
    /// Parent object value (may be Null for root).
    pub parent: serde_json::Value,
    /// Arguments passed to this field.
    pub arguments: HashMap<String, serde_json::Value>,
    /// Current field name.
    pub field_name: String,
    /// Path to this field in the response.
    pub path: Vec<PathSegment>,
}

/// Result of a field resolver.
#[derive(Debug, Clone)]
pub enum ResolverResult {
    /// Successfully resolved a value.
    Ok(serde_json::Value),
    /// Resolver produced an error.
    Err(String),
}

// ── Schema ───────────────────────────────────────────────────────

/// A resolver function type — receives context, returns a result.
type ResolverFn = Box<dyn Fn(&ResolverContext) -> ResolverResult + Send + Sync>;

/// Registered resolver for a specific type + field.
struct ResolverEntry {
    resolver: ResolverFn,
}

/// The GraphQL schema — type definitions + resolvers.
pub struct Schema {
    pub object_types: HashMap<String, ObjectTypeDef>,
    pub enum_types: HashMap<String, EnumTypeDef>,
    pub query_type: String,
    pub mutation_type: Option<String>,
    resolvers: HashMap<String, HashMap<String, ResolverEntry>>,
}

impl Schema {
    /// Create a new schema with the given query root type name.
    pub fn new(query_type: &str) -> Self {
        Self {
            object_types: HashMap::new(),
            enum_types: HashMap::new(),
            query_type: query_type.to_string(),
            mutation_type: None,
            resolvers: HashMap::new(),
        }
    }

    /// Set the mutation root type name.
    pub fn set_mutation_type(&mut self, name: &str) {
        self.mutation_type = Some(name.to_string());
    }

    /// Register an object type.
    pub fn add_object_type(&mut self, type_def: ObjectTypeDef) {
        self.object_types.insert(type_def.name.clone(), type_def);
    }

    /// Register an enum type.
    pub fn add_enum_type(&mut self, type_def: EnumTypeDef) {
        self.enum_types.insert(type_def.name.clone(), type_def);
    }

    /// Register a field resolver for a given type and field.
    pub fn add_resolver<F>(&mut self, type_name: &str, field_name: &str, resolver: F)
    where
        F: Fn(&ResolverContext) -> ResolverResult + Send + Sync + 'static,
    {
        let type_map = self
            .resolvers
            .entry(type_name.to_string())
            .or_insert_with(HashMap::new);
        type_map.insert(
            field_name.to_string(),
            ResolverEntry {
                resolver: Box::new(resolver),
            },
        );
    }

    /// Look up a resolver for a type + field.
    fn get_resolver(&self, type_name: &str, field_name: &str) -> Option<&ResolverEntry> {
        self.resolvers
            .get(type_name)
            .and_then(|m| m.get(field_name))
    }

    /// Get the object type definition by name.
    fn get_object_type(&self, name: &str) -> Option<&ObjectTypeDef> {
        self.object_types.get(name)
    }

    /// Get the field definition within an object type.
    fn get_field_def(&self, type_name: &str, field_name: &str) -> Option<&FieldDef> {
        self.object_types
            .get(type_name)
            .and_then(|ot| ot.fields.iter().find(|f| f.name == field_name))
    }
}

// ── Query Representation ─────────────────────────────────────────

/// A parsed field from a query (simplified, for execution).
#[derive(Debug, Clone)]
pub struct QueryField {
    pub alias: Option<String>,
    pub name: String,
    pub arguments: Vec<(String, serde_json::Value)>,
    pub sub_fields: Vec<QueryField>,
}

impl QueryField {
    /// The response key — alias if present, else the field name.
    pub fn response_key(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.name)
    }
}

/// An executable query.
#[derive(Debug, Clone)]
pub struct ExecutableQuery {
    pub operation: OperationKind,
    pub fields: Vec<QueryField>,
}

/// The kind of operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    Query,
    Mutation,
}

// ── Response ─────────────────────────────────────────────────────

/// The GraphQL response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub errors: Option<Vec<ResponseError>>,
}

/// An error in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<serde_json::Value>>,
}

// ── Execution ────────────────────────────────────────────────────

/// Execute a query against a schema.
pub fn execute(schema: &Schema, query: &ExecutableQuery) -> GraphqlResponse {
    let root_type = match query.operation {
        OperationKind::Query => &schema.query_type,
        OperationKind::Mutation => {
            if let Some(ref mt) = schema.mutation_type {
                mt
            } else {
                return GraphqlResponse {
                    data: None,
                    errors: Some(vec![ResponseError {
                        message: "no mutation type defined".to_string(),
                        path: None,
                    }]),
                };
            }
        }
    };

    let mut errors = Vec::new();
    let parent = serde_json::Value::Null;
    let path: Vec<PathSegment> = Vec::new();

    let data = resolve_selection_set(
        schema,
        root_type,
        &parent,
        &query.fields,
        &path,
        &mut errors,
    );

    let response_errors = if errors.is_empty() {
        None
    } else {
        Some(
            errors
                .into_iter()
                .map(|e| {
                    let path_vals: Vec<serde_json::Value> = e
                        .path
                        .iter()
                        .map(|seg| match seg {
                            PathSegment::Field(name) => {
                                serde_json::Value::String(name.clone())
                            }
                            PathSegment::Index(i) => {
                                serde_json::Value::Number((*i).into())
                            }
                        })
                        .collect();
                    ResponseError {
                        message: e.message,
                        path: if path_vals.is_empty() {
                            None
                        } else {
                            Some(path_vals)
                        },
                    }
                })
                .collect(),
        )
    };

    GraphqlResponse {
        data: Some(data),
        errors: response_errors,
    }
}

fn resolve_selection_set(
    schema: &Schema,
    type_name: &str,
    parent: &serde_json::Value,
    fields: &[QueryField],
    path: &[PathSegment],
    errors: &mut Vec<ExecutionError>,
) -> serde_json::Value {
    let mut map = serde_json::Map::new();

    for field in fields {
        let response_key = field.response_key().to_string();
        let mut field_path = path.to_vec();
        field_path.push(PathSegment::Field(response_key.clone()));

        // Check for __typename meta-field
        if field.name == "__typename" {
            map.insert(
                response_key,
                serde_json::Value::String(type_name.to_string()),
            );
            continue;
        }

        // Look up resolver
        let resolved = if let Some(entry) = schema.get_resolver(type_name, &field.name) {
            let mut arguments = HashMap::new();
            for (k, v) in &field.arguments {
                arguments.insert(k.clone(), v.clone());
            }
            let ctx = ResolverContext {
                parent: parent.clone(),
                arguments,
                field_name: field.name.clone(),
                path: field_path.clone(),
            };
            match (entry.resolver)(&ctx) {
                ResolverResult::Ok(val) => val,
                ResolverResult::Err(msg) => {
                    errors.push(ExecutionError {
                        message: msg,
                        path: field_path.clone(),
                    });
                    serde_json::Value::Null
                }
            }
        } else {
            // Default resolver: try to read from parent object
            match parent.get(&field.name) {
                Some(val) => val.clone(),
                None => {
                    errors.push(ExecutionError {
                        message: format!(
                            "no resolver for {type_name}.{}",
                            field.name
                        ),
                        path: field_path.clone(),
                    });
                    serde_json::Value::Null
                }
            }
        };

        // If there are sub-fields, we need to resolve them
        let final_value = if !field.sub_fields.is_empty() {
            resolve_nested(schema, type_name, &field.name, &resolved, &field.sub_fields, &field_path, errors)
        } else {
            resolved
        };

        map.insert(response_key, final_value);
    }

    serde_json::Value::Object(map)
}

fn resolve_nested(
    schema: &Schema,
    parent_type: &str,
    field_name: &str,
    value: &serde_json::Value,
    sub_fields: &[QueryField],
    path: &[PathSegment],
    errors: &mut Vec<ExecutionError>,
) -> serde_json::Value {
    match value {
        serde_json::Value::Null => serde_json::Value::Null,
        serde_json::Value::Array(items) => {
            let mut result = Vec::new();
            for (i, item) in items.iter().enumerate() {
                let mut item_path = path.to_vec();
                item_path.push(PathSegment::Index(i));
                let nested_type = infer_nested_type(schema, parent_type, field_name);
                result.push(resolve_selection_set(
                    schema,
                    &nested_type,
                    item,
                    sub_fields,
                    &item_path,
                    errors,
                ));
            }
            serde_json::Value::Array(result)
        }
        obj @ serde_json::Value::Object(_) => {
            let nested_type = infer_nested_type(schema, parent_type, field_name);
            resolve_selection_set(schema, &nested_type, obj, sub_fields, path, errors)
        }
        _ => value.clone(),
    }
}

/// Infer the nested object type name from the field definition.
fn infer_nested_type(schema: &Schema, parent_type: &str, field_name: &str) -> String {
    if let Some(field_def) = schema.get_field_def(parent_type, field_name) {
        extract_object_name(&field_def.field_type)
    } else {
        // Fallback: capitalize field name
        let mut result = field_name.to_string();
        if let Some(first) = result.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        result
    }
}

fn extract_object_name(gql_type: &GqlType) -> String {
    match gql_type {
        GqlType::Object(name) => name.clone(),
        GqlType::NonNull(inner) => extract_object_name(inner),
        GqlType::List(inner) => extract_object_name(inner),
        GqlType::Enum(name) => name.clone(),
        GqlType::Scalar(s) => s.name().to_string(),
    }
}

// ── Builder Helpers ──────────────────────────────────────────────

/// Build a field definition for an object type.
pub fn field(name: &str, field_type: GqlType) -> FieldDef {
    FieldDef {
        name: name.to_string(),
        field_type,
        arguments: Vec::new(),
    }
}

/// Build a field definition with arguments.
pub fn field_with_args(name: &str, field_type: GqlType, args: Vec<ArgumentDef>) -> FieldDef {
    FieldDef {
        name: name.to_string(),
        field_type,
        arguments: args,
    }
}

/// Build an argument definition.
pub fn arg(name: &str, arg_type: GqlType) -> ArgumentDef {
    ArgumentDef {
        name: name.to_string(),
        arg_type,
        default_value: None,
    }
}

/// Build an argument with a default value.
pub fn arg_with_default(name: &str, arg_type: GqlType, default: serde_json::Value) -> ArgumentDef {
    ArgumentDef {
        name: name.to_string(),
        arg_type,
        default_value: Some(default),
    }
}

/// Build an executable query from a list of fields.
pub fn query(fields: Vec<QueryField>) -> ExecutableQuery {
    ExecutableQuery {
        operation: OperationKind::Query,
        fields,
    }
}

/// Build an executable mutation from a list of fields.
pub fn mutation(fields: Vec<QueryField>) -> ExecutableQuery {
    ExecutableQuery {
        operation: OperationKind::Mutation,
        fields,
    }
}

/// Build a query field.
pub fn qf(name: &str) -> QueryField {
    QueryField {
        alias: None,
        name: name.to_string(),
        arguments: Vec::new(),
        sub_fields: Vec::new(),
    }
}

/// Build a query field with sub-fields.
pub fn qf_with_sub(name: &str, sub_fields: Vec<QueryField>) -> QueryField {
    QueryField {
        alias: None,
        name: name.to_string(),
        arguments: Vec::new(),
        sub_fields,
    }
}

/// Build a query field with arguments and sub-fields.
pub fn qf_full(
    name: &str,
    alias: Option<&str>,
    args: Vec<(&str, serde_json::Value)>,
    sub_fields: Vec<QueryField>,
) -> QueryField {
    QueryField {
        alias: alias.map(|s| s.to_string()),
        name: name.to_string(),
        arguments: args
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        sub_fields,
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user_schema() -> Schema {
        let mut schema = Schema::new("Query");

        schema.add_object_type(ObjectTypeDef {
            name: "Query".to_string(),
            fields: vec![
                field_with_args(
                    "user",
                    GqlType::Object("User".to_string()),
                    vec![arg("id", GqlType::NonNull(Box::new(GqlType::Scalar(ScalarType::Id))))],
                ),
                field("users", GqlType::List(Box::new(GqlType::Object("User".to_string())))),
                field("hello", GqlType::Scalar(ScalarType::String)),
            ],
        });

        schema.add_object_type(ObjectTypeDef {
            name: "User".to_string(),
            fields: vec![
                field("id", GqlType::NonNull(Box::new(GqlType::Scalar(ScalarType::Id)))),
                field("name", GqlType::Scalar(ScalarType::String)),
                field("email", GqlType::Scalar(ScalarType::String)),
                field(
                    "posts",
                    GqlType::List(Box::new(GqlType::Object("Post".to_string()))),
                ),
            ],
        });

        schema.add_object_type(ObjectTypeDef {
            name: "Post".to_string(),
            fields: vec![
                field("id", GqlType::NonNull(Box::new(GqlType::Scalar(ScalarType::Id)))),
                field("title", GqlType::Scalar(ScalarType::String)),
            ],
        });

        // Resolvers
        schema.add_resolver("Query", "hello", |_ctx| {
            ResolverResult::Ok(serde_json::Value::String("world".to_string()))
        });

        schema.add_resolver("Query", "user", |ctx| {
            let id = ctx
                .arguments
                .get("id")
                .cloned()
                .unwrap_or(serde_json::Value::String("1".into()));
            let id_str = match &id {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => "unknown".to_string(),
            };
            let user = serde_json::json!({
                "id": id_str,
                "name": format!("User {}", id_str),
                "email": format!("user{}@example.com", id_str),
            });
            ResolverResult::Ok(user)
        });

        schema.add_resolver("Query", "users", |_ctx| {
            let users = serde_json::json!([
                {"id": "1", "name": "Alice", "email": "alice@example.com"},
                {"id": "2", "name": "Bob", "email": "bob@example.com"},
            ]);
            ResolverResult::Ok(users)
        });

        schema.add_resolver("User", "posts", |ctx| {
            let user_id = ctx.parent.get("id").and_then(|v| v.as_str()).unwrap_or("0");
            let posts = serde_json::json!([
                {"id": format!("{user_id}-p1"), "title": format!("Post by {user_id}")},
            ]);
            ResolverResult::Ok(posts)
        });

        schema
    }

    #[test]
    fn execute_simple_scalar() {
        let schema = make_user_schema();
        let q = query(vec![qf("hello")]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["hello"], serde_json::Value::String("world".into()));
        assert!(response.errors.is_none());
    }

    #[test]
    fn execute_with_arguments() {
        let schema = make_user_schema();
        let q = query(vec![qf_full(
            "user",
            None,
            vec![("id", serde_json::json!("42"))],
            vec![qf("id"), qf("name")],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["user"]["id"], "42");
        assert_eq!(data["user"]["name"], "User 42");
    }

    #[test]
    fn execute_nested_objects() {
        let schema = make_user_schema();
        let q = query(vec![qf_full(
            "user",
            None,
            vec![("id", serde_json::json!("1"))],
            vec![
                qf("name"),
                qf_with_sub("posts", vec![qf("id"), qf("title")]),
            ],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        let posts = data["user"]["posts"].as_array().unwrap();
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0]["title"], "Post by 1");
    }

    #[test]
    fn execute_list_resolution() {
        let schema = make_user_schema();
        let q = query(vec![qf_with_sub(
            "users",
            vec![qf("id"), qf("name")],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        let users = data["users"].as_array().unwrap();
        assert_eq!(users.len(), 2);
        assert_eq!(users[0]["name"], "Alice");
        assert_eq!(users[1]["name"], "Bob");
    }

    #[test]
    fn execute_typename() {
        let schema = make_user_schema();
        let q = query(vec![qf_full(
            "user",
            None,
            vec![("id", serde_json::json!("1"))],
            vec![qf("__typename"), qf("name")],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["user"]["__typename"], "User");
    }

    #[test]
    fn execute_field_alias() {
        let schema = make_user_schema();
        let q = query(vec![qf_full(
            "hello",
            Some("greeting"),
            vec![],
            vec![],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["greeting"], "world");
    }

    #[test]
    fn execute_error_collection() {
        let mut schema = Schema::new("Query");
        schema.add_object_type(ObjectTypeDef {
            name: "Query".to_string(),
            fields: vec![field("failing", GqlType::Scalar(ScalarType::String))],
        });
        schema.add_resolver("Query", "failing", |_ctx| {
            ResolverResult::Err("something went wrong".to_string())
        });

        let q = query(vec![qf("failing")]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert!(data["failing"].is_null());
        let errors = response.errors.unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "something went wrong");
    }

    #[test]
    fn execute_no_resolver_uses_default() {
        let mut schema = Schema::new("Query");
        schema.add_object_type(ObjectTypeDef {
            name: "Query".to_string(),
            fields: vec![field(
                "item",
                GqlType::Object("Item".to_string()),
            )],
        });
        schema.add_object_type(ObjectTypeDef {
            name: "Item".to_string(),
            fields: vec![field("value", GqlType::Scalar(ScalarType::String))],
        });
        schema.add_resolver("Query", "item", |_ctx| {
            ResolverResult::Ok(serde_json::json!({"value": "hello"}))
        });

        let q = query(vec![qf_with_sub("item", vec![qf("value")])]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["item"]["value"], "hello");
    }

    #[test]
    fn execute_mutation_no_type_defined() {
        let schema = Schema::new("Query");
        let q = mutation(vec![qf("createUser")]);
        let response = execute(&schema, &q);
        assert!(response.errors.is_some());
        let err = &response.errors.unwrap()[0];
        assert!(err.message.contains("no mutation type"));
    }

    #[test]
    fn execute_mutation_with_resolver() {
        let mut schema = Schema::new("Query");
        schema.set_mutation_type("Mutation");
        schema.add_object_type(ObjectTypeDef {
            name: "Mutation".to_string(),
            fields: vec![field_with_args(
                "createUser",
                GqlType::Object("User".to_string()),
                vec![arg("name", GqlType::Scalar(ScalarType::String))],
            )],
        });
        schema.add_object_type(ObjectTypeDef {
            name: "User".to_string(),
            fields: vec![
                field("id", GqlType::Scalar(ScalarType::Id)),
                field("name", GqlType::Scalar(ScalarType::String)),
            ],
        });
        schema.add_resolver("Mutation", "createUser", |ctx| {
            let name = ctx
                .arguments
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            ResolverResult::Ok(serde_json::json!({"id": "new-1", "name": name}))
        });

        let q = mutation(vec![qf_full(
            "createUser",
            None,
            vec![("name", serde_json::json!("Charlie"))],
            vec![qf("id"), qf("name")],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["createUser"]["name"], "Charlie");
    }

    #[test]
    fn execute_multiple_root_fields() {
        let schema = make_user_schema();
        let q = query(vec![
            qf("hello"),
            qf_full(
                "user",
                None,
                vec![("id", serde_json::json!("5"))],
                vec![qf("name")],
            ),
        ]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        assert_eq!(data["hello"], "world");
        assert_eq!(data["user"]["name"], "User 5");
    }

    #[test]
    fn execute_deeply_nested() {
        let schema = make_user_schema();
        let q = query(vec![qf_with_sub(
            "users",
            vec![
                qf("name"),
                qf_with_sub("posts", vec![qf("title")]),
            ],
        )]);
        let response = execute(&schema, &q);
        let data = response.data.unwrap();
        let users = data["users"].as_array().unwrap();
        let posts = users[0]["posts"].as_array().unwrap();
        assert_eq!(posts[0]["title"], "Post by 1");
    }

    #[test]
    fn response_key_uses_alias() {
        let f = qf_full("user", Some("u"), vec![], vec![]);
        assert_eq!(f.response_key(), "u");
    }

    #[test]
    fn response_key_uses_name() {
        let f = qf("user");
        assert_eq!(f.response_key(), "user");
    }

    #[test]
    fn schema_builder_helpers() {
        let f = field("name", GqlType::Scalar(ScalarType::String));
        assert_eq!(f.name, "name");
        assert_eq!(f.arguments.len(), 0);

        let a = arg("id", GqlType::Scalar(ScalarType::Id));
        assert_eq!(a.name, "id");
        assert!(a.default_value.is_none());

        let ad = arg_with_default(
            "limit",
            GqlType::Scalar(ScalarType::Int),
            serde_json::json!(10),
        );
        assert_eq!(ad.default_value, Some(serde_json::json!(10)));
    }

    #[test]
    fn execution_error_display() {
        let err = ExecutionError {
            message: "bad".to_string(),
            path: vec![
                PathSegment::Field("user".to_string()),
                PathSegment::Index(0),
                PathSegment::Field("name".to_string()),
            ],
        };
        let s = err.to_string();
        assert!(s.contains("bad"));
        assert!(s.contains("user"));
        assert!(s.contains("[0]"));
        assert!(s.contains("name"));
    }

    #[test]
    fn scalar_type_names() {
        assert_eq!(ScalarType::String.name(), "String");
        assert_eq!(ScalarType::Int.name(), "Int");
        assert_eq!(ScalarType::Float.name(), "Float");
        assert_eq!(ScalarType::Boolean.name(), "Boolean");
        assert_eq!(ScalarType::Id.name(), "ID");
    }

    #[test]
    fn enum_type_in_schema() {
        let mut schema = Schema::new("Query");
        schema.add_enum_type(EnumTypeDef {
            name: "Role".to_string(),
            values: vec!["ADMIN".to_string(), "USER".to_string()],
        });
        assert!(schema.enum_types.contains_key("Role"));
        assert_eq!(schema.enum_types["Role"].values.len(), 2);
    }

    #[test]
    fn graphql_response_serialization() {
        let response = GraphqlResponse {
            data: Some(serde_json::json!({"hello": "world"})),
            errors: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("hello"));
        assert!(!json.contains("errors"));
    }
}
