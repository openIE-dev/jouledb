//! Role-Based Access Control (RBAC) System for JouleDB
//!
//! Provides comprehensive access control features:
//! - Users, Roles, and Permissions model
//! - Permission types: READ, WRITE, DELETE, CREATE, ALTER, DROP, ADMIN
//! - Resource-level permissions (database, table, column)
//! - Role hierarchy with inheritance
//! - Row-level security policies
//! - Permission checking middleware
//! - SQL GRANT/REVOKE support

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Permission Types
// ============================================================================

/// Permission type for access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PermissionType {
    /// Read data from resources
    Read,
    /// Write/update existing data
    Write,
    /// Delete data
    Delete,
    /// Create new records
    Create,
    /// Alter schema (modify tables, columns)
    Alter,
    /// Drop resources (tables, databases)
    Drop,
    /// Full administrative access
    Admin,
}

impl PermissionType {
    /// Get all permission types
    pub fn all() -> Vec<PermissionType> {
        vec![
            PermissionType::Read,
            PermissionType::Write,
            PermissionType::Delete,
            PermissionType::Create,
            PermissionType::Alter,
            PermissionType::Drop,
            PermissionType::Admin,
        ]
    }

    /// Parse from string (SQL-style)
    pub fn from_str(s: &str) -> Option<PermissionType> {
        match s.to_uppercase().as_str() {
            "READ" | "SELECT" => Some(PermissionType::Read),
            "WRITE" | "UPDATE" => Some(PermissionType::Write),
            "DELETE" => Some(PermissionType::Delete),
            "CREATE" | "INSERT" => Some(PermissionType::Create),
            "ALTER" => Some(PermissionType::Alter),
            "DROP" => Some(PermissionType::Drop),
            "ADMIN" | "ALL" | "ALL PRIVILEGES" => Some(PermissionType::Admin),
            _ => None,
        }
    }

    /// Convert to SQL-style string
    pub fn to_sql_string(&self) -> &'static str {
        match self {
            PermissionType::Read => "SELECT",
            PermissionType::Write => "UPDATE",
            PermissionType::Delete => "DELETE",
            PermissionType::Create => "INSERT",
            PermissionType::Alter => "ALTER",
            PermissionType::Drop => "DROP",
            PermissionType::Admin => "ALL PRIVILEGES",
        }
    }

    /// Check if this permission implies another
    pub fn implies(&self, other: &PermissionType) -> bool {
        if *self == PermissionType::Admin {
            return true; // Admin implies all permissions
        }
        self == other
    }
}

impl std::fmt::Display for PermissionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sql_string())
    }
}

// ============================================================================
// Resource Types
// ============================================================================

/// Resource type for permission scoping
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Global server-level resource
    Server,
    /// Database-level resource
    Database(String),
    /// Table-level resource
    Table { database: String, table: String },
    /// Column-level resource
    Column {
        database: String,
        table: String,
        column: String,
    },
}

impl ResourceType {
    /// Create a server resource
    pub fn server() -> Self {
        ResourceType::Server
    }

    /// Create a database resource
    pub fn database(name: impl Into<String>) -> Self {
        ResourceType::Database(name.into())
    }

    /// Create a table resource
    pub fn table(database: impl Into<String>, table: impl Into<String>) -> Self {
        ResourceType::Table {
            database: database.into(),
            table: table.into(),
        }
    }

    /// Create a column resource
    pub fn column(
        database: impl Into<String>,
        table: impl Into<String>,
        column: impl Into<String>,
    ) -> Self {
        ResourceType::Column {
            database: database.into(),
            table: table.into(),
            column: column.into(),
        }
    }

    /// Check if this resource contains another (hierarchical check)
    pub fn contains(&self, other: &ResourceType) -> bool {
        match (self, other) {
            (ResourceType::Server, _) => true,
            (ResourceType::Database(db1), ResourceType::Database(db2)) => db1 == db2,
            (ResourceType::Database(db1), ResourceType::Table { database, .. }) => db1 == database,
            (ResourceType::Database(db1), ResourceType::Column { database, .. }) => db1 == database,
            (
                ResourceType::Table {
                    database: db1,
                    table: t1,
                },
                ResourceType::Table {
                    database: db2,
                    table: t2,
                },
            ) => db1 == db2 && t1 == t2,
            (
                ResourceType::Table {
                    database: db1,
                    table: t1,
                },
                ResourceType::Column {
                    database: db2,
                    table: t2,
                    ..
                },
            ) => db1 == db2 && t1 == t2,
            (
                ResourceType::Column {
                    database: db1,
                    table: t1,
                    column: c1,
                },
                ResourceType::Column {
                    database: db2,
                    table: t2,
                    column: c2,
                },
            ) => db1 == db2 && t1 == t2 && c1 == c2,
            _ => false,
        }
    }

    /// Get the parent resource
    pub fn parent(&self) -> Option<ResourceType> {
        match self {
            ResourceType::Server => None,
            ResourceType::Database(_) => Some(ResourceType::Server),
            ResourceType::Table { database, .. } => Some(ResourceType::Database(database.clone())),
            ResourceType::Column {
                database, table, ..
            } => Some(ResourceType::Table {
                database: database.clone(),
                table: table.clone(),
            }),
        }
    }

    /// Convert to SQL-style string
    pub fn to_sql_string(&self) -> String {
        match self {
            ResourceType::Server => "*.*".to_string(),
            ResourceType::Database(db) => format!("{}.*", db),
            ResourceType::Table { database, table } => format!("{}.{}", database, table),
            ResourceType::Column {
                database,
                table,
                column,
            } => format!("{}.{}.{}", database, table, column),
        }
    }
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sql_string())
    }
}

// ============================================================================
// Permission
// ============================================================================

/// A permission grant combining type and resource
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permission {
    /// The type of permission
    pub permission_type: PermissionType,
    /// The resource this permission applies to
    pub resource: ResourceType,
    /// Whether this permission can be granted to others (WITH GRANT OPTION)
    pub with_grant_option: bool,
}

impl Permission {
    /// Create a new permission
    pub fn new(permission_type: PermissionType, resource: ResourceType) -> Self {
        Self {
            permission_type,
            resource,
            with_grant_option: false,
        }
    }

    /// Create a permission with grant option
    pub fn with_grant(permission_type: PermissionType, resource: ResourceType) -> Self {
        Self {
            permission_type,
            resource,
            with_grant_option: true,
        }
    }

    /// Check if this permission covers another permission
    pub fn covers(&self, other: &Permission) -> bool {
        // Check if permission type implies the other
        if !self.permission_type.implies(&other.permission_type) {
            return false;
        }
        // Check if resource contains the other
        self.resource.contains(&other.resource)
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ON {}", self.permission_type, self.resource)?;
        if self.with_grant_option {
            write!(f, " WITH GRANT OPTION")?;
        }
        Ok(())
    }
}

// ============================================================================
// Role
// ============================================================================

/// A role that groups permissions and can inherit from other roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Unique role name
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Direct permissions assigned to this role
    pub permissions: HashSet<Permission>,
    /// Roles that this role inherits from
    pub inherits_from: HashSet<String>,
    /// Whether this is a system role (cannot be deleted)
    pub is_system: bool,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Role {
    /// Create a new role
    pub fn new(name: impl Into<String>) -> Self {
        let now = current_timestamp();
        Self {
            name: name.into(),
            description: None,
            permissions: HashSet::new(),
            inherits_from: HashSet::new(),
            is_system: false,
            created_at: now,
            modified_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Create a system role
    pub fn system(name: impl Into<String>) -> Self {
        let mut role = Self::new(name);
        role.is_system = true;
        role
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a permission to this role
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.insert(permission);
        self.modified_at = current_timestamp();
    }

    /// Remove a permission from this role
    pub fn remove_permission(&mut self, permission: &Permission) -> bool {
        let removed = self.permissions.remove(permission);
        if removed {
            self.modified_at = current_timestamp();
        }
        removed
    }

    /// Add inheritance from another role
    pub fn inherit_from(&mut self, role_name: impl Into<String>) {
        self.inherits_from.insert(role_name.into());
        self.modified_at = current_timestamp();
    }

    /// Remove inheritance
    pub fn remove_inheritance(&mut self, role_name: &str) -> bool {
        let removed = self.inherits_from.remove(role_name);
        if removed {
            self.modified_at = current_timestamp();
        }
        removed
    }

    /// Check if this role has a direct permission (not inherited)
    pub fn has_direct_permission(&self, permission: &Permission) -> bool {
        self.permissions.iter().any(|p| p.covers(permission))
    }
}

impl PartialEq for Role {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Role {}

impl std::hash::Hash for Role {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

// ============================================================================
// User
// ============================================================================

/// A user with assigned roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique user ID
    pub id: String,
    /// Username
    pub username: String,
    /// Roles assigned to this user
    pub roles: HashSet<String>,
    /// Direct permissions (in addition to role permissions)
    pub direct_permissions: HashSet<Permission>,
    /// Whether the user is active
    pub is_active: bool,
    /// Whether this is a system user
    pub is_system: bool,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Last login timestamp
    pub last_login: Option<u64>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl User {
    /// Create a new user
    pub fn new(id: impl Into<String>, username: impl Into<String>) -> Self {
        let now = current_timestamp();
        Self {
            id: id.into(),
            username: username.into(),
            roles: HashSet::new(),
            direct_permissions: HashSet::new(),
            is_active: true,
            is_system: false,
            created_at: now,
            modified_at: now,
            last_login: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a system user
    pub fn system(id: impl Into<String>, username: impl Into<String>) -> Self {
        let mut user = Self::new(id, username);
        user.is_system = true;
        user
    }

    /// Assign a role to the user
    pub fn assign_role(&mut self, role_name: impl Into<String>) {
        self.roles.insert(role_name.into());
        self.modified_at = current_timestamp();
    }

    /// Remove a role from the user
    pub fn remove_role(&mut self, role_name: &str) -> bool {
        let removed = self.roles.remove(role_name);
        if removed {
            self.modified_at = current_timestamp();
        }
        removed
    }

    /// Add a direct permission
    pub fn add_permission(&mut self, permission: Permission) {
        self.direct_permissions.insert(permission);
        self.modified_at = current_timestamp();
    }

    /// Remove a direct permission
    pub fn remove_permission(&mut self, permission: &Permission) -> bool {
        let removed = self.direct_permissions.remove(permission);
        if removed {
            self.modified_at = current_timestamp();
        }
        removed
    }

    /// Check if user has a direct permission (not from roles)
    pub fn has_direct_permission(&self, permission: &Permission) -> bool {
        self.direct_permissions.iter().any(|p| p.covers(permission))
    }

    /// Record a login
    pub fn record_login(&mut self) {
        self.last_login = Some(current_timestamp());
    }

    /// Deactivate the user
    pub fn deactivate(&mut self) {
        self.is_active = false;
        self.modified_at = current_timestamp();
    }

    /// Activate the user
    pub fn activate(&mut self) {
        self.is_active = true;
        self.modified_at = current_timestamp();
    }
}

// ============================================================================
// Row-Level Security Policy
// ============================================================================

/// Condition type for row-level policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    /// User ID must match column value
    UserIdEquals { column: String },
    /// Column value must equal a specific value
    ValueEquals { column: String, value: String },
    /// Column value must be in a list
    ValueIn { column: String, values: Vec<String> },
    /// Column value must match a pattern
    ValueMatches { column: String, pattern: String },
    /// Custom SQL-like expression
    Expression(String),
    /// Always allow
    Always,
    /// Never allow
    Never,
    /// Combine conditions with AND
    And(Vec<PolicyCondition>),
    /// Combine conditions with OR
    Or(Vec<PolicyCondition>),
    /// Negate a condition
    Not(Box<PolicyCondition>),
}

impl PolicyCondition {
    /// Evaluate the condition against row data
    pub fn evaluate(&self, user_id: &str, row: &HashMap<String, String>) -> bool {
        match self {
            PolicyCondition::UserIdEquals { column } => {
                row.get(column).map(|v| v == user_id).unwrap_or(false)
            }
            PolicyCondition::ValueEquals { column, value } => {
                row.get(column).map(|v| v == value).unwrap_or(false)
            }
            PolicyCondition::ValueIn { column, values } => {
                row.get(column).map(|v| values.contains(v)).unwrap_or(false)
            }
            PolicyCondition::ValueMatches { column, pattern } => row
                .get(column)
                .map(|v| simple_pattern_match(pattern, v))
                .unwrap_or(false),
            PolicyCondition::Expression(expr) => {
                // Simple expression evaluator for common patterns
                evaluate_expression(expr, user_id, row)
            }
            PolicyCondition::Always => true,
            PolicyCondition::Never => false,
            PolicyCondition::And(conditions) => conditions.iter().all(|c| c.evaluate(user_id, row)),
            PolicyCondition::Or(conditions) => conditions.iter().any(|c| c.evaluate(user_id, row)),
            PolicyCondition::Not(condition) => !condition.evaluate(user_id, row),
        }
    }
}

/// Row-level security policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowLevelPolicy {
    /// Unique policy name
    pub name: String,
    /// Target table (database.table format)
    pub table: String,
    /// Database name
    pub database: String,
    /// Policy command type (SELECT, INSERT, UPDATE, DELETE, ALL)
    pub command: PolicyCommand,
    /// Roles this policy applies to (empty = all roles)
    pub roles: HashSet<String>,
    /// Condition for row visibility/access
    pub condition: PolicyCondition,
    /// Whether policy is enabled
    pub enabled: bool,
    /// Whether this is a permissive or restrictive policy
    pub policy_type: PolicyType,
    /// Creation timestamp
    pub created_at: u64,
    /// Description
    pub description: Option<String>,
}

/// Policy command type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyCommand {
    Select,
    Insert,
    Update,
    Delete,
    All,
}

impl PolicyCommand {
    /// Check if this command matches a permission type
    pub fn matches(&self, permission: PermissionType) -> bool {
        match self {
            PolicyCommand::All => true,
            PolicyCommand::Select => permission == PermissionType::Read,
            PolicyCommand::Insert => permission == PermissionType::Create,
            PolicyCommand::Update => permission == PermissionType::Write,
            PolicyCommand::Delete => permission == PermissionType::Delete,
        }
    }
}

/// Policy type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyType {
    /// Permissive policies are ORed together
    Permissive,
    /// Restrictive policies are ANDed together
    Restrictive,
}

impl RowLevelPolicy {
    /// Create a new row-level policy
    pub fn new(
        name: impl Into<String>,
        database: impl Into<String>,
        table: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            database: database.into(),
            table: table.into(),
            command: PolicyCommand::All,
            roles: HashSet::new(),
            condition: PolicyCondition::Always,
            enabled: true,
            policy_type: PolicyType::Permissive,
            created_at: current_timestamp(),
            description: None,
        }
    }

    /// Set the command type
    pub fn for_command(mut self, command: PolicyCommand) -> Self {
        self.command = command;
        self
    }

    /// Set the condition
    pub fn with_condition(mut self, condition: PolicyCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Add a role this policy applies to
    pub fn for_role(mut self, role: impl Into<String>) -> Self {
        self.roles.insert(role.into());
        self
    }

    /// Set policy type
    pub fn policy_type(mut self, policy_type: PolicyType) -> Self {
        self.policy_type = policy_type;
        self
    }

    /// Check if policy applies to a role
    pub fn applies_to_role(&self, role: &str) -> bool {
        self.roles.is_empty() || self.roles.contains(role)
    }

    /// Check if policy applies to any of the given roles
    pub fn applies_to_any_role(&self, roles: &HashSet<String>) -> bool {
        if self.roles.is_empty() {
            return true;
        }
        roles.iter().any(|r| self.roles.contains(r))
    }

    /// Evaluate the policy for a row
    pub fn evaluate(&self, user_id: &str, row: &HashMap<String, String>) -> bool {
        if !self.enabled {
            return true; // Disabled policies allow all
        }
        self.condition.evaluate(user_id, row)
    }
}

// ============================================================================
// RBAC Error
// ============================================================================

/// RBAC system error
#[derive(Debug, Clone, PartialEq)]
pub enum RBACError {
    /// User not found
    UserNotFound(String),
    /// Role not found
    RoleNotFound(String),
    /// Permission denied
    PermissionDenied {
        user: String,
        permission: PermissionType,
        resource: ResourceType,
    },
    /// User is not active
    UserInactive(String),
    /// Cannot modify system role/user
    CannotModifySystem(String),
    /// Role already exists
    RoleAlreadyExists(String),
    /// User already exists
    UserAlreadyExists(String),
    /// Circular inheritance detected
    CircularInheritance(String),
    /// Policy not found
    PolicyNotFound(String),
    /// Policy already exists
    PolicyAlreadyExists(String),
    /// Invalid grant (user doesn't have grant option)
    InvalidGrant(String),
    /// Parse error for SQL GRANT/REVOKE
    ParseError(String),
}

impl std::fmt::Display for RBACError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserNotFound(u) => write!(f, "User not found: {}", u),
            Self::RoleNotFound(r) => write!(f, "Role not found: {}", r),
            Self::PermissionDenied {
                user,
                permission,
                resource,
            } => {
                write!(
                    f,
                    "Permission denied: {} does not have {} on {}",
                    user, permission, resource
                )
            }
            Self::UserInactive(u) => write!(f, "User is inactive: {}", u),
            Self::CannotModifySystem(name) => {
                write!(f, "Cannot modify system role/user: {}", name)
            }
            Self::RoleAlreadyExists(r) => write!(f, "Role already exists: {}", r),
            Self::UserAlreadyExists(u) => write!(f, "User already exists: {}", u),
            Self::CircularInheritance(r) => {
                write!(f, "Circular inheritance detected involving role: {}", r)
            }
            Self::PolicyNotFound(p) => write!(f, "Policy not found: {}", p),
            Self::PolicyAlreadyExists(p) => write!(f, "Policy already exists: {}", p),
            Self::InvalidGrant(msg) => write!(f, "Invalid grant: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for RBACError {}

// ============================================================================
// RBAC Manager
// ============================================================================

/// Result type for RBAC operations
pub type RBACResult<T> = Result<T, RBACError>;

/// RBAC Manager - central access control management
pub struct RBACManager {
    /// Users: user_id -> User
    users: Arc<RwLock<HashMap<String, User>>>,
    /// Roles: role_name -> Role
    roles: Arc<RwLock<HashMap<String, Role>>>,
    /// Row-level policies: policy_name -> RowLevelPolicy
    policies: Arc<RwLock<HashMap<String, RowLevelPolicy>>>,
    /// Permission cache: (user_id, permission_key) -> allowed
    permission_cache: Arc<RwLock<HashMap<(String, String), bool>>>,
    /// Enable caching
    cache_enabled: bool,
}

impl RBACManager {
    /// Create a new RBAC manager
    pub fn new() -> Self {
        let manager = Self {
            users: Arc::new(RwLock::new(HashMap::new())),
            roles: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(HashMap::new())),
            permission_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_enabled: true,
        };
        manager.create_default_roles();
        manager
    }

    /// Create with caching disabled
    pub fn without_cache() -> Self {
        let mut manager = Self::new();
        manager.cache_enabled = false;
        manager
    }

    /// Create default system roles
    fn create_default_roles(&self) {
        let mut roles = crate::lock_util::write_lock(&self.roles);

        // Superuser role - has all permissions
        let mut superuser =
            Role::system("superuser").with_description("Full administrative access");
        superuser.add_permission(Permission::with_grant(
            PermissionType::Admin,
            ResourceType::Server,
        ));
        roles.insert("superuser".to_string(), superuser);

        // Admin role - can manage databases and users
        let mut admin = Role::system("admin").with_description("Database administration");
        admin.add_permission(Permission::new(PermissionType::Admin, ResourceType::Server));
        roles.insert("admin".to_string(), admin);

        // Read-only role
        let mut readonly = Role::system("readonly").with_description("Read-only access");
        readonly.add_permission(Permission::new(PermissionType::Read, ResourceType::Server));
        roles.insert("readonly".to_string(), readonly);

        // Read-write role (inherits from readonly)
        let mut readwrite = Role::system("readwrite").with_description("Read and write access");
        readwrite.inherit_from("readonly");
        readwrite.add_permission(Permission::new(PermissionType::Write, ResourceType::Server));
        readwrite.add_permission(Permission::new(
            PermissionType::Create,
            ResourceType::Server,
        ));
        readwrite.add_permission(Permission::new(
            PermissionType::Delete,
            ResourceType::Server,
        ));
        roles.insert("readwrite".to_string(), readwrite);

        // Create default superuser
        drop(roles);
        let mut users = crate::lock_util::write_lock(&self.users);
        let mut root = User::system("root", "root");
        root.assign_role("superuser");
        users.insert("root".to_string(), root);
    }

    // ========================================================================
    // User Management
    // ========================================================================

    /// Create a new user
    pub fn create_user(&self, user: User) -> RBACResult<()> {
        let mut users = crate::lock_util::write_lock(&self.users);
        if users.contains_key(&user.id) {
            return Err(RBACError::UserAlreadyExists(user.id));
        }
        users.insert(user.id.clone(), user);
        self.invalidate_cache();
        Ok(())
    }

    /// Get a user by ID
    pub fn get_user(&self, user_id: &str) -> RBACResult<User> {
        crate::lock_util::read_lock(&self.users)
            .get(user_id)
            .cloned()
            .ok_or_else(|| RBACError::UserNotFound(user_id.to_string()))
    }

    /// Update a user
    pub fn update_user(&self, user: User) -> RBACResult<()> {
        let mut users = crate::lock_util::write_lock(&self.users);
        if !users.contains_key(&user.id) {
            return Err(RBACError::UserNotFound(user.id));
        }
        if let Some(existing) = users.get(&user.id) {
            if existing.is_system {
                return Err(RBACError::CannotModifySystem(user.id));
            }
        }
        users.insert(user.id.clone(), user);
        self.invalidate_cache();
        Ok(())
    }

    /// Delete a user
    pub fn delete_user(&self, user_id: &str) -> RBACResult<()> {
        let mut users = crate::lock_util::write_lock(&self.users);
        if let Some(user) = users.get(user_id) {
            if user.is_system {
                return Err(RBACError::CannotModifySystem(user_id.to_string()));
            }
        } else {
            return Err(RBACError::UserNotFound(user_id.to_string()));
        }
        users.remove(user_id);
        self.invalidate_cache();
        Ok(())
    }

    /// Assign a role to a user
    pub fn assign_role_to_user(&self, user_id: &str, role_name: &str) -> RBACResult<()> {
        // Verify role exists
        if !crate::lock_util::read_lock(&self.roles).contains_key(role_name) {
            return Err(RBACError::RoleNotFound(role_name.to_string()));
        }

        let mut users = crate::lock_util::write_lock(&self.users);
        let user = users
            .get_mut(user_id)
            .ok_or_else(|| RBACError::UserNotFound(user_id.to_string()))?;
        user.assign_role(role_name);
        self.invalidate_cache();
        Ok(())
    }

    /// Remove a role from a user
    pub fn remove_role_from_user(&self, user_id: &str, role_name: &str) -> RBACResult<()> {
        let mut users = crate::lock_util::write_lock(&self.users);
        let user = users
            .get_mut(user_id)
            .ok_or_else(|| RBACError::UserNotFound(user_id.to_string()))?;
        user.remove_role(role_name);
        self.invalidate_cache();
        Ok(())
    }

    /// List all users
    pub fn list_users(&self) -> Vec<User> {
        crate::lock_util::read_lock(&self.users)
            .values()
            .cloned()
            .collect()
    }

    // ========================================================================
    // Role Management
    // ========================================================================

    /// Create a new role
    pub fn create_role(&self, role: Role) -> RBACResult<()> {
        let mut roles = crate::lock_util::write_lock(&self.roles);
        if roles.contains_key(&role.name) {
            return Err(RBACError::RoleAlreadyExists(role.name));
        }
        // Verify inherited roles exist
        for parent in &role.inherits_from {
            if !roles.contains_key(parent) {
                return Err(RBACError::RoleNotFound(parent.clone()));
            }
        }
        roles.insert(role.name.clone(), role);
        self.invalidate_cache();
        Ok(())
    }

    /// Get a role by name
    pub fn get_role(&self, role_name: &str) -> RBACResult<Role> {
        crate::lock_util::read_lock(&self.roles)
            .get(role_name)
            .cloned()
            .ok_or_else(|| RBACError::RoleNotFound(role_name.to_string()))
    }

    /// Update a role
    pub fn update_role(&self, role: Role) -> RBACResult<()> {
        let mut roles = crate::lock_util::write_lock(&self.roles);
        if let Some(existing) = roles.get(&role.name) {
            if existing.is_system {
                return Err(RBACError::CannotModifySystem(role.name));
            }
        } else {
            return Err(RBACError::RoleNotFound(role.name));
        }
        // Check for circular inheritance
        if self.would_create_cycle(&role, &roles) {
            return Err(RBACError::CircularInheritance(role.name));
        }
        roles.insert(role.name.clone(), role);
        self.invalidate_cache();
        Ok(())
    }

    /// Delete a role
    pub fn delete_role(&self, role_name: &str) -> RBACResult<()> {
        let mut roles = crate::lock_util::write_lock(&self.roles);
        if let Some(role) = roles.get(role_name) {
            if role.is_system {
                return Err(RBACError::CannotModifySystem(role_name.to_string()));
            }
        } else {
            return Err(RBACError::RoleNotFound(role_name.to_string()));
        }

        // Remove this role from any users
        let mut users = crate::lock_util::write_lock(&self.users);
        for user in users.values_mut() {
            user.remove_role(role_name);
        }
        drop(users);

        // Remove this role from any inheriting roles
        for role in roles.values_mut() {
            role.remove_inheritance(role_name);
        }

        roles.remove(role_name);
        self.invalidate_cache();
        Ok(())
    }

    /// Add role inheritance
    pub fn add_role_inheritance(&self, role_name: &str, parent_role: &str) -> RBACResult<()> {
        let mut roles = crate::lock_util::write_lock(&self.roles);

        // Verify both roles exist
        if !roles.contains_key(parent_role) {
            return Err(RBACError::RoleNotFound(parent_role.to_string()));
        }

        // First check if role exists and is not a system role
        {
            let role = roles
                .get(role_name)
                .ok_or_else(|| RBACError::RoleNotFound(role_name.to_string()))?;

            if role.is_system {
                return Err(RBACError::CannotModifySystem(role_name.to_string()));
            }

            // Check for cycles
            let mut test_role = role.clone();
            test_role.inherit_from(parent_role);
            if self.would_create_cycle(&test_role, &roles) {
                return Err(RBACError::CircularInheritance(role_name.to_string()));
            }
        }

        // Now we can safely get mutable reference (confirmed to exist above)
        let role = roles
            .get_mut(role_name)
            .expect("role confirmed to exist above");
        role.inherit_from(parent_role);
        self.invalidate_cache();
        Ok(())
    }

    /// Check if adding inheritance would create a cycle
    fn would_create_cycle(&self, role: &Role, roles: &HashMap<String, Role>) -> bool {
        // Check if following the inheritance chain from the role leads back to itself
        let mut visited = HashSet::new();
        let mut stack: Vec<String> = role.inherits_from.iter().cloned().collect();

        while let Some(current) = stack.pop() {
            // If we reach back to the role itself, we have a cycle
            if current == role.name {
                return true;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(r) = roles.get(&current) {
                for parent in &r.inherits_from {
                    stack.push(parent.clone());
                }
            }
        }
        false
    }

    /// List all roles
    pub fn list_roles(&self) -> Vec<Role> {
        crate::lock_util::read_lock(&self.roles)
            .values()
            .cloned()
            .collect()
    }

    /// Get all permissions for a role (including inherited)
    pub fn get_role_permissions(&self, role_name: &str) -> RBACResult<HashSet<Permission>> {
        let roles = crate::lock_util::read_lock(&self.roles);
        let role = roles
            .get(role_name)
            .ok_or_else(|| RBACError::RoleNotFound(role_name.to_string()))?;

        let mut permissions = role.permissions.clone();
        let mut visited = HashSet::new();
        let mut to_visit: Vec<String> = role.inherits_from.iter().cloned().collect();

        while let Some(parent_name) = to_visit.pop() {
            if visited.contains(&parent_name) {
                continue;
            }
            visited.insert(parent_name.clone());

            if let Some(parent) = roles.get(&parent_name) {
                permissions.extend(parent.permissions.clone());
                to_visit.extend(parent.inherits_from.iter().cloned());
            }
        }

        Ok(permissions)
    }

    // ========================================================================
    // Permission Checking
    // ========================================================================

    /// Check if a user has a specific permission
    pub fn check_permission(
        &self,
        user_id: &str,
        permission_type: PermissionType,
        resource: &ResourceType,
    ) -> RBACResult<bool> {
        let users = crate::lock_util::read_lock(&self.users);
        let user = users
            .get(user_id)
            .ok_or_else(|| RBACError::UserNotFound(user_id.to_string()))?;

        if !user.is_active {
            return Err(RBACError::UserInactive(user_id.to_string()));
        }

        // Check cache
        if self.cache_enabled {
            let cache_key = (
                user_id.to_string(),
                format!("{:?}:{}", permission_type, resource),
            );
            if let Some(&cached) =
                crate::lock_util::read_lock(&self.permission_cache).get(&cache_key)
            {
                return Ok(cached);
            }
        }

        let permission = Permission::new(permission_type, resource.clone());

        // Check direct user permissions
        if user.has_direct_permission(&permission) {
            self.cache_result(user_id, &permission, true);
            return Ok(true);
        }

        // Check role permissions
        let roles = crate::lock_util::read_lock(&self.roles);
        for role_name in &user.roles {
            if let Some(role) = roles.get(role_name) {
                if self.role_has_permission_recursive(
                    role,
                    &permission,
                    &roles,
                    &mut HashSet::new(),
                ) {
                    self.cache_result(user_id, &permission, true);
                    return Ok(true);
                }
            }
        }

        self.cache_result(user_id, &permission, false);
        Ok(false)
    }

    /// Recursively check if a role has a permission (including inheritance)
    fn role_has_permission_recursive(
        &self,
        role: &Role,
        permission: &Permission,
        all_roles: &HashMap<String, Role>,
        visited: &mut HashSet<String>,
    ) -> bool {
        if visited.contains(&role.name) {
            return false;
        }
        visited.insert(role.name.clone());

        // Check direct permissions
        if role.has_direct_permission(permission) {
            return true;
        }

        // Check inherited roles
        for parent_name in &role.inherits_from {
            if let Some(parent) = all_roles.get(parent_name) {
                if self.role_has_permission_recursive(parent, permission, all_roles, visited) {
                    return true;
                }
            }
        }

        false
    }

    /// Require permission (returns error if denied)
    pub fn require_permission(
        &self,
        user_id: &str,
        permission_type: PermissionType,
        resource: &ResourceType,
    ) -> RBACResult<()> {
        if self.check_permission(user_id, permission_type, resource)? {
            Ok(())
        } else {
            Err(RBACError::PermissionDenied {
                user: user_id.to_string(),
                permission: permission_type,
                resource: resource.clone(),
            })
        }
    }

    /// Get all effective permissions for a user
    pub fn get_user_permissions(&self, user_id: &str) -> RBACResult<HashSet<Permission>> {
        let users = crate::lock_util::read_lock(&self.users);
        let user = users
            .get(user_id)
            .ok_or_else(|| RBACError::UserNotFound(user_id.to_string()))?;

        let mut permissions = user.direct_permissions.clone();

        // Add permissions from all roles
        for role_name in &user.roles {
            if let Ok(role_perms) = self.get_role_permissions(role_name) {
                permissions.extend(role_perms);
            }
        }

        Ok(permissions)
    }

    /// Cache a permission check result
    fn cache_result(&self, user_id: &str, permission: &Permission, allowed: bool) {
        if self.cache_enabled {
            let cache_key = (
                user_id.to_string(),
                format!("{:?}:{}", permission.permission_type, permission.resource),
            );
            crate::lock_util::write_lock(&self.permission_cache).insert(cache_key, allowed);
        }
    }

    /// Invalidate the permission cache
    pub fn invalidate_cache(&self) {
        if self.cache_enabled {
            crate::lock_util::write_lock(&self.permission_cache).clear();
        }
    }

    // ========================================================================
    // Row-Level Security Policies
    // ========================================================================

    /// Create a row-level policy
    pub fn create_policy(&self, policy: RowLevelPolicy) -> RBACResult<()> {
        let mut policies = crate::lock_util::write_lock(&self.policies);
        if policies.contains_key(&policy.name) {
            return Err(RBACError::PolicyAlreadyExists(policy.name));
        }
        policies.insert(policy.name.clone(), policy);
        Ok(())
    }

    /// Get a policy by name
    pub fn get_policy(&self, name: &str) -> RBACResult<RowLevelPolicy> {
        crate::lock_util::read_lock(&self.policies)
            .get(name)
            .cloned()
            .ok_or_else(|| RBACError::PolicyNotFound(name.to_string()))
    }

    /// Delete a policy
    pub fn delete_policy(&self, name: &str) -> RBACResult<()> {
        let mut policies = crate::lock_util::write_lock(&self.policies);
        if policies.remove(name).is_none() {
            return Err(RBACError::PolicyNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Enable/disable a policy
    pub fn set_policy_enabled(&self, name: &str, enabled: bool) -> RBACResult<()> {
        let mut policies = crate::lock_util::write_lock(&self.policies);
        let policy = policies
            .get_mut(name)
            .ok_or_else(|| RBACError::PolicyNotFound(name.to_string()))?;
        policy.enabled = enabled;
        Ok(())
    }

    /// List all policies
    pub fn list_policies(&self) -> Vec<RowLevelPolicy> {
        crate::lock_util::read_lock(&self.policies)
            .values()
            .cloned()
            .collect()
    }

    /// Get policies for a specific table
    pub fn get_table_policies(&self, database: &str, table: &str) -> Vec<RowLevelPolicy> {
        crate::lock_util::read_lock(&self.policies)
            .values()
            .filter(|p| p.database == database && p.table == table && p.enabled)
            .cloned()
            .collect()
    }

    /// Check if a row is accessible based on RLS policies
    pub fn check_row_access(
        &self,
        user_id: &str,
        database: &str,
        table: &str,
        permission_type: PermissionType,
        row: &HashMap<String, String>,
    ) -> RBACResult<bool> {
        let users = crate::lock_util::read_lock(&self.users);
        let user = users
            .get(user_id)
            .ok_or_else(|| RBACError::UserNotFound(user_id.to_string()))?;

        if !user.is_active {
            return Err(RBACError::UserInactive(user_id.to_string()));
        }

        let policies = self.get_table_policies(database, table);

        // If no policies, allow access
        if policies.is_empty() {
            return Ok(true);
        }

        // Separate permissive and restrictive policies
        let (permissive, restrictive): (Vec<_>, Vec<_>) = policies
            .into_iter()
            .filter(|p| p.command.matches(permission_type))
            .filter(|p| p.applies_to_any_role(&user.roles))
            .partition(|p| p.policy_type == PolicyType::Permissive);

        // All restrictive policies must pass
        let restrictive_pass = restrictive.iter().all(|p| p.evaluate(user_id, row));

        if !restrictive_pass {
            return Ok(false);
        }

        // If no permissive policies, allow (only restrictive applied)
        if permissive.is_empty() {
            return Ok(true);
        }

        // At least one permissive policy must pass
        let permissive_pass = permissive.iter().any(|p| p.evaluate(user_id, row));

        Ok(permissive_pass)
    }

    // ========================================================================
    // SQL GRANT/REVOKE Support
    // ========================================================================

    /// Execute a GRANT statement
    /// Format: GRANT permission ON resource TO user/role [WITH GRANT OPTION]
    pub fn execute_grant(&self, sql: &str) -> RBACResult<()> {
        let parsed = parse_grant_revoke(sql)?;

        match parsed {
            GrantRevoke::Grant {
                permissions,
                resource,
                grantee,
                with_grant_option,
            } => {
                for perm_type in permissions {
                    let permission = if with_grant_option {
                        Permission::with_grant(perm_type, resource.clone())
                    } else {
                        Permission::new(perm_type, resource.clone())
                    };

                    // Try as role first, then as user
                    if crate::lock_util::read_lock(&self.roles).contains_key(&grantee) {
                        let mut roles = crate::lock_util::write_lock(&self.roles);
                        let role = roles
                            .get_mut(&grantee)
                            .expect("grantee confirmed to exist in roles");
                        if role.is_system {
                            return Err(RBACError::CannotModifySystem(grantee));
                        }
                        role.add_permission(permission);
                    } else if crate::lock_util::read_lock(&self.users).contains_key(&grantee) {
                        let mut users = crate::lock_util::write_lock(&self.users);
                        let user = users
                            .get_mut(&grantee)
                            .expect("grantee confirmed to exist in users");
                        user.add_permission(permission);
                    } else {
                        return Err(RBACError::UserNotFound(grantee));
                    }
                }
                self.invalidate_cache();
                Ok(())
            }
            GrantRevoke::Revoke { .. } => Err(RBACError::ParseError(
                "Expected GRANT statement".to_string(),
            )),
        }
    }

    /// Execute a REVOKE statement
    /// Format: REVOKE permission ON resource FROM user/role
    pub fn execute_revoke(&self, sql: &str) -> RBACResult<()> {
        let parsed = parse_grant_revoke(sql)?;

        match parsed {
            GrantRevoke::Revoke {
                permissions,
                resource,
                grantee,
            } => {
                for perm_type in permissions {
                    let permission = Permission::new(perm_type, resource.clone());

                    // Try as role first, then as user
                    if crate::lock_util::read_lock(&self.roles).contains_key(&grantee) {
                        let mut roles = crate::lock_util::write_lock(&self.roles);
                        let role = roles
                            .get_mut(&grantee)
                            .expect("grantee confirmed to exist in roles");
                        if role.is_system {
                            return Err(RBACError::CannotModifySystem(grantee));
                        }
                        role.remove_permission(&permission);
                    } else if crate::lock_util::read_lock(&self.users).contains_key(&grantee) {
                        let mut users = crate::lock_util::write_lock(&self.users);
                        let user = users
                            .get_mut(&grantee)
                            .expect("grantee confirmed to exist in users");
                        user.remove_permission(&permission);
                    } else {
                        return Err(RBACError::UserNotFound(grantee));
                    }
                }
                self.invalidate_cache();
                Ok(())
            }
            GrantRevoke::Grant { .. } => Err(RBACError::ParseError(
                "Expected REVOKE statement".to_string(),
            )),
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get RBAC statistics
    pub fn stats(&self) -> RBACStats {
        RBACStats {
            user_count: crate::lock_util::read_lock(&self.users).len(),
            role_count: crate::lock_util::read_lock(&self.roles).len(),
            policy_count: crate::lock_util::read_lock(&self.policies).len(),
            cache_size: crate::lock_util::read_lock(&self.permission_cache).len(),
        }
    }
}

impl Default for RBACManager {
    fn default() -> Self {
        Self::new()
    }
}

/// RBAC statistics
#[derive(Debug, Clone)]
pub struct RBACStats {
    pub user_count: usize,
    pub role_count: usize,
    pub policy_count: usize,
    pub cache_size: usize,
}

// ============================================================================
// Permission Checking Middleware
// ============================================================================

/// Context for permission checking
#[derive(Debug, Clone)]
pub struct AccessContext {
    /// User ID
    pub user_id: String,
    /// Current database
    pub database: Option<String>,
    /// Session roles (can be different from user's default roles)
    pub session_roles: Option<HashSet<String>>,
}

impl AccessContext {
    /// Create a new access context
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            database: None,
            session_roles: None,
        }
    }

    /// Set the current database
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    /// Set session-specific roles
    pub fn with_roles(mut self, roles: HashSet<String>) -> Self {
        self.session_roles = Some(roles);
        self
    }
}

/// Permission checking middleware
pub struct PermissionMiddleware {
    rbac: Arc<RBACManager>,
}

impl PermissionMiddleware {
    /// Create new middleware
    pub fn new(rbac: Arc<RBACManager>) -> Self {
        Self { rbac }
    }

    /// Check table-level permission
    pub fn check_table_access(
        &self,
        ctx: &AccessContext,
        table: &str,
        permission: PermissionType,
    ) -> RBACResult<()> {
        let database = ctx.database.as_deref().unwrap_or("default");
        let resource = ResourceType::table(database, table);
        self.rbac
            .require_permission(&ctx.user_id, permission, &resource)
    }

    /// Check column-level permission
    pub fn check_column_access(
        &self,
        ctx: &AccessContext,
        table: &str,
        column: &str,
        permission: PermissionType,
    ) -> RBACResult<()> {
        let database = ctx.database.as_deref().unwrap_or("default");
        let resource = ResourceType::column(database, table, column);
        self.rbac
            .require_permission(&ctx.user_id, permission, &resource)
    }

    /// Check database-level permission
    pub fn check_database_access(
        &self,
        ctx: &AccessContext,
        permission: PermissionType,
    ) -> RBACResult<()> {
        let database = ctx.database.as_deref().unwrap_or("default");
        let resource = ResourceType::database(database);
        self.rbac
            .require_permission(&ctx.user_id, permission, &resource)
    }

    /// Check row-level access
    pub fn check_row_access(
        &self,
        ctx: &AccessContext,
        table: &str,
        permission: PermissionType,
        row: &HashMap<String, String>,
    ) -> RBACResult<bool> {
        let database = ctx.database.as_deref().unwrap_or("default");
        self.rbac
            .check_row_access(&ctx.user_id, database, table, permission, row)
    }

    /// Filter rows based on RLS policies
    pub fn filter_rows(
        &self,
        ctx: &AccessContext,
        table: &str,
        permission: PermissionType,
        rows: Vec<HashMap<String, String>>,
    ) -> RBACResult<Vec<HashMap<String, String>>> {
        let database = ctx.database.as_deref().unwrap_or("default");
        let mut result = Vec::new();

        for row in rows {
            if self
                .rbac
                .check_row_access(&ctx.user_id, database, table, permission, &row)?
            {
                result.push(row);
            }
        }

        Ok(result)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Simple pattern matching (supports * and ? wildcards)
fn simple_pattern_match(pattern: &str, text: &str) -> bool {
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    pattern_match_helper(&pattern_chars, &text_chars, 0, 0)
}

fn pattern_match_helper(pattern: &[char], text: &[char], pi: usize, ti: usize) -> bool {
    if pi == pattern.len() {
        return ti == text.len();
    }

    if pattern[pi] == '*' {
        // Match zero or more characters
        for i in ti..=text.len() {
            if pattern_match_helper(pattern, text, pi + 1, i) {
                return true;
            }
        }
        return false;
    }

    if ti < text.len() && (pattern[pi] == '?' || pattern[pi] == text[ti]) {
        return pattern_match_helper(pattern, text, pi + 1, ti + 1);
    }

    false
}

/// Evaluate a simple expression
fn evaluate_expression(expr: &str, user_id: &str, row: &HashMap<String, String>) -> bool {
    // Simple expression parser for common patterns
    let expr = expr.trim();

    // Handle user_id() function
    let expr = expr.replace("user_id()", &format!("'{}'", user_id));

    // Parse simple comparisons: column = 'value' or column = column
    if let Some((left, right)) = expr.split_once('=') {
        let left = left.trim().trim_matches('\'').trim_matches('"');
        let right = right.trim().trim_matches('\'').trim_matches('"');

        let left_val = row.get(left).map(|s| s.as_str()).unwrap_or(left);
        let right_val = row.get(right).map(|s| s.as_str()).unwrap_or(right);

        return left_val == right_val;
    }

    // Default to true for unparseable expressions
    true
}

/// Parsed GRANT/REVOKE statement
enum GrantRevoke {
    Grant {
        permissions: Vec<PermissionType>,
        resource: ResourceType,
        grantee: String,
        with_grant_option: bool,
    },
    Revoke {
        permissions: Vec<PermissionType>,
        resource: ResourceType,
        grantee: String,
    },
}

/// Parse a GRANT or REVOKE statement
fn parse_grant_revoke(sql: &str) -> RBACResult<GrantRevoke> {
    let sql = sql.trim();
    let upper = sql.to_uppercase();
    let tokens: Vec<&str> = sql.split_whitespace().collect();

    if tokens.is_empty() {
        return Err(RBACError::ParseError("Empty statement".to_string()));
    }

    let is_grant = upper.starts_with("GRANT");
    let is_revoke = upper.starts_with("REVOKE");

    if !is_grant && !is_revoke {
        return Err(RBACError::ParseError(
            "Expected GRANT or REVOKE".to_string(),
        ));
    }

    // Find ON, TO/FROM positions
    let upper_tokens: Vec<String> = tokens.iter().map(|t| t.to_uppercase()).collect();
    let on_pos = upper_tokens
        .iter()
        .position(|t| t == "ON")
        .ok_or_else(|| RBACError::ParseError("Missing ON clause".to_string()))?;

    let target_keyword = if is_grant { "TO" } else { "FROM" };
    let target_pos = upper_tokens
        .iter()
        .position(|t| t == target_keyword)
        .ok_or_else(|| RBACError::ParseError(format!("Missing {} clause", target_keyword)))?;

    // Parse permissions (between GRANT/REVOKE and ON)
    let perm_tokens = &tokens[1..on_pos];
    let perm_str = perm_tokens.join(" ").replace(',', " ");
    let permissions: Vec<PermissionType> = perm_str
        .split_whitespace()
        .filter_map(|s| PermissionType::from_str(s))
        .collect();

    if permissions.is_empty() {
        return Err(RBACError::ParseError(
            "No valid permissions found".to_string(),
        ));
    }

    // Parse resource (between ON and TO/FROM)
    let resource_tokens = &tokens[on_pos + 1..target_pos];
    let resource_str = resource_tokens.join(".");
    let resource = parse_resource(&resource_str)?;

    // Parse grantee
    let mut grantee_end = tokens.len();
    let with_grant_option = if is_grant {
        if upper.contains("WITH GRANT OPTION") {
            // Find where "WITH" starts
            if let Some(with_pos) = upper_tokens.iter().position(|t| t == "WITH") {
                grantee_end = with_pos;
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    let grantee = tokens[target_pos + 1..grantee_end].join(" ");

    if grantee.is_empty() {
        return Err(RBACError::ParseError("Missing grantee".to_string()));
    }

    if is_grant {
        Ok(GrantRevoke::Grant {
            permissions,
            resource,
            grantee,
            with_grant_option,
        })
    } else {
        Ok(GrantRevoke::Revoke {
            permissions,
            resource,
            grantee,
        })
    }
}

/// Parse a resource string
fn parse_resource(s: &str) -> RBACResult<ResourceType> {
    let s = s.trim();

    if s == "*" || s == "*.*" {
        return Ok(ResourceType::Server);
    }

    let parts: Vec<&str> = s.split('.').collect();

    match parts.len() {
        1 => {
            if parts[0] == "*" {
                Ok(ResourceType::Server)
            } else {
                Ok(ResourceType::Database(parts[0].to_string()))
            }
        }
        2 => {
            if parts[1] == "*" {
                Ok(ResourceType::Database(parts[0].to_string()))
            } else {
                Ok(ResourceType::Table {
                    database: parts[0].to_string(),
                    table: parts[1].to_string(),
                })
            }
        }
        3 => Ok(ResourceType::Column {
            database: parts[0].to_string(),
            table: parts[1].to_string(),
            column: parts[2].to_string(),
        }),
        _ => Err(RBACError::ParseError(format!(
            "Invalid resource format: {}",
            s
        ))),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_type_parsing() {
        assert_eq!(
            PermissionType::from_str("SELECT"),
            Some(PermissionType::Read)
        );
        assert_eq!(PermissionType::from_str("read"), Some(PermissionType::Read));
        assert_eq!(
            PermissionType::from_str("INSERT"),
            Some(PermissionType::Create)
        );
        assert_eq!(
            PermissionType::from_str("ALL PRIVILEGES"),
            Some(PermissionType::Admin)
        );
        assert_eq!(PermissionType::from_str("INVALID"), None);
    }

    #[test]
    fn test_permission_implies() {
        assert!(PermissionType::Admin.implies(&PermissionType::Read));
        assert!(PermissionType::Admin.implies(&PermissionType::Write));
        assert!(PermissionType::Admin.implies(&PermissionType::Admin));
        assert!(PermissionType::Read.implies(&PermissionType::Read));
        assert!(!PermissionType::Read.implies(&PermissionType::Write));
    }

    #[test]
    fn test_resource_hierarchy() {
        let server = ResourceType::Server;
        let database = ResourceType::database("mydb");
        let table = ResourceType::table("mydb", "users");
        let column = ResourceType::column("mydb", "users", "email");

        // Server contains everything
        assert!(server.contains(&server));
        assert!(server.contains(&database));
        assert!(server.contains(&table));
        assert!(server.contains(&column));

        // Database contains its tables and columns
        assert!(database.contains(&database));
        assert!(database.contains(&table));
        assert!(database.contains(&column));
        assert!(!database.contains(&server));

        // Table contains its columns
        assert!(table.contains(&table));
        assert!(table.contains(&column));
        assert!(!table.contains(&database));
        assert!(!table.contains(&server));

        // Column only contains itself
        assert!(column.contains(&column));
        assert!(!column.contains(&table));
    }

    #[test]
    fn test_permission_covers() {
        let admin_server = Permission::new(PermissionType::Admin, ResourceType::Server);
        let read_table =
            Permission::new(PermissionType::Read, ResourceType::table("mydb", "users"));
        let write_table =
            Permission::new(PermissionType::Write, ResourceType::table("mydb", "users"));

        assert!(admin_server.covers(&read_table));
        assert!(admin_server.covers(&write_table));
        assert!(!read_table.covers(&write_table));
        assert!(!read_table.covers(&admin_server));
    }

    #[test]
    fn test_role_creation_and_permissions() {
        let mut role = Role::new("developer");
        role.add_permission(Permission::new(
            PermissionType::Read,
            ResourceType::database("dev_db"),
        ));
        role.add_permission(Permission::new(
            PermissionType::Write,
            ResourceType::database("dev_db"),
        ));

        assert_eq!(role.permissions.len(), 2);
        assert!(role.has_direct_permission(&Permission::new(
            PermissionType::Read,
            ResourceType::table("dev_db", "any_table"),
        )));
    }

    #[test]
    fn test_user_role_assignment() {
        let mut user = User::new("u1", "alice");
        user.assign_role("developer");
        user.assign_role("analyst");

        assert!(user.roles.contains("developer"));
        assert!(user.roles.contains("analyst"));
        assert_eq!(user.roles.len(), 2);

        user.remove_role("developer");
        assert!(!user.roles.contains("developer"));
        assert_eq!(user.roles.len(), 1);
    }

    #[test]
    fn test_rbac_manager_default_roles() {
        let rbac = RBACManager::new();

        assert!(rbac.get_role("superuser").is_ok());
        assert!(rbac.get_role("admin").is_ok());
        assert!(rbac.get_role("readonly").is_ok());
        assert!(rbac.get_role("readwrite").is_ok());
    }

    #[test]
    fn test_rbac_user_management() {
        let rbac = RBACManager::new();

        let user = User::new("u1", "testuser");
        rbac.create_user(user).unwrap();

        let retrieved = rbac.get_user("u1").unwrap();
        assert_eq!(retrieved.username, "testuser");

        // Duplicate should fail
        let user2 = User::new("u1", "another");
        assert!(matches!(
            rbac.create_user(user2),
            Err(RBACError::UserAlreadyExists(_))
        ));
    }

    #[test]
    fn test_rbac_permission_check() {
        let rbac = RBACManager::new();

        // Root user should have all permissions
        let result = rbac
            .check_permission("root", PermissionType::Admin, &ResourceType::Server)
            .unwrap();
        assert!(result);

        // Create a limited user
        let mut user = User::new("limited", "limited_user");
        user.assign_role("readonly");
        rbac.create_user(user).unwrap();

        // Should have read permission
        let result = rbac
            .check_permission("limited", PermissionType::Read, &ResourceType::Server)
            .unwrap();
        assert!(result);

        // Should not have write permission
        let result = rbac
            .check_permission("limited", PermissionType::Write, &ResourceType::Server)
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_role_inheritance() {
        let rbac = RBACManager::new();

        // Create parent role
        let mut parent = Role::new("parent");
        parent.add_permission(Permission::new(
            PermissionType::Read,
            ResourceType::database("test"),
        ));
        rbac.create_role(parent).unwrap();

        // Create child role that inherits from parent
        let mut child = Role::new("child");
        child.inherit_from("parent");
        child.add_permission(Permission::new(
            PermissionType::Write,
            ResourceType::database("test"),
        ));
        rbac.create_role(child).unwrap();

        // User with child role should have both read and write
        let mut user = User::new("u1", "testuser");
        user.assign_role("child");
        rbac.create_user(user).unwrap();

        assert!(
            rbac.check_permission("u1", PermissionType::Read, &ResourceType::database("test"))
                .unwrap()
        );
        assert!(
            rbac.check_permission("u1", PermissionType::Write, &ResourceType::database("test"))
                .unwrap()
        );
    }

    #[test]
    fn test_circular_inheritance_detection() {
        let rbac = RBACManager::new();

        let role_a = Role::new("role_a");
        rbac.create_role(role_a).unwrap();

        let mut role_b = Role::new("role_b");
        role_b.inherit_from("role_a");
        rbac.create_role(role_b).unwrap();

        // Try to make role_a inherit from role_b (would create cycle)
        let result = rbac.add_role_inheritance("role_a", "role_b");
        assert!(matches!(result, Err(RBACError::CircularInheritance(_))));
    }

    #[test]
    fn test_row_level_policy() {
        let rbac = RBACManager::new();

        // Create a policy that only allows users to see their own rows
        let policy = RowLevelPolicy::new("user_isolation", "testdb", "users")
            .for_command(PolicyCommand::Select)
            .with_condition(PolicyCondition::UserIdEquals {
                column: "owner_id".to_string(),
            });

        rbac.create_policy(policy).unwrap();

        // Create test user
        let user = User::new("user1", "testuser");
        rbac.create_user(user).unwrap();

        // Row owned by user1
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), "1".to_string());
        row1.insert("owner_id".to_string(), "user1".to_string());

        // Row owned by someone else
        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), "2".to_string());
        row2.insert("owner_id".to_string(), "user2".to_string());

        assert!(
            rbac.check_row_access("user1", "testdb", "users", PermissionType::Read, &row1)
                .unwrap()
        );
        assert!(
            !rbac
                .check_row_access("user1", "testdb", "users", PermissionType::Read, &row2)
                .unwrap()
        );
    }

    #[test]
    fn test_grant_statement_parsing() {
        let rbac = RBACManager::new();

        // Create a user to grant permissions to
        let user = User::new("alice", "alice");
        rbac.create_user(user).unwrap();

        // Execute GRANT
        rbac.execute_grant("GRANT SELECT ON mydb.users TO alice")
            .unwrap();

        // Check the permission was added
        let alice = rbac.get_user("alice").unwrap();
        assert!(alice.has_direct_permission(&Permission::new(
            PermissionType::Read,
            ResourceType::table("mydb", "users"),
        )));
    }

    #[test]
    fn test_revoke_statement_parsing() {
        let rbac = RBACManager::new();

        // Create a user with permissions
        let mut user = User::new("bob", "bob");
        user.add_permission(Permission::new(
            PermissionType::Read,
            ResourceType::database("testdb"),
        ));
        rbac.create_user(user).unwrap();

        // Execute REVOKE
        rbac.execute_revoke("REVOKE SELECT ON testdb FROM bob")
            .unwrap();

        // Check the permission was removed
        let bob = rbac.get_user("bob").unwrap();
        assert!(!bob.has_direct_permission(&Permission::new(
            PermissionType::Read,
            ResourceType::database("testdb"),
        )));
    }

    #[test]
    fn test_permission_middleware() {
        let rbac = Arc::new(RBACManager::new());
        let middleware = PermissionMiddleware::new(rbac.clone());

        // Create a user with table-level read permission
        let mut user = User::new("reader", "reader");
        user.add_permission(Permission::new(
            PermissionType::Read,
            ResourceType::table("mydb", "data"),
        ));
        rbac.create_user(user).unwrap();

        let ctx = AccessContext::new("reader").with_database("mydb");

        // Should have read access to the table
        assert!(
            middleware
                .check_table_access(&ctx, "data", PermissionType::Read)
                .is_ok()
        );

        // Should not have write access
        assert!(
            middleware
                .check_table_access(&ctx, "data", PermissionType::Write)
                .is_err()
        );
    }

    #[test]
    fn test_policy_condition_and_or() {
        let cond = PolicyCondition::And(vec![
            PolicyCondition::ValueIn {
                column: "status".to_string(),
                values: vec!["active".to_string(), "pending".to_string()],
            },
            PolicyCondition::Or(vec![
                PolicyCondition::UserIdEquals {
                    column: "owner".to_string(),
                },
                PolicyCondition::ValueEquals {
                    column: "public".to_string(),
                    value: "true".to_string(),
                },
            ]),
        ]);

        let mut row = HashMap::new();
        row.insert("status".to_string(), "active".to_string());
        row.insert("owner".to_string(), "user1".to_string());
        row.insert("public".to_string(), "false".to_string());

        assert!(cond.evaluate("user1", &row));
        assert!(!cond.evaluate("user2", &row));

        // Make it public
        row.insert("public".to_string(), "true".to_string());
        assert!(cond.evaluate("user2", &row));
    }

    #[test]
    fn test_system_role_protection() {
        let rbac = RBACManager::new();

        // Should not be able to delete system roles
        assert!(matches!(
            rbac.delete_role("superuser"),
            Err(RBACError::CannotModifySystem(_))
        ));

        // Should not be able to delete system users
        assert!(matches!(
            rbac.delete_user("root"),
            Err(RBACError::CannotModifySystem(_))
        ));
    }

    #[test]
    fn test_user_activation_deactivation() {
        let rbac = RBACManager::new();

        let user = User::new("temp", "temporary");
        rbac.create_user(user).unwrap();

        // Grant permission
        rbac.assign_role_to_user("temp", "readonly").unwrap();

        // Should work while active
        assert!(
            rbac.check_permission("temp", PermissionType::Read, &ResourceType::Server)
                .is_ok()
        );

        // Deactivate
        let mut user = rbac.get_user("temp").unwrap();
        user.deactivate();
        rbac.update_user(user).unwrap();

        // Should fail while inactive
        assert!(matches!(
            rbac.check_permission("temp", PermissionType::Read, &ResourceType::Server),
            Err(RBACError::UserInactive(_))
        ));
    }

    #[test]
    fn test_simple_pattern_match() {
        assert!(simple_pattern_match("*.txt", "file.txt"));
        assert!(simple_pattern_match("*.txt", "document.txt"));
        assert!(!simple_pattern_match("*.txt", "file.doc"));
        assert!(simple_pattern_match("file?", "file1"));
        assert!(simple_pattern_match("file?", "filea"));
        assert!(!simple_pattern_match("file?", "file12"));
        assert!(simple_pattern_match("*", "anything"));
        assert!(simple_pattern_match("exact", "exact"));
    }

    #[test]
    fn test_get_user_permissions() {
        let rbac = RBACManager::new();

        // Create a user with both direct permissions and role-based permissions
        let mut user = User::new("combo", "combo_user");
        user.assign_role("readonly");
        user.add_permission(Permission::new(
            PermissionType::Write,
            ResourceType::table("special", "table"),
        ));
        rbac.create_user(user).unwrap();

        let perms = rbac.get_user_permissions("combo").unwrap();

        // Should have the direct write permission
        assert!(
            perms
                .iter()
                .any(|p| p.permission_type == PermissionType::Write
                    && p.resource == ResourceType::table("special", "table"))
        );

        // Should have read permission from readonly role
        assert!(
            perms
                .iter()
                .any(|p| p.permission_type == PermissionType::Read)
        );
    }
}
