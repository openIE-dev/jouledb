//! Project scaffolding.
//!
//! Template definitions with placeholders, variable substitution,
//! directory structure creation plans, file generation from templates,
//! interactive prompts, and template validation. Pure Rust — no
//! filesystem or template engine dependencies.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from scaffold operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaffoldError {
    /// Missing required variable.
    MissingVariable(String),
    /// Template validation failed.
    ValidationError(String),
    /// Duplicate file path in plan.
    DuplicatePath(String),
    /// Invalid template syntax.
    TemplateSyntaxError { position: usize, message: String },
    /// Prompt error.
    PromptError(String),
}

impl fmt::Display for ScaffoldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingVariable(name) => write!(f, "missing variable: {name}"),
            Self::ValidationError(msg) => write!(f, "validation error: {msg}"),
            Self::DuplicatePath(path) => write!(f, "duplicate path: {path}"),
            Self::TemplateSyntaxError { position, message } => {
                write!(f, "template syntax error at pos {position}: {message}")
            }
            Self::PromptError(msg) => write!(f, "prompt error: {msg}"),
        }
    }
}

impl std::error::Error for ScaffoldError {}

// ── Variable Definition ─────────────────────────────────────────

/// A variable used in templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDef {
    /// Variable name (used in {{name}} placeholders).
    pub name: String,
    /// Description shown in interactive prompts.
    pub description: String,
    /// Default value.
    pub default: Option<String>,
    /// Whether the variable is required.
    pub required: bool,
    /// Validation: allowed values (empty = any).
    pub choices: Vec<String>,
    /// Transform to apply after collection.
    pub transform: VariableTransform,
}

/// Transform to apply to a variable value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableTransform {
    /// No transformation.
    None,
    /// Convert to lowercase.
    Lowercase,
    /// Convert to uppercase.
    Uppercase,
    /// Convert to snake_case (from space/kebab/camel).
    SnakeCase,
    /// Convert to kebab-case.
    KebabCase,
    /// Convert to PascalCase.
    PascalCase,
}

impl VariableDef {
    /// Create a required variable.
    pub fn required(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            default: None,
            required: true,
            choices: Vec::new(),
            transform: VariableTransform::None,
        }
    }

    /// Create an optional variable with a default.
    pub fn optional(name: &str, description: &str, default: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            default: Some(default.to_string()),
            required: false,
            choices: Vec::new(),
            transform: VariableTransform::None,
        }
    }

    /// Set allowed choices.
    pub fn with_choices(mut self, choices: Vec<String>) -> Self {
        self.choices = choices;
        self
    }

    /// Set transform.
    pub fn with_transform(mut self, transform: VariableTransform) -> Self {
        self.transform = transform;
        self
    }
}

// ── Variable Transforms ─────────────────────────────────────────

/// Apply a variable transform to a string value.
pub fn apply_transform(value: &str, transform: VariableTransform) -> String {
    match transform {
        VariableTransform::None => value.to_string(),
        VariableTransform::Lowercase => value.to_lowercase(),
        VariableTransform::Uppercase => value.to_uppercase(),
        VariableTransform::SnakeCase => to_snake_case(value),
        VariableTransform::KebabCase => to_kebab_case(value),
        VariableTransform::PascalCase => to_pascal_case(value),
    }
}

/// Convert a string to snake_case.
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let mut prev_was_upper = false;
    let mut prev_was_sep = true;

    for ch in s.chars() {
        if ch == '-' || ch == ' ' || ch == '_' {
            if !result.is_empty() && !result.ends_with('_') {
                result.push('_');
            }
            prev_was_sep = true;
            prev_was_upper = false;
        } else if ch.is_uppercase() {
            if !prev_was_upper && !prev_was_sep && !result.is_empty() {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
            prev_was_upper = true;
            prev_was_sep = false;
        } else {
            result.push(ch);
            prev_was_upper = false;
            prev_was_sep = false;
        }
    }

    result
}

/// Convert a string to kebab-case.
pub fn to_kebab_case(s: &str) -> String {
    to_snake_case(s).replace('_', "-")
}

/// Convert a string to PascalCase.
pub fn to_pascal_case(s: &str) -> String {
    let snake = to_snake_case(s);
    snake
        .split('_')
        .filter(|p| !p.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    format!("{upper}{}", chars.as_str())
                }
                None => String::new(),
            }
        })
        .collect()
}

// ── Template Substitution ───────────────────────────────────────

/// Substitute {{variable}} placeholders in a template string.
pub fn substitute(template: &str, vars: &HashMap<String, String>) -> Result<String, ScaffoldError> {
    let mut result = String::with_capacity(template.len());
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '{' && chars[i + 1] == '{' {
            // Find closing }}.
            let start = i + 2;
            let mut end = start;
            while end + 1 < chars.len() {
                if chars[end] == '}' && chars[end + 1] == '}' {
                    break;
                }
                end += 1;
            }
            if end + 1 >= chars.len() && !(chars[end] == '}' && end + 1 < chars.len() && chars[end + 1] == '}') {
                return Err(ScaffoldError::TemplateSyntaxError {
                    position: i,
                    message: "unclosed {{ placeholder".to_string(),
                });
            }

            let var_expr: String = chars[start..end].iter().collect();
            let var_name = var_expr.trim();

            // Support transforms: {{name|snake_case}}
            let (name, transform) = if let Some(pipe) = var_name.find('|') {
                let n = var_name[..pipe].trim();
                let t = var_name[pipe + 1..].trim();
                let tf = match t {
                    "lower" | "lowercase" => VariableTransform::Lowercase,
                    "upper" | "uppercase" => VariableTransform::Uppercase,
                    "snake" | "snake_case" => VariableTransform::SnakeCase,
                    "kebab" | "kebab_case" | "kebab-case" => VariableTransform::KebabCase,
                    "pascal" | "pascal_case" | "PascalCase" => VariableTransform::PascalCase,
                    _ => VariableTransform::None,
                };
                (n, tf)
            } else {
                (var_name, VariableTransform::None)
            };

            let value = vars
                .get(name)
                .ok_or_else(|| ScaffoldError::MissingVariable(name.to_string()))?;

            result.push_str(&apply_transform(value, transform));
            i = end + 2;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    Ok(result)
}

/// Extract all variable names referenced in a template string.
pub fn extract_variables(template: &str) -> HashSet<String> {
    let mut vars = HashSet::new();
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '{' && chars[i + 1] == '{' {
            let start = i + 2;
            let mut end = start;
            while end + 1 < chars.len() {
                if chars[end] == '}' && chars[end + 1] == '}' {
                    break;
                }
                end += 1;
            }
            let var_expr: String = chars[start..end].iter().collect();
            let var_name = var_expr.trim();
            // Strip transform.
            let name = if let Some(pipe) = var_name.find('|') {
                var_name[..pipe].trim()
            } else {
                var_name
            };
            if !name.is_empty() {
                vars.insert(name.to_string());
            }
            i = end + 2;
        } else {
            i += 1;
        }
    }

    vars
}

// ── File Template ───────────────────────────────────────────────

/// A file template to be generated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTemplate {
    /// Path template (may contain {{variables}}).
    pub path: String,
    /// Content template.
    pub content: String,
    /// Whether this file is optional (only generated if a condition is met).
    pub conditional: Option<String>,
    /// File permissions hint (e.g., "executable").
    pub permissions: Option<String>,
}

impl FileTemplate {
    /// Create a new file template.
    pub fn new(path: &str, content: &str) -> Self {
        Self {
            path: path.to_string(),
            content: content.to_string(),
            conditional: None,
            permissions: None,
        }
    }

    /// Make this file conditional on a variable being set.
    pub fn when(mut self, condition: &str) -> Self {
        self.conditional = Some(condition.to_string());
        self
    }

    /// Set permissions hint.
    pub fn executable(mut self) -> Self {
        self.permissions = Some("executable".to_string());
        self
    }

    /// Get all variables referenced in path and content.
    pub fn referenced_variables(&self) -> HashSet<String> {
        let mut vars = extract_variables(&self.path);
        vars.extend(extract_variables(&self.content));
        vars
    }
}

// ── Directory Entry ─────────────────────────────────────────────

/// An entry in the directory structure plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirEntry {
    /// A directory.
    Dir {
        name: String,
        children: Vec<DirEntry>,
    },
    /// A file.
    File {
        name: String,
        template_idx: usize,
    },
}

impl DirEntry {
    /// Create a directory entry.
    pub fn dir(name: &str, children: Vec<DirEntry>) -> Self {
        Self::Dir {
            name: name.to_string(),
            children,
        }
    }

    /// Create a file entry.
    pub fn file(name: &str, template_idx: usize) -> Self {
        Self::File {
            name: name.to_string(),
            template_idx,
        }
    }
}

// ── Generated File ──────────────────────────────────────────────

/// A fully generated file (after substitution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedFile {
    pub path: String,
    pub content: String,
    pub permissions: Option<String>,
}

// ── Creation Plan ───────────────────────────────────────────────

/// The plan for creating a project structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreationPlan {
    /// Root directory name.
    pub root: String,
    /// List of directories to create (relative paths).
    pub directories: Vec<String>,
    /// List of files to generate.
    pub files: Vec<GeneratedFile>,
}

impl CreationPlan {
    /// Total number of files.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Total number of directories.
    pub fn dir_count(&self) -> usize {
        self.directories.len()
    }

    /// Display the plan as a tree.
    pub fn tree_display(&self) -> String {
        let mut out = format!("{}/\n", self.root);
        for dir in &self.directories {
            out.push_str(&format!("  {dir}/\n"));
        }
        for file in &self.files {
            out.push_str(&format!("  {}\n", file.path));
        }
        out
    }
}

// ── Prompt Definition ───────────────────────────────────────────

/// A prompt for collecting user input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptDef {
    pub variable: String,
    pub message: String,
    pub default: Option<String>,
    pub choices: Vec<String>,
    pub required: bool,
}

impl PromptDef {
    /// Create from a variable definition.
    pub fn from_var(var: &VariableDef) -> Self {
        Self {
            variable: var.name.clone(),
            message: var.description.clone(),
            default: var.default.clone(),
            choices: var.choices.clone(),
            required: var.required,
        }
    }

    /// Render the prompt text.
    pub fn render(&self) -> String {
        let mut text = self.message.clone();
        if !self.choices.is_empty() {
            text.push_str(&format!(" [{}]", self.choices.join("/")));
        }
        if let Some(def) = &self.default {
            text.push_str(&format!(" (default: {def})"));
        }
        text.push_str(": ");
        text
    }

    /// Validate an answer.
    pub fn validate(&self, answer: &str) -> Result<String, ScaffoldError> {
        let value = if answer.is_empty() {
            if let Some(def) = &self.default {
                def.clone()
            } else if self.required {
                return Err(ScaffoldError::PromptError(format!(
                    "variable '{}' is required",
                    self.variable
                )));
            } else {
                String::new()
            }
        } else {
            answer.to_string()
        };

        if !self.choices.is_empty() && !value.is_empty() && !self.choices.contains(&value) {
            return Err(ScaffoldError::PromptError(format!(
                "invalid choice '{}', expected one of: {}",
                value,
                self.choices.join(", ")
            )));
        }

        Ok(value)
    }
}

// ── Project Template ────────────────────────────────────────────

/// A complete project template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTemplate {
    pub name: String,
    pub description: String,
    pub variables: Vec<VariableDef>,
    pub files: Vec<FileTemplate>,
}

impl ProjectTemplate {
    /// Create a new project template.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            variables: Vec::new(),
            files: Vec::new(),
        }
    }

    /// Add a variable.
    pub fn var(mut self, var: VariableDef) -> Self {
        self.variables.push(var);
        self
    }

    /// Add a file template.
    pub fn file(mut self, file: FileTemplate) -> Self {
        self.files.push(file);
        self
    }

    /// Get all prompts needed for this template.
    pub fn prompts(&self) -> Vec<PromptDef> {
        self.variables.iter().map(|v| PromptDef::from_var(v)).collect()
    }

    /// Validate the template: check that all referenced variables are defined.
    pub fn validate(&self) -> Result<(), ScaffoldError> {
        let defined: HashSet<String> = self.variables.iter().map(|v| v.name.clone()).collect();

        for file in &self.files {
            let referenced = file.referenced_variables();
            for var in &referenced {
                if !defined.contains(var) {
                    return Err(ScaffoldError::ValidationError(format!(
                        "file '{}' references undefined variable '{var}'",
                        file.path
                    )));
                }
            }
        }

        // Check for duplicate paths.
        let mut paths = HashSet::new();
        for file in &self.files {
            if !paths.insert(&file.path) {
                return Err(ScaffoldError::DuplicatePath(file.path.clone()));
            }
        }

        Ok(())
    }

    /// Generate the creation plan with the given variables.
    pub fn generate(&self, vars: &HashMap<String, String>) -> Result<CreationPlan, ScaffoldError> {
        // Validate required variables.
        for var_def in &self.variables {
            if var_def.required && !vars.contains_key(&var_def.name) && var_def.default.is_none() {
                return Err(ScaffoldError::MissingVariable(var_def.name.clone()));
            }
        }

        // Build effective vars with defaults and transforms.
        let mut effective = HashMap::new();
        for var_def in &self.variables {
            let raw = vars
                .get(&var_def.name)
                .or(var_def.default.as_ref())
                .cloned()
                .unwrap_or_default();
            let transformed = apply_transform(&raw, var_def.transform);
            effective.insert(var_def.name.clone(), transformed);
        }
        // Also include any extra vars not in definitions.
        for (k, v) in vars {
            effective.entry(k.clone()).or_insert_with(|| v.clone());
        }

        let root = effective
            .get("project_name")
            .cloned()
            .unwrap_or_else(|| "project".to_string());

        let mut directories = HashSet::new();
        let mut files = Vec::new();

        for file_tmpl in &self.files {
            // Check conditional.
            if let Some(cond) = &file_tmpl.conditional {
                let cond_val = effective.get(cond).map(|s| s.as_str()).unwrap_or("");
                if cond_val.is_empty() || cond_val == "false" || cond_val == "no" {
                    continue;
                }
            }

            let path = substitute(&file_tmpl.path, &effective)?;
            let content = substitute(&file_tmpl.content, &effective)?;

            // Extract parent directories.
            if let Some(parent) = path.rsplit_once('/') {
                directories.insert(parent.0.to_string());
            }

            files.push(GeneratedFile {
                path,
                content,
                permissions: file_tmpl.permissions.clone(),
            });
        }

        let mut dir_list: Vec<String> = directories.into_iter().collect();
        dir_list.sort();

        Ok(CreationPlan {
            root,
            directories: dir_list,
            files,
        })
    }
}

// ── Built-in Templates ──────────────────────────────────────────

/// Create a basic Rust project template.
pub fn rust_project_template() -> ProjectTemplate {
    ProjectTemplate::new("rust-project", "A basic Rust project scaffold")
        .var(VariableDef::required("project_name", "Project name"))
        .var(VariableDef::optional("author", "Author name", ""))
        .var(VariableDef::optional("license", "License", "MIT").with_choices(vec![
            "MIT".to_string(),
            "Apache-2.0".to_string(),
            "GPL-3.0".to_string(),
        ]))
        .file(FileTemplate::new(
            "Cargo.toml",
            "[package]\nname = \"{{project_name}}\"\nversion = \"0.1.0\"\nedition = \"2024\"\nauthors = [\"{{author}}\"]\nlicense = \"{{license}}\"\n",
        ))
        .file(FileTemplate::new(
            "src/main.rs",
            "fn main() {\n    println!(\"Hello from {{project_name}}!\");\n}\n",
        ))
        .file(FileTemplate::new(
            "src/lib.rs",
            "//! {{project_name}} library.\n\npub fn hello() -> &'static str {\n    \"Hello from {{project_name}}\"\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn test_hello() {\n        assert!(hello().contains(\"{{project_name}}\"));\n    }\n}\n",
        ))
        .file(FileTemplate::new(".gitignore", "/target\n"))
}

/// Create a basic web project template.
pub fn web_project_template() -> ProjectTemplate {
    ProjectTemplate::new("web-project", "A basic web project scaffold")
        .var(VariableDef::required("project_name", "Project name"))
        .var(VariableDef::optional("port", "Server port", "8080"))
        .file(FileTemplate::new(
            "index.html",
            "<!DOCTYPE html>\n<html>\n<head>\n  <title>{{project_name}}</title>\n</head>\n<body>\n  <h1>Welcome to {{project_name}}</h1>\n</body>\n</html>\n",
        ))
        .file(FileTemplate::new(
            "config.json",
            "{\n  \"name\": \"{{project_name}}\",\n  \"port\": {{port}}\n}\n",
        ))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_basic() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "hello".to_string());
        let result = substitute("Hello, {{name}}!", &vars).unwrap();
        assert_eq!(result, "Hello, hello!");
    }

    #[test]
    fn substitute_multiple() {
        let mut vars = HashMap::new();
        vars.insert("first".to_string(), "John".to_string());
        vars.insert("last".to_string(), "Doe".to_string());
        let result = substitute("{{first}} {{last}}", &vars).unwrap();
        assert_eq!(result, "John Doe");
    }

    #[test]
    fn substitute_with_transform() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "MyProject".to_string());
        let result = substitute("{{name|snake_case}}", &vars).unwrap();
        assert_eq!(result, "my_project");
    }

    #[test]
    fn substitute_missing_variable() {
        let vars = HashMap::new();
        let err = substitute("{{missing}}", &vars).unwrap_err();
        assert!(matches!(err, ScaffoldError::MissingVariable(_)));
    }

    #[test]
    fn extract_variables_basic() {
        let vars = extract_variables("Hello {{name}}, welcome to {{project}}!");
        assert!(vars.contains("name"));
        assert!(vars.contains("project"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn extract_variables_with_transforms() {
        let vars = extract_variables("{{name|snake_case}}");
        assert!(vars.contains("name"));
    }

    #[test]
    fn to_snake_case_from_camel() {
        assert_eq!(to_snake_case("MyProjectName"), "my_project_name");
        assert_eq!(to_snake_case("helloWorld"), "hello_world");
    }

    #[test]
    fn to_snake_case_from_kebab() {
        assert_eq!(to_snake_case("my-project-name"), "my_project_name");
    }

    #[test]
    fn to_snake_case_from_spaces() {
        assert_eq!(to_snake_case("My Project Name"), "my_project_name");
    }

    #[test]
    fn to_kebab_case_basic() {
        assert_eq!(to_kebab_case("MyProject"), "my-project");
        assert_eq!(to_kebab_case("hello_world"), "hello-world");
    }

    #[test]
    fn to_pascal_case_basic() {
        assert_eq!(to_pascal_case("my_project"), "MyProject");
        assert_eq!(to_pascal_case("hello-world"), "HelloWorld");
    }

    #[test]
    fn template_validation_ok() {
        let tmpl = ProjectTemplate::new("test", "A test")
            .var(VariableDef::required("name", "Name"))
            .file(FileTemplate::new("{{name}}.rs", "// {{name}}"));
        assert!(tmpl.validate().is_ok());
    }

    #[test]
    fn template_validation_undefined_var() {
        let tmpl = ProjectTemplate::new("test", "A test")
            .file(FileTemplate::new("{{undefined}}.rs", "content"));
        let err = tmpl.validate().unwrap_err();
        assert!(matches!(err, ScaffoldError::ValidationError(_)));
    }

    #[test]
    fn template_validation_duplicate_path() {
        let tmpl = ProjectTemplate::new("test", "A test")
            .file(FileTemplate::new("same.rs", "content1"))
            .file(FileTemplate::new("same.rs", "content2"));
        let err = tmpl.validate().unwrap_err();
        assert!(matches!(err, ScaffoldError::DuplicatePath(_)));
    }

    #[test]
    fn generate_plan() {
        let tmpl = rust_project_template();
        let mut vars = HashMap::new();
        vars.insert("project_name".to_string(), "my_app".to_string());
        vars.insert("author".to_string(), "Test Author".to_string());
        vars.insert("license".to_string(), "MIT".to_string());
        let plan = tmpl.generate(&vars).unwrap();
        assert_eq!(plan.root, "my_app");
        assert!(plan.files.len() >= 3);
        // Check Cargo.toml content.
        let cargo = plan.files.iter().find(|f| f.path == "Cargo.toml").unwrap();
        assert!(cargo.content.contains("my_app"));
        assert!(cargo.content.contains("MIT"));
    }

    #[test]
    fn generate_missing_required() {
        let tmpl = ProjectTemplate::new("test", "Test")
            .var(VariableDef::required("name", "Name"))
            .file(FileTemplate::new("{{name}}.rs", ""));
        let vars = HashMap::new();
        let err = tmpl.generate(&vars).unwrap_err();
        assert!(matches!(err, ScaffoldError::MissingVariable(_)));
    }

    #[test]
    fn conditional_file_excluded() {
        let tmpl = ProjectTemplate::new("test", "Test")
            .var(VariableDef::optional("name", "Name", "x"))
            .var(VariableDef::optional("with_tests", "Include tests?", "false"))
            .file(FileTemplate::new("src/main.rs", "fn main() {}"))
            .file(FileTemplate::new("tests/test.rs", "#[test] fn it_works() {}").when("with_tests"));
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "proj".to_string());
        vars.insert("with_tests".to_string(), "false".to_string());
        let plan = tmpl.generate(&vars).unwrap();
        assert_eq!(plan.file_count(), 1); // only main.rs
    }

    #[test]
    fn conditional_file_included() {
        let tmpl = ProjectTemplate::new("test", "Test")
            .var(VariableDef::optional("name", "Name", "x"))
            .var(VariableDef::optional("with_tests", "Include tests?", "true"))
            .file(FileTemplate::new("src/main.rs", "fn main() {}"))
            .file(FileTemplate::new("tests/test.rs", "#[test] fn it_works() {}").when("with_tests"));
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "proj".to_string());
        vars.insert("with_tests".to_string(), "true".to_string());
        let plan = tmpl.generate(&vars).unwrap();
        assert_eq!(plan.file_count(), 2);
    }

    #[test]
    fn prompt_render() {
        let prompt = PromptDef {
            variable: "name".to_string(),
            message: "Project name".to_string(),
            default: Some("my-app".to_string()),
            choices: Vec::new(),
            required: true,
        };
        let text = prompt.render();
        assert!(text.contains("Project name"));
        assert!(text.contains("my-app"));
    }

    #[test]
    fn prompt_validate_required() {
        let prompt = PromptDef {
            variable: "name".to_string(),
            message: "Name".to_string(),
            default: None,
            choices: Vec::new(),
            required: true,
        };
        let err = prompt.validate("").unwrap_err();
        assert!(matches!(err, ScaffoldError::PromptError(_)));
    }

    #[test]
    fn prompt_validate_choices() {
        let prompt = PromptDef {
            variable: "license".to_string(),
            message: "License".to_string(),
            default: None,
            choices: vec!["MIT".to_string(), "Apache-2.0".to_string()],
            required: true,
        };
        assert!(prompt.validate("MIT").is_ok());
        assert!(prompt.validate("GPL").is_err());
    }

    #[test]
    fn prompt_default_applied() {
        let prompt = PromptDef {
            variable: "x".to_string(),
            message: "X".to_string(),
            default: Some("default_val".to_string()),
            choices: Vec::new(),
            required: false,
        };
        let result = prompt.validate("").unwrap();
        assert_eq!(result, "default_val");
    }

    #[test]
    fn creation_plan_tree() {
        let plan = CreationPlan {
            root: "my-project".to_string(),
            directories: vec!["src".to_string()],
            files: vec![GeneratedFile {
                path: "src/main.rs".to_string(),
                content: "fn main() {}".to_string(),
                permissions: None,
            }],
        };
        let tree = plan.tree_display();
        assert!(tree.contains("my-project/"));
        assert!(tree.contains("src/"));
        assert!(tree.contains("src/main.rs"));
    }

    #[test]
    fn variable_transform_applied_in_generate() {
        let tmpl = ProjectTemplate::new("test", "Test")
            .var(
                VariableDef::required("project_name", "Name")
                    .with_transform(VariableTransform::SnakeCase),
            )
            .file(FileTemplate::new("{{project_name}}.rs", "// {{project_name}}"));
        let mut vars = HashMap::new();
        vars.insert("project_name".to_string(), "MyProject".to_string());
        let plan = tmpl.generate(&vars).unwrap();
        assert_eq!(plan.files[0].path, "my_project.rs");
    }

    #[test]
    fn file_template_executable() {
        let ft = FileTemplate::new("script.sh", "#!/bin/sh").executable();
        assert_eq!(ft.permissions, Some("executable".to_string()));
    }

    #[test]
    fn web_template_works() {
        let tmpl = web_project_template();
        let mut vars = HashMap::new();
        vars.insert("project_name".to_string(), "my-site".to_string());
        vars.insert("port".to_string(), "3000".to_string());
        let plan = tmpl.generate(&vars).unwrap();
        let html = plan.files.iter().find(|f| f.path == "index.html").unwrap();
        assert!(html.content.contains("my-site"));
    }

    #[test]
    fn error_display() {
        let e = ScaffoldError::MissingVariable("name".to_string());
        assert!(format!("{e}").contains("name"));
    }

    #[test]
    fn dir_entry_creation() {
        let tree = DirEntry::dir(
            "src",
            vec![DirEntry::file("main.rs", 0), DirEntry::file("lib.rs", 1)],
        );
        if let DirEntry::Dir { name, children } = &tree {
            assert_eq!(name, "src");
            assert_eq!(children.len(), 2);
        } else {
            panic!("expected Dir");
        }
    }
}
