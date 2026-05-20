//! Code generation framework — template-based codegen, AST builder, import management,
//! formatting, output buffering, and file generation plan.
//!
//! Replaces JS codegen tools (Hygen, Plop, ts-morph, Yeoman) with a pure-Rust
//! code generation engine that tracks every emitted byte with energy awareness.

use std::collections::{BTreeMap, BTreeSet, HashMap};

// ── Errors ──────────────────────────────────────────────────────

/// Code generation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeGenError {
    /// Template variable not found.
    MissingVariable(String),
    /// Duplicate import path.
    DuplicateImport(String),
    /// Invalid AST node construction.
    InvalidNode(String),
    /// File already exists in generation plan.
    FileExists(String),
    /// Empty template body.
    EmptyTemplate,
    /// Unclosed template block.
    UnclosedBlock(String),
}

impl std::fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVariable(v) => write!(f, "missing template variable: {v}"),
            Self::DuplicateImport(p) => write!(f, "duplicate import: {p}"),
            Self::InvalidNode(msg) => write!(f, "invalid AST node: {msg}"),
            Self::FileExists(p) => write!(f, "file already in plan: {p}"),
            Self::EmptyTemplate => write!(f, "empty template body"),
            Self::UnclosedBlock(tag) => write!(f, "unclosed block: {tag}"),
        }
    }
}

// ── AST Nodes ───────────────────────────────────────────────────

/// Visibility of a declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Visibility {
    Private,
    Public,
    Crate,
}

impl Visibility {
    fn prefix(&self) -> &str {
        match self {
            Self::Private => "",
            Self::Public => "pub ",
            Self::Crate => "pub(crate) ",
        }
    }
}

/// A single function parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub type_name: String,
}

/// An AST node representing a code construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstNode {
    /// A struct with named fields.
    Struct {
        name: String,
        visibility: Visibility,
        fields: Vec<(Visibility, String, String)>,
        derives: Vec<String>,
        doc: Option<String>,
    },
    /// A function definition.
    Function {
        name: String,
        visibility: Visibility,
        params: Vec<Param>,
        return_type: Option<String>,
        body: String,
        doc: Option<String>,
        is_async: bool,
    },
    /// An enum with variants.
    Enum {
        name: String,
        visibility: Visibility,
        variants: Vec<(String, Option<String>)>,
        derives: Vec<String>,
        doc: Option<String>,
    },
    /// An impl block.
    Impl {
        target: String,
        trait_name: Option<String>,
        methods: Vec<AstNode>,
    },
    /// A constant declaration.
    Const {
        name: String,
        visibility: Visibility,
        type_name: String,
        value: String,
    },
    /// A type alias.
    TypeAlias {
        name: String,
        visibility: Visibility,
        target: String,
    },
    /// Raw code block (escape hatch).
    Raw(String),
}

// ── AST Builder ─────────────────────────────────────────────────

/// Fluent builder for constructing AST nodes.
#[derive(Debug, Clone)]
pub struct AstBuilder {
    nodes: Vec<AstNode>,
}

impl AstBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_struct(
        &mut self,
        name: impl Into<String>,
        visibility: Visibility,
    ) -> StructBuilder<'_> {
        StructBuilder {
            parent: self,
            name: name.into(),
            visibility,
            fields: Vec::new(),
            derives: Vec::new(),
            doc: None,
        }
    }

    pub fn add_function(
        &mut self,
        name: impl Into<String>,
        visibility: Visibility,
    ) -> FunctionBuilder<'_> {
        FunctionBuilder {
            parent: self,
            name: name.into(),
            visibility,
            params: Vec::new(),
            return_type: None,
            body: String::new(),
            doc: None,
            is_async: false,
        }
    }

    pub fn add_enum(
        &mut self,
        name: impl Into<String>,
        visibility: Visibility,
    ) -> EnumBuilder<'_> {
        EnumBuilder {
            parent: self,
            name: name.into(),
            visibility,
            variants: Vec::new(),
            derives: Vec::new(),
            doc: None,
        }
    }

    pub fn add_raw(&mut self, code: impl Into<String>) {
        self.nodes.push(AstNode::Raw(code.into()));
    }

    pub fn add_const(
        &mut self,
        name: impl Into<String>,
        vis: Visibility,
        type_name: impl Into<String>,
        value: impl Into<String>,
    ) {
        self.nodes.push(AstNode::Const {
            name: name.into(),
            visibility: vis,
            type_name: type_name.into(),
            value: value.into(),
        });
    }

    pub fn add_type_alias(
        &mut self,
        name: impl Into<String>,
        vis: Visibility,
        target: impl Into<String>,
    ) {
        self.nodes.push(AstNode::TypeAlias {
            name: name.into(),
            visibility: vis,
            target: target.into(),
        });
    }

    pub fn nodes(&self) -> &[AstNode] {
        &self.nodes
    }

    pub fn into_nodes(self) -> Vec<AstNode> {
        self.nodes
    }
}

impl Default for AstBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for struct nodes.
pub struct StructBuilder<'a> {
    parent: &'a mut AstBuilder,
    name: String,
    visibility: Visibility,
    fields: Vec<(Visibility, String, String)>,
    derives: Vec<String>,
    doc: Option<String>,
}

impl<'a> StructBuilder<'a> {
    pub fn field(
        mut self,
        vis: Visibility,
        name: impl Into<String>,
        ty: impl Into<String>,
    ) -> Self {
        self.fields.push((vis, name.into(), ty.into()));
        self
    }

    pub fn derive(mut self, d: impl Into<String>) -> Self {
        self.derives.push(d.into());
        self
    }

    pub fn doc(mut self, d: impl Into<String>) -> Self {
        self.doc = Some(d.into());
        self
    }

    pub fn build(self) {
        self.parent.nodes.push(AstNode::Struct {
            name: self.name,
            visibility: self.visibility,
            fields: self.fields,
            derives: self.derives,
            doc: self.doc,
        });
    }
}

/// Builder for function nodes.
pub struct FunctionBuilder<'a> {
    parent: &'a mut AstBuilder,
    name: String,
    visibility: Visibility,
    params: Vec<Param>,
    return_type: Option<String>,
    body: String,
    doc: Option<String>,
    is_async: bool,
}

impl<'a> FunctionBuilder<'a> {
    pub fn param(mut self, name: impl Into<String>, ty: impl Into<String>) -> Self {
        self.params.push(Param { name: name.into(), type_name: ty.into() });
        self
    }

    pub fn returns(mut self, ty: impl Into<String>) -> Self {
        self.return_type = Some(ty.into());
        self
    }

    pub fn body(mut self, b: impl Into<String>) -> Self {
        self.body = b.into();
        self
    }

    pub fn doc(mut self, d: impl Into<String>) -> Self {
        self.doc = Some(d.into());
        self
    }

    pub fn set_async(mut self, a: bool) -> Self {
        self.is_async = a;
        self
    }

    pub fn build(self) {
        self.parent.nodes.push(AstNode::Function {
            name: self.name,
            visibility: self.visibility,
            params: self.params,
            return_type: self.return_type,
            body: self.body,
            doc: self.doc,
            is_async: self.is_async,
        });
    }
}

/// Builder for enum nodes.
pub struct EnumBuilder<'a> {
    parent: &'a mut AstBuilder,
    name: String,
    visibility: Visibility,
    variants: Vec<(String, Option<String>)>,
    derives: Vec<String>,
    doc: Option<String>,
}

impl<'a> EnumBuilder<'a> {
    pub fn variant(mut self, name: impl Into<String>, payload: Option<String>) -> Self {
        self.variants.push((name.into(), payload));
        self
    }

    pub fn derive(mut self, d: impl Into<String>) -> Self {
        self.derives.push(d.into());
        self
    }

    pub fn doc(mut self, d: impl Into<String>) -> Self {
        self.doc = Some(d.into());
        self
    }

    pub fn build(self) {
        self.parent.nodes.push(AstNode::Enum {
            name: self.name,
            visibility: self.visibility,
            variants: self.variants,
            derives: self.derives,
            doc: self.doc,
        });
    }
}

// ── Import Manager ──────────────────────────────────────────────

/// Tracks and deduplicates import statements.
#[derive(Debug, Clone, Default)]
pub struct ImportManager {
    /// module_path -> set of imported names.
    imports: BTreeMap<String, BTreeSet<String>>,
}

impl ImportManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single import: `use module_path::name;`
    pub fn add(&mut self, module_path: impl Into<String>, name: impl Into<String>) {
        self.imports
            .entry(module_path.into())
            .or_default()
            .insert(name.into());
    }

    /// Add a glob import: `use module_path::*;`
    pub fn add_glob(&mut self, module_path: impl Into<String>) {
        self.imports.entry(module_path.into()).or_default().insert("*".to_string());
    }

    /// Render all imports as sorted `use` statements.
    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        for (path, names) in &self.imports {
            if names.len() == 1 {
                let name = names.iter().next().unwrap();
                lines.push(format!("use {path}::{name};"));
            } else {
                let joined: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                lines.push(format!("use {path}::{{{}}};", joined.join(", ")));
            }
        }
        lines.join("\n")
    }

    /// Number of distinct import paths.
    pub fn path_count(&self) -> usize {
        self.imports.len()
    }

    /// Total number of imported symbols.
    pub fn symbol_count(&self) -> usize {
        self.imports.values().map(|s| s.len()).sum()
    }
}

// ── Output Buffer ───────────────────────────────────────────────

/// Indentation-aware output buffer for code generation.
#[derive(Debug, Clone)]
pub struct OutputBuffer {
    lines: Vec<String>,
    indent_level: usize,
    indent_str: String,
}

impl OutputBuffer {
    pub fn new(indent_str: impl Into<String>) -> Self {
        Self { lines: Vec::new(), indent_level: 0, indent_str: indent_str.into() }
    }

    pub fn with_spaces(n: usize) -> Self {
        Self::new(" ".repeat(n))
    }

    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    pub fn dedent(&mut self) {
        self.indent_level = self.indent_level.saturating_sub(1);
    }

    pub fn line(&mut self, text: impl AsRef<str>) {
        let prefix = self.indent_str.repeat(self.indent_level);
        self.lines.push(format!("{prefix}{}", text.as_ref()));
    }

    pub fn blank(&mut self) {
        self.lines.push(String::new());
    }

    pub fn raw(&mut self, text: impl AsRef<str>) {
        self.lines.push(text.as_ref().to_string());
    }

    pub fn finish(self) -> String {
        self.lines.join("\n")
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

// ── Code Renderer ───────────────────────────────────────────────

/// Renders AST nodes and imports into a complete source file.
pub struct CodeRenderer {
    imports: ImportManager,
    nodes: Vec<AstNode>,
    header_comment: Option<String>,
}

impl CodeRenderer {
    pub fn new() -> Self {
        Self { imports: ImportManager::new(), nodes: Vec::new(), header_comment: None }
    }

    pub fn set_header(&mut self, comment: impl Into<String>) {
        self.header_comment = Some(comment.into());
    }

    pub fn imports_mut(&mut self) -> &mut ImportManager {
        &mut self.imports
    }

    pub fn add_node(&mut self, node: AstNode) {
        self.nodes.push(node);
    }

    pub fn add_nodes(&mut self, nodes: Vec<AstNode>) {
        self.nodes.extend(nodes);
    }

    /// Render the complete source file.
    pub fn render(&self) -> String {
        let mut buf = OutputBuffer::with_spaces(4);

        if let Some(hdr) = &self.header_comment {
            for line in hdr.lines() {
                buf.line(format!("// {line}"));
            }
            buf.blank();
        }

        let imports_str = self.imports.render();
        if !imports_str.is_empty() {
            buf.raw(imports_str);
            buf.blank();
        }

        for (i, node) in self.nodes.iter().enumerate() {
            if i > 0 {
                buf.blank();
            }
            render_node(node, &mut buf);
        }

        buf.blank();
        buf.finish()
    }
}

impl Default for CodeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

fn render_node(node: &AstNode, buf: &mut OutputBuffer) {
    match node {
        AstNode::Struct { name, visibility, fields, derives, doc } => {
            if let Some(d) = doc {
                buf.line(format!("/// {d}"));
            }
            if !derives.is_empty() {
                buf.line(format!("#[derive({})]", derives.join(", ")));
            }
            buf.line(format!("{}struct {name} {{", visibility.prefix()));
            buf.indent();
            for (fvis, fname, ftype) in fields {
                buf.line(format!("{}{fname}: {ftype},", fvis.prefix()));
            }
            buf.dedent();
            buf.line("}");
        }
        AstNode::Function { name, visibility, params, return_type, body, doc, is_async } => {
            if let Some(d) = doc {
                buf.line(format!("/// {d}"));
            }
            let async_kw = if *is_async { "async " } else { "" };
            let params_str: Vec<String> =
                params.iter().map(|p| format!("{}: {}", p.name, p.type_name)).collect();
            let ret = return_type.as_ref().map(|r| format!(" -> {r}")).unwrap_or_default();
            buf.line(format!(
                "{}{async_kw}fn {name}({}){ret} {{",
                visibility.prefix(),
                params_str.join(", ")
            ));
            buf.indent();
            for bline in body.lines() {
                buf.line(bline);
            }
            buf.dedent();
            buf.line("}");
        }
        AstNode::Enum { name, visibility, variants, derives, doc } => {
            if let Some(d) = doc {
                buf.line(format!("/// {d}"));
            }
            if !derives.is_empty() {
                buf.line(format!("#[derive({})]", derives.join(", ")));
            }
            buf.line(format!("{}enum {name} {{", visibility.prefix()));
            buf.indent();
            for (vname, payload) in variants {
                if let Some(p) = payload {
                    buf.line(format!("{vname}({p}),"));
                } else {
                    buf.line(format!("{vname},"));
                }
            }
            buf.dedent();
            buf.line("}");
        }
        AstNode::Impl { target, trait_name, methods } => {
            if let Some(tr) = trait_name {
                buf.line(format!("impl {tr} for {target} {{"));
            } else {
                buf.line(format!("impl {target} {{"));
            }
            buf.indent();
            for (i, m) in methods.iter().enumerate() {
                if i > 0 {
                    buf.blank();
                }
                render_node(m, buf);
            }
            buf.dedent();
            buf.line("}");
        }
        AstNode::Const { name, visibility, type_name, value } => {
            buf.line(format!("{}const {name}: {type_name} = {value};", visibility.prefix()));
        }
        AstNode::TypeAlias { name, visibility, target } => {
            buf.line(format!("{}type {name} = {target};", visibility.prefix()));
        }
        AstNode::Raw(code) => {
            for line in code.lines() {
                buf.line(line);
            }
        }
    }
}

// ── Template Engine ─────────────────────────────────────────────

/// Simple template engine with `{{variable}}` substitution and
/// `{{#if var}}...{{/if}}` conditional blocks.
#[derive(Debug, Clone)]
pub struct Template {
    body: String,
}

impl Template {
    pub fn new(body: impl Into<String>) -> Result<Self, CodeGenError> {
        let b = body.into();
        if b.trim().is_empty() {
            return Err(CodeGenError::EmptyTemplate);
        }
        // Check for unclosed blocks.
        let opens = b.matches("{{#if ").count();
        let closes = b.matches("{{/if}}").count();
        if opens != closes {
            return Err(CodeGenError::UnclosedBlock("if".to_string()));
        }
        Ok(Self { body: b })
    }

    /// Render the template with the given variable bindings.
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String, CodeGenError> {
        let mut result = self.body.clone();

        // Process conditional blocks first.
        loop {
            let Some(start) = result.find("{{#if ") else { break };
            let tag_end = result[start..]
                .find("}}")
                .ok_or_else(|| CodeGenError::UnclosedBlock("if".to_string()))?
                + start
                + 2;
            let var_name = result[start + 6..tag_end - 2].trim().to_string();
            let close_tag = "{{/if}}";
            let close_pos = result[tag_end..]
                .find(close_tag)
                .ok_or_else(|| CodeGenError::UnclosedBlock("if".to_string()))?
                + tag_end;

            let inner = &result[tag_end..close_pos];
            let keep = vars.get(&var_name).map(|v| !v.is_empty()).unwrap_or(false);
            let replacement = if keep { inner.to_string() } else { String::new() };
            result = format!("{}{replacement}{}", &result[..start], &result[close_pos + close_tag.len()..]);
        }

        // Then substitute variables.
        for (key, val) in vars {
            let placeholder = format!("{{{{{key}}}}}");
            result = result.replace(&placeholder, val);
        }

        // Check for any remaining placeholders.
        if let Some(pos) = result.find("{{") {
            if let Some(end) = result[pos..].find("}}") {
                let var = &result[pos + 2..pos + end];
                return Err(CodeGenError::MissingVariable(var.to_string()));
            }
        }

        Ok(result)
    }
}

// ── File Generation Plan ────────────────────────────────────────

/// A planned output file in the generation plan.
#[derive(Debug, Clone)]
pub struct PlannedFile {
    pub path: String,
    pub content: String,
    pub overwrite: bool,
}

/// A collection of files to be generated.
#[derive(Debug, Clone, Default)]
pub struct GenerationPlan {
    files: BTreeMap<String, PlannedFile>,
}

impl GenerationPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file to the plan.
    pub fn add_file(
        &mut self,
        path: impl Into<String>,
        content: impl Into<String>,
        overwrite: bool,
    ) -> Result<(), CodeGenError> {
        let p = path.into();
        if self.files.contains_key(&p) && !overwrite {
            return Err(CodeGenError::FileExists(p));
        }
        self.files.insert(
            p.clone(),
            PlannedFile { path: p, content: content.into(), overwrite },
        );
        Ok(())
    }

    /// Get a planned file by path.
    pub fn get_file(&self, path: &str) -> Option<&PlannedFile> {
        self.files.get(path)
    }

    /// Number of files in the plan.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Total bytes across all planned files.
    pub fn total_bytes(&self) -> usize {
        self.files.values().map(|f| f.content.len()).sum()
    }

    /// List all planned file paths (sorted).
    pub fn file_paths(&self) -> Vec<&str> {
        self.files.keys().map(|s| s.as_str()).collect()
    }

    /// Remove a file from the plan.
    pub fn remove_file(&mut self, path: &str) -> bool {
        self.files.remove(path).is_some()
    }

    /// Generate a summary of the plan.
    pub fn summary(&self) -> String {
        let mut buf = String::new();
        buf.push_str(&format!("Generation Plan: {} files\n", self.files.len()));
        for f in self.files.values() {
            let action = if f.overwrite { "overwrite" } else { "create" };
            buf.push_str(&format!(
                "  [{action}] {} ({} bytes)\n",
                f.path,
                f.content.len()
            ));
        }
        buf.push_str(&format!("Total: {} bytes", self.total_bytes()));
        buf
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_simple_substitution() {
        let tmpl = Template::new("Hello, {{name}}!").unwrap();
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());
        let result = tmpl.render(&vars).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_template_multiple_vars() {
        let tmpl = Template::new("{{greeting}}, {{name}}!").unwrap();
        let mut vars = HashMap::new();
        vars.insert("greeting".to_string(), "Hi".to_string());
        vars.insert("name".to_string(), "Alice".to_string());
        let result = tmpl.render(&vars).unwrap();
        assert_eq!(result, "Hi, Alice!");
    }

    #[test]
    fn test_template_missing_variable() {
        let tmpl = Template::new("Hello, {{name}}!").unwrap();
        let vars = HashMap::new();
        let err = tmpl.render(&vars).unwrap_err();
        assert!(matches!(err, CodeGenError::MissingVariable(_)));
    }

    #[test]
    fn test_template_empty_body() {
        let err = Template::new("   ").unwrap_err();
        assert!(matches!(err, CodeGenError::EmptyTemplate));
    }

    #[test]
    fn test_template_conditional_true() {
        let tmpl = Template::new("start{{#if show}}VISIBLE{{/if}}end").unwrap();
        let mut vars = HashMap::new();
        vars.insert("show".to_string(), "yes".to_string());
        let result = tmpl.render(&vars).unwrap();
        assert_eq!(result, "startVISIBLEend");
    }

    #[test]
    fn test_template_conditional_false() {
        let tmpl = Template::new("start{{#if show}}HIDDEN{{/if}}end").unwrap();
        let vars = HashMap::new();
        let result = tmpl.render(&vars).unwrap();
        assert_eq!(result, "startend");
    }

    #[test]
    fn test_template_conditional_empty_value() {
        let tmpl = Template::new("start{{#if show}}HIDDEN{{/if}}end").unwrap();
        let mut vars = HashMap::new();
        vars.insert("show".to_string(), String::new());
        let result = tmpl.render(&vars).unwrap();
        assert_eq!(result, "startend");
    }

    #[test]
    fn test_template_unclosed_block() {
        let err = Template::new("{{#if foo}}bar").unwrap_err();
        assert!(matches!(err, CodeGenError::UnclosedBlock(_)));
    }

    #[test]
    fn test_import_manager_single() {
        let mut mgr = ImportManager::new();
        mgr.add("std::collections", "HashMap");
        let output = mgr.render();
        assert_eq!(output, "use std::collections::HashMap;");
        assert_eq!(mgr.path_count(), 1);
        assert_eq!(mgr.symbol_count(), 1);
    }

    #[test]
    fn test_import_manager_multiple_from_same_module() {
        let mut mgr = ImportManager::new();
        mgr.add("std::collections", "HashMap");
        mgr.add("std::collections", "BTreeMap");
        let output = mgr.render();
        assert!(output.contains("BTreeMap"));
        assert!(output.contains("HashMap"));
        assert_eq!(mgr.path_count(), 1);
        assert_eq!(mgr.symbol_count(), 2);
    }

    #[test]
    fn test_import_manager_dedup() {
        let mut mgr = ImportManager::new();
        mgr.add("std::fmt", "Display");
        mgr.add("std::fmt", "Display");
        assert_eq!(mgr.symbol_count(), 1);
    }

    #[test]
    fn test_import_manager_glob() {
        let mut mgr = ImportManager::new();
        mgr.add_glob("crate::prelude");
        let output = mgr.render();
        assert_eq!(output, "use crate::prelude::*;");
    }

    #[test]
    fn test_import_manager_sorted() {
        let mut mgr = ImportManager::new();
        mgr.add("z_mod", "Z");
        mgr.add("a_mod", "A");
        let output = mgr.render();
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines[0].contains("a_mod"));
        assert!(lines[1].contains("z_mod"));
    }

    #[test]
    fn test_output_buffer_basic() {
        let mut buf = OutputBuffer::with_spaces(4);
        buf.line("fn main() {");
        buf.indent();
        buf.line("println!(\"hello\");");
        buf.dedent();
        buf.line("}");
        let output = buf.finish();
        assert!(output.contains("    println!(\"hello\");"));
        assert!(output.contains("fn main() {"));
    }

    #[test]
    fn test_output_buffer_blank_line() {
        let mut buf = OutputBuffer::with_spaces(2);
        buf.line("a");
        buf.blank();
        buf.line("b");
        let output = buf.finish();
        assert_eq!(output, "a\n\nb");
    }

    #[test]
    fn test_output_buffer_dedent_floor() {
        let mut buf = OutputBuffer::with_spaces(2);
        buf.dedent();
        buf.dedent();
        buf.line("no indent");
        let output = buf.finish();
        assert_eq!(output, "no indent");
    }

    #[test]
    fn test_ast_builder_struct() {
        let mut builder = AstBuilder::new();
        builder
            .add_struct("User", Visibility::Public)
            .derive("Debug")
            .derive("Clone")
            .field(Visibility::Public, "name", "String")
            .field(Visibility::Private, "age", "u32")
            .doc("A user record")
            .build();

        assert_eq!(builder.nodes().len(), 1);
        match &builder.nodes()[0] {
            AstNode::Struct { name, fields, derives, .. } => {
                assert_eq!(name, "User");
                assert_eq!(fields.len(), 2);
                assert_eq!(derives.len(), 2);
            }
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn test_ast_builder_function() {
        let mut builder = AstBuilder::new();
        builder
            .add_function("greet", Visibility::Public)
            .param("name", "&str")
            .returns("String")
            .body("format!(\"Hello, {name}!\")")
            .build();

        assert_eq!(builder.nodes().len(), 1);
        match &builder.nodes()[0] {
            AstNode::Function { name, params, return_type, .. } => {
                assert_eq!(name, "greet");
                assert_eq!(params.len(), 1);
                assert!(return_type.is_some());
            }
            _ => panic!("expected Function"),
        }
    }

    #[test]
    fn test_ast_builder_enum() {
        let mut builder = AstBuilder::new();
        builder
            .add_enum("Color", Visibility::Public)
            .variant("Red", None)
            .variant("Rgb", Some("u8, u8, u8".to_string()))
            .derive("Debug")
            .build();

        match &builder.nodes()[0] {
            AstNode::Enum { variants, .. } => {
                assert_eq!(variants.len(), 2);
                assert!(variants[1].1.is_some());
            }
            _ => panic!("expected Enum"),
        }
    }

    #[test]
    fn test_ast_builder_const() {
        let mut builder = AstBuilder::new();
        builder.add_const("MAX", Visibility::Public, "usize", "100");
        match &builder.nodes()[0] {
            AstNode::Const { name, value, .. } => {
                assert_eq!(name, "MAX");
                assert_eq!(value, "100");
            }
            _ => panic!("expected Const"),
        }
    }

    #[test]
    fn test_ast_builder_type_alias() {
        let mut builder = AstBuilder::new();
        builder.add_type_alias("Result", Visibility::Public, "std::result::Result<T, MyError>");
        match &builder.nodes()[0] {
            AstNode::TypeAlias { name, target, .. } => {
                assert_eq!(name, "Result");
                assert!(target.contains("MyError"));
            }
            _ => panic!("expected TypeAlias"),
        }
    }

    #[test]
    fn test_code_renderer_full() {
        let mut renderer = CodeRenderer::new();
        renderer.set_header("Auto-generated file");
        renderer.imports_mut().add("std::collections", "HashMap");
        renderer.add_node(AstNode::Struct {
            name: "Config".to_string(),
            visibility: Visibility::Public,
            fields: vec![(Visibility::Public, "debug".to_string(), "bool".to_string())],
            derives: vec!["Debug".to_string()],
            doc: Some("Configuration struct".to_string()),
        });

        let output = renderer.render();
        assert!(output.contains("// Auto-generated file"));
        assert!(output.contains("use std::collections::HashMap;"));
        assert!(output.contains("pub struct Config"));
        assert!(output.contains("pub debug: bool,"));
    }

    #[test]
    fn test_code_renderer_function() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::Function {
            name: "add".to_string(),
            visibility: Visibility::Public,
            params: vec![
                Param { name: "a".to_string(), type_name: "i32".to_string() },
                Param { name: "b".to_string(), type_name: "i32".to_string() },
            ],
            return_type: Some("i32".to_string()),
            body: "a + b".to_string(),
            doc: None,
            is_async: false,
        });
        let output = renderer.render();
        assert!(output.contains("pub fn add(a: i32, b: i32) -> i32 {"));
        assert!(output.contains("    a + b"));
    }

    #[test]
    fn test_code_renderer_async_function() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::Function {
            name: "fetch".to_string(),
            visibility: Visibility::Public,
            params: vec![],
            return_type: Some("Response".to_string()),
            body: "do_fetch()".to_string(),
            doc: None,
            is_async: true,
        });
        let output = renderer.render();
        assert!(output.contains("pub async fn fetch()"));
    }

    #[test]
    fn test_code_renderer_enum() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::Enum {
            name: "Status".to_string(),
            visibility: Visibility::Public,
            variants: vec![
                ("Active".to_string(), None),
                ("Error".to_string(), Some("String".to_string())),
            ],
            derives: vec!["Debug".to_string()],
            doc: None,
        });
        let output = renderer.render();
        assert!(output.contains("Active,"));
        assert!(output.contains("Error(String),"));
    }

    #[test]
    fn test_code_renderer_impl_block() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::Impl {
            target: "Config".to_string(),
            trait_name: None,
            methods: vec![AstNode::Function {
                name: "new".to_string(),
                visibility: Visibility::Public,
                params: vec![],
                return_type: Some("Self".to_string()),
                body: "Self { debug: false }".to_string(),
                doc: None,
                is_async: false,
            }],
        });
        let output = renderer.render();
        assert!(output.contains("impl Config {"));
        assert!(output.contains("pub fn new()"));
    }

    #[test]
    fn test_code_renderer_trait_impl() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::Impl {
            target: "Config".to_string(),
            trait_name: Some("Default".to_string()),
            methods: vec![],
        });
        let output = renderer.render();
        assert!(output.contains("impl Default for Config {"));
    }

    #[test]
    fn test_generation_plan_add_get() {
        let mut plan = GenerationPlan::new();
        plan.add_file("src/main.rs", "fn main() {}", false).unwrap();
        assert_eq!(plan.file_count(), 1);
        let f = plan.get_file("src/main.rs").unwrap();
        assert_eq!(f.content, "fn main() {}");
        assert!(!f.overwrite);
    }

    #[test]
    fn test_generation_plan_duplicate_error() {
        let mut plan = GenerationPlan::new();
        plan.add_file("a.rs", "x", false).unwrap();
        let err = plan.add_file("a.rs", "y", false).unwrap_err();
        assert!(matches!(err, CodeGenError::FileExists(_)));
    }

    #[test]
    fn test_generation_plan_overwrite() {
        let mut plan = GenerationPlan::new();
        plan.add_file("a.rs", "old", true).unwrap();
        plan.add_file("a.rs", "new", true).unwrap();
        assert_eq!(plan.get_file("a.rs").unwrap().content, "new");
    }

    #[test]
    fn test_generation_plan_total_bytes() {
        let mut plan = GenerationPlan::new();
        plan.add_file("a.rs", "abc", false).unwrap();
        plan.add_file("b.rs", "de", false).unwrap();
        assert_eq!(plan.total_bytes(), 5);
    }

    #[test]
    fn test_generation_plan_remove() {
        let mut plan = GenerationPlan::new();
        plan.add_file("a.rs", "x", false).unwrap();
        assert!(plan.remove_file("a.rs"));
        assert!(!plan.remove_file("a.rs"));
        assert_eq!(plan.file_count(), 0);
    }

    #[test]
    fn test_generation_plan_paths_sorted() {
        let mut plan = GenerationPlan::new();
        plan.add_file("z.rs", "", false).unwrap();
        plan.add_file("a.rs", "", false).unwrap();
        plan.add_file("m.rs", "", false).unwrap();
        let paths = plan.file_paths();
        assert_eq!(paths, vec!["a.rs", "m.rs", "z.rs"]);
    }

    #[test]
    fn test_generation_plan_summary() {
        let mut plan = GenerationPlan::new();
        plan.add_file("src/lib.rs", "pub mod foo;", false).unwrap();
        let summary = plan.summary();
        assert!(summary.contains("1 files"));
        assert!(summary.contains("[create]"));
        assert!(summary.contains("src/lib.rs"));
    }

    #[test]
    fn test_visibility_prefix() {
        assert_eq!(Visibility::Private.prefix(), "");
        assert_eq!(Visibility::Public.prefix(), "pub ");
        assert_eq!(Visibility::Crate.prefix(), "pub(crate) ");
    }

    #[test]
    fn test_code_renderer_const() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::Const {
            name: "PI".to_string(),
            visibility: Visibility::Public,
            type_name: "f64".to_string(),
            value: "3.14159".to_string(),
        });
        let output = renderer.render();
        assert!(output.contains("pub const PI: f64 = 3.14159;"));
    }

    #[test]
    fn test_code_renderer_type_alias() {
        let mut renderer = CodeRenderer::new();
        renderer.add_node(AstNode::TypeAlias {
            name: "MyResult".to_string(),
            visibility: Visibility::Public,
            target: "Result<T, MyError>".to_string(),
        });
        let output = renderer.render();
        assert!(output.contains("pub type MyResult = Result<T, MyError>;"));
    }

    #[test]
    fn test_raw_node() {
        let mut builder = AstBuilder::new();
        builder.add_raw("// custom code\nlet x = 42;");
        let nodes = builder.into_nodes();
        assert_eq!(nodes.len(), 1);
        match &nodes[0] {
            AstNode::Raw(code) => assert!(code.contains("42")),
            _ => panic!("expected Raw"),
        }
    }

    #[test]
    fn test_output_buffer_line_count() {
        let mut buf = OutputBuffer::with_spaces(2);
        buf.line("a");
        buf.line("b");
        buf.blank();
        buf.line("c");
        assert_eq!(buf.line_count(), 4);
    }

    #[test]
    fn test_template_no_vars_needed() {
        let tmpl = Template::new("static content").unwrap();
        let vars = HashMap::new();
        let result = tmpl.render(&vars).unwrap();
        assert_eq!(result, "static content");
    }

    #[test]
    fn test_full_codegen_workflow() {
        // Build AST.
        let mut builder = AstBuilder::new();
        builder
            .add_struct("Point", Visibility::Public)
            .derive("Debug")
            .field(Visibility::Public, "x", "f64")
            .field(Visibility::Public, "y", "f64")
            .build();
        builder
            .add_function("distance", Visibility::Public)
            .param("a", "&Point")
            .param("b", "&Point")
            .returns("f64")
            .body("((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()")
            .build();

        // Render.
        let mut renderer = CodeRenderer::new();
        renderer.set_header("Generated by code_gen");
        renderer.add_nodes(builder.into_nodes());
        let output = renderer.render();

        assert!(output.contains("pub struct Point"));
        assert!(output.contains("pub fn distance"));
        assert!(output.contains("sqrt()"));

        // Add to plan.
        let mut plan = GenerationPlan::new();
        plan.add_file("src/point.rs", &output, false).unwrap();
        assert_eq!(plan.file_count(), 1);
        assert!(plan.total_bytes() > 100);
    }
}
