//! Shader program model — parse, validate, and preprocess GLSL shaders.
//!
//! Pure Rust replacement for glslang, shader-toy utilities, and three.js
//! ShaderMaterial. Models shader programs with uniforms, varyings, and
//! a basic GLSL preprocessor.

use std::collections::HashMap;
use std::fmt;

// ── Shader Types ─────────────────────────────────────────────

/// Type of shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

impl fmt::Display for ShaderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vertex => write!(f, "vertex"),
            Self::Fragment => write!(f, "fragment"),
            Self::Compute => write!(f, "compute"),
        }
    }
}

// ── Uniform Types ────────────────────────────────────────────

/// Type of a shader uniform variable.
#[derive(Debug, Clone, PartialEq)]
pub enum UniformType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    Int,
    Sampler2D,
}

impl fmt::Display for UniformType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float => write!(f, "float"),
            Self::Vec2 => write!(f, "vec2"),
            Self::Vec3 => write!(f, "vec3"),
            Self::Vec4 => write!(f, "vec4"),
            Self::Mat4 => write!(f, "mat4"),
            Self::Int => write!(f, "int"),
            Self::Sampler2D => write!(f, "sampler2D"),
        }
    }
}

/// Uniform value.
#[derive(Debug, Clone, PartialEq)]
pub enum UniformValue {
    Float(f64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Mat4([f64; 16]),
    Int(i32),
    Sampler2D(u32), // texture unit
}

/// A shader uniform declaration with optional value.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    pub name: String,
    pub uniform_type: UniformType,
    pub value: Option<UniformValue>,
}

impl Uniform {
    pub fn new(name: impl Into<String>, uniform_type: UniformType) -> Self {
        Self {
            name: name.into(),
            uniform_type,
            value: None,
        }
    }

    pub fn with_value(mut self, value: UniformValue) -> Self {
        self.value = Some(value);
        self
    }
}

// ── ShaderSource ─────────────────────────────────────────────

/// A single shader source (vertex, fragment, or compute).
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderSource {
    pub source_code: String,
    pub shader_type: ShaderType,
}

impl ShaderSource {
    pub fn new(source_code: impl Into<String>, shader_type: ShaderType) -> Self {
        Self {
            source_code: source_code.into(),
            shader_type,
        }
    }

    /// Check if the shader source has a `main` function.
    pub fn has_main(&self) -> bool {
        self.source_code.contains("void main")
    }

    /// Extract varying declarations from the source.
    pub fn varyings(&self) -> Vec<VaryingDecl> {
        let mut result = Vec::new();
        for line in self.source_code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("varying ") || trimmed.starts_with("out ") || trimmed.starts_with("in ") {
                let parts: Vec<&str> = trimmed
                    .trim_end_matches(';')
                    .split_whitespace()
                    .collect();
                if parts.len() >= 3 {
                    let qualifier = parts[0].to_string();
                    let var_type = parts[1].to_string();
                    let name = parts[2].to_string();
                    result.push(VaryingDecl { qualifier, var_type, name });
                }
            }
        }
        result
    }

    /// Extract uniform declarations from the source.
    pub fn uniforms(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();
        for line in self.source_code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("uniform ") {
                let parts: Vec<&str> = trimmed
                    .trim_end_matches(';')
                    .split_whitespace()
                    .collect();
                if parts.len() >= 3 {
                    result.push((parts[1].to_string(), parts[2].to_string()));
                }
            }
        }
        result
    }
}

/// A varying/in/out declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct VaryingDecl {
    pub qualifier: String,
    pub var_type: String,
    pub name: String,
}

// ── ShaderProgram ────────────────────────────────────────────

/// A complete shader program with vertex + fragment sources and uniforms.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderProgram {
    pub vertex: ShaderSource,
    pub fragment: ShaderSource,
    pub uniforms: Vec<Uniform>,
}

/// Shader validation error.
#[derive(Debug, Clone, PartialEq)]
pub enum ShaderError {
    MissingMain(ShaderType),
    VaryingMismatch {
        name: String,
        vertex_type: Option<String>,
        fragment_type: Option<String>,
    },
    PreprocessorError(String),
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMain(t) => write!(f, "{t} shader missing void main()"),
            Self::VaryingMismatch { name, vertex_type, fragment_type } =>
                write!(f, "varying '{name}' type mismatch: vertex={vertex_type:?} fragment={fragment_type:?}"),
            Self::PreprocessorError(msg) => write!(f, "preprocessor error: {msg}"),
        }
    }
}

impl std::error::Error for ShaderError {}

impl ShaderProgram {
    pub fn new(vertex: ShaderSource, fragment: ShaderSource) -> Self {
        Self {
            vertex,
            fragment,
            uniforms: Vec::new(),
        }
    }

    pub fn with_uniforms(mut self, uniforms: Vec<Uniform>) -> Self {
        self.uniforms = uniforms;
        self
    }

    /// Validate the shader program.
    pub fn validate(&self) -> Vec<ShaderError> {
        let mut errors = Vec::new();

        if !self.vertex.has_main() {
            errors.push(ShaderError::MissingMain(ShaderType::Vertex));
        }
        if !self.fragment.has_main() {
            errors.push(ShaderError::MissingMain(ShaderType::Fragment));
        }

        // Check varying matching: vertex `out` should match fragment `in`,
        // or both use `varying` with same type.
        let v_varyings = self.vertex.varyings();
        let f_varyings = self.fragment.varyings();

        // Build maps of outputs from vertex and inputs to fragment
        let v_out: HashMap<&str, &str> = v_varyings.iter()
            .filter(|v| v.qualifier == "varying" || v.qualifier == "out")
            .map(|v| (v.name.as_str(), v.var_type.as_str()))
            .collect();

        let f_in: HashMap<&str, &str> = f_varyings.iter()
            .filter(|v| v.qualifier == "varying" || v.qualifier == "in")
            .map(|v| (v.name.as_str(), v.var_type.as_str()))
            .collect();

        for (name, ftype) in &f_in {
            match v_out.get(name) {
                Some(vtype) => {
                    if vtype != ftype {
                        errors.push(ShaderError::VaryingMismatch {
                            name: name.to_string(),
                            vertex_type: Some(vtype.to_string()),
                            fragment_type: Some(ftype.to_string()),
                        });
                    }
                }
                None => {
                    errors.push(ShaderError::VaryingMismatch {
                        name: name.to_string(),
                        vertex_type: None,
                        fragment_type: Some(ftype.to_string()),
                    });
                }
            }
        }

        errors
    }

    /// Set a uniform value by name.
    pub fn set_uniform(&mut self, name: &str, value: UniformValue) -> bool {
        for u in &mut self.uniforms {
            if u.name == name {
                u.value = Some(value);
                return true;
            }
        }
        false
    }
}

// ── GLSL Preprocessor ────────────────────────────────────────

/// GLSL preprocessor that handles #define, #ifdef/#ifndef/#else/#endif, and #include.
pub struct GlslPreprocessor {
    defines: HashMap<String, String>,
    includes: HashMap<String, String>,
}

impl GlslPreprocessor {
    pub fn new() -> Self {
        Self {
            defines: HashMap::new(),
            includes: HashMap::new(),
        }
    }

    /// Add a #define macro.
    pub fn define(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.defines.insert(name.into(), value.into());
    }

    /// Register an include file.
    pub fn register_include(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.includes.insert(path.into(), content.into());
    }

    /// Process GLSL source with preprocessor directives.
    pub fn process(&self, source: &str) -> Result<String, ShaderError> {
        let mut output = String::new();
        let mut lines = source.lines().peekable();
        let mut ifdef_stack: Vec<IfDefState> = Vec::new();

        // Build effective defines (from source + constructor)
        let mut defines = self.defines.clone();

        while let Some(line) = lines.next() {
            let trimmed = line.trim();

            // #define
            if trimmed.starts_with("#define ") {
                let rest = trimmed.strip_prefix("#define ").unwrap().trim();
                let mut parts = rest.splitn(2, char::is_whitespace);
                let name = parts.next().unwrap_or("").to_string();
                let value = parts.next().unwrap_or("").trim().to_string();
                if !name.is_empty() {
                    defines.insert(name, value);
                }
                continue;
            }

            // #undef
            if trimmed.starts_with("#undef ") {
                let name = trimmed.strip_prefix("#undef ").unwrap().trim();
                defines.remove(name);
                continue;
            }

            // #ifdef
            if trimmed.starts_with("#ifdef ") {
                let name = trimmed.strip_prefix("#ifdef ").unwrap().trim();
                let active = defines.contains_key(name);
                ifdef_stack.push(IfDefState { active, else_seen: false });
                continue;
            }

            // #ifndef
            if trimmed.starts_with("#ifndef ") {
                let name = trimmed.strip_prefix("#ifndef ").unwrap().trim();
                let active = !defines.contains_key(name);
                ifdef_stack.push(IfDefState { active, else_seen: false });
                continue;
            }

            // #else
            if trimmed == "#else" {
                if let Some(state) = ifdef_stack.last_mut() {
                    state.active = !state.active;
                    state.else_seen = true;
                } else {
                    return Err(ShaderError::PreprocessorError("#else without #ifdef".into()));
                }
                continue;
            }

            // #endif
            if trimmed == "#endif" {
                if ifdef_stack.pop().is_none() {
                    return Err(ShaderError::PreprocessorError("#endif without #ifdef".into()));
                }
                continue;
            }

            // #include
            if trimmed.starts_with("#include ") {
                if ifdef_stack.iter().all(|s| s.active) {
                    let rest = trimmed.strip_prefix("#include ").unwrap().trim();
                    let path = rest.trim_matches('"').trim_matches('<').trim_matches('>');
                    match self.includes.get(path) {
                        Some(content) => {
                            output.push_str(content);
                            output.push('\n');
                        }
                        None => {
                            return Err(ShaderError::PreprocessorError(
                                format!("include not found: {path}")
                            ));
                        }
                    }
                }
                continue;
            }

            // Regular line — emit if active
            if ifdef_stack.iter().all(|s| s.active) {
                // Replace defines in the line
                let mut processed = line.to_string();
                for (name, value) in &defines {
                    if !value.is_empty() {
                        // Simple word-boundary replacement
                        processed = replace_identifier(&processed, name, value);
                    }
                }
                output.push_str(&processed);
                output.push('\n');
            }
        }

        if !ifdef_stack.is_empty() {
            return Err(ShaderError::PreprocessorError("unterminated #ifdef".into()));
        }

        Ok(output)
    }
}

impl Default for GlslPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

struct IfDefState {
    active: bool,
    #[allow(dead_code)]
    else_seen: bool,
}

/// Replace an identifier in source code (basic word-boundary replacement).
fn replace_identifier(source: &str, name: &str, replacement: &str) -> String {
    let mut result = String::new();
    let mut chars = source.chars().peekable();
    let name_chars: Vec<char> = name.chars().collect();

    while chars.peek().is_some() {
        // Try to match the identifier
        let remaining: String = chars.clone().collect();
        if remaining.starts_with(name) {
            // Check character before (already consumed, check result)
            let before_ok = result.is_empty() || {
                let last = result.chars().last().unwrap();
                !last.is_alphanumeric() && last != '_'
            };
            // Check character after
            let after_chars: String = chars.clone().skip(name_chars.len()).collect();
            let after_ok = after_chars.is_empty() || {
                let next = after_chars.chars().next().unwrap();
                !next.is_alphanumeric() && next != '_'
            };

            if before_ok && after_ok {
                result.push_str(replacement);
                for _ in 0..name_chars.len() {
                    chars.next();
                }
                continue;
            }
        }

        if let Some(ch) = chars.next() {
            result.push(ch);
        }
    }

    result
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_vertex() -> ShaderSource {
        ShaderSource::new(
            r#"
            varying vec2 vUv;
            uniform mat4 uModelViewMatrix;
            void main() {
                vUv = uv;
                gl_Position = uModelViewMatrix * vec4(position, 1.0);
            }
            "#,
            ShaderType::Vertex,
        )
    }

    fn simple_fragment() -> ShaderSource {
        ShaderSource::new(
            r#"
            varying vec2 vUv;
            uniform sampler2D uTexture;
            void main() {
                gl_FragColor = texture2D(uTexture, vUv);
            }
            "#,
            ShaderType::Fragment,
        )
    }

    #[test]
    fn shader_has_main() {
        let vs = simple_vertex();
        assert!(vs.has_main());
        let empty = ShaderSource::new("// no main here", ShaderType::Vertex);
        assert!(!empty.has_main());
    }

    #[test]
    fn extract_varyings() {
        let vs = simple_vertex();
        let varyings = vs.varyings();
        assert_eq!(varyings.len(), 1);
        assert_eq!(varyings[0].name, "vUv");
        assert_eq!(varyings[0].var_type, "vec2");
    }

    #[test]
    fn extract_uniforms_from_source() {
        let vs = simple_vertex();
        let uniforms = vs.uniforms();
        assert_eq!(uniforms.len(), 1);
        assert_eq!(uniforms[0].0, "mat4");
        assert_eq!(uniforms[0].1, "uModelViewMatrix");
    }

    #[test]
    fn validate_valid_program() {
        let prog = ShaderProgram::new(simple_vertex(), simple_fragment());
        let errors = prog.validate();
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn validate_missing_main() {
        let vs = ShaderSource::new("varying vec2 vUv;", ShaderType::Vertex);
        let fs = simple_fragment();
        let prog = ShaderProgram::new(vs, fs);
        let errors = prog.validate();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0], ShaderError::MissingMain(ShaderType::Vertex)));
    }

    #[test]
    fn validate_varying_mismatch() {
        let vs = ShaderSource::new(
            "out vec3 vColor;\nvoid main() {}",
            ShaderType::Vertex,
        );
        let fs = ShaderSource::new(
            "in vec4 vColor;\nvoid main() {}",
            ShaderType::Fragment,
        );
        let prog = ShaderProgram::new(vs, fs);
        let errors = prog.validate();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            ShaderError::VaryingMismatch { name, .. } => assert_eq!(name, "vColor"),
            _ => panic!("expected VaryingMismatch"),
        }
    }

    #[test]
    fn validate_missing_varying() {
        let vs = ShaderSource::new("void main() {}", ShaderType::Vertex);
        let fs = ShaderSource::new(
            "in vec2 vUv;\nvoid main() {}",
            ShaderType::Fragment,
        );
        let prog = ShaderProgram::new(vs, fs);
        let errors = prog.validate();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            ShaderError::VaryingMismatch { name, vertex_type, .. } => {
                assert_eq!(name, "vUv");
                assert!(vertex_type.is_none());
            }
            _ => panic!("expected VaryingMismatch"),
        }
    }

    #[test]
    fn set_uniform_value() {
        let mut prog = ShaderProgram::new(simple_vertex(), simple_fragment())
            .with_uniforms(vec![
                Uniform::new("uTime", UniformType::Float),
                Uniform::new("uResolution", UniformType::Vec2),
            ]);
        assert!(prog.set_uniform("uTime", UniformValue::Float(1.5)));
        assert!(!prog.set_uniform("uNonexistent", UniformValue::Float(0.0)));
        assert_eq!(prog.uniforms[0].value, Some(UniformValue::Float(1.5)));
    }

    #[test]
    fn preprocessor_define() {
        let mut pp = GlslPreprocessor::new();
        pp.define("MAX_LIGHTS", "4");
        let result = pp.process("int lights = MAX_LIGHTS;").unwrap();
        assert!(result.contains("int lights = 4;"));
    }

    #[test]
    fn preprocessor_ifdef() {
        let pp = GlslPreprocessor::new();
        let src = "#define USE_NORMAL\n#ifdef USE_NORMAL\nvec3 n = normal;\n#endif\n";
        let result = pp.process(src).unwrap();
        assert!(result.contains("vec3 n = normal;"));
    }

    #[test]
    fn preprocessor_ifdef_false() {
        let pp = GlslPreprocessor::new();
        let src = "#ifdef UNDEFINED_FLAG\nvec3 n = normal;\n#endif\n";
        let result = pp.process(src).unwrap();
        assert!(!result.contains("vec3 n = normal;"));
    }

    #[test]
    fn preprocessor_ifndef() {
        let pp = GlslPreprocessor::new();
        let src = "#ifndef SOME_FLAG\nfloat x = 1.0;\n#endif\n";
        let result = pp.process(src).unwrap();
        assert!(result.contains("float x = 1.0;"));
    }

    #[test]
    fn preprocessor_else() {
        let pp = GlslPreprocessor::new();
        let src = "#ifdef NOPE\nfloat a;\n#else\nfloat b;\n#endif\n";
        let result = pp.process(src).unwrap();
        assert!(!result.contains("float a;"));
        assert!(result.contains("float b;"));
    }

    #[test]
    fn preprocessor_include() {
        let mut pp = GlslPreprocessor::new();
        pp.register_include("common.glsl", "vec3 lightDir = vec3(0.0, 1.0, 0.0);");
        let src = "#include \"common.glsl\"\nvoid main() {}";
        let result = pp.process(src).unwrap();
        assert!(result.contains("vec3 lightDir"));
        assert!(result.contains("void main()"));
    }

    #[test]
    fn preprocessor_include_not_found() {
        let pp = GlslPreprocessor::new();
        let src = "#include \"missing.glsl\"\n";
        let result = pp.process(src);
        assert!(result.is_err());
    }

    #[test]
    fn preprocessor_unterminated_ifdef() {
        let pp = GlslPreprocessor::new();
        let src = "#ifdef FOO\nfloat x;\n";
        let result = pp.process(src);
        assert!(result.is_err());
    }

    #[test]
    fn uniform_type_display() {
        assert_eq!(UniformType::Float.to_string(), "float");
        assert_eq!(UniformType::Mat4.to_string(), "mat4");
        assert_eq!(UniformType::Sampler2D.to_string(), "sampler2D");
    }

    #[test]
    fn shader_type_display() {
        assert_eq!(ShaderType::Vertex.to_string(), "vertex");
        assert_eq!(ShaderType::Fragment.to_string(), "fragment");
        assert_eq!(ShaderType::Compute.to_string(), "compute");
    }
}
