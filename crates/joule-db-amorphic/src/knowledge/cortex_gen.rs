//! Cortex Generation: Pattern-Lang's 5 primitives → compiled IR → output.
//!
//! This replaces neural text generation for algorithmic queries.
//! "Implement GCD" → decompose("gcd") → compile() → Cortex IR → code.
//!
//! No neural network. No training. Correct by construction.
//! The 5 primitives (Converge, LinearScan, Direct, MapBuild, FilterBuild)
//! compose to any algorithm in the 1,135-pattern catalog.
//!
//! Requires `pattern-lang` feature.

#[cfg(feature = "pattern-lang")]
mod inner {
    /// Generate code for a named pattern via Cortex IR compilation.
    ///
    /// decompose(name) → PrimitiveProgram → compile() → Cortex IR Function
    /// → render to human-readable form.
    pub fn generate_code(pattern_name: &str) -> Option<GeneratedCode> {
        let program = pattern_core::primitives::decompose(pattern_name)?;
        let ir_function = pattern_core::primitives::compile(&program);

        // Render the IR function to readable form
        let ir_text = render_ir(&ir_function);

        // Extract metadata
        let args: Vec<(String, String)> = program
            .args
            .iter()
            .map(|(name, ty)| (name.clone(), format!("{:?}", ty)))
            .collect();

        Some(GeneratedCode {
            pattern_name: pattern_name.to_string(),
            primitive_kind: describe_primitive(&program.body),
            args,
            return_type: format!("{:?}", program.ret_type),
            ir_text,
            block_count: ir_function.blocks.len(),
            instruction_count: ir_function
                .blocks
                .iter()
                .map(|b| b.instrs.len())
                .sum(),
        })
    }

    /// Try to generate code from a natural language query.
    /// Extracts keywords → resolves via bridge → decomposes best match → compiles.
    pub fn generate_from_query(
        query: &str,
        resolver: &super::super::pattern_resolver::PatternLangResolver,
    ) -> Option<GeneratedCode> {
        use crate::ai::pattern_bridge::PatternResolver;

        let keywords: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_string())
            .collect();

        let matches = resolver.resolve(&keywords, None);
        let best = matches.first()?;

        if best.score < 0.3 {
            return None;
        }

        generate_code(&best.pattern_name)
    }

    /// List all patterns that can be compiled.
    pub fn available_patterns() -> Vec<&'static str> {
        pattern_core::primitives::all_canonical_names()
            .iter()
            .filter(|name| pattern_core::primitives::decompose(name).is_some())
            .copied()
            .collect()
    }

    /// Count of compilable patterns.
    pub fn compilable_count() -> usize {
        pattern_core::primitives::all_canonical_names()
            .iter()
            .filter(|name| pattern_core::primitives::decompose(name).is_some())
            .count()
    }

    /// Generated code output.
    #[derive(Clone, Debug)]
    pub struct GeneratedCode {
        pub pattern_name: String,
        pub primitive_kind: String,
        pub args: Vec<(String, String)>,
        pub return_type: String,
        pub ir_text: String,
        pub block_count: usize,
        pub instruction_count: usize,
    }

    impl GeneratedCode {
        pub fn render(&self) -> String {
            format!(
                "// Pattern: {} ({})\n// Args: {}\n// Returns: {}\n// Blocks: {}, Instructions: {}\n\n{}",
                self.pattern_name,
                self.primitive_kind,
                self.args
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect::<Vec<_>>()
                    .join(", "),
                self.return_type,
                self.block_count,
                self.instruction_count,
                self.ir_text,
            )
        }
    }

    fn describe_primitive(body: &pattern_core::primitives::Primitive) -> String {
        match body {
            pattern_core::primitives::Primitive::Converge { .. } => "Converge (while-loop)".into(),
            pattern_core::primitives::Primitive::LinearScan { .. } => "LinearScan (fold)".into(),
            pattern_core::primitives::Primitive::Direct { .. } => "Direct (expression)".into(),
            pattern_core::primitives::Primitive::MapBuild { .. } => "MapBuild (list comprehension)".into(),
            pattern_core::primitives::Primitive::FilterBuild { .. } => "FilterBuild (filter)".into(),
            pattern_core::primitives::Primitive::NestedScan { .. } => "NestedScan (nested loops)".into(),
            pattern_core::primitives::Primitive::TwoPointer { .. } => "TwoPointer (two-pointer)".into(),
            _ => "Unknown".into(),
        }
    }

    fn render_ir(func: &cortex_ir::ir::Function) -> String {
        let mut out = String::new();
        out.push_str(&format!("fn {}(", func.name));
        for (i, (name, ty)) in func.args.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{}: {:?}", name, ty));
        }
        out.push_str(&format!(") -> {:?} {{\n", func.ret_type));

        for block in &func.blocks {
            out.push_str(&format!("  block_{}:\n", block.id));
            for instr in &block.instrs {
                out.push_str(&format!("    {:?}\n", instr));
            }
            out.push_str(&format!("    {:?}\n", block.terminator));
        }

        out.push_str("}\n");
        out
    }
}

#[cfg(feature = "pattern-lang")]
pub use inner::{generate_code, generate_from_query, available_patterns, compilable_count, GeneratedCode};

/// Stub when pattern-lang feature is not enabled.
#[cfg(not(feature = "pattern-lang"))]
pub fn compilable_count() -> usize { 0 }

#[cfg(test)]
mod tests {
    #[test]
    fn test_compilable_count() {
        let count = super::compilable_count();
        eprintln!("Compilable patterns: {}", count);

        #[cfg(feature = "pattern-lang")]
        assert!(count > 0, "should have compilable patterns");
    }

    #[cfg(feature = "pattern-lang")]
    #[test]
    fn test_generate_gcd() {
        let code = super::generate_code("gcd");
        assert!(code.is_some(), "should compile gcd");
        let code = code.unwrap();
        eprintln!("{}", code.render());
        assert_eq!(code.primitive_kind, "Converge (while-loop)");
        assert!(code.block_count > 0);
        assert!(code.instruction_count > 0);
    }

    #[cfg(feature = "pattern-lang")]
    #[test]
    fn test_generate_from_query() {
        let resolver = super::super::pattern_resolver::PatternLangResolver::from_primitives();
        let code = super::generate_from_query("compute greatest common divisor", &resolver);
        eprintln!("Query 'compute greatest common divisor' → {:?}",
            code.as_ref().map(|c| &c.pattern_name));
    }

    #[cfg(feature = "pattern-lang")]
    #[test]
    fn test_available_patterns() {
        let patterns = super::available_patterns();
        eprintln!("Available compilable patterns: {}", patterns.len());
        for p in patterns.iter().take(10) {
            eprintln!("  {}", p);
        }
        assert!(!patterns.is_empty());
    }
}
