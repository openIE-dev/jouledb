//! Command-line argument parser with flags, options, subcommands, and help generation.
//!
//! Provides a declarative API for defining CLI arguments: positional args,
//! named flags (`--verbose`, `-v`), options with values (`--output file.txt`),
//! subcommands, required/optional parameters, default values, validation,
//! and automatic help text generation.

use std::collections::HashMap;
use std::fmt;

// ── Argument Types ──

/// The kind of argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgKind {
    /// Boolean flag (present or absent).
    Flag,
    /// Option that takes a value.
    Option,
    /// Positional argument.
    Positional,
}

/// A single argument definition.
#[derive(Debug, Clone)]
pub struct Arg {
    pub name: String,
    pub short: Option<char>,
    pub long: Option<String>,
    pub kind: ArgKind,
    pub help: String,
    pub required: bool,
    pub default: Option<String>,
    pub value_name: Option<String>,
    pub choices: Vec<String>,
}

impl Arg {
    /// Create a positional argument.
    pub fn positional(name: &str) -> Self {
        Self {
            name: name.to_string(),
            short: None,
            long: None,
            kind: ArgKind::Positional,
            help: String::new(),
            required: true,
            default: None,
            value_name: None,
            choices: Vec::new(),
        }
    }

    /// Create a boolean flag.
    pub fn flag(name: &str) -> Self {
        Self {
            name: name.to_string(),
            short: None,
            long: Some(format!("--{name}")),
            kind: ArgKind::Flag,
            help: String::new(),
            required: false,
            default: None,
            value_name: None,
            choices: Vec::new(),
        }
    }

    /// Create an option that takes a value.
    pub fn option(name: &str) -> Self {
        Self {
            name: name.to_string(),
            short: None,
            long: Some(format!("--{name}")),
            kind: ArgKind::Option,
            help: String::new(),
            required: false,
            default: None,
            value_name: Some("VALUE".to_string()),
            choices: Vec::new(),
        }
    }

    pub fn short(mut self, ch: char) -> Self {
        self.short = Some(ch);
        self
    }

    pub fn long(mut self, name: &str) -> Self {
        self.long = Some(format!("--{name}"));
        self
    }

    pub fn help(mut self, text: &str) -> Self {
        self.help = text.to_string();
        self
    }

    pub fn required(mut self, req: bool) -> Self {
        self.required = req;
        self
    }

    pub fn default_value(mut self, val: &str) -> Self {
        self.default = Some(val.to_string());
        self
    }

    pub fn value_name(mut self, name: &str) -> Self {
        self.value_name = Some(name.to_string());
        self
    }

    pub fn choices(mut self, opts: &[&str]) -> Self {
        self.choices = opts.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Check if this arg matches a token like `--name` or `-n`.
    fn matches_token(&self, token: &str) -> bool {
        if let Some(long) = &self.long {
            if token == long { return true; }
        }
        if let Some(short) = self.short {
            if token == format!("-{short}") { return true; }
        }
        false
    }
}

// ── Parse Result ──

/// Parsed argument values.
#[derive(Debug, Clone)]
pub struct ParseResult {
    pub flags: HashMap<String, bool>,
    pub options: HashMap<String, String>,
    pub positionals: Vec<String>,
    pub subcommand: Option<(String, Box<ParseResult>)>,
    pub errors: Vec<String>,
}

impl ParseResult {
    fn new() -> Self {
        Self {
            flags: HashMap::new(),
            options: HashMap::new(),
            positionals: Vec::new(),
            subcommand: None,
            errors: Vec::new(),
        }
    }

    /// Get a flag value (default false).
    pub fn get_flag(&self, name: &str) -> bool {
        self.flags.get(name).copied().unwrap_or(false)
    }

    /// Get an option value.
    pub fn get_option(&self, name: &str) -> Option<&str> {
        self.options.get(name).map(|s| s.as_str())
    }

    /// Get a positional by index.
    pub fn get_positional(&self, index: usize) -> Option<&str> {
        self.positionals.get(index).map(|s| s.as_str())
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// ── Command ──

/// A command (top-level or subcommand) with arguments and sub-subcommands.
#[derive(Debug, Clone)]
pub struct Command {
    pub name: String,
    pub version: Option<String>,
    pub description: String,
    pub args: Vec<Arg>,
    pub subcommands: Vec<Command>,
}

impl Command {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: None,
            description: String::new(),
            args: Vec::new(),
            subcommands: Vec::new(),
        }
    }

    pub fn version(mut self, ver: &str) -> Self {
        self.version = Some(ver.to_string());
        self
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn arg(mut self, arg: Arg) -> Self {
        self.args.push(arg);
        self
    }

    pub fn subcommand(mut self, cmd: Command) -> Self {
        self.subcommands.push(cmd);
        self
    }

    /// Parse a list of string arguments.
    pub fn parse(&self, args: &[&str]) -> ParseResult {
        let mut result = ParseResult::new();
        let mut i = 0;
        let mut positional_idx = 0;

        // Apply defaults.
        for a in &self.args {
            if let Some(def) = &a.default {
                match a.kind {
                    ArgKind::Option => { result.options.insert(a.name.clone(), def.clone()); }
                    ArgKind::Flag => { result.flags.insert(a.name.clone(), def == "true"); }
                    _ => {}
                }
            }
        }

        let positional_args: Vec<&Arg> = self.args.iter()
            .filter(|a| a.kind == ArgKind::Positional)
            .collect();

        while i < args.len() {
            let token = args[i];

            // Check for subcommand.
            if !token.starts_with('-') {
                if let Some(sub) = self.subcommands.iter().find(|c| c.name == token) {
                    let sub_result = sub.parse(&args[i + 1..]);
                    result.subcommand = Some((token.to_string(), Box::new(sub_result)));
                    break;
                }
            }

            // Check for help.
            if token == "--help" || token == "-h" {
                result.flags.insert("help".to_string(), true);
                i += 1;
                continue;
            }

            // Check for version.
            if token == "--version" || token == "-V" {
                result.flags.insert("version".to_string(), true);
                i += 1;
                continue;
            }

            // Handle `--key=value` syntax.
            if token.starts_with("--") && token.contains('=') {
                let eq_pos = token.find('=').unwrap();
                let key = &token[..eq_pos];
                let val = &token[eq_pos + 1..];
                if let Some(a) = self.args.iter().find(|a| a.long.as_deref() == Some(key)) {
                    if !a.choices.is_empty() && !a.choices.iter().any(|c| c == val) {
                        result.errors.push(format!(
                            "Invalid value '{}' for {}. Valid: {:?}", val, a.name, a.choices
                        ));
                    }
                    result.options.insert(a.name.clone(), val.to_string());
                } else {
                    result.errors.push(format!("Unknown option: {key}"));
                }
                i += 1;
                continue;
            }

            // Named arg.
            if token.starts_with('-') {
                if let Some(a) = self.args.iter().find(|a| a.matches_token(token)) {
                    match a.kind {
                        ArgKind::Flag => {
                            result.flags.insert(a.name.clone(), true);
                            i += 1;
                        }
                        ArgKind::Option => {
                            if i + 1 < args.len() {
                                let val = args[i + 1];
                                if !a.choices.is_empty() && !a.choices.iter().any(|c| c == val) {
                                    result.errors.push(format!(
                                        "Invalid value '{}' for {}. Valid: {:?}", val, a.name, a.choices
                                    ));
                                }
                                result.options.insert(a.name.clone(), val.to_string());
                                i += 2;
                            } else {
                                result.errors.push(format!("Missing value for {}", a.name));
                                i += 1;
                            }
                        }
                        _ => { i += 1; }
                    }
                } else {
                    result.errors.push(format!("Unknown flag: {token}"));
                    i += 1;
                }
            } else {
                // Positional.
                result.positionals.push(token.to_string());
                positional_idx += 1;
                i += 1;
            }
        }

        // Check required args.
        for a in &self.args {
            if a.required {
                match a.kind {
                    ArgKind::Positional => {
                        // Find index of this positional.
                        let idx = positional_args.iter()
                            .position(|p| p.name == a.name)
                            .unwrap_or(0);
                        if result.positionals.get(idx).is_none() {
                            result.errors.push(format!("Missing required argument: {}", a.name));
                        }
                    }
                    ArgKind::Option => {
                        if !result.options.contains_key(&a.name) {
                            result.errors.push(format!("Missing required option: --{}", a.name));
                        }
                    }
                    _ => {}
                }
            }
        }

        result
    }

    /// Generate help text.
    pub fn help_text(&self) -> String {
        let mut out = String::new();

        // Usage line.
        out.push_str(&format!("{}", self.name));
        if let Some(v) = &self.version {
            out.push_str(&format!(" {v}"));
        }
        out.push('\n');

        if !self.description.is_empty() {
            out.push_str(&self.description);
            out.push_str("\n\n");
        }

        // Usage.
        out.push_str(&format!("USAGE:\n    {}", self.name));
        let positionals: Vec<&Arg> = self.args.iter()
            .filter(|a| a.kind == ArgKind::Positional)
            .collect();
        let named: Vec<&Arg> = self.args.iter()
            .filter(|a| a.kind != ArgKind::Positional)
            .collect();

        if !named.is_empty() { out.push_str(" [OPTIONS]"); }
        if !self.subcommands.is_empty() { out.push_str(" <COMMAND>"); }
        for p in &positionals {
            if p.required {
                out.push_str(&format!(" <{}>", p.name.to_uppercase()));
            } else {
                out.push_str(&format!(" [{}]", p.name.to_uppercase()));
            }
        }
        out.push_str("\n\n");

        // Arguments.
        if !positionals.is_empty() {
            out.push_str("ARGUMENTS:\n");
            for p in &positionals {
                out.push_str(&format!("    {:<20} {}\n",
                    format!("<{}>", p.name), p.help
                ));
            }
            out.push('\n');
        }

        // Options.
        if !named.is_empty() {
            out.push_str("OPTIONS:\n");
            for a in &named {
                let short = a.short.map(|c| format!("-{c}, ")).unwrap_or_else(|| "    ".to_string());
                let long = a.long.as_deref().unwrap_or("");
                let val = match a.kind {
                    ArgKind::Option => format!(" <{}>", a.value_name.as_deref().unwrap_or("VALUE")),
                    _ => String::new(),
                };
                let default = a.default.as_ref()
                    .map(|d| format!(" [default: {d}]"))
                    .unwrap_or_default();
                out.push_str(&format!("    {short}{long}{val:<12} {}{default}\n", a.help));
            }
            out.push('\n');
        }

        // Subcommands.
        if !self.subcommands.is_empty() {
            out.push_str("COMMANDS:\n");
            for sub in &self.subcommands {
                out.push_str(&format!("    {:<20} {}\n", sub.name, sub.description));
            }
            out.push('\n');
        }

        out
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.help_text())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cmd() -> Command {
        Command::new("myapp")
            .version("1.0.0")
            .description("A test CLI app")
            .arg(Arg::positional("input").help("Input file"))
            .arg(Arg::flag("verbose").short('v').help("Enable verbose output"))
            .arg(Arg::option("output").short('o').help("Output file").value_name("FILE"))
    }

    #[test]
    fn parse_flag() {
        let cmd = test_cmd();
        let result = cmd.parse(&["file.txt", "--verbose"]);
        assert!(result.get_flag("verbose"));
    }

    #[test]
    fn parse_short_flag() {
        let cmd = test_cmd();
        let result = cmd.parse(&["file.txt", "-v"]);
        assert!(result.get_flag("verbose"));
    }

    #[test]
    fn parse_option_long() {
        let cmd = test_cmd();
        let result = cmd.parse(&["file.txt", "--output", "out.txt"]);
        assert_eq!(result.get_option("output"), Some("out.txt"));
    }

    #[test]
    fn parse_option_short() {
        let cmd = test_cmd();
        let result = cmd.parse(&["file.txt", "-o", "out.txt"]);
        assert_eq!(result.get_option("output"), Some("out.txt"));
    }

    #[test]
    fn parse_positional() {
        let cmd = test_cmd();
        let result = cmd.parse(&["input.txt"]);
        assert_eq!(result.get_positional(0), Some("input.txt"));
    }

    #[test]
    fn parse_missing_required() {
        let cmd = test_cmd();
        let result = cmd.parse(&["--verbose"]);
        assert!(result.has_errors());
        assert!(result.errors.iter().any(|e| e.contains("input")));
    }

    #[test]
    fn parse_default_value() {
        let cmd = Command::new("app")
            .arg(Arg::option("format").default_value("json"));
        let result = cmd.parse(&[]);
        assert_eq!(result.get_option("format"), Some("json"));
    }

    #[test]
    fn parse_override_default() {
        let cmd = Command::new("app")
            .arg(Arg::option("format").default_value("json"));
        let result = cmd.parse(&["--format", "csv"]);
        assert_eq!(result.get_option("format"), Some("csv"));
    }

    #[test]
    fn parse_equals_syntax() {
        let cmd = Command::new("app")
            .arg(Arg::option("output").short('o'));
        let result = cmd.parse(&["--output=file.txt"]);
        assert_eq!(result.get_option("output"), Some("file.txt"));
    }

    #[test]
    fn parse_subcommand() {
        let cmd = Command::new("git")
            .subcommand(
                Command::new("commit")
                    .arg(Arg::option("message").short('m'))
            );
        let result = cmd.parse(&["commit", "-m", "initial"]);
        assert!(result.subcommand.is_some());
        let (name, sub) = result.subcommand.unwrap();
        assert_eq!(name, "commit");
        assert_eq!(sub.get_option("message"), Some("initial"));
    }

    #[test]
    fn parse_unknown_flag() {
        let cmd = Command::new("app");
        let result = cmd.parse(&["--unknown"]);
        assert!(result.has_errors());
    }

    #[test]
    fn parse_choices_valid() {
        let cmd = Command::new("app")
            .arg(Arg::option("color").choices(&["red", "green", "blue"]));
        let result = cmd.parse(&["--color", "red"]);
        assert!(!result.has_errors());
        assert_eq!(result.get_option("color"), Some("red"));
    }

    #[test]
    fn parse_choices_invalid() {
        let cmd = Command::new("app")
            .arg(Arg::option("color").choices(&["red", "green", "blue"]));
        let result = cmd.parse(&["--color", "purple"]);
        assert!(result.has_errors());
    }

    #[test]
    fn help_text_contains_sections() {
        let cmd = test_cmd();
        let help = cmd.help_text();
        assert!(help.contains("myapp 1.0.0"));
        assert!(help.contains("USAGE:"));
        assert!(help.contains("OPTIONS:"));
        assert!(help.contains("ARGUMENTS:"));
        assert!(help.contains("--verbose"));
        assert!(help.contains("-v"));
        assert!(help.contains("--output"));
    }

    #[test]
    fn help_flag() {
        let cmd = test_cmd();
        let result = cmd.parse(&["--help"]);
        assert!(result.get_flag("help"));
    }

    #[test]
    fn display_trait() {
        let cmd = test_cmd();
        let out = format!("{cmd}");
        assert!(out.contains("myapp"));
    }
}
