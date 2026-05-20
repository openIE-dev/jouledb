//! Changelog generator.
//!
//! Parses conventional commits (feat/fix/chore/etc), groups by version,
//! detects breaking changes, generates markdown output, suggests version
//! bumps, and categorizes commits. Pure Rust — no git or external deps.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from changelog operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangelogError {
    /// Failed to parse a commit message.
    ParseError(String),
    /// Invalid version string.
    InvalidVersion(String),
    /// No commits to process.
    NoCommits,
}

impl fmt::Display for ChangelogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "commit parse error: {msg}"),
            Self::InvalidVersion(v) => write!(f, "invalid version: {v}"),
            Self::NoCommits => write!(f, "no commits to process"),
        }
    }
}

impl std::error::Error for ChangelogError {}

// ── Commit Type ─────────────────────────────────────────────────

/// Conventional commit type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommitType {
    /// New feature.
    Feat,
    /// Bug fix.
    Fix,
    /// Documentation.
    Docs,
    /// Code style (formatting, no logic change).
    Style,
    /// Refactoring (no feature or fix).
    Refactor,
    /// Performance improvement.
    Perf,
    /// Tests.
    Test,
    /// Build system or external dependencies.
    Build,
    /// CI configuration.
    Ci,
    /// Chores (maintenance).
    Chore,
    /// Reverting a previous commit.
    Revert,
    /// Unknown/custom type.
    Other(String),
}

impl CommitType {
    /// Parse from a string prefix.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "feat" | "feature" => Self::Feat,
            "fix" | "bugfix" => Self::Fix,
            "docs" | "doc" => Self::Docs,
            "style" => Self::Style,
            "refactor" => Self::Refactor,
            "perf" | "performance" => Self::Perf,
            "test" | "tests" => Self::Test,
            "build" => Self::Build,
            "ci" => Self::Ci,
            "chore" => Self::Chore,
            "revert" => Self::Revert,
            other => Self::Other(other.to_string()),
        }
    }

    /// Heading for the changelog section.
    pub fn heading(&self) -> &str {
        match self {
            Self::Feat => "Features",
            Self::Fix => "Bug Fixes",
            Self::Docs => "Documentation",
            Self::Style => "Styles",
            Self::Refactor => "Code Refactoring",
            Self::Perf => "Performance Improvements",
            Self::Test => "Tests",
            Self::Build => "Build System",
            Self::Ci => "Continuous Integration",
            Self::Chore => "Chores",
            Self::Revert => "Reverts",
            Self::Other(_) => "Other Changes",
        }
    }

    /// Display order priority (lower = shown first in changelog).
    pub fn display_order(&self) -> u8 {
        match self {
            Self::Feat => 0,
            Self::Fix => 1,
            Self::Perf => 2,
            Self::Refactor => 3,
            Self::Docs => 4,
            Self::Test => 5,
            Self::Build => 6,
            Self::Ci => 7,
            Self::Style => 8,
            Self::Chore => 9,
            Self::Revert => 10,
            Self::Other(_) => 11,
        }
    }
}

impl fmt::Display for CommitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Feat => write!(f, "feat"),
            Self::Fix => write!(f, "fix"),
            Self::Docs => write!(f, "docs"),
            Self::Style => write!(f, "style"),
            Self::Refactor => write!(f, "refactor"),
            Self::Perf => write!(f, "perf"),
            Self::Test => write!(f, "test"),
            Self::Build => write!(f, "build"),
            Self::Ci => write!(f, "ci"),
            Self::Chore => write!(f, "chore"),
            Self::Revert => write!(f, "revert"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

// ── Parsed Commit ───────────────────────────────────────────────

/// A parsed conventional commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConventionalCommit {
    /// Commit type (feat, fix, etc).
    pub commit_type: CommitType,
    /// Optional scope (e.g., "parser").
    pub scope: Option<String>,
    /// Short description.
    pub description: String,
    /// Body text (after blank line).
    pub body: Option<String>,
    /// Footer entries (e.g., "BREAKING CHANGE: ...").
    pub footers: Vec<Footer>,
    /// Whether this is a breaking change.
    pub breaking: bool,
    /// Optional commit hash (short or full).
    pub hash: Option<String>,
    /// Optional author.
    pub author: Option<String>,
    /// Optional date string.
    pub date: Option<String>,
}

/// A footer entry in a conventional commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Footer {
    pub key: String,
    pub value: String,
}

/// Parse a conventional commit message.
///
/// Format: `type(scope)!: description`
///
/// Where `!` indicates breaking change, scope is optional.
pub fn parse_commit(message: &str) -> Result<ConventionalCommit, ChangelogError> {
    let lines: Vec<&str> = message.lines().collect();
    if lines.is_empty() {
        return Err(ChangelogError::ParseError("empty message".to_string()));
    }

    let first_line = lines[0].trim();
    if first_line.is_empty() {
        return Err(ChangelogError::ParseError("empty first line".to_string()));
    }

    // Find the colon separator.
    let colon_pos = first_line
        .find(':')
        .ok_or_else(|| ChangelogError::ParseError(format!("no colon in: {first_line}")))?;

    let prefix = &first_line[..colon_pos];
    let description = first_line[colon_pos + 1..].trim().to_string();

    if description.is_empty() {
        return Err(ChangelogError::ParseError(
            "empty description after colon".to_string(),
        ));
    }

    // Parse prefix: type(scope)! or type! or type(scope) or type
    let mut breaking = prefix.ends_with('!');
    let prefix_clean = prefix.trim_end_matches('!');

    let (type_str, scope) = if let Some(paren_start) = prefix_clean.find('(') {
        let paren_end = prefix_clean.find(')').ok_or_else(|| {
            ChangelogError::ParseError("unclosed parenthesis in scope".to_string())
        })?;
        let type_part = &prefix_clean[..paren_start];
        let scope_part = &prefix_clean[paren_start + 1..paren_end];
        (type_part, Some(scope_part.to_string()))
    } else {
        (prefix_clean, None)
    };

    let commit_type = CommitType::parse(type_str);

    // Parse body and footers.
    let mut body_lines = Vec::new();
    let mut footers = Vec::new();
    let mut in_body = false;
    let mut past_blank = false;

    for line in &lines[1..] {
        if line.is_empty() {
            if in_body {
                past_blank = true;
            }
            in_body = true;
            continue;
        }
        if in_body {
            // Check for footer (KEY: value or KEY #value or BREAKING CHANGE: ...).
            if past_blank || line.starts_with("BREAKING CHANGE") {
                if let Some(colon) = line.find(':') {
                    let key = line[..colon].trim();
                    let value = line[colon + 1..].trim();
                    if key == "BREAKING CHANGE" || key == "BREAKING-CHANGE" {
                        breaking = true;
                    }
                    footers.push(Footer {
                        key: key.to_string(),
                        value: value.to_string(),
                    });
                    continue;
                }
            }
            body_lines.push(*line);
        }
    }

    let body = if body_lines.is_empty() {
        None
    } else {
        Some(body_lines.join("\n"))
    };

    Ok(ConventionalCommit {
        commit_type,
        scope,
        description,
        body,
        footers,
        breaking,
        hash: None,
        author: None,
        date: None,
    })
}

// ── Semantic Version ────────────────────────────────────────────

/// A semantic version (major.minor.patch).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
}

impl SemVer {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
        }
    }

    /// Parse from a version string (optional leading 'v').
    pub fn parse(s: &str) -> Result<Self, ChangelogError> {
        let s = s.trim().strip_prefix('v').unwrap_or(s.trim());

        let (version_part, pre_release) = if let Some(hyphen) = s.find('-') {
            (&s[..hyphen], Some(s[hyphen + 1..].to_string()))
        } else {
            (s, None)
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() != 3 {
            return Err(ChangelogError::InvalidVersion(s.to_string()));
        }

        let major = parts[0]
            .parse::<u32>()
            .map_err(|_| ChangelogError::InvalidVersion(s.to_string()))?;
        let minor = parts[1]
            .parse::<u32>()
            .map_err(|_| ChangelogError::InvalidVersion(s.to_string()))?;
        let patch = parts[2]
            .parse::<u32>()
            .map_err(|_| ChangelogError::InvalidVersion(s.to_string()))?;

        Ok(Self {
            major,
            minor,
            patch,
            pre_release,
        })
    }

    /// Bump the major version.
    pub fn bump_major(&self) -> Self {
        Self {
            major: self.major + 1,
            minor: 0,
            patch: 0,
            pre_release: None,
        }
    }

    /// Bump the minor version.
    pub fn bump_minor(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor + 1,
            patch: 0,
            pre_release: None,
        }
    }

    /// Bump the patch version.
    pub fn bump_patch(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor,
            patch: self.patch + 1,
            pre_release: None,
        }
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{pre}")?;
        }
        Ok(())
    }
}

// ── Version Bump Suggestion ─────────────────────────────────────

/// Suggested version bump based on commit types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BumpKind {
    Major,
    Minor,
    Patch,
    None,
}

impl fmt::Display for BumpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Major => write!(f, "major"),
            Self::Minor => write!(f, "minor"),
            Self::Patch => write!(f, "patch"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Determine the version bump based on a set of commits.
pub fn suggest_bump(commits: &[ConventionalCommit]) -> BumpKind {
    if commits.is_empty() {
        return BumpKind::None;
    }

    // Breaking change = major.
    if commits.iter().any(|c| c.breaking) {
        return BumpKind::Major;
    }

    // Feature = minor.
    if commits.iter().any(|c| c.commit_type == CommitType::Feat) {
        return BumpKind::Minor;
    }

    // Fix or perf = patch.
    if commits
        .iter()
        .any(|c| c.commit_type == CommitType::Fix || c.commit_type == CommitType::Perf)
    {
        return BumpKind::Patch;
    }

    BumpKind::None
}

// ── Version Group ───────────────────────────────────────────────

/// A group of commits under a version heading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionGroup {
    pub version: String,
    pub date: Option<String>,
    pub commits: Vec<ConventionalCommit>,
}

impl VersionGroup {
    /// Group commits by their type.
    pub fn by_type(&self) -> Vec<(CommitType, Vec<&ConventionalCommit>)> {
        let mut map: HashMap<CommitType, Vec<&ConventionalCommit>> = HashMap::new();
        for c in &self.commits {
            map.entry(c.commit_type.clone())
                .or_default()
                .push(c);
        }
        let mut groups: Vec<(CommitType, Vec<&ConventionalCommit>)> = map.into_iter().collect();
        groups.sort_by_key(|(t, _)| t.display_order());
        groups
    }

    /// Get breaking changes.
    pub fn breaking_changes(&self) -> Vec<&ConventionalCommit> {
        self.commits.iter().filter(|c| c.breaking).collect()
    }
}

// ── Changelog ───────────────────────────────────────────────────

/// A complete changelog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Changelog {
    pub title: String,
    pub description: Option<String>,
    pub versions: Vec<VersionGroup>,
}

impl Changelog {
    /// Create a new changelog.
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            description: None,
            versions: Vec::new(),
        }
    }

    /// Add a version group.
    pub fn add_version(&mut self, group: VersionGroup) {
        self.versions.push(group);
    }

    /// Total number of commits across all versions.
    pub fn total_commits(&self) -> usize {
        self.versions.iter().map(|v| v.commits.len()).sum()
    }

    /// Render the changelog as markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# {}\n\n", self.title));

        if let Some(desc) = &self.description {
            out.push_str(&format!("{desc}\n\n"));
        }

        for version in &self.versions {
            // Version heading.
            let date_part = version
                .date
                .as_deref()
                .map_or(String::new(), |d| format!(" ({d})"));
            out.push_str(&format!("## {}{}\n\n", version.version, date_part));

            // Breaking changes section.
            let breaking = version.breaking_changes();
            if !breaking.is_empty() {
                out.push_str("### BREAKING CHANGES\n\n");
                for commit in &breaking {
                    let scope_part = commit
                        .scope
                        .as_deref()
                        .map_or(String::new(), |s| format!("**{s}:** "));
                    out.push_str(&format!("- {}{}\n", scope_part, commit.description));
                }
                out.push('\n');
            }

            // Group by type.
            for (commit_type, commits) in version.by_type() {
                out.push_str(&format!("### {}\n\n", commit_type.heading()));
                for commit in &commits {
                    let scope_part = commit
                        .scope
                        .as_deref()
                        .map_or(String::new(), |s| format!("**{s}:** "));
                    let hash_part = commit
                        .hash
                        .as_deref()
                        .map_or(String::new(), |h| format!(" ({h})"));
                    out.push_str(&format!(
                        "- {}{}{}\n",
                        scope_part, commit.description, hash_part
                    ));
                }
                out.push('\n');
            }
        }

        out
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Convenience builder for a changelog from raw commit messages.
pub struct ChangelogBuilder {
    title: String,
    commits: Vec<(String, String)>, // (version_tag, message)
}

impl ChangelogBuilder {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            commits: Vec::new(),
        }
    }

    /// Add a commit with its version tag.
    pub fn add_commit(&mut self, version: &str, message: &str) {
        self.commits
            .push((version.to_string(), message.to_string()));
    }

    /// Build the changelog.
    pub fn build(&self) -> Result<Changelog, ChangelogError> {
        if self.commits.is_empty() {
            return Err(ChangelogError::NoCommits);
        }

        let mut version_map: HashMap<String, Vec<ConventionalCommit>> = HashMap::new();

        for (version, message) in &self.commits {
            let commit = parse_commit(message)?;
            version_map
                .entry(version.clone())
                .or_default()
                .push(commit);
        }

        let mut changelog = Changelog::new(&self.title);

        // Sort versions (by SemVer if possible, else lexically).
        let mut versions: Vec<String> = version_map.keys().cloned().collect();
        versions.sort_by(|a, b| {
            let av = SemVer::parse(a);
            let bv = SemVer::parse(b);
            match (av, bv) {
                (Ok(av), Ok(bv)) => bv.cmp(&av), // newest first
                _ => b.cmp(a),
            }
        });

        for version in versions {
            let commits = version_map.remove(&version).unwrap_or_default();
            changelog.add_version(VersionGroup {
                version,
                date: None,
                commits,
            });
        }

        Ok(changelog)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_feat() {
        let commit = parse_commit("feat: add new login page").unwrap();
        assert_eq!(commit.commit_type, CommitType::Feat);
        assert_eq!(commit.description, "add new login page");
        assert!(commit.scope.is_none());
        assert!(!commit.breaking);
    }

    #[test]
    fn parse_with_scope() {
        let commit = parse_commit("fix(parser): handle edge case").unwrap();
        assert_eq!(commit.commit_type, CommitType::Fix);
        assert_eq!(commit.scope, Some("parser".to_string()));
        assert_eq!(commit.description, "handle edge case");
    }

    #[test]
    fn parse_breaking_bang() {
        let commit = parse_commit("feat!: remove deprecated API").unwrap();
        assert!(commit.breaking);
    }

    #[test]
    fn parse_breaking_footer() {
        let msg = "feat: new auth system\n\nBREAKING CHANGE: old tokens are invalidated";
        let commit = parse_commit(msg).unwrap();
        assert!(commit.breaking);
        assert_eq!(commit.footers.len(), 1);
        assert_eq!(commit.footers[0].key, "BREAKING CHANGE");
    }

    #[test]
    fn parse_with_body() {
        let msg = "fix: correct calculation\n\nThe previous formula was wrong.\nThis fixes it.";
        let commit = parse_commit(msg).unwrap();
        assert!(commit.body.is_some());
        assert!(commit.body.unwrap().contains("previous formula"));
    }

    #[test]
    fn parse_error_no_colon() {
        let err = parse_commit("no colon here").unwrap_err();
        assert!(matches!(err, ChangelogError::ParseError(_)));
    }

    #[test]
    fn parse_error_empty() {
        let err = parse_commit("").unwrap_err();
        assert!(matches!(err, ChangelogError::ParseError(_)));
    }

    #[test]
    fn parse_error_empty_description() {
        let err = parse_commit("feat:").unwrap_err();
        assert!(matches!(err, ChangelogError::ParseError(_)));
    }

    #[test]
    fn commit_type_parse() {
        assert_eq!(CommitType::parse("feat"), CommitType::Feat);
        assert_eq!(CommitType::parse("fix"), CommitType::Fix);
        assert_eq!(CommitType::parse("docs"), CommitType::Docs);
        assert_eq!(CommitType::parse("chore"), CommitType::Chore);
        assert_eq!(
            CommitType::parse("custom"),
            CommitType::Other("custom".to_string())
        );
    }

    #[test]
    fn commit_type_heading() {
        assert_eq!(CommitType::Feat.heading(), "Features");
        assert_eq!(CommitType::Fix.heading(), "Bug Fixes");
    }

    #[test]
    fn semver_parse() {
        let v = SemVer::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn semver_parse_with_v() {
        let v = SemVer::parse("v2.0.0").unwrap();
        assert_eq!(v.major, 2);
    }

    #[test]
    fn semver_parse_prerelease() {
        let v = SemVer::parse("1.0.0-beta.1").unwrap();
        assert_eq!(v.pre_release, Some("beta.1".to_string()));
    }

    #[test]
    fn semver_bump() {
        let v = SemVer::new(1, 2, 3);
        assert_eq!(v.bump_major(), SemVer::new(2, 0, 0));
        assert_eq!(v.bump_minor(), SemVer::new(1, 3, 0));
        assert_eq!(v.bump_patch(), SemVer::new(1, 2, 4));
    }

    #[test]
    fn semver_display() {
        let v = SemVer::new(1, 2, 3);
        assert_eq!(format!("{v}"), "1.2.3");
        let v2 = SemVer {
            pre_release: Some("rc.1".to_string()),
            ..v
        };
        assert_eq!(format!("{v2}"), "1.2.3-rc.1");
    }

    #[test]
    fn semver_invalid() {
        assert!(SemVer::parse("not a version").is_err());
        assert!(SemVer::parse("1.2").is_err());
    }

    #[test]
    fn suggest_bump_breaking() {
        let commits = vec![parse_commit("feat!: breaking change").unwrap()];
        assert_eq!(suggest_bump(&commits), BumpKind::Major);
    }

    #[test]
    fn suggest_bump_feature() {
        let commits = vec![parse_commit("feat: new feature").unwrap()];
        assert_eq!(suggest_bump(&commits), BumpKind::Minor);
    }

    #[test]
    fn suggest_bump_fix() {
        let commits = vec![parse_commit("fix: bug fix").unwrap()];
        assert_eq!(suggest_bump(&commits), BumpKind::Patch);
    }

    #[test]
    fn suggest_bump_none() {
        let commits = vec![parse_commit("chore: update deps").unwrap()];
        assert_eq!(suggest_bump(&commits), BumpKind::None);
    }

    #[test]
    fn suggest_bump_empty() {
        assert_eq!(suggest_bump(&[]), BumpKind::None);
    }

    #[test]
    fn changelog_markdown() {
        let mut cl = Changelog::new("Changelog");
        let c1 = parse_commit("feat(auth): add OAuth2 support").unwrap();
        let c2 = parse_commit("fix: correct typo").unwrap();
        cl.add_version(VersionGroup {
            version: "1.1.0".to_string(),
            date: Some("2026-03-09".to_string()),
            commits: vec![c1, c2],
        });
        let md = cl.to_markdown();
        assert!(md.contains("# Changelog"));
        assert!(md.contains("## 1.1.0"));
        assert!(md.contains("2026-03-09"));
        assert!(md.contains("Features"));
        assert!(md.contains("Bug Fixes"));
        assert!(md.contains("**auth:**"));
    }

    #[test]
    fn changelog_breaking_section() {
        let mut cl = Changelog::new("Changelog");
        let c = parse_commit("feat!: remove deprecated API").unwrap();
        cl.add_version(VersionGroup {
            version: "2.0.0".to_string(),
            date: None,
            commits: vec![c],
        });
        let md = cl.to_markdown();
        assert!(md.contains("BREAKING CHANGES"));
    }

    #[test]
    fn changelog_builder() {
        let mut builder = ChangelogBuilder::new("My App");
        builder.add_commit("1.0.0", "feat: initial release");
        builder.add_commit("1.0.0", "fix: typo");
        builder.add_commit("1.1.0", "feat: new feature");
        let cl = builder.build().unwrap();
        assert_eq!(cl.versions.len(), 2);
        assert_eq!(cl.total_commits(), 3);
    }

    #[test]
    fn changelog_builder_no_commits() {
        let builder = ChangelogBuilder::new("Empty");
        let err = builder.build().unwrap_err();
        assert_eq!(err, ChangelogError::NoCommits);
    }

    #[test]
    fn version_group_by_type() {
        let c1 = parse_commit("feat: feature A").unwrap();
        let c2 = parse_commit("fix: fix B").unwrap();
        let c3 = parse_commit("feat: feature C").unwrap();
        let group = VersionGroup {
            version: "1.0.0".to_string(),
            date: None,
            commits: vec![c1, c2, c3],
        };
        let by_type = group.by_type();
        // feat should come before fix (lower display_order).
        let first_type = &by_type[0].0;
        assert_eq!(*first_type, CommitType::Feat);
    }

    #[test]
    fn bump_kind_display() {
        assert_eq!(format!("{}", BumpKind::Major), "major");
        assert_eq!(format!("{}", BumpKind::None), "none");
    }

    #[test]
    fn error_display() {
        let e = ChangelogError::InvalidVersion("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn commit_with_hash() {
        let mut commit = parse_commit("feat: something").unwrap();
        commit.hash = Some("abc1234".to_string());
        let mut cl = Changelog::new("Test");
        cl.add_version(VersionGroup {
            version: "1.0.0".to_string(),
            date: None,
            commits: vec![commit],
        });
        let md = cl.to_markdown();
        assert!(md.contains("abc1234"));
    }
}
