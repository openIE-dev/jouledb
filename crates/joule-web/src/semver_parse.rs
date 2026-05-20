//! Semantic versioning parser and comparator.
//!
//! Parse, compare, version ranges (^, ~, >=, <, ||), pre-release ordering,
//! build metadata, satisfies-range check, and version bumping.

use std::cmp::Ordering;
use std::fmt;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SemverError {
    #[error("invalid version string: '{0}'")]
    InvalidVersion(String),
    #[error("invalid range expression: '{0}'")]
    InvalidRange(String),
}

// ── Version ─────────────────────────────────────────────────────

/// A semantic version (major.minor.patch-pre+build).
#[derive(Debug, Clone, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre_release: Vec<PreReleasePart>,
    pub build_metadata: Option<String>,
}

/// A pre-release identifier component.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PreReleasePart {
    Numeric(u64),
    Alpha(String),
}

impl Ord for PreReleasePart {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Numeric(a), Self::Numeric(b)) => a.cmp(b),
            (Self::Alpha(a), Self::Alpha(b)) => a.cmp(b),
            (Self::Numeric(_), Self::Alpha(_)) => Ordering::Less,
            (Self::Alpha(_), Self::Numeric(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for PreReleasePart {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for PreReleasePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(n) => write!(f, "{}", n),
            Self::Alpha(s) => write!(f, "{}", s),
        }
    }
}

impl Version {
    /// Parse a version string.
    pub fn parse(input: &str) -> Result<Self, SemverError> {
        let s = input.trim().trim_start_matches('v').trim_start_matches('V');
        if s.is_empty() {
            return Err(SemverError::InvalidVersion(input.to_string()));
        }

        let (version_pre, build_meta) = if let Some(plus_pos) = s.find('+') {
            (&s[..plus_pos], Some(s[plus_pos + 1..].to_string()))
        } else {
            (s, None)
        };

        let (version_str, pre_release_str) = if let Some(dash_pos) = version_pre.find('-') {
            (&version_pre[..dash_pos], Some(&version_pre[dash_pos + 1..]))
        } else {
            (version_pre, None)
        };

        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.len() > 3 || parts.is_empty() {
            return Err(SemverError::InvalidVersion(input.to_string()));
        }

        let major = parts[0].parse::<u64>()
            .map_err(|_| SemverError::InvalidVersion(input.to_string()))?;
        let minor = if parts.len() > 1 {
            parts[1].parse::<u64>().map_err(|_| SemverError::InvalidVersion(input.to_string()))?
        } else { 0 };
        let patch = if parts.len() > 2 {
            parts[2].parse::<u64>().map_err(|_| SemverError::InvalidVersion(input.to_string()))?
        } else { 0 };

        let pre_release = if let Some(pre_str) = pre_release_str {
            pre_str.split('.').map(|p| {
                if let Ok(n) = p.parse::<u64>() {
                    PreReleasePart::Numeric(n)
                } else {
                    PreReleasePart::Alpha(p.to_string())
                }
            }).collect()
        } else {
            Vec::new()
        };

        Ok(Self { major, minor, patch, pre_release, build_metadata: build_meta })
    }

    /// Create a version from major.minor.patch components.
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self { major, minor, patch, pre_release: Vec::new(), build_metadata: None }
    }

    /// Is this a pre-release version?
    pub fn is_pre_release(&self) -> bool {
        !self.pre_release.is_empty()
    }

    /// Bump the major version (minor and patch reset to 0).
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Bump the minor version (patch resets to 0).
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Bump the patch version.
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Return version string without build metadata.
    pub fn to_string_no_build(&self) -> String {
        let mut s = format!("{}.{}.{}", self.major, self.minor, self.patch);
        if !self.pre_release.is_empty() {
            s.push('-');
            let parts: Vec<String> = self.pre_release.iter().map(|p| p.to_string()).collect();
            s.push_str(&parts.join("."));
        }
        s
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        // Build metadata is ignored for equality per semver spec.
        self.major == other.major
            && self.minor == other.minor
            && self.patch == other.patch
            && self.pre_release == other.pre_release
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        // Major, minor, patch comparison
        let core = self.major.cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch));
        if core != Ordering::Equal { return core; }

        // Pre-release: version without pre-release has HIGHER precedence
        match (self.pre_release.is_empty(), other.pre_release.is_empty()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                for (a, b) in self.pre_release.iter().zip(other.pre_release.iter()) {
                    let c = a.cmp(b);
                    if c != Ordering::Equal { return c; }
                }
                self.pre_release.len().cmp(&other.pre_release.len())
            }
        }
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre_release.is_empty() {
            write!(f, "-")?;
            for (i, part) in self.pre_release.iter().enumerate() {
                if i > 0 { write!(f, ".")?; }
                write!(f, "{}", part)?;
            }
        }
        if let Some(build) = &self.build_metadata {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

// ── Range ───────────────────────────────────────────────────────

/// A version range, supporting ^, ~, >=, <=, <, >, =, and || (union).
#[derive(Debug, Clone)]
pub struct VersionRange {
    constraints: Vec<Vec<Constraint>>,
}

#[derive(Debug, Clone)]
enum Constraint {
    Gte(Version),
    Lte(Version),
    Gt(Version),
    Lt(Version),
    Eq(Version),
}

impl Constraint {
    fn matches(&self, v: &Version) -> bool {
        match self {
            Self::Gte(c) => v >= c,
            Self::Lte(c) => v <= c,
            Self::Gt(c) => v > c,
            Self::Lt(c) => v < c,
            Self::Eq(c) => v == c,
        }
    }
}

impl VersionRange {
    /// Parse a version range expression.
    ///
    /// Supports: `^1.2.3`, `~1.2.3`, `>=1.0.0`, `<2.0.0`, `1.2.3`, `>=1.0.0 <2.0.0`, `^1 || ^2`.
    pub fn parse(input: &str) -> Result<Self, SemverError> {
        let or_groups: Vec<&str> = input.split("||").collect();
        let mut constraints = Vec::new();

        for group in or_groups {
            let group = group.trim();
            if group.is_empty() { continue; }
            let parts = split_range_parts(group);
            let mut group_constraints = Vec::new();

            for part in parts {
                let part = part.trim();
                if part.is_empty() { continue; }
                let cs = parse_single_range(part)?;
                group_constraints.extend(cs);
            }

            if !group_constraints.is_empty() {
                constraints.push(group_constraints);
            }
        }

        Ok(Self { constraints })
    }

    /// Check if a version satisfies this range.
    pub fn satisfies(&self, version: &Version) -> bool {
        if self.constraints.is_empty() { return true; }
        self.constraints.iter().any(|group| {
            group.iter().all(|c| c.matches(version))
        })
    }
}

fn split_range_parts(group: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = group.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            i += 1;
            continue;
        }
        current.push(chars[i]);
        i += 1;
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn parse_single_range(part: &str) -> Result<Vec<Constraint>, SemverError> {
    if let Some(rest) = part.strip_prefix("^") {
        let v = Version::parse(rest)?;
        // Caret: >=version, <next major (or minor if major==0)
        let upper = if v.major == 0 {
            if v.minor == 0 {
                Version::new(0, 0, v.patch + 1)
            } else {
                Version::new(0, v.minor + 1, 0)
            }
        } else {
            Version::new(v.major + 1, 0, 0)
        };
        Ok(vec![Constraint::Gte(v), Constraint::Lt(upper)])
    } else if let Some(rest) = part.strip_prefix("~") {
        let v = Version::parse(rest)?;
        // Tilde: >=version, <next minor
        let upper = Version::new(v.major, v.minor + 1, 0);
        Ok(vec![Constraint::Gte(v), Constraint::Lt(upper)])
    } else if let Some(rest) = part.strip_prefix(">=") {
        let v = Version::parse(rest)?;
        Ok(vec![Constraint::Gte(v)])
    } else if let Some(rest) = part.strip_prefix("<=") {
        let v = Version::parse(rest)?;
        Ok(vec![Constraint::Lte(v)])
    } else if let Some(rest) = part.strip_prefix(">") {
        let v = Version::parse(rest)?;
        Ok(vec![Constraint::Gt(v)])
    } else if let Some(rest) = part.strip_prefix("<") {
        let v = Version::parse(rest)?;
        Ok(vec![Constraint::Lt(v)])
    } else if let Some(rest) = part.strip_prefix("=") {
        let v = Version::parse(rest)?;
        Ok(vec![Constraint::Eq(v)])
    } else {
        let v = Version::parse(part)?;
        Ok(vec![Constraint::Eq(v)])
    }
}

impl fmt::Display for VersionRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let groups: Vec<String> = self.constraints.iter().map(|g| {
            let parts: Vec<String> = g.iter().map(|c| match c {
                Constraint::Gte(v) => format!(">={}", v),
                Constraint::Lte(v) => format!("<={}", v),
                Constraint::Gt(v) => format!(">{}", v),
                Constraint::Lt(v) => format!("<{}", v),
                Constraint::Eq(v) => format!("={}", v),
            }).collect();
            parts.join(" ")
        }).collect();
        write!(f, "{}", groups.join(" || "))
    }
}

// ── Utility Functions ───────────────────────────────────────────

/// Find the maximum version satisfying a range from a list.
pub fn max_satisfying(versions: &[Version], range: &VersionRange) -> Option<Version> {
    versions.iter()
        .filter(|v| range.satisfies(v))
        .max()
        .cloned()
}

/// Find the minimum version satisfying a range from a list.
pub fn min_satisfying(versions: &[Version], range: &VersionRange) -> Option<Version> {
    versions.iter()
        .filter(|v| range.satisfies(v))
        .min()
        .cloned()
}

/// Sort versions in ascending order.
pub fn sort_versions(versions: &mut [Version]) {
    versions.sort();
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_parse_with_v_prefix() {
        let v = Version::parse("v2.0.0").unwrap();
        assert_eq!(v.major, 2);
    }

    #[test]
    fn test_parse_pre_release() {
        let v = Version::parse("1.0.0-alpha.1").unwrap();
        assert!(v.is_pre_release());
        assert_eq!(v.pre_release.len(), 2);
    }

    #[test]
    fn test_parse_build_metadata() {
        let v = Version::parse("1.0.0+build.42").unwrap();
        assert_eq!(v.build_metadata.as_deref(), Some("build.42"));
    }

    #[test]
    fn test_display() {
        let v = Version::parse("1.2.3-alpha.1+build").unwrap();
        assert_eq!(v.to_string(), "1.2.3-alpha.1+build");
    }

    #[test]
    fn test_comparison_basic() {
        let a = Version::parse("1.0.0").unwrap();
        let b = Version::parse("2.0.0").unwrap();
        assert!(a < b);
    }

    #[test]
    fn test_comparison_minor() {
        let a = Version::parse("1.1.0").unwrap();
        let b = Version::parse("1.2.0").unwrap();
        assert!(a < b);
    }

    #[test]
    fn test_comparison_patch() {
        let a = Version::parse("1.0.1").unwrap();
        let b = Version::parse("1.0.2").unwrap();
        assert!(a < b);
    }

    #[test]
    fn test_pre_release_lower_than_release() {
        let pre = Version::parse("1.0.0-alpha").unwrap();
        let rel = Version::parse("1.0.0").unwrap();
        assert!(pre < rel);
    }

    #[test]
    fn test_pre_release_ordering() {
        let a = Version::parse("1.0.0-alpha").unwrap();
        let b = Version::parse("1.0.0-alpha.1").unwrap();
        let c = Version::parse("1.0.0-beta").unwrap();
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_equality_ignores_build() {
        let a = Version::parse("1.0.0+build1").unwrap();
        let b = Version::parse("1.0.0+build2").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_bump_major() {
        let v = Version::parse("1.2.3").unwrap().bump_major();
        assert_eq!(v, Version::new(2, 0, 0));
    }

    #[test]
    fn test_bump_minor() {
        let v = Version::parse("1.2.3").unwrap().bump_minor();
        assert_eq!(v, Version::new(1, 3, 0));
    }

    #[test]
    fn test_bump_patch() {
        let v = Version::parse("1.2.3").unwrap().bump_patch();
        assert_eq!(v, Version::new(1, 2, 4));
    }

    #[test]
    fn test_caret_range() {
        let range = VersionRange::parse("^1.2.3").unwrap();
        assert!(range.satisfies(&Version::parse("1.2.3").unwrap()));
        assert!(range.satisfies(&Version::parse("1.9.0").unwrap()));
        assert!(!range.satisfies(&Version::parse("2.0.0").unwrap()));
        assert!(!range.satisfies(&Version::parse("1.2.2").unwrap()));
    }

    #[test]
    fn test_tilde_range() {
        let range = VersionRange::parse("~1.2.3").unwrap();
        assert!(range.satisfies(&Version::parse("1.2.3").unwrap()));
        assert!(range.satisfies(&Version::parse("1.2.9").unwrap()));
        assert!(!range.satisfies(&Version::parse("1.3.0").unwrap()));
    }

    #[test]
    fn test_gte_lt_range() {
        let range = VersionRange::parse(">=1.0.0 <2.0.0").unwrap();
        assert!(range.satisfies(&Version::parse("1.5.0").unwrap()));
        assert!(!range.satisfies(&Version::parse("0.9.0").unwrap()));
        assert!(!range.satisfies(&Version::parse("2.0.0").unwrap()));
    }

    #[test]
    fn test_or_range() {
        let range = VersionRange::parse("^1.0.0 || ^2.0.0").unwrap();
        assert!(range.satisfies(&Version::parse("1.5.0").unwrap()));
        assert!(range.satisfies(&Version::parse("2.5.0").unwrap()));
        assert!(!range.satisfies(&Version::parse("3.0.0").unwrap()));
    }

    #[test]
    fn test_exact_range() {
        let range = VersionRange::parse("1.2.3").unwrap();
        assert!(range.satisfies(&Version::parse("1.2.3").unwrap()));
        assert!(!range.satisfies(&Version::parse("1.2.4").unwrap()));
    }

    #[test]
    fn test_max_satisfying() {
        let versions: Vec<Version> = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]
            .iter().map(|s| Version::parse(s).unwrap()).collect();
        let range = VersionRange::parse("^1.0.0").unwrap();
        let max = max_satisfying(&versions, &range).unwrap();
        assert_eq!(max, Version::parse("1.2.0").unwrap());
    }

    #[test]
    fn test_min_satisfying() {
        let versions: Vec<Version> = ["1.0.0", "1.1.0", "2.0.0"]
            .iter().map(|s| Version::parse(s).unwrap()).collect();
        let range = VersionRange::parse(">=1.1.0").unwrap();
        let min = min_satisfying(&versions, &range).unwrap();
        assert_eq!(min, Version::parse("1.1.0").unwrap());
    }

    #[test]
    fn test_sort() {
        let mut versions: Vec<Version> = ["2.0.0", "1.0.0", "1.1.0", "1.0.0-alpha"]
            .iter().map(|s| Version::parse(s).unwrap()).collect();
        sort_versions(&mut versions);
        assert_eq!(versions[0], Version::parse("1.0.0-alpha").unwrap());
        assert_eq!(versions[1], Version::parse("1.0.0").unwrap());
        assert_eq!(versions[2], Version::parse("1.1.0").unwrap());
        assert_eq!(versions[3], Version::parse("2.0.0").unwrap());
    }

    #[test]
    fn test_invalid_version() {
        assert!(Version::parse("not.a.version").is_err());
        assert!(Version::parse("").is_err());
    }

    #[test]
    fn test_caret_zero_major() {
        let range = VersionRange::parse("^0.2.3").unwrap();
        assert!(range.satisfies(&Version::parse("0.2.5").unwrap()));
        assert!(!range.satisfies(&Version::parse("0.3.0").unwrap()));
    }

    #[test]
    fn test_to_string_no_build() {
        let v = Version::parse("1.0.0-beta+build").unwrap();
        assert_eq!(v.to_string_no_build(), "1.0.0-beta");
    }
}
