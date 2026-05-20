// SPDX-License-Identifier: MIT
//! Semantic Versioning -- parse, compare, range matching.
//!
//! Full SemVer 2.0.0: major.minor.patch, pre-release identifiers, build
//! metadata. Range operators: caret (^), tilde (~), wildcard (*), comparators
//! (>=, <=, >, <, =), hyphen ranges, AND/OR composition.

use std::cmp::Ordering;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemverError {
    InvalidVersion(String),
    InvalidRange(String),
}

impl fmt::Display for SemverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersion(s) => write!(f, "invalid version: {s}"),
            Self::InvalidRange(s) => write!(f, "invalid range: {s}"),
        }
    }
}

// ── Version ─────────────────────────────────────────────────────────────────

/// A parsed semantic version.
#[derive(Debug, Clone, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Vec<PreRelease>,
    pub build: Vec<String>,
}

/// A pre-release identifier -- either numeric or alphanumeric.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PreRelease {
    Num(u64),
    Alpha(String),
}

impl Ord for PreRelease {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (PreRelease::Num(a), PreRelease::Num(b)) => a.cmp(b),
            (PreRelease::Alpha(a), PreRelease::Alpha(b)) => a.cmp(b),
            (PreRelease::Num(_), PreRelease::Alpha(_)) => Ordering::Less,
            (PreRelease::Alpha(_), PreRelease::Num(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for PreRelease {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for PreRelease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreRelease::Num(n) => write!(f, "{n}"),
            PreRelease::Alpha(s) => write!(f, "{s}"),
        }
    }
}

impl Version {
    /// Parse a version string like "1.2.3-alpha.1+build.42".
    pub fn parse(input: &str) -> Result<Self, SemverError> {
        let input = input.trim().strip_prefix('v').unwrap_or(input.trim());
        let (version_part, build) = if let Some((vp, bp)) = input.split_once('+') {
            (vp, bp.split('.').map(String::from).collect())
        } else {
            (input, Vec::new())
        };

        let (core_part, pre) = if let Some((cp, pp)) = version_part.split_once('-') {
            let pre_ids: Vec<PreRelease> = pp
                .split('.')
                .map(|s| {
                    if let Ok(n) = s.parse::<u64>() {
                        PreRelease::Num(n)
                    } else {
                        PreRelease::Alpha(s.to_string())
                    }
                })
                .collect();
            (cp, pre_ids)
        } else {
            (version_part, Vec::new())
        };

        let parts: Vec<&str> = core_part.split('.').collect();
        if parts.len() != 3 {
            return Err(SemverError::InvalidVersion(input.to_string()));
        }
        let major = parts[0].parse().map_err(|_| SemverError::InvalidVersion(input.to_string()))?;
        let minor = parts[1].parse().map_err(|_| SemverError::InvalidVersion(input.to_string()))?;
        let patch = parts[2].parse().map_err(|_| SemverError::InvalidVersion(input.to_string()))?;

        Ok(Version { major, minor, patch, pre, build })
    }

    /// Whether this version has pre-release identifiers.
    pub fn is_prerelease(&self) -> bool {
        !self.pre.is_empty()
    }

    /// Whether this version has build metadata.
    pub fn has_build(&self) -> bool {
        !self.build.is_empty()
    }

    /// Compare versions ignoring build metadata (per spec).
    pub fn cmp_precedence(&self, other: &Version) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            o => return o,
        }
        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            o => return o,
        }
        match self.patch.cmp(&other.patch) {
            Ordering::Equal => {}
            o => return o,
        }
        // Pre-release: version without pre-release has higher precedence
        match (self.pre.is_empty(), other.pre.is_empty()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                for (a, b) in self.pre.iter().zip(&other.pre) {
                    match a.cmp(b) {
                        Ordering::Equal => continue,
                        o => return o,
                    }
                }
                self.pre.len().cmp(&other.pre.len())
            }
        }
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_precedence(other) == Ordering::Equal
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_precedence(other)
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
        if !self.pre.is_empty() {
            write!(f, "-")?;
            for (i, p) in self.pre.iter().enumerate() {
                if i > 0 { write!(f, ".")?; }
                write!(f, "{p}")?;
            }
        }
        if !self.build.is_empty() {
            write!(f, "+")?;
            for (i, b) in self.build.iter().enumerate() {
                if i > 0 { write!(f, ".")?; }
                write!(f, "{b}")?;
            }
        }
        Ok(())
    }
}

// ── Range ───────────────────────────────────────────────────────────────────

/// A version range that can test whether versions satisfy it.
#[derive(Debug, Clone)]
pub struct VersionRange {
    comparators: Vec<Vec<Comparator>>, // OR of AND groups
}

#[derive(Debug, Clone)]
struct Comparator {
    op: RangeOp,
    major: u64,
    minor: Option<u64>,
    patch: Option<u64>,
    pre: Vec<PreRelease>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RangeOp {
    Eq,
    Gt,
    Gte,
    Lt,
    Lte,
    Caret,
    Tilde,
}

impl VersionRange {
    /// Parse a range string. Supports:
    /// - Exact: `1.2.3`
    /// - Comparators: `>=1.0.0`, `<2.0.0`, `>1.0.0 <2.0.0`
    /// - Caret: `^1.2.3`
    /// - Tilde: `~1.2.3`
    /// - Wildcard: `1.2.*`, `1.*`, `*`
    /// - OR: `>=1.0.0 || <0.5.0`
    /// - Hyphen: `1.0.0 - 2.0.0`
    pub fn parse(input: &str) -> Result<Self, SemverError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(SemverError::InvalidRange("empty range".into()));
        }

        let or_parts: Vec<&str> = input.split("||").collect();
        let mut comparators = Vec::new();
        for part in or_parts {
            let part = part.trim();
            if part.is_empty() { continue; }
            let and_group = parse_and_group(part)?;
            comparators.push(and_group);
        }
        if comparators.is_empty() {
            return Err(SemverError::InvalidRange(input.into()));
        }
        Ok(VersionRange { comparators })
    }

    /// Check whether a version satisfies this range.
    pub fn satisfies(&self, version: &Version) -> bool {
        self.comparators.iter().any(|and_group| {
            and_group.iter().all(|comp| comp.matches(version))
        })
    }
}

impl fmt::Display for VersionRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, group) in self.comparators.iter().enumerate() {
            if i > 0 { write!(f, " || ")?; }
            for (j, comp) in group.iter().enumerate() {
                if j > 0 { write!(f, " ")?; }
                write!(f, "{comp}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Comparator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_str = match self.op {
            RangeOp::Eq => "=",
            RangeOp::Gt => ">",
            RangeOp::Gte => ">=",
            RangeOp::Lt => "<",
            RangeOp::Lte => "<=",
            RangeOp::Caret => "^",
            RangeOp::Tilde => "~",
        };
        write!(f, "{}{}", op_str, self.major)?;
        if let Some(min) = self.minor { write!(f, ".{min}")?; }
        if let Some(pat) = self.patch { write!(f, ".{pat}")?; }
        Ok(())
    }
}

fn parse_and_group(input: &str) -> Result<Vec<Comparator>, SemverError> {
    // Check for hyphen range: "1.0.0 - 2.0.0"
    if let Some(hyphen_idx) = input.find(" - ") {
        let lo = input[..hyphen_idx].trim();
        let hi = input[hyphen_idx + 3..].trim();
        let lo_comp = parse_single_comparator(&format!(">={lo}"))?;
        let hi_comp = parse_single_comparator(&format!("<={hi}"))?;
        return Ok(vec![lo_comp, hi_comp]);
    }

    let tokens: Vec<&str> = input.split_whitespace().collect();
    let mut comps = Vec::new();
    for token in tokens {
        comps.push(parse_single_comparator(token)?);
    }
    Ok(comps)
}

fn parse_single_comparator(input: &str) -> Result<Comparator, SemverError> {
    let input = input.trim();
    let (op, rest) = if input.starts_with(">=") {
        (RangeOp::Gte, &input[2..])
    } else if input.starts_with("<=") {
        (RangeOp::Lte, &input[2..])
    } else if input.starts_with('>') {
        (RangeOp::Gt, &input[1..])
    } else if input.starts_with('<') {
        (RangeOp::Lt, &input[1..])
    } else if input.starts_with('=') {
        (RangeOp::Eq, &input[1..])
    } else if input.starts_with('^') {
        (RangeOp::Caret, &input[1..])
    } else if input.starts_with('~') {
        (RangeOp::Tilde, &input[1..])
    } else {
        (RangeOp::Eq, input) // bare version = exact match
    };

    let rest = rest.trim().strip_prefix('v').unwrap_or(rest.trim());

    // Handle wildcard
    if rest == "*" || rest == "x" || rest == "X" {
        return Ok(Comparator { op: RangeOp::Gte, major: 0, minor: Some(0), patch: Some(0), pre: vec![] });
    }

    // Split off pre-release
    let (version_part, pre) = if let Some((vp, pp)) = rest.split_once('-') {
        let pre_ids: Vec<PreRelease> = pp
            .split('.')
            .map(|s| {
                if let Ok(n) = s.parse::<u64>() { PreRelease::Num(n) } else { PreRelease::Alpha(s.to_string()) }
            })
            .collect();
        (vp, pre_ids)
    } else {
        (rest, Vec::new())
    };

    let parts: Vec<&str> = version_part.split('.').collect();
    let major: u64 = parts[0].parse().map_err(|_| SemverError::InvalidRange(input.to_string()))?;
    let minor = if parts.len() > 1 {
        let s = parts[1];
        if s == "*" || s == "x" || s == "X" { None } else { Some(s.parse().map_err(|_| SemverError::InvalidRange(input.to_string()))?) }
    } else {
        None
    };
    let patch = if parts.len() > 2 {
        let s = parts[2];
        if s == "*" || s == "x" || s == "X" { None } else { Some(s.parse().map_err(|_| SemverError::InvalidRange(input.to_string()))?) }
    } else {
        None
    };

    Ok(Comparator { op, major, minor, patch, pre })
}

impl Comparator {
    fn matches(&self, v: &Version) -> bool {
        match self.op {
            RangeOp::Eq => self.matches_eq(v),
            RangeOp::Gt => self.to_version().map_or(false, |cv| v.cmp_precedence(&cv) == Ordering::Greater),
            RangeOp::Gte => self.to_version().map_or(false, |cv| v.cmp_precedence(&cv) != Ordering::Less),
            RangeOp::Lt => self.to_version().map_or(false, |cv| v.cmp_precedence(&cv) == Ordering::Less),
            RangeOp::Lte => self.to_version().map_or(false, |cv| v.cmp_precedence(&cv) != Ordering::Greater),
            RangeOp::Caret => self.matches_caret(v),
            RangeOp::Tilde => self.matches_tilde(v),
        }
    }

    fn matches_eq(&self, v: &Version) -> bool {
        if v.major != self.major { return false; }
        if let Some(min) = self.minor {
            if v.minor != min { return false; }
        }
        if let Some(pat) = self.patch {
            if v.patch != pat { return false; }
        }
        if !self.pre.is_empty() && self.pre != v.pre { return false; }
        true
    }

    fn matches_caret(&self, v: &Version) -> bool {
        let minor = self.minor.unwrap_or(0);
        let patch = self.patch.unwrap_or(0);
        // ^1.2.3 := >=1.2.3, <2.0.0
        // ^0.2.3 := >=0.2.3, <0.3.0
        // ^0.0.3 := >=0.0.3, <0.0.4
        let lo = Version { major: self.major, minor, patch, pre: self.pre.clone(), build: vec![] };
        if v.cmp_precedence(&lo) == Ordering::Less { return false; }

        let hi = if self.major != 0 {
            Version { major: self.major + 1, minor: 0, patch: 0, pre: vec![], build: vec![] }
        } else if minor != 0 || self.minor.is_none() {
            Version { major: 0, minor: minor + 1, patch: 0, pre: vec![], build: vec![] }
        } else {
            Version { major: 0, minor: 0, patch: patch + 1, pre: vec![], build: vec![] }
        };
        v.cmp_precedence(&hi) == Ordering::Less
    }

    fn matches_tilde(&self, v: &Version) -> bool {
        let minor = self.minor.unwrap_or(0);
        let patch = self.patch.unwrap_or(0);
        // ~1.2.3 := >=1.2.3, <1.3.0
        let lo = Version { major: self.major, minor, patch, pre: self.pre.clone(), build: vec![] };
        if v.cmp_precedence(&lo) == Ordering::Less { return false; }

        let hi = if self.minor.is_some() {
            Version { major: self.major, minor: minor + 1, patch: 0, pre: vec![], build: vec![] }
        } else {
            Version { major: self.major + 1, minor: 0, patch: 0, pre: vec![], build: vec![] }
        };
        v.cmp_precedence(&hi) == Ordering::Less
    }

    fn to_version(&self) -> Option<Version> {
        Some(Version {
            major: self.major,
            minor: self.minor.unwrap_or(0),
            patch: self.patch.unwrap_or(0),
            pre: self.pre.clone(),
            build: vec![],
        })
    }
}

/// Convenience: check if a version string satisfies a range string.
pub fn satisfies(version: &str, range: &str) -> Result<bool, SemverError> {
    let v = Version::parse(version)?;
    let r = VersionRange::parse(range)?;
    Ok(r.satisfies(&v))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.pre.is_empty());
        assert!(v.build.is_empty());
    }

    #[test]
    fn parse_with_v_prefix() {
        let v = Version::parse("v1.0.0").unwrap();
        assert_eq!(v.major, 1);
    }

    #[test]
    fn parse_prerelease() {
        let v = Version::parse("1.0.0-alpha.1").unwrap();
        assert_eq!(v.pre.len(), 2);
        assert_eq!(v.pre[0], PreRelease::Alpha("alpha".into()));
        assert_eq!(v.pre[1], PreRelease::Num(1));
        assert!(v.is_prerelease());
    }

    #[test]
    fn parse_build() {
        let v = Version::parse("1.0.0+build.42").unwrap();
        assert_eq!(v.build, vec!["build", "42"]);
        assert!(v.has_build());
    }

    #[test]
    fn parse_pre_and_build() {
        let v = Version::parse("1.0.0-beta.2+exp.sha.5114f85").unwrap();
        assert_eq!(v.pre.len(), 2);
        assert_eq!(v.build.len(), 3);
    }

    #[test]
    fn parse_invalid() {
        assert!(Version::parse("1.2").is_err());
        assert!(Version::parse("abc").is_err());
        assert!(Version::parse("").is_err());
    }

    #[test]
    fn display() {
        let v = Version::parse("1.2.3-alpha.1+build").unwrap();
        assert_eq!(format!("{v}"), "1.2.3-alpha.1+build");
    }

    #[test]
    fn display_simple() {
        let v = Version::parse("0.1.0").unwrap();
        assert_eq!(format!("{v}"), "0.1.0");
    }

    #[test]
    fn compare_basic() {
        let a = Version::parse("1.0.0").unwrap();
        let b = Version::parse("2.0.0").unwrap();
        assert!(a < b);
    }

    #[test]
    fn compare_minor() {
        let a = Version::parse("1.0.0").unwrap();
        let b = Version::parse("1.1.0").unwrap();
        assert!(a < b);
    }

    #[test]
    fn compare_patch() {
        let a = Version::parse("1.0.0").unwrap();
        let b = Version::parse("1.0.1").unwrap();
        assert!(a < b);
    }

    #[test]
    fn compare_pre_vs_release() {
        let pre = Version::parse("1.0.0-alpha").unwrap();
        let rel = Version::parse("1.0.0").unwrap();
        assert!(pre < rel);
    }

    #[test]
    fn compare_pre_identifiers() {
        let a = Version::parse("1.0.0-alpha").unwrap();
        let b = Version::parse("1.0.0-alpha.1").unwrap();
        let c = Version::parse("1.0.0-alpha.beta").unwrap();
        let d = Version::parse("1.0.0-beta").unwrap();
        let e = Version::parse("1.0.0-beta.2").unwrap();
        let f = Version::parse("1.0.0-beta.11").unwrap();
        let g = Version::parse("1.0.0-rc.1").unwrap();
        assert!(a < b);
        assert!(b < c);
        assert!(c < d);
        assert!(d < e);
        assert!(e < f);
        assert!(f < g);
    }

    #[test]
    fn compare_build_ignored() {
        let a = Version::parse("1.0.0+aaa").unwrap();
        let b = Version::parse("1.0.0+zzz").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn equality() {
        let a = Version::parse("1.2.3").unwrap();
        let b = Version::parse("1.2.3").unwrap();
        assert_eq!(a, b);
    }

    // ── Range tests ─────────────────────────────────────────────────────

    #[test]
    fn exact_range() {
        assert!(satisfies("1.2.3", "1.2.3").unwrap());
        assert!(!satisfies("1.2.4", "1.2.3").unwrap());
    }

    #[test]
    fn gte_range() {
        assert!(satisfies("1.0.0", ">=1.0.0").unwrap());
        assert!(satisfies("2.0.0", ">=1.0.0").unwrap());
        assert!(!satisfies("0.9.9", ">=1.0.0").unwrap());
    }

    #[test]
    fn lt_range() {
        assert!(satisfies("0.9.9", "<1.0.0").unwrap());
        assert!(!satisfies("1.0.0", "<1.0.0").unwrap());
    }

    #[test]
    fn gt_range() {
        assert!(satisfies("1.0.1", ">1.0.0").unwrap());
        assert!(!satisfies("1.0.0", ">1.0.0").unwrap());
    }

    #[test]
    fn lte_range() {
        assert!(satisfies("1.0.0", "<=1.0.0").unwrap());
        assert!(satisfies("0.9.0", "<=1.0.0").unwrap());
        assert!(!satisfies("1.0.1", "<=1.0.0").unwrap());
    }

    #[test]
    fn and_range() {
        assert!(satisfies("1.5.0", ">=1.0.0 <2.0.0").unwrap());
        assert!(!satisfies("2.0.0", ">=1.0.0 <2.0.0").unwrap());
        assert!(!satisfies("0.9.0", ">=1.0.0 <2.0.0").unwrap());
    }

    #[test]
    fn or_range() {
        assert!(satisfies("1.0.0", ">=2.0.0 || <=1.0.0").unwrap());
        assert!(satisfies("3.0.0", ">=2.0.0 || <=1.0.0").unwrap());
        assert!(!satisfies("1.5.0", ">=2.0.0 || <=1.0.0").unwrap());
    }

    #[test]
    fn caret_major() {
        // ^1.2.3 := >=1.2.3, <2.0.0
        assert!(satisfies("1.2.3", "^1.2.3").unwrap());
        assert!(satisfies("1.9.9", "^1.2.3").unwrap());
        assert!(!satisfies("2.0.0", "^1.2.3").unwrap());
        assert!(!satisfies("1.2.2", "^1.2.3").unwrap());
    }

    #[test]
    fn caret_zero_minor() {
        // ^0.2.3 := >=0.2.3, <0.3.0
        assert!(satisfies("0.2.3", "^0.2.3").unwrap());
        assert!(satisfies("0.2.9", "^0.2.3").unwrap());
        assert!(!satisfies("0.3.0", "^0.2.3").unwrap());
    }

    #[test]
    fn caret_zero_zero() {
        // ^0.0.3 := >=0.0.3, <0.0.4
        assert!(satisfies("0.0.3", "^0.0.3").unwrap());
        assert!(!satisfies("0.0.4", "^0.0.3").unwrap());
    }

    #[test]
    fn tilde_range() {
        // ~1.2.3 := >=1.2.3, <1.3.0
        assert!(satisfies("1.2.3", "~1.2.3").unwrap());
        assert!(satisfies("1.2.9", "~1.2.3").unwrap());
        assert!(!satisfies("1.3.0", "~1.2.3").unwrap());
    }

    #[test]
    fn tilde_minor_only() {
        // ~1.2 := >=1.2.0, <1.3.0
        assert!(satisfies("1.2.0", "~1.2").unwrap());
        assert!(satisfies("1.2.9", "~1.2").unwrap());
        assert!(!satisfies("1.3.0", "~1.2").unwrap());
    }

    #[test]
    fn wildcard_star() {
        assert!(satisfies("1.0.0", "*").unwrap());
        assert!(satisfies("999.999.999", "*").unwrap());
    }

    #[test]
    fn wildcard_partial() {
        // 1.2.* matches 1.2.x
        assert!(satisfies("1.2.0", "1.2.*").unwrap());
        assert!(satisfies("1.2.99", "1.2.*").unwrap());
        assert!(!satisfies("1.3.0", "1.2.*").unwrap());
    }

    #[test]
    fn hyphen_range() {
        assert!(satisfies("1.0.0", "1.0.0 - 2.0.0").unwrap());
        assert!(satisfies("1.5.0", "1.0.0 - 2.0.0").unwrap());
        assert!(satisfies("2.0.0", "1.0.0 - 2.0.0").unwrap());
        assert!(!satisfies("2.0.1", "1.0.0 - 2.0.0").unwrap());
        assert!(!satisfies("0.9.9", "1.0.0 - 2.0.0").unwrap());
    }

    #[test]
    fn prerelease_in_range() {
        assert!(satisfies("1.0.0-alpha.1", ">=1.0.0-alpha").unwrap());
        assert!(!satisfies("1.0.0-alpha", ">=1.0.0").unwrap());
    }

    #[test]
    fn range_parse_error() {
        assert!(VersionRange::parse("").is_err());
    }

    #[test]
    fn range_display() {
        let r = VersionRange::parse(">=1.0.0 <2.0.0").unwrap();
        let s = format!("{r}");
        assert!(s.contains(">=1.0.0"));
        assert!(s.contains("<2.0.0"));
    }

    #[test]
    fn error_display() {
        let e = SemverError::InvalidVersion("bad".into());
        assert_eq!(format!("{e}"), "invalid version: bad");
    }

    #[test]
    fn sort_versions() {
        let mut versions: Vec<Version> = vec![
            Version::parse("2.0.0").unwrap(),
            Version::parse("1.0.0").unwrap(),
            Version::parse("1.0.0-alpha").unwrap(),
            Version::parse("1.1.0").unwrap(),
        ];
        versions.sort();
        assert_eq!(versions[0].to_string(), "1.0.0-alpha");
        assert_eq!(versions[1].to_string(), "1.0.0");
        assert_eq!(versions[2].to_string(), "1.1.0");
        assert_eq!(versions[3].to_string(), "2.0.0");
    }

    #[test]
    fn caret_no_minor() {
        // ^1 := >=1.0.0, <2.0.0
        assert!(satisfies("1.0.0", "^1").unwrap());
        assert!(satisfies("1.9.9", "^1").unwrap());
        assert!(!satisfies("2.0.0", "^1").unwrap());
    }

    #[test]
    fn complex_or_and() {
        // (>=1.0.0 <1.5.0) || (>=2.0.0 <3.0.0)
        let range = ">=1.0.0 <1.5.0 || >=2.0.0 <3.0.0";
        assert!(satisfies("1.2.3", range).unwrap());
        assert!(!satisfies("1.6.0", range).unwrap());
        assert!(satisfies("2.5.0", range).unwrap());
        assert!(!satisfies("3.0.0", range).unwrap());
    }
}
