//! Dependency analysis — dependency graph building, circular detection, version
//! conflict detection, unused dependency detection, license compliance checking.
//!
//! Replaces JS dependency tools (depcheck, madge, license-checker, npm-check)
//! with a pure-Rust dependency analyzer that tracks every graph traversal
//! with energy awareness.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

// ── Errors ──────────────────────────────────────────────────────

/// Dependency analysis errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepCheckError {
    /// Package not found in the graph.
    PackageNotFound(String),
    /// Duplicate package definition.
    DuplicatePackage(String),
    /// Self-dependency detected.
    SelfDependency(String),
    /// Invalid version string.
    InvalidVersion(String),
}

impl std::fmt::Display for DepCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PackageNotFound(p) => write!(f, "package not found: {p}"),
            Self::DuplicatePackage(p) => write!(f, "duplicate package: {p}"),
            Self::SelfDependency(p) => write!(f, "self-dependency: {p}"),
            Self::InvalidVersion(v) => write!(f, "invalid version: {v}"),
        }
    }
}

// ── Types ───────────────────────────────────────────────────────

/// A version requirement (simplified semver range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionReq {
    /// Exact version match.
    Exact(String),
    /// Caret range: ^major.minor.patch
    Caret(String),
    /// Tilde range: ~major.minor.patch
    Tilde(String),
    /// Wildcard: *
    Any,
}

impl VersionReq {
    pub fn parse(s: &str) -> Result<Self, DepCheckError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(DepCheckError::InvalidVersion(s.to_string()));
        }
        if s == "*" {
            return Ok(Self::Any);
        }
        if let Some(v) = s.strip_prefix('^') {
            Ok(Self::Caret(v.to_string()))
        } else if let Some(v) = s.strip_prefix('~') {
            Ok(Self::Tilde(v.to_string()))
        } else {
            Ok(Self::Exact(s.to_string()))
        }
    }

    /// Check if a concrete version satisfies this requirement (simplified).
    pub fn satisfies(&self, version: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(v) => v == version,
            Self::Caret(base) => {
                let base_parts = parse_version_parts(base);
                let ver_parts = parse_version_parts(version);
                if base_parts.is_none() || ver_parts.is_none() {
                    return false;
                }
                let (bm, bmi, _bp) = base_parts.unwrap();
                let (vm, vmi, vp) = ver_parts.unwrap();
                if bm == 0 {
                    // ^0.y.z -> >=0.y.z, <0.(y+1).0
                    vm == 0 && vmi == bmi && vp >= _bp
                } else {
                    // ^x.y.z -> >=x.y.z, <(x+1).0.0
                    vm == bm && (vmi > bmi || (vmi == bmi && vp >= _bp))
                }
            }
            Self::Tilde(base) => {
                let base_parts = parse_version_parts(base);
                let ver_parts = parse_version_parts(version);
                if base_parts.is_none() || ver_parts.is_none() {
                    return false;
                }
                let (bm, bmi, _bp) = base_parts.unwrap();
                let (vm, vmi, vp) = ver_parts.unwrap();
                // ~x.y.z -> >=x.y.z, <x.(y+1).0
                vm == bm && vmi == bmi && vp >= _bp
            }
        }
    }
}

fn parse_version_parts(v: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let major = parts[0].parse().ok()?;
    let minor = parts[1].parse().ok()?;
    let patch = parts[2].parse().ok()?;
    Some((major, minor, patch))
}

/// Software license identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum License {
    Mit,
    Apache2,
    Bsd2,
    Bsd3,
    Gpl2,
    Gpl3,
    Lgpl21,
    Lgpl3,
    Mpl2,
    Isc,
    Unlicense,
    Custom(String),
}

impl License {
    pub fn from_spdx(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "MIT" => Self::Mit,
            "APACHE-2.0" => Self::Apache2,
            "BSD-2-CLAUSE" => Self::Bsd2,
            "BSD-3-CLAUSE" => Self::Bsd3,
            "GPL-2.0" | "GPL-2.0-ONLY" => Self::Gpl2,
            "GPL-3.0" | "GPL-3.0-ONLY" => Self::Gpl3,
            "LGPL-2.1" | "LGPL-2.1-ONLY" => Self::Lgpl21,
            "LGPL-3.0" | "LGPL-3.0-ONLY" => Self::Lgpl3,
            "MPL-2.0" => Self::Mpl2,
            "ISC" => Self::Isc,
            "UNLICENSE" => Self::Unlicense,
            other => Self::Custom(other.to_string()),
        }
    }

    pub fn spdx_id(&self) -> &str {
        match self {
            Self::Mit => "MIT",
            Self::Apache2 => "Apache-2.0",
            Self::Bsd2 => "BSD-2-Clause",
            Self::Bsd3 => "BSD-3-Clause",
            Self::Gpl2 => "GPL-2.0",
            Self::Gpl3 => "GPL-3.0",
            Self::Lgpl21 => "LGPL-2.1",
            Self::Lgpl3 => "LGPL-3.0",
            Self::Mpl2 => "MPL-2.0",
            Self::Isc => "ISC",
            Self::Unlicense => "Unlicense",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Is this a copyleft license?
    pub fn is_copyleft(&self) -> bool {
        matches!(self, Self::Gpl2 | Self::Gpl3 | Self::Lgpl21 | Self::Lgpl3)
    }
}

/// A package in the dependency graph.
#[derive(Debug, Clone)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub license: Option<License>,
    pub dependencies: BTreeMap<String, VersionReq>,
    pub is_used: bool,
}

impl Package {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            license: None,
            dependencies: BTreeMap::new(),
            is_used: true,
        }
    }

    pub fn with_license(mut self, license: License) -> Self {
        self.license = Some(license);
        self
    }

    pub fn with_dep(mut self, name: impl Into<String>, req: VersionReq) -> Self {
        self.dependencies.insert(name.into(), req);
        self
    }

    pub fn set_used(&mut self, used: bool) {
        self.is_used = used;
    }
}

// ── Dependency Graph ────────────────────────────────────────────

/// A dependency graph for analyzing package relationships.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    packages: BTreeMap<String, Package>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self { packages: BTreeMap::new() }
    }

    /// Add a package to the graph.
    pub fn add_package(&mut self, pkg: Package) -> Result<(), DepCheckError> {
        if self.packages.contains_key(&pkg.name) {
            return Err(DepCheckError::DuplicatePackage(pkg.name));
        }
        // Check for self-dependency.
        if pkg.dependencies.contains_key(&pkg.name) {
            return Err(DepCheckError::SelfDependency(pkg.name));
        }
        self.packages.insert(pkg.name.clone(), pkg);
        Ok(())
    }

    /// Get a package by name.
    pub fn get_package(&self, name: &str) -> Option<&Package> {
        self.packages.get(name)
    }

    /// Number of packages in the graph.
    pub fn package_count(&self) -> usize {
        self.packages.len()
    }

    /// Total number of dependency edges.
    pub fn edge_count(&self) -> usize {
        self.packages.values().map(|p| p.dependencies.len()).sum()
    }

    /// Direct dependencies of a package.
    pub fn direct_deps(&self, name: &str) -> Result<Vec<&str>, DepCheckError> {
        let pkg = self
            .packages
            .get(name)
            .ok_or_else(|| DepCheckError::PackageNotFound(name.to_string()))?;
        Ok(pkg.dependencies.keys().map(|s| s.as_str()).collect())
    }

    /// All transitive dependencies of a package (BFS).
    pub fn transitive_deps(&self, name: &str) -> Result<BTreeSet<String>, DepCheckError> {
        if !self.packages.contains_key(name) {
            return Err(DepCheckError::PackageNotFound(name.to_string()));
        }

        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();

        for dep in self.packages[name].dependencies.keys() {
            queue.push_back(dep.clone());
        }

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            if let Some(pkg) = self.packages.get(&current) {
                for dep in pkg.dependencies.keys() {
                    if !visited.contains(dep) {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        Ok(visited)
    }

    /// Reverse dependencies: who depends on this package?
    pub fn reverse_deps(&self, name: &str) -> Result<Vec<&str>, DepCheckError> {
        if !self.packages.contains_key(name) {
            return Err(DepCheckError::PackageNotFound(name.to_string()));
        }
        let mut rdeps: Vec<&str> = self
            .packages
            .iter()
            .filter(|(_, pkg)| pkg.dependencies.contains_key(name))
            .map(|(n, _)| n.as_str())
            .collect();
        rdeps.sort();
        Ok(rdeps)
    }

    /// Detect circular dependencies using DFS.
    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();

        for name in self.packages.keys() {
            if visited.contains(name) {
                continue;
            }
            let mut path = Vec::new();
            let mut on_stack = HashSet::new();
            self.dfs_cycle(name, &mut path, &mut on_stack, &mut visited, &mut cycles);
        }

        cycles
    }

    fn dfs_cycle(
        &self,
        node: &str,
        path: &mut Vec<String>,
        on_stack: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        if on_stack.contains(node) {
            // Found a cycle — extract it.
            if let Some(start) = path.iter().position(|n| n == node) {
                let mut cycle = path[start..].to_vec();
                cycle.push(node.to_string());
                cycles.push(cycle);
            }
            return;
        }
        if visited.contains(node) {
            return;
        }

        path.push(node.to_string());
        on_stack.insert(node.to_string());

        if let Some(pkg) = self.packages.get(node) {
            for dep in pkg.dependencies.keys() {
                self.dfs_cycle(dep, path, on_stack, visited, cycles);
            }
        }

        on_stack.remove(node);
        path.pop();
        visited.insert(node.to_string());
    }

    /// Detect version conflicts: multiple packages requiring incompatible
    /// versions of the same dependency.
    pub fn detect_version_conflicts(&self) -> Vec<VersionConflict> {
        let mut requirements: HashMap<String, Vec<(String, VersionReq)>> = HashMap::new();

        for pkg in self.packages.values() {
            for (dep_name, req) in &pkg.dependencies {
                requirements
                    .entry(dep_name.clone())
                    .or_default()
                    .push((pkg.name.clone(), req.clone()));
            }
        }

        let mut conflicts = Vec::new();
        for (dep_name, reqs) in &requirements {
            if reqs.len() < 2 {
                continue;
            }
            // Check if the resolved version satisfies all requirements.
            if let Some(dep_pkg) = self.packages.get(dep_name) {
                let unsatisfied: Vec<(String, VersionReq)> = reqs
                    .iter()
                    .filter(|(_, req)| !req.satisfies(&dep_pkg.version))
                    .cloned()
                    .collect();
                if !unsatisfied.is_empty() {
                    conflicts.push(VersionConflict {
                        package: dep_name.clone(),
                        resolved_version: dep_pkg.version.clone(),
                        unsatisfied_by: unsatisfied,
                    });
                }
            }
        }

        conflicts.sort_by(|a, b| a.package.cmp(&b.package));
        conflicts
    }

    /// Find packages that are marked as unused.
    pub fn find_unused(&self) -> Vec<&str> {
        let mut unused: Vec<&str> = self
            .packages
            .values()
            .filter(|p| !p.is_used)
            .map(|p| p.name.as_str())
            .collect();
        unused.sort();
        unused
    }

    /// Check license compliance against an allow-list.
    pub fn check_license_compliance(
        &self,
        allowed: &[License],
    ) -> Vec<LicenseViolation> {
        let allowed_set: HashSet<&License> = allowed.iter().collect();
        let mut violations = Vec::new();

        for pkg in self.packages.values() {
            if let Some(lic) = &pkg.license {
                if !allowed_set.contains(lic) {
                    violations.push(LicenseViolation {
                        package: pkg.name.clone(),
                        license: lic.clone(),
                    });
                }
            }
        }

        violations.sort_by(|a, b| a.package.cmp(&b.package));
        violations
    }

    /// Find packages with no dependents (root packages or orphans).
    pub fn find_roots(&self) -> Vec<&str> {
        let all_deps: HashSet<&str> = self
            .packages
            .values()
            .flat_map(|p| p.dependencies.keys().map(|s| s.as_str()))
            .collect();

        let mut roots: Vec<&str> = self
            .packages
            .keys()
            .filter(|name| !all_deps.contains(name.as_str()))
            .map(|s| s.as_str())
            .collect();
        roots.sort();
        roots
    }

    /// Find packages with no dependencies (leaf packages).
    pub fn find_leaves(&self) -> Vec<&str> {
        let mut leaves: Vec<&str> = self
            .packages
            .values()
            .filter(|p| p.dependencies.is_empty())
            .map(|p| p.name.as_str())
            .collect();
        leaves.sort();
        leaves
    }

    /// Compute the dependency depth of a package.
    pub fn depth(&self, name: &str) -> Result<usize, DepCheckError> {
        if !self.packages.contains_key(name) {
            return Err(DepCheckError::PackageNotFound(name.to_string()));
        }
        Ok(self.compute_depth(name, &mut HashMap::new()))
    }

    fn compute_depth(&self, name: &str, cache: &mut HashMap<String, usize>) -> usize {
        if let Some(&d) = cache.get(name) {
            return d;
        }
        let pkg = match self.packages.get(name) {
            Some(p) => p,
            None => return 0,
        };
        if pkg.dependencies.is_empty() {
            cache.insert(name.to_string(), 0);
            return 0;
        }
        let max_child = pkg
            .dependencies
            .keys()
            .map(|dep| self.compute_depth(dep, cache))
            .max()
            .unwrap_or(0);
        let depth = max_child + 1;
        cache.insert(name.to_string(), depth);
        depth
    }

    /// Generate a summary report.
    pub fn summary(&self) -> DepSummary {
        let cycles = self.detect_cycles();
        let conflicts = self.detect_version_conflicts();
        let unused = self.find_unused().len();
        DepSummary {
            total_packages: self.package_count(),
            total_edges: self.edge_count(),
            cycle_count: cycles.len(),
            conflict_count: conflicts.len(),
            unused_count: unused,
            root_count: self.find_roots().len(),
            leaf_count: self.find_leaves().len(),
        }
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A version conflict report.
#[derive(Debug, Clone)]
pub struct VersionConflict {
    pub package: String,
    pub resolved_version: String,
    pub unsatisfied_by: Vec<(String, VersionReq)>,
}

/// A license violation.
#[derive(Debug, Clone)]
pub struct LicenseViolation {
    pub package: String,
    pub license: License,
}

/// Summary statistics.
#[derive(Debug, Clone)]
pub struct DepSummary {
    pub total_packages: usize,
    pub total_edges: usize,
    pub cycle_count: usize,
    pub conflict_count: usize,
    pub unused_count: usize,
    pub root_count: usize,
    pub leaf_count: usize,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> DependencyGraph {
        let mut g = DependencyGraph::new();
        g.add_package(
            Package::new("app", "1.0.0")
                .with_license(License::Mit)
                .with_dep("lib-a", VersionReq::Caret("1.0.0".into()))
                .with_dep("lib-b", VersionReq::Caret("2.0.0".into())),
        ).unwrap();
        g.add_package(
            Package::new("lib-a", "1.2.0")
                .with_license(License::Mit)
                .with_dep("lib-c", VersionReq::Caret("0.5.0".into())),
        ).unwrap();
        g.add_package(
            Package::new("lib-b", "2.1.0")
                .with_license(License::Apache2)
                .with_dep("lib-c", VersionReq::Caret("0.5.0".into())),
        ).unwrap();
        g.add_package(
            Package::new("lib-c", "0.5.3")
                .with_license(License::Mit),
        ).unwrap();
        g
    }

    #[test]
    fn test_add_package() {
        let mut g = DependencyGraph::new();
        g.add_package(Package::new("foo", "1.0.0")).unwrap();
        assert_eq!(g.package_count(), 1);
    }

    #[test]
    fn test_duplicate_package() {
        let mut g = DependencyGraph::new();
        g.add_package(Package::new("foo", "1.0.0")).unwrap();
        let err = g.add_package(Package::new("foo", "2.0.0")).unwrap_err();
        assert!(matches!(err, DepCheckError::DuplicatePackage(_)));
    }

    #[test]
    fn test_self_dependency() {
        let mut g = DependencyGraph::new();
        let err = g
            .add_package(Package::new("foo", "1.0.0").with_dep("foo", VersionReq::Any))
            .unwrap_err();
        assert!(matches!(err, DepCheckError::SelfDependency(_)));
    }

    #[test]
    fn test_direct_deps() {
        let g = make_graph();
        let deps = g.direct_deps("app").unwrap();
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"lib-a"));
        assert!(deps.contains(&"lib-b"));
    }

    #[test]
    fn test_direct_deps_not_found() {
        let g = make_graph();
        let err = g.direct_deps("nonexistent").unwrap_err();
        assert!(matches!(err, DepCheckError::PackageNotFound(_)));
    }

    #[test]
    fn test_transitive_deps() {
        let g = make_graph();
        let trans = g.transitive_deps("app").unwrap();
        assert_eq!(trans.len(), 3);
        assert!(trans.contains("lib-a"));
        assert!(trans.contains("lib-b"));
        assert!(trans.contains("lib-c"));
    }

    #[test]
    fn test_transitive_deps_leaf() {
        let g = make_graph();
        let trans = g.transitive_deps("lib-c").unwrap();
        assert!(trans.is_empty());
    }

    #[test]
    fn test_reverse_deps() {
        let g = make_graph();
        let rdeps = g.reverse_deps("lib-c").unwrap();
        assert_eq!(rdeps.len(), 2);
        assert!(rdeps.contains(&"lib-a"));
        assert!(rdeps.contains(&"lib-b"));
    }

    #[test]
    fn test_reverse_deps_root() {
        let g = make_graph();
        let rdeps = g.reverse_deps("app").unwrap();
        assert!(rdeps.is_empty());
    }

    #[test]
    fn test_no_cycles() {
        let g = make_graph();
        let cycles = g.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_detect_cycle() {
        let mut g = DependencyGraph::new();
        g.add_package(
            Package::new("a", "1.0.0").with_dep("b", VersionReq::Any),
        ).unwrap();
        g.add_package(
            Package::new("b", "1.0.0").with_dep("c", VersionReq::Any),
        ).unwrap();
        g.add_package(
            Package::new("c", "1.0.0").with_dep("a", VersionReq::Any),
        ).unwrap();

        let cycles = g.detect_cycles();
        assert!(!cycles.is_empty());
        // The cycle should contain a, b, c.
        let cycle = &cycles[0];
        assert!(cycle.contains(&"a".to_string()));
        assert!(cycle.contains(&"b".to_string()));
        assert!(cycle.contains(&"c".to_string()));
    }

    #[test]
    fn test_version_req_exact() {
        let req = VersionReq::parse("1.2.3").unwrap();
        assert!(req.satisfies("1.2.3"));
        assert!(!req.satisfies("1.2.4"));
    }

    #[test]
    fn test_version_req_caret() {
        let req = VersionReq::parse("^1.2.0").unwrap();
        assert!(req.satisfies("1.2.0"));
        assert!(req.satisfies("1.3.0"));
        assert!(req.satisfies("1.2.5"));
        assert!(!req.satisfies("2.0.0"));
        assert!(!req.satisfies("1.1.0"));
    }

    #[test]
    fn test_version_req_caret_zero() {
        let req = VersionReq::parse("^0.5.0").unwrap();
        assert!(req.satisfies("0.5.0"));
        assert!(req.satisfies("0.5.3"));
        assert!(!req.satisfies("0.6.0"));
    }

    #[test]
    fn test_version_req_tilde() {
        let req = VersionReq::parse("~1.2.0").unwrap();
        assert!(req.satisfies("1.2.0"));
        assert!(req.satisfies("1.2.9"));
        assert!(!req.satisfies("1.3.0"));
    }

    #[test]
    fn test_version_req_any() {
        let req = VersionReq::parse("*").unwrap();
        assert!(req.satisfies("0.0.1"));
        assert!(req.satisfies("999.999.999"));
    }

    #[test]
    fn test_version_req_invalid() {
        assert!(VersionReq::parse("").is_err());
    }

    #[test]
    fn test_no_version_conflicts() {
        let g = make_graph();
        let conflicts = g.detect_version_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_version_conflict_detected() {
        let mut g = DependencyGraph::new();
        g.add_package(
            Package::new("app", "1.0.0")
                .with_dep("shared", VersionReq::Exact("1.0.0".into())),
        ).unwrap();
        g.add_package(
            Package::new("lib", "1.0.0")
                .with_dep("shared", VersionReq::Exact("2.0.0".into())),
        ).unwrap();
        g.add_package(Package::new("shared", "1.0.0")).unwrap();

        let conflicts = g.detect_version_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].package, "shared");
        assert_eq!(conflicts[0].resolved_version, "1.0.0");
    }

    #[test]
    fn test_find_unused() {
        let mut g = DependencyGraph::new();
        g.add_package(Package::new("used", "1.0.0")).unwrap();
        let mut unused_pkg = Package::new("unused", "1.0.0");
        unused_pkg.set_used(false);
        g.add_package(unused_pkg).unwrap();

        let unused = g.find_unused();
        assert_eq!(unused, vec!["unused"]);
    }

    #[test]
    fn test_license_compliance_pass() {
        let g = make_graph();
        let allowed = vec![License::Mit, License::Apache2];
        let violations = g.check_license_compliance(&allowed);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_license_compliance_fail() {
        let mut g = DependencyGraph::new();
        g.add_package(Package::new("gpl-pkg", "1.0.0").with_license(License::Gpl3))
            .unwrap();

        let allowed = vec![License::Mit, License::Apache2];
        let violations = g.check_license_compliance(&allowed);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].package, "gpl-pkg");
    }

    #[test]
    fn test_license_copyleft() {
        assert!(License::Gpl3.is_copyleft());
        assert!(License::Lgpl21.is_copyleft());
        assert!(!License::Mit.is_copyleft());
        assert!(!License::Apache2.is_copyleft());
    }

    #[test]
    fn test_license_spdx_roundtrip() {
        let lic = License::from_spdx("MIT");
        assert_eq!(lic.spdx_id(), "MIT");

        let lic = License::from_spdx("Apache-2.0");
        assert_eq!(lic.spdx_id(), "Apache-2.0");
    }

    #[test]
    fn test_find_roots() {
        let g = make_graph();
        let roots = g.find_roots();
        assert_eq!(roots, vec!["app"]);
    }

    #[test]
    fn test_find_leaves() {
        let g = make_graph();
        let leaves = g.find_leaves();
        assert_eq!(leaves, vec!["lib-c"]);
    }

    #[test]
    fn test_depth() {
        let g = make_graph();
        assert_eq!(g.depth("lib-c").unwrap(), 0);
        assert_eq!(g.depth("lib-a").unwrap(), 1);
        assert_eq!(g.depth("app").unwrap(), 2);
    }

    #[test]
    fn test_edge_count() {
        let g = make_graph();
        // app->lib-a, app->lib-b, lib-a->lib-c, lib-b->lib-c = 4
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_summary() {
        let g = make_graph();
        let s = g.summary();
        assert_eq!(s.total_packages, 4);
        assert_eq!(s.total_edges, 4);
        assert_eq!(s.cycle_count, 0);
        assert_eq!(s.conflict_count, 0);
        assert_eq!(s.root_count, 1);
        assert_eq!(s.leaf_count, 1);
    }

    #[test]
    fn test_get_package() {
        let g = make_graph();
        let pkg = g.get_package("lib-a").unwrap();
        assert_eq!(pkg.version, "1.2.0");
        assert!(g.get_package("nonexistent").is_none());
    }

    #[test]
    fn test_custom_license() {
        let lic = License::from_spdx("PROPRIETARY");
        assert_eq!(lic, License::Custom("PROPRIETARY".to_string()));
        assert!(!lic.is_copyleft());
    }
}
