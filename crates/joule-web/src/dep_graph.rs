//! Dependency graph analyzer.
//!
//! Package/module dependency analysis: circular dependency detection,
//! dependency tree visualization, unused dependency detection, version
//! conflict detection, and DOT graph export. Pure Rust — no package
//! manager dependencies.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from dependency graph operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepGraphError {
    /// Package not found.
    PackageNotFound(String),
    /// Duplicate package.
    DuplicatePackage(String),
    /// Circular dependency detected.
    CircularDependency(Vec<String>),
    /// Version conflict.
    VersionConflict {
        package: String,
        versions: Vec<String>,
    },
}

impl fmt::Display for DepGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PackageNotFound(name) => write!(f, "package not found: {name}"),
            Self::DuplicatePackage(name) => write!(f, "duplicate package: {name}"),
            Self::CircularDependency(cycle) => {
                write!(f, "circular dependency: {}", cycle.join(" -> "))
            }
            Self::VersionConflict { package, versions } => {
                write!(
                    f,
                    "version conflict for '{package}': {}",
                    versions.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for DepGraphError {}

// ── Dependency Spec ─────────────────────────────────────────────

/// A single dependency specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DepSpec {
    /// Name of the required package.
    pub name: String,
    /// Required version (semver range string or exact).
    pub version: String,
    /// Whether this is optional (e.g., dev dependency).
    pub optional: bool,
    /// Dependency kind.
    pub kind: DepKind,
}

/// Kind of dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DepKind {
    /// Normal runtime dependency.
    Normal,
    /// Development dependency.
    Dev,
    /// Build dependency.
    Build,
    /// Peer dependency.
    Peer,
    /// Optional feature dependency.
    Optional,
}

impl fmt::Display for DepKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Dev => write!(f, "dev"),
            Self::Build => write!(f, "build"),
            Self::Peer => write!(f, "peer"),
            Self::Optional => write!(f, "optional"),
        }
    }
}

// ── Package ─────────────────────────────────────────────────────

/// A package in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub dependencies: Vec<DepSpec>,
    /// Set of feature flags enabled.
    pub features: HashSet<String>,
}

impl Package {
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            dependencies: Vec::new(),
            features: HashSet::new(),
        }
    }

    /// Add a dependency.
    pub fn add_dep(&mut self, dep: DepSpec) {
        self.dependencies.push(dep);
    }

    /// Add a normal dependency (shorthand).
    pub fn depends_on(&mut self, name: &str, version: &str) {
        self.dependencies.push(DepSpec {
            name: name.to_string(),
            version: version.to_string(),
            optional: false,
            kind: DepKind::Normal,
        });
    }

    /// Add a dev dependency (shorthand).
    pub fn dev_depends_on(&mut self, name: &str, version: &str) {
        self.dependencies.push(DepSpec {
            name: name.to_string(),
            version: version.to_string(),
            optional: false,
            kind: DepKind::Dev,
        });
    }
}

// ── Dependency Graph ────────────────────────────────────────────

/// The dependency graph.
#[derive(Debug, Clone, Default)]
pub struct DepGraph {
    packages: HashMap<String, Package>,
}

impl DepGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a package to the graph.
    pub fn add_package(&mut self, pkg: Package) -> Result<(), DepGraphError> {
        if self.packages.contains_key(&pkg.name) {
            return Err(DepGraphError::DuplicatePackage(pkg.name.clone()));
        }
        self.packages.insert(pkg.name.clone(), pkg);
        Ok(())
    }

    /// Get a package by name.
    pub fn get_package(&self, name: &str) -> Option<&Package> {
        self.packages.get(name)
    }

    /// List all package names sorted.
    pub fn package_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.packages.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Number of packages.
    pub fn len(&self) -> usize {
        self.packages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.packages.is_empty()
    }

    // ── Analysis ────────────────────────────────────────────────

    /// Get direct dependencies of a package.
    pub fn direct_deps(&self, name: &str) -> Result<Vec<&str>, DepGraphError> {
        let pkg = self
            .packages
            .get(name)
            .ok_or_else(|| DepGraphError::PackageNotFound(name.to_string()))?;
        Ok(pkg
            .dependencies
            .iter()
            .map(|d| d.name.as_str())
            .collect())
    }

    /// Get transitive (all reachable) dependencies of a package.
    pub fn transitive_deps(&self, name: &str) -> Result<HashSet<String>, DepGraphError> {
        if !self.packages.contains_key(name) {
            return Err(DepGraphError::PackageNotFound(name.to_string()));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(name.to_string());

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if let Some(pkg) = self.packages.get(&current) {
                for dep in &pkg.dependencies {
                    if !visited.contains(&dep.name) {
                        queue.push_back(dep.name.clone());
                    }
                }
            }
        }

        visited.remove(name);
        Ok(visited)
    }

    /// Get reverse dependencies (packages that depend on the given package).
    pub fn reverse_deps(&self, name: &str) -> Vec<&str> {
        self.packages
            .values()
            .filter(|pkg| pkg.dependencies.iter().any(|d| d.name == name))
            .map(|pkg| pkg.name.as_str())
            .collect()
    }

    /// Detect all circular dependencies in the graph.
    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        let mut global_visited = HashSet::new();

        for name in self.packages.keys() {
            if global_visited.contains(name) {
                continue;
            }
            let mut path = Vec::new();
            let mut path_set = HashSet::new();
            let mut visited = HashSet::new();
            self.dfs_cycles(name, &mut path, &mut path_set, &mut visited, &mut cycles);
            global_visited.extend(visited);
        }

        cycles
    }

    fn dfs_cycles(
        &self,
        node: &str,
        path: &mut Vec<String>,
        path_set: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        if path_set.contains(node) {
            // Found a cycle — extract it.
            let cycle_start = path.iter().position(|n| n == node).unwrap();
            let mut cycle: Vec<String> = path[cycle_start..].to_vec();
            cycle.push(node.to_string());
            cycles.push(cycle);
            return;
        }

        if visited.contains(node) {
            return;
        }

        path.push(node.to_string());
        path_set.insert(node.to_string());

        if let Some(pkg) = self.packages.get(node) {
            for dep in &pkg.dependencies {
                self.dfs_cycles(&dep.name, path, path_set, visited, cycles);
            }
        }

        path.pop();
        path_set.remove(node);
        visited.insert(node.to_string());
    }

    /// Detect unused dependencies: packages in the graph that are not depended on
    /// by any other package (except root packages).
    pub fn unused_deps(&self, roots: &[&str]) -> Vec<&str> {
        // Collect all packages that are reachable from roots.
        let mut reachable = HashSet::new();
        for root in roots {
            if let Ok(deps) = self.transitive_deps(root) {
                reachable.extend(deps);
            }
            reachable.insert(root.to_string());
        }

        self.packages
            .keys()
            .filter(|name| !reachable.contains(name.as_str()))
            .map(|s| s.as_str())
            .collect()
    }

    /// Detect version conflicts: different packages requiring different
    /// versions of the same dependency.
    pub fn detect_version_conflicts(&self) -> Vec<DepGraphError> {
        let mut dep_versions: HashMap<String, HashSet<String>> = HashMap::new();

        for pkg in self.packages.values() {
            for dep in &pkg.dependencies {
                dep_versions
                    .entry(dep.name.clone())
                    .or_default()
                    .insert(dep.version.clone());
            }
        }

        let mut conflicts = Vec::new();
        for (name, versions) in &dep_versions {
            if versions.len() > 1 {
                let mut version_list: Vec<String> = versions.iter().cloned().collect();
                version_list.sort();
                conflicts.push(DepGraphError::VersionConflict {
                    package: name.clone(),
                    versions: version_list,
                });
            }
        }

        conflicts
    }

    /// Compute the depth of each package from the given root.
    pub fn depth_from(&self, root: &str) -> Result<HashMap<String, usize>, DepGraphError> {
        if !self.packages.contains_key(root) {
            return Err(DepGraphError::PackageNotFound(root.to_string()));
        }

        let mut depths = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back((root.to_string(), 0usize));

        while let Some((name, depth)) = queue.pop_front() {
            if depths.contains_key(&name) {
                continue;
            }
            depths.insert(name.clone(), depth);
            if let Some(pkg) = self.packages.get(&name) {
                for dep in &pkg.dependencies {
                    if !depths.contains_key(&dep.name) {
                        queue.push_back((dep.name.clone(), depth + 1));
                    }
                }
            }
        }

        Ok(depths)
    }

    // ── Visualization ───────────────────────────────────────────

    /// Generate a text tree visualization.
    pub fn tree_text(&self, root: &str) -> Result<String, DepGraphError> {
        if !self.packages.contains_key(root) {
            return Err(DepGraphError::PackageNotFound(root.to_string()));
        }

        let mut output = String::new();
        let mut visited = HashSet::new();
        self.build_tree_text(root, &mut output, "", true, &mut visited);
        Ok(output)
    }

    fn build_tree_text(
        &self,
        name: &str,
        output: &mut String,
        prefix: &str,
        is_last: bool,
        visited: &mut HashSet<String>,
    ) {
        let connector = if prefix.is_empty() {
            ""
        } else if is_last {
            "`-- "
        } else {
            "|-- "
        };

        let version = self
            .packages
            .get(name)
            .map_or("?", |p| p.version.as_str());

        let circular = visited.contains(name);
        let suffix = if circular { " (circular)" } else { "" };

        output.push_str(&format!("{prefix}{connector}{name}@{version}{suffix}\n"));

        if circular {
            return;
        }

        visited.insert(name.to_string());

        if let Some(pkg) = self.packages.get(name) {
            let deps: Vec<&DepSpec> = pkg.dependencies.iter().collect();
            for (i, dep) in deps.iter().enumerate() {
                let is_last_dep = i == deps.len() - 1;
                let new_prefix = if prefix.is_empty() {
                    String::new()
                } else if is_last {
                    format!("{prefix}    ")
                } else {
                    format!("{prefix}|   ")
                };
                self.build_tree_text(&dep.name, output, &new_prefix, is_last_dep, visited);
            }
        }

        visited.remove(name);
    }

    /// Export the graph in DOT format for Graphviz.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph dependencies {\n");
        out.push_str("    rankdir=LR;\n");
        out.push_str("    node [shape=box];\n\n");

        // Sort for deterministic output.
        let mut names: Vec<&str> = self.packages.keys().map(|s| s.as_str()).collect();
        names.sort();

        for name in &names {
            if let Some(pkg) = self.packages.get(*name) {
                let label = format!("{}\\n{}", pkg.name, pkg.version);
                out.push_str(&format!("    \"{name}\" [label=\"{label}\"];\n"));
            }
        }

        out.push('\n');

        for name in &names {
            if let Some(pkg) = self.packages.get(*name) {
                let mut dep_names: Vec<&str> =
                    pkg.dependencies.iter().map(|d| d.name.as_str()).collect();
                dep_names.sort();
                for dep_name in dep_names {
                    let dep = pkg.dependencies.iter().find(|d| d.name == dep_name).unwrap();
                    let style = match dep.kind {
                        DepKind::Dev => " [style=dashed]",
                        DepKind::Optional => " [style=dotted]",
                        _ => "",
                    };
                    out.push_str(&format!(
                        "    \"{name}\" -> \"{dep_name}\"{style};\n"
                    ));
                }
            }
        }

        out.push_str("}\n");
        out
    }

    /// Export the graph in DOT format for a subtree rooted at `root`.
    pub fn to_dot_subtree(&self, root: &str) -> Result<String, DepGraphError> {
        let reachable = self.transitive_deps(root)?;
        let mut nodes: HashSet<String> = reachable;
        nodes.insert(root.to_string());

        let mut out = String::from("digraph dependencies {\n");
        out.push_str("    rankdir=LR;\n");
        out.push_str("    node [shape=box];\n\n");

        let mut sorted_nodes: Vec<&str> = nodes.iter().map(|s| s.as_str()).collect();
        sorted_nodes.sort();

        for name in &sorted_nodes {
            if let Some(pkg) = self.packages.get(*name) {
                let label = format!("{}\\n{}", pkg.name, pkg.version);
                out.push_str(&format!("    \"{name}\" [label=\"{label}\"];\n"));
            }
        }

        out.push('\n');

        for name in &sorted_nodes {
            if let Some(pkg) = self.packages.get(*name) {
                for dep in &pkg.dependencies {
                    if nodes.contains(&dep.name) {
                        out.push_str(&format!(
                            "    \"{name}\" -> \"{}\";\n",
                            dep.name
                        ));
                    }
                }
            }
        }

        out.push_str("}\n");
        Ok(out)
    }
}

// ── Statistics ──────────────────────────────────────────────────

/// Statistics about a dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepStats {
    pub total_packages: usize,
    pub total_edges: usize,
    pub max_depth: usize,
    pub packages_with_no_deps: usize,
    pub packages_with_most_deps: Vec<(String, usize)>,
    pub most_depended_on: Vec<(String, usize)>,
}

impl DepGraph {
    /// Compute statistics about the graph.
    pub fn stats(&self, root: &str) -> Result<DepStats, DepGraphError> {
        let total_packages = self.packages.len();
        let total_edges: usize = self.packages.values().map(|p| p.dependencies.len()).sum();

        let packages_with_no_deps = self
            .packages
            .values()
            .filter(|p| p.dependencies.is_empty())
            .count();

        let depths = self.depth_from(root)?;
        let max_depth = depths.values().copied().max().unwrap_or(0);

        // Packages with most deps.
        let mut by_dep_count: Vec<(String, usize)> = self
            .packages
            .values()
            .map(|p| (p.name.clone(), p.dependencies.len()))
            .collect();
        by_dep_count.sort_by(|a, b| b.1.cmp(&a.1));
        by_dep_count.truncate(5);

        // Most depended on.
        let mut dep_count: HashMap<String, usize> = HashMap::new();
        for pkg in self.packages.values() {
            for dep in &pkg.dependencies {
                *dep_count.entry(dep.name.clone()).or_insert(0) += 1;
            }
        }
        let mut most_depended: Vec<(String, usize)> = dep_count.into_iter().collect();
        most_depended.sort_by(|a, b| b.1.cmp(&a.1));
        most_depended.truncate(5);

        Ok(DepStats {
            total_packages,
            total_edges,
            max_depth,
            packages_with_no_deps,
            packages_with_most_deps: by_dep_count,
            most_depended_on: most_depended,
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> DepGraph {
        let mut graph = DepGraph::new();

        let mut app = Package::new("app", "1.0.0");
        app.depends_on("web-framework", "2.0.0");
        app.depends_on("database", "3.1.0");
        graph.add_package(app).unwrap();

        let mut web = Package::new("web-framework", "2.0.0");
        web.depends_on("http", "1.5.0");
        web.depends_on("json", "0.9.0");
        graph.add_package(web).unwrap();

        let mut db = Package::new("database", "3.1.0");
        db.depends_on("json", "0.9.0");
        graph.add_package(db).unwrap();

        let http = Package::new("http", "1.5.0");
        graph.add_package(http).unwrap();

        let json = Package::new("json", "0.9.0");
        graph.add_package(json).unwrap();

        graph
    }

    #[test]
    fn add_and_get_package() {
        let graph = make_graph();
        assert_eq!(graph.len(), 5);
        let pkg = graph.get_package("app").unwrap();
        assert_eq!(pkg.version, "1.0.0");
    }

    #[test]
    fn duplicate_package() {
        let mut graph = DepGraph::new();
        graph.add_package(Package::new("a", "1.0.0")).unwrap();
        let err = graph.add_package(Package::new("a", "2.0.0")).unwrap_err();
        assert!(matches!(err, DepGraphError::DuplicatePackage(_)));
    }

    #[test]
    fn direct_deps() {
        let graph = make_graph();
        let deps = graph.direct_deps("app").unwrap();
        assert!(deps.contains(&"web-framework"));
        assert!(deps.contains(&"database"));
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn transitive_deps() {
        let graph = make_graph();
        let trans = graph.transitive_deps("app").unwrap();
        assert!(trans.contains("web-framework"));
        assert!(trans.contains("database"));
        assert!(trans.contains("http"));
        assert!(trans.contains("json"));
        assert_eq!(trans.len(), 4);
    }

    #[test]
    fn reverse_deps() {
        let graph = make_graph();
        let mut rdeps = graph.reverse_deps("json");
        rdeps.sort();
        assert_eq!(rdeps, vec!["database", "web-framework"]);
    }

    #[test]
    fn package_not_found() {
        let graph = make_graph();
        let err = graph.direct_deps("nonexistent").unwrap_err();
        assert!(matches!(err, DepGraphError::PackageNotFound(_)));
    }

    #[test]
    fn detect_no_cycles() {
        let graph = make_graph();
        let cycles = graph.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn detect_cycle() {
        let mut graph = DepGraph::new();
        let mut a = Package::new("a", "1.0.0");
        a.depends_on("b", "1.0.0");
        let mut b = Package::new("b", "1.0.0");
        b.depends_on("c", "1.0.0");
        let mut c = Package::new("c", "1.0.0");
        c.depends_on("a", "1.0.0");
        graph.add_package(a).unwrap();
        graph.add_package(b).unwrap();
        graph.add_package(c).unwrap();

        let cycles = graph.detect_cycles();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn version_conflict_detection() {
        let mut graph = DepGraph::new();
        let mut a = Package::new("a", "1.0.0");
        a.depends_on("c", "1.0.0");
        let mut b = Package::new("b", "1.0.0");
        b.depends_on("c", "2.0.0");
        let c = Package::new("c", "1.0.0");
        graph.add_package(a).unwrap();
        graph.add_package(b).unwrap();
        graph.add_package(c).unwrap();

        let conflicts = graph.detect_version_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(&conflicts[0], DepGraphError::VersionConflict { package, .. } if package == "c"));
    }

    #[test]
    fn no_version_conflicts() {
        let graph = make_graph();
        let conflicts = graph.detect_version_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn unused_deps() {
        let mut graph = make_graph();
        // Add an orphan package.
        graph
            .add_package(Package::new("orphan", "0.1.0"))
            .unwrap();
        let unused = graph.unused_deps(&["app"]);
        assert!(unused.contains(&"orphan"));
    }

    #[test]
    fn depth_from() {
        let graph = make_graph();
        let depths = graph.depth_from("app").unwrap();
        assert_eq!(depths["app"], 0);
        assert_eq!(depths["web-framework"], 1);
        assert_eq!(depths["database"], 1);
        assert_eq!(depths["http"], 2);
        assert_eq!(depths["json"], 2);
    }

    #[test]
    fn tree_text() {
        let graph = make_graph();
        let tree = graph.tree_text("app").unwrap();
        assert!(tree.contains("app@1.0.0"));
        assert!(tree.contains("web-framework@2.0.0"));
        assert!(tree.contains("json@0.9.0"));
    }

    #[test]
    fn dot_export() {
        let graph = make_graph();
        let dot = graph.to_dot();
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("\"app\""));
        assert!(dot.contains("\"web-framework\""));
        assert!(dot.contains("->"));
    }

    #[test]
    fn dot_subtree() {
        let graph = make_graph();
        let dot = graph.to_dot_subtree("web-framework").unwrap();
        assert!(dot.contains("web-framework"));
        assert!(dot.contains("http"));
        assert!(!dot.contains("database")); // not in subtree
    }

    #[test]
    fn dev_dependency_dot() {
        let mut graph = DepGraph::new();
        let mut a = Package::new("a", "1.0.0");
        a.add_dep(DepSpec {
            name: "test-lib".to_string(),
            version: "1.0.0".to_string(),
            optional: false,
            kind: DepKind::Dev,
        });
        let b = Package::new("test-lib", "1.0.0");
        graph.add_package(a).unwrap();
        graph.add_package(b).unwrap();

        let dot = graph.to_dot();
        assert!(dot.contains("dashed"));
    }

    #[test]
    fn stats() {
        let graph = make_graph();
        let stats = graph.stats("app").unwrap();
        assert_eq!(stats.total_packages, 5);
        assert_eq!(stats.total_edges, 5);
        assert_eq!(stats.max_depth, 2);
        assert_eq!(stats.packages_with_no_deps, 2); // http, json
    }

    #[test]
    fn package_names_sorted() {
        let graph = make_graph();
        let names = graph.package_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn dep_kind_display() {
        assert_eq!(format!("{}", DepKind::Normal), "normal");
        assert_eq!(format!("{}", DepKind::Dev), "dev");
    }

    #[test]
    fn error_display() {
        let e = DepGraphError::CircularDependency(vec!["a".to_string(), "b".to_string()]);
        assert!(format!("{e}").contains("a -> b"));
    }

    #[test]
    fn empty_graph() {
        let graph = DepGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn package_features() {
        let mut pkg = Package::new("my-app", "1.0.0");
        pkg.features.insert("ssl".to_string());
        assert!(pkg.features.contains("ssl"));
    }
}
