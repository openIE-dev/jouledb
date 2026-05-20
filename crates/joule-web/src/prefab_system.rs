//! Prefab/template system for game objects.
//!
//! Prefabs are serializable templates (name + component key-value pairs).
//! Supports instantiation, per-instance property overrides, nested prefabs,
//! prefab variants (inherit from base, override fields), JSON serialization,
//! a prefab library/registry, and diff between instance and source.

use std::collections::HashMap;
use std::fmt;

// ── Component value ────────────────────────────────────────────

/// A property value that can be stored in a component.
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl PropValue {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PropValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            PropValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            PropValue::Text(v) => Some(v.as_str()),
            _ => None,
        }
    }

    fn to_json_value(&self) -> String {
        match self {
            PropValue::Int(v) => format!("{v}"),
            PropValue::Float(v) => {
                // Ensure the float always has a decimal point.
                let s = format!("{v}");
                if s.contains('.') { s } else { format!("{s}.0") }
            }
            PropValue::Bool(v) => format!("{v}"),
            PropValue::Text(v) => format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")),
        }
    }

    fn from_json_value(s: &str) -> Option<Self> {
        let s = s.trim();
        if s == "true" {
            return Some(PropValue::Bool(true));
        }
        if s == "false" {
            return Some(PropValue::Bool(false));
        }
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            let inner = &s[1..s.len() - 1];
            let unescaped = inner.replace("\\\"", "\"").replace("\\\\", "\\");
            return Some(PropValue::Text(unescaped));
        }
        if s.contains('.') {
            if let Ok(f) = s.parse::<f64>() {
                return Some(PropValue::Float(f));
            }
        }
        if let Ok(i) = s.parse::<i64>() {
            return Some(PropValue::Int(i));
        }
        None
    }
}

impl fmt::Display for PropValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropValue::Int(v) => write!(f, "{v}"),
            PropValue::Float(v) => write!(f, "{v}"),
            PropValue::Bool(v) => write!(f, "{v}"),
            PropValue::Text(v) => write!(f, "{v}"),
        }
    }
}

// ── Component ──────────────────────────────────────────────────

/// A named component with key-value properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Component {
    pub name: String,
    pub properties: HashMap<String, PropValue>,
}

impl Component {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            properties: HashMap::new(),
        }
    }

    pub fn with_prop(mut self, key: &str, value: PropValue) -> Self {
        self.properties.insert(key.to_string(), value);
        self
    }

    pub fn set(&mut self, key: &str, value: PropValue) {
        self.properties.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&PropValue> {
        self.properties.get(key)
    }

    /// Merge another component's properties on top (override).
    pub fn merge(&mut self, other: &Component) {
        for (k, v) in &other.properties {
            self.properties.insert(k.clone(), v.clone());
        }
    }
}

// ── Prefab ─────────────────────────────────────────────────────

/// A prefab template: name, components, optional children, and optional base variant.
#[derive(Debug, Clone, PartialEq)]
pub struct Prefab {
    pub name: String,
    pub components: Vec<Component>,
    /// Nested child prefab names (resolved from the library at instantiation).
    pub children: Vec<String>,
    /// If this prefab is a variant, the name of the base prefab.
    pub base: Option<String>,
}

impl Prefab {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            components: Vec::new(),
            children: Vec::new(),
            base: None,
        }
    }

    pub fn with_component(mut self, component: Component) -> Self {
        self.components.push(component);
        self
    }

    pub fn with_child(mut self, child_name: &str) -> Self {
        self.children.push(child_name.to_string());
        self
    }

    pub fn variant_of(mut self, base_name: &str) -> Self {
        self.base = Some(base_name.to_string());
        self
    }

    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }

    /// Find a component by name.
    pub fn component(&self, name: &str) -> Option<&Component> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Find a mutable component by name.
    pub fn component_mut(&mut self, name: &str) -> Option<&mut Component> {
        self.components.iter_mut().find(|c| c.name == name)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"name\": \"{}\",\n", self.name));
        if let Some(ref base) = self.base {
            out.push_str(&format!("  \"base\": \"{base}\",\n"));
        }
        // Components.
        out.push_str("  \"components\": [\n");
        for (i, comp) in self.components.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"name\": \"{}\",\n", comp.name));
            out.push_str("      \"properties\": {\n");
            let mut props: Vec<_> = comp.properties.iter().collect();
            props.sort_by_key(|(k, _)| (*k).clone());
            for (j, (k, v)) in props.iter().enumerate() {
                let comma = if j + 1 < props.len() { "," } else { "" };
                out.push_str(&format!(
                    "        \"{}\": {}{}\n",
                    k,
                    v.to_json_value(),
                    comma
                ));
            }
            out.push_str("      }\n");
            let comma = if i + 1 < self.components.len() { "," } else { "" };
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");
        // Children.
        out.push_str("  \"children\": [");
        for (i, child) in self.children.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\"{child}\""));
        }
        out.push_str("]\n");
        out.push('}');
        out
    }

    /// Deserialize from a simple JSON string (subset parser).
    pub fn from_json(json: &str) -> Option<Self> {
        let name = extract_string_field(json, "name")?;
        let base = extract_string_field(json, "base");

        let mut prefab = Prefab::new(&name);
        prefab.base = base;

        // Parse components array.
        if let Some(comp_str) = extract_array(json, "components") {
            for obj in split_json_objects(&comp_str) {
                let cname = extract_string_field(&obj, "name")?;
                let mut comp = Component::new(&cname);
                if let Some(props_str) = extract_object(&obj, "properties") {
                    for (k, v_str) in parse_kv_pairs(&props_str) {
                        if let Some(val) = PropValue::from_json_value(&v_str) {
                            comp.set(&k, val);
                        }
                    }
                }
                prefab.components.push(comp);
            }
        }

        // Parse children array.
        if let Some(children_str) = extract_array(json, "children") {
            for child in parse_string_array(&children_str) {
                prefab.children.push(child);
            }
        }

        Some(prefab)
    }
}

// ── Minimal JSON helpers (no external deps) ────────────────────

fn extract_string_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{field}\": \"");
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_array(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{field}\": [");
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let mut depth = 1i32;
    for (i, ch) in rest.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn extract_object(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{field}\": {{");
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let mut depth = 1i32;
    for (i, ch) in rest.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn split_json_objects(s: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0i32;
    let mut start = None;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s_idx) = start {
                        objects.push(s[s_idx..=i].to_string());
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

fn parse_kv_pairs(s: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some(colon) = part.find(':') {
            let key = part[..colon].trim().trim_matches('"');
            let val = part[colon + 1..].trim();
            if !key.is_empty() {
                pairs.push((key.to_string(), val.to_string()));
            }
        }
    }
    pairs
}

fn parse_string_array(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim().trim_matches('"');
        if !part.is_empty() {
            result.push(part.to_string());
        }
    }
    result
}

// ── Instance ───────────────────────────────────────────────────

/// An instantiated entity from a prefab with optional property overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefabInstance {
    pub id: u64,
    pub source_prefab: String,
    pub components: Vec<Component>,
    pub children: Vec<u64>,
}

// ── Diff ───────────────────────────────────────────────────────

/// A difference between an instance and its source prefab.
#[derive(Debug, Clone, PartialEq)]
pub enum PrefabDiff {
    /// A property was changed from the source value.
    PropertyChanged {
        component: String,
        key: String,
        source: PropValue,
        instance: PropValue,
    },
    /// A property was added that the source does not have.
    PropertyAdded {
        component: String,
        key: String,
        value: PropValue,
    },
    /// A property from the source was removed.
    PropertyRemoved {
        component: String,
        key: String,
        value: PropValue,
    },
    /// A component was added.
    ComponentAdded(String),
    /// A component was removed.
    ComponentRemoved(String),
}

// ── Prefab library ─────────────────────────────────────────────

/// Registry of all available prefabs and instance tracking.
pub struct PrefabLibrary {
    prefabs: HashMap<String, Prefab>,
    next_instance_id: u64,
}

impl fmt::Debug for PrefabLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefabLibrary")
            .field("prefabs", &self.prefabs.len())
            .finish()
    }
}

impl PrefabLibrary {
    pub fn new() -> Self {
        Self {
            prefabs: HashMap::new(),
            next_instance_id: 1,
        }
    }

    /// Register a prefab.
    pub fn register(&mut self, prefab: Prefab) {
        self.prefabs.insert(prefab.name.clone(), prefab);
    }

    /// Unregister a prefab by name.
    pub fn unregister(&mut self, name: &str) -> Option<Prefab> {
        self.prefabs.remove(name)
    }

    /// Get a prefab by name.
    pub fn get(&self, name: &str) -> Option<&Prefab> {
        self.prefabs.get(name)
    }

    /// Number of registered prefabs.
    pub fn count(&self) -> usize {
        self.prefabs.len()
    }

    /// All registered prefab names.
    pub fn names(&self) -> Vec<String> {
        self.prefabs.keys().cloned().collect()
    }

    /// Resolve a prefab, applying variant inheritance from its base chain.
    pub fn resolve(&self, name: &str) -> Option<Prefab> {
        let prefab = self.prefabs.get(name)?.clone();
        if let Some(ref base_name) = prefab.base {
            let base = self.resolve(base_name)?;
            let merged = merge_prefab(&base, &prefab);
            Some(merged)
        } else {
            Some(prefab)
        }
    }

    /// Instantiate a prefab, resolving variants and returning a new instance.
    pub fn instantiate(&mut self, name: &str) -> Option<PrefabInstance> {
        let resolved = self.resolve(name)?;
        let id = self.next_instance_id;
        self.next_instance_id += 1;

        // Recursively instantiate children.
        let child_names: Vec<String> = resolved.children.clone();
        let mut child_ids = Vec::new();
        for child_name in &child_names {
            if let Some(child_inst) = self.instantiate(child_name) {
                child_ids.push(child_inst.id);
            }
        }

        Some(PrefabInstance {
            id,
            source_prefab: name.to_string(),
            components: resolved.components,
            children: child_ids,
        })
    }

    /// Apply overrides to an instance's components.
    pub fn apply_overrides(
        instance: &mut PrefabInstance,
        component_name: &str,
        overrides: &[(String, PropValue)],
    ) {
        if let Some(comp) = instance
            .components
            .iter_mut()
            .find(|c| c.name == component_name)
        {
            for (key, val) in overrides {
                comp.set(key, val.clone());
            }
        }
    }

    /// Compute diff between an instance and its source prefab.
    pub fn diff(&self, instance: &PrefabInstance) -> Vec<PrefabDiff> {
        let resolved = match self.resolve(&instance.source_prefab) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut diffs = Vec::new();

        // Check each source component.
        for src_comp in &resolved.components {
            if let Some(inst_comp) = instance.components.iter().find(|c| c.name == src_comp.name) {
                // Compare properties.
                let mut all_keys: Vec<String> = src_comp.properties.keys().cloned().collect();
                for k in inst_comp.properties.keys() {
                    if !all_keys.contains(k) {
                        all_keys.push(k.clone());
                    }
                }
                all_keys.sort();
                for key in &all_keys {
                    match (src_comp.properties.get(key), inst_comp.properties.get(key)) {
                        (Some(sv), Some(iv)) if sv != iv => {
                            diffs.push(PrefabDiff::PropertyChanged {
                                component: src_comp.name.clone(),
                                key: key.clone(),
                                source: sv.clone(),
                                instance: iv.clone(),
                            });
                        }
                        (None, Some(iv)) => {
                            diffs.push(PrefabDiff::PropertyAdded {
                                component: src_comp.name.clone(),
                                key: key.clone(),
                                value: iv.clone(),
                            });
                        }
                        (Some(sv), None) => {
                            diffs.push(PrefabDiff::PropertyRemoved {
                                component: src_comp.name.clone(),
                                key: key.clone(),
                                value: sv.clone(),
                            });
                        }
                        _ => {}
                    }
                }
            } else {
                diffs.push(PrefabDiff::ComponentRemoved(src_comp.name.clone()));
            }
        }

        // Check for added components.
        for inst_comp in &instance.components {
            if !resolved.components.iter().any(|c| c.name == inst_comp.name) {
                diffs.push(PrefabDiff::ComponentAdded(inst_comp.name.clone()));
            }
        }

        diffs
    }
}

impl Default for PrefabLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge variant prefab on top of base prefab.
fn merge_prefab(base: &Prefab, variant: &Prefab) -> Prefab {
    let mut merged = Prefab::new(&variant.name);
    merged.base = variant.base.clone();

    // Start with base components.
    let mut comp_map: HashMap<String, Component> = HashMap::new();
    for c in &base.components {
        comp_map.insert(c.name.clone(), c.clone());
    }
    // Override/add variant components.
    for c in &variant.components {
        if let Some(existing) = comp_map.get_mut(&c.name) {
            existing.merge(c);
        } else {
            comp_map.insert(c.name.clone(), c.clone());
        }
    }
    // Collect in sorted order for determinism.
    let mut names: Vec<String> = comp_map.keys().cloned().collect();
    names.sort();
    for name in names {
        merged.components.push(comp_map.remove(&name).unwrap());
    }

    // Children: variant overrides base.
    merged.children = if variant.children.is_empty() {
        base.children.clone()
    } else {
        variant.children.clone()
    };

    merged
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn transform_comp(x: f64, y: f64) -> Component {
        Component::new("Transform")
            .with_prop("x", PropValue::Float(x))
            .with_prop("y", PropValue::Float(y))
    }

    fn health_comp(hp: i64) -> Component {
        Component::new("Health").with_prop("hp", PropValue::Int(hp))
    }

    #[test]
    fn create_prefab() {
        let p = Prefab::new("Enemy")
            .with_component(transform_comp(0.0, 0.0))
            .with_component(health_comp(100));
        assert_eq!(p.components.len(), 2);
        assert_eq!(p.name, "Enemy");
    }

    #[test]
    fn component_get_set() {
        let mut c = Component::new("Stats");
        c.set("str", PropValue::Int(10));
        assert_eq!(c.get("str"), Some(&PropValue::Int(10)));
        assert_eq!(c.get("dex"), None);
    }

    #[test]
    fn prop_value_accessors() {
        assert_eq!(PropValue::Int(5).as_int(), Some(5));
        assert_eq!(PropValue::Float(1.5).as_float(), Some(1.5));
        assert_eq!(PropValue::Bool(true).as_bool(), Some(true));
        assert_eq!(PropValue::Text("hi".into()).as_text(), Some("hi"));
        assert_eq!(PropValue::Int(5).as_float(), None);
    }

    #[test]
    fn prefab_to_json_and_back() {
        let p = Prefab::new("Coin")
            .with_component(
                Component::new("Value")
                    .with_prop("amount", PropValue::Int(10))
                    .with_prop("name", PropValue::Text("Gold".into())),
            )
            .with_child("Sparkle");
        let json = p.to_json();
        let parsed = Prefab::from_json(&json).unwrap();
        assert_eq!(parsed.name, "Coin");
        assert_eq!(parsed.children, vec!["Sparkle"]);
        let comp = parsed.component("Value").unwrap();
        assert_eq!(comp.get("amount"), Some(&PropValue::Int(10)));
        assert_eq!(comp.get("name"), Some(&PropValue::Text("Gold".into())));
    }

    #[test]
    fn prefab_json_bool_and_float() {
        let p = Prefab::new("Test").with_component(
            Component::new("Flags")
                .with_prop("active", PropValue::Bool(true))
                .with_prop("speed", PropValue::Float(3.5)),
        );
        let json = p.to_json();
        let parsed = Prefab::from_json(&json).unwrap();
        let comp = parsed.component("Flags").unwrap();
        assert_eq!(comp.get("active"), Some(&PropValue::Bool(true)));
        let speed = comp.get("speed").unwrap().as_float().unwrap();
        assert!((speed - 3.5).abs() < 1e-9);
    }

    #[test]
    fn library_register_and_get() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("Hero"));
        assert_eq!(lib.count(), 1);
        assert!(lib.get("Hero").is_some());
    }

    #[test]
    fn library_unregister() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("Hero"));
        let removed = lib.unregister("Hero");
        assert!(removed.is_some());
        assert_eq!(lib.count(), 0);
    }

    #[test]
    fn instantiate_prefab() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("Enemy").with_component(health_comp(50)));
        let inst = lib.instantiate("Enemy").unwrap();
        assert_eq!(inst.source_prefab, "Enemy");
        assert_eq!(inst.components.len(), 1);
    }

    #[test]
    fn instance_ids_increment() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("A"));
        let i1 = lib.instantiate("A").unwrap();
        let i2 = lib.instantiate("A").unwrap();
        assert!(i2.id > i1.id);
    }

    #[test]
    fn apply_overrides() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("Enemy").with_component(health_comp(100)));
        let mut inst = lib.instantiate("Enemy").unwrap();
        PrefabLibrary::apply_overrides(
            &mut inst,
            "Health",
            &[("hp".to_string(), PropValue::Int(200))],
        );
        let hp = inst
            .components
            .iter()
            .find(|c| c.name == "Health")
            .unwrap()
            .get("hp")
            .unwrap()
            .as_int()
            .unwrap();
        assert_eq!(hp, 200);
    }

    #[test]
    fn variant_inheritance() {
        let mut lib = PrefabLibrary::new();
        lib.register(
            Prefab::new("BaseEnemy")
                .with_component(health_comp(100))
                .with_component(transform_comp(0.0, 0.0)),
        );
        lib.register(
            Prefab::new("StrongEnemy")
                .variant_of("BaseEnemy")
                .with_component(health_comp(500)),
        );
        let resolved = lib.resolve("StrongEnemy").unwrap();
        // Health overridden, Transform inherited.
        let hp = resolved
            .component("Health")
            .unwrap()
            .get("hp")
            .unwrap()
            .as_int()
            .unwrap();
        assert_eq!(hp, 500);
        assert!(resolved.component("Transform").is_some());
    }

    #[test]
    fn nested_prefab_children() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("Wheel"));
        lib.register(Prefab::new("Car").with_child("Wheel"));
        let inst = lib.instantiate("Car").unwrap();
        assert_eq!(inst.children.len(), 1);
    }

    #[test]
    fn diff_no_changes() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("A").with_component(health_comp(100)));
        let inst = lib.instantiate("A").unwrap();
        let diffs = lib.diff(&inst);
        assert!(diffs.is_empty());
    }

    #[test]
    fn diff_property_changed() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("A").with_component(health_comp(100)));
        let mut inst = lib.instantiate("A").unwrap();
        PrefabLibrary::apply_overrides(
            &mut inst,
            "Health",
            &[("hp".to_string(), PropValue::Int(999))],
        );
        let diffs = lib.diff(&inst);
        assert!(diffs.iter().any(|d| matches!(d, PrefabDiff::PropertyChanged { key, .. } if key == "hp")));
    }

    #[test]
    fn diff_component_added() {
        let mut lib = PrefabLibrary::new();
        lib.register(Prefab::new("A").with_component(health_comp(100)));
        let mut inst = lib.instantiate("A").unwrap();
        inst.components.push(Component::new("Armor"));
        let diffs = lib.diff(&inst);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, PrefabDiff::ComponentAdded(n) if n == "Armor")));
    }

    #[test]
    fn diff_component_removed() {
        let mut lib = PrefabLibrary::new();
        lib.register(
            Prefab::new("A")
                .with_component(health_comp(100))
                .with_component(transform_comp(0.0, 0.0)),
        );
        let mut inst = lib.instantiate("A").unwrap();
        inst.components.retain(|c| c.name != "Transform");
        let diffs = lib.diff(&inst);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, PrefabDiff::ComponentRemoved(n) if n == "Transform")));
    }

    #[test]
    fn component_merge() {
        let mut a = Component::new("Stats")
            .with_prop("str", PropValue::Int(10))
            .with_prop("dex", PropValue::Int(8));
        let b = Component::new("Stats").with_prop("str", PropValue::Int(20));
        a.merge(&b);
        assert_eq!(a.get("str"), Some(&PropValue::Int(20)));
        assert_eq!(a.get("dex"), Some(&PropValue::Int(8)));
    }

    #[test]
    fn prop_value_display() {
        assert_eq!(PropValue::Int(42).to_string(), "42");
        assert_eq!(PropValue::Bool(false).to_string(), "false");
    }

    #[test]
    fn instantiate_nonexistent_returns_none() {
        let mut lib = PrefabLibrary::new();
        assert!(lib.instantiate("Missing").is_none());
    }

    #[test]
    fn prefab_find_component() {
        let p = Prefab::new("T")
            .with_component(health_comp(10))
            .with_component(transform_comp(1.0, 2.0));
        assert!(p.component("Health").is_some());
        assert!(p.component("Missing").is_none());
    }
}
