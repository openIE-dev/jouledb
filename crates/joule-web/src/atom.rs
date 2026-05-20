//! Jotai/Recoil-style atomic state management.
//!
//! Provides `Atom<T>`, `DerivedAtom`, `AsyncAtomStatus`, and `AtomStore`
//! for fine-grained, composable state without a single monolithic store.

use std::any::Any;
use std::collections::HashMap;

// ── Atom ──

/// Unique identifier for an atom.
pub type AtomId = u64;

/// A typed atom descriptor with a default value.
#[derive(Debug, Clone)]
pub struct Atom<T: Clone + 'static> {
    pub id: AtomId,
    pub default_value: T,
    pub label: Option<String>,
}

impl<T: Clone + 'static> Atom<T> {
    pub fn new(id: AtomId, default_value: T) -> Self {
        Self {
            id,
            default_value,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ── Async Atom Status ──

/// Status for an asynchronous atom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncAtomStatus<T: Clone> {
    Loading,
    Ready(T),
    Error(String),
}

// ── Derived Atom ──

/// A derived atom that computes its value from other atoms.
pub struct DerivedAtom<T: Clone + 'static> {
    pub id: AtomId,
    /// IDs of atoms this derived atom depends on.
    pub dependencies: Vec<AtomId>,
    /// Compute function that receives a reader and returns the derived value.
    pub compute: Box<dyn Fn(&dyn Fn(AtomId) -> Box<dyn Any>) -> T>,
}

// ── Atom Family ──

/// A factory that creates parameterized atoms from a key.
pub struct AtomFamily<K, T: Clone + 'static> {
    base_id: AtomId,
    default_fn: Box<dyn Fn(&K) -> T>,
    created: HashMap<u64, AtomId>,
}

impl<K, T: Clone + 'static> AtomFamily<K, T>
where
    K: std::hash::Hash + Eq,
{
    /// Create a new atom family with a base ID and a function producing defaults.
    pub fn new(base_id: AtomId, default_fn: impl Fn(&K) -> T + 'static) -> Self {
        Self {
            base_id,
            default_fn: Box::new(default_fn),
            created: HashMap::new(),
        }
    }

    /// Get or create an atom for the given key. The atom ID is deterministic
    /// based on the hash of the key combined with the base ID.
    pub fn get(&mut self, key: &K) -> Atom<T>
    where
        K: std::hash::Hash,
    {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let key_hash = std::hash::Hasher::finish(&hasher);
        let atom_id = self.base_id.wrapping_add(key_hash);
        self.created.insert(key_hash, atom_id);
        let default = (self.default_fn)(key);
        Atom::new(atom_id, default)
    }

    /// Number of atoms created by this family.
    pub fn len(&self) -> usize {
        self.created.len()
    }

    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
    }
}

// ── AtomStore ──

/// Central store holding current values of all atoms.
pub struct AtomStore {
    values: HashMap<AtomId, Box<dyn Any>>,
    defaults: HashMap<AtomId, Box<dyn Any>>,
    /// Derived atom definitions.
    derived: HashMap<AtomId, (Vec<AtomId>, Box<dyn Fn(&dyn Fn(AtomId) -> Box<dyn Any>) -> Box<dyn Any>>)>,
    /// Async atom statuses.
    async_status: HashMap<AtomId, Box<dyn Any>>,
}

impl Default for AtomStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomStore {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            defaults: HashMap::new(),
            derived: HashMap::new(),
            async_status: HashMap::new(),
        }
    }

    /// Read an atom's current value, or its default if not yet set.
    pub fn read<T: Clone + 'static>(&self, atom: &Atom<T>) -> T {
        if let Some(boxed) = self.values.get(&atom.id) {
            if let Some(val) = boxed.downcast_ref::<T>() {
                return val.clone();
            }
        }
        atom.default_value.clone()
    }

    /// Write a value to an atom.
    pub fn write<T: Clone + 'static>(&mut self, atom: &Atom<T>, value: T) {
        self.defaults
            .entry(atom.id)
            .or_insert_with(|| Box::new(atom.default_value.clone()));
        self.values.insert(atom.id, Box::new(value));
    }

    /// Reset an atom to its default value.
    pub fn reset<T: Clone + 'static>(&mut self, atom: &Atom<T>) {
        self.values.insert(atom.id, Box::new(atom.default_value.clone()));
    }

    /// Register a derived atom.
    pub fn register_derived<T: Clone + 'static>(&mut self, derived: DerivedAtom<T>) {
        let compute = derived.compute;
        let wrapper: Box<dyn Fn(&dyn Fn(AtomId) -> Box<dyn Any>) -> Box<dyn Any>> =
            Box::new(move |reader| {
                let val = compute(reader);
                Box::new(val)
            });
        self.derived.insert(derived.id, (derived.dependencies, wrapper));
    }

    /// Read a derived atom's computed value.
    pub fn read_derived<T: Clone + 'static>(&self, id: AtomId) -> Option<T> {
        let (_, compute) = self.derived.get(&id)?;
        let reader = |dep_id: AtomId| -> Box<dyn Any> {
            if let Some(val) = self.values.get(&dep_id) {
                // Clone the box contents — we need to return owned data.
                // We'll try common types.
                clone_any_box(val)
            } else {
                Box::new(())
            }
        };
        let result = compute(&reader);
        result.downcast::<T>().ok().map(|b| *b)
    }

    /// Set the async status for an atom.
    pub fn set_async_status<T: Clone + 'static>(&mut self, id: AtomId, status: AsyncAtomStatus<T>) {
        self.async_status.insert(id, Box::new(status));
    }

    /// Get the async status for an atom.
    pub fn get_async_status<T: Clone + 'static>(&self, id: AtomId) -> Option<AsyncAtomStatus<T>> {
        self.async_status
            .get(&id)
            .and_then(|b| b.downcast_ref::<AsyncAtomStatus<T>>())
            .cloned()
    }

    /// Check whether an atom has been explicitly set.
    pub fn has(&self, id: AtomId) -> bool {
        self.values.contains_key(&id)
    }

    /// Remove an atom's value from the store entirely.
    pub fn remove(&mut self, id: AtomId) {
        self.values.remove(&id);
        self.defaults.remove(&id);
    }
}

/// Clone a boxed Any by trying common types. For the store's purposes,
/// we support i32, i64, f64, String, bool, and Vec<i32>.
fn clone_any_box(val: &Box<dyn Any>) -> Box<dyn Any> {
    if let Some(v) = val.downcast_ref::<i32>() {
        return Box::new(*v);
    }
    if let Some(v) = val.downcast_ref::<i64>() {
        return Box::new(*v);
    }
    if let Some(v) = val.downcast_ref::<f64>() {
        return Box::new(*v);
    }
    if let Some(v) = val.downcast_ref::<String>() {
        return Box::new(v.clone());
    }
    if let Some(v) = val.downcast_ref::<bool>() {
        return Box::new(*v);
    }
    if let Some(v) = val.downcast_ref::<u64>() {
        return Box::new(*v);
    }
    Box::new(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atom_read_default() {
        let store = AtomStore::new();
        let counter = Atom::new(1, 0i32);
        assert_eq!(store.read(&counter), 0);
    }

    #[test]
    fn atom_write_and_read() {
        let mut store = AtomStore::new();
        let counter = Atom::new(1, 0i32);
        store.write(&counter, 42);
        assert_eq!(store.read(&counter), 42);
    }

    #[test]
    fn atom_reset() {
        let mut store = AtomStore::new();
        let counter = Atom::new(1, 10i32);
        store.write(&counter, 99);
        assert_eq!(store.read(&counter), 99);
        store.reset(&counter);
        assert_eq!(store.read(&counter), 10);
    }

    #[test]
    fn atom_with_label() {
        let a = Atom::new(1, 0i32).with_label("counter");
        assert_eq!(a.label.as_deref(), Some("counter"));
    }

    #[test]
    fn derived_atom() {
        let mut store = AtomStore::new();
        let a = Atom::new(1, 3i32);
        let b = Atom::new(2, 7i32);
        store.write(&a, 3);
        store.write(&b, 7);

        let derived = DerivedAtom {
            id: 100,
            dependencies: vec![1, 2],
            compute: Box::new(|reader| {
                let va: i32 = *reader(1).downcast::<i32>().unwrap();
                let vb: i32 = *reader(2).downcast::<i32>().unwrap();
                va + vb
            }),
        };
        store.register_derived(derived);

        let result: Option<i32> = store.read_derived(100);
        assert_eq!(result, Some(10));
    }

    #[test]
    fn derived_atom_updates_with_deps() {
        let mut store = AtomStore::new();
        let a = Atom::new(1, 0i32);
        store.write(&a, 5);

        let derived = DerivedAtom {
            id: 100,
            dependencies: vec![1],
            compute: Box::new(|reader| {
                let v: i32 = *reader(1).downcast::<i32>().unwrap();
                v * 2
            }),
        };
        store.register_derived(derived);

        assert_eq!(store.read_derived::<i32>(100), Some(10));

        store.write(&a, 20);
        assert_eq!(store.read_derived::<i32>(100), Some(40));
    }

    #[test]
    fn async_atom_status() {
        let mut store = AtomStore::new();
        store.set_async_status::<String>(50, AsyncAtomStatus::Loading);
        assert_eq!(
            store.get_async_status::<String>(50),
            Some(AsyncAtomStatus::Loading)
        );

        store.set_async_status(50, AsyncAtomStatus::Ready("done".to_string()));
        assert_eq!(
            store.get_async_status::<String>(50),
            Some(AsyncAtomStatus::Ready("done".to_string()))
        );

        store.set_async_status::<String>(50, AsyncAtomStatus::Error("fail".into()));
        assert_eq!(
            store.get_async_status::<String>(50),
            Some(AsyncAtomStatus::Error("fail".into()))
        );
    }

    #[test]
    fn atom_family() {
        let mut store = AtomStore::new();
        let mut family = AtomFamily::new(1000, |key: &String| format!("default-{key}"));
        assert!(family.is_empty());

        let atom_a = family.get(&"user_1".to_string());
        let atom_b = family.get(&"user_2".to_string());
        assert_eq!(family.len(), 2);

        assert_eq!(store.read(&atom_a), "default-user_1");
        store.write(&atom_a, "Alice".to_string());
        assert_eq!(store.read(&atom_a), "Alice");
        assert_eq!(store.read(&atom_b), "default-user_2");
    }

    #[test]
    fn has_and_remove() {
        let mut store = AtomStore::new();
        let a = Atom::new(1, 0i32);
        assert!(!store.has(1));
        store.write(&a, 5);
        assert!(store.has(1));
        store.remove(1);
        assert!(!store.has(1));
        // Reading after remove returns default
        assert_eq!(store.read(&a), 0);
    }

    #[test]
    fn multiple_atoms_independent() {
        let mut store = AtomStore::new();
        let name = Atom::new(1, String::new());
        let age = Atom::new(2, 0i32);
        let active = Atom::new(3, false);

        store.write(&name, "Alice".to_string());
        store.write(&age, 30);
        store.write(&active, true);

        assert_eq!(store.read(&name), "Alice");
        assert_eq!(store.read(&age), 30);
        assert_eq!(store.read(&active), true);
    }

    #[test]
    fn nonexistent_derived_returns_none() {
        let store = AtomStore::new();
        assert_eq!(store.read_derived::<i32>(999), None);
    }
}
