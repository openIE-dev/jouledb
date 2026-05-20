//! Functional lenses and optics for Rust.
//!
//! Provides `Lens` (get/set for nested fields), `Prism` (for enum variants),
//! `Traversal` (for collections), lens composition, `over` (modify through
//! a lens), and `JsonLens` for dynamic access into `serde_json::Value` trees.

use serde_json::Value;
use std::fmt;

// ── Lens ────────────────────────────────────────────────────────────────────

/// A lens focuses on a part `B` within a whole `A`.
///
/// It provides `get` to extract and `set` to produce a new whole with
/// the focused part replaced.
pub struct Lens<A, B> {
    getter: Box<dyn Fn(&A) -> B>,
    setter: Box<dyn Fn(&A, B) -> A>,
}

impl<A: 'static, B: 'static> Lens<A, B> {
    /// Create a lens from a getter and a setter.
    pub fn new(
        getter: impl Fn(&A) -> B + 'static,
        setter: impl Fn(&A, B) -> A + 'static,
    ) -> Self {
        Self {
            getter: Box::new(getter),
            setter: Box::new(setter),
        }
    }

    /// Get the focused value from the whole.
    pub fn get(&self, whole: &A) -> B {
        (self.getter)(whole)
    }

    /// Set the focused value, returning a new whole.
    pub fn set(&self, whole: &A, value: B) -> A {
        (self.setter)(whole, value)
    }

    /// Modify the focused value by applying a function, returning a new whole.
    pub fn over(&self, whole: &A, f: impl FnOnce(B) -> B) -> A {
        let current = self.get(whole);
        self.set(whole, f(current))
    }

    /// Compose this lens with another lens that focuses deeper.
    pub fn compose<C: 'static>(self, other: Lens<B, C>) -> Lens<A, C>
    where
        A: Clone,
        B: Clone,
    {
        use std::rc::Rc;

        let g1: Rc<dyn Fn(&A) -> B> = Rc::from(self.getter);
        let s1: Rc<dyn Fn(&A, B) -> A> = Rc::from(self.setter);
        let g2: Rc<dyn Fn(&B) -> C> = Rc::from(other.getter);
        let s2: Rc<dyn Fn(&B, C) -> B> = Rc::from(other.setter);

        let g1_for_setter = g1.clone();
        Lens {
            getter: Box::new(move |a: &A| {
                let b = g1(a);
                g2(&b)
            }),
            setter: Box::new(move |a: &A, c: C| {
                let b = g1_for_setter(a);
                let new_b = s2(&b, c);
                s1(a, new_b)
            }),
        }
    }
}

impl<A: 'static, B: fmt::Debug + 'static> Lens<A, B> {
    /// Debug-view the focused value.
    pub fn view_debug(&self, whole: &A) -> String {
        format!("{:?}", self.get(whole))
    }
}

// ── Prism ───────────────────────────────────────────────────────────────────

/// A prism focuses on one variant of a sum type.
///
/// `preview` tries to extract the value (returning `None` if the variant
/// doesn't match) and `review` constructs the whole from the part.
pub struct Prism<A, B> {
    preview_fn: Box<dyn Fn(&A) -> Option<B>>,
    review_fn: Box<dyn Fn(B) -> A>,
}

impl<A: 'static, B: 'static> Prism<A, B> {
    /// Create a prism from preview and review functions.
    pub fn new(
        preview: impl Fn(&A) -> Option<B> + 'static,
        review: impl Fn(B) -> A + 'static,
    ) -> Self {
        Self {
            preview_fn: Box::new(preview),
            review_fn: Box::new(review),
        }
    }

    /// Try to extract the focused variant.
    pub fn preview(&self, whole: &A) -> Option<B> {
        (self.preview_fn)(whole)
    }

    /// Construct the whole from the part.
    pub fn review(&self, part: B) -> A {
        (self.review_fn)(part)
    }

    /// Modify the focused variant if it matches, otherwise return unchanged.
    pub fn over(&self, whole: &A, f: impl FnOnce(B) -> B) -> A
    where
        A: Clone,
    {
        match self.preview(whole) {
            Some(b) => (self.review_fn)(f(b)),
            None => whole.clone(),
        }
    }

    /// Compose with another prism (prism after prism).
    pub fn compose<C: 'static>(self, other: Prism<B, C>) -> Prism<A, C>
    where
        B: Clone + 'static,
    {
        let p1 = self.preview_fn;
        let r1 = self.review_fn;
        let p2 = other.preview_fn;
        let r2 = other.review_fn;

        Prism {
            preview_fn: Box::new(move |a: &A| {
                let b = (p1)(a)?;
                (p2)(&b)
            }),
            review_fn: Box::new(move |c: C| {
                let b = (r2)(c);
                (r1)(b)
            }),
        }
    }
}

// ── Traversal ───────────────────────────────────────────────────────────────

/// A traversal focuses on zero or more parts within a whole.
///
/// Useful for operating on all elements of a collection within a struct.
pub struct Traversal<A, B> {
    get_all_fn: Box<dyn Fn(&A) -> Vec<B>>,
    modify_all_fn: Box<dyn Fn(&A, &dyn Fn(&B) -> B) -> A>,
}

impl<A: 'static, B: 'static> Traversal<A, B> {
    /// Create a traversal from get-all and modify-all functions.
    pub fn new(
        get_all: impl Fn(&A) -> Vec<B> + 'static,
        modify_all: impl Fn(&A, &dyn Fn(&B) -> B) -> A + 'static,
    ) -> Self {
        Self {
            get_all_fn: Box::new(get_all),
            modify_all_fn: Box::new(modify_all),
        }
    }

    /// Get all focused values.
    pub fn get_all(&self, whole: &A) -> Vec<B> {
        (self.get_all_fn)(whole)
    }

    /// Modify all focused values.
    pub fn modify_all(&self, whole: &A, f: &dyn Fn(&B) -> B) -> A {
        (self.modify_all_fn)(whole, f)
    }

    /// Count the focused elements.
    pub fn count(&self, whole: &A) -> usize {
        self.get_all(whole).len()
    }

    /// Check if any focused element satisfies a predicate.
    pub fn any(&self, whole: &A, pred: impl Fn(&B) -> bool) -> bool {
        self.get_all(whole).iter().any(pred)
    }

    /// Check if all focused elements satisfy a predicate.
    pub fn all(&self, whole: &A, pred: impl Fn(&B) -> bool) -> bool {
        self.get_all(whole).iter().all(pred)
    }

    /// Find the first focused element satisfying a predicate.
    pub fn find(&self, whole: &A, pred: impl Fn(&B) -> bool) -> Option<B> {
        self.get_all(whole).into_iter().find(|b| pred(b))
    }
}

// ── JSON Lens ───────────────────────────────────────────────────────────────

/// A path-based lens into a `serde_json::Value` tree.
///
/// Supports dotted paths like `"user.address.city"` and array indices
/// like `"users.0.name"`.
#[derive(Debug, Clone)]
pub struct JsonLens {
    segments: Vec<JsonSegment>,
}

/// A single segment in a JSON path.
#[derive(Debug, Clone)]
enum JsonSegment {
    Key(String),
    Index(usize),
}

impl JsonLens {
    /// Parse a dotted path into a `JsonLens`.
    ///
    /// Segments that parse as `usize` are treated as array indices,
    /// everything else as object keys.
    pub fn new(path: &str) -> Self {
        let segments = path
            .split('.')
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Ok(idx) = s.parse::<usize>() {
                    JsonSegment::Index(idx)
                } else {
                    JsonSegment::Key(s.to_string())
                }
            })
            .collect();
        Self { segments }
    }

    /// Get the focused value, or `None` if the path doesn't exist.
    pub fn get<'a>(&self, root: &'a Value) -> Option<&'a Value> {
        let mut current = root;
        for seg in &self.segments {
            current = match seg {
                JsonSegment::Key(k) => current.get(k)?,
                JsonSegment::Index(i) => current.get(i)?,
            };
        }
        Some(current)
    }

    /// Set the focused value, returning a new `Value` tree.
    /// Creates intermediate objects/arrays as needed.
    pub fn set(&self, root: &Value, value: Value) -> Value {
        if self.segments.is_empty() {
            return value;
        }
        self.set_recursive(root, &self.segments, value)
    }

    fn set_recursive(&self, current: &Value, segments: &[JsonSegment], value: Value) -> Value {
        if segments.is_empty() {
            return value;
        }
        let (head, tail) = (&segments[0], &segments[1..]);
        match head {
            JsonSegment::Key(k) => {
                let mut obj = match current {
                    Value::Object(map) => map.clone(),
                    _ => serde_json::Map::new(),
                };
                let child = obj.get(k).cloned().unwrap_or(Value::Null);
                let new_child = if tail.is_empty() {
                    value
                } else {
                    self.set_recursive(&child, tail, value)
                };
                obj.insert(k.clone(), new_child);
                Value::Object(obj)
            }
            JsonSegment::Index(i) => {
                let mut arr = match current {
                    Value::Array(a) => a.clone(),
                    _ => Vec::new(),
                };
                // Extend with nulls if needed.
                while arr.len() <= *i {
                    arr.push(Value::Null);
                }
                let child = &arr[*i];
                arr[*i] = if tail.is_empty() {
                    value
                } else {
                    self.set_recursive(child, tail, value)
                };
                Value::Array(arr)
            }
        }
    }

    /// Modify the focused value using a function.
    pub fn over(&self, root: &Value, f: impl FnOnce(&Value) -> Value) -> Value {
        match self.get(root) {
            Some(v) => {
                let new_val = f(v);
                self.set(root, new_val)
            }
            None => root.clone(),
        }
    }

    /// Compose two JSON lenses (concatenate their paths).
    pub fn compose(&self, other: &JsonLens) -> JsonLens {
        let mut segments = self.segments.clone();
        segments.extend(other.segments.clone());
        JsonLens { segments }
    }

    /// Delete the focused key/index, returning a new tree.
    pub fn delete(&self, root: &Value) -> Value {
        if self.segments.is_empty() {
            return Value::Null;
        }
        self.delete_recursive(root, &self.segments)
    }

    fn delete_recursive(&self, current: &Value, segments: &[JsonSegment]) -> Value {
        if segments.len() == 1 {
            match &segments[0] {
                JsonSegment::Key(k) => {
                    if let Value::Object(map) = current {
                        let mut m = map.clone();
                        m.remove(k);
                        return Value::Object(m);
                    }
                    current.clone()
                }
                JsonSegment::Index(i) => {
                    if let Value::Array(arr) = current {
                        let mut a = arr.clone();
                        if *i < a.len() {
                            a.remove(*i);
                        }
                        return Value::Array(a);
                    }
                    current.clone()
                }
            }
        } else {
            let (head, tail) = (&segments[0], &segments[1..]);
            match head {
                JsonSegment::Key(k) => {
                    if let Value::Object(map) = current {
                        let mut m = map.clone();
                        if let Some(child) = m.get(k) {
                            let new_child = self.delete_recursive(child, tail);
                            m.insert(k.clone(), new_child);
                        }
                        return Value::Object(m);
                    }
                    current.clone()
                }
                JsonSegment::Index(i) => {
                    if let Value::Array(arr) = current {
                        let mut a = arr.clone();
                        if *i < a.len() {
                            let child = &a[*i];
                            a[*i] = self.delete_recursive(child, tail);
                        }
                        return Value::Array(a);
                    }
                    current.clone()
                }
            }
        }
    }

    /// Check if the path exists in the value tree.
    pub fn exists(&self, root: &Value) -> bool {
        self.get(root).is_some()
    }

    /// Get the path as a string.
    pub fn path_string(&self) -> String {
        self.segments
            .iter()
            .map(|s| match s {
                JsonSegment::Key(k) => k.clone(),
                JsonSegment::Index(i) => i.to_string(),
            })
            .collect::<Vec<_>>()
            .join(".")
    }
}

// ── Helper: lens for tuples ─────────────────────────────────────────────────

/// Create a lens focusing on the first element of a pair.
pub fn fst_lens<A: Clone + 'static, B: Clone + 'static>() -> Lens<(A, B), A> {
    Lens::new(
        |pair: &(A, B)| pair.0.clone(),
        |pair: &(A, B), a: A| (a, pair.1.clone()),
    )
}

/// Create a lens focusing on the second element of a pair.
pub fn snd_lens<A: Clone + 'static, B: Clone + 'static>() -> Lens<(A, B), B> {
    Lens::new(
        |pair: &(A, B)| pair.1.clone(),
        |pair: &(A, B), b: B| (pair.0.clone(), b),
    )
}

/// Create a lens for a named field on a struct using closures.
/// This is a convenience alias for `Lens::new`.
pub fn field_lens<A: 'static, B: 'static>(
    getter: impl Fn(&A) -> B + 'static,
    setter: impl Fn(&A, B) -> A + 'static,
) -> Lens<A, B> {
    Lens::new(getter, setter)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- Struct for testing ---

    #[derive(Debug, Clone, PartialEq)]
    struct Address {
        city: String,
        zip: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Person {
        name: String,
        age: u32,
        address: Address,
    }

    fn name_lens() -> Lens<Person, String> {
        Lens::new(
            |p: &Person| p.name.clone(),
            |p: &Person, n: String| Person {
                name: n,
                ..p.clone()
            },
        )
    }

    fn address_lens() -> Lens<Person, Address> {
        Lens::new(
            |p: &Person| p.address.clone(),
            |p: &Person, a: Address| Person {
                address: a,
                ..p.clone()
            },
        )
    }

    fn city_lens() -> Lens<Address, String> {
        Lens::new(
            |a: &Address| a.city.clone(),
            |a: &Address, c: String| Address {
                city: c,
                ..a.clone()
            },
        )
    }

    fn make_person() -> Person {
        Person {
            name: "Alice".to_string(),
            age: 30,
            address: Address {
                city: "Boston".to_string(),
                zip: "02101".to_string(),
            },
        }
    }

    #[test]
    fn lens_get() {
        let p = make_person();
        assert_eq!(name_lens().get(&p), "Alice");
    }

    #[test]
    fn lens_set() {
        let p = make_person();
        let p2 = name_lens().set(&p, "Bob".to_string());
        assert_eq!(p2.name, "Bob");
        // Original is unchanged.
        assert_eq!(p.name, "Alice");
    }

    #[test]
    fn lens_over() {
        let p = make_person();
        let p2 = name_lens().over(&p, |n| n.to_uppercase());
        assert_eq!(p2.name, "ALICE");
    }

    #[test]
    fn lens_compose() {
        let p = make_person();
        let city = address_lens().compose(city_lens());
        assert_eq!(city.get(&p), "Boston");
        let p2 = city.set(&p, "NYC".to_string());
        assert_eq!(p2.address.city, "NYC");
    }

    #[test]
    fn lens_compose_over() {
        let p = make_person();
        let city = address_lens().compose(city_lens());
        let p2 = city.over(&p, |c| format!("{c}!"));
        assert_eq!(p2.address.city, "Boston!");
    }

    #[test]
    fn fst_snd_lens() {
        let pair = (10, "hello");
        let f = fst_lens();
        let s = snd_lens();
        assert_eq!(f.get(&pair), 10);
        assert_eq!(s.get(&pair), "hello");
        let pair2 = f.set(&pair, 20);
        assert_eq!(pair2, (20, "hello"));
    }

    #[test]
    fn prism_preview_some() {
        let prism: Prism<Result<i32, String>, i32> = Prism::new(
            |r: &Result<i32, String>| r.as_ref().ok().copied(),
            |v: i32| Ok(v),
        );
        assert_eq!(prism.preview(&Ok(42)), Some(42));
    }

    #[test]
    fn prism_preview_none() {
        let prism: Prism<Result<i32, String>, i32> = Prism::new(
            |r: &Result<i32, String>| r.as_ref().ok().copied(),
            |v: i32| Ok(v),
        );
        assert_eq!(prism.preview(&Err("no".to_string())), None);
    }

    #[test]
    fn prism_review() {
        let prism: Prism<Result<i32, String>, i32> = Prism::new(
            |r: &Result<i32, String>| r.as_ref().ok().copied(),
            |v: i32| Ok(v),
        );
        assert_eq!(prism.review(99), Ok(99));
    }

    #[test]
    fn prism_over_matches() {
        let prism: Prism<Result<i32, String>, i32> = Prism::new(
            |r: &Result<i32, String>| r.as_ref().ok().copied(),
            |v: i32| Ok(v),
        );
        let r = prism.over(&Ok(10), |x| x * 2);
        assert_eq!(r, Ok(20));
    }

    #[test]
    fn prism_over_no_match() {
        let prism: Prism<Result<i32, String>, i32> = Prism::new(
            |r: &Result<i32, String>| r.as_ref().ok().copied(),
            |v: i32| Ok(v),
        );
        let r: Result<i32, String> = Err("fail".to_string());
        let r2 = prism.over(&r, |x| x * 2);
        assert_eq!(r2, Err("fail".to_string()));
    }

    #[test]
    fn traversal_get_all() {
        let trav: Traversal<Vec<i32>, i32> = Traversal::new(
            |v: &Vec<i32>| v.clone(),
            |v: &Vec<i32>, f: &dyn Fn(&i32) -> i32| v.iter().map(f).collect(),
        );
        assert_eq!(trav.get_all(&vec![1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn traversal_modify_all() {
        let trav: Traversal<Vec<i32>, i32> = Traversal::new(
            |v: &Vec<i32>| v.clone(),
            |v: &Vec<i32>, f: &dyn Fn(&i32) -> i32| v.iter().map(f).collect(),
        );
        let result = trav.modify_all(&vec![1, 2, 3], &|x| x * 10);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn traversal_any_all() {
        let trav: Traversal<Vec<i32>, i32> = Traversal::new(
            |v: &Vec<i32>| v.clone(),
            |v: &Vec<i32>, f: &dyn Fn(&i32) -> i32| v.iter().map(f).collect(),
        );
        assert!(trav.any(&vec![1, 2, 3], |x| *x == 2));
        assert!(trav.all(&vec![1, 2, 3], |x| *x > 0));
        assert!(!trav.all(&vec![1, 2, 3], |x| *x > 1));
    }

    #[test]
    fn json_lens_get() {
        let data = json!({"user": {"name": "Alice", "age": 30}});
        let lens = JsonLens::new("user.name");
        assert_eq!(lens.get(&data), Some(&json!("Alice")));
    }

    #[test]
    fn json_lens_get_nested() {
        let data = json!({"a": {"b": {"c": 42}}});
        let lens = JsonLens::new("a.b.c");
        assert_eq!(lens.get(&data), Some(&json!(42)));
    }

    #[test]
    fn json_lens_get_array() {
        let data = json!({"users": [{"name": "A"}, {"name": "B"}]});
        let lens = JsonLens::new("users.1.name");
        assert_eq!(lens.get(&data), Some(&json!("B")));
    }

    #[test]
    fn json_lens_get_missing() {
        let data = json!({"a": 1});
        let lens = JsonLens::new("b.c");
        assert_eq!(lens.get(&data), None);
    }

    #[test]
    fn json_lens_set() {
        let data = json!({"user": {"name": "Alice"}});
        let lens = JsonLens::new("user.name");
        let result = lens.set(&data, json!("Bob"));
        assert_eq!(result, json!({"user": {"name": "Bob"}}));
    }

    #[test]
    fn json_lens_set_creates_path() {
        let data = json!({});
        let lens = JsonLens::new("a.b");
        let result = lens.set(&data, json!(42));
        assert_eq!(result, json!({"a": {"b": 42}}));
    }

    #[test]
    fn json_lens_over() {
        let data = json!({"count": 10});
        let lens = JsonLens::new("count");
        let result = lens.over(&data, |v| {
            json!(v.as_i64().unwrap_or(0) + 1)
        });
        assert_eq!(result, json!({"count": 11}));
    }

    #[test]
    fn json_lens_compose() {
        let l1 = JsonLens::new("user");
        let l2 = JsonLens::new("address.city");
        let composed = l1.compose(&l2);
        let data = json!({"user": {"address": {"city": "Boston"}}});
        assert_eq!(composed.get(&data), Some(&json!("Boston")));
    }

    #[test]
    fn json_lens_delete() {
        let data = json!({"a": 1, "b": 2});
        let lens = JsonLens::new("b");
        let result = lens.delete(&data);
        assert_eq!(result, json!({"a": 1}));
    }

    #[test]
    fn json_lens_exists() {
        let data = json!({"a": {"b": 1}});
        assert!(JsonLens::new("a.b").exists(&data));
        assert!(!JsonLens::new("a.c").exists(&data));
    }

    #[test]
    fn json_lens_path_string() {
        let lens = JsonLens::new("user.address.city");
        assert_eq!(lens.path_string(), "user.address.city");
    }

    #[test]
    fn json_lens_delete_nested() {
        let data = json!({"a": {"b": 1, "c": 2}});
        let lens = JsonLens::new("a.b");
        let result = lens.delete(&data);
        assert_eq!(result, json!({"a": {"c": 2}}));
    }

    #[test]
    fn traversal_find() {
        let trav: Traversal<Vec<i32>, i32> = Traversal::new(
            |v: &Vec<i32>| v.clone(),
            |v: &Vec<i32>, f: &dyn Fn(&i32) -> i32| v.iter().map(f).collect(),
        );
        assert_eq!(trav.find(&vec![1, 2, 3], |x| *x == 2), Some(2));
        assert_eq!(trav.find(&vec![1, 2, 3], |x| *x == 5), None);
    }

    #[test]
    fn traversal_count() {
        let trav: Traversal<Vec<i32>, i32> = Traversal::new(
            |v: &Vec<i32>| v.clone(),
            |v: &Vec<i32>, f: &dyn Fn(&i32) -> i32| v.iter().map(f).collect(),
        );
        assert_eq!(trav.count(&vec![1, 2, 3]), 3);
        assert_eq!(trav.count(&Vec::<i32>::new()), 0);
    }

    #[test]
    fn lens_view_debug() {
        let p = make_person();
        let s = name_lens().view_debug(&p);
        assert!(s.contains("Alice"));
    }

    #[test]
    fn prism_compose() {
        // Prism into Option then prism into Result inside that.
        // Simplified: Prism<Option<Result<i32, String>>, Result<i32, String>>
        // composed with Prism<Result<i32, String>, i32>
        let p1: Prism<Option<Result<i32, String>>, Result<i32, String>> = Prism::new(
            |o: &Option<Result<i32, String>>| o.clone(),
            |r: Result<i32, String>| Some(r),
        );
        let p2: Prism<Result<i32, String>, i32> = Prism::new(
            |r: &Result<i32, String>| r.as_ref().ok().copied(),
            |v: i32| Ok(v),
        );
        let composed = p1.compose(p2);
        assert_eq!(composed.preview(&Some(Ok(42))), Some(42));
        assert_eq!(composed.preview(&Some(Err("x".to_string()))), None);
        assert_eq!(composed.preview(&None), None);
        assert_eq!(composed.review(10), Some(Ok(10)));
    }
}
