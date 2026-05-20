//! Heap snapshot model: objects, retained size via dominator tree,
//! grouping by type, snapshot diffing, and root object detection.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Heap Object ──────────────────────────────────────────────────

/// A single object in the heap.
#[derive(Debug, Clone)]
pub struct HeapObject {
    pub id: u64,
    pub type_name: String,
    pub shallow_size: usize,
    pub references: Vec<u64>,
}

impl HeapObject {
    pub fn new(id: u64, type_name: &str, shallow_size: usize) -> Self {
        Self {
            id,
            type_name: type_name.to_string(),
            shallow_size,
            references: Vec::new(),
        }
    }

    pub fn with_refs(mut self, refs: &[u64]) -> Self {
        self.references = refs.to_vec();
        self
    }
}

// ── Type Group ───────────────────────────────────────────────────

/// Summary of objects grouped by type.
#[derive(Debug, Clone)]
pub struct TypeGroup {
    pub type_name: String,
    pub count: usize,
    pub total_shallow_size: usize,
}

// ── Snapshot Diff ────────────────────────────────────────────────

/// Difference between two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub new_objects: Vec<u64>,
    pub removed_objects: Vec<u64>,
    pub size_delta_by_type: HashMap<String, i64>,
    pub total_size_delta: i64,
}

// ── Heap Snapshot ────────────────────────────────────────────────

/// A snapshot of the heap at a point in time.
#[derive(Debug, Clone)]
pub struct HeapSnapshot {
    objects: HashMap<u64, HeapObject>,
    /// Virtual root that can reach all real roots.
    root_id: u64,
}

impl HeapSnapshot {
    /// Create a new snapshot. A synthetic root (id=0) is added that references
    /// all root objects (objects not referenced by any other object).
    pub fn new(objects: Vec<HeapObject>) -> Self {
        let mut obj_map: HashMap<u64, HeapObject> = HashMap::new();
        let mut referenced: HashSet<u64> = HashSet::new();

        for obj in &objects {
            for r in &obj.references {
                referenced.insert(*r);
            }
        }

        let root_refs: Vec<u64> = objects
            .iter()
            .filter(|o| !referenced.contains(&o.id))
            .map(|o| o.id)
            .collect();

        for obj in objects {
            obj_map.insert(obj.id, obj);
        }

        // Synthetic root
        let root_id = 0;
        obj_map.insert(
            root_id,
            HeapObject {
                id: root_id,
                type_name: "(root)".to_string(),
                shallow_size: 0,
                references: root_refs,
            },
        );

        Self {
            objects: obj_map,
            root_id,
        }
    }

    /// Get an object by id.
    pub fn get(&self, id: u64) -> Option<&HeapObject> {
        self.objects.get(&id)
    }

    /// All object ids (excluding synthetic root).
    pub fn object_ids(&self) -> Vec<u64> {
        self.objects
            .keys()
            .copied()
            .filter(|id| *id != self.root_id)
            .collect()
    }

    /// Root objects (not referenced by any other object, excluding synthetic root).
    pub fn root_objects(&self) -> Vec<u64> {
        self.objects
            .get(&self.root_id)
            .map(|r| r.references.clone())
            .unwrap_or_default()
    }

    /// Total shallow size of all objects.
    pub fn total_shallow_size(&self) -> usize {
        self.objects
            .values()
            .filter(|o| o.id != self.root_id)
            .map(|o| o.shallow_size)
            .sum()
    }

    /// Group objects by type name.
    pub fn group_by_type(&self) -> Vec<TypeGroup> {
        let mut groups: HashMap<&str, (usize, usize)> = HashMap::new();
        for obj in self.objects.values() {
            if obj.id == self.root_id {
                continue;
            }
            let entry = groups.entry(&obj.type_name).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += obj.shallow_size;
        }
        let mut result: Vec<TypeGroup> = groups
            .into_iter()
            .map(|(name, (count, size))| TypeGroup {
                type_name: name.to_string(),
                count,
                total_shallow_size: size,
            })
            .collect();
        result.sort_by(|a, b| b.total_shallow_size.cmp(&a.total_shallow_size));
        result
    }

    /// Compute the dominator tree from the synthetic root using the iterative
    /// algorithm (Cooper, Harvey, Kennedy). Returns a map: node -> immediate dominator.
    pub fn dominator_tree(&self) -> HashMap<u64, u64> {
        let all_ids: Vec<u64> = self.objects.keys().copied().collect();
        if all_ids.is_empty() {
            return HashMap::new();
        }

        // BFS to get reverse post-order
        let mut visited = HashSet::new();
        let rpo;
        // Use a DFS post-order then reverse
        let mut post_order = Vec::new();
        let mut dfs_stack: Vec<(u64, bool)> = vec![(self.root_id, false)];
        while let Some((node, processed)) = dfs_stack.pop() {
            if processed {
                post_order.push(node);
                continue;
            }
            if !visited.insert(node) {
                continue;
            }
            dfs_stack.push((node, true));
            if let Some(obj) = self.objects.get(&node) {
                for child in obj.references.iter().rev() {
                    if !visited.contains(child) && self.objects.contains_key(child) {
                        dfs_stack.push((*child, false));
                    }
                }
            }
        }
        post_order.reverse();
        rpo = post_order;

        // Build reverse graph (predecessors)
        let mut preds: HashMap<u64, Vec<u64>> = HashMap::new();
        for obj in self.objects.values() {
            for child in &obj.references {
                if self.objects.contains_key(child) {
                    preds.entry(*child).or_default().push(obj.id);
                }
            }
        }

        // RPO index
        let rpo_idx: HashMap<u64, usize> = rpo.iter().enumerate().map(|(i, id)| (*id, i)).collect();

        let mut idom: HashMap<u64, u64> = HashMap::new();
        idom.insert(self.root_id, self.root_id);

        let intersect = |mut a: u64, mut b: u64, idom: &HashMap<u64, u64>| -> u64 {
            while a != b {
                while rpo_idx.get(&a).copied().unwrap_or(usize::MAX)
                    > rpo_idx.get(&b).copied().unwrap_or(usize::MAX)
                {
                    a = match idom.get(&a) {
                        Some(&d) => d,
                        None => return self.root_id,
                    };
                }
                while rpo_idx.get(&b).copied().unwrap_or(usize::MAX)
                    > rpo_idx.get(&a).copied().unwrap_or(usize::MAX)
                {
                    b = match idom.get(&b) {
                        Some(&d) => d,
                        None => return self.root_id,
                    };
                }
            }
            a
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &node in &rpo {
                if node == self.root_id {
                    continue;
                }
                let pred_list = preds.get(&node);
                if pred_list.is_none() {
                    continue;
                }
                let pred_list = pred_list.unwrap();

                let mut new_idom: Option<u64> = None;
                for &p in pred_list {
                    if idom.contains_key(&p) {
                        new_idom = Some(match new_idom {
                            None => p,
                            Some(cur) => intersect(cur, p, &idom),
                        });
                    }
                }

                if let Some(ni) = new_idom {
                    if idom.get(&node) != Some(&ni) {
                        idom.insert(node, ni);
                        changed = true;
                    }
                }
            }
        }

        // Remove synthetic root's self-domination
        idom.remove(&self.root_id);
        idom
    }

    /// Compute the retained size of a specific object: the total shallow size
    /// of all objects that would be garbage-collected if this object were removed.
    pub fn retained_size(&self, target_id: u64) -> usize {
        let dom = self.dominator_tree();

        // Build dominator children map
        let mut dom_children: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&node, &parent) in &dom {
            dom_children.entry(parent).or_default().push(node);
        }

        // Sum subtree under target_id in dominator tree
        let mut total = 0usize;
        let mut queue = VecDeque::new();
        queue.push_back(target_id);
        while let Some(node) = queue.pop_front() {
            if let Some(obj) = self.objects.get(&node) {
                if node != self.root_id {
                    total += obj.shallow_size;
                }
            }
            if let Some(children) = dom_children.get(&node) {
                for &child in children {
                    queue.push_back(child);
                }
            }
        }

        total
    }

    /// Top N objects by retained size.
    pub fn top_retained(&self, n: usize) -> Vec<(u64, usize)> {
        let ids = self.object_ids();
        let mut sizes: Vec<(u64, usize)> = ids.iter().map(|id| (*id, self.retained_size(*id))).collect();
        sizes.sort_by(|a, b| b.1.cmp(&a.1));
        sizes.truncate(n);
        sizes
    }

    /// Diff two snapshots.
    pub fn diff(old: &HeapSnapshot, new_snap: &HeapSnapshot) -> SnapshotDiff {
        let old_ids: HashSet<u64> = old.object_ids().into_iter().collect();
        let new_ids: HashSet<u64> = new_snap.object_ids().into_iter().collect();

        let new_objects: Vec<u64> = new_ids.difference(&old_ids).copied().collect();
        let removed_objects: Vec<u64> = old_ids.difference(&new_ids).copied().collect();

        // Size by type in each snapshot
        let old_by_type = type_size_map(old);
        let new_by_type = type_size_map(new_snap);

        let mut all_types: HashSet<&str> = HashSet::new();
        for k in old_by_type.keys() {
            all_types.insert(k.as_str());
        }
        for k in new_by_type.keys() {
            all_types.insert(k.as_str());
        }

        let mut size_delta_by_type = HashMap::new();
        let mut total_delta: i64 = 0;
        for t in all_types {
            let old_sz = old_by_type.get(t).copied().unwrap_or(0) as i64;
            let new_sz = new_by_type.get(t).copied().unwrap_or(0) as i64;
            let delta = new_sz - old_sz;
            if delta != 0 {
                size_delta_by_type.insert(t.to_string(), delta);
            }
            total_delta += delta;
        }

        SnapshotDiff {
            new_objects,
            removed_objects,
            size_delta_by_type,
            total_size_delta: total_delta,
        }
    }
}

fn type_size_map(snap: &HeapSnapshot) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for obj in snap.objects.values() {
        if obj.id == snap.root_id {
            continue;
        }
        *m.entry(obj.type_name.clone()).or_insert(0) += obj.shallow_size;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_snapshot() -> HeapSnapshot {
        // A -> B -> C, A -> D
        HeapSnapshot::new(vec![
            HeapObject::new(1, "Node", 100).with_refs(&[2, 4]),
            HeapObject::new(2, "Node", 200).with_refs(&[3]),
            HeapObject::new(3, "Leaf", 50),
            HeapObject::new(4, "Leaf", 80),
        ])
    }

    #[test]
    fn test_total_shallow_size() {
        let snap = simple_snapshot();
        assert_eq!(snap.total_shallow_size(), 430);
    }

    #[test]
    fn test_root_objects() {
        let snap = simple_snapshot();
        let roots = snap.root_objects();
        assert_eq!(roots, vec![1]);
    }

    #[test]
    fn test_group_by_type() {
        let snap = simple_snapshot();
        let groups = snap.group_by_type();
        assert_eq!(groups.len(), 2);
        // Node: 2 objects, 300 total
        let node_group = groups.iter().find(|g| g.type_name == "Node").unwrap();
        assert_eq!(node_group.count, 2);
        assert_eq!(node_group.total_shallow_size, 300);
    }

    #[test]
    fn test_dominator_tree() {
        let snap = simple_snapshot();
        let dom = snap.dominator_tree();
        // Object 1 is dominated by root (0)
        assert_eq!(dom[&1], 0);
        // Object 2 is dominated by 1
        assert_eq!(dom[&2], 1);
        // Object 3 is dominated by 2
        assert_eq!(dom[&3], 2);
        // Object 4 is dominated by 1
        assert_eq!(dom[&4], 1);
    }

    #[test]
    fn test_retained_size_root() {
        let snap = simple_snapshot();
        // Object 1 retains everything (100 + 200 + 50 + 80 = 430)
        assert_eq!(snap.retained_size(1), 430);
    }

    #[test]
    fn test_retained_size_leaf() {
        let snap = simple_snapshot();
        // Object 3 (leaf) retains only itself
        assert_eq!(snap.retained_size(3), 50);
    }

    #[test]
    fn test_retained_size_intermediate() {
        let snap = simple_snapshot();
        // Object 2 retains itself (200) + object 3 (50)
        assert_eq!(snap.retained_size(2), 250);
    }

    #[test]
    fn test_top_retained() {
        let snap = simple_snapshot();
        let top = snap.top_retained(2);
        assert_eq!(top.len(), 2);
        // Object 1 should be first (retains 430)
        assert_eq!(top[0].0, 1);
        assert_eq!(top[0].1, 430);
    }

    #[test]
    fn test_diff_new_objects() {
        let old = HeapSnapshot::new(vec![
            HeapObject::new(1, "A", 100),
        ]);
        let new_snap = HeapSnapshot::new(vec![
            HeapObject::new(1, "A", 100),
            HeapObject::new(2, "B", 200),
        ]);
        let diff = HeapSnapshot::diff(&old, &new_snap);
        assert_eq!(diff.new_objects, vec![2]);
        assert!(diff.removed_objects.is_empty());
        assert_eq!(diff.total_size_delta, 200);
    }

    #[test]
    fn test_diff_removed_objects() {
        let old = HeapSnapshot::new(vec![
            HeapObject::new(1, "A", 100),
            HeapObject::new(2, "B", 200),
        ]);
        let new_snap = HeapSnapshot::new(vec![
            HeapObject::new(1, "A", 100),
        ]);
        let diff = HeapSnapshot::diff(&old, &new_snap);
        assert!(diff.new_objects.is_empty());
        assert_eq!(diff.removed_objects, vec![2]);
        assert_eq!(diff.total_size_delta, -200);
    }

    #[test]
    fn test_diff_size_by_type() {
        let old = HeapSnapshot::new(vec![
            HeapObject::new(1, "String", 100),
        ]);
        let new_snap = HeapSnapshot::new(vec![
            HeapObject::new(1, "String", 100),
            HeapObject::new(2, "String", 300),
        ]);
        let diff = HeapSnapshot::diff(&old, &new_snap);
        assert_eq!(diff.size_delta_by_type["String"], 300);
    }

    #[test]
    fn test_shared_reference_dominator() {
        // A -> C, B -> C (C has two parents, both roots)
        let snap = HeapSnapshot::new(vec![
            HeapObject::new(1, "X", 10).with_refs(&[3]),
            HeapObject::new(2, "X", 20).with_refs(&[3]),
            HeapObject::new(3, "Y", 30),
        ]);
        // C is dominated by root (since both A and B are roots)
        let dom = snap.dominator_tree();
        assert_eq!(dom[&3], 0);
        // Neither A nor B retains C
        assert_eq!(snap.retained_size(1), 10);
        assert_eq!(snap.retained_size(2), 20);
    }

    #[test]
    fn test_empty_snapshot() {
        let snap = HeapSnapshot::new(vec![]);
        assert_eq!(snap.total_shallow_size(), 0);
        assert!(snap.root_objects().is_empty());
        assert!(snap.group_by_type().is_empty());
    }
}
