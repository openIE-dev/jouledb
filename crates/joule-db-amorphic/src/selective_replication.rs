//! Selective Replication — push relevant holograms to edge nodes.
//!
//! Not every edge node needs every record. An edge in France gets:
//! - Trending French content
//! - Content licensed for FR territory
//! - Category overrides from the replication policy
//!
//! Combined with Edge Query Routing: similarity, trending, and rights
//! queries are served locally without origin round-trips.

use crate::temporal_fields::TemporalStore;
use crate::trending::{TrendWindow, TrendingIndex};
use crate::{AmorphicStore, RecordId};
use std::collections::HashSet;
use std::time::Duration;

/// What content to replicate to an edge node.
#[derive(Debug, Clone)]
pub enum ContentFilter {
    /// Top-k trending content in a time window
    Trending {
        window: TrendWindow,
        k: usize,
    },
    /// Content in a specific category
    Category(String),
    /// Content licensed for a specific territory
    Territory(String),
    /// Explicit record IDs
    Explicit(Vec<RecordId>),
    /// All content (full replica)
    All,
}

/// Replication policy for an edge node.
#[derive(Debug, Clone)]
pub struct ReplicationPolicy {
    /// Edge region identifier (e.g., "eu-west-1", "ap-tokyo")
    pub region: String,
    /// Content filters (combined with OR logic)
    pub filters: Vec<ContentFilter>,
    /// Maximum records to replicate
    pub max_records: usize,
    /// How often to refresh the replication set
    pub refresh_interval: Duration,
}

impl ReplicationPolicy {
    pub fn new(region: &str) -> Self {
        Self {
            region: region.to_string(),
            filters: vec![],
            max_records: 10_000,
            refresh_interval: Duration::from_secs(60),
        }
    }

    pub fn with_trending(mut self, window: TrendWindow, k: usize) -> Self {
        self.filters.push(ContentFilter::Trending { window, k });
        self
    }

    pub fn with_territory(mut self, territory: &str) -> Self {
        self.filters.push(ContentFilter::Territory(territory.to_string()));
        self
    }

    pub fn with_category(mut self, category: &str) -> Self {
        self.filters.push(ContentFilter::Category(category.to_string()));
        self
    }

    pub fn with_explicit(mut self, ids: Vec<RecordId>) -> Self {
        self.filters.push(ContentFilter::Explicit(ids));
        self
    }

    pub fn with_max_records(mut self, max: usize) -> Self {
        self.max_records = max;
        self
    }
}

/// Compute the set of records that should be replicated to an edge node
/// based on its replication policy.
pub fn compute_replication_set(
    store: &AmorphicStore,
    policy: &ReplicationPolicy,
    trending: Option<&TrendingIndex>,
    temporal: Option<&TemporalStore>,
    now_ms: u64,
) -> Vec<RecordId> {
    let mut result_set = HashSet::new();

    for filter in &policy.filters {
        match filter {
            ContentFilter::Trending { window, k } => {
                if let Some(idx) = trending {
                    let items = idx.query_trending(*k, *window, None);
                    for item in items {
                        result_set.insert(item.record_id);
                    }
                }
            }
            ContentFilter::Category(cat) => {
                // Find records with matching category field
                let matches = store.query_equals("category", &crate::Value::String(cat.clone()));
                for record in matches.records() {
                    result_set.insert(record.id);
                }
            }
            ContentFilter::Territory(territory) => {
                if let Some(ts) = temporal {
                    // Find all records with active streaming rights for this territory
                    for &id in store.records.keys() {
                        if ts.can_stream(id, territory, now_ms) {
                            result_set.insert(id);
                        }
                    }
                }
            }
            ContentFilter::Explicit(ids) => {
                for &id in ids {
                    result_set.insert(id);
                }
            }
            ContentFilter::All => {
                for &id in store.records.keys() {
                    result_set.insert(id);
                }
            }
        }
    }

    // Enforce max_records limit
    let mut ids: Vec<RecordId> = result_set.into_iter().collect();
    ids.truncate(policy.max_records);
    ids
}

/// Determines how to route a query at the edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryRoute {
    /// Answer entirely from local edge store
    Local,
    /// Forward to origin (edge doesn't have the data)
    Origin,
    /// Try local first, fall back to origin if not found
    LocalThenOrigin,
}

/// Edge query router: decides whether to serve locally or forward to origin.
pub struct EdgeRouter {
    /// Set of record IDs available at this edge (Bloom filter would be ideal, HashSet for now)
    local_records: HashSet<RecordId>,
    /// Whether trending index is available locally
    has_trending: bool,
    /// Whether temporal store is available locally
    has_temporal: bool,
}

impl EdgeRouter {
    pub fn new(local_records: HashSet<RecordId>, has_trending: bool, has_temporal: bool) -> Self {
        Self {
            local_records,
            has_trending,
            has_temporal,
        }
    }

    /// Update the set of locally available records.
    pub fn update_local_records(&mut self, ids: HashSet<RecordId>) {
        self.local_records = ids;
    }

    /// Route a similarity search query.
    /// Always local — holograms are pre-replicated for this purpose.
    pub fn route_similarity(&self) -> QueryRoute {
        QueryRoute::Local
    }

    /// Route a trending query.
    pub fn route_trending(&self) -> QueryRoute {
        if self.has_trending {
            QueryRoute::Local
        } else {
            QueryRoute::Origin
        }
    }

    /// Route a rights/license check.
    pub fn route_rights_check(&self) -> QueryRoute {
        if self.has_temporal {
            QueryRoute::Local
        } else {
            QueryRoute::Origin
        }
    }

    /// Route a specific record lookup.
    pub fn route_record_lookup(&self, id: RecordId) -> QueryRoute {
        if self.local_records.contains(&id) {
            QueryRoute::Local
        } else {
            QueryRoute::Origin
        }
    }

    /// Route a batch feature lookup.
    pub fn route_batch_lookup(&self, ids: &[RecordId]) -> QueryRoute {
        let local_count = ids.iter().filter(|id| self.local_records.contains(id)).count();
        if local_count == ids.len() {
            QueryRoute::Local
        } else if local_count > ids.len() / 2 {
            QueryRoute::LocalThenOrigin
        } else {
            QueryRoute::Origin
        }
    }

    /// Route a SQL query (always origin — edge doesn't have full data).
    pub fn route_sql(&self) -> QueryRoute {
        QueryRoute::Origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    #[test]
    fn test_replication_policy_builder() {
        let policy = ReplicationPolicy::new("eu-west-1")
            .with_trending(TrendWindow::OneHour, 100)
            .with_territory("FR")
            .with_max_records(5000);

        assert_eq!(policy.region, "eu-west-1");
        assert_eq!(policy.filters.len(), 2);
        assert_eq!(policy.max_records, 5000);
    }

    #[test]
    fn test_selective_replication_explicit() {
        let store = AmorphicStore::new();
        let policy = ReplicationPolicy::new("test")
            .with_explicit(vec![1, 2, 3]);

        let set = compute_replication_set(&store, &policy, None, None, 0);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_edge_routing() {
        let local = HashSet::from([1, 2, 3, 4, 5]);
        let router = EdgeRouter::new(local, true, true);

        assert_eq!(router.route_similarity(), QueryRoute::Local);
        assert_eq!(router.route_trending(), QueryRoute::Local);
        assert_eq!(router.route_rights_check(), QueryRoute::Local);
        assert_eq!(router.route_record_lookup(3), QueryRoute::Local);
        assert_eq!(router.route_record_lookup(99), QueryRoute::Origin);
        assert_eq!(router.route_sql(), QueryRoute::Origin);
    }

    #[test]
    fn test_batch_routing() {
        let local = HashSet::from([1, 2, 3]);
        let router = EdgeRouter::new(local, false, false);

        // All local
        assert_eq!(router.route_batch_lookup(&[1, 2, 3]), QueryRoute::Local);
        // Mostly local
        assert_eq!(
            router.route_batch_lookup(&[1, 2, 99]),
            QueryRoute::LocalThenOrigin
        );
        // Mostly remote
        assert_eq!(
            router.route_batch_lookup(&[98, 99, 100]),
            QueryRoute::Origin
        );
    }
}
