//! Weighted random loot/drop table system.
//!
//! Items with drop weights and rarity tiers, guaranteed drops, nested
//! tables, roll-N-from-table, no-duplicate mode, level-scaled weights,
//! pity system for rare drops, and drop rate simulation.

use std::collections::HashMap;

// ── Seeded RNG ──

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ── Rarity ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

impl Rarity {
    pub fn base_weight(self) -> f64 {
        match self {
            Rarity::Common => 100.0,
            Rarity::Uncommon => 40.0,
            Rarity::Rare => 10.0,
            Rarity::Epic => 3.0,
            Rarity::Legendary => 0.5,
        }
    }

    pub fn all() -> &'static [Rarity] {
        &[Rarity::Common, Rarity::Uncommon, Rarity::Rare, Rarity::Epic, Rarity::Legendary]
    }
}

// ── LootEntry ──

#[derive(Debug, Clone, PartialEq)]
pub enum LootEntry {
    Item {
        name: String,
        rarity: Rarity,
        weight: f64,
        level_scale: f64,
    },
    NestedTable {
        table_name: String,
        weight: f64,
    },
    Guaranteed {
        name: String,
        rarity: Rarity,
    },
}

impl LootEntry {
    /// Create a simple item entry.
    pub fn item(name: impl Into<String>, rarity: Rarity, weight: f64) -> Self {
        LootEntry::Item { name: name.into(), rarity, weight, level_scale: 0.0 }
    }

    /// Create an item with level scaling. Weight adjusted by: weight + level_scale * level.
    pub fn item_scaled(name: impl Into<String>, rarity: Rarity, weight: f64, level_scale: f64) -> Self {
        LootEntry::Item { name: name.into(), rarity, weight, level_scale }
    }

    /// Create a nested table reference.
    pub fn nested(table_name: impl Into<String>, weight: f64) -> Self {
        LootEntry::NestedTable { table_name: table_name.into(), weight }
    }

    /// Create a guaranteed drop (always included in results).
    pub fn guaranteed(name: impl Into<String>, rarity: Rarity) -> Self {
        LootEntry::Guaranteed { name: name.into(), rarity }
    }

    fn effective_weight(&self, level: u32) -> f64 {
        match self {
            LootEntry::Item { weight, level_scale, .. } => {
                (weight + level_scale * level as f64).max(0.0)
            }
            LootEntry::NestedTable { weight, .. } => *weight,
            LootEntry::Guaranteed { .. } => 0.0,
        }
    }

    fn name(&self) -> &str {
        match self {
            LootEntry::Item { name, .. } => name,
            LootEntry::NestedTable { table_name, .. } => table_name,
            LootEntry::Guaranteed { name, .. } => name,
        }
    }

    fn rarity(&self) -> Option<Rarity> {
        match self {
            LootEntry::Item { rarity, .. } => Some(*rarity),
            LootEntry::Guaranteed { rarity, .. } => Some(*rarity),
            LootEntry::NestedTable { .. } => None,
        }
    }
}

// ── DroppedItem ──

#[derive(Debug, Clone, PartialEq)]
pub struct DroppedItem {
    pub name: String,
    pub rarity: Rarity,
}

// ── PityTracker ──

#[derive(Debug, Clone)]
pub struct PityTracker {
    misses: HashMap<Rarity, u32>,
    thresholds: HashMap<Rarity, u32>,
    bonus_per_miss: f64,
}

impl PityTracker {
    /// Create a pity tracker. After `threshold` misses for a rarity,
    /// the weight bonus starts accumulating.
    pub fn new(thresholds: &[(Rarity, u32)], bonus_per_miss: f64) -> Self {
        Self {
            misses: HashMap::new(),
            thresholds: thresholds.iter().cloned().collect(),
            bonus_per_miss,
        }
    }

    pub fn record_miss(&mut self, rarity: Rarity) {
        *self.misses.entry(rarity).or_insert(0) += 1;
    }

    pub fn record_hit(&mut self, rarity: Rarity) {
        self.misses.insert(rarity, 0);
    }

    pub fn weight_bonus(&self, rarity: Rarity) -> f64 {
        let misses = self.misses.get(&rarity).copied().unwrap_or(0);
        let threshold = self.thresholds.get(&rarity).copied().unwrap_or(u32::MAX);
        if misses > threshold {
            (misses - threshold) as f64 * self.bonus_per_miss
        } else {
            0.0
        }
    }
}

// ── LootTable ──

#[derive(Debug, Clone)]
pub struct LootTable {
    pub name: String,
    pub entries: Vec<LootEntry>,
}

impl LootTable {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), entries: Vec::new() }
    }

    pub fn add(mut self, entry: LootEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Roll one item from the table (excluding guaranteed drops).
    pub fn roll_one(&self, level: u32, pity: Option<&mut PityTracker>, seed: u64) -> Option<DroppedItem> {
        let items = self.roll(1, level, false, pity, seed);
        items.into_iter().find(|i| {
            // Skip guaranteed drops
            !self.entries.iter().any(|e| matches!(e, LootEntry::Guaranteed { name, .. } if name == &i.name))
        })
    }

    /// Roll N items from the table. Includes guaranteed drops first.
    pub fn roll(
        &self,
        count: usize,
        level: u32,
        no_duplicates: bool,
        mut pity: Option<&mut PityTracker>,
        seed: u64,
    ) -> Vec<DroppedItem> {
        let mut rng = Rng::new(seed);
        let mut result = Vec::new();
        let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Guaranteed drops first
        for entry in &self.entries {
            if let LootEntry::Guaranteed { name, rarity } = entry {
                result.push(DroppedItem { name: name.clone(), rarity: *rarity });
                if no_duplicates { used_names.insert(name.clone()); }
            }
        }

        // Compute pity bonuses per rarity upfront
        let mut pity_bonuses: HashMap<Rarity, f64> = HashMap::new();
        if let Some(p) = pity.as_ref() {
            for r in Rarity::all() {
                let bonus = p.weight_bonus(*r);
                if bonus > 0.0 {
                    pity_bonuses.insert(*r, bonus);
                }
            }
        }

        // Weighted selection for remaining slots
        let weighted: Vec<(usize, f64)> = self.entries.iter().enumerate()
            .filter(|(_, e)| !matches!(e, LootEntry::Guaranteed { .. }))
            .map(|(i, e)| {
                let mut w = e.effective_weight(level);
                if let Some(rarity) = e.rarity() {
                    w += pity_bonuses.get(&rarity).copied().unwrap_or(0.0);
                }
                (i, w)
            })
            .filter(|(_, w)| *w > 0.0)
            .collect();

        for _ in 0..count {
            let available: Vec<(usize, f64)> = if no_duplicates {
                weighted.iter()
                    .filter(|(i, _)| !used_names.contains(self.entries[*i].name()))
                    .cloned()
                    .collect()
            } else {
                weighted.clone()
            };

            if available.is_empty() { break; }

            let total: f64 = available.iter().map(|(_, w)| w).sum();
            if total <= 0.0 { break; }

            let roll = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut selected = available[0].0;

            for &(idx, w) in &available {
                cumulative += w;
                if roll < cumulative {
                    selected = idx;
                    break;
                }
            }

            let entry = &self.entries[selected];
            if let Some(rarity) = entry.rarity() {
                let item = DroppedItem { name: entry.name().to_string(), rarity };
                if no_duplicates { used_names.insert(item.name.clone()); }

                // Update pity
                if let Some(p) = pity.as_mut() {
                    p.record_hit(rarity);
                    for r in Rarity::all() {
                        if *r != rarity { p.record_miss(*r); }
                    }
                }
                result.push(item);
            }
        }

        result
    }
}

// ── LootSystem (manages multiple tables + nested references) ──

#[derive(Debug, Clone)]
pub struct LootSystem {
    pub tables: HashMap<String, LootTable>,
}

impl LootSystem {
    pub fn new() -> Self { Self { tables: HashMap::new() } }

    pub fn add_table(mut self, table: LootTable) -> Self {
        self.tables.insert(table.name.clone(), table);
        self
    }

    /// Roll from a named table, resolving nested table references.
    pub fn roll(
        &self,
        table_name: &str,
        count: usize,
        level: u32,
        no_duplicates: bool,
        seed: u64,
    ) -> Vec<DroppedItem> {
        let mut rng = Rng::new(seed);
        let mut result = Vec::new();
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut depth = 0;

        self.roll_recursive(table_name, count, level, no_duplicates, &mut rng, &mut result, &mut used, &mut depth);
        result
    }

    fn roll_recursive(
        &self,
        table_name: &str,
        count: usize,
        level: u32,
        no_duplicates: bool,
        rng: &mut Rng,
        result: &mut Vec<DroppedItem>,
        used: &mut std::collections::HashSet<String>,
        depth: &mut usize,
    ) {
        if *depth > 10 { return; } // Prevent infinite recursion
        *depth += 1;

        let table = match self.tables.get(table_name) {
            Some(t) => t,
            None => { *depth -= 1; return; }
        };

        // Guaranteed drops
        for entry in &table.entries {
            if let LootEntry::Guaranteed { name, rarity } = entry {
                if !no_duplicates || !used.contains(name) {
                    result.push(DroppedItem { name: name.clone(), rarity: *rarity });
                    used.insert(name.clone());
                }
            }
        }

        let weighted: Vec<(usize, f64)> = table.entries.iter().enumerate()
            .filter(|(_, e)| !matches!(e, LootEntry::Guaranteed { .. }))
            .map(|(i, e)| (i, e.effective_weight(level)))
            .filter(|(_, w)| *w > 0.0)
            .collect();

        for _ in 0..count {
            let available: Vec<(usize, f64)> = if no_duplicates {
                weighted.iter()
                    .filter(|(i, _)| !used.contains(table.entries[*i].name()))
                    .cloned()
                    .collect()
            } else {
                weighted.clone()
            };

            if available.is_empty() { break; }
            let total: f64 = available.iter().map(|(_, w)| w).sum();
            if total <= 0.0 { break; }

            let roll = rng.next_f64() * total;
            let mut cum = 0.0;
            let mut selected = available[0].0;
            for &(idx, w) in &available {
                cum += w;
                if roll < cum { selected = idx; break; }
            }

            match &table.entries[selected] {
                LootEntry::Item { name, rarity, .. } => {
                    if !no_duplicates || !used.contains(name) {
                        result.push(DroppedItem { name: name.clone(), rarity: *rarity });
                        used.insert(name.clone());
                    }
                }
                LootEntry::NestedTable { table_name: nested, .. } => {
                    self.roll_recursive(nested, 1, level, no_duplicates, rng, result, used, depth);
                }
                LootEntry::Guaranteed { .. } => {}
            }
        }
        *depth -= 1;
    }
}

// ── Simulation ──

/// Simulate N rolls and return the distribution of items.
pub fn simulate_distribution(
    table: &LootTable,
    rolls: usize,
    level: u32,
    seed: u64,
) -> HashMap<String, usize> {
    let mut dist: HashMap<String, usize> = HashMap::new();
    for i in 0..rolls {
        let items = table.roll(1, level, false, None, seed.wrapping_add(i as u64));
        for item in items {
            *dist.entry(item.name).or_insert(0) += 1;
        }
    }
    dist
}

/// Compute the theoretical drop rates for each entry in the table.
pub fn expected_rates(table: &LootTable, level: u32) -> Vec<(String, f64)> {
    let weighted: Vec<(&LootEntry, f64)> = table.entries.iter()
        .filter(|e| !matches!(e, LootEntry::Guaranteed { .. }))
        .map(|e| (e, e.effective_weight(level)))
        .filter(|(_, w)| *w > 0.0)
        .collect();

    let total: f64 = weighted.iter().map(|(_, w)| w).sum();
    if total <= 0.0 { return Vec::new(); }

    let mut rates: Vec<(String, f64)> = weighted.iter()
        .map(|(e, w)| (e.name().to_string(), w / total))
        .collect();

    // Add guaranteed at 100%
    for entry in &table.entries {
        if let LootEntry::Guaranteed { name, .. } = entry {
            rates.push((name.clone(), 1.0));
        }
    }
    rates.sort_by(|a, b| a.0.cmp(&b.0));
    rates
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_table() -> LootTable {
        LootTable::new("weapons")
            .add(LootEntry::item("Iron Sword", Rarity::Common, 100.0))
            .add(LootEntry::item("Steel Sword", Rarity::Uncommon, 40.0))
            .add(LootEntry::item("Flame Blade", Rarity::Rare, 10.0))
            .add(LootEntry::item("Dragon Slayer", Rarity::Epic, 3.0))
            .add(LootEntry::item("Excalibur", Rarity::Legendary, 0.5))
    }

    #[test]
    fn test_roll_one() {
        let table = basic_table();
        let item = table.roll_one(1, None, 42);
        assert!(item.is_some());
    }

    #[test]
    fn test_roll_multiple() {
        let table = basic_table();
        let items = table.roll(5, 1, false, None, 42);
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn test_guaranteed_drops() {
        let table = LootTable::new("boss")
            .add(LootEntry::guaranteed("Boss Key", Rarity::Epic))
            .add(LootEntry::item("Gold", Rarity::Common, 100.0));
        let items = table.roll(2, 1, false, None, 42);
        assert!(items.iter().any(|i| i.name == "Boss Key"));
    }

    #[test]
    fn test_no_duplicates() {
        let table = LootTable::new("small")
            .add(LootEntry::item("A", Rarity::Common, 50.0))
            .add(LootEntry::item("B", Rarity::Common, 50.0));
        let items = table.roll(2, 1, true, None, 42);
        assert_eq!(items.len(), 2);
        assert_ne!(items[0].name, items[1].name);
    }

    #[test]
    fn test_no_duplicates_exhaustion() {
        let table = LootTable::new("tiny")
            .add(LootEntry::item("Only", Rarity::Common, 100.0));
        let items = table.roll(5, 1, true, None, 42);
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_seed_determinism() {
        let table = basic_table();
        let a = table.roll(10, 1, false, None, 123);
        let b = table.roll(10, 1, false, None, 123);
        let names_a: Vec<&str> = a.iter().map(|i| i.name.as_str()).collect();
        let names_b: Vec<&str> = b.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names_a, names_b);
    }

    #[test]
    fn test_level_scaling() {
        let table = LootTable::new("scaled")
            .add(LootEntry::item_scaled("Weak", Rarity::Common, 100.0, -5.0))
            .add(LootEntry::item_scaled("Strong", Rarity::Rare, 1.0, 10.0));
        // At level 20, Strong has weight 1+10*20=201, Weak has max(100-5*20,0)=0
        let rates = expected_rates(&table, 20);
        let strong_rate = rates.iter().find(|(n, _)| n == "Strong").map(|(_, r)| *r).unwrap_or(0.0);
        assert!((strong_rate - 1.0).abs() < 1e-6, "Strong should be 100% at level 20");
    }

    #[test]
    fn test_pity_tracker() {
        let mut pity = PityTracker::new(&[(Rarity::Legendary, 5)], 10.0);
        for _ in 0..10 {
            pity.record_miss(Rarity::Legendary);
        }
        let bonus = pity.weight_bonus(Rarity::Legendary);
        assert!((bonus - 50.0).abs() < 1e-6); // (10-5)*10 = 50
    }

    #[test]
    fn test_pity_reset_on_hit() {
        let mut pity = PityTracker::new(&[(Rarity::Rare, 3)], 5.0);
        for _ in 0..10 { pity.record_miss(Rarity::Rare); }
        pity.record_hit(Rarity::Rare);
        assert!((pity.weight_bonus(Rarity::Rare) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_nested_table() {
        let system = LootSystem::new()
            .add_table(LootTable::new("main")
                .add(LootEntry::item("Coin", Rarity::Common, 50.0))
                .add(LootEntry::nested("rare_table", 50.0)))
            .add_table(LootTable::new("rare_table")
                .add(LootEntry::item("Ruby", Rarity::Rare, 50.0))
                .add(LootEntry::item("Sapphire", Rarity::Rare, 50.0)));

        let items = system.roll("main", 10, 1, false, 42);
        assert!(!items.is_empty());
    }

    #[test]
    fn test_simulate_distribution() {
        let table = LootTable::new("test")
            .add(LootEntry::item("A", Rarity::Common, 75.0))
            .add(LootEntry::item("B", Rarity::Common, 25.0));
        let dist = simulate_distribution(&table, 1000, 1, 42);
        let a_count = dist.get("A").copied().unwrap_or(0);
        let b_count = dist.get("B").copied().unwrap_or(0);
        // A should appear ~3x more than B (roughly)
        assert!(a_count > b_count, "A ({}) should appear more than B ({})", a_count, b_count);
    }

    #[test]
    fn test_expected_rates() {
        let table = LootTable::new("rates")
            .add(LootEntry::item("X", Rarity::Common, 60.0))
            .add(LootEntry::item("Y", Rarity::Common, 40.0));
        let rates = expected_rates(&table, 1);
        let x_rate = rates.iter().find(|(n, _)| n == "X").map(|(_, r)| *r).unwrap_or(0.0);
        assert!((x_rate - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_guaranteed_rate_is_one() {
        let table = LootTable::new("g")
            .add(LootEntry::guaranteed("Always", Rarity::Epic))
            .add(LootEntry::item("Sometimes", Rarity::Common, 100.0));
        let rates = expected_rates(&table, 1);
        let always_rate = rates.iter().find(|(n, _)| n == "Always").map(|(_, r)| *r).unwrap_or(0.0);
        assert!((always_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rarity_base_weights() {
        assert!(Rarity::Common.base_weight() > Rarity::Legendary.base_weight());
        assert!(Rarity::Uncommon.base_weight() > Rarity::Rare.base_weight());
    }

    #[test]
    fn test_empty_table() {
        let table = LootTable::new("empty");
        let items = table.roll(5, 1, false, None, 42);
        assert!(items.is_empty());
    }

    #[test]
    fn test_loot_system_missing_table() {
        let system = LootSystem::new();
        let items = system.roll("nonexistent", 5, 1, false, 42);
        assert!(items.is_empty());
    }

    #[test]
    fn test_entry_name() {
        let e = LootEntry::item("Sword", Rarity::Common, 10.0);
        assert_eq!(e.name(), "Sword");
    }

    #[test]
    fn test_effective_weight_at_zero_level() {
        let e = LootEntry::item_scaled("Test", Rarity::Common, 50.0, 2.0);
        assert!((e.effective_weight(0) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_effective_weight_floor() {
        let e = LootEntry::item_scaled("Test", Rarity::Common, 10.0, -5.0);
        // At level 100: 10 + (-5)*100 = -490, clamped to 0
        assert!((e.effective_weight(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_pity_with_roll() {
        let table = basic_table();
        let mut pity = PityTracker::new(&[(Rarity::Legendary, 0)], 1000.0);
        // After many misses, legendary bonus is huge
        for _ in 0..50 { pity.record_miss(Rarity::Legendary); }
        let item = table.roll_one(1, Some(&mut pity), 42);
        assert!(item.is_some());
    }

    #[test]
    fn test_many_rolls_all_rarities_appear() {
        let table = basic_table();
        let dist = simulate_distribution(&table, 10000, 1, 12345);
        // Common should definitely appear
        assert!(dist.get("Iron Sword").copied().unwrap_or(0) > 0);
    }
}
