//! Crafting recipe system — recipes, ingredients, queues, skills, byproducts.
//!
//! Replaces crafting.js / Phaser-craft plugins with pure Rust.
//! Recipe definitions, discovery, skill requirements, success rates,
//! byproducts, categories, crafting queues, and auto-craft.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CraftingError {
    RecipeNotFound(u64),
    RecipeNotDiscovered(u64),
    MissingIngredient { item_id: u64, have: u32, need: u32 },
    SkillTooLow { skill: String, have: u32, need: u32 },
    QueueFull { max: usize },
    NothingInQueue,
    DuplicateRecipe(u64),
    InvalidRecipe(String),
}

impl fmt::Display for CraftingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RecipeNotFound(id) => write!(f, "recipe not found: {id}"),
            Self::RecipeNotDiscovered(id) => write!(f, "recipe not yet discovered: {id}"),
            Self::MissingIngredient { item_id, have, need } => {
                write!(f, "need {need} of item {item_id}, have {have}")
            }
            Self::SkillTooLow { skill, have, need } => {
                write!(f, "{skill} level {have} < required {need}")
            }
            Self::QueueFull { max } => write!(f, "crafting queue full ({max})"),
            Self::NothingInQueue => write!(f, "nothing in crafting queue"),
            Self::DuplicateRecipe(id) => write!(f, "duplicate recipe id: {id}"),
            Self::InvalidRecipe(msg) => write!(f, "invalid recipe: {msg}"),
        }
    }
}

impl std::error::Error for CraftingError {}

// ── PRNG ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns float in [0.0, 1.0)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 1_000_000) as f64 / 1_000_000.0
    }
}

// ── Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CraftCategory {
    Blacksmithing,
    Alchemy,
    Cooking,
    Tailoring,
    Woodworking,
    Enchanting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ingredient {
    pub item_id: u64,
    pub quantity: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraftOutput {
    pub item_id: u64,
    pub quantity: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Byproduct {
    pub item_id: u64,
    pub quantity: u32,
    /// Chance in [0.0, 1.0]
    pub chance: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillReq {
    pub skill_name: String,
    pub level: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    pub id: u64,
    pub name: String,
    pub category: CraftCategory,
    pub ingredients: Vec<Ingredient>,
    pub outputs: Vec<CraftOutput>,
    pub byproducts: Vec<Byproduct>,
    pub duration_ms: u64,
    /// Base success rate in [0.0, 1.0]
    pub base_success_rate: f64,
    /// Bonus per skill level above requirement
    pub skill_bonus_per_level: f64,
    pub skill_requirements: Vec<SkillReq>,
    /// Items that trigger auto-discovery when all are possessed
    pub discovery_items: Vec<u64>,
}

impl Recipe {
    pub fn new(id: u64, name: &str, category: CraftCategory) -> Self {
        Self {
            id,
            name: name.to_string(),
            category,
            ingredients: Vec::new(),
            outputs: Vec::new(),
            byproducts: Vec::new(),
            duration_ms: 1000,
            base_success_rate: 1.0,
            skill_bonus_per_level: 0.05,
            skill_requirements: Vec::new(),
            discovery_items: Vec::new(),
        }
    }

    pub fn with_ingredient(mut self, item_id: u64, qty: u32) -> Self {
        self.ingredients.push(Ingredient { item_id, quantity: qty });
        self
    }

    pub fn with_output(mut self, item_id: u64, qty: u32) -> Self {
        self.outputs.push(CraftOutput { item_id, quantity: qty });
        self
    }

    pub fn with_byproduct(mut self, item_id: u64, qty: u32, chance: f64) -> Self {
        self.byproducts.push(Byproduct { item_id, quantity: qty, chance: chance.clamp(0.0, 1.0) });
        self
    }

    pub fn with_duration(mut self, ms: u64) -> Self { self.duration_ms = ms; self }

    pub fn with_success_rate(mut self, rate: f64) -> Self {
        self.base_success_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_skill_req(mut self, skill: &str, level: u32) -> Self {
        self.skill_requirements.push(SkillReq { skill_name: skill.to_string(), level });
        self
    }

    pub fn with_discovery_items(mut self, items: Vec<u64>) -> Self {
        self.discovery_items = items;
        self
    }
}

// ── Craft Result ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraftResult {
    pub recipe_id: u64,
    pub success: bool,
    pub outputs: Vec<CraftOutput>,
    pub byproducts: Vec<CraftOutput>,
}

// ── Queue Entry ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueEntry {
    pub recipe_id: u64,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
}

// ── Crafting System ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CraftingSystem {
    recipes: HashMap<u64, Recipe>,
    discovered: HashMap<u64, bool>,
    queue: Vec<QueueEntry>,
    max_queue: usize,
    rng: Rng,
}

impl CraftingSystem {
    pub fn new(max_queue: usize, seed: u64) -> Self {
        Self {
            recipes: HashMap::new(),
            discovered: HashMap::new(),
            queue: Vec::new(),
            max_queue,
            rng: Rng::new(seed),
        }
    }

    pub fn add_recipe(&mut self, recipe: Recipe) -> Result<(), CraftingError> {
        if recipe.ingredients.is_empty() {
            return Err(CraftingError::InvalidRecipe("no ingredients".into()));
        }
        if recipe.outputs.is_empty() {
            return Err(CraftingError::InvalidRecipe("no outputs".into()));
        }
        if self.recipes.contains_key(&recipe.id) {
            return Err(CraftingError::DuplicateRecipe(recipe.id));
        }
        let id = recipe.id;
        self.recipes.insert(id, recipe);
        self.discovered.insert(id, false);
        Ok(())
    }

    pub fn discover_recipe(&mut self, recipe_id: u64) -> Result<(), CraftingError> {
        if !self.recipes.contains_key(&recipe_id) {
            return Err(CraftingError::RecipeNotFound(recipe_id));
        }
        self.discovered.insert(recipe_id, true);
        Ok(())
    }

    pub fn is_discovered(&self, recipe_id: u64) -> bool {
        self.discovered.get(&recipe_id).copied().unwrap_or(false)
    }

    /// Check discovery based on items the player possesses.
    pub fn check_discovery(&mut self, possessed_items: &[u64]) -> Vec<u64> {
        let mut newly_discovered = Vec::new();
        let recipe_ids: Vec<u64> = self.recipes.keys().copied().collect();
        for rid in recipe_ids {
            if self.is_discovered(rid) { continue; }
            let recipe = &self.recipes[&rid];
            if recipe.discovery_items.is_empty() { continue; }
            if recipe.discovery_items.iter().all(|item| possessed_items.contains(item)) {
                self.discovered.insert(rid, true);
                newly_discovered.push(rid);
            }
        }
        newly_discovered
    }

    pub fn get_recipe(&self, id: u64) -> Option<&Recipe> {
        self.recipes.get(&id)
    }

    pub fn recipes_by_category(&self, cat: CraftCategory) -> Vec<&Recipe> {
        let mut results: Vec<&Recipe> = self.recipes.values()
            .filter(|r| r.category == cat)
            .collect();
        results.sort_by_key(|r| r.id);
        results
    }

    pub fn discovered_recipes(&self) -> Vec<&Recipe> {
        let mut results: Vec<&Recipe> = self.recipes.values()
            .filter(|r| self.is_discovered(r.id))
            .collect();
        results.sort_by_key(|r| r.id);
        results
    }

    /// Check if a recipe can be crafted with given inventory and skills.
    pub fn can_craft(
        &self,
        recipe_id: u64,
        inventory: &HashMap<u64, u32>,
        skills: &HashMap<String, u32>,
    ) -> Result<(), CraftingError> {
        let recipe = self.recipes.get(&recipe_id)
            .ok_or(CraftingError::RecipeNotFound(recipe_id))?;
        if !self.is_discovered(recipe_id) {
            return Err(CraftingError::RecipeNotDiscovered(recipe_id));
        }
        for req in &recipe.skill_requirements {
            let have = skills.get(&req.skill_name).copied().unwrap_or(0);
            if have < req.level {
                return Err(CraftingError::SkillTooLow {
                    skill: req.skill_name.clone(), have, need: req.level,
                });
            }
        }
        for ing in &recipe.ingredients {
            let have = inventory.get(&ing.item_id).copied().unwrap_or(0);
            if have < ing.quantity {
                return Err(CraftingError::MissingIngredient {
                    item_id: ing.item_id, have, need: ing.quantity,
                });
            }
        }
        Ok(())
    }

    /// Execute a craft immediately (skip queue).
    pub fn craft_immediate(
        &mut self,
        recipe_id: u64,
        inventory: &mut HashMap<u64, u32>,
        skills: &HashMap<String, u32>,
    ) -> Result<CraftResult, CraftingError> {
        self.can_craft(recipe_id, inventory, skills)?;
        let recipe = self.recipes[&recipe_id].clone();

        // Consume ingredients
        for ing in &recipe.ingredients {
            let entry = inventory.get_mut(&ing.item_id).unwrap();
            *entry -= ing.quantity;
            if *entry == 0 {
                inventory.remove(&ing.item_id);
            }
        }

        // Calculate success
        let mut success_rate = recipe.base_success_rate;
        for req in &recipe.skill_requirements {
            let have = skills.get(&req.skill_name).copied().unwrap_or(0);
            if have > req.level {
                let bonus_levels = have - req.level;
                success_rate = (success_rate + bonus_levels as f64 * recipe.skill_bonus_per_level).min(1.0);
            }
        }

        let roll = self.rng.next_f64();
        let success = roll < success_rate;

        let mut result = CraftResult {
            recipe_id,
            success,
            outputs: Vec::new(),
            byproducts: Vec::new(),
        };

        if success {
            // Grant outputs
            for out in &recipe.outputs {
                *inventory.entry(out.item_id).or_insert(0) += out.quantity;
                result.outputs.push(out.clone());
            }
            // Roll byproducts
            for bp in &recipe.byproducts {
                let bp_roll = self.rng.next_f64();
                if bp_roll < bp.chance {
                    *inventory.entry(bp.item_id).or_insert(0) += bp.quantity;
                    result.byproducts.push(CraftOutput { item_id: bp.item_id, quantity: bp.quantity });
                }
            }
        }

        Ok(result)
    }

    /// Queue a craft job.
    pub fn queue_craft(&mut self, recipe_id: u64, current_time_ms: u64, inventory: &HashMap<u64, u32>, skills: &HashMap<String, u32>) -> Result<&QueueEntry, CraftingError> {
        self.can_craft(recipe_id, inventory, skills)?;
        if self.queue.len() >= self.max_queue {
            return Err(CraftingError::QueueFull { max: self.max_queue });
        }
        let recipe = &self.recipes[&recipe_id];
        let start = if let Some(last) = self.queue.last() {
            last.end_time_ms
        } else {
            current_time_ms
        };
        let entry = QueueEntry {
            recipe_id,
            start_time_ms: start,
            end_time_ms: start + recipe.duration_ms,
        };
        self.queue.push(entry);
        Ok(self.queue.last().unwrap())
    }

    /// Check and collect completed queue items.
    pub fn collect_completed(&mut self, current_time_ms: u64) -> Vec<QueueEntry> {
        let (done, pending): (Vec<QueueEntry>, Vec<QueueEntry>) = self.queue
            .drain(..)
            .partition(|e| current_time_ms >= e.end_time_ms);
        self.queue = pending;
        done
    }

    pub fn queue_len(&self) -> usize { self.queue.len() }
    pub fn queue_entries(&self) -> &[QueueEntry] { &self.queue }

    /// Queue multiple of the same recipe.
    pub fn auto_craft(&mut self, recipe_id: u64, count: u32, current_time_ms: u64, inventory: &HashMap<u64, u32>, skills: &HashMap<String, u32>) -> Result<u32, CraftingError> {
        // Verify recipe exists and is discovered, and skills met
        self.can_craft(recipe_id, inventory, skills)?;
        let recipe = self.recipes[&recipe_id].clone();
        let mut queued = 0u32;
        // Check how many we can craft with given ingredients
        let mut max_craftable = u32::MAX;
        for ing in &recipe.ingredients {
            let have = inventory.get(&ing.item_id).copied().unwrap_or(0);
            max_craftable = max_craftable.min(have / ing.quantity);
        }
        let to_queue = count.min(max_craftable);
        for _ in 0..to_queue {
            if self.queue.len() >= self.max_queue { break; }
            let start = if let Some(last) = self.queue.last() {
                last.end_time_ms
            } else {
                current_time_ms
            };
            self.queue.push(QueueEntry {
                recipe_id,
                start_time_ms: start,
                end_time_ms: start + recipe.duration_ms,
            });
            queued += 1;
        }
        Ok(queued)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iron_sword_recipe() -> Recipe {
        Recipe::new(1, "Iron Sword", CraftCategory::Blacksmithing)
            .with_ingredient(100, 3) // 3 iron ingots
            .with_ingredient(101, 1) // 1 leather strip
            .with_output(200, 1)     // 1 iron sword
            .with_duration(5000)
    }

    fn potion_recipe() -> Recipe {
        Recipe::new(2, "Health Potion", CraftCategory::Alchemy)
            .with_ingredient(110, 2) // 2 herbs
            .with_ingredient(111, 1) // 1 water
            .with_output(210, 1)
            .with_byproduct(211, 1, 0.3) // 30% chance of residue
            .with_duration(2000)
            .with_skill_req("alchemy", 3)
    }

    fn setup() -> (CraftingSystem, HashMap<u64, u32>, HashMap<String, u32>) {
        let mut sys = CraftingSystem::new(5, 42);
        sys.add_recipe(iron_sword_recipe()).unwrap();
        sys.add_recipe(potion_recipe()).unwrap();
        sys.discover_recipe(1).unwrap();
        sys.discover_recipe(2).unwrap();
        let mut inv = HashMap::new();
        inv.insert(100, 10); // 10 iron ingots
        inv.insert(101, 5);  // 5 leather strips
        inv.insert(110, 8);  // 8 herbs
        inv.insert(111, 4);  // 4 water
        let mut skills = HashMap::new();
        skills.insert("blacksmithing".to_string(), 5);
        skills.insert("alchemy".to_string(), 5);
        (sys, inv, skills)
    }

    #[test]
    fn add_recipe() {
        let mut sys = CraftingSystem::new(5, 1);
        sys.add_recipe(iron_sword_recipe()).unwrap();
        assert!(sys.get_recipe(1).is_some());
    }

    #[test]
    fn duplicate_recipe() {
        let mut sys = CraftingSystem::new(5, 1);
        sys.add_recipe(iron_sword_recipe()).unwrap();
        let err = sys.add_recipe(iron_sword_recipe()).unwrap_err();
        assert!(matches!(err, CraftingError::DuplicateRecipe(1)));
    }

    #[test]
    fn invalid_recipe_no_ingredients() {
        let mut sys = CraftingSystem::new(5, 1);
        let r = Recipe::new(99, "Bad", CraftCategory::Cooking).with_output(1, 1);
        let err = sys.add_recipe(r).unwrap_err();
        assert!(matches!(err, CraftingError::InvalidRecipe(_)));
    }

    #[test]
    fn invalid_recipe_no_outputs() {
        let mut sys = CraftingSystem::new(5, 1);
        let r = Recipe::new(99, "Bad", CraftCategory::Cooking).with_ingredient(1, 1);
        let err = sys.add_recipe(r).unwrap_err();
        assert!(matches!(err, CraftingError::InvalidRecipe(_)));
    }

    #[test]
    fn discovery() {
        let mut sys = CraftingSystem::new(5, 1);
        sys.add_recipe(iron_sword_recipe()).unwrap();
        assert!(!sys.is_discovered(1));
        sys.discover_recipe(1).unwrap();
        assert!(sys.is_discovered(1));
    }

    #[test]
    fn auto_discovery_by_items() {
        let mut sys = CraftingSystem::new(5, 1);
        let r = iron_sword_recipe();
        let mut r2 = Recipe::new(3, "Steel Sword", CraftCategory::Blacksmithing)
            .with_ingredient(100, 5)
            .with_output(201, 1)
            .with_discovery_items(vec![100, 101]);
        sys.add_recipe(iron_sword_recipe()).unwrap();
        sys.add_recipe(r2).unwrap();
        let found = sys.check_discovery(&[100, 101]);
        assert!(found.contains(&3));
        assert!(sys.is_discovered(3));
    }

    #[test]
    fn can_craft_ok() {
        let (sys, inv, skills) = setup();
        assert!(sys.can_craft(1, &inv, &skills).is_ok());
    }

    #[test]
    fn can_craft_missing_ingredient() {
        let (sys, _, skills) = setup();
        let inv = HashMap::new();
        let err = sys.can_craft(1, &inv, &skills).unwrap_err();
        assert!(matches!(err, CraftingError::MissingIngredient { .. }));
    }

    #[test]
    fn can_craft_skill_too_low() {
        let (sys, inv, _) = setup();
        let mut skills = HashMap::new();
        skills.insert("alchemy".to_string(), 1);
        let err = sys.can_craft(2, &inv, &skills).unwrap_err();
        assert!(matches!(err, CraftingError::SkillTooLow { .. }));
    }

    #[test]
    fn can_craft_not_discovered() {
        let mut sys = CraftingSystem::new(5, 1);
        sys.add_recipe(iron_sword_recipe()).unwrap();
        let inv: HashMap<u64, u32> = [(100, 10), (101, 5)].into_iter().collect();
        let skills = HashMap::new();
        let err = sys.can_craft(1, &inv, &skills).unwrap_err();
        assert!(matches!(err, CraftingError::RecipeNotDiscovered(1)));
    }

    #[test]
    fn craft_immediate_success() {
        let (mut sys, mut inv, skills) = setup();
        // Force success rate 1.0
        let result = sys.craft_immediate(1, &mut inv, &skills).unwrap();
        assert!(result.success);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].item_id, 200);
        // Check ingredients consumed
        assert_eq!(*inv.get(&100).unwrap(), 7); // 10 - 3
        assert_eq!(*inv.get(&101).unwrap(), 4); // 5 - 1
        // Check output added
        assert_eq!(*inv.get(&200).unwrap(), 1);
    }

    #[test]
    fn craft_immediate_consumes_all() {
        let (mut sys, mut inv, skills) = setup();
        inv.insert(100, 3);
        inv.insert(101, 1);
        let result = sys.craft_immediate(1, &mut inv, &skills).unwrap();
        assert!(result.success);
        assert!(!inv.contains_key(&100));
        assert!(!inv.contains_key(&101));
    }

    #[test]
    fn queue_craft() {
        let (mut sys, inv, skills) = setup();
        let entry = sys.queue_craft(1, 0, &inv, &skills).unwrap();
        assert_eq!(entry.recipe_id, 1);
        assert_eq!(entry.start_time_ms, 0);
        assert_eq!(entry.end_time_ms, 5000);
        assert_eq!(sys.queue_len(), 1);
    }

    #[test]
    fn queue_chains() {
        let (mut sys, inv, skills) = setup();
        sys.queue_craft(1, 0, &inv, &skills).unwrap();
        sys.queue_craft(2, 0, &inv, &skills).unwrap();
        let entries = sys.queue_entries();
        assert_eq!(entries[0].end_time_ms, 5000);
        assert_eq!(entries[1].start_time_ms, 5000);
        assert_eq!(entries[1].end_time_ms, 7000); // 5000 + 2000
    }

    #[test]
    fn queue_full() {
        let (mut sys, inv, skills) = setup();
        for _ in 0..5 {
            sys.queue_craft(1, 0, &inv, &skills).unwrap();
        }
        let err = sys.queue_craft(1, 0, &inv, &skills).unwrap_err();
        assert!(matches!(err, CraftingError::QueueFull { max: 5 }));
    }

    #[test]
    fn collect_completed() {
        let (mut sys, inv, skills) = setup();
        sys.queue_craft(1, 0, &inv, &skills).unwrap(); // ends at 5000
        sys.queue_craft(2, 0, &inv, &skills).unwrap(); // ends at 7000
        let done = sys.collect_completed(5500);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].recipe_id, 1);
        assert_eq!(sys.queue_len(), 1); // potion still pending
    }

    #[test]
    fn auto_craft_multiple() {
        let (mut sys, inv, skills) = setup();
        let queued = sys.auto_craft(1, 3, 0, &inv, &skills).unwrap();
        assert_eq!(queued, 3);
        assert_eq!(sys.queue_len(), 3);
        let entries = sys.queue_entries();
        assert_eq!(entries[2].end_time_ms, 15000); // 3 * 5000
    }

    #[test]
    fn auto_craft_limited_by_ingredients() {
        let (mut sys, mut inv, skills) = setup();
        inv.insert(100, 6); // only 6 ingots → max 2 swords
        let queued = sys.auto_craft(1, 10, 0, &inv, &skills).unwrap();
        assert_eq!(queued, 2);
    }

    #[test]
    fn auto_craft_limited_by_queue() {
        let (mut sys, inv, skills) = setup();
        // Fill queue with 3
        sys.auto_craft(1, 3, 0, &inv, &skills).unwrap();
        // Queue max is 5, so only 2 more
        let queued = sys.auto_craft(1, 10, 0, &inv, &skills).unwrap();
        assert_eq!(queued, 2);
    }

    #[test]
    fn recipes_by_category() {
        let (sys, _, _) = setup();
        let bs = sys.recipes_by_category(CraftCategory::Blacksmithing);
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0].name, "Iron Sword");
        let al = sys.recipes_by_category(CraftCategory::Alchemy);
        assert_eq!(al.len(), 1);
    }

    #[test]
    fn discovered_recipes_list() {
        let (sys, _, _) = setup();
        let disc = sys.discovered_recipes();
        assert_eq!(disc.len(), 2);
    }

    #[test]
    fn recipe_not_found() {
        let (sys, inv, skills) = setup();
        let err = sys.can_craft(999, &inv, &skills).unwrap_err();
        assert!(matches!(err, CraftingError::RecipeNotFound(999)));
    }

    #[test]
    fn byproduct_with_high_chance() {
        // Seed chosen so byproduct triggers (tested deterministically)
        let mut sys = CraftingSystem::new(5, 12345);
        let r = Recipe::new(10, "Test", CraftCategory::Cooking)
            .with_ingredient(1, 1)
            .with_output(2, 1)
            .with_byproduct(3, 1, 1.0) // 100% chance
            .with_success_rate(1.0);
        sys.add_recipe(r).unwrap();
        sys.discover_recipe(10).unwrap();
        let mut inv: HashMap<u64, u32> = [(1, 5)].into_iter().collect();
        let skills = HashMap::new();
        let result = sys.craft_immediate(10, &mut inv, &skills).unwrap();
        assert!(result.success);
        assert_eq!(result.byproducts.len(), 1);
        assert_eq!(result.byproducts[0].item_id, 3);
    }

    #[test]
    fn success_rate_zero() {
        let mut sys = CraftingSystem::new(5, 99);
        let r = Recipe::new(20, "Hard Craft", CraftCategory::Enchanting)
            .with_ingredient(1, 1)
            .with_output(2, 1)
            .with_success_rate(0.0);
        sys.add_recipe(r).unwrap();
        sys.discover_recipe(20).unwrap();
        let mut inv: HashMap<u64, u32> = [(1, 5)].into_iter().collect();
        let skills = HashMap::new();
        let result = sys.craft_immediate(20, &mut inv, &skills).unwrap();
        assert!(!result.success);
        assert!(result.outputs.is_empty());
        // Ingredients still consumed
        assert_eq!(*inv.get(&1).unwrap(), 4);
    }
}
