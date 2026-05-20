//! RPG-style inventory system — grid/list inventory, equipment, stacking, weight.
//!
//! Replaces InventoryJS / RPG-inventory-engine with pure Rust.
//! Grid-based and list-based storage, item stacking, weight limits,
//! equipment slots, sorting, item search, and serialization.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InventoryError {
    ItemNotFound(u64),
    SlotOccupied { x: usize, y: usize },
    SlotOutOfBounds { x: usize, y: usize },
    InventoryFull,
    WeightLimitExceeded { current: u64, added: u64, limit: u64 },
    StackFull { item_id: u64, current: u32, max: u32 },
    IncompatibleSlot { item_type: ItemType, slot: EquipSlot },
    AlreadyEquipped(EquipSlot),
    NothingEquipped(EquipSlot),
    InvalidGrid { width: usize, height: usize },
    CannotStack { a: u64, b: u64 },
}

impl fmt::Display for InventoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ItemNotFound(id) => write!(f, "item not found: {id}"),
            Self::SlotOccupied { x, y } => write!(f, "slot ({x},{y}) occupied"),
            Self::SlotOutOfBounds { x, y } => write!(f, "slot ({x},{y}) out of bounds"),
            Self::InventoryFull => write!(f, "inventory is full"),
            Self::WeightLimitExceeded { current, added, limit } => {
                write!(f, "weight {current}+{added} exceeds limit {limit}")
            }
            Self::StackFull { item_id, current, max } => {
                write!(f, "stack full for item {item_id}: {current}/{max}")
            }
            Self::IncompatibleSlot { item_type, slot } => {
                write!(f, "{item_type:?} cannot equip to {slot:?}")
            }
            Self::AlreadyEquipped(s) => write!(f, "slot {s:?} already has equipment"),
            Self::NothingEquipped(s) => write!(f, "nothing equipped in {s:?}"),
            Self::InvalidGrid { width, height } => {
                write!(f, "invalid grid {width}x{height}")
            }
            Self::CannotStack { a, b } => write!(f, "items {a} and {b} cannot stack"),
        }
    }
}

impl std::error::Error for InventoryError {}

// ── Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemType {
    Weapon,
    Armor,
    Consumable,
    Material,
    Quest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquipSlot {
    Head,
    Chest,
    Legs,
    Hands,
    Weapon,
    Shield,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ItemDef {
    pub id: u64,
    pub name: String,
    pub stack_size: u32,
    pub weight: u64,
    pub rarity: Rarity,
    pub item_type: ItemType,
}

impl ItemDef {
    pub fn new(id: u64, name: &str, item_type: ItemType) -> Self {
        Self {
            id,
            name: name.to_string(),
            stack_size: 1,
            weight: 1,
            rarity: Rarity::Common,
            item_type,
        }
    }

    pub fn with_stack_size(mut self, s: u32) -> Self { self.stack_size = s; self }
    pub fn with_weight(mut self, w: u64) -> Self { self.weight = w; self }
    pub fn with_rarity(mut self, r: Rarity) -> Self { self.rarity = r; self }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ItemStack {
    pub item_id: u64,
    pub quantity: u32,
}

// ── Item Registry ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ItemRegistry {
    items: HashMap<u64, ItemDef>,
}

impl ItemRegistry {
    pub fn new() -> Self { Self { items: HashMap::new() } }

    pub fn register(&mut self, def: ItemDef) {
        self.items.insert(def.id, def);
    }

    pub fn get(&self, id: u64) -> Option<&ItemDef> {
        self.items.get(&id)
    }
}

// ── Grid Inventory ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GridInventory {
    width: usize,
    height: usize,
    slots: Vec<Option<ItemStack>>,
    weight_limit: u64,
    current_weight: u64,
    registry: ItemRegistry,
}

impl GridInventory {
    pub fn new(width: usize, height: usize, weight_limit: u64, registry: ItemRegistry) -> Result<Self, InventoryError> {
        if width == 0 || height == 0 {
            return Err(InventoryError::InvalidGrid { width, height });
        }
        let total = width * height;
        Ok(Self {
            width,
            height,
            slots: vec![None; total],
            weight_limit,
            current_weight: 0,
            registry,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn weight_limit(&self) -> u64 { self.weight_limit }
    pub fn current_weight(&self) -> u64 { self.current_weight }

    fn idx(&self, x: usize, y: usize) -> Result<usize, InventoryError> {
        if x >= self.width || y >= self.height {
            return Err(InventoryError::SlotOutOfBounds { x, y });
        }
        Ok(y * self.width + x)
    }

    pub fn get_slot(&self, x: usize, y: usize) -> Result<Option<&ItemStack>, InventoryError> {
        let i = self.idx(x, y)?;
        Ok(self.slots[i].as_ref())
    }

    fn item_weight(&self, item_id: u64, qty: u32) -> Result<u64, InventoryError> {
        let def = self.registry.get(item_id).ok_or(InventoryError::ItemNotFound(item_id))?;
        Ok(def.weight * qty as u64)
    }

    pub fn add_at(&mut self, x: usize, y: usize, item_id: u64, qty: u32) -> Result<(), InventoryError> {
        let i = self.idx(x, y)?;
        let def = self.registry.get(item_id).ok_or(InventoryError::ItemNotFound(item_id))?.clone();
        let add_weight = def.weight * qty as u64;
        if self.current_weight + add_weight > self.weight_limit {
            return Err(InventoryError::WeightLimitExceeded {
                current: self.current_weight, added: add_weight, limit: self.weight_limit,
            });
        }
        if let Some(ref mut stack) = self.slots[i] {
            if stack.item_id != item_id {
                return Err(InventoryError::CannotStack { a: stack.item_id, b: item_id });
            }
            if stack.quantity + qty > def.stack_size {
                return Err(InventoryError::StackFull {
                    item_id, current: stack.quantity, max: def.stack_size,
                });
            }
            stack.quantity += qty;
        } else {
            if qty > def.stack_size {
                return Err(InventoryError::StackFull { item_id, current: 0, max: def.stack_size });
            }
            self.slots[i] = Some(ItemStack { item_id, quantity: qty });
        }
        self.current_weight += add_weight;
        Ok(())
    }

    pub fn add_auto(&mut self, item_id: u64, mut qty: u32) -> Result<u32, InventoryError> {
        let def = self.registry.get(item_id).ok_or(InventoryError::ItemNotFound(item_id))?.clone();
        let total_weight = def.weight * qty as u64;
        if self.current_weight + total_weight > self.weight_limit {
            return Err(InventoryError::WeightLimitExceeded {
                current: self.current_weight, added: total_weight, limit: self.weight_limit,
            });
        }
        // Try stacking into existing slots first
        for slot in self.slots.iter_mut() {
            if qty == 0 { break; }
            if let Some(stack) = slot {
                if stack.item_id == item_id && stack.quantity < def.stack_size {
                    let space = def.stack_size - stack.quantity;
                    let add = space.min(qty);
                    stack.quantity += add;
                    qty -= add;
                }
            }
        }
        // Fill empty slots
        for slot in self.slots.iter_mut() {
            if qty == 0 { break; }
            if slot.is_none() {
                let add = def.stack_size.min(qty);
                *slot = Some(ItemStack { item_id, quantity: add });
                qty -= add;
            }
        }
        let added = def.weight * (total_weight / def.weight - qty as u64).max(0);
        // Recompute actual weight added
        let actually_added_qty = (total_weight / def.weight) as u32 - qty;
        let _ = added; // suppress
        self.current_weight += def.weight * actually_added_qty as u64;
        Ok(qty) // returns remainder that didn't fit
    }

    pub fn remove_at(&mut self, x: usize, y: usize, qty: u32) -> Result<ItemStack, InventoryError> {
        let i = self.idx(x, y)?;
        let stack = self.slots[i].as_ref().ok_or(InventoryError::InventoryFull)?;
        let item_id = stack.item_id;
        let have = stack.quantity;
        let remove = qty.min(have);
        let w = self.item_weight(item_id, remove)?;
        if have <= qty {
            let removed = self.slots[i].take().unwrap();
            self.current_weight = self.current_weight.saturating_sub(w);
            Ok(removed)
        } else {
            self.slots[i].as_mut().unwrap().quantity -= remove;
            self.current_weight = self.current_weight.saturating_sub(w);
            Ok(ItemStack { item_id, quantity: remove })
        }
    }

    pub fn move_item(&mut self, fx: usize, fy: usize, tx: usize, ty: usize) -> Result<(), InventoryError> {
        let fi = self.idx(fx, fy)?;
        let ti = self.idx(tx, ty)?;
        if self.slots[fi].is_none() {
            return Err(InventoryError::ItemNotFound(0));
        }
        if self.slots[ti].is_some() {
            // Swap
            self.slots.swap(fi, ti);
        } else {
            self.slots[ti] = self.slots[fi].take();
        }
        Ok(())
    }

    pub fn find<F: Fn(&ItemStack, &ItemDef) -> bool>(&self, pred: F) -> Vec<(usize, usize, &ItemStack)> {
        let mut results = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(stack) = slot {
                if let Some(def) = self.registry.get(stack.item_id) {
                    if pred(stack, def) {
                        let x = i % self.width;
                        let y = i / self.width;
                        results.push((x, y, stack));
                    }
                }
            }
        }
        results
    }

    pub fn sorted_items(&self, mode: SortMode) -> Vec<(usize, &ItemStack)> {
        let mut items: Vec<(usize, &ItemStack)> = self.slots.iter().enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|st| (i, st)))
            .collect();
        let reg = &self.registry;
        items.sort_by(|a, b| {
            let da = reg.get(a.1.item_id);
            let db = reg.get(b.1.item_id);
            match mode {
                SortMode::ByName => {
                    let na = da.map(|d| d.name.as_str()).unwrap_or("");
                    let nb = db.map(|d| d.name.as_str()).unwrap_or("");
                    na.cmp(nb)
                }
                SortMode::ByRarity => {
                    let ra = da.map(|d| d.rarity).unwrap_or(Rarity::Common);
                    let rb = db.map(|d| d.rarity).unwrap_or(Rarity::Common);
                    rb.cmp(&ra) // higher rarity first
                }
                SortMode::ByType => {
                    let ta = da.map(|d| d.item_type as u8).unwrap_or(0);
                    let tb = db.map(|d| d.item_type as u8).unwrap_or(0);
                    ta.cmp(&tb)
                }
            }
        });
        items
    }

    pub fn total_slots(&self) -> usize { self.width * self.height }
    pub fn used_slots(&self) -> usize { self.slots.iter().filter(|s| s.is_some()).count() }
    pub fn free_slots(&self) -> usize { self.total_slots() - self.used_slots() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMode {
    ByName,
    ByRarity,
    ByType,
}

// ── List Inventory ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ListInventory {
    items: Vec<ItemStack>,
    capacity: usize,
    weight_limit: u64,
    current_weight: u64,
    registry: ItemRegistry,
}

impl ListInventory {
    pub fn new(capacity: usize, weight_limit: u64, registry: ItemRegistry) -> Self {
        Self { items: Vec::new(), capacity, weight_limit, current_weight: 0, registry }
    }

    pub fn items(&self) -> &[ItemStack] { &self.items }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn current_weight(&self) -> u64 { self.current_weight }

    pub fn add(&mut self, item_id: u64, mut qty: u32) -> Result<(), InventoryError> {
        let def = self.registry.get(item_id).ok_or(InventoryError::ItemNotFound(item_id))?.clone();
        let add_weight = def.weight * qty as u64;
        if self.current_weight + add_weight > self.weight_limit {
            return Err(InventoryError::WeightLimitExceeded {
                current: self.current_weight, added: add_weight, limit: self.weight_limit,
            });
        }
        for stack in self.items.iter_mut() {
            if qty == 0 { break; }
            if stack.item_id == item_id && stack.quantity < def.stack_size {
                let space = def.stack_size - stack.quantity;
                let add = space.min(qty);
                stack.quantity += add;
                qty -= add;
            }
        }
        while qty > 0 {
            if self.items.len() >= self.capacity {
                return Err(InventoryError::InventoryFull);
            }
            let add = def.stack_size.min(qty);
            self.items.push(ItemStack { item_id, quantity: add });
            qty -= add;
        }
        self.current_weight += add_weight;
        Ok(())
    }

    pub fn remove(&mut self, item_id: u64, mut qty: u32) -> Result<u32, InventoryError> {
        let def = self.registry.get(item_id).ok_or(InventoryError::ItemNotFound(item_id))?;
        let unit_w = def.weight;
        let mut removed = 0u32;
        self.items.retain_mut(|stack| {
            if qty == 0 || stack.item_id != item_id { return true; }
            if stack.quantity <= qty {
                qty -= stack.quantity;
                removed += stack.quantity;
                false
            } else {
                stack.quantity -= qty;
                removed += qty;
                qty = 0;
                true
            }
        });
        self.current_weight = self.current_weight.saturating_sub(unit_w * removed as u64);
        Ok(removed)
    }

    pub fn count(&self, item_id: u64) -> u32 {
        self.items.iter().filter(|s| s.item_id == item_id).map(|s| s.quantity).sum()
    }

    pub fn find<F: Fn(&ItemStack) -> bool>(&self, pred: F) -> Vec<&ItemStack> {
        self.items.iter().filter(|s| pred(s)).collect()
    }
}

// ── Equipment ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Equipment {
    slots: HashMap<EquipSlot, ItemStack>,
    registry: ItemRegistry,
}

impl Equipment {
    pub fn new(registry: ItemRegistry) -> Self {
        Self { slots: HashMap::new(), registry }
    }

    fn compatible(item_type: ItemType, slot: EquipSlot) -> bool {
        matches!((item_type, slot),
            (ItemType::Armor, EquipSlot::Head) |
            (ItemType::Armor, EquipSlot::Chest) |
            (ItemType::Armor, EquipSlot::Legs) |
            (ItemType::Armor, EquipSlot::Hands) |
            (ItemType::Weapon, EquipSlot::Weapon) |
            (ItemType::Armor, EquipSlot::Shield)
        )
    }

    pub fn equip(&mut self, slot: EquipSlot, item_id: u64) -> Result<Option<ItemStack>, InventoryError> {
        let def = self.registry.get(item_id).ok_or(InventoryError::ItemNotFound(item_id))?;
        if !Self::compatible(def.item_type, slot) {
            return Err(InventoryError::IncompatibleSlot { item_type: def.item_type, slot });
        }
        let prev = self.slots.insert(slot, ItemStack { item_id, quantity: 1 });
        Ok(prev)
    }

    pub fn unequip(&mut self, slot: EquipSlot) -> Result<ItemStack, InventoryError> {
        self.slots.remove(&slot).ok_or(InventoryError::NothingEquipped(slot))
    }

    pub fn get(&self, slot: EquipSlot) -> Option<&ItemStack> {
        self.slots.get(&slot)
    }

    pub fn equipped_slots(&self) -> Vec<EquipSlot> {
        let mut slots: Vec<EquipSlot> = self.slots.keys().copied().collect();
        slots.sort_by_key(|s| *s as u8);
        slots
    }

    pub fn total_weight(&self) -> u64 {
        self.slots.values().map(|stack| {
            self.registry.get(stack.item_id).map(|d| d.weight).unwrap_or(0)
        }).sum()
    }
}

// ── Serialization ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SerializedInventory {
    pub slots: Vec<Option<(u64, u32)>>,
    pub width: usize,
    pub height: usize,
}

impl GridInventory {
    pub fn serialize(&self) -> SerializedInventory {
        SerializedInventory {
            slots: self.slots.iter().map(|s| s.as_ref().map(|st| (st.item_id, st.quantity))).collect(),
            width: self.width,
            height: self.height,
        }
    }

    pub fn deserialize(data: &SerializedInventory, weight_limit: u64, registry: ItemRegistry) -> Result<Self, InventoryError> {
        let mut inv = Self::new(data.width, data.height, weight_limit, registry)?;
        for (i, slot) in data.slots.iter().enumerate() {
            if let Some((item_id, qty)) = slot {
                let x = i % data.width;
                let y = i / data.width;
                inv.add_at(x, y, *item_id, *qty)?;
            }
        }
        Ok(inv)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> ItemRegistry {
        let mut reg = ItemRegistry::new();
        reg.register(ItemDef::new(1, "Iron Sword", ItemType::Weapon).with_weight(5).with_stack_size(1));
        reg.register(ItemDef::new(2, "Health Potion", ItemType::Consumable).with_weight(1).with_stack_size(10));
        reg.register(ItemDef::new(3, "Iron Ore", ItemType::Material).with_weight(3).with_stack_size(20).with_rarity(Rarity::Common));
        reg.register(ItemDef::new(4, "Dragon Scale", ItemType::Material).with_weight(2).with_stack_size(5).with_rarity(Rarity::Legendary));
        reg.register(ItemDef::new(5, "Steel Helm", ItemType::Armor).with_weight(4).with_stack_size(1).with_rarity(Rarity::Uncommon));
        reg.register(ItemDef::new(6, "Quest Scroll", ItemType::Quest).with_weight(0).with_stack_size(1));
        reg.register(ItemDef::new(7, "Leather Gloves", ItemType::Armor).with_weight(2));
        reg.register(ItemDef::new(8, "Chain Mail", ItemType::Armor).with_weight(10).with_rarity(Rarity::Rare));
        reg
    }

    #[test]
    fn grid_create() {
        let reg = test_registry();
        let inv = GridInventory::new(4, 3, 100, reg).unwrap();
        assert_eq!(inv.total_slots(), 12);
        assert_eq!(inv.free_slots(), 12);
    }

    #[test]
    fn grid_zero_dimension() {
        let reg = test_registry();
        assert!(GridInventory::new(0, 3, 100, reg).is_err());
    }

    #[test]
    fn add_item_at_slot() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 1, 1).unwrap();
        let slot = inv.get_slot(0, 0).unwrap().unwrap();
        assert_eq!(slot.item_id, 1);
        assert_eq!(slot.quantity, 1);
        assert_eq!(inv.current_weight(), 5);
    }

    #[test]
    fn stacking() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(1, 0, 2, 5).unwrap();
        inv.add_at(1, 0, 2, 3).unwrap();
        let slot = inv.get_slot(1, 0).unwrap().unwrap();
        assert_eq!(slot.quantity, 8);
    }

    #[test]
    fn stack_overflow() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 2, 10).unwrap();
        let err = inv.add_at(0, 0, 2, 1).unwrap_err();
        assert!(matches!(err, InventoryError::StackFull { .. }));
    }

    #[test]
    fn cannot_stack_different_items() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 200, reg).unwrap();
        inv.add_at(0, 0, 1, 1).unwrap();
        let err = inv.add_at(0, 0, 2, 1).unwrap_err();
        assert!(matches!(err, InventoryError::CannotStack { .. }));
    }

    #[test]
    fn weight_limit() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 10, reg).unwrap();
        inv.add_at(0, 0, 1, 1).unwrap(); // 5 weight
        inv.add_at(1, 0, 1, 1).unwrap(); // 5 more = 10, at limit
        assert_eq!(inv.current_weight(), 10);
        // Third sword would exceed limit
        let err = inv.add_at(2, 0, 1, 1).unwrap_err();
        assert!(matches!(err, InventoryError::WeightLimitExceeded { .. }));
    }

    #[test]
    fn remove_partial() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 2, 8).unwrap();
        let removed = inv.remove_at(0, 0, 3).unwrap();
        assert_eq!(removed.quantity, 3);
        assert_eq!(inv.get_slot(0, 0).unwrap().unwrap().quantity, 5);
    }

    #[test]
    fn remove_all() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 2, 5).unwrap();
        let removed = inv.remove_at(0, 0, 10).unwrap();
        assert_eq!(removed.quantity, 5);
        assert!(inv.get_slot(0, 0).unwrap().is_none());
    }

    #[test]
    fn move_item_to_empty() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 1, 1).unwrap();
        inv.move_item(0, 0, 2, 1).unwrap();
        assert!(inv.get_slot(0, 0).unwrap().is_none());
        assert_eq!(inv.get_slot(2, 1).unwrap().unwrap().item_id, 1);
    }

    #[test]
    fn move_item_swap() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 1, 1).unwrap();
        inv.add_at(1, 0, 2, 3).unwrap();
        inv.move_item(0, 0, 1, 0).unwrap();
        assert_eq!(inv.get_slot(0, 0).unwrap().unwrap().item_id, 2);
        assert_eq!(inv.get_slot(1, 0).unwrap().unwrap().item_id, 1);
    }

    #[test]
    fn out_of_bounds() {
        let reg = test_registry();
        let inv = GridInventory::new(4, 3, 100, reg).unwrap();
        let err = inv.get_slot(5, 0).unwrap_err();
        assert!(matches!(err, InventoryError::SlotOutOfBounds { .. }));
    }

    #[test]
    fn find_items() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 100, reg).unwrap();
        inv.add_at(0, 0, 2, 5).unwrap();
        inv.add_at(1, 0, 3, 2).unwrap();
        inv.add_at(2, 0, 2, 3).unwrap();
        let found = inv.find(|stack, _def| stack.item_id == 2);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn sorted_by_name() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 200, reg).unwrap();
        inv.add_at(0, 0, 4, 1).unwrap(); // Dragon Scale
        inv.add_at(1, 0, 2, 1).unwrap(); // Health Potion
        inv.add_at(2, 0, 1, 1).unwrap(); // Iron Sword
        let sorted = inv.sorted_items(SortMode::ByName);
        let names: Vec<&str> = sorted.iter().map(|(_, s)| {
            inv.registry.get(s.item_id).unwrap().name.as_str()
        }).collect();
        assert_eq!(names, vec!["Dragon Scale", "Health Potion", "Iron Sword"]);
    }

    #[test]
    fn sorted_by_rarity() {
        let reg = test_registry();
        let mut inv = GridInventory::new(4, 3, 200, reg).unwrap();
        inv.add_at(0, 0, 3, 1).unwrap(); // Common
        inv.add_at(1, 0, 4, 1).unwrap(); // Legendary
        inv.add_at(2, 0, 5, 1).unwrap(); // Uncommon
        let sorted = inv.sorted_items(SortMode::ByRarity);
        let rarities: Vec<Rarity> = sorted.iter().map(|(_, s)| {
            inv.registry.get(s.item_id).unwrap().rarity
        }).collect();
        assert_eq!(rarities, vec![Rarity::Legendary, Rarity::Uncommon, Rarity::Common]);
    }

    #[test]
    fn list_inventory_basic() {
        let reg = test_registry();
        let mut inv = ListInventory::new(10, 100, reg);
        inv.add(2, 5).unwrap();
        assert_eq!(inv.count(2), 5);
        assert_eq!(inv.current_weight(), 5);
    }

    #[test]
    fn list_inventory_remove() {
        let reg = test_registry();
        let mut inv = ListInventory::new(10, 100, reg);
        inv.add(3, 10).unwrap();
        let removed = inv.remove(3, 4).unwrap();
        assert_eq!(removed, 4);
        assert_eq!(inv.count(3), 6);
    }

    #[test]
    fn list_full() {
        let reg = test_registry();
        let mut inv = ListInventory::new(1, 1000, reg);
        inv.add(1, 1).unwrap();
        let err = inv.add(5, 1).unwrap_err();
        assert!(matches!(err, InventoryError::InventoryFull));
    }

    #[test]
    fn equip_weapon() {
        let reg = test_registry();
        let mut eq = Equipment::new(reg);
        let prev = eq.equip(EquipSlot::Weapon, 1).unwrap();
        assert!(prev.is_none());
        assert_eq!(eq.get(EquipSlot::Weapon).unwrap().item_id, 1);
    }

    #[test]
    fn equip_incompatible() {
        let reg = test_registry();
        let mut eq = Equipment::new(reg);
        let err = eq.equip(EquipSlot::Head, 1).unwrap_err(); // weapon in head
        assert!(matches!(err, InventoryError::IncompatibleSlot { .. }));
    }

    #[test]
    fn unequip() {
        let reg = test_registry();
        let mut eq = Equipment::new(reg);
        eq.equip(EquipSlot::Weapon, 1).unwrap();
        let item = eq.unequip(EquipSlot::Weapon).unwrap();
        assert_eq!(item.item_id, 1);
        assert!(eq.get(EquipSlot::Weapon).is_none());
    }

    #[test]
    fn unequip_empty() {
        let reg = test_registry();
        let mut eq = Equipment::new(reg);
        let err = eq.unequip(EquipSlot::Head).unwrap_err();
        assert!(matches!(err, InventoryError::NothingEquipped(_)));
    }

    #[test]
    fn equipment_weight() {
        let reg = test_registry();
        let mut eq = Equipment::new(reg);
        eq.equip(EquipSlot::Weapon, 1).unwrap();  // 5
        eq.equip(EquipSlot::Head, 5).unwrap();     // 4
        assert_eq!(eq.total_weight(), 9);
    }

    #[test]
    fn serialize_roundtrip() {
        let reg = test_registry();
        let mut inv = GridInventory::new(3, 2, 100, reg.clone()).unwrap();
        inv.add_at(0, 0, 2, 5).unwrap();
        inv.add_at(2, 1, 3, 3).unwrap();
        let data = inv.serialize();
        let inv2 = GridInventory::deserialize(&data, 100, reg).unwrap();
        assert_eq!(inv2.get_slot(0, 0).unwrap().unwrap().quantity, 5);
        assert_eq!(inv2.get_slot(2, 1).unwrap().unwrap().item_id, 3);
        assert!(inv2.get_slot(1, 0).unwrap().is_none());
    }

    #[test]
    fn add_auto_fills_stacks() {
        let reg = test_registry();
        let mut inv = GridInventory::new(2, 1, 200, reg).unwrap();
        inv.add_at(0, 0, 2, 7).unwrap(); // 7/10 potions
        let remainder = inv.add_auto(2, 5).unwrap(); // add 5 more
        // slot 0 fills to 10, slot 1 gets 2
        assert_eq!(inv.get_slot(0, 0).unwrap().unwrap().quantity, 10);
        assert_eq!(inv.get_slot(1, 0).unwrap().unwrap().quantity, 2);
        assert_eq!(remainder, 0);
    }

    #[test]
    fn item_registry_lookup() {
        let reg = test_registry();
        let def = reg.get(4).unwrap();
        assert_eq!(def.name, "Dragon Scale");
        assert_eq!(def.rarity, Rarity::Legendary);
        assert!(reg.get(999).is_none());
    }
}
