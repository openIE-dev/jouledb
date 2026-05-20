//! Minimap rendering system: top-down or radar-style, circular/rectangular
//! viewport, entity icons, fog of war, zoom, rotation, click-to-ping,
//! and border styling.
//!
//! Pure layout math — outputs draw commands; actual pixel pushing is
//! the renderer's job.

use std::collections::HashMap;

// ── Minimap Shape ──────────────────────────────────────────────

/// Clipping mask shape for the minimap viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimapShape {
    Rectangular,
    Circular,
}

/// Whether the minimap rotates with the player or stays north-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationMode {
    /// Minimap always shows north at the top.
    NorthUp,
    /// Minimap rotates so the player always faces up.
    RotateWithPlayer,
}

// ── Entity Icon ────────────────────────────────────────────────

/// Visual representation on the minimap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IconShape {
    Circle,
    Triangle,
    Square,
    Diamond,
    Star,
}

/// RGBA color for minimap icons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimapColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl MinimapColor {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub fn with_alpha(mut self, a: f64) -> Self {
        self.a = a;
        self
    }
}

/// Category of an entity for minimap purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityCategory {
    Player,
    Ally,
    Enemy,
    Item,
    Objective,
    Custom(u32),
}

/// An entity to display on the minimap.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimapEntity {
    pub id: String,
    pub category: EntityCategory,
    /// World position.
    pub world_x: f64,
    pub world_y: f64,
    /// Facing direction in radians (0 = up/north).
    pub rotation: f64,
    pub shape: IconShape,
    pub color: MinimapColor,
    /// Base icon size in minimap pixels.
    pub icon_size: f64,
    pub visible: bool,
}

impl MinimapEntity {
    pub fn new(id: &str, cat: EntityCategory, x: f64, y: f64) -> Self {
        let (shape, color) = default_style(cat);
        Self {
            id: id.to_string(),
            category: cat,
            world_x: x,
            world_y: y,
            rotation: 0.0,
            shape,
            color,
            icon_size: 6.0,
            visible: true,
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.world_x = x;
        self.world_y = y;
        self
    }

    pub fn with_rotation(mut self, rad: f64) -> Self {
        self.rotation = rad;
        self
    }

    pub fn with_style(mut self, shape: IconShape, color: MinimapColor, size: f64) -> Self {
        self.shape = shape;
        self.color = color;
        self.icon_size = size;
        self
    }
}

fn default_style(cat: EntityCategory) -> (IconShape, MinimapColor) {
    match cat {
        EntityCategory::Player => (IconShape::Triangle, MinimapColor::new(0.0, 1.0, 0.0)),
        EntityCategory::Ally => (IconShape::Circle, MinimapColor::new(0.0, 0.8, 1.0)),
        EntityCategory::Enemy => (IconShape::Diamond, MinimapColor::new(1.0, 0.2, 0.2)),
        EntityCategory::Item => (IconShape::Square, MinimapColor::new(1.0, 1.0, 0.0)),
        EntityCategory::Objective => (IconShape::Star, MinimapColor::new(1.0, 0.8, 0.0)),
        EntityCategory::Custom(_) => (IconShape::Circle, MinimapColor::new(0.7, 0.7, 0.7)),
    }
}

// ── Fog of War ─────────────────────────────────────────────────

/// Fog of war state for a grid cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FogState {
    /// Never seen.
    Hidden,
    /// Previously explored but not currently visible.
    Explored,
    /// Currently visible.
    Visible,
}

/// Grid-based fog of war.
#[derive(Debug, Clone)]
pub struct FogOfWar {
    /// Grid dimensions.
    pub cols: u32,
    pub rows: u32,
    /// World-space size of each fog cell.
    pub cell_size: f64,
    /// Origin in world space.
    pub origin_x: f64,
    pub origin_y: f64,
    cells: Vec<FogState>,
}

impl FogOfWar {
    pub fn new(cols: u32, rows: u32, cell_size: f64) -> Self {
        Self {
            cols,
            rows,
            cell_size,
            origin_x: 0.0,
            origin_y: 0.0,
            cells: vec![FogState::Hidden; (cols * rows) as usize],
        }
    }

    pub fn with_origin(mut self, x: f64, y: f64) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self
    }

    fn index(&self, col: u32, row: u32) -> Option<usize> {
        if col < self.cols && row < self.rows {
            Some((row * self.cols + col) as usize)
        } else {
            None
        }
    }

    pub fn get_cell(&self, col: u32, row: u32) -> FogState {
        self.index(col, row).map_or(FogState::Hidden, |i| self.cells[i])
    }

    pub fn set_cell(&mut self, col: u32, row: u32, state: FogState) {
        if let Some(i) = self.index(col, row) {
            self.cells[i] = state;
        }
    }

    /// Reveal cells within `radius` of a world position.
    pub fn reveal_around(&mut self, world_x: f64, world_y: f64, radius: f64) {
        let c_col = ((world_x - self.origin_x) / self.cell_size).floor() as i64;
        let c_row = ((world_y - self.origin_y) / self.cell_size).floor() as i64;
        let r_cells = (radius / self.cell_size).ceil() as i64;

        for dr in -r_cells..=r_cells {
            for dc in -r_cells..=r_cells {
                let col = c_col + dc;
                let row = c_row + dr;
                if col < 0 || row < 0 || col >= self.cols as i64 || row >= self.rows as i64 {
                    continue;
                }
                let cell_cx = self.origin_x + (col as f64 + 0.5) * self.cell_size;
                let cell_cy = self.origin_y + (row as f64 + 0.5) * self.cell_size;
                let dist = ((cell_cx - world_x).powi(2) + (cell_cy - world_y).powi(2)).sqrt();
                if dist <= radius {
                    let idx = (row as u32 * self.cols + col as u32) as usize;
                    self.cells[idx] = FogState::Visible;
                }
            }
        }
    }

    /// Demote all `Visible` cells to `Explored`.
    pub fn demote_visible(&mut self) {
        for cell in &mut self.cells {
            if *cell == FogState::Visible {
                *cell = FogState::Explored;
            }
        }
    }

    /// Query fog state at a world position.
    pub fn fog_at(&self, world_x: f64, world_y: f64) -> FogState {
        let col = ((world_x - self.origin_x) / self.cell_size).floor() as i64;
        let row = ((world_y - self.origin_y) / self.cell_size).floor() as i64;
        if col < 0 || row < 0 || col >= self.cols as i64 || row >= self.rows as i64 {
            return FogState::Hidden;
        }
        self.get_cell(col as u32, row as u32)
    }
}

// ── Ping ───────────────────────────────────────────────────────

/// A click-to-ping marker.
#[derive(Debug, Clone, PartialEq)]
pub struct Ping {
    pub world_x: f64,
    pub world_y: f64,
    pub color: MinimapColor,
    pub duration: f64,
    pub elapsed: f64,
}

impl Ping {
    pub fn new(wx: f64, wy: f64, color: MinimapColor, duration: f64) -> Self {
        Self { world_x: wx, world_y: wy, color, duration, elapsed: 0.0 }
    }

    pub fn is_expired(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Current animation progress [0, 1].
    pub fn progress(&self) -> f64 {
        if self.duration <= 0.0 { 1.0 } else { (self.elapsed / self.duration).clamp(0.0, 1.0) }
    }
}

// ── Draw Commands ──────────────────────────────────────────────

/// A minimap icon draw command (output to renderer).
#[derive(Debug, Clone, PartialEq)]
pub struct MinimapIconCmd {
    pub entity_id: String,
    /// Position in minimap-local pixels.
    pub minimap_x: f64,
    pub minimap_y: f64,
    pub size: f64,
    pub rotation: f64,
    pub shape: IconShape,
    pub color: MinimapColor,
}

/// A minimap ping draw command.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimapPingCmd {
    pub minimap_x: f64,
    pub minimap_y: f64,
    pub radius: f64,
    pub color: MinimapColor,
    pub alpha: f64,
}

// ── Border Style ───────────────────────────────────────────────

/// Visual styling for the minimap border.
#[derive(Debug, Clone, PartialEq)]
pub struct BorderStyle {
    pub color: MinimapColor,
    pub thickness: f64,
    /// For circular minimaps, adds a ring; for rectangular, a rectangle outline.
    pub visible: bool,
}

impl Default for BorderStyle {
    fn default() -> Self {
        Self {
            color: MinimapColor::new(0.8, 0.8, 0.8),
            thickness: 2.0,
            visible: true,
        }
    }
}

// ── Minimap System ─────────────────────────────────────────────

/// The minimap renderer/layout engine.
#[derive(Debug, Clone)]
pub struct Minimap {
    /// Minimap viewport position on screen (top-left).
    pub screen_x: f64,
    pub screen_y: f64,
    /// Minimap viewport size in screen pixels.
    pub view_width: f64,
    pub view_height: f64,
    /// Shape of the minimap viewport.
    pub shape: MinimapShape,
    /// Center of the minimap in world coordinates (usually the player).
    pub center_x: f64,
    pub center_y: f64,
    /// World units visible per minimap pixel.
    pub zoom: f64,
    /// Rotation mode.
    pub rotation_mode: RotationMode,
    /// Current map rotation in radians (for RotateWithPlayer).
    pub map_rotation: f64,
    /// Distance-based icon scaling: icons shrink beyond this distance.
    pub icon_scale_distance: f64,
    /// Minimum icon scale factor.
    pub min_icon_scale: f64,

    pub border: BorderStyle,
    entities: HashMap<String, MinimapEntity>,
    pub fog: Option<FogOfWar>,
    pings: Vec<Ping>,
}

impl Minimap {
    pub fn new(screen_x: f64, screen_y: f64, size: f64) -> Self {
        Self {
            screen_x,
            screen_y,
            view_width: size,
            view_height: size,
            shape: MinimapShape::Circular,
            center_x: 0.0,
            center_y: 0.0,
            zoom: 1.0,
            rotation_mode: RotationMode::NorthUp,
            map_rotation: 0.0,
            icon_scale_distance: 0.0,
            min_icon_scale: 0.5,
            border: BorderStyle::default(),
            entities: HashMap::new(),
            fog: None,
            pings: Vec::new(),
        }
    }

    pub fn with_shape(mut self, shape: MinimapShape) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_zoom(mut self, zoom: f64) -> Self {
        self.zoom = zoom.max(0.01);
        self
    }

    pub fn with_rotation_mode(mut self, mode: RotationMode) -> Self {
        self.rotation_mode = mode;
        self
    }

    pub fn set_center(&mut self, x: f64, y: f64) {
        self.center_x = x;
        self.center_y = y;
    }

    pub fn set_map_rotation(&mut self, rad: f64) {
        self.map_rotation = rad;
    }

    pub fn set_zoom(&mut self, zoom: f64) {
        self.zoom = zoom.max(0.01);
    }

    pub fn set_fog(&mut self, fog: FogOfWar) {
        self.fog = Some(fog);
    }

    // ── Entity Management ───────────────────────────────────

    pub fn add_entity(&mut self, entity: MinimapEntity) {
        self.entities.insert(entity.id.clone(), entity);
    }

    pub fn remove_entity(&mut self, id: &str) -> bool {
        self.entities.remove(id).is_some()
    }

    pub fn update_entity_position(&mut self, id: &str, x: f64, y: f64) {
        if let Some(e) = self.entities.get_mut(id) {
            e.world_x = x;
            e.world_y = y;
        }
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    // ── Ping ────────────────────────────────────────────────

    pub fn add_ping(&mut self, world_x: f64, world_y: f64, color: MinimapColor, duration: f64) {
        self.pings.push(Ping::new(world_x, world_y, color, duration));
    }

    /// Advance pings and remove expired ones.
    pub fn update(&mut self, dt: f64) {
        for ping in &mut self.pings {
            ping.elapsed += dt;
        }
        self.pings.retain(|p| !p.is_expired());
    }

    // ── Coordinate Transforms ───────────────────────────────

    /// Transform a world position to minimap-local pixel coordinates.
    pub fn world_to_minimap(&self, world_x: f64, world_y: f64) -> (f64, f64) {
        let dx = world_x - self.center_x;
        let dy = world_y - self.center_y;

        // Apply rotation
        let (rdx, rdy) = match self.rotation_mode {
            RotationMode::NorthUp => (dx, dy),
            RotationMode::RotateWithPlayer => {
                let cos = self.map_rotation.cos();
                let sin = self.map_rotation.sin();
                (dx * cos - dy * sin, dx * sin + dy * cos)
            }
        };

        // Scale: world units → minimap pixels
        let mx = rdx / self.zoom + self.view_width / 2.0;
        let my = rdy / self.zoom + self.view_height / 2.0;
        (mx, my)
    }

    /// Transform minimap-local pixel coordinates to world position.
    pub fn minimap_to_world(&self, mx: f64, my: f64) -> (f64, f64) {
        let rdx = (mx - self.view_width / 2.0) * self.zoom;
        let rdy = (my - self.view_height / 2.0) * self.zoom;

        let (dx, dy) = match self.rotation_mode {
            RotationMode::NorthUp => (rdx, rdy),
            RotationMode::RotateWithPlayer => {
                let cos = (-self.map_rotation).cos();
                let sin = (-self.map_rotation).sin();
                (rdx * cos - rdy * sin, rdx * sin + rdy * cos)
            }
        };

        (self.center_x + dx, self.center_y + dy)
    }

    /// Test whether a minimap-local position is inside the viewport.
    pub fn is_in_viewport(&self, mx: f64, my: f64) -> bool {
        match self.shape {
            MinimapShape::Rectangular => {
                mx >= 0.0 && mx <= self.view_width && my >= 0.0 && my <= self.view_height
            }
            MinimapShape::Circular => {
                let cx = self.view_width / 2.0;
                let cy = self.view_height / 2.0;
                let r = self.view_width.min(self.view_height) / 2.0;
                let dist = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
                dist <= r
            }
        }
    }

    /// Distance-based icon scaling.
    fn icon_scale_for_distance(&self, world_dist: f64) -> f64 {
        if self.icon_scale_distance <= 0.0 {
            return 1.0;
        }
        let t = (world_dist / self.icon_scale_distance).clamp(0.0, 1.0);
        let scale = 1.0 - t * (1.0 - self.min_icon_scale);
        scale.max(self.min_icon_scale)
    }

    // ── Rendering ───────────────────────────────────────────

    /// Generate draw commands for all visible entities.
    pub fn render_icons(&self) -> Vec<MinimapIconCmd> {
        let mut cmds = Vec::new();

        for entity in self.entities.values() {
            if !entity.visible {
                continue;
            }

            // Check fog of war
            if let Some(fog) = &self.fog {
                let state = fog.fog_at(entity.world_x, entity.world_y);
                if state == FogState::Hidden {
                    continue;
                }
                // Explored but not visible: skip enemies
                if state == FogState::Explored && entity.category == EntityCategory::Enemy {
                    continue;
                }
            }

            let (mx, my) = self.world_to_minimap(entity.world_x, entity.world_y);

            if !self.is_in_viewport(mx, my) {
                continue;
            }

            let world_dist = ((entity.world_x - self.center_x).powi(2)
                + (entity.world_y - self.center_y).powi(2))
            .sqrt();
            let dist_scale = self.icon_scale_for_distance(world_dist);

            cmds.push(MinimapIconCmd {
                entity_id: entity.id.clone(),
                minimap_x: mx,
                minimap_y: my,
                size: entity.icon_size * dist_scale,
                rotation: entity.rotation,
                shape: entity.shape,
                color: entity.color,
            });
        }

        cmds
    }

    /// Generate draw commands for active pings.
    pub fn render_pings(&self) -> Vec<MinimapPingCmd> {
        self.pings
            .iter()
            .filter(|p| !p.is_expired())
            .map(|p| {
                let (mx, my) = self.world_to_minimap(p.world_x, p.world_y);
                let progress = p.progress();
                MinimapPingCmd {
                    minimap_x: mx,
                    minimap_y: my,
                    radius: 10.0 + progress * 20.0,
                    color: p.color,
                    alpha: 1.0 - progress,
                }
            })
            .collect()
    }

    /// Handle a click on the minimap, returns the world position pinged.
    pub fn click_to_ping(
        &mut self,
        screen_click_x: f64,
        screen_click_y: f64,
        color: MinimapColor,
        duration: f64,
    ) -> Option<(f64, f64)> {
        let local_x = screen_click_x - self.screen_x;
        let local_y = screen_click_y - self.screen_y;
        if !self.is_in_viewport(local_x, local_y) {
            return None;
        }
        let (wx, wy) = self.minimap_to_world(local_x, local_y);
        self.add_ping(wx, wy, color, duration);
        Some((wx, wy))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn basic_minimap() -> Minimap {
        Minimap::new(10.0, 10.0, 100.0).with_zoom(10.0)
    }

    #[test]
    fn world_to_minimap_center() {
        let mm = basic_minimap();
        let (mx, my) = mm.world_to_minimap(0.0, 0.0);
        assert!((mx - 50.0).abs() < EPS);
        assert!((my - 50.0).abs() < EPS);
    }

    #[test]
    fn world_to_minimap_offset() {
        let mm = basic_minimap();
        // 100 world units east → 100/10 = 10 minimap pixels right of center
        let (mx, _) = mm.world_to_minimap(100.0, 0.0);
        assert!((mx - 60.0).abs() < EPS);
    }

    #[test]
    fn minimap_to_world_round_trip() {
        let mm = basic_minimap();
        let (mx, my) = mm.world_to_minimap(150.0, -75.0);
        let (wx, wy) = mm.minimap_to_world(mx, my);
        assert!((wx - 150.0).abs() < EPS);
        assert!((wy - (-75.0)).abs() < EPS);
    }

    #[test]
    fn rotate_with_player() {
        let mut mm = basic_minimap().with_rotation_mode(RotationMode::RotateWithPlayer);
        mm.set_map_rotation(std::f64::consts::FRAC_PI_2); // 90°
        let (mx, my) = mm.world_to_minimap(100.0, 0.0);
        // Rotated 90°: east becomes down on minimap... or up depending on sign
        // cos(90)=0, sin(90)=1: rdx = 100*0 - 0*1 = 0, rdy = 100*1 + 0*0 = 100
        // mx = 0/10+50 = 50, my = 100/10+50 = 60
        assert!((mx - 50.0).abs() < EPS);
        assert!((my - 60.0).abs() < EPS);
    }

    #[test]
    fn circular_viewport_test() {
        let mm = basic_minimap().with_shape(MinimapShape::Circular);
        assert!(mm.is_in_viewport(50.0, 50.0)); // center
        assert!(!mm.is_in_viewport(0.0, 0.0)); // corner
    }

    #[test]
    fn rectangular_viewport_test() {
        let mm = basic_minimap().with_shape(MinimapShape::Rectangular);
        assert!(mm.is_in_viewport(50.0, 50.0));
        assert!(mm.is_in_viewport(0.0, 0.0));
        assert!(!mm.is_in_viewport(-1.0, 50.0));
    }

    #[test]
    fn add_remove_entity() {
        let mut mm = basic_minimap();
        mm.add_entity(MinimapEntity::new("player", EntityCategory::Player, 0.0, 0.0));
        assert_eq!(mm.entity_count(), 1);
        assert!(mm.remove_entity("player"));
        assert_eq!(mm.entity_count(), 0);
    }

    #[test]
    fn update_entity_position() {
        let mut mm = basic_minimap();
        mm.add_entity(MinimapEntity::new("p", EntityCategory::Player, 0.0, 0.0));
        mm.update_entity_position("p", 100.0, 200.0);
        let icons = mm.render_icons();
        assert_eq!(icons.len(), 1);
        // Should be offset from center
        assert!((icons[0].minimap_x - 60.0).abs() < EPS);
    }

    #[test]
    fn entity_default_styles() {
        let e = MinimapEntity::new("e", EntityCategory::Enemy, 0.0, 0.0);
        assert_eq!(e.shape, IconShape::Diamond);
        assert!((e.color.r - 1.0).abs() < EPS);
    }

    #[test]
    fn render_icons_excludes_hidden() {
        let mut mm = basic_minimap();
        let mut entity = MinimapEntity::new("hidden", EntityCategory::Item, 0.0, 0.0);
        entity.visible = false;
        mm.add_entity(entity);
        let icons = mm.render_icons();
        assert!(icons.is_empty());
    }

    #[test]
    fn render_icons_excludes_out_of_viewport() {
        let mut mm = basic_minimap();
        // 1000 world units away at zoom=10 → 100 minimap px from center, past edge
        mm.add_entity(MinimapEntity::new("far", EntityCategory::Enemy, 1000.0, 0.0));
        let icons = mm.render_icons();
        assert!(icons.is_empty());
    }

    #[test]
    fn fog_of_war_hidden() {
        let mut mm = basic_minimap();
        let fog = FogOfWar::new(10, 10, 100.0);
        mm.set_fog(fog);
        mm.add_entity(MinimapEntity::new("e", EntityCategory::Enemy, 50.0, 50.0));
        // Everything is hidden by default
        let icons = mm.render_icons();
        assert!(icons.is_empty());
    }

    #[test]
    fn fog_of_war_reveal() {
        let mut mm = basic_minimap();
        let mut fog = FogOfWar::new(10, 10, 100.0);
        fog.reveal_around(50.0, 50.0, 150.0);
        mm.set_fog(fog);
        mm.add_entity(MinimapEntity::new("e", EntityCategory::Item, 50.0, 50.0));
        let icons = mm.render_icons();
        assert_eq!(icons.len(), 1);
    }

    #[test]
    fn fog_explored_hides_enemies() {
        let mut mm = basic_minimap();
        let mut fog = FogOfWar::new(10, 10, 100.0);
        fog.reveal_around(50.0, 50.0, 200.0);
        fog.demote_visible(); // Explored, not Visible
        mm.set_fog(fog);
        mm.add_entity(MinimapEntity::new("e", EntityCategory::Enemy, 50.0, 50.0));
        let icons = mm.render_icons();
        assert!(icons.is_empty());
    }

    #[test]
    fn fog_explored_shows_items() {
        let mut mm = basic_minimap();
        let mut fog = FogOfWar::new(10, 10, 100.0);
        fog.reveal_around(50.0, 50.0, 200.0);
        fog.demote_visible();
        mm.set_fog(fog);
        mm.add_entity(MinimapEntity::new("i", EntityCategory::Item, 50.0, 50.0));
        let icons = mm.render_icons();
        assert_eq!(icons.len(), 1);
    }

    #[test]
    fn ping_lifecycle() {
        let mut mm = basic_minimap();
        mm.add_ping(0.0, 0.0, MinimapColor::new(1.0, 1.0, 1.0), 1.0);
        let pings = mm.render_pings();
        assert_eq!(pings.len(), 1);
        mm.update(1.1);
        let pings = mm.render_pings();
        assert!(pings.is_empty());
    }

    #[test]
    fn ping_alpha_decays() {
        let mut mm = basic_minimap();
        mm.add_ping(0.0, 0.0, MinimapColor::new(1.0, 0.0, 0.0), 1.0);
        mm.update(0.5);
        let pings = mm.render_pings();
        assert!((pings[0].alpha - 0.5).abs() < EPS);
    }

    #[test]
    fn click_to_ping_inside() {
        let mut mm = basic_minimap();
        // Click at minimap center (screen 60, 60 → local 50, 50)
        let result = mm.click_to_ping(60.0, 60.0, MinimapColor::new(1.0, 1.0, 1.0), 2.0);
        assert!(result.is_some());
        let (wx, wy) = result.unwrap();
        assert!((wx).abs() < EPS);
        assert!((wy).abs() < EPS);
    }

    #[test]
    fn click_to_ping_outside() {
        let mut mm = basic_minimap().with_shape(MinimapShape::Circular);
        // Click at corner (screen 10, 10 → local 0, 0) which is outside circle
        let result = mm.click_to_ping(10.0, 10.0, MinimapColor::new(1.0, 1.0, 1.0), 2.0);
        assert!(result.is_none());
    }

    #[test]
    fn distance_icon_scaling() {
        let mut mm = basic_minimap();
        mm.icon_scale_distance = 500.0;
        mm.min_icon_scale = 0.3;
        let s = mm.icon_scale_for_distance(250.0);
        // t = 250/500 = 0.5, scale = 1 - 0.5*(1-0.3) = 1-0.35 = 0.65
        assert!((s - 0.65).abs() < EPS);
    }

    #[test]
    fn distance_icon_scaling_zero_distance() {
        let mut mm = basic_minimap();
        mm.icon_scale_distance = 500.0;
        let s = mm.icon_scale_for_distance(0.0);
        assert!((s - 1.0).abs() < EPS);
    }

    #[test]
    fn distance_icon_scaling_disabled() {
        let mm = basic_minimap();
        assert!((mm.icon_scale_for_distance(9999.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn fog_of_war_query() {
        let mut fog = FogOfWar::new(5, 5, 10.0);
        fog.set_cell(2, 3, FogState::Visible);
        assert_eq!(fog.get_cell(2, 3), FogState::Visible);
        assert_eq!(fog.get_cell(0, 0), FogState::Hidden);
    }

    #[test]
    fn fog_of_war_out_of_bounds() {
        let fog = FogOfWar::new(5, 5, 10.0);
        assert_eq!(fog.get_cell(10, 10), FogState::Hidden);
        assert_eq!(fog.fog_at(-100.0, -100.0), FogState::Hidden);
    }

    #[test]
    fn zoom_affects_scale() {
        let mm1 = basic_minimap().with_zoom(10.0);
        let mm2 = basic_minimap().with_zoom(20.0);
        let (mx1, _) = mm1.world_to_minimap(100.0, 0.0);
        let (mx2, _) = mm2.world_to_minimap(100.0, 0.0);
        // Higher zoom → less minimap displacement
        assert!((mx1 - 50.0).abs() > (mx2 - 50.0).abs());
    }

    #[test]
    fn border_style_default() {
        let mm = basic_minimap();
        assert!(mm.border.visible);
        assert!((mm.border.thickness - 2.0).abs() < EPS);
    }
}
