//! Sprite system — sprite sheets, animations, batch rendering, and z-ordering.
//!
//! Pure Rust replacement for PixiJS sprites, Phaser sprite system, and
//! similar 2D game sprite libraries. Fully headless — no GPU calls.

use std::fmt;

// ── Color ────────────────────────────────────────────────────

/// RGBA color for tinting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

// ── Rect ─────────────────────────────────────────────────────

/// Axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }
}

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::zero()
    }
}

// ── Sprite ───────────────────────────────────────────────────

/// A 2D sprite with transform, flip, tint, and z-ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct Sprite {
    pub texture_id: u64,
    pub src_rect: Rect,
    pub position: Vec2,
    pub scale: Vec2,
    pub rotation: f32,
    pub anchor: Vec2,
    pub flip_h: bool,
    pub flip_v: bool,
    pub z_order: i32,
    pub visible: bool,
    pub tint_color: Color,
    pub opacity: f32,
}

impl Sprite {
    pub fn new(texture_id: u64, src_rect: Rect) -> Self {
        Self {
            texture_id,
            src_rect,
            position: Vec2::zero(),
            scale: Vec2::new(1.0, 1.0),
            rotation: 0.0,
            anchor: Vec2::new(0.5, 0.5),
            flip_h: false,
            flip_v: false,
            z_order: 0,
            visible: true,
            tint_color: Color::WHITE,
            opacity: 1.0,
        }
    }

    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = Vec2::new(x, y);
        self
    }

    pub fn with_scale(mut self, sx: f32, sy: f32) -> Self {
        self.scale = Vec2::new(sx, sy);
        self
    }

    pub fn with_rotation(mut self, angle: f32) -> Self {
        self.rotation = angle;
        self
    }

    pub fn with_anchor(mut self, ax: f32, ay: f32) -> Self {
        self.anchor = Vec2::new(ax, ay);
        self
    }

    pub fn with_z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    pub fn with_tint(mut self, color: Color) -> Self {
        self.tint_color = color;
        self
    }

    pub fn with_flip(mut self, h: bool, v: bool) -> Self {
        self.flip_h = h;
        self.flip_v = v;
        self
    }

    /// Get the world-space bounding rect (ignoring rotation).
    pub fn world_rect(&self) -> Rect {
        let w = self.src_rect.width * self.scale.x;
        let h = self.src_rect.height * self.scale.y;
        let x = self.position.x - w * self.anchor.x;
        let y = self.position.y - h * self.anchor.y;
        Rect::new(x, y, w, h)
    }
}

// ── SpriteSheet ──────────────────────────────────────────────

/// A sprite sheet that defines a grid of frames.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteSheet {
    pub texture_id: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub columns: u32,
    pub rows: u32,
    pub padding: u32,
    pub margin: u32,
}

impl SpriteSheet {
    pub fn new(texture_id: u64, frame_width: u32, frame_height: u32, columns: u32, rows: u32) -> Self {
        Self {
            texture_id,
            frame_width,
            frame_height,
            columns,
            rows,
            padding: 0,
            margin: 0,
        }
    }

    pub fn with_padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Total number of frames in the sheet.
    pub fn frame_count(&self) -> u32 {
        self.columns * self.rows
    }

    /// Get the source rectangle for a given frame index.
    pub fn frame_rect(&self, index: u32) -> Option<Rect> {
        if index >= self.frame_count() {
            return None;
        }
        let col = index % self.columns;
        let row = index / self.columns;
        let x = self.margin + col * (self.frame_width + self.padding);
        let y = self.margin + row * (self.frame_height + self.padding);
        Some(Rect::new(x as f32, y as f32, self.frame_width as f32, self.frame_height as f32))
    }

    /// Create a sprite from this sheet at a given frame.
    pub fn create_sprite(&self, frame: u32) -> Option<Sprite> {
        let rect = self.frame_rect(frame)?;
        Some(Sprite::new(self.texture_id, rect))
    }
}

// ── Animation ────────────────────────────────────────────────

/// Loop mode for sprite animations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Play once and stop on last frame.
    Once,
    /// Loop forever.
    Loop,
    /// Play forward then backward, repeating.
    PingPong,
}

/// Defines a sprite animation as a sequence of frame indices.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteAnimation {
    pub name: String,
    pub frames: Vec<u32>,
    pub fps: f32,
    pub loop_mode: LoopMode,
}

impl SpriteAnimation {
    pub fn new(name: impl Into<String>, frames: Vec<u32>, fps: f32) -> Self {
        Self {
            name: name.into(),
            frames,
            fps,
            loop_mode: LoopMode::Loop,
        }
    }

    pub fn with_loop_mode(mut self, mode: LoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Duration of one full animation cycle in seconds.
    pub fn duration(&self) -> f32 {
        if self.fps <= 0.0 || self.frames.is_empty() {
            return 0.0;
        }
        self.frames.len() as f32 / self.fps
    }
}

// ── AnimatedSprite ───────────────────────────────────────────

/// A sprite that plays through animation frames.
#[derive(Debug, Clone)]
pub struct AnimatedSprite {
    pub sprite: Sprite,
    pub sheet: SpriteSheet,
    pub animation: SpriteAnimation,
    elapsed: f32,
    current_frame_index: usize,
    playing: bool,
    forward: bool,
    finished: bool,
}

impl AnimatedSprite {
    pub fn new(sheet: SpriteSheet, animation: SpriteAnimation) -> Self {
        let first_frame = animation.frames.first().copied().unwrap_or(0);
        let rect = sheet.frame_rect(first_frame).unwrap_or(Rect::new(0.0, 0.0, 0.0, 0.0));
        let sprite = Sprite::new(sheet.texture_id, rect);

        Self {
            sprite,
            sheet,
            animation,
            elapsed: 0.0,
            current_frame_index: 0,
            playing: true,
            forward: true,
            finished: false,
        }
    }

    /// Update the animation by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        if !self.playing || self.animation.frames.is_empty() || self.animation.fps <= 0.0 {
            return;
        }

        self.elapsed += dt;
        let frame_duration = 1.0 / self.animation.fps;
        let total_frames = self.animation.frames.len();

        while self.elapsed >= frame_duration {
            self.elapsed -= frame_duration;

            match self.animation.loop_mode {
                LoopMode::Once => {
                    if self.current_frame_index + 1 < total_frames {
                        self.current_frame_index += 1;
                    } else {
                        self.finished = true;
                        self.playing = false;
                    }
                }
                LoopMode::Loop => {
                    self.current_frame_index = (self.current_frame_index + 1) % total_frames;
                }
                LoopMode::PingPong => {
                    if self.forward {
                        if self.current_frame_index + 1 < total_frames {
                            self.current_frame_index += 1;
                        } else {
                            self.forward = false;
                            if self.current_frame_index > 0 {
                                self.current_frame_index -= 1;
                            }
                        }
                    } else {
                        if self.current_frame_index > 0 {
                            self.current_frame_index -= 1;
                        } else {
                            self.forward = true;
                            if self.current_frame_index + 1 < total_frames {
                                self.current_frame_index += 1;
                            }
                        }
                    }
                }
            }
        }

        // Update sprite source rect
        let frame_id = self.animation.frames[self.current_frame_index];
        if let Some(rect) = self.sheet.frame_rect(frame_id) {
            self.sprite.src_rect = rect;
        }
    }

    /// Current frame index within the animation.
    pub fn current_frame(&self) -> usize {
        self.current_frame_index
    }

    /// Whether the animation has finished (only relevant for LoopMode::Once).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Whether the animation is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        self.playing = true;
        self.finished = false;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Reset to the first frame.
    pub fn reset(&mut self) {
        self.current_frame_index = 0;
        self.elapsed = 0.0;
        self.finished = false;
        self.forward = true;
        self.playing = true;
    }
}

// ── Batch Renderer ───────────────────────────────────────────

/// Collects sprites and sorts them by z-order for batch rendering.
#[derive(Debug, Clone)]
pub struct SpriteBatch {
    sprites: Vec<Sprite>,
}

impl SpriteBatch {
    pub fn new() -> Self {
        Self { sprites: Vec::new() }
    }

    pub fn add(&mut self, sprite: Sprite) {
        if sprite.visible {
            self.sprites.push(sprite);
        }
    }

    /// Sort by z_order (ascending) for correct draw order.
    pub fn sort(&mut self) {
        self.sprites.sort_by_key(|s| s.z_order);
    }

    /// Get the sorted sprite draw order.
    pub fn draw_order(&mut self) -> &[Sprite] {
        self.sort();
        &self.sprites
    }

    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    pub fn clear(&mut self) {
        self.sprites.clear();
    }
}

impl Default for SpriteBatch {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sheet() -> SpriteSheet {
        SpriteSheet::new(1, 32, 32, 4, 4)
    }

    #[test]
    fn sprite_creation() {
        let s = Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0))
            .with_position(100.0, 200.0)
            .with_z_order(5);
        assert_eq!(s.position.x, 100.0);
        assert_eq!(s.position.y, 200.0);
        assert_eq!(s.z_order, 5);
        assert!(s.visible);
    }

    #[test]
    fn sprite_world_rect() {
        let s = Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0))
            .with_position(100.0, 100.0)
            .with_anchor(0.5, 0.5);
        let wr = s.world_rect();
        assert_eq!(wr.x, 84.0);
        assert_eq!(wr.y, 84.0);
        assert_eq!(wr.width, 32.0);
        assert_eq!(wr.height, 32.0);
    }

    #[test]
    fn sprite_scaled_world_rect() {
        let s = Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0))
            .with_position(0.0, 0.0)
            .with_scale(2.0, 2.0)
            .with_anchor(0.0, 0.0);
        let wr = s.world_rect();
        assert_eq!(wr.width, 64.0);
        assert_eq!(wr.height, 64.0);
    }

    #[test]
    fn sprite_flip() {
        let s = Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0))
            .with_flip(true, false);
        assert!(s.flip_h);
        assert!(!s.flip_v);
    }

    #[test]
    fn sprite_sheet_frame_count() {
        let sheet = test_sheet();
        assert_eq!(sheet.frame_count(), 16);
    }

    #[test]
    fn sprite_sheet_frame_rect() {
        let sheet = test_sheet();
        let r = sheet.frame_rect(0).unwrap();
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 32.0);

        let r5 = sheet.frame_rect(5).unwrap();
        // col=1, row=1
        assert_eq!(r5.x, 32.0);
        assert_eq!(r5.y, 32.0);
    }

    #[test]
    fn sprite_sheet_frame_rect_out_of_bounds() {
        let sheet = test_sheet();
        assert!(sheet.frame_rect(16).is_none());
    }

    #[test]
    fn sprite_sheet_with_padding() {
        let sheet = SpriteSheet::new(1, 32, 32, 4, 4).with_padding(2).with_margin(1);
        let r = sheet.frame_rect(1).unwrap();
        // x = margin + col * (width + padding) = 1 + 1 * 34 = 35
        assert_eq!(r.x, 35.0);
        assert_eq!(r.y, 1.0);
    }

    #[test]
    fn animation_duration() {
        let anim = SpriteAnimation::new("walk", vec![0, 1, 2, 3], 10.0);
        assert!((anim.duration() - 0.4).abs() < 0.001);
    }

    #[test]
    fn animated_sprite_tick_loop() {
        let sheet = test_sheet();
        let anim = SpriteAnimation::new("walk", vec![0, 1, 2, 3], 4.0)
            .with_loop_mode(LoopMode::Loop);
        let mut animated = AnimatedSprite::new(sheet, anim);

        assert_eq!(animated.current_frame(), 0);
        animated.tick(0.26); // 0.26s at 4fps = 1+ frame
        assert_eq!(animated.current_frame(), 1);
        animated.tick(0.25);
        assert_eq!(animated.current_frame(), 2);
        animated.tick(0.25);
        assert_eq!(animated.current_frame(), 3);
        animated.tick(0.25); // Loops back
        assert_eq!(animated.current_frame(), 0);
    }

    #[test]
    fn animated_sprite_tick_once() {
        let sheet = test_sheet();
        let anim = SpriteAnimation::new("die", vec![0, 1, 2], 4.0)
            .with_loop_mode(LoopMode::Once);
        let mut animated = AnimatedSprite::new(sheet, anim);

        animated.tick(0.26);
        assert_eq!(animated.current_frame(), 1);
        animated.tick(0.25);
        assert_eq!(animated.current_frame(), 2);
        animated.tick(0.25); // Should stay at frame 2
        assert_eq!(animated.current_frame(), 2);
        assert!(animated.is_finished());
        assert!(!animated.is_playing());
    }

    #[test]
    fn animated_sprite_pause_resume() {
        let sheet = test_sheet();
        let anim = SpriteAnimation::new("idle", vec![0, 1], 2.0);
        let mut animated = AnimatedSprite::new(sheet, anim);

        animated.tick(0.6);
        let frame_before = animated.current_frame();
        animated.pause();
        animated.tick(1.0); // Should not advance
        assert_eq!(animated.current_frame(), frame_before);
        animated.play();
        animated.tick(0.6);
        assert_ne!(animated.current_frame(), frame_before);
    }

    #[test]
    fn animated_sprite_reset() {
        let sheet = test_sheet();
        let anim = SpriteAnimation::new("run", vec![0, 1, 2, 3], 4.0);
        let mut animated = AnimatedSprite::new(sheet, anim);

        animated.tick(0.5);
        assert_ne!(animated.current_frame(), 0);
        animated.reset();
        assert_eq!(animated.current_frame(), 0);
        assert!(animated.is_playing());
    }

    #[test]
    fn sprite_batch_z_order() {
        let mut batch = SpriteBatch::new();
        batch.add(Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0)).with_z_order(3));
        batch.add(Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0)).with_z_order(1));
        batch.add(Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0)).with_z_order(2));

        let order = batch.draw_order();
        assert_eq!(order[0].z_order, 1);
        assert_eq!(order[1].z_order, 2);
        assert_eq!(order[2].z_order, 3);
    }

    #[test]
    fn sprite_batch_hidden_not_added() {
        let mut batch = SpriteBatch::new();
        let mut s = Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0));
        s.visible = false;
        batch.add(s);
        assert!(batch.is_empty());
    }

    #[test]
    fn sprite_tint() {
        let s = Sprite::new(1, Rect::new(0.0, 0.0, 32.0, 32.0))
            .with_tint(Color::RED);
        assert_eq!(s.tint_color, Color::RED);
    }

    #[test]
    fn create_sprite_from_sheet() {
        let sheet = test_sheet();
        let sprite = sheet.create_sprite(5).unwrap();
        assert_eq!(sprite.texture_id, 1);
        assert_eq!(sprite.src_rect.x, 32.0);
        assert_eq!(sprite.src_rect.y, 32.0);
    }
}
