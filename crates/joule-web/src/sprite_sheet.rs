// sprite_sheet.rs — Sprite sheet: frame rects, named animations with
// frame sequences, frame timing/duration, atlas packing (bin-packing),
// UV coordinate calculation, animation playback state with looping.

/// A rectangle within an atlas (pixel coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl FrameRect {
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    pub fn area(&self) -> u32 {
        self.w * self.h
    }

    pub fn right(&self) -> u32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }
}

/// UV coordinates (0.0..1.0 range) for a frame within an atlas.
#[derive(Debug, Clone, Copy)]
pub struct UvRect {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

impl UvRect {
    /// Calculate UV rect for a frame given the atlas dimensions.
    pub fn from_frame(frame: &FrameRect, atlas_w: u32, atlas_h: u32) -> Self {
        Self {
            u_min: frame.x as f32 / atlas_w as f32,
            v_min: frame.y as f32 / atlas_h as f32,
            u_max: (frame.x + frame.w) as f32 / atlas_w as f32,
            v_max: (frame.y + frame.h) as f32 / atlas_h as f32,
        }
    }

    pub fn width(&self) -> f32 {
        self.u_max - self.u_min
    }

    pub fn height(&self) -> f32 {
        self.v_max - self.v_min
    }
}

/// A single frame in an animation (index into the sprite sheet's frame list + duration).
#[derive(Debug, Clone, Copy)]
pub struct AnimFrame {
    pub frame_index: usize,
    /// Duration of this frame in milliseconds.
    pub duration_ms: u32,
}

/// A named animation sequence.
#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub frames: Vec<AnimFrame>,
    pub looping: bool,
}

impl Animation {
    pub fn new(name: &str, looping: bool) -> Self {
        Self {
            name: name.to_string(),
            frames: Vec::new(),
            looping,
        }
    }

    pub fn add_frame(&mut self, frame_index: usize, duration_ms: u32) {
        self.frames.push(AnimFrame {
            frame_index,
            duration_ms,
        });
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Total duration of one loop of the animation.
    pub fn total_duration_ms(&self) -> u32 {
        self.frames.iter().map(|f| f.duration_ms).sum()
    }

    /// Get the frame index for a given elapsed time (in ms).
    /// Returns `None` if non-looping and past the end.
    pub fn frame_at_time(&self, elapsed_ms: u32) -> Option<usize> {
        if self.frames.is_empty() {
            return None;
        }
        let total = self.total_duration_ms();
        if total == 0 {
            return Some(self.frames[0].frame_index);
        }

        let effective = if self.looping {
            elapsed_ms % total
        } else if elapsed_ms >= total {
            return self.frames.last().map(|f| f.frame_index);
        } else {
            elapsed_ms
        };

        let mut accumulated = 0u32;
        for f in &self.frames {
            accumulated += f.duration_ms;
            if effective < accumulated {
                return Some(f.frame_index);
            }
        }
        self.frames.last().map(|f| f.frame_index)
    }
}

/// Sprite sheet: a collection of frame rectangles and named animations.
#[derive(Debug, Clone)]
pub struct SpriteSheet {
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub frames: Vec<FrameRect>,
    pub animations: Vec<Animation>,
}

impl SpriteSheet {
    pub fn new(atlas_width: u32, atlas_height: u32) -> Self {
        Self {
            atlas_width,
            atlas_height,
            frames: Vec::new(),
            animations: Vec::new(),
        }
    }

    pub fn add_frame(&mut self, rect: FrameRect) -> usize {
        let idx = self.frames.len();
        self.frames.push(rect);
        idx
    }

    pub fn add_animation(&mut self, anim: Animation) {
        self.animations.push(anim);
    }

    pub fn find_animation(&self, name: &str) -> Option<&Animation> {
        self.animations.iter().find(|a| a.name == name)
    }

    pub fn frame_uv(&self, frame_index: usize) -> Option<UvRect> {
        self.frames
            .get(frame_index)
            .map(|f| UvRect::from_frame(f, self.atlas_width, self.atlas_height))
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn animation_count(&self) -> usize {
        self.animations.len()
    }
}

// ---------------------------------------------------------------------------
// Animation playback state
// ---------------------------------------------------------------------------

/// Playback state for an animation.
#[derive(Debug, Clone)]
pub struct PlaybackState {
    pub animation_name: String,
    pub elapsed_ms: u32,
    pub playing: bool,
    pub speed: f32,
}

impl PlaybackState {
    pub fn new(animation_name: &str) -> Self {
        Self {
            animation_name: animation_name.to_string(),
            elapsed_ms: 0,
            playing: true,
            speed: 1.0,
        }
    }

    pub fn advance(&mut self, delta_ms: u32) {
        if self.playing {
            let scaled = (delta_ms as f32 * self.speed) as u32;
            self.elapsed_ms += scaled;
        }
    }

    pub fn reset(&mut self) {
        self.elapsed_ms = 0;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn resume(&mut self) {
        self.playing = true;
    }

    /// Resolve the current frame rect from a sprite sheet.
    pub fn current_frame<'a>(&self, sheet: &'a SpriteSheet) -> Option<&'a FrameRect> {
        let anim = sheet.find_animation(&self.animation_name)?;
        let frame_idx = anim.frame_at_time(self.elapsed_ms)?;
        sheet.frames.get(frame_idx)
    }

    pub fn current_uv(&self, sheet: &SpriteSheet) -> Option<UvRect> {
        let anim = sheet.find_animation(&self.animation_name)?;
        let frame_idx = anim.frame_at_time(self.elapsed_ms)?;
        sheet.frame_uv(frame_idx)
    }

    pub fn is_finished(&self, sheet: &SpriteSheet) -> bool {
        if let Some(anim) = sheet.find_animation(&self.animation_name) {
            if anim.looping {
                false
            } else {
                self.elapsed_ms >= anim.total_duration_ms()
            }
        } else {
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Atlas packing (simple shelf / bin-packing)
// ---------------------------------------------------------------------------

/// A packed sprite: original index + packed position.
#[derive(Debug, Clone)]
pub struct PackedSprite {
    pub original_index: usize,
    pub rect: FrameRect,
}

/// Result of atlas packing.
#[derive(Debug, Clone)]
pub struct PackResult {
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub sprites: Vec<PackedSprite>,
}

impl PackResult {
    pub fn utilization(&self) -> f32 {
        let used: u32 = self.sprites.iter().map(|s| s.rect.area()).sum();
        let total = self.atlas_width * self.atlas_height;
        if total == 0 {
            return 0.0;
        }
        used as f32 / total as f32
    }
}

/// Simple shelf-based bin packing. Sorts sprites by height descending,
/// then packs left-to-right in rows.
pub fn pack_atlas(sprites: &[(u32, u32)], max_width: u32) -> PackResult {
    if sprites.is_empty() {
        return PackResult {
            atlas_width: 0,
            atlas_height: 0,
            sprites: Vec::new(),
        };
    }

    // Sort by height descending (keep original indices).
    let mut indexed: Vec<(usize, u32, u32)> = sprites
        .iter()
        .enumerate()
        .map(|(i, &(w, h))| (i, w, h))
        .collect();
    indexed.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.1.cmp(&a.1)));

    let mut packed = Vec::new();
    let mut shelf_x: u32 = 0;
    let mut shelf_y: u32 = 0;
    let mut shelf_h: u32 = 0;
    let mut atlas_w: u32 = 0;
    let mut atlas_h: u32 = 0;

    for (orig_idx, w, h) in indexed {
        if shelf_x + w > max_width {
            // New shelf.
            shelf_y += shelf_h;
            shelf_x = 0;
            shelf_h = 0;
        }

        packed.push(PackedSprite {
            original_index: orig_idx,
            rect: FrameRect::new(shelf_x, shelf_y, w, h),
        });

        shelf_x += w;
        if h > shelf_h {
            shelf_h = h;
        }
        if shelf_x > atlas_w {
            atlas_w = shelf_x;
        }
        let bottom = shelf_y + shelf_h;
        if bottom > atlas_h {
            atlas_h = bottom;
        }
    }

    // Sort by original index for deterministic output.
    packed.sort_by_key(|p| p.original_index);

    PackResult {
        atlas_width: atlas_w,
        atlas_height: atlas_h,
        sprites: packed,
    }
}

// ---------------------------------------------------------------------------
// Grid-based sprite sheet helper
// ---------------------------------------------------------------------------

/// Create a uniform-grid sprite sheet (all frames same size).
pub fn grid_sheet(
    atlas_w: u32,
    atlas_h: u32,
    frame_w: u32,
    frame_h: u32,
) -> SpriteSheet {
    let mut sheet = SpriteSheet::new(atlas_w, atlas_h);
    let cols = atlas_w / frame_w;
    let rows = atlas_h / frame_h;
    for row in 0..rows {
        for col in 0..cols {
            sheet.add_frame(FrameRect::new(
                col * frame_w,
                row * frame_h,
                frame_w,
                frame_h,
            ));
        }
    }
    sheet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rect() {
        let f = FrameRect::new(10, 20, 32, 32);
        assert_eq!(f.area(), 1024);
        assert_eq!(f.right(), 42);
        assert_eq!(f.bottom(), 52);
    }

    #[test]
    fn test_uv_rect() {
        let frame = FrameRect::new(0, 0, 64, 64);
        let uv = UvRect::from_frame(&frame, 256, 256);
        assert!((uv.u_min - 0.0).abs() < f32::EPSILON);
        assert!((uv.v_min - 0.0).abs() < f32::EPSILON);
        assert!((uv.u_max - 0.25).abs() < f32::EPSILON);
        assert!((uv.v_max - 0.25).abs() < f32::EPSILON);
        assert!((uv.width() - 0.25).abs() < f32::EPSILON);
        assert!((uv.height() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_animation_basic() {
        let mut anim = Animation::new("walk", true);
        anim.add_frame(0, 100); anim.add_frame(1, 100); anim.add_frame(2, 100);
        assert_eq!(anim.frame_count(), 3);
        assert_eq!(anim.total_duration_ms(), 300);
    }

    #[test]
    fn test_animation_frame_at_time() {
        let mut anim = Animation::new("run", false);
        anim.add_frame(0, 100);
        anim.add_frame(1, 100);
        anim.add_frame(2, 100);

        assert_eq!(anim.frame_at_time(0), Some(0));
        assert_eq!(anim.frame_at_time(50), Some(0));
        assert_eq!(anim.frame_at_time(100), Some(1));
        assert_eq!(anim.frame_at_time(250), Some(2));
        // Past end, non-looping: returns last.
        assert_eq!(anim.frame_at_time(500), Some(2));
    }

    #[test]
    fn test_animation_looping() {
        let mut anim = Animation::new("idle", true);
        anim.add_frame(0, 100);
        anim.add_frame(1, 100);
        // total = 200ms

        assert_eq!(anim.frame_at_time(0), Some(0));
        assert_eq!(anim.frame_at_time(150), Some(1));
        // At 200ms, loops back: 200 % 200 = 0 -> frame 0
        assert_eq!(anim.frame_at_time(200), Some(0));
        // 250 % 200 = 50 -> frame 0
        assert_eq!(anim.frame_at_time(250), Some(0));
    }

    #[test]
    fn test_animation_empty() {
        let anim = Animation::new("empty", true);
        assert_eq!(anim.frame_at_time(0), None);
        assert_eq!(anim.total_duration_ms(), 0);
    }

    #[test]
    fn test_sprite_sheet() {
        let mut sheet = SpriteSheet::new(256, 256);
        let idx0 = sheet.add_frame(FrameRect::new(0, 0, 64, 64));
        let idx1 = sheet.add_frame(FrameRect::new(64, 0, 64, 64));
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(sheet.frame_count(), 2);

        let uv = sheet.frame_uv(0).unwrap();
        assert!((uv.u_max - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sprite_sheet_animation() {
        let mut sheet = SpriteSheet::new(256, 256);
        sheet.add_frame(FrameRect::new(0, 0, 64, 64));
        sheet.add_frame(FrameRect::new(64, 0, 64, 64));
        let mut anim = Animation::new("walk", true);
        anim.add_frame(0, 100);
        anim.add_frame(1, 100);
        sheet.add_animation(anim);

        assert_eq!(sheet.animation_count(), 1);
        assert!(sheet.find_animation("walk").is_some());
        assert!(sheet.find_animation("run").is_none());
    }

    #[test]
    fn test_playback_state() {
        let mut sheet = SpriteSheet::new(256, 256);
        sheet.add_frame(FrameRect::new(0, 0, 64, 64));
        sheet.add_frame(FrameRect::new(64, 0, 64, 64));
        let mut anim = Animation::new("walk", true);
        anim.add_frame(0, 100); anim.add_frame(1, 100);
        sheet.add_animation(anim);
        let mut state = PlaybackState::new("walk");
        assert!(state.playing);
        assert_eq!(state.current_frame(&sheet).unwrap().x, 0);
        state.advance(150);
        assert_eq!(state.current_frame(&sheet).unwrap().x, 64);
    }

    #[test]
    fn test_playback_controls() {
        let mut state = PlaybackState::new("a");
        state.pause(); state.advance(200);
        assert_eq!(state.elapsed_ms, 0);
        state.resume(); state.advance(50);
        assert_eq!(state.elapsed_ms, 50);
        state.speed = 2.0; state.advance(100);
        assert_eq!(state.elapsed_ms, 250);
        state.reset();
        assert_eq!(state.elapsed_ms, 0);
    }

    #[test]
    fn test_playback_finished_and_looping() {
        let mut sheet = SpriteSheet::new(128, 128);
        sheet.add_frame(FrameRect::new(0, 0, 32, 32));
        let mut once = Animation::new("once", false);
        once.add_frame(0, 100);
        sheet.add_animation(once);
        let mut lp = Animation::new("lp", true);
        lp.add_frame(0, 100);
        sheet.add_animation(lp);
        let mut s1 = PlaybackState::new("once");
        assert!(!s1.is_finished(&sheet));
        s1.advance(100);
        assert!(s1.is_finished(&sheet));
        let mut s2 = PlaybackState::new("lp");
        s2.advance(99999);
        assert!(!s2.is_finished(&sheet));
        assert!(s2.current_uv(&sheet).is_some());
    }

    // ---- Atlas packing ----

    #[test]
    fn test_pack_empty() {
        let result = pack_atlas(&[], 256);
        assert_eq!(result.atlas_width, 0);
        assert_eq!(result.atlas_height, 0);
        assert!(result.sprites.is_empty());
    }

    #[test]
    fn test_pack_single() {
        let result = pack_atlas(&[(32, 32)], 256);
        assert_eq!(result.sprites.len(), 1);
        assert_eq!(result.atlas_width, 32);
        assert_eq!(result.atlas_height, 32);
        assert_eq!(result.sprites[0].rect.x, 0);
        assert_eq!(result.sprites[0].rect.y, 0);
    }

    #[test]
    fn test_pack_multiple_fit_one_row() {
        let result = pack_atlas(&[(32, 32), (32, 32), (32, 32)], 256);
        assert_eq!(result.sprites.len(), 3);
        assert_eq!(result.atlas_height, 32);
        assert!(result.atlas_width <= 96);
    }

    #[test]
    fn test_pack_wraps_to_new_shelf() {
        let result = pack_atlas(&[(64, 64), (64, 64), (64, 64)], 128);
        assert_eq!(result.sprites.len(), 3);
        // 2 fit on first row, 1 on second.
        assert!(result.atlas_height >= 128);
    }

    #[test]
    fn test_pack_utilization() {
        let sprites: Vec<(u32, u32)> = vec![(32, 32); 4];
        let result = pack_atlas(&sprites, 128);
        let util = result.utilization();
        assert!(util > 0.0 && util <= 1.0, "utilization={util}");
    }

    #[test]
    fn test_pack_preserves_original_index() {
        let result = pack_atlas(&[(16, 32), (32, 16), (16, 16)], 256);
        assert_eq!(result.sprites.len(), 3);
        // Sorted by original index.
        for (i, s) in result.sprites.iter().enumerate() {
            assert_eq!(s.original_index, i);
        }
    }

    // ---- Grid sheet ----

    #[test]
    fn test_grid_sheet() {
        let sheet = grid_sheet(128, 128, 32, 32);
        assert_eq!(sheet.frame_count(), 16); // 4x4
        assert_eq!(sheet.frames[0], FrameRect::new(0, 0, 32, 32));
        assert_eq!(sheet.frames[1], FrameRect::new(32, 0, 32, 32));
        assert_eq!(sheet.frames[4], FrameRect::new(0, 32, 32, 32));
    }

    #[test]
    fn test_grid_sheet_non_square() {
        let sheet = grid_sheet(256, 128, 64, 64);
        assert_eq!(sheet.frame_count(), 8); // 4 cols x 2 rows
    }
}
