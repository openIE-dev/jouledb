//! Carousel / Slider: auto-play, looping, multi-slide view.
//!
//! Pure state machine for a carousel widget: next/prev navigation,
//! timed auto-advance via `tick()`, pause/resume, and progress tracking.

// ── Types ───────────────────────────────────────────────────────

/// A single slide in the carousel.
#[derive(Debug, Clone)]
pub struct CarouselItem {
    pub id: String,
    pub content: String,
}

impl CarouselItem {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { id: id.into(), content: content.into() }
    }
}

/// Configuration for carousel behaviour.
#[derive(Debug, Clone)]
pub struct CarouselConfig {
    pub auto_play: bool,
    pub interval_ms: u64,
    pub loop_: bool,
    pub slides_per_view: usize,
    pub gap: f64,
}

impl Default for CarouselConfig {
    fn default() -> Self {
        Self {
            auto_play: false,
            interval_ms: 3000,
            loop_: true,
            slides_per_view: 1,
            gap: 0.0,
        }
    }
}

/// Carousel state machine.
#[derive(Debug, Clone)]
pub struct Carousel {
    items: Vec<CarouselItem>,
    current_index: usize,
    config: CarouselConfig,
    paused: bool,
    elapsed_ms: u64,
}

// ── Implementation ──────────────────────────────────────────────

impl Carousel {
    pub fn new(config: CarouselConfig) -> Self {
        Self {
            items: Vec::new(),
            current_index: 0,
            config,
            paused: false,
            elapsed_ms: 0,
        }
    }

    pub fn add_item(&mut self, item: CarouselItem) {
        self.items.push(item);
    }

    /// Advance to next slide. Returns false if at end and not looping.
    pub fn next(&mut self) -> bool {
        if self.items.is_empty() { return false; }
        if self.current_index + 1 < self.items.len() {
            self.current_index += 1;
            self.elapsed_ms = 0;
            true
        } else if self.config.loop_ {
            self.current_index = 0;
            self.elapsed_ms = 0;
            true
        } else {
            false
        }
    }

    /// Go to previous slide. Returns false if at start and not looping.
    pub fn previous(&mut self) -> bool {
        if self.items.is_empty() { return false; }
        if self.current_index > 0 {
            self.current_index -= 1;
            self.elapsed_ms = 0;
            true
        } else if self.config.loop_ {
            self.current_index = self.items.len() - 1;
            self.elapsed_ms = 0;
            true
        } else {
            false
        }
    }

    pub fn go_to(&mut self, index: usize) {
        if index < self.items.len() {
            self.current_index = index;
            self.elapsed_ms = 0;
        }
    }

    pub fn current(&self) -> Option<&CarouselItem> {
        self.items.get(self.current_index)
    }

    pub fn current_index(&self) -> usize { self.current_index }
    pub fn total_slides(&self) -> usize { self.items.len() }

    /// Advance time. If auto_play and not paused, auto-advance when interval reached.
    /// Returns true if the slide advanced.
    pub fn tick(&mut self, dt_ms: u64) -> bool {
        if !self.config.auto_play || self.paused || self.items.is_empty() {
            return false;
        }
        self.elapsed_ms += dt_ms;
        if self.elapsed_ms >= self.config.interval_ms {
            self.elapsed_ms = 0;
            self.next()
        } else {
            false
        }
    }

    pub fn pause(&mut self) { self.paused = true; }
    pub fn resume(&mut self) { self.paused = false; }
    pub fn is_paused(&self) -> bool { self.paused }

    /// Indices of currently visible slides based on slides_per_view.
    pub fn visible_indices(&self) -> Vec<usize> {
        if self.items.is_empty() { return Vec::new(); }
        let end = (self.current_index + self.config.slides_per_view).min(self.items.len());
        (self.current_index..end).collect()
    }

    /// Number of pagination dots (total slides accounting for slides_per_view).
    pub fn dot_count(&self) -> usize {
        if self.items.is_empty() || self.config.slides_per_view == 0 { return 0; }
        let n = self.items.len();
        let spv = self.config.slides_per_view;
        if n <= spv { 1 } else { n - spv + 1 }
    }

    /// Progress through the carousel: 0.0 at start, 1.0 at end.
    pub fn progress(&self) -> f64 {
        if self.items.len() <= 1 { return 0.0; }
        self.current_index as f64 / (self.items.len() - 1) as f64
    }

    pub fn items(&self) -> &[CarouselItem] { &self.items }
    pub fn config(&self) -> &CarouselConfig { &self.config }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn carousel3(loop_: bool, auto_play: bool) -> Carousel {
        let config = CarouselConfig {
            auto_play,
            interval_ms: 1000,
            loop_,
            slides_per_view: 1,
            gap: 0.0,
        };
        let mut c = Carousel::new(config);
        c.add_item(CarouselItem::new("s1", "Slide 1"));
        c.add_item(CarouselItem::new("s2", "Slide 2"));
        c.add_item(CarouselItem::new("s3", "Slide 3"));
        c
    }

    #[test]
    fn next_previous() {
        let mut c = carousel3(false, false);
        assert_eq!(c.current_index(), 0);
        assert!(c.next());
        assert_eq!(c.current_index(), 1);
        assert!(c.previous());
        assert_eq!(c.current_index(), 0);
    }

    #[test]
    fn loop_wraps() {
        let mut c = carousel3(true, false);
        c.go_to(2);
        assert!(c.next());
        assert_eq!(c.current_index(), 0);
        assert!(c.previous());
        assert_eq!(c.current_index(), 2);
    }

    #[test]
    fn no_loop_stops() {
        let mut c = carousel3(false, false);
        c.go_to(2);
        assert!(!c.next());
        assert_eq!(c.current_index(), 2);
        c.go_to(0);
        assert!(!c.previous());
        assert_eq!(c.current_index(), 0);
    }

    #[test]
    fn auto_play_tick_advances() {
        let mut c = carousel3(true, true);
        assert!(!c.tick(500));
        assert!(c.tick(500));
        assert_eq!(c.current_index(), 1);
    }

    #[test]
    fn pause_stops_tick() {
        let mut c = carousel3(true, true);
        c.pause();
        assert!(c.is_paused());
        assert!(!c.tick(2000));
        assert_eq!(c.current_index(), 0);
        c.resume();
        assert!(c.tick(1000));
        assert_eq!(c.current_index(), 1);
    }

    #[test]
    fn visible_indices() {
        let config = CarouselConfig {
            slides_per_view: 2,
            ..Default::default()
        };
        let mut c = Carousel::new(config);
        c.add_item(CarouselItem::new("a", "A"));
        c.add_item(CarouselItem::new("b", "B"));
        c.add_item(CarouselItem::new("c", "C"));
        assert_eq!(c.visible_indices(), vec![0, 1]);
        c.next();
        assert_eq!(c.visible_indices(), vec![1, 2]);
    }

    #[test]
    fn go_to() {
        let mut c = carousel3(false, false);
        c.go_to(2);
        assert_eq!(c.current().unwrap().id, "s3");
        c.go_to(99);
        assert_eq!(c.current_index(), 2);
    }

    #[test]
    fn progress() {
        let mut c = carousel3(false, false);
        assert_eq!(c.progress(), 0.0);
        c.go_to(1);
        assert_eq!(c.progress(), 0.5);
        c.go_to(2);
        assert_eq!(c.progress(), 1.0);
    }

    #[test]
    fn empty_carousel() {
        let mut c = Carousel::new(CarouselConfig::default());
        assert!(c.current().is_none());
        assert!(!c.next());
        assert!(!c.previous());
        assert_eq!(c.visible_indices().len(), 0);
        assert_eq!(c.dot_count(), 0);
        assert_eq!(c.progress(), 0.0);
    }

    #[test]
    fn dot_count() {
        let mut c = carousel3(false, false);
        assert_eq!(c.dot_count(), 3);
        c.config.slides_per_view = 2;
        assert_eq!(c.dot_count(), 2);
    }

    #[test]
    fn slides_per_view_clamps() {
        let config = CarouselConfig {
            slides_per_view: 5,
            ..Default::default()
        };
        let mut c = Carousel::new(config);
        c.add_item(CarouselItem::new("a", "A"));
        c.add_item(CarouselItem::new("b", "B"));
        assert_eq!(c.visible_indices(), vec![0, 1]);
    }

    #[test]
    fn tick_no_autoplay() {
        let mut c = carousel3(true, false);
        assert!(!c.tick(5000));
        assert_eq!(c.current_index(), 0);
    }
}
