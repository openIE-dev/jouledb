//! GPIO simulation: pin model (input/output/alternate), digital read/write,
//! pull-up/pull-down, interrupt simulation (edge trigger), pin groups,
//! debounce, PWM output simulation, and pin state history.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

// ── Types ──

/// GPIO pin direction/mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinMode {
    Input,
    Output,
    Alternate,
    Analog,
    Disabled,
}

impl PinMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Alternate => "alternate",
            Self::Analog => "analog",
            Self::Disabled => "disabled",
        }
    }
}

/// Digital logic level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Low,
    High,
}

impl Level {
    pub fn as_bool(&self) -> bool {
        matches!(self, Self::High)
    }

    pub fn from_bool(val: bool) -> Self {
        if val { Self::High } else { Self::Low }
    }

    pub fn invert(&self) -> Self {
        match self {
            Self::Low => Self::High,
            Self::High => Self::Low,
        }
    }
}

/// Pull resistor configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pull {
    None,
    Up,
    Down,
}

/// Edge trigger type for interrupts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeTrigger {
    Rising,
    Falling,
    Both,
}

impl EdgeTrigger {
    pub fn matches(&self, old: Level, new: Level) -> bool {
        match self {
            Self::Rising => old == Level::Low && new == Level::High,
            Self::Falling => old == Level::High && new == Level::Low,
            Self::Both => old != new,
        }
    }
}

/// A recorded interrupt event.
#[derive(Debug, Clone)]
pub struct InterruptEvent {
    pub pin: u8,
    pub trigger: EdgeTrigger,
    pub level: Level,
    pub timestamp: DateTime<Utc>,
}

/// A recorded pin state change.
#[derive(Debug, Clone)]
pub struct PinStateChange {
    pub pin: u8,
    pub old_level: Level,
    pub new_level: Level,
    pub timestamp: DateTime<Utc>,
}

/// PWM configuration for a pin.
#[derive(Debug, Clone)]
pub struct PwmConfig {
    pub frequency_hz: u32,
    pub duty_cycle: f64,
    pub enabled: bool,
}

impl PwmConfig {
    pub fn new(frequency_hz: u32, duty_cycle: f64) -> Self {
        Self {
            frequency_hz,
            duty_cycle: duty_cycle.clamp(0.0, 1.0),
            enabled: true,
        }
    }

    /// Period in microseconds.
    pub fn period_us(&self) -> f64 {
        if self.frequency_hz == 0 {
            return 0.0;
        }
        1_000_000.0 / self.frequency_hz as f64
    }

    /// High time in microseconds.
    pub fn high_time_us(&self) -> f64 {
        self.period_us() * self.duty_cycle
    }

    /// Low time in microseconds.
    pub fn low_time_us(&self) -> f64 {
        self.period_us() * (1.0 - self.duty_cycle)
    }
}

/// Internal representation of a single GPIO pin.
#[derive(Debug, Clone)]
struct PinState {
    mode: PinMode,
    level: Level,
    pull: Pull,
    interrupt: Option<EdgeTrigger>,
    pwm: Option<PwmConfig>,
    debounce_ms: u32,
    last_change: Option<DateTime<Utc>>,
    alternate_function: Option<String>,
}

impl PinState {
    fn new() -> Self {
        Self {
            mode: PinMode::Disabled,
            level: Level::Low,
            pull: Pull::None,
            interrupt: None,
            pwm: None,
            debounce_ms: 0,
            last_change: None,
            alternate_function: None,
        }
    }

    fn effective_level(&self) -> Level {
        if self.mode == PinMode::Input && self.pull != Pull::None {
            // In input mode with a pull, if not externally driven, report pull level.
            // For simulation, the explicit level overrides, but pull defines default.
            return self.level;
        }
        self.level
    }
}

// ── Pin Group ──

/// A named group of GPIO pins for bulk operations.
#[derive(Debug, Clone)]
pub struct PinGroup {
    pub name: String,
    pub pins: Vec<u8>,
}

impl PinGroup {
    pub fn new(name: &str, pins: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            pins,
        }
    }
}

// ── GPIO Controller ──

/// GPIO controller simulator managing multiple pins.
pub struct GpioController {
    pins: HashMap<u8, PinState>,
    groups: HashMap<String, PinGroup>,
    history: Vec<PinStateChange>,
    interrupts: Vec<InterruptEvent>,
    max_history: usize,
}

impl GpioController {
    pub fn new(pin_count: u8) -> Self {
        let mut pins = HashMap::new();
        for i in 0..pin_count {
            pins.insert(i, PinState::new());
        }
        Self {
            pins,
            groups: HashMap::new(),
            history: Vec::new(),
            interrupts: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
    }

    /// Configure a pin's mode.
    pub fn set_mode(&mut self, pin: u8, mode: PinMode) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        state.mode = mode;
        if mode == PinMode::Disabled {
            state.pwm = None;
            state.interrupt = None;
        }
        Ok(())
    }

    /// Set the pull resistor for an input pin.
    pub fn set_pull(&mut self, pin: u8, pull: Pull) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        if state.mode != PinMode::Input {
            return Err(GpioError::WrongMode { pin, expected: PinMode::Input, actual: state.mode });
        }
        state.pull = pull;
        // If pull is set and level hasn't been explicitly driven, apply pull level.
        match pull {
            Pull::Up => state.level = Level::High,
            Pull::Down => state.level = Level::Low,
            Pull::None => {}
        }
        Ok(())
    }

    /// Write a digital level to an output pin.
    pub fn write(&mut self, pin: u8, level: Level) -> Result<(), GpioError> {
        // Check mode first without holding mutable borrow.
        let mode = self.pins.get(&pin).ok_or(GpioError::InvalidPin(pin))?.mode;
        if mode != PinMode::Output {
            return Err(GpioError::WrongMode { pin, expected: PinMode::Output, actual: mode });
        }

        let old_level = self.pins[&pin].level;
        let now = Utc::now();

        // Check debounce.
        let debounce_ms = self.pins[&pin].debounce_ms;
        if debounce_ms > 0 {
            if let Some(last_change) = self.pins[&pin].last_change {
                let elapsed = now.signed_duration_since(last_change).num_milliseconds();
                if elapsed < debounce_ms as i64 {
                    return Ok(()); // Debounced — ignore.
                }
            }
        }

        let state = self.pins.get_mut(&pin).unwrap();
        state.level = level;
        state.last_change = Some(now);

        if old_level != level {
            self.record_change(pin, old_level, level, now);
        }

        Ok(())
    }

    /// Read the digital level of a pin.
    pub fn read(&self, pin: u8) -> Result<Level, GpioError> {
        let state = self.pins.get(&pin).ok_or(GpioError::InvalidPin(pin))?;
        Ok(state.effective_level())
    }

    /// Simulate an external signal driving an input pin.
    pub fn drive_input(&mut self, pin: u8, level: Level) -> Result<(), GpioError> {
        let mode = self.pins.get(&pin).ok_or(GpioError::InvalidPin(pin))?.mode;
        if mode != PinMode::Input {
            return Err(GpioError::WrongMode { pin, expected: PinMode::Input, actual: mode });
        }

        let old_level = self.pins[&pin].level;
        let now = Utc::now();

        // Debounce.
        let debounce_ms = self.pins[&pin].debounce_ms;
        if debounce_ms > 0 {
            if let Some(last_change) = self.pins[&pin].last_change {
                let elapsed = now.signed_duration_since(last_change).num_milliseconds();
                if elapsed < debounce_ms as i64 {
                    return Ok(());
                }
            }
        }

        let int_trigger = self.pins[&pin].interrupt;
        let state = self.pins.get_mut(&pin).unwrap();
        state.level = level;
        state.last_change = Some(now);

        if old_level != level {
            self.record_change(pin, old_level, level, now);

            // Check interrupt.
            if let Some(trigger) = int_trigger {
                if trigger.matches(old_level, level) {
                    self.interrupts.push(InterruptEvent {
                        pin,
                        trigger,
                        level,
                        timestamp: now,
                    });
                }
            }
        }

        Ok(())
    }

    /// Enable edge-triggered interrupt on a pin.
    pub fn enable_interrupt(&mut self, pin: u8, trigger: EdgeTrigger) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        state.interrupt = Some(trigger);
        Ok(())
    }

    /// Disable interrupt on a pin.
    pub fn disable_interrupt(&mut self, pin: u8) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        state.interrupt = None;
        Ok(())
    }

    /// Set debounce time for a pin.
    pub fn set_debounce(&mut self, pin: u8, ms: u32) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        state.debounce_ms = ms;
        Ok(())
    }

    /// Configure PWM on an output pin.
    pub fn configure_pwm(&mut self, pin: u8, config: PwmConfig) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        if state.mode != PinMode::Output {
            return Err(GpioError::WrongMode { pin, expected: PinMode::Output, actual: state.mode });
        }
        state.pwm = Some(config);
        Ok(())
    }

    /// Get PWM config for a pin.
    pub fn pwm_config(&self, pin: u8) -> Result<Option<&PwmConfig>, GpioError> {
        let state = self.pins.get(&pin).ok_or(GpioError::InvalidPin(pin))?;
        Ok(state.pwm.as_ref())
    }

    /// Set the duty cycle for a PWM pin.
    pub fn set_pwm_duty(&mut self, pin: u8, duty: f64) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        match &mut state.pwm {
            Some(pwm) => {
                pwm.duty_cycle = duty.clamp(0.0, 1.0);
                Ok(())
            }
            None => Err(GpioError::NoPwm(pin)),
        }
    }

    /// Set alternate function for a pin.
    pub fn set_alternate(&mut self, pin: u8, function: &str) -> Result<(), GpioError> {
        let state = self.pins.get_mut(&pin).ok_or(GpioError::InvalidPin(pin))?;
        state.mode = PinMode::Alternate;
        state.alternate_function = Some(function.to_string());
        Ok(())
    }

    /// Register a pin group.
    pub fn add_group(&mut self, group: PinGroup) {
        self.groups.insert(group.name.clone(), group);
    }

    /// Write the same level to all pins in a group.
    pub fn write_group(&mut self, group_name: &str, level: Level) -> Result<(), GpioError> {
        let pins = self.groups.get(group_name)
            .ok_or(GpioError::InvalidGroup(group_name.to_string()))?
            .pins.clone();
        for pin in pins {
            // Only write to output pins — skip others silently.
            if let Some(state) = self.pins.get(&pin) {
                if state.mode == PinMode::Output {
                    self.write(pin, level)?;
                }
            }
        }
        Ok(())
    }

    /// Read all pins in a group as a bit vector (LSB = first pin in group).
    pub fn read_group(&self, group_name: &str) -> Result<Vec<Level>, GpioError> {
        let group = self.groups.get(group_name)
            .ok_or(GpioError::InvalidGroup(group_name.to_string()))?;
        let mut levels = Vec::with_capacity(group.pins.len());
        for pin in &group.pins {
            levels.push(self.read(*pin)?);
        }
        Ok(levels)
    }

    /// Get the pin mode.
    pub fn pin_mode(&self, pin: u8) -> Result<PinMode, GpioError> {
        let state = self.pins.get(&pin).ok_or(GpioError::InvalidPin(pin))?;
        Ok(state.mode)
    }

    /// Get pin state history.
    pub fn history(&self) -> &[PinStateChange] {
        &self.history
    }

    /// Get interrupt events.
    pub fn interrupts(&self) -> &[InterruptEvent] {
        &self.interrupts
    }

    /// Get history for a specific pin.
    pub fn pin_history(&self, pin: u8) -> Vec<&PinStateChange> {
        self.history.iter().filter(|h| h.pin == pin).collect()
    }

    /// Clear history and interrupt log.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.interrupts.clear();
    }

    pub fn pin_count(&self) -> usize {
        self.pins.len()
    }

    fn record_change(&mut self, pin: u8, old: Level, new: Level, ts: DateTime<Utc>) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(PinStateChange {
            pin,
            old_level: old,
            new_level: new,
            timestamp: ts,
        });
    }
}

/// GPIO errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpioError {
    InvalidPin(u8),
    WrongMode { pin: u8, expected: PinMode, actual: PinMode },
    NoPwm(u8),
    InvalidGroup(String),
}

impl std::fmt::Display for GpioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPin(p) => write!(f, "invalid pin: {}", p),
            Self::WrongMode { pin, expected, actual } => {
                write!(f, "pin {}: expected mode {}, got {}", pin, expected.as_str(), actual.as_str())
            }
            Self::NoPwm(p) => write!(f, "pin {} has no PWM configured", p),
            Self::InvalidGroup(g) => write!(f, "invalid group: {}", g),
        }
    }
}

impl std::error::Error for GpioError {}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_mode_default() {
        let gpio = GpioController::new(8);
        assert_eq!(gpio.pin_mode(0).unwrap(), PinMode::Disabled);
    }

    #[test]
    fn set_output_and_write() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Output).unwrap();
        gpio.write(0, Level::High).unwrap();
        assert_eq!(gpio.read(0).unwrap(), Level::High);
    }

    #[test]
    fn write_to_input_fails() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        assert!(gpio.write(0, Level::High).is_err());
    }

    #[test]
    fn invalid_pin() {
        let gpio = GpioController::new(4);
        assert!(gpio.read(10).is_err());
    }

    #[test]
    fn pull_up() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.set_pull(0, Pull::Up).unwrap();
        assert_eq!(gpio.read(0).unwrap(), Level::High);
    }

    #[test]
    fn pull_down() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.set_pull(0, Pull::Down).unwrap();
        assert_eq!(gpio.read(0).unwrap(), Level::Low);
    }

    #[test]
    fn drive_input() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.drive_input(0, Level::High).unwrap();
        assert_eq!(gpio.read(0).unwrap(), Level::High);
    }

    #[test]
    fn interrupt_rising() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.enable_interrupt(0, EdgeTrigger::Rising).unwrap();
        gpio.drive_input(0, Level::High).unwrap();
        assert_eq!(gpio.interrupts().len(), 1);
        assert_eq!(gpio.interrupts()[0].trigger, EdgeTrigger::Rising);
    }

    #[test]
    fn interrupt_falling() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.drive_input(0, Level::High).unwrap();
        gpio.enable_interrupt(0, EdgeTrigger::Falling).unwrap();
        gpio.drive_input(0, Level::Low).unwrap();
        // Should have falling edge interrupt.
        let falling: Vec<_> = gpio.interrupts().iter()
            .filter(|i| i.trigger == EdgeTrigger::Falling)
            .collect();
        assert_eq!(falling.len(), 1);
    }

    #[test]
    fn interrupt_both() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.enable_interrupt(0, EdgeTrigger::Both).unwrap();
        gpio.drive_input(0, Level::High).unwrap();
        gpio.drive_input(0, Level::Low).unwrap();
        assert_eq!(gpio.interrupts().len(), 2);
    }

    #[test]
    fn disable_interrupt() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Input).unwrap();
        gpio.enable_interrupt(0, EdgeTrigger::Rising).unwrap();
        gpio.disable_interrupt(0).unwrap();
        gpio.drive_input(0, Level::High).unwrap();
        assert!(gpio.interrupts().is_empty());
    }

    #[test]
    fn pin_history_tracking() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Output).unwrap();
        gpio.write(0, Level::High).unwrap();
        gpio.write(0, Level::Low).unwrap();
        assert_eq!(gpio.history().len(), 2);
        assert_eq!(gpio.pin_history(0).len(), 2);
    }

    #[test]
    fn pwm_configuration() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Output).unwrap();
        gpio.configure_pwm(0, PwmConfig::new(1000, 0.5)).unwrap();
        let pwm = gpio.pwm_config(0).unwrap().unwrap();
        assert_eq!(pwm.frequency_hz, 1000);
        assert!((pwm.duty_cycle - 0.5).abs() < 1e-10);
    }

    #[test]
    fn pwm_timing() {
        let pwm = PwmConfig::new(1000, 0.25);
        assert!((pwm.period_us() - 1000.0).abs() < 1e-6);
        assert!((pwm.high_time_us() - 250.0).abs() < 1e-6);
        assert!((pwm.low_time_us() - 750.0).abs() < 1e-6);
    }

    #[test]
    fn set_pwm_duty() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Output).unwrap();
        gpio.configure_pwm(0, PwmConfig::new(1000, 0.5)).unwrap();
        gpio.set_pwm_duty(0, 0.75).unwrap();
        let pwm = gpio.pwm_config(0).unwrap().unwrap();
        assert!((pwm.duty_cycle - 0.75).abs() < 1e-10);
    }

    #[test]
    fn pin_group_write_read() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Output).unwrap();
        gpio.set_mode(1, PinMode::Output).unwrap();
        gpio.set_mode(2, PinMode::Output).unwrap();
        gpio.add_group(PinGroup::new("leds", vec![0, 1, 2]));
        gpio.write_group("leds", Level::High).unwrap();
        let levels = gpio.read_group("leds").unwrap();
        assert!(levels.iter().all(|l| *l == Level::High));
    }

    #[test]
    fn level_invert() {
        assert_eq!(Level::High.invert(), Level::Low);
        assert_eq!(Level::Low.invert(), Level::High);
    }

    #[test]
    fn level_bool_conversion() {
        assert!(Level::High.as_bool());
        assert!(!Level::Low.as_bool());
        assert_eq!(Level::from_bool(true), Level::High);
        assert_eq!(Level::from_bool(false), Level::Low);
    }

    #[test]
    fn edge_trigger_matches() {
        assert!(EdgeTrigger::Rising.matches(Level::Low, Level::High));
        assert!(!EdgeTrigger::Rising.matches(Level::High, Level::Low));
        assert!(EdgeTrigger::Falling.matches(Level::High, Level::Low));
        assert!(EdgeTrigger::Both.matches(Level::Low, Level::High));
        assert!(EdgeTrigger::Both.matches(Level::High, Level::Low));
    }

    #[test]
    fn alternate_function() {
        let mut gpio = GpioController::new(8);
        gpio.set_alternate(0, "SPI_MOSI").unwrap();
        assert_eq!(gpio.pin_mode(0).unwrap(), PinMode::Alternate);
    }

    #[test]
    fn clear_history() {
        let mut gpio = GpioController::new(8);
        gpio.set_mode(0, PinMode::Output).unwrap();
        gpio.write(0, Level::High).unwrap();
        gpio.clear_history();
        assert!(gpio.history().is_empty());
    }
}
