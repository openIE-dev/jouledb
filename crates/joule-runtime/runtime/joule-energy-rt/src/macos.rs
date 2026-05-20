//! macOS energy monitoring for Apple Silicon (M1/M2/M3/M4/M5)
//!
//! This module provides energy and power monitoring on macOS using:
//! - IOKit for accessing the System Management Controller (SMC)
//! - IOReport for reading Apple Silicon power metrics
//! - IOKit thermal sensors for temperature readings
//!
//! Apple Silicon chips expose energy counters through IOReport, which provides
//! access to CPU, GPU, ANE (Apple Neural Engine), and package power data.

#![cfg(target_os = "macos")]

use crate::error::{Error, Result};
use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use std::ffi::CStr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// IOKit bindings for macOS
#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(
        name: *const libc::c_char,
    ) -> core_foundation_sys::dictionary::CFMutableDictionaryRef;
    fn IOServiceGetMatchingServices(
        mainPort: u32,
        matching: core_foundation_sys::dictionary::CFDictionaryRef,
        existing: *mut u32,
    ) -> i32;
    fn IOIteratorNext(iterator: u32) -> u32;
    fn IORegistryEntryCreateCFProperties(
        entry: u32,
        properties: *mut core_foundation_sys::dictionary::CFMutableDictionaryRef,
        allocator: core_foundation_sys::base::CFAllocatorRef,
        options: u32,
    ) -> i32;
    fn IOObjectRelease(object: u32) -> i32;
    fn IORegistryEntryGetName(entry: u32, name: *mut libc::c_char) -> i32;
}

// IOReport bindings for Apple Silicon power metrics
#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: core_foundation_sys::string::CFStringRef,
        subgroup: core_foundation_sys::string::CFStringRef,
        channel_id: u64,
        channel_id2: u64,
        channel_id3: u64,
    ) -> core_foundation_sys::dictionary::CFDictionaryRef;

    fn IOReportCreateSubscription(
        desiredChannels: *const libc::c_void,
        subbedChannels: *mut core_foundation_sys::dictionary::CFDictionaryRef,
        channelCount: *mut i32,
        options: u32,
    ) -> *mut libc::c_void;

    fn IOReportCreateSamples(
        subscription: *mut libc::c_void,
        sampleBuffer: core_foundation_sys::dictionary::CFDictionaryRef,
        options: u32,
    ) -> core_foundation_sys::dictionary::CFDictionaryRef;

    fn IOReportCreateSamplesDelta(
        prev: core_foundation_sys::dictionary::CFDictionaryRef,
        current: core_foundation_sys::dictionary::CFDictionaryRef,
        options: u32,
    ) -> core_foundation_sys::dictionary::CFDictionaryRef;

    fn IOReportIterate(
        samples: core_foundation_sys::dictionary::CFDictionaryRef,
        callback: extern "C" fn(core_foundation_sys::dictionary::CFDictionaryRef) -> i32,
    ) -> i32;

    #[allow(dead_code)]
    fn IOReportChannelGetGroup(
        channel: core_foundation_sys::dictionary::CFDictionaryRef,
    ) -> core_foundation_sys::string::CFStringRef;
    fn IOReportChannelGetSubGroup(
        channel: core_foundation_sys::dictionary::CFDictionaryRef,
    ) -> core_foundation_sys::string::CFStringRef;
    #[allow(dead_code)]
    fn IOReportChannelGetChannelName(
        channel: core_foundation_sys::dictionary::CFDictionaryRef,
    ) -> core_foundation_sys::string::CFStringRef;
    fn IOReportSimpleGetIntegerValue(
        channel: core_foundation_sys::dictionary::CFDictionaryRef,
        value: *mut i64,
    ) -> i32;
}

/// Constants for IOKit
const K_IO_MAIN_PORT_DEFAULT: u32 = 0;
const K_IO_RETURN_SUCCESS: i32 = 0;

/// Apple Silicon power channel groups
const ENERGY_MODEL_GROUP: &str = "Energy Model";
#[allow(dead_code)]
const CPU_ENERGY_SUBGROUP: &str = "CPU Energy";
#[allow(dead_code)]
const GPU_ENERGY_SUBGROUP: &str = "GPU Energy";

/// SMC key constants for temperature (FourCC format)
const SMC_KEY_CPU_TEMP: &[u8; 4] = b"TC0P"; // CPU proximity temperature
const SMC_KEY_CPU_DIE_TEMP: &[u8; 4] = b"TC0D"; // CPU die temperature
#[allow(dead_code)]
const SMC_KEY_GPU_TEMP: &[u8; 4] = b"TG0P"; // GPU proximity temperature

/// Energy sample data
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergySample {
    /// CPU energy in microjoules
    pub cpu_energy_uj: u64,
    /// GPU energy in microjoules
    pub gpu_energy_uj: u64,
    /// ANE (Apple Neural Engine) energy in microjoules
    pub ane_energy_uj: u64,
    /// Total package energy in microjoules
    pub package_energy_uj: u64,
    /// Sample duration in nanoseconds
    pub duration_ns: u64,
}

impl EnergySample {
    /// Get total energy in joules
    pub fn total_energy_joules(&self) -> f64 {
        self.package_energy_uj as f64 / 1_000_000.0
    }

    /// Get average power in watts
    pub fn average_power_watts(&self) -> Option<f64> {
        if self.duration_ns == 0 {
            return None;
        }
        let energy_j = self.total_energy_joules();
        let duration_s = self.duration_ns as f64 / 1_000_000_000.0;
        Some(energy_j / duration_s)
    }
}

// Thread-local storage for energy accumulation during IOReport iteration
thread_local! {
    static ENERGY_ACCUMULATOR: std::cell::RefCell<EnergySample> = const { std::cell::RefCell::new(EnergySample {
        cpu_energy_uj: 0,
        gpu_energy_uj: 0,
        ane_energy_uj: 0,
        package_energy_uj: 0,
        duration_ns: 0,
    }) };
}

/// Callback function for IOReport iteration
/// This is called for each channel in the sample
extern "C" fn energy_iterate_callback(
    channel: core_foundation_sys::dictionary::CFDictionaryRef,
) -> i32 {
    unsafe {
        let subgroup_ref = IOReportChannelGetSubGroup(channel);
        if subgroup_ref.is_null() {
            return 0;
        }

        let subgroup = CFString::wrap_under_get_rule(subgroup_ref);
        let subgroup_str = subgroup.to_string();

        let mut value: i64 = 0;
        if IOReportSimpleGetIntegerValue(channel, &mut value) != 0 {
            return 0;
        }

        // Energy values are typically in microjoules
        let energy_uj = value.unsigned_abs();

        ENERGY_ACCUMULATOR.with(|acc| {
            let mut sample = acc.borrow_mut();
            if subgroup_str.contains("CPU") {
                sample.cpu_energy_uj += energy_uj;
            } else if subgroup_str.contains("GPU") {
                sample.gpu_energy_uj += energy_uj;
            } else if subgroup_str.contains("ANE") {
                sample.ane_energy_uj += energy_uj;
            }
            sample.package_energy_uj += energy_uj;
        });

        0 // Continue iteration
    }
}

/// Energy reading state
struct EnergyState {
    subscription: *mut libc::c_void,
    last_sample: Option<core_foundation_sys::dictionary::CFDictionaryRef>,
    last_time: Instant,
    cumulative_energy_uj: AtomicU64,
}

// SAFETY: The IOReport subscription handle is thread-safe when accessed through proper synchronization
unsafe impl Send for EnergyState {}
unsafe impl Sync for EnergyState {}

/// macOS energy reader using IOReport for Apple Silicon
pub struct MacOSEnergyReader {
    state: Mutex<EnergyState>,
    is_apple_silicon: bool,
}

impl MacOSEnergyReader {
    /// Create a new macOS energy reader
    ///
    /// This detects whether we're running on Apple Silicon and initializes
    /// the appropriate energy monitoring subsystem.
    pub fn new() -> Result<Self> {
        let is_apple_silicon = Self::detect_apple_silicon();

        if !is_apple_silicon {
            return Err(Error::Unsupported(
                "macOS energy monitoring requires Apple Silicon (M1/M2/M3/M4/M5)".to_string(),
            ));
        }

        // Initialize IOReport subscription for energy metrics
        let subscription = Self::create_energy_subscription()?;

        let state = EnergyState {
            subscription,
            last_sample: None,
            last_time: Instant::now(),
            cumulative_energy_uj: AtomicU64::new(0),
        };

        Ok(Self {
            state: Mutex::new(state),
            is_apple_silicon,
        })
    }

    /// Detect if running on Apple Silicon
    fn detect_apple_silicon() -> bool {
        // Check CPU brand string for Apple Silicon
        #[cfg(target_arch = "aarch64")]
        {
            // On ARM64 macOS, we're definitely on Apple Silicon
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // On x86_64 macOS, we're on Intel
            false
        }
    }

    /// Create IOReport subscription for energy channels
    fn create_energy_subscription() -> Result<*mut libc::c_void> {
        unsafe {
            let group = CFString::new(ENERGY_MODEL_GROUP);
            let group_ref = group.as_concrete_TypeRef();

            // Get all energy model channels
            let channels = IOReportCopyChannelsInGroup(group_ref, std::ptr::null(), 0, 0, 0);

            if channels.is_null() {
                return Err(Error::Unsupported(
                    "Failed to get IOReport energy channels. This may require running as root or enabling 'Full Disk Access'.".to_string(),
                ));
            }

            let mut subscribed_channels: core_foundation_sys::dictionary::CFDictionaryRef =
                std::ptr::null();
            let mut channel_count: i32 = 0;

            let subscription = IOReportCreateSubscription(
                channels as *const libc::c_void,
                &mut subscribed_channels,
                &mut channel_count,
                0,
            );

            // Release the channels dictionary
            core_foundation_sys::base::CFRelease(channels as *const libc::c_void);

            if subscription.is_null() {
                return Err(Error::Unsupported(
                    "Failed to create IOReport subscription for energy monitoring".to_string(),
                ));
            }

            if channel_count == 0 {
                return Err(Error::Unsupported(
                    "No energy channels available. Ensure you're running on Apple Silicon."
                        .to_string(),
                ));
            }

            Ok(subscription)
        }
    }

    /// Read current energy sample
    fn read_sample(&self) -> Result<EnergySample> {
        let mut state = self.state.lock().map_err(|_| {
            Error::Unsupported("Failed to acquire lock for energy reading".to_string())
        })?;

        unsafe {
            let current_sample = IOReportCreateSamples(state.subscription, std::ptr::null(), 0);

            if current_sample.is_null() {
                return Err(Error::Unsupported(
                    "Failed to create IOReport sample".to_string(),
                ));
            }

            let sample = if let Some(prev_sample) = state.last_sample {
                // Calculate delta from previous sample
                let delta = IOReportCreateSamplesDelta(prev_sample, current_sample, 0);

                // Release previous sample
                core_foundation_sys::base::CFRelease(prev_sample as *const libc::c_void);

                if delta.is_null() {
                    core_foundation_sys::base::CFRelease(current_sample as *const libc::c_void);
                    return Err(Error::Unsupported(
                        "Failed to calculate energy delta".to_string(),
                    ));
                }

                // Parse the delta sample
                let parsed = Self::parse_energy_sample(delta);

                core_foundation_sys::base::CFRelease(delta as *const libc::c_void);
                parsed
            } else {
                // First sample - return zero delta
                EnergySample {
                    cpu_energy_uj: 0,
                    gpu_energy_uj: 0,
                    ane_energy_uj: 0,
                    package_energy_uj: 0,
                    duration_ns: 0,
                }
            };

            // Update state
            state.last_sample = Some(current_sample);
            let now = Instant::now();
            let elapsed = now.duration_since(state.last_time);
            state.last_time = now;

            // Update cumulative energy
            let total_uj = sample.cpu_energy_uj + sample.gpu_energy_uj + sample.ane_energy_uj;
            state
                .cumulative_energy_uj
                .fetch_add(total_uj, Ordering::Relaxed);

            Ok(EnergySample {
                duration_ns: elapsed.as_nanos() as u64,
                ..sample
            })
        }
    }

    /// Parse energy values from IOReport sample
    fn parse_energy_sample(
        sample: core_foundation_sys::dictionary::CFDictionaryRef,
    ) -> EnergySample {
        // Reset global accumulator before iteration
        ENERGY_ACCUMULATOR.with(|acc| {
            *acc.borrow_mut() = EnergySample::default();
        });

        unsafe {
            IOReportIterate(sample, energy_iterate_callback);
        }

        // Retrieve accumulated values
        ENERGY_ACCUMULATOR.with(|acc| *acc.borrow())
    }

    /// Read cumulative energy in joules
    pub fn read_cumulative_energy(&self) -> Result<f64> {
        // Take a new sample to update cumulative energy
        self.read_sample()?;

        let state = self.state.lock().map_err(|_| {
            Error::Unsupported("Failed to acquire lock for energy reading".to_string())
        })?;

        let energy_uj = state.cumulative_energy_uj.load(Ordering::Relaxed);
        Ok(energy_uj as f64 / 1_000_000.0) // Convert microjoules to joules
    }

    /// Read current power in watts
    pub fn read_current_power(&self) -> Result<f64> {
        let sample = self.read_sample()?;

        if sample.duration_ns == 0 {
            return Err(Error::Unsupported(
                "Cannot calculate power from first sample".to_string(),
            ));
        }

        // Calculate power: P = E / t
        let energy_j = sample.package_energy_uj as f64 / 1_000_000.0;
        let duration_s = sample.duration_ns as f64 / 1_000_000_000.0;

        Ok(energy_j / duration_s)
    }
}

impl Drop for MacOSEnergyReader {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            unsafe {
                if let Some(sample) = state.last_sample {
                    core_foundation_sys::base::CFRelease(sample as *const libc::c_void);
                }
                // Note: IOReport subscriptions are cleaned up automatically
            }
            state.last_sample = None;
        }
    }
}

/// macOS thermal reader using IOKit SMC access
pub struct MacOSThermalReader {
    smc_connection: u32,
}

impl MacOSThermalReader {
    /// Create a new thermal reader
    pub fn new() -> Result<Self> {
        let smc_connection = Self::connect_to_smc()?;
        Ok(Self { smc_connection })
    }

    /// Connect to the SMC service
    fn connect_to_smc() -> Result<u32> {
        unsafe {
            let service_name = b"AppleSMC\0";
            let matching = IOServiceMatching(service_name.as_ptr() as *const libc::c_char);

            if matching.is_null() {
                return Err(Error::Unsupported(
                    "Failed to create IOService matching dictionary for SMC".to_string(),
                ));
            }

            let mut iterator: u32 = 0;
            let result =
                IOServiceGetMatchingServices(K_IO_MAIN_PORT_DEFAULT, matching, &mut iterator);

            if result != K_IO_RETURN_SUCCESS {
                return Err(Error::Unsupported(format!(
                    "Failed to find SMC service: error code {}",
                    result
                )));
            }

            let service = IOIteratorNext(iterator);
            IOObjectRelease(iterator);

            if service == 0 {
                return Err(Error::Unsupported("SMC service not found".to_string()));
            }

            Ok(service)
        }
    }

    /// Read CPU temperature in Celsius
    pub fn read_cpu_temperature(&self) -> Result<f64> {
        // Try multiple temperature keys
        let keys = [SMC_KEY_CPU_DIE_TEMP, SMC_KEY_CPU_TEMP];

        for key in &keys {
            if let Ok(temp) = self.read_smc_temperature(key) {
                return Ok(temp);
            }
        }

        // Fall back to IOKit thermal zones
        self.read_thermal_zone_temperature()
    }

    /// Read GPU temperature in Celsius
    pub fn read_gpu_temperature(&self) -> Result<f64> {
        self.read_smc_temperature(SMC_KEY_GPU_TEMP)
    }

    /// Read temperature from SMC key
    fn read_smc_temperature(&self, _key: &[u8; 4]) -> Result<f64> {
        // SMC temperature reading requires low-level SMC protocol
        // This is a simplified implementation that uses IOKit properties
        unsafe {
            let mut properties: core_foundation_sys::dictionary::CFMutableDictionaryRef =
                std::ptr::null_mut();
            let result = IORegistryEntryCreateCFProperties(
                self.smc_connection,
                &mut properties,
                std::ptr::null(),
                0,
            );

            if result != K_IO_RETURN_SUCCESS || properties.is_null() {
                return Err(Error::Unsupported(
                    "Failed to read SMC properties".to_string(),
                ));
            }

            let dict = CFDictionary::<CFString, CFType>::wrap_under_create_rule(
                properties as core_foundation_sys::dictionary::CFDictionaryRef,
            );

            // Look for temperature keys in properties
            let temp_key = CFString::new("Temperature");
            if let Some(value) = dict.find(&temp_key) {
                if let Some(number) = value.downcast::<CFNumber>() {
                    if let Some(temp) = number.to_f64() {
                        // SMC temperatures are often in 1/100th degree or direct Celsius
                        let normalized_temp = if temp > 200.0 { temp / 100.0 } else { temp };
                        return Ok(normalized_temp);
                    }
                }
            }

            Err(Error::Unsupported(
                "Temperature not found in SMC properties".to_string(),
            ))
        }
    }

    /// Read temperature from IOKit thermal zones (alternative method)
    fn read_thermal_zone_temperature(&self) -> Result<f64> {
        unsafe {
            // Try to find AppleARMIODevice for thermal sensors
            let service_name = b"AppleARMIODevice\0";
            let matching = IOServiceMatching(service_name.as_ptr() as *const libc::c_char);

            if matching.is_null() {
                return Err(Error::Unsupported(
                    "Failed to create thermal matching dictionary".to_string(),
                ));
            }

            let mut iterator: u32 = 0;
            let result =
                IOServiceGetMatchingServices(K_IO_MAIN_PORT_DEFAULT, matching, &mut iterator);

            if result != K_IO_RETURN_SUCCESS {
                return Err(Error::Unsupported(
                    "Failed to find thermal services".to_string(),
                ));
            }

            let mut best_temp: Option<f64> = None;

            loop {
                let service = IOIteratorNext(iterator);
                if service == 0 {
                    break;
                }

                // Get service name
                let mut name_buf = [0i8; 128];
                if IORegistryEntryGetName(service, name_buf.as_mut_ptr()) == K_IO_RETURN_SUCCESS {
                    let name = CStr::from_ptr(name_buf.as_ptr());
                    let name_str = name.to_string_lossy();

                    // Look for thermal-related devices
                    if name_str.contains("thermal") || name_str.contains("temp") {
                        if let Ok(temp) = self.read_service_temperature(service) {
                            best_temp = Some(temp);
                            IOObjectRelease(service);
                            break;
                        }
                    }
                }

                IOObjectRelease(service);
            }

            IOObjectRelease(iterator);

            best_temp.ok_or_else(|| Error::Unsupported("No thermal sensors found".to_string()))
        }
    }

    /// Read temperature from a specific IOKit service
    fn read_service_temperature(&self, service: u32) -> Result<f64> {
        unsafe {
            let mut properties: core_foundation_sys::dictionary::CFMutableDictionaryRef =
                std::ptr::null_mut();
            let result =
                IORegistryEntryCreateCFProperties(service, &mut properties, std::ptr::null(), 0);

            if result != K_IO_RETURN_SUCCESS || properties.is_null() {
                return Err(Error::Unsupported(
                    "Failed to read service properties".to_string(),
                ));
            }

            let dict = CFDictionary::<CFString, CFType>::wrap_under_create_rule(
                properties as core_foundation_sys::dictionary::CFDictionaryRef,
            );

            // Common temperature property names
            let temp_keys = [
                "temperature",
                "Temperature",
                "current-temp",
                "die-temperature",
            ];

            for key_name in &temp_keys {
                let key = CFString::new(key_name);
                if let Some(value) = dict.find(&key) {
                    if let Some(number) = value.downcast::<CFNumber>() {
                        if let Some(temp) = number.to_f64() {
                            let normalized_temp = if temp > 200.0 { temp / 100.0 } else { temp };
                            if (0.0..=150.0).contains(&normalized_temp) {
                                return Ok(normalized_temp);
                            }
                        }
                    }
                }
            }

            Err(Error::Unsupported(
                "No valid temperature found in service".to_string(),
            ))
        }
    }
}

impl Drop for MacOSThermalReader {
    fn drop(&mut self) {
        unsafe {
            IOObjectRelease(self.smc_connection);
        }
    }
}

/// High-level macOS energy monitor combining energy and thermal readings
pub struct MacOSEnergyMonitor {
    energy_reader: MacOSEnergyReader,
    thermal_reader: Option<MacOSThermalReader>,
}

impl MacOSEnergyMonitor {
    /// Create a new macOS energy monitor
    pub fn new() -> Result<Self> {
        let energy_reader = MacOSEnergyReader::new()?;
        let thermal_reader = MacOSThermalReader::new().ok(); // Thermal is optional

        Ok(Self {
            energy_reader,
            thermal_reader,
        })
    }

    /// Read cumulative energy in joules
    pub fn read_energy(&self) -> Result<f64> {
        self.energy_reader.read_cumulative_energy()
    }

    /// Read current power in watts
    pub fn read_power(&self) -> Result<f64> {
        self.energy_reader.read_current_power()
    }

    /// Read CPU temperature in Celsius
    pub fn read_temperature(&self) -> Result<f64> {
        match &self.thermal_reader {
            Some(reader) => reader.read_cpu_temperature(),
            None => Err(Error::Unsupported(
                "Thermal monitoring not available".to_string(),
            )),
        }
    }

    /// Check if running on Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        self.energy_reader.is_apple_silicon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_apple_silicon() {
        let is_arm = cfg!(target_arch = "aarch64");
        assert_eq!(MacOSEnergyReader::detect_apple_silicon(), is_arm);
    }

    #[test]
    #[ignore] // Requires Apple Silicon Mac
    fn test_energy_reader() {
        let reader = MacOSEnergyReader::new().expect("Failed to create energy reader");
        let energy = reader
            .read_cumulative_energy()
            .expect("Failed to read energy");
        println!("Cumulative energy: {:.6} J", energy);
    }

    #[test]
    #[ignore] // Requires Apple Silicon Mac
    fn test_power_reading() {
        let reader = MacOSEnergyReader::new().expect("Failed to create energy reader");

        // Take initial sample
        let _ = reader.read_sample();

        // Wait a bit
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Read power
        let power = reader.read_current_power().expect("Failed to read power");
        println!("Current power: {:.2} W", power);
        assert!(power > 0.0 && power < 200.0); // Reasonable range for Apple Silicon
    }

    #[test]
    #[ignore] // Requires Apple Silicon Mac
    fn test_thermal_reader() {
        let reader = MacOSThermalReader::new().expect("Failed to create thermal reader");
        let temp = reader
            .read_cpu_temperature()
            .expect("Failed to read temperature");
        println!("CPU temperature: {:.1}C", temp);
        assert!(temp > 20.0 && temp < 120.0); // Reasonable temperature range
    }

    #[test]
    #[ignore] // Requires Apple Silicon Mac
    fn test_full_monitor() {
        let monitor = MacOSEnergyMonitor::new().expect("Failed to create monitor");

        assert!(monitor.is_apple_silicon());

        let energy = monitor.read_energy().expect("Failed to read energy");
        println!("Energy: {:.6} J", energy);

        std::thread::sleep(std::time::Duration::from_millis(100));

        let power = monitor.read_power().expect("Failed to read power");
        println!("Power: {:.2} W", power);

        if let Ok(temp) = monitor.read_temperature() {
            println!("Temperature: {:.1}C", temp);
        }
    }
}
