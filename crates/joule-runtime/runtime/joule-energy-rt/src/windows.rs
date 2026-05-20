//! Windows energy monitoring using EMI (Energy Metering Interface)
//!
//! This module provides energy monitoring on Windows 10+ using the Energy Metering
//! Interface (EMI) which exposes hardware energy counters from the platform.
//!
//! EMI is available on systems with compatible hardware energy meters, typically
//! modern Intel and AMD processors with RAPL-like capabilities exposed through
//! the Windows power management stack.

#![cfg(target_os = "windows")]

use crate::error::{Error, Result};
use std::ffi::OsStr;
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::OsStrExt;
use std::ptr::null_mut;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Windows API type definitions
type HANDLE = *mut std::ffi::c_void;
type DWORD = u32;
type BOOL = i32;
type HDEVINFO = *mut std::ffi::c_void;

const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;
const DIGCF_PRESENT: DWORD = 0x02;
const DIGCF_DEVICEINTERFACE: DWORD = 0x10;
const GENERIC_READ: DWORD = 0x80000000;
const FILE_SHARE_READ: DWORD = 0x01;
const FILE_SHARE_WRITE: DWORD = 0x02;
const OPEN_EXISTING: DWORD = 3;
const FILE_ATTRIBUTE_NORMAL: DWORD = 0x80;

// EMI GUID: {45BD8344-7ED6-49CF-A440-C276C933B053}
const EMI_DEVICE_INTERFACE_GUID: GUID = GUID {
    data1: 0x45BD8344,
    data2: 0x7ED6,
    data3: 0x49CF,
    data4: [0xA4, 0x40, 0xC2, 0x76, 0xC9, 0x33, 0xB0, 0x53],
};

// EMI IOCTL codes
const FILE_DEVICE_EMI: DWORD = 0x8000; // Private device type for EMI
const METHOD_BUFFERED: DWORD = 0;
const FILE_READ_ACCESS: DWORD = 0x01;

// CTL_CODE macro equivalent
const fn ctl_code(device_type: DWORD, function: DWORD, method: DWORD, access: DWORD) -> DWORD {
    (device_type << 16) | (access << 14) | (function << 2) | method
}

const IOCTL_EMI_GET_VERSION: DWORD =
    ctl_code(FILE_DEVICE_EMI, 0x01, METHOD_BUFFERED, FILE_READ_ACCESS);
const IOCTL_EMI_GET_METADATA_SIZE: DWORD =
    ctl_code(FILE_DEVICE_EMI, 0x02, METHOD_BUFFERED, FILE_READ_ACCESS);
const IOCTL_EMI_GET_METADATA: DWORD =
    ctl_code(FILE_DEVICE_EMI, 0x03, METHOD_BUFFERED, FILE_READ_ACCESS);
const IOCTL_EMI_GET_MEASUREMENT: DWORD =
    ctl_code(FILE_DEVICE_EMI, 0x04, METHOD_BUFFERED, FILE_READ_ACCESS);

// EMI version constants
const EMI_VERSION_V1: u16 = 0x0001;
const EMI_VERSION_V2: u16 = 0x0002;

/// GUID structure matching Windows definition
#[repr(C)]
#[derive(Clone, Copy)]
struct GUID {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

/// SP_DEVICE_INTERFACE_DATA structure
#[repr(C)]
struct SP_DEVICE_INTERFACE_DATA {
    cb_size: DWORD,
    interface_class_guid: GUID,
    flags: DWORD,
    reserved: usize,
}

/// SP_DEVICE_INTERFACE_DETAIL_DATA_W header
#[repr(C)]
struct SP_DEVICE_INTERFACE_DETAIL_DATA_W {
    cb_size: DWORD,
    device_path: [u16; 1], // Variable length, at least 1
}

/// EMI version structure
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct EMI_VERSION {
    emi_version: u16,
}

/// EMI metadata size structure
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct EMI_METADATA_SIZE {
    metadata_size: DWORD,
}

/// EMI metadata V1 structure
#[repr(C)]
#[derive(Clone, Copy)]
struct EMI_METADATA_V1 {
    measurement_unit: u8,
    hardware_oem: [u16; 16],
    hardware_model: [u16; 16],
    hardware_revision: u16,
    metering_hardware_vendor: u16,
}

/// EMI channel metadata V2
#[repr(C)]
#[derive(Clone, Copy)]
struct EMI_CHANNEL_MEASUREMENT_DATA {
    absolute_energy: u64, // Energy in picowatt-hours
    absolute_time: u64,   // Time in 100ns intervals
}

/// EMI measurement data V1
#[repr(C)]
#[derive(Clone, Copy)]
struct EMI_MEASUREMENT_DATA_V1 {
    absolute_energy: u64, // Energy in picowatt-hours
    absolute_time: u64,   // Time in 100ns intervals
}

// Windows API bindings
#[link(name = "setupapi")]
unsafe extern "system" {
    fn SetupDiGetClassDevsW(
        class_guid: *const GUID,
        enumerator: *const u16,
        hwnd_parent: HANDLE,
        flags: DWORD,
    ) -> HDEVINFO;

    fn SetupDiEnumDeviceInterfaces(
        device_info_set: HDEVINFO,
        device_info_data: *const std::ffi::c_void,
        interface_class_guid: *const GUID,
        member_index: DWORD,
        device_interface_data: *mut SP_DEVICE_INTERFACE_DATA,
    ) -> BOOL;

    fn SetupDiGetDeviceInterfaceDetailW(
        device_info_set: HDEVINFO,
        device_interface_data: *const SP_DEVICE_INTERFACE_DATA,
        device_interface_detail_data: *mut SP_DEVICE_INTERFACE_DETAIL_DATA_W,
        device_interface_detail_data_size: DWORD,
        required_size: *mut DWORD,
        device_info_data: *mut std::ffi::c_void,
    ) -> BOOL;

    fn SetupDiDestroyDeviceInfoList(device_info_set: HDEVINFO) -> BOOL;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateFileW(
        file_name: *const u16,
        desired_access: DWORD,
        share_mode: DWORD,
        security_attributes: *mut std::ffi::c_void,
        creation_disposition: DWORD,
        flags_and_attributes: DWORD,
        template_file: HANDLE,
    ) -> HANDLE;

    fn CloseHandle(handle: HANDLE) -> BOOL;

    fn DeviceIoControl(
        device: HANDLE,
        io_control_code: DWORD,
        in_buffer: *const std::ffi::c_void,
        in_buffer_size: DWORD,
        out_buffer: *mut std::ffi::c_void,
        out_buffer_size: DWORD,
        bytes_returned: *mut DWORD,
        overlapped: *mut std::ffi::c_void,
    ) -> BOOL;

    fn GetLastError() -> DWORD;
}

/// Convert a Rust string to a null-terminated wide string
fn to_wide_string(s: &str) -> Vec<u16> {
    OsStr::new(s)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

/// EMI device handle wrapper
struct EMIDevice {
    handle: HANDLE,
    version: u16,
    channel_count: u32,
}

impl EMIDevice {
    /// Open an EMI device by path
    fn open(device_path: &[u16]) -> Result<Self> {
        unsafe {
            let handle = CreateFileW(
                device_path.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                null_mut(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                null_mut(),
            );

            if handle == INVALID_HANDLE_VALUE {
                return Err(Error::Permission(format!(
                    "Failed to open EMI device: error code {}",
                    GetLastError()
                )));
            }

            // Query EMI version
            let mut version: EMI_VERSION = zeroed();
            let mut bytes_returned: DWORD = 0;

            let result = DeviceIoControl(
                handle,
                IOCTL_EMI_GET_VERSION,
                null_mut(),
                0,
                &mut version as *mut _ as *mut std::ffi::c_void,
                size_of::<EMI_VERSION>() as DWORD,
                &mut bytes_returned,
                null_mut(),
            );

            if result == 0 {
                CloseHandle(handle);
                return Err(Error::Unsupported(format!(
                    "Failed to query EMI version: error code {}",
                    GetLastError()
                )));
            }

            // Determine channel count based on version
            let channel_count = if version.emi_version >= EMI_VERSION_V2 {
                // V2 supports multiple channels, query metadata
                Self::query_channel_count(handle)?
            } else {
                1 // V1 has single channel
            };

            Ok(Self {
                handle,
                version: version.emi_version,
                channel_count,
            })
        }
    }

    /// Query channel count from EMI V2 metadata
    fn query_channel_count(handle: HANDLE) -> Result<u32> {
        unsafe {
            let mut metadata_size: EMI_METADATA_SIZE = zeroed();
            let mut bytes_returned: DWORD = 0;

            let result = DeviceIoControl(
                handle,
                IOCTL_EMI_GET_METADATA_SIZE,
                null_mut(),
                0,
                &mut metadata_size as *mut _ as *mut std::ffi::c_void,
                size_of::<EMI_METADATA_SIZE>() as DWORD,
                &mut bytes_returned,
                null_mut(),
            );

            if result == 0 {
                // Fall back to 1 channel if metadata query fails
                return Ok(1);
            }

            // For V2, channel count is encoded in metadata
            // Simplified: assume 1 channel for now
            Ok(1)
        }
    }

    /// Read energy measurement (in picowatt-hours)
    fn read_measurement(&self) -> Result<(u64, u64)> {
        unsafe {
            let mut measurement: EMI_MEASUREMENT_DATA_V1 = zeroed();
            let mut bytes_returned: DWORD = 0;

            let result = DeviceIoControl(
                self.handle,
                IOCTL_EMI_GET_MEASUREMENT,
                null_mut(),
                0,
                &mut measurement as *mut _ as *mut std::ffi::c_void,
                size_of::<EMI_MEASUREMENT_DATA_V1>() as DWORD,
                &mut bytes_returned,
                null_mut(),
            );

            if result == 0 {
                return Err(Error::Unsupported(format!(
                    "Failed to read EMI measurement: error code {}",
                    GetLastError()
                )));
            }

            Ok((measurement.absolute_energy, measurement.absolute_time))
        }
    }
}

impl Drop for EMIDevice {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

// SAFETY: Handle is Send/Sync when properly synchronized
unsafe impl Send for EMIDevice {}
unsafe impl Sync for EMIDevice {}

/// Enumerate EMI devices on the system
fn enumerate_emi_devices() -> Result<Vec<Vec<u16>>> {
    let mut devices = Vec::new();

    unsafe {
        let device_info_set = SetupDiGetClassDevsW(
            &EMI_DEVICE_INTERFACE_GUID,
            null_mut(),
            null_mut(),
            DIGCF_PRESENT | DIGCF_DEVICEINTERFACE,
        );

        if device_info_set == INVALID_HANDLE_VALUE || device_info_set.is_null() {
            return Err(Error::Unsupported(
                "No EMI devices found on this system. EMI requires Windows 10+ with compatible hardware.".to_string(),
            ));
        }

        let mut member_index: DWORD = 0;
        loop {
            let mut interface_data: SP_DEVICE_INTERFACE_DATA = zeroed();
            interface_data.cb_size = size_of::<SP_DEVICE_INTERFACE_DATA>() as DWORD;

            let result = SetupDiEnumDeviceInterfaces(
                device_info_set,
                null_mut(),
                &EMI_DEVICE_INTERFACE_GUID,
                member_index,
                &mut interface_data,
            );

            if result == 0 {
                break; // No more devices
            }

            // Get required buffer size
            let mut required_size: DWORD = 0;
            SetupDiGetDeviceInterfaceDetailW(
                device_info_set,
                &interface_data,
                null_mut(),
                0,
                &mut required_size,
                null_mut(),
            );

            if required_size == 0 {
                member_index += 1;
                continue;
            }

            // Allocate buffer for device path
            let buffer_size = required_size as usize;
            let mut buffer: Vec<u8> = vec![0; buffer_size];
            let detail_data = buffer.as_mut_ptr() as *mut SP_DEVICE_INTERFACE_DETAIL_DATA_W;

            // Set cb_size to the fixed part of the structure
            // On 64-bit Windows, this is 8 bytes due to alignment
            // On 32-bit Windows, this is 6 bytes
            #[cfg(target_pointer_width = "64")]
            {
                (*detail_data).cb_size = 8;
            }
            #[cfg(target_pointer_width = "32")]
            {
                (*detail_data).cb_size = 6;
            }

            let result = SetupDiGetDeviceInterfaceDetailW(
                device_info_set,
                &interface_data,
                detail_data,
                required_size,
                null_mut(),
                null_mut(),
            );

            if result != 0 {
                // Extract device path (wide string after cb_size field)
                let path_ptr = std::ptr::addr_of!((*detail_data).device_path) as *const u16;
                let mut path = Vec::new();
                let mut i = 0;
                loop {
                    let c = *path_ptr.add(i);
                    if c == 0 {
                        path.push(0);
                        break;
                    }
                    path.push(c);
                    i += 1;
                    if i > 512 {
                        break; // Safety limit
                    }
                }
                devices.push(path);
            }

            member_index += 1;
        }

        SetupDiDestroyDeviceInfoList(device_info_set);
    }

    if devices.is_empty() {
        Err(Error::Unsupported(
            "No EMI devices found. Your hardware may not support Windows EMI energy metering."
                .to_string(),
        ))
    } else {
        Ok(devices)
    }
}

/// Energy reading state
struct EnergyState {
    device: EMIDevice,
    last_energy_pwh: u64,            // Last energy reading in picowatt-hours
    last_time_100ns: u64,            // Last time in 100ns intervals
    cumulative_energy_uj: AtomicU64, // Cumulative energy in microjoules
    start_time: Instant,
}

// SAFETY: State is protected by Mutex
unsafe impl Send for EnergyState {}
unsafe impl Sync for EnergyState {}

/// Windows energy reader using EMI
pub struct WindowsEnergyReader {
    state: Mutex<EnergyState>,
}

impl WindowsEnergyReader {
    /// Create a new Windows EMI energy reader
    pub fn new() -> Result<Self> {
        // Enumerate EMI devices
        let device_paths = enumerate_emi_devices()?;

        // Open the first available EMI device
        let device = EMIDevice::open(&device_paths[0])?;

        // Take initial measurement
        let (initial_energy, initial_time) = device.read_measurement()?;

        let state = EnergyState {
            device,
            last_energy_pwh: initial_energy,
            last_time_100ns: initial_time,
            cumulative_energy_uj: AtomicU64::new(0),
            start_time: Instant::now(),
        };

        Ok(Self {
            state: Mutex::new(state),
        })
    }

    /// Read current energy sample
    fn read_sample(&self) -> Result<EnergySample> {
        let mut state = self.state.lock().map_err(|_| {
            Error::Unsupported("Failed to acquire lock for energy reading".to_string())
        })?;

        let (current_energy_pwh, current_time_100ns) = state.device.read_measurement()?;

        // Calculate delta (handle counter wrap)
        let delta_energy_pwh = if current_energy_pwh >= state.last_energy_pwh {
            current_energy_pwh - state.last_energy_pwh
        } else {
            // Counter wrapped - assume small delta (this is rare)
            current_energy_pwh
        };

        let delta_time_100ns = if current_time_100ns >= state.last_time_100ns {
            current_time_100ns - state.last_time_100ns
        } else {
            current_time_100ns
        };

        // Convert picowatt-hours to microjoules
        // 1 pWh = 3.6e-9 J = 3.6e-3 uJ
        // More precisely: 1 pWh = 3600 pJ = 3600e-12 J = 3.6e-9 J = 0.0036 uJ
        let delta_energy_uj = (delta_energy_pwh as f64 * 0.0036) as u64;

        // Update cumulative energy
        state
            .cumulative_energy_uj
            .fetch_add(delta_energy_uj, Ordering::Relaxed);

        // Update last readings
        state.last_energy_pwh = current_energy_pwh;
        state.last_time_100ns = current_time_100ns;

        // Convert 100ns intervals to nanoseconds
        let duration_ns = delta_time_100ns * 100;

        Ok(EnergySample {
            energy_uj: delta_energy_uj,
            duration_ns,
        })
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
                "Cannot calculate power from zero-duration sample".to_string(),
            ));
        }

        // Calculate power: P = E / t
        let energy_j = sample.energy_uj as f64 / 1_000_000.0;
        let duration_s = sample.duration_ns as f64 / 1_000_000_000.0;

        Ok(energy_j / duration_s)
    }
}

/// Energy sample data
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergySample {
    /// Energy in microjoules
    pub energy_uj: u64,
    /// Sample duration in nanoseconds
    pub duration_ns: u64,
}

impl EnergySample {
    /// Get energy in joules
    pub fn energy_joules(&self) -> f64 {
        self.energy_uj as f64 / 1_000_000.0
    }

    /// Get average power in watts
    pub fn average_power_watts(&self) -> Option<f64> {
        if self.duration_ns == 0 {
            return None;
        }
        let energy_j = self.energy_joules();
        let duration_s = self.duration_ns as f64 / 1_000_000_000.0;
        Some(energy_j / duration_s)
    }
}

/// Windows thermal reader using WMI
pub struct WindowsThermalReader;

impl WindowsThermalReader {
    /// Create a new Windows thermal reader
    pub fn new() -> Result<Self> {
        // WMI thermal reading requires COM initialization and WMI queries
        // This is a simplified placeholder - full implementation would use
        // wmi crate or direct COM bindings
        Ok(Self)
    }

    /// Read CPU temperature in Celsius
    ///
    /// Note: Windows doesn't expose temperature through a standard API.
    /// This would need to use:
    /// - WMI MSAcpi_ThermalZoneTemperature (often returns 0 on modern systems)
    /// - Hardware-specific drivers (Intel/AMD)
    /// - Third-party libraries like LibreHardwareMonitor
    pub fn read_cpu_temperature(&self) -> Result<f64> {
        // WMI query would go here
        // For now, return unsupported as reliable cross-hardware temp reading
        // requires hardware-specific implementations
        Err(Error::Unsupported(
            "Temperature reading on Windows requires hardware-specific drivers".to_string(),
        ))
    }
}

/// High-level Windows energy monitor combining energy and thermal readings
pub struct WindowsEnergyMonitor {
    energy_reader: WindowsEnergyReader,
    thermal_reader: Option<WindowsThermalReader>,
}

impl WindowsEnergyMonitor {
    /// Create a new Windows energy monitor
    pub fn new() -> Result<Self> {
        let energy_reader = WindowsEnergyReader::new()?;
        let thermal_reader = WindowsThermalReader::new().ok(); // Thermal is optional

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Windows with EMI-compatible hardware
    fn test_enumerate_emi_devices() {
        let devices = enumerate_emi_devices();
        match devices {
            Ok(paths) => {
                println!("Found {} EMI device(s)", paths.len());
                for (i, path) in paths.iter().enumerate() {
                    let path_str: String = path
                        .iter()
                        .take_while(|&&c| c != 0)
                        .map(|&c| char::from_u32(c as u32).unwrap_or('?'))
                        .collect();
                    println!("  Device {}: {}", i, path_str);
                }
            }
            Err(e) => {
                println!(
                    "EMI enumeration failed (expected on non-EMI systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires Windows with EMI-compatible hardware
    fn test_energy_reader() {
        let reader = WindowsEnergyReader::new().expect("Failed to create energy reader");
        let energy = reader
            .read_cumulative_energy()
            .expect("Failed to read energy");
        println!("Cumulative energy: {:.6} J", energy);
    }

    #[test]
    #[ignore] // Requires Windows with EMI-compatible hardware
    fn test_power_reading() {
        let reader = WindowsEnergyReader::new().expect("Failed to create energy reader");

        // Take initial sample
        let _ = reader.read_sample();

        // Wait a bit
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Read power
        let power = reader.read_current_power().expect("Failed to read power");
        println!("Current power: {:.2} W", power);
        assert!(power > 0.0 && power < 500.0); // Reasonable range
    }

    #[test]
    #[ignore] // Requires Windows with EMI-compatible hardware
    fn test_full_monitor() {
        let monitor = WindowsEnergyMonitor::new().expect("Failed to create monitor");

        let energy = monitor.read_energy().expect("Failed to read energy");
        println!("Energy: {:.6} J", energy);

        std::thread::sleep(std::time::Duration::from_millis(100));

        let power = monitor.read_power().expect("Failed to read power");
        println!("Power: {:.2} W", power);

        // Temperature may not be available
        match monitor.read_temperature() {
            Ok(temp) => println!("Temperature: {:.1}C", temp),
            Err(e) => println!("Temperature not available: {}", e),
        }
    }
}
