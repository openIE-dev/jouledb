//! NVIDIA GPU energy through NVML, loaded at run time with `libloading`:
//! no link-time dependency, so builds and hosts without NVIDIA are
//! unaffected (the loader just reports why it found nothing).
//!
//! Library search: `JOULE_NVML_LIB`, then the sonames through the dynamic
//! loader, then the usual driver/toolkit/WSL directories (CUDA `stubs/`
//! libraries are skipped: they load but cannot initialise).
//!
//! Readings: `nvmlDeviceGetTotalEnergyConsumption` (millijoule counter)
//! where supported, else `nvmlDeviceGetPowerUsage` (milliwatts) sampled at
//! the job's ends. Every reading is sanity-checked against the enforced
//! power limit; absurd values (e.g. a 590 W instantaneous draw on a 35 W
//! laptop GPU) are rejected and the reason is reported.

use std::ffi::{c_char, c_void};
use std::path::PathBuf;

/// One GPU's NVML readings. Implemented by the real library and by fakes.
pub trait GpuEnergy: Send + Sync {
    /// NVML device name, e.g. `NVIDIA GeForce RTX 4050 Laptop GPU`.
    fn name(&self) -> String;
    /// PCI bus id, e.g. `00000000:01:00.0`.
    fn bus_id(&self) -> String;
    /// Library path that was loaded (or `fake`).
    fn library(&self) -> String;
    /// Total energy since driver load, millijoules.
    fn energy_mj(&self) -> Option<u64>;
    /// Current power draw, milliwatts.
    fn power_mw(&self) -> Option<u32>;
    /// Enforced power limit, milliwatts.
    fn limit_mw(&self) -> Option<u32>;
}

/// A reading of the GPU rail at one instant.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuSnapshot {
    pub energy_mj: Option<u64>,
    pub power_mw: Option<u32>,
}

pub fn snapshot(gpu: &dyn GpuEnergy) -> GpuSnapshot {
    GpuSnapshot { energy_mj: gpu.energy_mj(), power_mw: gpu.power_mw() }
}

/// GPU energy over a window.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuJoules {
    pub joules: f64,
    /// `nvml-energy` (counter) or `nvml-power` (sampled draw).
    pub source: &'static str,
    /// Why a reading was rejected or degraded, if any.
    pub note: Option<String>,
}

/// Largest believable draw: twice the enforced limit (or 1 kW unknown).
fn ceiling_w(limit_mw: Option<u32>) -> f64 {
    limit_mw.map(|l| 2.0 * f64::from(l) / 1000.0).filter(|w| *w > 0.0).unwrap_or(1000.0)
}

/// Sane power sample in watts, or the reason it was rejected.
pub fn sane_power_w(mw: Option<u32>, limit_mw: Option<u32>) -> Result<f64, String> {
    let w = f64::from(mw.ok_or("power usage unsupported")?) / 1000.0;
    let max = ceiling_w(limit_mw);
    if w > max {
        return Err(format!("rejected NVML power {w:.1} W > 2x enforced limit ({:.0} W)", max / 2.0));
    }
    Ok(w)
}

/// Energy of the GPU between two snapshots `seconds` apart. Prefers the
/// energy counter; falls back to the mean of the sane power samples.
/// `None` only when nothing usable was read.
pub fn between(a: GpuSnapshot, b: GpuSnapshot, seconds: f64, limit_mw: Option<u32>) -> Option<GpuJoules> {
    let max = ceiling_w(limit_mw);
    let mut note = None;
    if let (Some(x), Some(y)) = (a.energy_mj, b.energy_mj) {
        if y < x {
            note = Some(format!("rejected NVML energy counter: went backwards ({x} -> {y} mJ)"));
        } else {
            let j = (y - x) as f64 / 1000.0;
            // Counter updates are coarse: a short job may see one update
            // that covers more than its own window; allow 50 ms of slack.
            if j > 0.0 && j <= max * (seconds + 0.05) {
                return Some(GpuJoules { joules: j, source: "nvml-energy", note: None });
            }
            note = Some(if j == 0.0 {
                format!("NVML energy counter did not advance in {:.1} ms (coarse updates)", seconds * 1e3)
            } else {
                format!(
                    "rejected NVML energy step {j:.3} J over {:.1} ms (> {:.0} W ceiling; coarse counter update)",
                    seconds * 1e3,
                    max
                )
            });
        }
    }
    let pa = sane_power_w(a.power_mw, limit_mw);
    let pb = sane_power_w(b.power_mw, limit_mw);
    let samples: Vec<f64> = [&pa, &pb].iter().filter_map(|r| r.as_ref().ok().copied()).collect();
    let rejected: Vec<String> = [pa, pb].into_iter().filter_map(|r| r.err()).collect();
    let note: Option<String> = match (note, rejected.first()) {
        (Some(n), Some(r)) => Some(format!("{n}; {r}")),
        (Some(n), None) => Some(n),
        (None, Some(r)) => Some(r.clone()),
        (None, None) => None,
    };
    if samples.is_empty() {
        // A counter that did not advance is still a (zero) reading.
        if let (Some(x), Some(y)) = (a.energy_mj, b.energy_mj) {
            if x == y {
                return Some(GpuJoules { joules: 0.0, source: "nvml-energy", note });
            }
        }
        return None;
    }
    let w = samples.iter().sum::<f64>() / samples.len() as f64;
    Some(GpuJoules { joules: w * seconds, source: "nvml-power", note })
}

// ── Real NVML ─────────────────────────────────────────────────────────

type Device = *mut c_void;
type FnInit = unsafe extern "C" fn() -> u32;
type FnCount = unsafe extern "C" fn(*mut u32) -> u32;
type FnHandle = unsafe extern "C" fn(u32, *mut Device) -> u32;
type FnName = unsafe extern "C" fn(Device, *mut c_char, u32) -> u32;
type FnU64 = unsafe extern "C" fn(Device, *mut u64) -> u32;
type FnU32 = unsafe extern "C" fn(Device, *mut u32) -> u32;
type FnPci = unsafe extern "C" fn(Device, *mut PciInfo) -> u32;

/// `nvmlPciInfo_t` (v3).
#[repr(C)]
pub struct PciInfo {
    bus_id_legacy: [c_char; 16],
    domain: u32,
    bus: u32,
    device: u32,
    /// `(device id << 16) | vendor id`.
    pci_device_id: u32,
    pci_subsystem_id: u32,
    bus_id: [c_char; 32],
}

/// A loaded NVML device.
pub struct Nvml {
    _lib: libloading::Library,
    dev: Device,
    name: String,
    bus_id: String,
    path: String,
    energy: Option<FnU64>,
    power: Option<FnU32>,
    limit: Option<FnU32>,
}

// SAFETY: NVML is thread safe; the handle is an opaque pointer owned by
// the library, which lives as long as this struct.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

/// Candidate library paths, in search order.
pub fn candidate_paths() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("JOULE_NVML_LIB") {
        out.push(PathBuf::from(p));
    }
    #[cfg(target_os = "windows")]
    {
        out.push("nvml.dll".into());
        out.push(r"C:\Windows\System32\nvml.dll".into());
        out.push(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll".into());
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Through the dynamic loader (ldconfig cache, LD_LIBRARY_PATH).
        out.push("libnvidia-ml.so.1".into());
        out.push("libnvidia-ml.so".into());
        let dirs = [
            "/usr/lib/x86_64-linux-gnu",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/usr/lib/wsl/lib",
            "/usr/lib/nvidia",
            "/usr/local/nvidia/lib64",
            "/run/opengl-driver/lib",
        ];
        for d in dirs {
            out.push(PathBuf::from(d).join("libnvidia-ml.so.1"));
        }
        // CUDA toolkits, /opt and snaps; versioned names too.
        let globs = ["/usr/local", "/opt", "/snap"];
        for root in globs {
            if let Ok(entries) = std::fs::read_dir(root) {
                for e in entries.flatten() {
                    for sub in ["lib64", "lib", "targets/x86_64-linux/lib", "current/usr/lib/x86_64-linux-gnu"] {
                        out.push(e.path().join(sub).join("libnvidia-ml.so.1"));
                        out.push(e.path().join(sub).join("libnvidia-ml.so"));
                    }
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir("/usr/lib/x86_64-linux-gnu") {
            for e in entries.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("libnvidia-ml.so.") && n != "libnvidia-ml.so.1" {
                    out.push(e.path());
                }
            }
        }
    }
    out.retain(|p| !p.to_string_lossy().contains("/stubs/"));
    out.dedup();
    out
}

fn cstr(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl Nvml {
    /// Load NVML and pick the device matching the wgpu adapter
    /// (`(name, vendor, device id)`): by PCI device id, then by name, then
    /// the only device. The error lists every path tried and why.
    pub fn open(adapter: Option<(String, u32, u32)>) -> Result<Nvml, String> {
        let mut tried = Vec::new();
        for path in candidate_paths() {
            let explicit = path.components().count() > 1;
            if explicit && !path.exists() {
                continue;
            }
            match Self::open_path(&path, adapter.as_ref()) {
                Ok(n) => return Ok(n),
                Err(e) => tried.push(format!("{}: {e}", path.display())),
            }
        }
        Err(if tried.is_empty() { "no libnvidia-ml found".into() } else { tried.join("; ") })
    }

    fn open_path(path: &std::path::Path, adapter: Option<&(String, u32, u32)>) -> Result<Nvml, String> {
        // SAFETY: loading NVML runs its (trusted, driver-provided) initialisers.
        let lib = unsafe { libloading::Library::new(path) }.map_err(|e| e.to_string())?;
        // SAFETY: symbol types match the NVML C API.
        unsafe {
            let sym = |name: &[u8]| -> Result<*mut c_void, String> {
                lib.get::<*mut c_void>(name)
                    .map(|s| *s)
                    .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(&name[..name.len() - 1])))
            };
            let init: FnInit = std::mem::transmute(sym(b"nvmlInit_v2\0")?);
            let rc = init();
            if rc != 0 {
                return Err(format!("nvmlInit_v2 returned {rc}"));
            }
            let count: FnCount = std::mem::transmute(sym(b"nvmlDeviceGetCount_v2\0")?);
            let handle: FnHandle = std::mem::transmute(sym(b"nvmlDeviceGetHandleByIndex_v2\0")?);
            let name_fn: FnName = std::mem::transmute(sym(b"nvmlDeviceGetName\0")?);
            let pci_fn: Option<FnPci> = sym(b"nvmlDeviceGetPciInfo_v3\0").ok().map(|p| std::mem::transmute(p));
            let mut n = 0u32;
            if count(&mut n) != 0 || n == 0 {
                return Err("no NVML devices".into());
            }
            let mut devices = Vec::new();
            for i in 0..n {
                let mut dev: Device = std::ptr::null_mut();
                if handle(i, &mut dev) != 0 {
                    continue;
                }
                let mut buf = [0 as c_char; 96];
                let name = if name_fn(dev, buf.as_mut_ptr(), buf.len() as u32) == 0 { cstr(&buf) } else { String::new() };
                let (bus, pci_dev) = match pci_fn {
                    Some(f) => {
                        let mut info: PciInfo = std::mem::zeroed();
                        if f(dev, &mut info) == 0 { (cstr(&info.bus_id), info.pci_device_id) } else { (String::new(), 0) }
                    }
                    None => (String::new(), 0),
                };
                devices.push((dev, name, bus, pci_dev));
            }
            let pick = match adapter {
                Some((aname, vendor, device)) => devices
                    .iter()
                    .position(|d| *vendor == 0x10de && d.3 >> 16 == *device && d.3 & 0xffff == 0x10de)
                    .or_else(|| devices.iter().position(|d| !d.1.is_empty() && aname.contains(d.1.as_str())))
                    .or_else(|| (devices.len() == 1).then_some(0)),
                None => (devices.len() == 1).then_some(0),
            }
            .ok_or_else(|| format!("no NVML device matches the wgpu adapter ({} devices)", devices.len()))?;
            let (dev, name, bus_id, _) = devices.swap_remove(pick);
            let energy = sym(b"nvmlDeviceGetTotalEnergyConsumption\0").ok().map(|p| std::mem::transmute::<_, FnU64>(p));
            let power = sym(b"nvmlDeviceGetPowerUsage\0").ok().map(|p| std::mem::transmute::<_, FnU32>(p));
            let limit = sym(b"nvmlDeviceGetEnforcedPowerLimit\0").ok().map(|p| std::mem::transmute::<_, FnU32>(p));
            Ok(Nvml { dev, name, bus_id, path: path.display().to_string(), energy, power, limit, _lib: lib })
        }
    }
}

impl GpuEnergy for Nvml {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn bus_id(&self) -> String {
        self.bus_id.clone()
    }
    fn library(&self) -> String {
        self.path.clone()
    }
    fn energy_mj(&self) -> Option<u64> {
        let f = self.energy?;
        let mut v = 0u64;
        // SAFETY: valid device handle and out-pointer.
        (unsafe { f(self.dev, &mut v) } == 0).then_some(v)
    }
    fn power_mw(&self) -> Option<u32> {
        let f = self.power?;
        let mut v = 0u32;
        // SAFETY: as above.
        (unsafe { f(self.dev, &mut v) } == 0).then_some(v)
    }
    fn limit_mw(&self) -> Option<u32> {
        let f = self.limit?;
        let mut v = 0u32;
        // SAFETY: as above.
        (unsafe { f(self.dev, &mut v) } == 0).then_some(v)
    }
}

/// A scripted NVML for tests: the energy counter advances `watts` of real
/// time; the power reading is fixed (e.g. absurd).
pub struct FakeNvml {
    pub start: std::time::Instant,
    pub watts: f64,
    pub power_mw: Option<u32>,
    pub limit_mw: Option<u32>,
    pub counter: bool,
}

impl GpuEnergy for FakeNvml {
    fn name(&self) -> String {
        "Fake RTX".into()
    }
    fn bus_id(&self) -> String {
        "00000000:01:00.0".into()
    }
    fn library(&self) -> String {
        "fake".into()
    }
    fn energy_mj(&self) -> Option<u64> {
        self.counter.then(|| (self.start.elapsed().as_secs_f64() * self.watts * 1000.0) as u64)
    }
    fn power_mw(&self) -> Option<u32> {
        self.power_mw
    }
    fn limit_mw(&self) -> Option<u32> {
        self.limit_mw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_is_preferred_and_absurd_power_is_rejected_with_reason() {
        let limit = Some(35_000);
        let a = GpuSnapshot { energy_mj: Some(1_000), power_mw: Some(590_010) };
        let b = GpuSnapshot { energy_mj: Some(1_120), power_mw: Some(590_010) };
        let g = between(a, b, 0.010, limit).expect("reading");
        assert_eq!((g.source, g.joules), ("nvml-energy", 0.12));
        // Counter not advanced in a short job: the sane power sample is used.
        let c = GpuSnapshot { energy_mj: Some(1_000), power_mw: Some(10_000) };
        let g = between(c, c, 0.010, limit).expect("reading");
        assert_eq!(g.source, "nvml-power");
        assert!((g.joules - 0.1).abs() < 1e-9 && g.note.unwrap_or_default().contains("did not advance"));
        // No counter: both power samples are absurd -> nothing usable.
        let a = GpuSnapshot { energy_mj: None, power_mw: Some(590_010) };
        assert!(between(a, a, 0.010, limit).is_none());
        assert!(sane_power_w(Some(590_010), limit).unwrap_err().contains("590.0 W"));
        // One sane sample is used, the other rejected and reported.
        let b = GpuSnapshot { energy_mj: None, power_mw: Some(12_000) };
        let g = between(a, b, 0.5, limit).expect("reading");
        assert_eq!(g.source, "nvml-power");
        assert!((g.joules - 6.0).abs() < 1e-9);
        assert!(g.note.as_deref().unwrap_or("").contains("rejected NVML power"));
    }

    #[test]
    fn counter_sanity_and_backwards() {
        let limit = Some(35_000);
        // 10 J in 10 ms is 1 kW: rejected, falls back to sane power.
        let a = GpuSnapshot { energy_mj: Some(0), power_mw: Some(8_000) };
        let b = GpuSnapshot { energy_mj: Some(10_000), power_mw: Some(8_000) };
        let g = between(a, b, 0.010, limit).expect("reading");
        assert_eq!(g.source, "nvml-power");
        assert!(g.note.as_deref().unwrap_or("").contains("rejected NVML energy"));
        let b = GpuSnapshot { energy_mj: Some(0), power_mw: Some(8_000) };
        let a2 = GpuSnapshot { energy_mj: Some(5), power_mw: Some(8_000) };
        let g = between(a2, b, 0.010, limit).expect("reading");
        assert!(g.note.as_deref().unwrap_or("").contains("backwards"));
    }

    #[test]
    fn search_skips_stubs_and_honours_env() {
        let paths = candidate_paths();
        assert!(paths.iter().all(|p| !p.to_string_lossy().contains("/stubs/")));
        #[cfg(not(target_os = "windows"))]
        assert!(paths.iter().any(|p| p.ends_with("libnvidia-ml.so.1")));
    }

    #[test]
    fn missing_nvml_is_an_error_not_a_panic() {
        // The box has no NVIDIA driver: a clear error, never a crash.
        if let Err(e) = Nvml::open(None) {
            assert!(!e.is_empty());
        }
    }
}
