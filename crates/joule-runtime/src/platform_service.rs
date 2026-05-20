//! Platform service integration — auto-start the daemon on login.
//!
//! Generates and installs:
//! - macOS: `~/Library/LaunchAgents/com.jouledb.daemon.plist`
//! - Linux: `~/.config/systemd/user/jouled.service`

use crate::RuntimeError;
use std::path::PathBuf;

/// Service identifier.
const LAUNCHD_LABEL: &str = "com.jouledb.daemon";
const SYSTEMD_UNIT: &str = "jouled.service";

/// Generate a macOS launchd plist for the daemon.
pub fn generate_launchd_plist(daemon_binary: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
        <string>daemon</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{home}/.jouledb/daemon.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{home}/.jouledb/daemon.stderr.log</string>
    <key>WorkingDirectory</key>
    <string>{home}</string>
</dict>
</plist>
"#,
        label = LAUNCHD_LABEL,
        binary = daemon_binary,
        home = home_dir_str(),
    )
}

/// Generate a Linux systemd user unit for the daemon.
pub fn generate_systemd_unit(daemon_binary: &str) -> String {
    format!(
        r#"[Unit]
Description=JouleDB Daemon — Universal Database Energy Layer
After=network.target

[Service]
Type=simple
ExecStart={binary} daemon start --foreground
Restart=always
RestartSec=5
WatchdogSec=30
StandardOutput=append:{home}/.jouledb/daemon.stdout.log
StandardError=append:{home}/.jouledb/daemon.stderr.log
WorkingDirectory={home}

[Install]
WantedBy=default.target
"#,
        binary = daemon_binary,
        home = home_dir_str(),
    )
}

fn home_dir_str() -> String {
    #[cfg(unix)]
    {
        std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
    }
    #[cfg(not(unix))]
    {
        std::env::var("USERPROFILE").unwrap_or_else(|_| "C:\\".to_string())
    }
}

/// Path where the launchd plist should be installed.
fn launchd_plist_path() -> PathBuf {
    let home = home_dir_str();
    PathBuf::from(home)
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{}.plist", LAUNCHD_LABEL))
}

/// Path where the systemd unit should be installed.
fn systemd_unit_path() -> PathBuf {
    let home = home_dir_str();
    PathBuf::from(home)
        .join(".config")
        .join("systemd")
        .join("user")
        .join(SYSTEMD_UNIT)
}

/// Find the daemon binary path (assumes `jouledb-cli` is on PATH or in current dir).
fn find_daemon_binary() -> Result<String, RuntimeError> {
    // Check if jouledb-cli is on PATH
    if let Ok(output) = std::process::Command::new("which")
        .arg("jouledb-cli")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }

    // Check current executable
    if let Ok(exe) = std::env::current_exe() {
        return Ok(exe.to_string_lossy().to_string());
    }

    Err(RuntimeError::ProcessError(
        "cannot find jouledb-cli binary".into(),
    ))
}

/// Install the daemon as an OS service (auto-start on login).
pub fn install_service() -> Result<String, RuntimeError> {
    let binary = find_daemon_binary()?;

    #[cfg(target_os = "macos")]
    {
        install_launchd(&binary)
    }
    #[cfg(target_os = "linux")]
    {
        install_systemd(&binary)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err(RuntimeError::ConfigError(
            "service installation not supported on this platform".into(),
        ))
    }
}

/// Uninstall the daemon OS service.
pub fn uninstall_service() -> Result<String, RuntimeError> {
    #[cfg(target_os = "macos")]
    {
        uninstall_launchd()
    }
    #[cfg(target_os = "linux")]
    {
        uninstall_systemd()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err(RuntimeError::ConfigError(
            "service uninstallation not supported on this platform".into(),
        ))
    }
}

#[cfg(target_os = "macos")]
fn install_launchd(binary: &str) -> Result<String, RuntimeError> {
    let plist_path = launchd_plist_path();
    let plist_content = generate_launchd_plist(binary);

    // Ensure parent directory exists
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&plist_path, &plist_content)?;

    // Load the service
    let status = std::process::Command::new("launchctl")
        .args(["load", "-w"])
        .arg(&plist_path)
        .status()?;

    if status.success() {
        Ok(format!("Installed: {}", plist_path.display()))
    } else {
        Err(RuntimeError::ProcessError("launchctl load failed".into()))
    }
}

#[cfg(target_os = "macos")]
fn uninstall_launchd() -> Result<String, RuntimeError> {
    let plist_path = launchd_plist_path();

    if plist_path.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w"])
            .arg(&plist_path)
            .status();
        std::fs::remove_file(&plist_path)?;
        Ok(format!("Uninstalled: {}", plist_path.display()))
    } else {
        Ok("Service was not installed".into())
    }
}

#[cfg(target_os = "linux")]
fn install_systemd(binary: &str) -> Result<String, RuntimeError> {
    let unit_path = systemd_unit_path();
    let unit_content = generate_systemd_unit(binary);

    if let Some(parent) = unit_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&unit_path, &unit_content)?;

    // Reload and enable
    let _ = std::process::Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status();

    let status = std::process::Command::new("systemctl")
        .args(["--user", "enable", SYSTEMD_UNIT])
        .status()?;

    if status.success() {
        Ok(format!("Installed and enabled: {}", unit_path.display()))
    } else {
        Err(RuntimeError::ProcessError("systemctl enable failed".into()))
    }
}

#[cfg(target_os = "linux")]
fn uninstall_systemd() -> Result<String, RuntimeError> {
    let unit_path = systemd_unit_path();

    if unit_path.exists() {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "stop", SYSTEMD_UNIT])
            .status();
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", SYSTEMD_UNIT])
            .status();
        std::fs::remove_file(&unit_path)?;
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status();
        Ok(format!("Uninstalled: {}", unit_path.display()))
    } else {
        Ok("Service was not installed".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launchd_plist_contains_expected_fields() {
        let plist = generate_launchd_plist("/usr/local/bin/jouledb-cli");
        assert!(plist.contains(LAUNCHD_LABEL));
        assert!(plist.contains("/usr/local/bin/jouledb-cli"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
        assert!(plist.contains("<key>KeepAlive</key>"));
        assert!(plist.contains("daemon.stdout.log"));
        assert!(plist.contains("daemon.stderr.log"));
        assert!(plist.contains("--foreground"));
    }

    #[test]
    fn test_systemd_unit_contains_expected_fields() {
        let unit = generate_systemd_unit("/usr/local/bin/jouledb-cli");
        assert!(unit.contains("/usr/local/bin/jouledb-cli"));
        assert!(unit.contains("Restart=always"));
        assert!(unit.contains("WatchdogSec=30"));
        assert!(unit.contains("RestartSec=5"));
        assert!(unit.contains("default.target"));
        assert!(unit.contains("--foreground"));
    }

    #[test]
    fn test_launchd_plist_is_valid_xml() {
        let plist = generate_launchd_plist("/bin/test");
        assert!(plist.starts_with("<?xml"));
        assert!(plist.contains("</plist>"));
    }

    #[test]
    fn test_systemd_unit_has_sections() {
        let unit = generate_systemd_unit("/bin/test");
        assert!(unit.contains("[Unit]"));
        assert!(unit.contains("[Service]"));
        assert!(unit.contains("[Install]"));
    }
}
