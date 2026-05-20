//! I2C bus simulation: master/slave model, address space, read/write
//! transactions, register-based access, bus arbitration, clock stretching
//! concept, multi-slave bus, and transaction logging.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

// ── Types ──

/// I2C address (7-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct I2cAddress(pub u8);

impl I2cAddress {
    /// Create a 7-bit I2C address. Returns None if > 127 or reserved (0..=7, 120..=127).
    pub fn new(addr: u8) -> Option<Self> {
        if addr > 127 {
            return None;
        }
        // 0x00..=0x07 and 0x78..=0x7F are reserved in 7-bit addressing.
        if addr <= 7 || addr >= 120 {
            return None;
        }
        Some(Self(addr))
    }

    /// Create without validation (for testing).
    pub fn unchecked(addr: u8) -> Self {
        Self(addr)
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for I2cAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:02X}", self.0)
    }
}

/// Direction of an I2C transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    Write,
    Read,
}

impl TransferDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Write => "write",
            Self::Read => "read",
        }
    }
}

/// Result of a bus transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckNack {
    Ack,
    Nack,
}

/// A logged I2C transaction.
#[derive(Debug, Clone)]
pub struct TransactionLog {
    pub address: I2cAddress,
    pub direction: TransferDirection,
    pub register: Option<u8>,
    pub data: Vec<u8>,
    pub result: TransactionResult,
    pub timestamp: DateTime<Utc>,
}

/// Outcome of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionResult {
    Success,
    Nack,
    BusError,
    ArbitrationLost,
    Timeout,
}

impl TransactionResult {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Nack => "nack",
            Self::BusError => "bus_error",
            Self::ArbitrationLost => "arbitration_lost",
            Self::Timeout => "timeout",
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Success)
    }
}

/// I2C bus errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum I2cError {
    NoDeviceAtAddress(u8),
    InvalidRegister { address: u8, register: u8 },
    BusBusy,
    ArbitrationLost,
    Nack(u8),
    ReadOnlyRegister { address: u8, register: u8 },
    DataTooLong { max: usize, got: usize },
    ClockStretch,
    BusError(String),
}

impl std::fmt::Display for I2cError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDeviceAtAddress(a) => write!(f, "no device at address 0x{:02X}", a),
            Self::InvalidRegister { address, register } => {
                write!(f, "invalid register 0x{:02X} on device 0x{:02X}", register, address)
            }
            Self::BusBusy => write!(f, "bus is busy"),
            Self::ArbitrationLost => write!(f, "arbitration lost"),
            Self::Nack(a) => write!(f, "NACK from address 0x{:02X}", a),
            Self::ReadOnlyRegister { address, register } => {
                write!(f, "register 0x{:02X} on device 0x{:02X} is read-only", register, address)
            }
            Self::DataTooLong { max, got } => {
                write!(f, "data too long: max {} bytes, got {}", max, got)
            }
            Self::ClockStretch => write!(f, "clock stretching timeout"),
            Self::BusError(msg) => write!(f, "bus error: {}", msg),
        }
    }
}

impl std::error::Error for I2cError {}

/// Register access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterAccess {
    ReadWrite,
    ReadOnly,
    WriteOnly,
}

/// A single register in a slave device.
#[derive(Debug, Clone)]
pub struct Register {
    pub address: u8,
    pub value: u8,
    pub access: RegisterAccess,
    pub name: String,
}

impl Register {
    pub fn new(address: u8, name: &str, access: RegisterAccess) -> Self {
        Self {
            address,
            value: 0,
            access,
            name: name.to_string(),
        }
    }

    pub fn with_default(mut self, value: u8) -> Self {
        self.value = value;
        self
    }
}

/// A simulated I2C slave device.
#[derive(Debug, Clone)]
pub struct I2cSlave {
    pub address: I2cAddress,
    pub name: String,
    registers: HashMap<u8, Register>,
    /// Whether this device is currently clock-stretching.
    clock_stretching: bool,
    online: bool,
}

impl I2cSlave {
    pub fn new(address: I2cAddress, name: &str) -> Self {
        Self {
            address,
            name: name.to_string(),
            registers: HashMap::new(),
            clock_stretching: false,
            online: true,
        }
    }

    pub fn add_register(&mut self, reg: Register) {
        self.registers.insert(reg.address, reg);
    }

    pub fn set_clock_stretching(&mut self, stretching: bool) {
        self.clock_stretching = stretching;
    }

    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }

    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    fn read_register(&self, reg_addr: u8) -> Result<u8, I2cError> {
        let reg = self.registers.get(&reg_addr).ok_or(I2cError::InvalidRegister {
            address: self.address.0,
            register: reg_addr,
        })?;
        if reg.access == RegisterAccess::WriteOnly {
            return Err(I2cError::InvalidRegister {
                address: self.address.0,
                register: reg_addr,
            });
        }
        Ok(reg.value)
    }

    fn write_register(&mut self, reg_addr: u8, value: u8) -> Result<(), I2cError> {
        let reg = self.registers.get_mut(&reg_addr).ok_or(I2cError::InvalidRegister {
            address: self.address.0,
            register: reg_addr,
        })?;
        if reg.access == RegisterAccess::ReadOnly {
            return Err(I2cError::ReadOnlyRegister {
                address: self.address.0,
                register: reg_addr,
            });
        }
        reg.value = value;
        Ok(())
    }
}

// ── I2C Bus ──

/// Simulated I2C bus with master/slave communication.
pub struct I2cBus {
    slaves: HashMap<u8, I2cSlave>,
    log: Vec<TransactionLog>,
    max_log: usize,
    bus_busy: bool,
    /// Simulate arbitration: if multiple masters try at once (for testing).
    arbitration_holder: Option<String>,
}

impl I2cBus {
    pub fn new() -> Self {
        Self {
            slaves: HashMap::new(),
            log: Vec::new(),
            max_log: 1000,
            bus_busy: false,
            arbitration_holder: None,
        }
    }

    pub fn set_max_log(&mut self, max: usize) {
        self.max_log = max;
    }

    /// Attach a slave device to the bus.
    pub fn attach(&mut self, slave: I2cSlave) -> Result<(), I2cError> {
        let addr = slave.address.0;
        if self.slaves.contains_key(&addr) {
            return Err(I2cError::BusError(format!("address 0x{:02X} already occupied", addr)));
        }
        self.slaves.insert(addr, slave);
        Ok(())
    }

    /// Detach a slave from the bus.
    pub fn detach(&mut self, address: I2cAddress) -> Option<I2cSlave> {
        self.slaves.remove(&address.0)
    }

    /// Write data to a slave register.
    pub fn write_register(&mut self, address: I2cAddress, register: u8, data: &[u8]) -> Result<(), I2cError> {
        if self.bus_busy {
            return Err(I2cError::BusBusy);
        }

        self.bus_busy = true;

        let addr_val = address.0;
        let result = self.perform_write(addr_val, register, data);

        let tx_result = match &result {
            Ok(()) => TransactionResult::Success,
            Err(I2cError::NoDeviceAtAddress(_)) => TransactionResult::Nack,
            Err(I2cError::ArbitrationLost) => TransactionResult::ArbitrationLost,
            Err(I2cError::ClockStretch) => TransactionResult::Timeout,
            Err(_) => TransactionResult::BusError,
        };

        self.record_log(address, TransferDirection::Write, Some(register), data.to_vec(), tx_result);
        self.bus_busy = false;
        result
    }

    fn perform_write(&mut self, addr: u8, register: u8, data: &[u8]) -> Result<(), I2cError> {
        let slave = self.slaves.get_mut(&addr)
            .ok_or(I2cError::NoDeviceAtAddress(addr))?;

        if !slave.online {
            return Err(I2cError::Nack(addr));
        }

        if slave.clock_stretching {
            return Err(I2cError::ClockStretch);
        }

        if data.is_empty() {
            return slave.write_register(register, 0);
        }

        // Write sequential registers starting from `register`.
        for (i, byte) in data.iter().enumerate() {
            let reg_addr = register.wrapping_add(i as u8);
            slave.write_register(reg_addr, *byte)?;
        }

        Ok(())
    }

    /// Read from a slave register.
    pub fn read_register(&mut self, address: I2cAddress, register: u8, length: usize) -> Result<Vec<u8>, I2cError> {
        if self.bus_busy {
            return Err(I2cError::BusBusy);
        }

        self.bus_busy = true;

        let addr_val = address.0;
        let result = self.perform_read(addr_val, register, length);

        let (tx_result, data) = match &result {
            Ok(d) => (TransactionResult::Success, d.clone()),
            Err(I2cError::NoDeviceAtAddress(_)) => (TransactionResult::Nack, Vec::new()),
            Err(I2cError::ClockStretch) => (TransactionResult::Timeout, Vec::new()),
            Err(_) => (TransactionResult::BusError, Vec::new()),
        };

        self.record_log(address, TransferDirection::Read, Some(register), data, tx_result);
        self.bus_busy = false;
        result
    }

    fn perform_read(&self, addr: u8, register: u8, length: usize) -> Result<Vec<u8>, I2cError> {
        let slave = self.slaves.get(&addr)
            .ok_or(I2cError::NoDeviceAtAddress(addr))?;

        if !slave.online {
            return Err(I2cError::Nack(addr));
        }

        if slave.clock_stretching {
            return Err(I2cError::ClockStretch);
        }

        let mut data = Vec::with_capacity(length);
        for i in 0..length {
            let reg_addr = register.wrapping_add(i as u8);
            data.push(slave.read_register(reg_addr)?);
        }

        Ok(data)
    }

    /// Scan the bus for devices that respond.
    pub fn scan(&mut self) -> Vec<I2cAddress> {
        let mut found = Vec::new();
        for (addr, slave) in &self.slaves {
            if slave.online {
                found.push(I2cAddress(*addr));
            }
        }
        found.sort_by_key(|a| a.0);
        found
    }

    /// Simulate bus arbitration: acquire the bus for a named master.
    pub fn acquire(&mut self, master_name: &str) -> Result<(), I2cError> {
        if self.arbitration_holder.is_some() {
            return Err(I2cError::ArbitrationLost);
        }
        self.arbitration_holder = Some(master_name.to_string());
        Ok(())
    }

    /// Release bus arbitration.
    pub fn release(&mut self) {
        self.arbitration_holder = None;
    }

    /// Get the transaction log.
    pub fn log(&self) -> &[TransactionLog] {
        &self.log
    }

    /// Transaction count.
    pub fn transaction_count(&self) -> usize {
        self.log.len()
    }

    /// Count of successful transactions.
    pub fn success_count(&self) -> usize {
        self.log.iter().filter(|t| t.result.is_ok()).count()
    }

    /// Count of failed transactions.
    pub fn error_count(&self) -> usize {
        self.log.iter().filter(|t| !t.result.is_ok()).count()
    }

    pub fn slave_count(&self) -> usize {
        self.slaves.len()
    }

    /// Clear the transaction log.
    pub fn clear_log(&mut self) {
        self.log.clear();
    }

    /// Get a reference to a slave for inspection.
    pub fn slave(&self, address: I2cAddress) -> Option<&I2cSlave> {
        self.slaves.get(&address.0)
    }

    fn record_log(&mut self, address: I2cAddress, direction: TransferDirection, register: Option<u8>, data: Vec<u8>, result: TransactionResult) {
        if self.log.len() >= self.max_log {
            self.log.remove(0);
        }
        self.log.push(TransactionLog {
            address,
            direction,
            register,
            data,
            result,
            timestamp: Utc::now(),
        });
    }
}

impl Default for I2cBus {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_sensor(addr: u8) -> I2cSlave {
        let address = I2cAddress::unchecked(addr);
        let mut slave = I2cSlave::new(address, "TMP102");
        slave.add_register(Register::new(0x00, "temperature", RegisterAccess::ReadOnly).with_default(25));
        slave.add_register(Register::new(0x01, "config", RegisterAccess::ReadWrite).with_default(0x60));
        slave.add_register(Register::new(0x02, "t_low", RegisterAccess::ReadWrite).with_default(75));
        slave.add_register(Register::new(0x03, "t_high", RegisterAccess::ReadWrite).with_default(80));
        slave
    }

    #[test]
    fn address_validation() {
        assert!(I2cAddress::new(0x48).is_some());
        assert!(I2cAddress::new(5).is_none());     // reserved
        assert!(I2cAddress::new(125).is_none());    // reserved
        assert!(I2cAddress::new(200).is_none());    // > 127
    }

    #[test]
    fn address_display() {
        let addr = I2cAddress::unchecked(0x48);
        assert_eq!(format!("{}", addr), "0x48");
    }

    #[test]
    fn attach_and_scan() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        bus.attach(make_temp_sensor(0x49)).unwrap();
        let found = bus.scan();
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn read_register() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        let data = bus.read_register(addr, 0x00, 1).unwrap();
        assert_eq!(data, vec![25]);
    }

    #[test]
    fn write_register() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        bus.write_register(addr, 0x01, &[0xFF]).unwrap();
        let data = bus.read_register(addr, 0x01, 1).unwrap();
        assert_eq!(data, vec![0xFF]);
    }

    #[test]
    fn write_to_read_only_register() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        let result = bus.write_register(addr, 0x00, &[42]);
        assert!(result.is_err());
    }

    #[test]
    fn read_invalid_register() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        let result = bus.read_register(addr, 0xFF, 1);
        assert!(result.is_err());
    }

    #[test]
    fn no_device_at_address() {
        let mut bus = I2cBus::new();
        let addr = I2cAddress::unchecked(0x48);
        let result = bus.read_register(addr, 0x00, 1);
        assert!(result.is_err());
    }

    #[test]
    fn multi_byte_read() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        let data = bus.read_register(addr, 0x00, 4).unwrap();
        assert_eq!(data.len(), 4);
        assert_eq!(data[0], 25);  // reg 0x00
        assert_eq!(data[1], 0x60); // reg 0x01
    }

    #[test]
    fn multi_byte_write() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        // Write to config (0x01) and t_low (0x02).
        bus.write_register(addr, 0x01, &[0xAA, 0xBB]).unwrap();
        let config = bus.read_register(addr, 0x01, 1).unwrap();
        let t_low = bus.read_register(addr, 0x02, 1).unwrap();
        assert_eq!(config, vec![0xAA]);
        assert_eq!(t_low, vec![0xBB]);
    }

    #[test]
    fn transaction_logging() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        bus.read_register(addr, 0x00, 1).unwrap();
        bus.write_register(addr, 0x01, &[0xFF]).unwrap();
        assert_eq!(bus.transaction_count(), 2);
        assert_eq!(bus.success_count(), 2);
    }

    #[test]
    fn error_count() {
        let mut bus = I2cBus::new();
        let addr = I2cAddress::unchecked(0x48);
        let _ = bus.read_register(addr, 0x00, 1);
        assert_eq!(bus.error_count(), 1);
    }

    #[test]
    fn bus_arbitration() {
        let mut bus = I2cBus::new();
        bus.acquire("master1").unwrap();
        assert!(bus.acquire("master2").is_err());
        bus.release();
        bus.acquire("master2").unwrap();
    }

    #[test]
    fn clock_stretching() {
        let mut bus = I2cBus::new();
        let mut sensor = make_temp_sensor(0x48);
        sensor.set_clock_stretching(true);
        bus.attach(sensor).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        let result = bus.read_register(addr, 0x00, 1);
        assert!(result.is_err());
    }

    #[test]
    fn offline_device() {
        let mut bus = I2cBus::new();
        let mut sensor = make_temp_sensor(0x48);
        sensor.set_online(false);
        bus.attach(sensor).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        let result = bus.read_register(addr, 0x00, 1);
        assert!(result.is_err());
        // Should not appear in scan.
        assert!(bus.scan().is_empty());
    }

    #[test]
    fn detach_slave() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let removed = bus.detach(I2cAddress::unchecked(0x48));
        assert!(removed.is_some());
        assert_eq!(bus.slave_count(), 0);
    }

    #[test]
    fn duplicate_address_fails() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let result = bus.attach(make_temp_sensor(0x48));
        assert!(result.is_err());
    }

    #[test]
    fn clear_log() {
        let mut bus = I2cBus::new();
        bus.attach(make_temp_sensor(0x48)).unwrap();
        let addr = I2cAddress::unchecked(0x48);
        bus.read_register(addr, 0x00, 1).unwrap();
        bus.clear_log();
        assert_eq!(bus.transaction_count(), 0);
    }

    #[test]
    fn register_default_value() {
        let reg = Register::new(0x00, "test", RegisterAccess::ReadWrite).with_default(42);
        assert_eq!(reg.value, 42);
    }

    #[test]
    fn transaction_result_is_ok() {
        assert!(TransactionResult::Success.is_ok());
        assert!(!TransactionResult::Nack.is_ok());
        assert!(!TransactionResult::BusError.is_ok());
    }
}
