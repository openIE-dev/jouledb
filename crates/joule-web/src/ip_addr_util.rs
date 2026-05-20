//! IP address utilities — parsing, CIDR, subnet math, classification.
//!
//! Pure Rust IPv4/IPv6 address manipulation. Supports CIDR notation,
//! subnet calculation, network/broadcast addresses, address-in-subnet
//! checks, classification (private/loopback/multicast/link-local), and
//! IPv4-mapped IPv6 addresses.

use std::fmt;

// ── IPv4 Address ──────────────────────────────────────────────

/// A 32-bit IPv4 address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ipv4Addr {
    octets: [u8; 4],
}

impl Ipv4Addr {
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self { octets: [a, b, c, d] }
    }

    pub const fn octets(&self) -> [u8; 4] {
        self.octets
    }

    pub const fn to_u32(&self) -> u32 {
        ((self.octets[0] as u32) << 24)
            | ((self.octets[1] as u32) << 16)
            | ((self.octets[2] as u32) << 8)
            | (self.octets[3] as u32)
    }

    pub const fn from_u32(val: u32) -> Self {
        Self {
            octets: [
                (val >> 24) as u8,
                (val >> 16) as u8,
                (val >> 8) as u8,
                val as u8,
            ],
        }
    }

    /// Parse from dotted-decimal string "a.b.c.d".
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return None;
        }
        let a = parts[0].parse::<u8>().ok()?;
        let b = parts[1].parse::<u8>().ok()?;
        let c = parts[2].parse::<u8>().ok()?;
        let d = parts[3].parse::<u8>().ok()?;
        Some(Self::new(a, b, c, d))
    }

    // ── Classification ────────────────────────────────────────

    /// 127.0.0.0/8
    pub fn is_loopback(&self) -> bool {
        self.octets[0] == 127
    }

    /// 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
    pub fn is_private(&self) -> bool {
        self.octets[0] == 10
            || (self.octets[0] == 172 && (self.octets[1] & 0xf0) == 16)
            || (self.octets[0] == 192 && self.octets[1] == 168)
    }

    /// 224.0.0.0/4
    pub fn is_multicast(&self) -> bool {
        (self.octets[0] & 0xf0) == 224
    }

    /// 169.254.0.0/16
    pub fn is_link_local(&self) -> bool {
        self.octets[0] == 169 && self.octets[1] == 254
    }

    /// 255.255.255.255
    pub fn is_broadcast(&self) -> bool {
        self.octets == [255, 255, 255, 255]
    }

    /// 0.0.0.0
    pub fn is_unspecified(&self) -> bool {
        self.octets == [0, 0, 0, 0]
    }

    /// Convert to IPv4-mapped IPv6 address (::ffff:a.b.c.d).
    pub fn to_ipv6_mapped(&self) -> Ipv6Addr {
        let mut segments = [0u16; 8];
        segments[5] = 0xffff;
        segments[6] = ((self.octets[0] as u16) << 8) | (self.octets[1] as u16);
        segments[7] = ((self.octets[2] as u16) << 8) | (self.octets[3] as u16);
        Ipv6Addr { segments }
    }

    pub const LOCALHOST: Self = Self::new(127, 0, 0, 1);
    pub const UNSPECIFIED: Self = Self::new(0, 0, 0, 0);
    pub const BROADCAST: Self = Self::new(255, 255, 255, 255);
}

impl fmt::Display for Ipv4Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}.{}", self.octets[0], self.octets[1], self.octets[2], self.octets[3])
    }
}

// ── IPv6 Address ──────────────────────────────────────────────

/// A 128-bit IPv6 address stored as 8 × u16 segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ipv6Addr {
    segments: [u16; 8],
}

impl Ipv6Addr {
    pub const fn new(s: [u16; 8]) -> Self {
        Self { segments: s }
    }

    pub const fn segments(&self) -> [u16; 8] {
        self.segments
    }

    /// Parse colon-hex notation "a:b:c:d:e:f:g:h" with :: expansion.
    pub fn parse(s: &str) -> Option<Self> {
        if s.is_empty() {
            return None;
        }

        // Handle :: expansion.
        if s.contains("::") {
            let parts: Vec<&str> = s.split("::").collect();
            if parts.len() > 2 {
                return None; // Only one :: allowed.
            }

            let left: Vec<u16> = if parts[0].is_empty() {
                Vec::new()
            } else {
                let mut v = Vec::new();
                for p in parts[0].split(':') {
                    v.push(u16::from_str_radix(p, 16).ok()?);
                }
                v
            };

            let right: Vec<u16> = if parts.len() < 2 || parts[1].is_empty() {
                Vec::new()
            } else {
                let mut v = Vec::new();
                for p in parts[1].split(':') {
                    v.push(u16::from_str_radix(p, 16).ok()?);
                }
                v
            };

            let fill = 8usize.checked_sub(left.len() + right.len())?;
            let mut segments = [0u16; 8];
            for (i, val) in left.iter().enumerate() {
                segments[i] = *val;
            }
            let offset = left.len() + fill;
            for (i, val) in right.iter().enumerate() {
                segments[offset + i] = *val;
            }
            Some(Self { segments })
        } else {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() != 8 {
                return None;
            }
            let mut segments = [0u16; 8];
            for (i, p) in parts.iter().enumerate() {
                segments[i] = u16::from_str_radix(p, 16).ok()?;
            }
            Some(Self { segments })
        }
    }

    pub fn is_loopback(&self) -> bool {
        self.segments == [0, 0, 0, 0, 0, 0, 0, 1]
    }

    pub fn is_unspecified(&self) -> bool {
        self.segments == [0; 8]
    }

    /// ff00::/8
    pub fn is_multicast(&self) -> bool {
        (self.segments[0] & 0xff00) == 0xff00
    }

    /// fe80::/10
    pub fn is_link_local(&self) -> bool {
        (self.segments[0] & 0xffc0) == 0xfe80
    }

    /// ::ffff:0:0/96
    pub fn is_ipv4_mapped(&self) -> bool {
        self.segments[0] == 0
            && self.segments[1] == 0
            && self.segments[2] == 0
            && self.segments[3] == 0
            && self.segments[4] == 0
            && self.segments[5] == 0xffff
    }

    /// Extract the IPv4 address from an IPv4-mapped IPv6 address.
    pub fn to_ipv4_mapped(&self) -> Option<Ipv4Addr> {
        if !self.is_ipv4_mapped() {
            return None;
        }
        let a = (self.segments[6] >> 8) as u8;
        let b = self.segments[6] as u8;
        let c = (self.segments[7] >> 8) as u8;
        let d = self.segments[7] as u8;
        Some(Ipv4Addr::new(a, b, c, d))
    }

    pub const LOCALHOST: Self = Self { segments: [0, 0, 0, 0, 0, 0, 0, 1] };
    pub const UNSPECIFIED: Self = Self { segments: [0; 8] };
}

impl fmt::Display for Ipv6Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Find the longest run of zeros for :: compression.
        let mut best_start = None;
        let mut best_len = 0usize;
        let mut cur_start = 0;
        let mut cur_len = 0usize;

        for (i, seg) in self.segments.iter().enumerate() {
            if *seg == 0 {
                if cur_len == 0 {
                    cur_start = i;
                }
                cur_len += 1;
                if cur_len > best_len {
                    best_len = cur_len;
                    best_start = Some(cur_start);
                }
            } else {
                cur_len = 0;
            }
        }

        if best_len < 2 {
            // No compression.
            for (i, seg) in self.segments.iter().enumerate() {
                if i > 0 { write!(f, ":")?; }
                write!(f, "{:x}", seg)?;
            }
        } else {
            let start = best_start.unwrap();
            let end = start + best_len;

            for i in 0..start {
                if i > 0 { write!(f, ":")?; }
                write!(f, "{:x}", self.segments[i])?;
            }
            write!(f, "::")?;
            let mut first = true;
            for i in end..8 {
                if !first { write!(f, ":")?; }
                write!(f, "{:x}", self.segments[i])?;
                first = false;
            }
        }

        Ok(())
    }
}

// ── IpAddr enum ───────────────────────────────────────────────

/// Either an IPv4 or IPv6 address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IpAddr {
    V4(Ipv4Addr),
    V6(Ipv6Addr),
}

impl IpAddr {
    /// Parse from string — tries IPv4, then IPv6.
    pub fn parse(s: &str) -> Option<Self> {
        if let Some(v4) = Ipv4Addr::parse(s) {
            Some(Self::V4(v4))
        } else {
            Ipv6Addr::parse(s).map(Self::V6)
        }
    }

    pub fn is_loopback(&self) -> bool {
        match self {
            Self::V4(a) => a.is_loopback(),
            Self::V6(a) => a.is_loopback(),
        }
    }

    pub fn is_multicast(&self) -> bool {
        match self {
            Self::V4(a) => a.is_multicast(),
            Self::V6(a) => a.is_multicast(),
        }
    }

    pub fn is_unspecified(&self) -> bool {
        match self {
            Self::V4(a) => a.is_unspecified(),
            Self::V6(a) => a.is_unspecified(),
        }
    }
}

impl fmt::Display for IpAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V4(a) => write!(f, "{}", a),
            Self::V6(a) => write!(f, "{}", a),
        }
    }
}

// ── CIDR ──────────────────────────────────────────────────────

/// An IPv4 CIDR block (address + prefix length).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cidr {
    pub addr: Ipv4Addr,
    pub prefix_len: u8,
}

impl Cidr {
    /// Create a CIDR from address and prefix length (0–32).
    pub fn new(addr: Ipv4Addr, prefix_len: u8) -> Option<Self> {
        if prefix_len > 32 {
            return None;
        }
        Some(Self { addr, prefix_len })
    }

    /// Parse "a.b.c.d/n" notation.
    pub fn parse(s: &str) -> Option<Self> {
        let (addr_str, prefix_str) = s.split_once('/')?;
        let addr = Ipv4Addr::parse(addr_str)?;
        let prefix_len: u8 = prefix_str.parse().ok()?;
        Self::new(addr, prefix_len)
    }

    /// The subnet mask as a 32-bit value.
    pub fn mask(&self) -> u32 {
        if self.prefix_len == 0 {
            0
        } else {
            !0u32 << (32 - self.prefix_len)
        }
    }

    /// The subnet mask as an Ipv4Addr.
    pub fn mask_addr(&self) -> Ipv4Addr {
        Ipv4Addr::from_u32(self.mask())
    }

    /// Network address (first address in the subnet).
    pub fn network(&self) -> Ipv4Addr {
        Ipv4Addr::from_u32(self.addr.to_u32() & self.mask())
    }

    /// Broadcast address (last address in the subnet).
    pub fn broadcast(&self) -> Ipv4Addr {
        Ipv4Addr::from_u32(self.addr.to_u32() | !self.mask())
    }

    /// First usable host address (network + 1). None for /31 and /32.
    pub fn first_host(&self) -> Option<Ipv4Addr> {
        if self.prefix_len >= 31 {
            return None;
        }
        Some(Ipv4Addr::from_u32(self.network().to_u32() + 1))
    }

    /// Last usable host address (broadcast - 1). None for /31 and /32.
    pub fn last_host(&self) -> Option<Ipv4Addr> {
        if self.prefix_len >= 31 {
            return None;
        }
        Some(Ipv4Addr::from_u32(self.broadcast().to_u32() - 1))
    }

    /// Number of addresses in the subnet.
    pub fn host_count(&self) -> u64 {
        if self.prefix_len == 0 {
            1u64 << 32
        } else {
            1u64 << (32 - self.prefix_len)
        }
    }

    /// Number of usable host addresses (total - 2, minimum 0).
    pub fn usable_hosts(&self) -> u64 {
        self.host_count().saturating_sub(2)
    }

    /// Check if an address belongs to this subnet.
    pub fn contains(&self, addr: Ipv4Addr) -> bool {
        (addr.to_u32() & self.mask()) == (self.addr.to_u32() & self.mask())
    }

    /// Check if this CIDR fully contains another CIDR (is a supernet).
    pub fn contains_cidr(&self, other: &Cidr) -> bool {
        self.prefix_len <= other.prefix_len && self.contains(other.network())
    }
}

impl fmt::Display for Cidr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.network(), self.prefix_len)
    }
}

// ── Subnet Calculator ─────────────────────────────────────────

/// Split a CIDR into two equal subnets.
pub fn split_subnet(cidr: &Cidr) -> Option<(Cidr, Cidr)> {
    if cidr.prefix_len >= 32 {
        return None;
    }
    let new_prefix = cidr.prefix_len + 1;
    let net = cidr.network().to_u32();
    let half = 1u32 << (32 - new_prefix);
    Some((
        Cidr::new(Ipv4Addr::from_u32(net), new_prefix)?,
        Cidr::new(Ipv4Addr::from_u32(net + half), new_prefix)?,
    ))
}

/// Summarize two adjacent subnets of equal prefix length into one.
pub fn summarize(a: &Cidr, b: &Cidr) -> Option<Cidr> {
    if a.prefix_len != b.prefix_len || a.prefix_len == 0 {
        return None;
    }
    let parent_prefix = a.prefix_len - 1;
    let parent_mask = if parent_prefix == 0 { 0 } else { !0u32 << (32 - parent_prefix) };
    let net_a = a.network().to_u32() & parent_mask;
    let net_b = b.network().to_u32() & parent_mask;
    if net_a != net_b {
        return None;
    }
    Cidr::new(Ipv4Addr::from_u32(net_a), parent_prefix)
}

/// Check if two CIDRs overlap.
pub fn cidrs_overlap(a: &Cidr, b: &Cidr) -> bool {
    a.contains(b.network()) || a.contains(b.broadcast()) || b.contains(a.network())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_parse() {
        let addr = Ipv4Addr::parse("192.168.1.1").unwrap();
        assert_eq!(addr.octets(), [192, 168, 1, 1]);
    }

    #[test]
    fn test_ipv4_parse_invalid() {
        assert!(Ipv4Addr::parse("256.0.0.1").is_none());
        assert!(Ipv4Addr::parse("1.2.3").is_none());
        assert!(Ipv4Addr::parse("abc").is_none());
    }

    #[test]
    fn test_ipv4_display() {
        let addr = Ipv4Addr::new(10, 0, 0, 1);
        assert_eq!(format!("{}", addr), "10.0.0.1");
    }

    #[test]
    fn test_ipv4_to_u32_roundtrip() {
        let addr = Ipv4Addr::new(192, 168, 1, 100);
        let val = addr.to_u32();
        assert_eq!(Ipv4Addr::from_u32(val), addr);
    }

    #[test]
    fn test_ipv4_classification() {
        assert!(Ipv4Addr::new(127, 0, 0, 1).is_loopback());
        assert!(Ipv4Addr::new(10, 0, 0, 1).is_private());
        assert!(Ipv4Addr::new(172, 16, 0, 1).is_private());
        assert!(Ipv4Addr::new(192, 168, 0, 1).is_private());
        assert!(!Ipv4Addr::new(8, 8, 8, 8).is_private());
        assert!(Ipv4Addr::new(224, 0, 0, 1).is_multicast());
        assert!(Ipv4Addr::new(169, 254, 1, 1).is_link_local());
        assert!(Ipv4Addr::BROADCAST.is_broadcast());
        assert!(Ipv4Addr::UNSPECIFIED.is_unspecified());
    }

    #[test]
    fn test_ipv6_parse_full() {
        let addr = Ipv6Addr::parse("2001:0db8:0000:0000:0000:0000:0000:0001").unwrap();
        assert_eq!(addr.segments(), [0x2001, 0x0db8, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_ipv6_parse_compressed() {
        let addr = Ipv6Addr::parse("2001:db8::1").unwrap();
        assert_eq!(addr.segments(), [0x2001, 0x0db8, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_ipv6_parse_loopback() {
        let addr = Ipv6Addr::parse("::1").unwrap();
        assert!(addr.is_loopback());
    }

    #[test]
    fn test_ipv6_parse_unspecified() {
        let addr = Ipv6Addr::parse("::").unwrap();
        assert!(addr.is_unspecified());
    }

    #[test]
    fn test_ipv6_display_compression() {
        let addr = Ipv6Addr::new([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]);
        let s = format!("{}", addr);
        assert_eq!(s, "2001:db8::1");
    }

    #[test]
    fn test_ipv6_multicast() {
        let addr = Ipv6Addr::new([0xff02, 0, 0, 0, 0, 0, 0, 1]);
        assert!(addr.is_multicast());
    }

    #[test]
    fn test_ipv6_link_local() {
        let addr = Ipv6Addr::new([0xfe80, 0, 0, 0, 0, 0, 0, 1]);
        assert!(addr.is_link_local());
    }

    #[test]
    fn test_ipv4_mapped_ipv6() {
        let v4 = Ipv4Addr::new(192, 168, 1, 1);
        let v6 = v4.to_ipv6_mapped();
        assert!(v6.is_ipv4_mapped());
        let back = v6.to_ipv4_mapped().unwrap();
        assert_eq!(back, v4);
    }

    #[test]
    fn test_cidr_parse() {
        let cidr = Cidr::parse("192.168.1.0/24").unwrap();
        assert_eq!(cidr.prefix_len, 24);
        assert_eq!(cidr.network(), Ipv4Addr::new(192, 168, 1, 0));
    }

    #[test]
    fn test_cidr_mask() {
        let cidr = Cidr::parse("10.0.0.0/8").unwrap();
        assert_eq!(cidr.mask_addr(), Ipv4Addr::new(255, 0, 0, 0));

        let cidr24 = Cidr::parse("192.168.1.0/24").unwrap();
        assert_eq!(cidr24.mask_addr(), Ipv4Addr::new(255, 255, 255, 0));
    }

    #[test]
    fn test_cidr_network_broadcast() {
        let cidr = Cidr::parse("192.168.1.100/24").unwrap();
        assert_eq!(cidr.network(), Ipv4Addr::new(192, 168, 1, 0));
        assert_eq!(cidr.broadcast(), Ipv4Addr::new(192, 168, 1, 255));
    }

    #[test]
    fn test_cidr_host_range() {
        let cidr = Cidr::parse("192.168.1.0/24").unwrap();
        assert_eq!(cidr.first_host().unwrap(), Ipv4Addr::new(192, 168, 1, 1));
        assert_eq!(cidr.last_host().unwrap(), Ipv4Addr::new(192, 168, 1, 254));
    }

    #[test]
    fn test_cidr_host_count() {
        let cidr = Cidr::parse("10.0.0.0/24").unwrap();
        assert_eq!(cidr.host_count(), 256);
        assert_eq!(cidr.usable_hosts(), 254);
    }

    #[test]
    fn test_cidr_contains() {
        let cidr = Cidr::parse("10.0.0.0/8").unwrap();
        assert!(cidr.contains(Ipv4Addr::new(10, 1, 2, 3)));
        assert!(!cidr.contains(Ipv4Addr::new(11, 0, 0, 1)));
    }

    #[test]
    fn test_cidr_contains_cidr() {
        let big = Cidr::parse("10.0.0.0/8").unwrap();
        let small = Cidr::parse("10.1.0.0/16").unwrap();
        assert!(big.contains_cidr(&small));
        assert!(!small.contains_cidr(&big));
    }

    #[test]
    fn test_split_subnet() {
        let cidr = Cidr::parse("192.168.0.0/24").unwrap();
        let (a, b) = split_subnet(&cidr).unwrap();
        assert_eq!(a.prefix_len, 25);
        assert_eq!(a.network(), Ipv4Addr::new(192, 168, 0, 0));
        assert_eq!(b.network(), Ipv4Addr::new(192, 168, 0, 128));
    }

    #[test]
    fn test_summarize() {
        let a = Cidr::parse("192.168.0.0/25").unwrap();
        let b = Cidr::parse("192.168.0.128/25").unwrap();
        let sum = summarize(&a, &b).unwrap();
        assert_eq!(sum.prefix_len, 24);
        assert_eq!(sum.network(), Ipv4Addr::new(192, 168, 0, 0));
    }

    #[test]
    fn test_cidrs_overlap() {
        let a = Cidr::parse("10.0.0.0/8").unwrap();
        let b = Cidr::parse("10.1.0.0/16").unwrap();
        assert!(cidrs_overlap(&a, &b));

        let c = Cidr::parse("192.168.0.0/16").unwrap();
        assert!(!cidrs_overlap(&a, &c));
    }

    #[test]
    fn test_ip_addr_parse() {
        let v4 = IpAddr::parse("1.2.3.4").unwrap();
        assert!(matches!(v4, IpAddr::V4(_)));

        let v6 = IpAddr::parse("::1").unwrap();
        assert!(matches!(v6, IpAddr::V6(_)));
    }

    #[test]
    fn test_cidr_slash32() {
        let cidr = Cidr::parse("10.0.0.1/32").unwrap();
        assert_eq!(cidr.host_count(), 1);
        assert!(cidr.first_host().is_none());
        assert!(cidr.contains(Ipv4Addr::new(10, 0, 0, 1)));
        assert!(!cidr.contains(Ipv4Addr::new(10, 0, 0, 2)));
    }

    #[test]
    fn test_cidr_display() {
        let cidr = Cidr::parse("192.168.1.100/24").unwrap();
        assert_eq!(format!("{}", cidr), "192.168.1.0/24");
    }

    #[test]
    fn test_ipv6_non_ipv4_mapped() {
        let addr = Ipv6Addr::new([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]);
        assert!(!addr.is_ipv4_mapped());
        assert!(addr.to_ipv4_mapped().is_none());
    }
}
