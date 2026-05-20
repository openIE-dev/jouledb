//! SMTP protocol model.
//!
//! Replaces `lettre` / `nodemailer` with a pure-Rust SMTP command builder
//! and response parser. Supports EHLO, MAIL FROM, RCPT TO, DATA, QUIT, RSET,
//! response parsing (status code + text), MIME message building, multipart
//! boundaries, base64 content encoding, and email address validation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// SMTP protocol errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtpError {
    /// Invalid email address.
    InvalidAddress(String),
    /// Invalid SMTP response.
    InvalidResponse(String),
    /// Server returned an error status.
    ServerError { code: u16, message: String },
    /// Invalid MIME boundary.
    InvalidBoundary(String),
    /// Missing required field.
    MissingField(String),
}

impl fmt::Display for SmtpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAddress(a) => write!(f, "invalid email address: {a}"),
            Self::InvalidResponse(r) => write!(f, "invalid SMTP response: {r}"),
            Self::ServerError { code, message } => write!(f, "SMTP error {code}: {message}"),
            Self::InvalidBoundary(b) => write!(f, "invalid boundary: {b}"),
            Self::MissingField(field) => write!(f, "missing field: {field}"),
        }
    }
}

impl std::error::Error for SmtpError {}

// ── Email Address Validation ────────────────────────────────

/// Validate an email address (simplified RFC 5321).
pub fn validate_email(address: &str) -> Result<(), SmtpError> {
    let address = address.trim();
    if address.is_empty() {
        return Err(SmtpError::InvalidAddress("empty address".into()));
    }
    let at_pos = address
        .rfind('@')
        .ok_or_else(|| SmtpError::InvalidAddress(format!("no @ in {address}")))?;
    let local = &address[..at_pos];
    let domain = &address[at_pos + 1..];

    if local.is_empty() {
        return Err(SmtpError::InvalidAddress("empty local part".into()));
    }
    if local.len() > 64 {
        return Err(SmtpError::InvalidAddress("local part > 64 chars".into()));
    }
    if domain.is_empty() {
        return Err(SmtpError::InvalidAddress("empty domain".into()));
    }
    if domain.len() > 255 {
        return Err(SmtpError::InvalidAddress("domain > 255 chars".into()));
    }
    if !domain.contains('.') {
        return Err(SmtpError::InvalidAddress("domain has no dot".into()));
    }
    // Check for basic invalid chars in local part
    for ch in local.chars() {
        if ch.is_control() || ch == ' ' {
            return Err(SmtpError::InvalidAddress(format!("invalid char in local: {ch:?}")));
        }
    }
    // Domain labels
    for label in domain.split('.') {
        if label.is_empty() {
            return Err(SmtpError::InvalidAddress("empty domain label".into()));
        }
        if label.len() > 63 {
            return Err(SmtpError::InvalidAddress("domain label > 63 chars".into()));
        }
    }
    Ok(())
}

// ── SMTP Commands ───────────────────────────────────────────

/// An SMTP command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtpCommand {
    Ehlo(String),
    Helo(String),
    MailFrom { address: String, params: Vec<String> },
    RcptTo { address: String, params: Vec<String> },
    Data,
    Rset,
    Quit,
    Noop,
    Vrfy(String),
    StartTls,
    Auth { mechanism: String, initial_response: Option<String> },
}

impl SmtpCommand {
    /// Serialize command to SMTP wire format (with \r\n).
    pub fn to_string(&self) -> String {
        match self {
            Self::Ehlo(domain) => format!("EHLO {domain}\r\n"),
            Self::Helo(domain) => format!("HELO {domain}\r\n"),
            Self::MailFrom { address, params } => {
                let mut cmd = format!("MAIL FROM:<{address}>");
                for p in params {
                    cmd.push(' ');
                    cmd.push_str(p);
                }
                cmd.push_str("\r\n");
                cmd
            }
            Self::RcptTo { address, params } => {
                let mut cmd = format!("RCPT TO:<{address}>");
                for p in params {
                    cmd.push(' ');
                    cmd.push_str(p);
                }
                cmd.push_str("\r\n");
                cmd
            }
            Self::Data => "DATA\r\n".to_string(),
            Self::Rset => "RSET\r\n".to_string(),
            Self::Quit => "QUIT\r\n".to_string(),
            Self::Noop => "NOOP\r\n".to_string(),
            Self::Vrfy(addr) => format!("VRFY {addr}\r\n"),
            Self::StartTls => "STARTTLS\r\n".to_string(),
            Self::Auth { mechanism, initial_response } => {
                if let Some(resp) = initial_response {
                    format!("AUTH {mechanism} {resp}\r\n")
                } else {
                    format!("AUTH {mechanism}\r\n")
                }
            }
        }
    }

    /// Parse an SMTP command from a line (without trailing \r\n).
    pub fn parse(line: &str) -> Result<Self, SmtpError> {
        let line = line.trim_end_matches("\r\n").trim_end_matches('\n');
        let upper = line.to_uppercase();

        if upper.starts_with("EHLO ") {
            return Ok(Self::Ehlo(line[5..].trim().to_string()));
        }
        if upper.starts_with("HELO ") {
            return Ok(Self::Helo(line[5..].trim().to_string()));
        }
        if upper.starts_with("MAIL FROM:") {
            let rest = line[10..].trim();
            let (addr, params) = parse_angle_address(rest)?;
            return Ok(Self::MailFrom { address: addr, params });
        }
        if upper.starts_with("RCPT TO:") {
            let rest = line[8..].trim();
            let (addr, params) = parse_angle_address(rest)?;
            return Ok(Self::RcptTo { address: addr, params });
        }
        if upper == "DATA" {
            return Ok(Self::Data);
        }
        if upper == "RSET" {
            return Ok(Self::Rset);
        }
        if upper == "QUIT" {
            return Ok(Self::Quit);
        }
        if upper == "NOOP" {
            return Ok(Self::Noop);
        }
        if upper.starts_with("VRFY ") {
            return Ok(Self::Vrfy(line[5..].trim().to_string()));
        }
        if upper == "STARTTLS" {
            return Ok(Self::StartTls);
        }
        if upper.starts_with("AUTH ") {
            let parts: Vec<&str> = line[5..].trim().splitn(2, ' ').collect();
            return Ok(Self::Auth {
                mechanism: parts[0].to_string(),
                initial_response: parts.get(1).map(|s| s.to_string()),
            });
        }
        Err(SmtpError::InvalidResponse(format!("unknown command: {line}")))
    }
}

fn parse_angle_address(s: &str) -> Result<(String, Vec<String>), SmtpError> {
    if !s.starts_with('<') {
        return Err(SmtpError::InvalidAddress(format!("missing < in: {s}")));
    }
    let end = s
        .find('>')
        .ok_or_else(|| SmtpError::InvalidAddress(format!("missing > in: {s}")))?;
    let addr = s[1..end].to_string();
    let rest = s[end + 1..].trim();
    let params: Vec<String> = if rest.is_empty() {
        Vec::new()
    } else {
        rest.split_whitespace().map(|p| p.to_string()).collect()
    };
    Ok((addr, params))
}

// ── SMTP Response ───────────────────────────────────────────

/// A parsed SMTP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmtpResponse {
    pub code: u16,
    pub lines: Vec<String>,
}

impl SmtpResponse {
    /// Parse an SMTP response (may be multi-line).
    pub fn parse(text: &str) -> Result<Self, SmtpError> {
        let mut code: Option<u16> = None;
        let mut lines = Vec::new();

        for line in text.lines() {
            if line.len() < 3 {
                return Err(SmtpError::InvalidResponse(line.to_string()));
            }
            let line_code: u16 = line[..3]
                .parse()
                .map_err(|_| SmtpError::InvalidResponse(line.to_string()))?;
            if let Some(c) = code {
                if c != line_code {
                    return Err(SmtpError::InvalidResponse(
                        format!("inconsistent codes: {c} vs {line_code}"),
                    ));
                }
            } else {
                code = Some(line_code);
            }
            let text = if line.len() > 4 { line[4..].to_string() } else { String::new() };
            lines.push(text);
        }

        let code = code.ok_or_else(|| SmtpError::InvalidResponse("empty response".into()))?;
        Ok(Self { code, lines })
    }

    /// Whether this is a positive (2xx/3xx) response.
    pub fn is_positive(&self) -> bool {
        self.code >= 200 && self.code < 400
    }

    /// Whether this is a permanent error (5xx).
    pub fn is_permanent_error(&self) -> bool {
        self.code >= 500 && self.code < 600
    }

    /// Whether this is a transient error (4xx).
    pub fn is_transient_error(&self) -> bool {
        self.code >= 400 && self.code < 500
    }

    /// Get the primary text of the response.
    pub fn text(&self) -> &str {
        self.lines.last().map(|s| s.as_str()).unwrap_or("")
    }
}

// ── Base64 Encoding ─────────────────────────────────────────

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode data to base64.
pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() { data[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if i + 1 < data.len() {
            out.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if i + 2 < data.len() {
            out.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

/// Decode base64 to bytes.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, SmtpError> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if s.len() % 4 != 0 {
        return Err(SmtpError::InvalidResponse("base64 length not multiple of 4".into()));
    }
    let mut out = Vec::new();
    let chars: Vec<u8> = s.bytes().collect();
    let mut i = 0;
    while i < chars.len() {
        let a = b64_val(chars[i])?;
        let b = b64_val(chars[i + 1])?;
        let c_byte = chars[i + 2];
        let d_byte = chars[i + 3];

        out.push(((a << 2) | (b >> 4)) as u8);
        if c_byte != b'=' {
            let c = b64_val(c_byte)?;
            out.push((((b & 0x0F) << 4) | (c >> 2)) as u8);
            if d_byte != b'=' {
                let d = b64_val(d_byte)?;
                out.push((((c & 0x03) << 6) | d) as u8);
            }
        }
        i += 4;
    }
    Ok(out)
}

fn b64_val(ch: u8) -> Result<u32, SmtpError> {
    match ch {
        b'A'..=b'Z' => Ok((ch - b'A') as u32),
        b'a'..=b'z' => Ok((ch - b'a' + 26) as u32),
        b'0'..=b'9' => Ok((ch - b'0' + 52) as u32),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(SmtpError::InvalidResponse(format!("invalid base64 char: {ch}"))),
    }
}

// ── MIME Message Builder ────────────────────────────────────

/// Content transfer encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentEncoding {
    SevenBit,
    EightBit,
    Base64,
    QuotedPrintable,
}

impl ContentEncoding {
    pub fn as_str(&self) -> &str {
        match self {
            Self::SevenBit => "7bit",
            Self::EightBit => "8bit",
            Self::Base64 => "base64",
            Self::QuotedPrintable => "quoted-printable",
        }
    }
}

/// A MIME part.
#[derive(Debug, Clone)]
pub struct MimePart {
    pub content_type: String,
    pub encoding: ContentEncoding,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl MimePart {
    pub fn text(content: &str) -> Self {
        Self {
            content_type: "text/plain; charset=UTF-8".to_string(),
            encoding: ContentEncoding::SevenBit,
            headers: Vec::new(),
            body: content.as_bytes().to_vec(),
        }
    }

    pub fn html(content: &str) -> Self {
        Self {
            content_type: "text/html; charset=UTF-8".to_string(),
            encoding: ContentEncoding::SevenBit,
            headers: Vec::new(),
            body: content.as_bytes().to_vec(),
        }
    }

    pub fn attachment(filename: &str, content_type: &str, data: &[u8]) -> Self {
        Self {
            content_type: content_type.to_string(),
            encoding: ContentEncoding::Base64,
            headers: vec![(
                "Content-Disposition".to_string(),
                format!("attachment; filename=\"{filename}\""),
            )],
            body: data.to_vec(),
        }
    }

    fn render(&self) -> String {
        let mut out = format!("Content-Type: {}\r\n", self.content_type);
        out.push_str(&format!(
            "Content-Transfer-Encoding: {}\r\n",
            self.encoding.as_str()
        ));
        for (name, value) in &self.headers {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
        out.push_str("\r\n");
        match self.encoding {
            ContentEncoding::Base64 => {
                let encoded = base64_encode(&self.body);
                // Wrap at 76 chars
                for chunk in encoded.as_bytes().chunks(76) {
                    out.push_str(std::str::from_utf8(chunk).unwrap_or(""));
                    out.push_str("\r\n");
                }
            }
            _ => {
                out.push_str(&String::from_utf8_lossy(&self.body));
            }
        }
        out
    }
}

/// A complete MIME email message.
#[derive(Debug, Clone)]
pub struct MimeMessage {
    pub from: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub subject: String,
    pub message_id: Option<String>,
    pub date: Option<String>,
    pub extra_headers: Vec<(String, String)>,
    pub parts: Vec<MimePart>,
}

impl MimeMessage {
    pub fn new(from: &str, to: &str, subject: &str) -> Self {
        Self {
            from: from.to_string(),
            to: vec![to.to_string()],
            cc: Vec::new(),
            subject: subject.to_string(),
            message_id: None,
            date: None,
            extra_headers: Vec::new(),
            parts: Vec::new(),
        }
    }

    pub fn add_part(mut self, part: MimePart) -> Self {
        self.parts.push(part);
        self
    }

    pub fn with_cc(mut self, cc: &str) -> Self {
        self.cc.push(cc.to_string());
        self
    }

    pub fn with_message_id(mut self, id: &str) -> Self {
        self.message_id = Some(id.to_string());
        self
    }

    /// Generate a multipart boundary string.
    pub fn generate_boundary(seed: u64) -> String {
        // Deterministic boundary for testing
        format!("----=_Part_{seed:016x}")
    }

    /// Render the complete message as RFC 2822 text.
    pub fn render(&self, boundary_seed: u64) -> String {
        let mut out = String::new();
        out.push_str(&format!("From: {}\r\n", self.from));
        out.push_str(&format!("To: {}\r\n", self.to.join(", ")));
        if !self.cc.is_empty() {
            out.push_str(&format!("Cc: {}\r\n", self.cc.join(", ")));
        }
        out.push_str(&format!("Subject: {}\r\n", self.subject));
        if let Some(id) = &self.message_id {
            out.push_str(&format!("Message-ID: {id}\r\n"));
        }
        if let Some(date) = &self.date {
            out.push_str(&format!("Date: {date}\r\n"));
        }
        out.push_str("MIME-Version: 1.0\r\n");

        for (name, value) in &self.extra_headers {
            out.push_str(&format!("{name}: {value}\r\n"));
        }

        if self.parts.len() <= 1 {
            // Single part
            if let Some(part) = self.parts.first() {
                out.push_str(&format!("Content-Type: {}\r\n", part.content_type));
                out.push_str(&format!(
                    "Content-Transfer-Encoding: {}\r\n",
                    part.encoding.as_str()
                ));
                out.push_str("\r\n");
                match part.encoding {
                    ContentEncoding::Base64 => {
                        let encoded = base64_encode(&part.body);
                        for chunk in encoded.as_bytes().chunks(76) {
                            out.push_str(std::str::from_utf8(chunk).unwrap_or(""));
                            out.push_str("\r\n");
                        }
                    }
                    _ => {
                        out.push_str(&String::from_utf8_lossy(&part.body));
                        out.push_str("\r\n");
                    }
                }
            } else {
                out.push_str("\r\n");
            }
        } else {
            let boundary = Self::generate_boundary(boundary_seed);
            out.push_str(&format!(
                "Content-Type: multipart/mixed; boundary=\"{boundary}\"\r\n"
            ));
            out.push_str("\r\n");
            for part in &self.parts {
                out.push_str(&format!("--{boundary}\r\n"));
                out.push_str(&part.render());
            }
            out.push_str(&format!("--{boundary}--\r\n"));
        }
        out
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_email_valid() {
        assert!(validate_email("user@example.com").is_ok());
        assert!(validate_email("foo.bar+tag@sub.domain.org").is_ok());
    }

    #[test]
    fn validate_email_invalid() {
        assert!(validate_email("").is_err());
        assert!(validate_email("noatsign").is_err());
        assert!(validate_email("@domain.com").is_err());
        assert!(validate_email("user@").is_err());
        assert!(validate_email("user@nodot").is_err());
    }

    #[test]
    fn smtp_command_ehlo() {
        let cmd = SmtpCommand::Ehlo("mail.example.com".into());
        assert_eq!(cmd.to_string(), "EHLO mail.example.com\r\n");
        let parsed = SmtpCommand::parse("EHLO mail.example.com").unwrap();
        assert_eq!(parsed, cmd);
    }

    #[test]
    fn smtp_command_mail_from() {
        let cmd = SmtpCommand::MailFrom {
            address: "sender@example.com".into(),
            params: vec!["SIZE=1024".into()],
        };
        assert_eq!(cmd.to_string(), "MAIL FROM:<sender@example.com> SIZE=1024\r\n");
    }

    #[test]
    fn smtp_command_rcpt_to() {
        let cmd = SmtpCommand::RcptTo {
            address: "rcpt@example.com".into(),
            params: Vec::new(),
        };
        assert_eq!(cmd.to_string(), "RCPT TO:<rcpt@example.com>\r\n");
    }

    #[test]
    fn smtp_command_parse_mail_from() {
        let parsed = SmtpCommand::parse("MAIL FROM:<test@test.com>").unwrap();
        assert_eq!(
            parsed,
            SmtpCommand::MailFrom { address: "test@test.com".into(), params: Vec::new() }
        );
    }

    #[test]
    fn smtp_response_parse() {
        let resp = SmtpResponse::parse("250 OK").unwrap();
        assert_eq!(resp.code, 250);
        assert_eq!(resp.text(), "OK");
        assert!(resp.is_positive());
    }

    #[test]
    fn smtp_response_multiline() {
        let resp = SmtpResponse::parse("250-mail.example.com\n250-SIZE 10240000\n250 AUTH PLAIN").unwrap();
        assert_eq!(resp.code, 250);
        assert_eq!(resp.lines.len(), 3);
        assert_eq!(resp.text(), "AUTH PLAIN");
    }

    #[test]
    fn smtp_response_error() {
        let resp = SmtpResponse::parse("550 Mailbox not found").unwrap();
        assert!(resp.is_permanent_error());
        assert!(!resp.is_positive());
    }

    #[test]
    fn base64_roundtrip() {
        let data = b"Hello, World!";
        let encoded = base64_encode(data);
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn base64_empty() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_decode("").unwrap(), b"");
    }

    #[test]
    fn mime_simple_text() {
        let msg = MimeMessage::new("sender@test.com", "rcpt@test.com", "Test")
            .add_part(MimePart::text("Hello"));
        let rendered = msg.render(0);
        assert!(rendered.contains("From: sender@test.com"));
        assert!(rendered.contains("To: rcpt@test.com"));
        assert!(rendered.contains("Subject: Test"));
        assert!(rendered.contains("Hello"));
    }

    #[test]
    fn mime_multipart() {
        let msg = MimeMessage::new("a@b.com", "c@d.com", "Multi")
            .add_part(MimePart::text("body text"))
            .add_part(MimePart::attachment("doc.pdf", "application/pdf", b"\x25PDF"));
        let rendered = msg.render(42);
        assert!(rendered.contains("multipart/mixed"));
        assert!(rendered.contains("body text"));
        assert!(rendered.contains("Content-Disposition: attachment"));
    }

    #[test]
    fn smtp_command_quit_rset() {
        assert_eq!(SmtpCommand::Quit.to_string(), "QUIT\r\n");
        assert_eq!(SmtpCommand::Rset.to_string(), "RSET\r\n");
        assert_eq!(SmtpCommand::parse("QUIT").unwrap(), SmtpCommand::Quit);
        assert_eq!(SmtpCommand::parse("RSET").unwrap(), SmtpCommand::Rset);
    }
}
