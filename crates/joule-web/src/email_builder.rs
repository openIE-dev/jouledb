//! Email construction — MIME multipart, headers, text/HTML body, attachments,
//! inline images, address parsing (RFC 5322), message-ID generation.
//!
//! Pure-Rust replacement for lettre/mail-builder for constructing email messages.

use std::collections::HashMap;
use std::fmt;

// ── Email Address (RFC 5322) ────────────────────────────────────

/// A parsed email address with optional display name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmailAddress {
    pub name: Option<String>,
    pub local: String,
    pub domain: String,
}

impl EmailAddress {
    /// Create an address from bare email `user@domain`.
    pub fn new(email: &str) -> Option<Self> {
        parse_bare_address(email)
    }

    /// Create with a display name.
    pub fn with_name(name: &str, email: &str) -> Option<Self> {
        let mut addr = parse_bare_address(email)?;
        addr.name = Some(name.to_string());
        Some(addr)
    }

    /// Parse an RFC 5322 address like `"Display Name" <user@domain>` or `user@domain`.
    pub fn parse(input: &str) -> Option<Self> {
        let input = input.trim();
        if let Some(lt_pos) = input.rfind('<') {
            let gt_pos = input.find('>')?;
            if gt_pos <= lt_pos { return None; }
            let email_part = &input[lt_pos + 1..gt_pos];
            let name_part = input[..lt_pos].trim().trim_matches('"').trim();
            let mut addr = parse_bare_address(email_part)?;
            if !name_part.is_empty() {
                addr.name = Some(name_part.to_string());
            }
            Some(addr)
        } else {
            parse_bare_address(input)
        }
    }

    /// Format as RFC 5322 string.
    pub fn to_rfc5322(&self) -> String {
        let email = format!("{}@{}", self.local, self.domain);
        match &self.name {
            Some(n) => format!("\"{}\" <{}>", n, email),
            None => email,
        }
    }

    pub fn email(&self) -> String {
        format!("{}@{}", self.local, self.domain)
    }
}

impl fmt::Display for EmailAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_rfc5322())
    }
}

fn parse_bare_address(email: &str) -> Option<EmailAddress> {
    let email = email.trim();
    let at_pos = email.rfind('@')?;
    if at_pos == 0 || at_pos == email.len() - 1 { return None; }
    let local = &email[..at_pos];
    let domain = &email[at_pos + 1..];
    if local.is_empty() || domain.is_empty() { return None; }
    if !domain.contains('.') && !domain.eq_ignore_ascii_case("localhost") { return None; }
    if local.len() > 64 || domain.len() > 255 { return None; }
    Some(EmailAddress { name: None, local: local.to_string(), domain: domain.to_string() })
}

// ── MIME Content Types ──────────────────────────────────────────

/// MIME content type for body parts.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    TextPlain,
    TextHtml,
    Custom(String),
}

impl ContentType {
    pub fn as_str(&self) -> &str {
        match self {
            ContentType::TextPlain => "text/plain; charset=utf-8",
            ContentType::TextHtml => "text/html; charset=utf-8",
            ContentType::Custom(s) => s.as_str(),
        }
    }
}

// ── Content Transfer Encoding ───────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferEncoding { SevenBit, QuotedPrintable, Base64 }

impl TransferEncoding {
    pub fn as_str(&self) -> &str {
        match self {
            TransferEncoding::SevenBit => "7bit",
            TransferEncoding::QuotedPrintable => "quoted-printable",
            TransferEncoding::Base64 => "base64",
        }
    }
}

// ── Base64 Encoding ─────────────────────────────────────────────

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Base64-encode and wrap at 76-char lines for MIME.
pub fn base64_encode_mime(data: &[u8]) -> String {
    let raw = base64_encode(data);
    let mut wrapped = String::with_capacity(raw.len() + raw.len() / 76 * 2);
    for (i, ch) in raw.chars().enumerate() {
        if i > 0 && i % 76 == 0 { wrapped.push_str("\r\n"); }
        wrapped.push(ch);
    }
    wrapped
}

// ── Quoted-Printable Encoding ───────────────────────────────────

pub fn quoted_printable_encode(data: &[u8]) -> String {
    let mut result = String::new();
    let mut col = 0;
    for &b in data {
        if b == b'\r' || b == b'\n' {
            result.push(b as char);
            col = 0;
            continue;
        }
        let needs_encoding = b < 0x20 || b > 0x7E || b == b'=';
        let token = if needs_encoding {
            format!("={:02X}", b)
        } else {
            String::from(b as char)
        };
        if col + token.len() > 75 {
            result.push_str("=\r\n");
            col = 0;
        }
        result.push_str(&token);
        col += token.len();
    }
    result
}

// ── Message-ID Generation ───────────────────────────────────────

/// Generate a message-ID using a domain and a unique value.
pub fn generate_message_id(domain: &str, unique: &str) -> String {
    format!("<{unique}@{domain}>")
}

// ── Attachment ──────────────────────────────────────────────────

/// An email attachment.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub filename: String,
    pub content_type: String,
    pub data: Vec<u8>,
    pub content_id: Option<String>,
}

impl Attachment {
    pub fn new(filename: &str, content_type: &str, data: Vec<u8>) -> Self {
        Self { filename: filename.into(), content_type: content_type.into(), data, content_id: None }
    }

    /// Create an inline attachment with a Content-ID for embedding in HTML.
    pub fn inline_image(filename: &str, content_type: &str, data: Vec<u8>, cid: &str) -> Self {
        Self {
            filename: filename.into(),
            content_type: content_type.into(),
            data,
            content_id: Some(cid.to_string()),
        }
    }
}

// ── Email Builder ───────────────────────────────────────────────

/// Builder for constructing RFC 5322 / MIME email messages.
pub struct EmailBuilder {
    from: Option<EmailAddress>,
    to: Vec<EmailAddress>,
    cc: Vec<EmailAddress>,
    bcc: Vec<EmailAddress>,
    reply_to: Option<EmailAddress>,
    subject: String,
    headers: HashMap<String, String>,
    text_body: Option<String>,
    html_body: Option<String>,
    attachments: Vec<Attachment>,
    message_id: Option<String>,
}

impl EmailBuilder {
    pub fn new() -> Self {
        Self {
            from: None, to: Vec::new(), cc: Vec::new(), bcc: Vec::new(),
            reply_to: None, subject: String::new(), headers: HashMap::new(),
            text_body: None, html_body: None, attachments: Vec::new(),
            message_id: None,
        }
    }

    pub fn from(mut self, addr: EmailAddress) -> Self { self.from = Some(addr); self }
    pub fn to(mut self, addr: EmailAddress) -> Self { self.to.push(addr); self }
    pub fn cc(mut self, addr: EmailAddress) -> Self { self.cc.push(addr); self }
    pub fn bcc(mut self, addr: EmailAddress) -> Self { self.bcc.push(addr); self }
    pub fn reply_to(mut self, addr: EmailAddress) -> Self { self.reply_to = Some(addr); self }
    pub fn subject(mut self, subject: &str) -> Self { self.subject = subject.into(); self }
    pub fn message_id(mut self, id: &str) -> Self { self.message_id = Some(id.into()); self }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn text_body(mut self, body: &str) -> Self { self.text_body = Some(body.into()); self }
    pub fn html_body(mut self, body: &str) -> Self { self.html_body = Some(body.into()); self }
    pub fn attachment(mut self, att: Attachment) -> Self { self.attachments.push(att); self }

    /// Build the complete MIME message as a string.
    pub fn build(&self) -> Result<String, EmailError> {
        let from = self.from.as_ref().ok_or(EmailError::MissingFrom)?;
        if self.to.is_empty() { return Err(EmailError::MissingTo); }

        let boundary_mixed = "----=_Part_Mixed_001";
        let boundary_alt = "----=_Part_Alt_002";

        let mut msg = String::new();

        // Headers
        msg.push_str(&format!("From: {}\r\n", from.to_rfc5322()));
        let to_list: Vec<String> = self.to.iter().map(|a| a.to_rfc5322()).collect();
        msg.push_str(&format!("To: {}\r\n", to_list.join(", ")));
        if !self.cc.is_empty() {
            let cc_list: Vec<String> = self.cc.iter().map(|a| a.to_rfc5322()).collect();
            msg.push_str(&format!("Cc: {}\r\n", cc_list.join(", ")));
        }
        msg.push_str(&format!("Subject: {}\r\n", self.subject));
        if let Some(ref reply) = self.reply_to {
            msg.push_str(&format!("Reply-To: {}\r\n", reply.to_rfc5322()));
        }
        if let Some(ref mid) = self.message_id {
            msg.push_str(&format!("Message-ID: {}\r\n", mid));
        }
        // Sort custom headers for deterministic output
        let mut sorted_headers: Vec<_> = self.headers.iter().collect();
        sorted_headers.sort_by_key(|(k, _)| (*k).clone());
        for (k, v) in sorted_headers {
            msg.push_str(&format!("{k}: {v}\r\n"));
        }
        msg.push_str("MIME-Version: 1.0\r\n");

        let has_attachments = !self.attachments.is_empty();
        let has_both_bodies = self.text_body.is_some() && self.html_body.is_some();

        if has_attachments {
            msg.push_str(&format!("Content-Type: multipart/mixed; boundary=\"{boundary_mixed}\"\r\n\r\n"));
            msg.push_str(&format!("--{boundary_mixed}\r\n"));
            self.write_body_part(&mut msg, boundary_alt, has_both_bodies);
            for att in &self.attachments {
                msg.push_str(&format!("--{boundary_mixed}\r\n"));
                self.write_attachment(&mut msg, att);
            }
            msg.push_str(&format!("--{boundary_mixed}--\r\n"));
        } else if has_both_bodies {
            msg.push_str(&format!("Content-Type: multipart/alternative; boundary=\"{boundary_alt}\"\r\n\r\n"));
            if let Some(ref text) = self.text_body {
                msg.push_str(&format!("--{boundary_alt}\r\n"));
                msg.push_str("Content-Type: text/plain; charset=utf-8\r\n");
                msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
                msg.push_str(&quoted_printable_encode(text.as_bytes()));
                msg.push_str("\r\n");
            }
            if let Some(ref html) = self.html_body {
                msg.push_str(&format!("--{boundary_alt}\r\n"));
                msg.push_str("Content-Type: text/html; charset=utf-8\r\n");
                msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
                msg.push_str(&quoted_printable_encode(html.as_bytes()));
                msg.push_str("\r\n");
            }
            msg.push_str(&format!("--{boundary_alt}--\r\n"));
        } else if let Some(ref html) = self.html_body {
            msg.push_str("Content-Type: text/html; charset=utf-8\r\n");
            msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
            msg.push_str(&quoted_printable_encode(html.as_bytes()));
        } else if let Some(ref text) = self.text_body {
            msg.push_str("Content-Type: text/plain; charset=utf-8\r\n");
            msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
            msg.push_str(&quoted_printable_encode(text.as_bytes()));
        } else {
            msg.push_str("Content-Type: text/plain; charset=utf-8\r\n\r\n");
        }

        Ok(msg)
    }

    fn write_body_part(&self, msg: &mut String, boundary_alt: &str, has_both: bool) {
        if has_both {
            msg.push_str(&format!("Content-Type: multipart/alternative; boundary=\"{boundary_alt}\"\r\n\r\n"));
            if let Some(ref text) = self.text_body {
                msg.push_str(&format!("--{boundary_alt}\r\n"));
                msg.push_str("Content-Type: text/plain; charset=utf-8\r\n");
                msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
                msg.push_str(&quoted_printable_encode(text.as_bytes()));
                msg.push_str("\r\n");
            }
            if let Some(ref html) = self.html_body {
                msg.push_str(&format!("--{boundary_alt}\r\n"));
                msg.push_str("Content-Type: text/html; charset=utf-8\r\n");
                msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
                msg.push_str(&quoted_printable_encode(html.as_bytes()));
                msg.push_str("\r\n");
            }
            msg.push_str(&format!("--{boundary_alt}--\r\n"));
        } else if let Some(ref html) = self.html_body {
            msg.push_str("Content-Type: text/html; charset=utf-8\r\n");
            msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
            msg.push_str(&quoted_printable_encode(html.as_bytes()));
            msg.push_str("\r\n");
        } else if let Some(ref text) = self.text_body {
            msg.push_str("Content-Type: text/plain; charset=utf-8\r\n");
            msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n\r\n");
            msg.push_str(&quoted_printable_encode(text.as_bytes()));
            msg.push_str("\r\n");
        }
    }

    fn write_attachment(&self, msg: &mut String, att: &Attachment) {
        msg.push_str(&format!("Content-Type: {}; name=\"{}\"\r\n", att.content_type, att.filename));
        if let Some(ref cid) = att.content_id {
            msg.push_str(&format!("Content-ID: <{}>\r\n", cid));
            msg.push_str("Content-Disposition: inline\r\n");
        } else {
            msg.push_str(&format!("Content-Disposition: attachment; filename=\"{}\"\r\n", att.filename));
        }
        msg.push_str("Content-Transfer-Encoding: base64\r\n\r\n");
        msg.push_str(&base64_encode_mime(&att.data));
        msg.push_str("\r\n");
    }
}

impl Default for EmailBuilder {
    fn default() -> Self { Self::new() }
}

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum EmailError {
    MissingFrom,
    MissingTo,
    InvalidAddress(String),
}

impl fmt::Display for EmailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmailError::MissingFrom => write!(f, "From address is required"),
            EmailError::MissingTo => write!(f, "At least one To address is required"),
            EmailError::InvalidAddress(a) => write!(f, "Invalid address: {a}"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_email() {
        let addr = EmailAddress::new("user@example.com").unwrap();
        assert_eq!(addr.local, "user");
        assert_eq!(addr.domain, "example.com");
        assert!(addr.name.is_none());
    }

    #[test]
    fn parse_email_with_name() {
        let addr = EmailAddress::parse("\"John Doe\" <john@example.com>").unwrap();
        assert_eq!(addr.name.as_deref(), Some("John Doe"));
        assert_eq!(addr.local, "john");
        assert_eq!(addr.domain, "example.com");
    }

    #[test]
    fn parse_email_angle_brackets_only() {
        let addr = EmailAddress::parse("<alice@example.org>").unwrap();
        assert_eq!(addr.local, "alice");
        assert!(addr.name.is_none());
    }

    #[test]
    fn parse_invalid_email() {
        assert!(EmailAddress::new("nodomain").is_none());
        assert!(EmailAddress::new("@example.com").is_none());
        assert!(EmailAddress::new("user@").is_none());
    }

    #[test]
    fn email_rfc5322_format() {
        let addr = EmailAddress::with_name("Alice", "alice@example.com").unwrap();
        assert_eq!(addr.to_rfc5322(), "\"Alice\" <alice@example.com>");
    }

    #[test]
    fn email_bare_format() {
        let addr = EmailAddress::new("bob@example.com").unwrap();
        assert_eq!(addr.to_rfc5322(), "bob@example.com");
    }

    #[test]
    fn email_display() {
        let addr = EmailAddress::new("test@example.com").unwrap();
        assert_eq!(format!("{addr}"), "test@example.com");
    }

    #[test]
    fn base64_encode_simple() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"Man"), "TWFu");
    }

    #[test]
    fn base64_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn base64_mime_wrapping() {
        let data = vec![0u8; 100];
        let encoded = base64_encode_mime(&data);
        for line in encoded.split("\r\n") {
            assert!(line.len() <= 76);
        }
    }

    #[test]
    fn quoted_printable_ascii() {
        let result = quoted_printable_encode(b"Hello World");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn quoted_printable_special() {
        let result = quoted_printable_encode(b"price = $10");
        assert!(result.contains("=3D"));
    }

    #[test]
    fn quoted_printable_high_bytes() {
        let result = quoted_printable_encode(&[0xC3, 0xA9]);
        assert_eq!(result, "=C3=A9");
    }

    #[test]
    fn message_id_generation() {
        let mid = generate_message_id("example.com", "abc123");
        assert_eq!(mid, "<abc123@example.com>");
    }

    #[test]
    fn build_simple_text_email() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("recipient@example.com").unwrap())
            .subject("Test")
            .text_body("Hello, World!")
            .build()
            .unwrap();
        assert!(email.contains("From: sender@example.com"));
        assert!(email.contains("To: recipient@example.com"));
        assert!(email.contains("Subject: Test"));
        assert!(email.contains("text/plain"));
        assert!(email.contains("MIME-Version: 1.0"));
    }

    #[test]
    fn build_html_email() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("recipient@example.com").unwrap())
            .subject("HTML Test")
            .html_body("<h1>Hello</h1>")
            .build()
            .unwrap();
        assert!(email.contains("text/html"));
    }

    #[test]
    fn build_multipart_alternative() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("recipient@example.com").unwrap())
            .subject("Multi")
            .text_body("Plain text")
            .html_body("<p>HTML</p>")
            .build()
            .unwrap();
        assert!(email.contains("multipart/alternative"));
        assert!(email.contains("text/plain"));
        assert!(email.contains("text/html"));
    }

    #[test]
    fn build_with_attachment() {
        let att = Attachment::new("file.txt", "text/plain", b"content".to_vec());
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("recipient@example.com").unwrap())
            .subject("With Attachment")
            .text_body("See attached")
            .attachment(att)
            .build()
            .unwrap();
        assert!(email.contains("multipart/mixed"));
        assert!(email.contains("Content-Disposition: attachment"));
        assert!(email.contains("file.txt"));
    }

    #[test]
    fn build_with_inline_image() {
        let att = Attachment::inline_image("logo.png", "image/png", vec![0x89, 0x50, 0x4E], "logo1");
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("recipient@example.com").unwrap())
            .subject("Inline")
            .html_body("<img src=\"cid:logo1\">")
            .attachment(att)
            .build()
            .unwrap();
        assert!(email.contains("Content-ID: <logo1>"));
        assert!(email.contains("Content-Disposition: inline"));
    }

    #[test]
    fn build_with_cc_and_reply_to() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("to@example.com").unwrap())
            .cc(EmailAddress::new("cc@example.com").unwrap())
            .reply_to(EmailAddress::new("reply@example.com").unwrap())
            .subject("CC Test")
            .text_body("test")
            .build()
            .unwrap();
        assert!(email.contains("Cc: cc@example.com"));
        assert!(email.contains("Reply-To: reply@example.com"));
    }

    #[test]
    fn build_with_message_id() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("to@example.com").unwrap())
            .subject("MID")
            .message_id("<unique123@example.com>")
            .text_body("test")
            .build()
            .unwrap();
        assert!(email.contains("Message-ID: <unique123@example.com>"));
    }

    #[test]
    fn build_with_custom_header() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("to@example.com").unwrap())
            .subject("Custom")
            .header("X-Custom", "value123")
            .text_body("test")
            .build()
            .unwrap();
        assert!(email.contains("X-Custom: value123"));
    }

    #[test]
    fn build_missing_from_error() {
        let result = EmailBuilder::new()
            .to(EmailAddress::new("to@example.com").unwrap())
            .subject("No From")
            .build();
        assert_eq!(result, Err(EmailError::MissingFrom));
    }

    #[test]
    fn build_missing_to_error() {
        let result = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .subject("No To")
            .build();
        assert_eq!(result, Err(EmailError::MissingTo));
    }

    #[test]
    fn multiple_recipients() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("a@example.com").unwrap())
            .to(EmailAddress::new("b@example.com").unwrap())
            .subject("Multi To")
            .text_body("test")
            .build()
            .unwrap();
        assert!(email.contains("a@example.com"));
        assert!(email.contains("b@example.com"));
    }

    #[test]
    fn attachment_creation() {
        let att = Attachment::new("report.pdf", "application/pdf", vec![1, 2, 3]);
        assert_eq!(att.filename, "report.pdf");
        assert_eq!(att.content_type, "application/pdf");
        assert!(att.content_id.is_none());
    }

    #[test]
    fn inline_attachment_creation() {
        let att = Attachment::inline_image("img.jpg", "image/jpeg", vec![1, 2], "img001");
        assert_eq!(att.content_id.as_deref(), Some("img001"));
    }

    #[test]
    fn email_address_email_method() {
        let addr = EmailAddress::new("user@example.com").unwrap();
        assert_eq!(addr.email(), "user@example.com");
    }

    #[test]
    fn error_display() {
        assert_eq!(format!("{}", EmailError::MissingFrom), "From address is required");
        assert_eq!(format!("{}", EmailError::MissingTo), "At least one To address is required");
    }

    #[test]
    fn content_type_strings() {
        assert_eq!(ContentType::TextPlain.as_str(), "text/plain; charset=utf-8");
        assert_eq!(ContentType::TextHtml.as_str(), "text/html; charset=utf-8");
        assert_eq!(ContentType::Custom("application/json".into()).as_str(), "application/json");
    }

    #[test]
    fn transfer_encoding_strings() {
        assert_eq!(TransferEncoding::SevenBit.as_str(), "7bit");
        assert_eq!(TransferEncoding::Base64.as_str(), "base64");
        assert_eq!(TransferEncoding::QuotedPrintable.as_str(), "quoted-printable");
    }

    #[test]
    fn build_empty_body_email() {
        let email = EmailBuilder::new()
            .from(EmailAddress::new("sender@example.com").unwrap())
            .to(EmailAddress::new("recipient@example.com").unwrap())
            .subject("Empty")
            .build()
            .unwrap();
        assert!(email.contains("text/plain"));
    }

    #[test]
    fn base64_rfc_test_vectors() {
        // From RFC 4648
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
