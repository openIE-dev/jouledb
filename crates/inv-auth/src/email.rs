//! Email delivery for passwordless authentication.
//!
//! Two modes:
//! - `SmtpEmailSender`: Real SMTP delivery via lettre (production)
//! - `DevEmailSender`: Captures emails in memory (development/testing)

use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Email types
// ---------------------------------------------------------------------------

/// An email message to be sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    pub to: String,
    pub subject: String,
    pub body_html: String,
    pub body_text: String,
}

/// Configuration for SMTP email delivery.
#[derive(Debug, Clone)]
pub struct SmtpConfig {
    /// SMTP server hostname (e.g., "smtp.gmail.com", "email-smtp.us-east-1.amazonaws.com").
    pub host: String,
    /// SMTP port (587 for STARTTLS, 465 for implicit TLS).
    pub port: u16,
    /// SMTP username (often the full email address).
    pub username: String,
    /// SMTP password or app-specific password.
    pub password: String,
    /// "From" email address.
    pub from_email: String,
    /// "From" display name.
    pub from_name: String,
    /// Whether to use STARTTLS (port 587) or implicit TLS (port 465).
    pub starttls: bool,
}

impl Default for SmtpConfig {
    fn default() -> Self {
        Self {
            host: "localhost".into(),
            port: 587,
            username: String::new(),
            password: String::new(),
            from_email: "noreply@invisible.dev".into(),
            from_name: "Invisible Infrastructure".into(),
            starttls: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum EmailError {
    #[error("SMTP transport error: {0}")]
    Transport(String),
    #[error("invalid email address: {0}")]
    InvalidAddress(String),
    #[error("email service not configured")]
    NotConfigured,
}

// ---------------------------------------------------------------------------
// Email sender trait
// ---------------------------------------------------------------------------

/// Trait for sending emails. Implemented by both SMTP and dev senders.
pub trait EmailSender: Send + Sync {
    /// Send an email message.
    fn send(&self, message: &EmailMessage) -> Result<(), EmailError>;

    /// Check if this is a development/capture sender.
    fn is_dev_mode(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Dev email sender (captures emails for testing)
// ---------------------------------------------------------------------------

/// Development email sender that captures all emails in memory.
/// Perfect for testing — no SMTP server required.
pub struct DevEmailSender {
    captured: Arc<RwLock<Vec<EmailMessage>>>,
}

impl DevEmailSender {
    pub fn new() -> Self {
        Self {
            captured: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get all captured emails.
    pub fn captured_emails(&self) -> Vec<EmailMessage> {
        self.captured.read().unwrap().clone()
    }

    /// Get the last captured email.
    pub fn last_email(&self) -> Option<EmailMessage> {
        self.captured.read().unwrap().last().cloned()
    }

    /// Clear captured emails.
    pub fn clear(&self) {
        self.captured.write().unwrap().clear();
    }

    /// Count of captured emails.
    pub fn count(&self) -> usize {
        self.captured.read().unwrap().len()
    }
}

impl Default for DevEmailSender {
    fn default() -> Self {
        Self::new()
    }
}

impl EmailSender for DevEmailSender {
    fn send(&self, message: &EmailMessage) -> Result<(), EmailError> {
        tracing::info!(
            to = message.to,
            subject = message.subject,
            "[dev-email] Captured email (not sent)"
        );
        self.captured.write().unwrap().push(message.clone());
        Ok(())
    }

    fn is_dev_mode(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// SMTP email sender (real delivery via lettre)
// ---------------------------------------------------------------------------

/// Production SMTP email sender using lettre.
pub struct SmtpEmailSender {
    config: SmtpConfig,
}

impl SmtpEmailSender {
    pub fn new(config: SmtpConfig) -> Self {
        Self { config }
    }
}

impl EmailSender for SmtpEmailSender {
    fn send(&self, message: &EmailMessage) -> Result<(), EmailError> {
        use lettre::message::Mailbox;
        use lettre::message::header::ContentType;
        use lettre::transport::smtp::authentication::Credentials;
        use lettre::{Message, SmtpTransport, Transport};

        let from: Mailbox = format!("{} <{}>", self.config.from_name, self.config.from_email)
            .parse()
            .map_err(|e: lettre::address::AddressError| {
                EmailError::InvalidAddress(e.to_string())
            })?;
        let to: Mailbox = message
            .to
            .parse()
            .map_err(|e: lettre::address::AddressError| {
                EmailError::InvalidAddress(e.to_string())
            })?;

        let email = Message::builder()
            .from(from)
            .to(to)
            .subject(&message.subject)
            .header(ContentType::TEXT_HTML)
            .body(message.body_html.clone())
            .map_err(|e| EmailError::Transport(e.to_string()))?;

        let creds = Credentials::new(self.config.username.clone(), self.config.password.clone());

        let transport = if self.config.starttls {
            SmtpTransport::starttls_relay(&self.config.host)
                .map_err(|e| EmailError::Transport(e.to_string()))?
                .port(self.config.port)
                .credentials(creds)
                .build()
        } else {
            SmtpTransport::relay(&self.config.host)
                .map_err(|e| EmailError::Transport(e.to_string()))?
                .port(self.config.port)
                .credentials(creds)
                .build()
        };

        transport
            .send(&email)
            .map_err(|e| EmailError::Transport(e.to_string()))?;

        tracing::info!(
            to = message.to,
            subject = message.subject,
            "Email sent via SMTP"
        );
        Ok(())
    }

    fn is_dev_mode(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Email template helpers
// ---------------------------------------------------------------------------

/// Build a magic link email.
pub fn magic_link_email(email: &str, base_url: &str, nonce: &str, signature: &str) -> EmailMessage {
    let link = format!(
        "{}/auth/verify?email={}&nonce={}&signature={}",
        base_url,
        urlencoded(email),
        nonce,
        signature
    );

    EmailMessage {
        to: email.to_string(),
        subject: "Sign in to Invisible Infrastructure".into(),
        body_html: format!(
            r#"<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 20px;">
<h2 style="color: #1a1a2e;">Sign in to Invisible Infrastructure</h2>
<p>Click the button below to sign in. This link expires in 15 minutes.</p>
<a href="{link}" style="display: inline-block; background: #6366f1; color: white; padding: 12px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 16px 0;">Sign In</a>
<p style="color: #666; font-size: 13px;">If you didn't request this, you can safely ignore this email.</p>
<p style="color: #999; font-size: 11px;">Or copy this link: {link}</p>
</div>"#
        ),
        body_text: format!(
            "Sign in to Invisible Infrastructure\n\nClick this link to sign in (expires in 15 minutes):\n{}\n\nIf you didn't request this, ignore this email.",
            link
        ),
    }
}

/// Build an OTP email.
pub fn otp_email(email: &str, code: &str) -> EmailMessage {
    EmailMessage {
        to: email.to_string(),
        subject: "Your verification code for Invisible Infrastructure".into(),
        body_html: format!(
            r#"<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 20px;">
<h2 style="color: #1a1a2e;">Verification Code</h2>
<p>Use the code below to verify your identity. It expires in 5 minutes.</p>
<div style="background: #f1f5f9; padding: 20px; border-radius: 12px; text-align: center; margin: 24px 0;">
<span style="font-size: 36px; font-weight: 700; letter-spacing: 8px; color: #1a1a2e;">{code}</span>
</div>
<p style="color: #666; font-size: 13px;">If you didn't request this code, you can safely ignore this email.</p>
</div>"#
        ),
        body_text: format!(
            "Verification Code\n\nYour code: {}\n\nThis code expires in 5 minutes. If you didn't request this, ignore this email.",
            code
        ),
    }
}

/// Simple URL encoding for email addresses in links.
fn urlencoded(s: &str) -> String {
    s.replace('@', "%40").replace('+', "%2B")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_sender_captures_emails() {
        let sender = DevEmailSender::new();
        assert_eq!(sender.count(), 0);
        assert!(sender.is_dev_mode());

        let msg = EmailMessage {
            to: "alice@example.com".into(),
            subject: "Test".into(),
            body_html: "<p>Hello</p>".into(),
            body_text: "Hello".into(),
        };
        sender.send(&msg).unwrap();

        assert_eq!(sender.count(), 1);
        let captured = sender.last_email().unwrap();
        assert_eq!(captured.to, "alice@example.com");
        assert_eq!(captured.subject, "Test");
    }

    #[test]
    fn dev_sender_clear() {
        let sender = DevEmailSender::new();
        sender
            .send(&EmailMessage {
                to: "a@b.com".into(),
                subject: "S".into(),
                body_html: "H".into(),
                body_text: "T".into(),
            })
            .unwrap();
        assert_eq!(sender.count(), 1);
        sender.clear();
        assert_eq!(sender.count(), 0);
    }

    #[test]
    fn magic_link_email_template() {
        let msg = magic_link_email(
            "alice@example.com",
            "https://app.invisible.dev",
            "abc123",
            "sig456",
        );
        assert_eq!(msg.to, "alice@example.com");
        assert!(msg.subject.contains("Sign in"));
        assert!(msg.body_html.contains("abc123"));
        assert!(msg.body_html.contains("sig456"));
        assert!(msg.body_html.contains("alice%40example.com"));
        assert!(msg.body_text.contains("abc123"));
    }

    #[test]
    fn otp_email_template() {
        let msg = otp_email("bob@example.com", "123456");
        assert_eq!(msg.to, "bob@example.com");
        assert!(msg.subject.contains("verification code"));
        assert!(msg.body_html.contains("123456"));
        assert!(msg.body_text.contains("123456"));
    }

    #[test]
    fn dev_sender_default() {
        let sender = DevEmailSender::default();
        assert!(sender.is_dev_mode());
        assert_eq!(sender.count(), 0);
    }

    #[test]
    fn smtp_config_default() {
        let config = SmtpConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 587);
        assert_eq!(config.from_email, "noreply@invisible.dev");
        assert!(config.starttls);
    }

    #[test]
    fn multiple_captured_emails() {
        let sender = DevEmailSender::new();
        for i in 0..5 {
            sender
                .send(&EmailMessage {
                    to: format!("user{i}@example.com"),
                    subject: format!("Subject {i}"),
                    body_html: format!("<p>{i}</p>"),
                    body_text: format!("{i}"),
                })
                .unwrap();
        }
        assert_eq!(sender.count(), 5);
        let all = sender.captured_emails();
        assert_eq!(all.len(), 5);
        assert_eq!(all[0].to, "user0@example.com");
        assert_eq!(all[4].to, "user4@example.com");
    }
}
