use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::AuthError;
use crate::rbac::Role;

/// An authenticated user session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    pub id: Uuid,
    /// Organization the session belongs to.
    pub org: String,
    /// User who owns the session.
    pub user: String,
    /// Role assigned to the user for this session.
    pub role: Role,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// When the session was last active (last request).
    pub last_activity: DateTime<Utc>,
    /// IP address the session originated from.
    pub ip_addr: Option<IpAddr>,
    /// User-Agent string from the client.
    pub user_agent: Option<String>,
    /// Whether the session is still active.
    pub active: bool,
}

/// Manages user sessions for Invisible Infrastructure.
#[derive(Clone)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<Uuid, Session>>>,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new session.
    pub fn create(
        &self,
        org: &str,
        user: &str,
        role: Role,
        ip_addr: Option<IpAddr>,
        user_agent: Option<String>,
    ) -> Session {
        let now = Utc::now();
        let session = Session {
            id: Uuid::new_v4(),
            org: org.to_string(),
            user: user.to_string(),
            role,
            created_at: now,
            last_activity: now,
            ip_addr,
            user_agent,
            active: true,
        };

        {
            let mut sessions = self.sessions.write().unwrap();
            sessions.insert(session.id, session.clone());
        }

        info!(
            session_id = %session.id,
            org = org,
            user = user,
            "Session created"
        );

        session
    }

    /// Get a session by its ID.
    pub fn get(&self, session_id: Uuid) -> Result<Session, AuthError> {
        let sessions = self.sessions.read().unwrap();
        sessions
            .get(&session_id)
            .cloned()
            .ok_or(AuthError::SessionNotFound)
    }

    /// List all active sessions for a user within an org.
    pub fn list_for_user(&self, org: &str, user: &str) -> Vec<Session> {
        let sessions = self.sessions.read().unwrap();
        sessions
            .values()
            .filter(|s| s.org == org && s.user == user && s.active)
            .cloned()
            .collect()
    }

    /// List all active sessions for an organization.
    pub fn list_for_org(&self, org: &str) -> Vec<Session> {
        let sessions = self.sessions.read().unwrap();
        sessions
            .values()
            .filter(|s| s.org == org && s.active)
            .cloned()
            .collect()
    }

    /// Update the last activity timestamp for a session (keep-alive / touch).
    pub fn touch(&self, session_id: Uuid) -> Result<(), AuthError> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(&session_id)
            .ok_or(AuthError::SessionNotFound)?;

        if !session.active {
            return Err(AuthError::SessionNotFound);
        }

        session.last_activity = Utc::now();
        Ok(())
    }

    /// Terminate a specific session.
    pub fn terminate(&self, session_id: Uuid) -> Result<(), AuthError> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(&session_id)
            .ok_or(AuthError::SessionNotFound)?;
        session.active = false;
        info!(session_id = %session_id, "Session terminated");
        Ok(())
    }

    /// Terminate all sessions for a user within an org.
    ///
    /// Returns the number of sessions terminated.
    pub fn terminate_all_for_user(&self, org: &str, user: &str) -> usize {
        let mut sessions = self.sessions.write().unwrap();
        let mut count = 0;
        for session in sessions.values_mut() {
            if session.org == org && session.user == user && session.active {
                session.active = false;
                count += 1;
            }
        }
        if count > 0 {
            info!(
                org = org,
                user = user,
                count = count,
                "All user sessions terminated"
            );
        }
        count
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_session() {
        let mgr = SessionManager::new();
        let session = mgr.create("acme", "user-1", Role::Operator, None, None);

        assert_eq!(session.org, "acme");
        assert_eq!(session.user, "user-1");
        assert_eq!(session.role, Role::Operator);
        assert!(session.active);

        let retrieved = mgr.get(session.id).unwrap();
        assert_eq!(retrieved.id, session.id);
    }

    #[test]
    fn get_nonexistent_session_fails() {
        let mgr = SessionManager::new();
        let err = mgr.get(Uuid::new_v4()).unwrap_err();
        assert!(matches!(err, AuthError::SessionNotFound));
    }

    #[test]
    fn list_for_user_returns_active_sessions() {
        let mgr = SessionManager::new();
        let s1 = mgr.create("acme", "user-1", Role::Operator, None, None);
        let _s2 = mgr.create("acme", "user-1", Role::Operator, None, None);
        let _s3 = mgr.create("acme", "user-2", Role::Viewer, None, None);

        // Terminate one of user-1's sessions
        mgr.terminate(s1.id).unwrap();

        let user1_sessions = mgr.list_for_user("acme", "user-1");
        assert_eq!(user1_sessions.len(), 1);
    }

    #[test]
    fn list_for_org_returns_all_active() {
        let mgr = SessionManager::new();
        mgr.create("acme", "user-1", Role::Operator, None, None);
        mgr.create("acme", "user-2", Role::Viewer, None, None);
        mgr.create("other-org", "user-3", Role::Admin, None, None);

        let acme_sessions = mgr.list_for_org("acme");
        assert_eq!(acme_sessions.len(), 2);

        let other_sessions = mgr.list_for_org("other-org");
        assert_eq!(other_sessions.len(), 1);
    }

    #[test]
    fn touch_updates_last_activity() {
        let mgr = SessionManager::new();
        let session = mgr.create("acme", "user-1", Role::Operator, None, None);
        let original_activity = session.last_activity;

        // Sleep briefly so timestamps differ
        std::thread::sleep(std::time::Duration::from_millis(10));

        mgr.touch(session.id).unwrap();

        let updated = mgr.get(session.id).unwrap();
        assert!(updated.last_activity >= original_activity);
    }

    #[test]
    fn touch_terminated_session_fails() {
        let mgr = SessionManager::new();
        let session = mgr.create("acme", "user-1", Role::Operator, None, None);
        mgr.terminate(session.id).unwrap();

        let err = mgr.touch(session.id).unwrap_err();
        assert!(matches!(err, AuthError::SessionNotFound));
    }

    #[test]
    fn terminate_session() {
        let mgr = SessionManager::new();
        let session = mgr.create("acme", "user-1", Role::Operator, None, None);

        mgr.terminate(session.id).unwrap();

        let retrieved = mgr.get(session.id).unwrap();
        assert!(!retrieved.active);
    }

    #[test]
    fn terminate_all_for_user() {
        let mgr = SessionManager::new();
        mgr.create("acme", "user-1", Role::Operator, None, None);
        mgr.create("acme", "user-1", Role::Operator, None, None);
        mgr.create("acme", "user-2", Role::Viewer, None, None);

        let count = mgr.terminate_all_for_user("acme", "user-1");
        assert_eq!(count, 2);

        let user1_sessions = mgr.list_for_user("acme", "user-1");
        assert!(user1_sessions.is_empty());

        // user-2 is unaffected
        let user2_sessions = mgr.list_for_user("acme", "user-2");
        assert_eq!(user2_sessions.len(), 1);
    }

    #[test]
    fn session_with_ip_and_user_agent() {
        let mgr = SessionManager::new();
        let ip: IpAddr = "192.168.1.1".parse().unwrap();
        let ua = "InvisibleCLI/0.1.0".to_string();

        let session = mgr.create("acme", "user-1", Role::Admin, Some(ip), Some(ua.clone()));

        assert_eq!(session.ip_addr, Some(ip));
        assert_eq!(session.user_agent, Some(ua));
    }

    #[test]
    fn terminate_nonexistent_session_fails() {
        let mgr = SessionManager::new();
        let err = mgr.terminate(Uuid::new_v4()).unwrap_err();
        assert!(matches!(err, AuthError::SessionNotFound));
    }

    #[test]
    fn terminate_all_for_user_with_no_sessions() {
        let mgr = SessionManager::new();
        let count = mgr.terminate_all_for_user("acme", "nonexistent");
        assert_eq!(count, 0);
    }
}
