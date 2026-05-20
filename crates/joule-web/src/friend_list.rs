//! Friend list management — requests, blocking, groups, mutual friends.
//!
//! Replaces Steam friend-list / Discord relationships with pure Rust.
//! Send/accept/reject/cancel friend requests, block/unblock, friend groups,
//! mutual-friend suggestions, online friends filter, activity feed.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FriendError {
    AlreadyFriends(String, String),
    NotFriends(String, String),
    RequestAlreadySent(String, String),
    RequestNotFound(String, String),
    UserBlocked(String),
    SelfFriend,
    GroupNotFound(String),
}

impl fmt::Display for FriendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyFriends(a, b) => write!(f, "{a} and {b} are already friends"),
            Self::NotFriends(a, b) => write!(f, "{a} and {b} are not friends"),
            Self::RequestAlreadySent(a, b) => write!(f, "request from {a} to {b} already exists"),
            Self::RequestNotFound(a, b) => write!(f, "no request from {a} to {b}"),
            Self::UserBlocked(u) => write!(f, "user is blocked: {u}"),
            Self::SelfFriend => write!(f, "cannot friend yourself"),
            Self::GroupNotFound(g) => write!(f, "group not found: {g}"),
        }
    }
}

impl std::error::Error for FriendError {}

// ── RequestStatus ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Pending,
    Accepted,
    Rejected,
}

impl fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Accepted => write!(f, "accepted"),
            Self::Rejected => write!(f, "rejected"),
        }
    }
}

// ── FriendRequest ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FriendRequest {
    pub from: String,
    pub to: String,
    pub status: RequestStatus,
    pub sent_at: u64,
}

impl FriendRequest {
    pub fn new(from: &str, to: &str, sent_at: u64) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            status: RequestStatus::Pending,
            sent_at,
        }
    }
}

impl fmt::Display for FriendRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {} ({})", self.from, self.to, self.status)
    }
}

// ── FriendActivity ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FriendActivity {
    pub user_id: String,
    pub last_seen: u64,
    pub current_activity: Option<String>,
}

impl FriendActivity {
    pub fn new(user_id: &str, last_seen: u64) -> Self {
        Self {
            user_id: user_id.to_string(),
            last_seen,
            current_activity: None,
        }
    }

    pub fn with_activity(mut self, activity: &str) -> Self {
        self.current_activity = Some(activity.to_string());
        self
    }
}

// ── FriendList ──────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct FriendList {
    friends: HashSet<String>,
    blocked: HashSet<String>,
    groups: HashMap<String, HashSet<String>>,
    activities: HashMap<String, FriendActivity>,
    online_set: HashSet<String>,
}

impl FriendList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_friend(&mut self, user_id: &str) {
        self.friends.insert(user_id.to_string());
    }

    pub fn remove_friend(&mut self, user_id: &str) -> bool {
        // Also remove from all groups
        for group in self.groups.values_mut() {
            group.remove(user_id);
        }
        self.friends.remove(user_id)
    }

    pub fn is_friend(&self, user_id: &str) -> bool {
        self.friends.contains(user_id)
    }

    pub fn block(&mut self, user_id: &str) {
        self.blocked.insert(user_id.to_string());
        self.friends.remove(user_id);
    }

    pub fn unblock(&mut self, user_id: &str) {
        self.blocked.remove(user_id);
    }

    pub fn is_blocked(&self, user_id: &str) -> bool {
        self.blocked.contains(user_id)
    }

    pub fn friend_count(&self) -> usize {
        self.friends.len()
    }

    pub fn friends(&self) -> &HashSet<String> {
        &self.friends
    }

    pub fn create_group(&mut self, name: &str) {
        self.groups.entry(name.to_string()).or_default();
    }

    pub fn add_to_group(&mut self, group: &str, user_id: &str) -> Result<(), FriendError> {
        let g = self.groups.get_mut(group).ok_or_else(|| FriendError::GroupNotFound(group.to_string()))?;
        g.insert(user_id.to_string());
        Ok(())
    }

    pub fn group_members(&self, group: &str) -> Option<&HashSet<String>> {
        self.groups.get(group)
    }

    pub fn set_activity(&mut self, user_id: &str, activity: FriendActivity) {
        self.activities.insert(user_id.to_string(), activity);
    }

    pub fn get_activity(&self, user_id: &str) -> Option<&FriendActivity> {
        self.activities.get(user_id)
    }

    pub fn set_online(&mut self, user_id: &str, online: bool) {
        if online {
            self.online_set.insert(user_id.to_string());
        } else {
            self.online_set.remove(user_id);
        }
    }

    pub fn online_friends(&self) -> Vec<&str> {
        self.friends.iter()
            .filter(|f| self.online_set.contains(f.as_str()))
            .map(|f| f.as_str())
            .collect()
    }
}

// ── FriendManager ───────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct FriendManager {
    lists: HashMap<String, FriendList>,
    requests: Vec<FriendRequest>,
}

impl FriendManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_list(&mut self, user_id: &str) -> &mut FriendList {
        self.lists.entry(user_id.to_string()).or_default()
    }

    pub fn get_list(&self, user_id: &str) -> Option<&FriendList> {
        self.lists.get(user_id)
    }

    pub fn send_request(&mut self, from: &str, to: &str, sent_at: u64) -> Result<(), FriendError> {
        if from == to {
            return Err(FriendError::SelfFriend);
        }
        // Check if blocked
        if let Some(target_list) = self.lists.get(to) {
            if target_list.is_blocked(from) {
                return Err(FriendError::UserBlocked(from.to_string()));
            }
        }
        // Check already friends
        if let Some(from_list) = self.lists.get(from) {
            if from_list.is_friend(to) {
                return Err(FriendError::AlreadyFriends(from.to_string(), to.to_string()));
            }
        }
        // Check duplicate request
        if self.requests.iter().any(|r| r.from == from && r.to == to && r.status == RequestStatus::Pending) {
            return Err(FriendError::RequestAlreadySent(from.to_string(), to.to_string()));
        }
        self.requests.push(FriendRequest::new(from, to, sent_at));
        Ok(())
    }

    pub fn accept_request(&mut self, from: &str, to: &str) -> Result<(), FriendError> {
        let req = self.requests.iter_mut()
            .find(|r| r.from == from && r.to == to && r.status == RequestStatus::Pending)
            .ok_or_else(|| FriendError::RequestNotFound(from.to_string(), to.to_string()))?;
        req.status = RequestStatus::Accepted;

        self.ensure_list(from).add_friend(to);
        self.ensure_list(to).add_friend(from);
        Ok(())
    }

    pub fn reject_request(&mut self, from: &str, to: &str) -> Result<(), FriendError> {
        let req = self.requests.iter_mut()
            .find(|r| r.from == from && r.to == to && r.status == RequestStatus::Pending)
            .ok_or_else(|| FriendError::RequestNotFound(from.to_string(), to.to_string()))?;
        req.status = RequestStatus::Rejected;
        Ok(())
    }

    pub fn cancel_request(&mut self, from: &str, to: &str) -> Result<(), FriendError> {
        let idx = self.requests.iter()
            .position(|r| r.from == from && r.to == to && r.status == RequestStatus::Pending)
            .ok_or_else(|| FriendError::RequestNotFound(from.to_string(), to.to_string()))?;
        self.requests.remove(idx);
        Ok(())
    }

    pub fn remove_friend(&mut self, user_a: &str, user_b: &str) -> Result<(), FriendError> {
        let list_a = self.lists.get_mut(user_a).ok_or_else(|| FriendError::NotFriends(user_a.to_string(), user_b.to_string()))?;
        if !list_a.remove_friend(user_b) {
            return Err(FriendError::NotFriends(user_a.to_string(), user_b.to_string()));
        }
        if let Some(list_b) = self.lists.get_mut(user_b) {
            list_b.remove_friend(user_a);
        }
        Ok(())
    }

    pub fn block_user(&mut self, blocker: &str, target: &str) {
        self.ensure_list(blocker).block(target);
        // Remove friendship in both directions
        if let Some(target_list) = self.lists.get_mut(target) {
            target_list.remove_friend(blocker);
        }
    }

    pub fn unblock_user(&mut self, blocker: &str, target: &str) {
        if let Some(list) = self.lists.get_mut(blocker) {
            list.unblock(target);
        }
    }

    pub fn pending_requests_for(&self, user_id: &str) -> Vec<&FriendRequest> {
        self.requests.iter()
            .filter(|r| r.to == user_id && r.status == RequestStatus::Pending)
            .collect()
    }

    pub fn mutual_friends(&self, user_a: &str, user_b: &str) -> Vec<String> {
        let a_friends = match self.lists.get(user_a) {
            Some(l) => l.friends(),
            None => return Vec::new(),
        };
        let b_friends = match self.lists.get(user_b) {
            Some(l) => l.friends(),
            None => return Vec::new(),
        };
        a_friends.intersection(b_friends).cloned().collect()
    }

    /// Suggest friends based on mutual connections.
    pub fn suggest_friends(&self, user_id: &str, max: usize) -> Vec<(String, usize)> {
        let my_friends = match self.lists.get(user_id) {
            Some(l) => l.friends().clone(),
            None => return Vec::new(),
        };
        let mut scores: HashMap<String, usize> = HashMap::new();
        for friend in &my_friends {
            if let Some(their_list) = self.lists.get(friend) {
                for fof in their_list.friends() {
                    if fof != user_id && !my_friends.contains(fof) {
                        *scores.entry(fof.clone()).or_default() += 1;
                    }
                }
            }
        }
        let mut suggestions: Vec<_> = scores.into_iter().collect();
        suggestions.sort_by(|a, b| b.1.cmp(&a.1));
        suggestions.truncate(max);
        suggestions
    }

    pub fn import_friends(&mut self, user_id: &str, friend_ids: &[&str]) {
        // First pass: add friends to user's list.
        {
            let list = self.ensure_list(user_id);
            for fid in friend_ids {
                list.add_friend(fid);
            }
        }
        // Second pass: add user to each friend's list.
        for fid in friend_ids {
            self.ensure_list(fid).add_friend(user_id);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr_with_friends() -> FriendManager {
        let mut m = FriendManager::new();
        m.send_request("alice", "bob", 100).unwrap();
        m.accept_request("alice", "bob").unwrap();
        m
    }

    #[test]
    fn test_send_and_accept() {
        let m = mgr_with_friends();
        assert!(m.get_list("alice").unwrap().is_friend("bob"));
        assert!(m.get_list("bob").unwrap().is_friend("alice"));
    }

    #[test]
    fn test_reject_request() {
        let mut m = FriendManager::new();
        m.send_request("alice", "bob", 100).unwrap();
        m.reject_request("alice", "bob").unwrap();
        assert!(m.get_list("bob").is_none() || !m.get_list("bob").unwrap().is_friend("alice"));
    }

    #[test]
    fn test_cancel_request() {
        let mut m = FriendManager::new();
        m.send_request("alice", "bob", 100).unwrap();
        m.cancel_request("alice", "bob").unwrap();
        assert!(m.pending_requests_for("bob").is_empty());
    }

    #[test]
    fn test_self_friend_error() {
        let mut m = FriendManager::new();
        assert_eq!(m.send_request("alice", "alice", 100), Err(FriendError::SelfFriend));
    }

    #[test]
    fn test_duplicate_request() {
        let mut m = FriendManager::new();
        m.send_request("alice", "bob", 100).unwrap();
        assert!(m.send_request("alice", "bob", 101).is_err());
    }

    #[test]
    fn test_already_friends() {
        let mut m = mgr_with_friends();
        assert!(m.send_request("alice", "bob", 200).is_err());
    }

    #[test]
    fn test_remove_friend() {
        let mut m = mgr_with_friends();
        m.remove_friend("alice", "bob").unwrap();
        assert!(!m.get_list("alice").unwrap().is_friend("bob"));
    }

    #[test]
    fn test_block_user() {
        let mut m = mgr_with_friends();
        m.block_user("alice", "bob");
        assert!(m.get_list("alice").unwrap().is_blocked("bob"));
        assert!(!m.get_list("alice").unwrap().is_friend("bob"));
    }

    #[test]
    fn test_blocked_user_cannot_request() {
        let mut m = FriendManager::new();
        m.block_user("bob", "alice");
        assert!(m.send_request("alice", "bob", 100).is_err());
    }

    #[test]
    fn test_unblock() {
        let mut m = FriendManager::new();
        m.block_user("alice", "bob");
        m.unblock_user("alice", "bob");
        assert!(!m.get_list("alice").unwrap().is_blocked("bob"));
    }

    #[test]
    fn test_mutual_friends() {
        let mut m = mgr_with_friends();
        m.send_request("alice", "charlie", 200).unwrap();
        m.accept_request("alice", "charlie").unwrap();
        m.send_request("bob", "charlie", 201).unwrap();
        m.accept_request("bob", "charlie").unwrap();
        let mutual = m.mutual_friends("alice", "bob");
        assert!(mutual.contains(&"charlie".to_string()));
    }

    #[test]
    fn test_friend_groups() {
        let mut list = FriendList::new();
        list.add_friend("bob");
        list.create_group("close");
        list.add_to_group("close", "bob").unwrap();
        assert!(list.group_members("close").unwrap().contains("bob"));
    }

    #[test]
    fn test_group_not_found() {
        let mut list = FriendList::new();
        assert!(list.add_to_group("nope", "bob").is_err());
    }

    #[test]
    fn test_online_friends() {
        let mut list = FriendList::new();
        list.add_friend("bob");
        list.add_friend("charlie");
        list.set_online("bob", true);
        let online = list.online_friends();
        assert_eq!(online.len(), 1);
        assert!(online.contains(&"bob"));
    }

    #[test]
    fn test_friend_activity() {
        let mut list = FriendList::new();
        let activity = FriendActivity::new("bob", 1000).with_activity("Playing Chess");
        list.set_activity("bob", activity);
        let a = list.get_activity("bob").unwrap();
        assert_eq!(a.current_activity.as_deref(), Some("Playing Chess"));
    }

    #[test]
    fn test_suggest_friends() {
        let mut m = FriendManager::new();
        // alice-bob, alice-charlie, bob-dave, charlie-dave
        m.import_friends("alice", &["bob", "charlie"]);
        m.import_friends("bob", &["dave"]);
        m.import_friends("charlie", &["dave"]);
        let suggestions = m.suggest_friends("alice", 5);
        assert!(suggestions.iter().any(|(u, _)| u == "dave"));
    }

    #[test]
    fn test_import_friends() {
        let mut m = FriendManager::new();
        m.import_friends("alice", &["bob", "charlie"]);
        assert_eq!(m.get_list("alice").unwrap().friend_count(), 2);
        assert!(m.get_list("bob").unwrap().is_friend("alice"));
    }

    #[test]
    fn test_pending_requests() {
        let mut m = FriendManager::new();
        m.send_request("alice", "bob", 100).unwrap();
        m.send_request("charlie", "bob", 101).unwrap();
        assert_eq!(m.pending_requests_for("bob").len(), 2);
    }

    #[test]
    fn test_display_request() {
        let r = FriendRequest::new("alice", "bob", 100);
        let s = format!("{r}");
        assert!(s.contains("alice"));
        assert!(s.contains("pending"));
    }

    #[test]
    fn test_remove_friend_not_friends() {
        let mut m = FriendManager::new();
        assert!(m.remove_friend("alice", "bob").is_err());
    }
}
