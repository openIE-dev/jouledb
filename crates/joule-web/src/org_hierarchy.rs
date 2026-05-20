// org_hierarchy.rs — Organization hierarchy: Org with teams,
// Team with members, nested teams via parent_id, member roles,
// permission inheritance, org-level settings with team overrides.

use std::collections::HashMap;

/// A role within a team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Owner,
    Admin,
    Member,
    Viewer,
}

impl Role {
    /// Numeric privilege level (higher = more privilege).
    pub fn level(&self) -> u8 {
        match self {
            Self::Owner => 4,
            Self::Admin => 3,
            Self::Member => 2,
            Self::Viewer => 1,
        }
    }

    pub fn can_manage_members(&self) -> bool {
        self.level() >= Self::Admin.level()
    }

    pub fn can_edit(&self) -> bool {
        self.level() >= Self::Member.level()
    }

    pub fn can_view(&self) -> bool {
        self.level() >= Self::Viewer.level()
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Owner => "owner",
            Self::Admin => "admin",
            Self::Member => "member",
            Self::Viewer => "viewer",
        };
        write!(f, "{s}")
    }
}

/// A member of a team.
#[derive(Debug, Clone)]
pub struct TeamMember {
    pub user_id: String,
    pub display_name: String,
    pub role: Role,
}

/// A team within an organization.
#[derive(Debug, Clone)]
pub struct Team {
    pub id: String,
    pub name: String,
    pub parent_id: Option<String>,
    pub members: Vec<TeamMember>,
    /// Team-level setting overrides (key -> value).
    pub settings: HashMap<String, String>,
}

impl Team {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            parent_id: None,
            members: Vec::new(),
            settings: HashMap::new(),
        }
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    pub fn add_member(&mut self, user_id: &str, display_name: &str, role: Role) {
        self.members.push(TeamMember {
            user_id: user_id.to_string(),
            display_name: display_name.to_string(),
            role,
        });
    }

    pub fn remove_member(&mut self, user_id: &str) -> bool {
        let before = self.members.len();
        self.members.retain(|m| m.user_id != user_id);
        self.members.len() < before
    }

    pub fn find_member(&self, user_id: &str) -> Option<&TeamMember> {
        self.members.iter().find(|m| m.user_id == user_id)
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    pub fn set_setting(&mut self, key: &str, value: &str) {
        self.settings.insert(key.to_string(), value.to_string());
    }

    pub fn get_setting(&self, key: &str) -> Option<&str> {
        self.settings.get(key).map(|s| s.as_str())
    }

    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    pub fn members_with_role(&self, role: Role) -> Vec<&TeamMember> {
        self.members.iter().filter(|m| m.role == role).collect()
    }
}

/// An organization containing teams.
#[derive(Debug, Clone)]
pub struct Org {
    pub id: String,
    pub name: String,
    pub teams: Vec<Team>,
    /// Org-level default settings.
    pub default_settings: HashMap<String, String>,
}

impl Org {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            teams: Vec::new(),
            default_settings: HashMap::new(),
        }
    }

    pub fn add_team(&mut self, team: Team) {
        self.teams.push(team);
    }

    pub fn find_team(&self, team_id: &str) -> Option<&Team> {
        self.teams.iter().find(|t| t.id == team_id)
    }

    pub fn find_team_mut(&mut self, team_id: &str) -> Option<&mut Team> {
        self.teams.iter_mut().find(|t| t.id == team_id)
    }

    pub fn team_count(&self) -> usize {
        self.teams.len()
    }

    /// Set an org-level default setting.
    pub fn set_default(&mut self, key: &str, value: &str) {
        self.default_settings
            .insert(key.to_string(), value.to_string());
    }

    /// Resolve a setting for a team: team override > org default.
    pub fn resolve_setting(&self, team_id: &str, key: &str) -> Option<&str> {
        // Check team override first.
        if let Some(team) = self.find_team(team_id) {
            if let Some(val) = team.get_setting(key) {
                return Some(val);
            }
        }
        // Fall back to org default.
        self.default_settings.get(key).map(|s| s.as_str())
    }

    /// Resolve a setting walking up the team hierarchy:
    /// team override > parent override > ... > org default.
    pub fn resolve_setting_inherited(&self, team_id: &str, key: &str) -> Option<String> {
        let mut current_id = Some(team_id.to_string());
        while let Some(tid) = current_id {
            if let Some(team) = self.find_team(&tid) {
                if let Some(val) = team.get_setting(key) {
                    return Some(val.to_string());
                }
                current_id = team.parent_id.clone();
            } else {
                break;
            }
        }
        self.default_settings.get(key).map(|s| s.to_string())
    }

    /// Get all child teams of a given parent.
    pub fn children_of(&self, parent_id: &str) -> Vec<&Team> {
        self.teams
            .iter()
            .filter(|t| t.parent_id.as_deref() == Some(parent_id))
            .collect()
    }

    /// Get root teams (no parent).
    pub fn root_teams(&self) -> Vec<&Team> {
        self.teams.iter().filter(|t| t.is_root()).collect()
    }

    /// All descendants of a team (recursive).
    pub fn descendants_of(&self, team_id: &str) -> Vec<&Team> {
        let mut result = Vec::new();
        let mut stack = vec![team_id.to_string()];
        while let Some(tid) = stack.pop() {
            for child in self.children_of(&tid) {
                result.push(child);
                stack.push(child.id.clone());
            }
        }
        result
    }

    /// Ancestor chain for a team (from immediate parent up to root).
    pub fn ancestors_of(&self, team_id: &str) -> Vec<&Team> {
        let mut result = Vec::new();
        let mut current = self.find_team(team_id).and_then(|t| t.parent_id.clone());
        while let Some(pid) = current {
            if let Some(parent) = self.find_team(&pid) {
                result.push(parent);
                current = parent.parent_id.clone();
            } else {
                break;
            }
        }
        result
    }

    /// Find the effective role of a user in a team, inheriting from ancestors.
    /// If the user has a role in a parent team but not the target team,
    /// the parent role is inherited. The highest-privilege role wins.
    pub fn effective_role(&self, team_id: &str, user_id: &str) -> Option<Role> {
        let mut best: Option<Role> = None;

        // Check the team itself.
        if let Some(team) = self.find_team(team_id) {
            if let Some(member) = team.find_member(user_id) {
                best = Some(member.role);
            }
        }

        // Walk ancestors.
        for ancestor in self.ancestors_of(team_id) {
            if let Some(member) = ancestor.find_member(user_id) {
                let candidate = member.role;
                match &best {
                    None => best = Some(candidate),
                    Some(current) => {
                        if candidate.level() > current.level() {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }

        best
    }

    /// All unique user IDs across all teams.
    pub fn all_member_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .teams
            .iter()
            .flat_map(|t| t.members.iter().map(|m| m.user_id.clone()))
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Total member count (may include duplicates across teams).
    pub fn total_memberships(&self) -> usize {
        self.teams.iter().map(|t| t.member_count()).sum()
    }

    /// Depth of the team hierarchy (0 = flat).
    pub fn max_depth(&self) -> usize {
        let mut max = 0;
        for team in &self.teams {
            let depth = self.ancestors_of(&team.id).len();
            if depth > max {
                max = depth;
            }
        }
        max
    }
}

// ---------------------------------------------------------------------------
// Permission check helpers
// ---------------------------------------------------------------------------

pub fn can_user_edit(org: &Org, team_id: &str, user_id: &str) -> bool {
    org.effective_role(team_id, user_id)
        .map(|r| r.can_edit())
        .unwrap_or(false)
}

pub fn can_user_manage(org: &Org, team_id: &str, user_id: &str) -> bool {
    org.effective_role(team_id, user_id)
        .map(|r| r.can_manage_members())
        .unwrap_or(false)
}

pub fn can_user_view(org: &Org, team_id: &str, user_id: &str) -> bool {
    org.effective_role(team_id, user_id)
        .map(|r| r.can_view())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_org() -> Org {
        let mut org = Org::new("org-1", "Acme Corp");
        org.set_default("timezone", "UTC");
        org.set_default("language", "en");

        let mut eng = Team::new("eng", "Engineering");
        eng.add_member("alice", "Alice", Role::Owner);
        eng.add_member("bob", "Bob", Role::Admin);
        eng.set_setting("language", "rust");
        org.add_team(eng);

        let mut frontend = Team::new("fe", "Frontend").with_parent("eng");
        frontend.add_member("carol", "Carol", Role::Member);
        org.add_team(frontend);

        let mut backend = Team::new("be", "Backend").with_parent("eng");
        backend.add_member("dave", "Dave", Role::Member);
        backend.add_member("eve", "Eve", Role::Viewer);
        backend.set_setting("language", "go");
        org.add_team(backend);

        let mut infra = Team::new("infra", "Infra").with_parent("be");
        infra.add_member("frank", "Frank", Role::Admin);
        org.add_team(infra);

        org
    }

    #[test]
    fn test_role_levels() {
        assert!(Role::Owner.level() > Role::Admin.level());
        assert!(Role::Admin.level() > Role::Member.level());
        assert!(Role::Member.level() > Role::Viewer.level());
    }

    #[test]
    fn test_role_permissions() {
        assert!(Role::Owner.can_manage_members());
        assert!(Role::Admin.can_manage_members());
        assert!(!Role::Member.can_manage_members());
        assert!(Role::Member.can_edit());
        assert!(!Role::Viewer.can_edit());
        assert!(Role::Viewer.can_view());
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::Owner.to_string(), "owner");
        assert_eq!(Role::Admin.to_string(), "admin");
    }

    #[test]
    fn test_team_basic() {
        let org = sample_org();
        let eng = org.find_team("eng").unwrap();
        assert_eq!(eng.member_count(), 2);
        assert!(eng.is_root());
        assert!(eng.find_member("alice").is_some());
    }

    #[test]
    fn test_team_add_remove() {
        let mut team = Team::new("t", "Test");
        team.add_member("u1", "User1", Role::Member);
        assert_eq!(team.member_count(), 1);
        assert!(team.remove_member("u1"));
        assert_eq!(team.member_count(), 0);
        assert!(!team.remove_member("u1")); // already gone
    }

    #[test]
    fn test_team_members_with_role() {
        let org = sample_org();
        let be = org.find_team("be").unwrap();
        let viewers = be.members_with_role(Role::Viewer);
        assert_eq!(viewers.len(), 1);
        assert_eq!(viewers[0].user_id, "eve");
    }

    #[test]
    fn test_org_team_count() {
        let org = sample_org();
        assert_eq!(org.team_count(), 4);
    }

    #[test]
    fn test_root_teams() {
        let org = sample_org();
        let roots = org.root_teams();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "eng");
    }

    #[test]
    fn test_children_of() {
        let org = sample_org();
        let eng_children = org.children_of("eng");
        assert_eq!(eng_children.len(), 2);
        let child_ids: Vec<&str> = eng_children.iter().map(|t| t.id.as_str()).collect();
        assert!(child_ids.contains(&"fe"));
        assert!(child_ids.contains(&"be"));
    }

    #[test]
    fn test_descendants_of() {
        let org = sample_org();
        let desc = org.descendants_of("eng");
        assert_eq!(desc.len(), 3); // fe, be, infra
    }

    #[test]
    fn test_ancestors_of() {
        let org = sample_org();
        let ancestors = org.ancestors_of("infra");
        assert_eq!(ancestors.len(), 2); // be, eng
        assert_eq!(ancestors[0].id, "be");
        assert_eq!(ancestors[1].id, "eng");
    }

    #[test]
    fn test_ancestors_of_root() {
        let org = sample_org();
        let ancestors = org.ancestors_of("eng");
        assert!(ancestors.is_empty());
    }

    #[test]
    fn test_resolve_setting_team_override() {
        let org = sample_org();
        // "be" overrides "language" to "go"
        assert_eq!(org.resolve_setting("be", "language"), Some("go"));
    }

    #[test]
    fn test_resolve_setting_org_default() {
        let org = sample_org();
        // "fe" has no "timezone" override -> org default "UTC"
        assert_eq!(org.resolve_setting("fe", "timezone"), Some("UTC"));
    }

    #[test]
    fn test_resolve_setting_inherited() {
        let org = sample_org();
        // "infra" has no "language" -> parent "be" has "go"
        assert_eq!(
            org.resolve_setting_inherited("infra", "language"),
            Some("go".to_string())
        );
        // "fe" has no "language" -> parent "eng" has "rust"
        assert_eq!(
            org.resolve_setting_inherited("fe", "language"),
            Some("rust".to_string())
        );
    }

    #[test]
    fn test_resolve_setting_inherited_falls_to_org() {
        let org = sample_org();
        // "timezone" not overridden by any team -> org default "UTC"
        assert_eq!(
            org.resolve_setting_inherited("infra", "timezone"),
            Some("UTC".to_string())
        );
    }

    #[test]
    fn test_resolve_setting_missing() {
        let org = sample_org();
        assert!(org.resolve_setting("fe", "nonexistent").is_none());
    }

    #[test]
    fn test_effective_role_direct() {
        let org = sample_org();
        let role = org.effective_role("be", "dave").unwrap();
        assert_eq!(role, Role::Member);
    }

    #[test]
    fn test_effective_role_inherited() {
        let org = sample_org();
        // alice is Owner in "eng", which is ancestor of "infra"
        let role = org.effective_role("infra", "alice").unwrap();
        assert_eq!(role, Role::Owner);
    }

    #[test]
    fn test_effective_role_none() {
        let org = sample_org();
        assert!(org.effective_role("fe", "unknown-user").is_none());
    }

    #[test]
    fn test_effective_role_highest_wins() {
        let org = sample_org();
        // bob is Admin in "eng". If bob were also a Viewer in "be",
        // the Admin from "eng" should win.
        // (In our sample, bob is only in "eng" as Admin.)
        let role = org.effective_role("be", "bob").unwrap();
        assert_eq!(role, Role::Admin);
    }

    #[test]
    fn test_can_user_helpers() {
        let org = sample_org();
        assert!(can_user_edit(&org, "eng", "alice"));
        assert!(can_user_manage(&org, "eng", "bob"));
        assert!(!can_user_manage(&org, "be", "dave")); // Member can't manage
        assert!(can_user_view(&org, "be", "eve"));
        assert!(!can_user_view(&org, "fe", "nobody"));
    }

    #[test]
    fn test_all_member_ids() {
        let org = sample_org();
        let ids = org.all_member_ids();
        assert!(ids.contains(&"alice".to_string()));
        assert!(ids.contains(&"frank".to_string()));
        // No duplicates.
        let len = ids.len();
        let mut deduped = ids.clone();
        deduped.dedup();
        assert_eq!(len, deduped.len());
    }

    #[test]
    fn test_total_memberships() {
        let org = sample_org();
        // alice, bob in eng; carol in fe; dave, eve in be; frank in infra = 6
        assert_eq!(org.total_memberships(), 6);
    }

    #[test]
    fn test_max_depth() {
        let org = sample_org();
        // eng(0) -> be(1) -> infra(2)
        assert_eq!(org.max_depth(), 2);
    }

    #[test]
    fn test_empty_org() {
        let org = Org::new("empty", "Empty Org");
        assert_eq!(org.team_count(), 0);
        assert!(org.root_teams().is_empty());
        assert_eq!(org.max_depth(), 0);
        assert!(org.all_member_ids().is_empty());
    }
}
