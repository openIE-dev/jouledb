//! Component model with lifecycle hooks, context, and error boundaries.

use crate::vdom::VNode;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

// ── Component trait ─────────────────────────────────────────────

/// A renderable component.
pub trait Component {
    fn render(&self) -> VNode;
}

/// Marker trait for component props.
pub trait Props: Clone + 'static {}

// Blanket impl: anything Clone + 'static is Props.
impl<T: Clone + 'static> Props for T {}

/// A concrete component definition holding typed props, optional key, and children.
pub struct ComponentDef<P: Props> {
    pub props: P,
    pub key: Option<String>,
    pub children: Vec<VNode>,
}

impl<P: Props> ComponentDef<P> {
    pub fn new(props: P) -> Self {
        Self {
            props,
            key: None,
            children: Vec::new(),
        }
    }

    pub fn key(mut self, k: &str) -> Self {
        self.key = Some(k.to_string());
        self
    }

    pub fn child(mut self, node: VNode) -> Self {
        self.children.push(node);
        self
    }

    pub fn children(mut self, nodes: Vec<VNode>) -> Self {
        self.children = nodes;
        self
    }
}

// ── Lifecycle ───────────────────────────────────────────────────

/// Manages mount and cleanup hooks for a component.
pub struct LifecycleManager {
    mount_hooks: Vec<Box<dyn FnOnce()>>,
    cleanup_hooks: Vec<Box<dyn FnOnce()>>,
}

impl LifecycleManager {
    pub fn new() -> Self {
        Self {
            mount_hooks: Vec::new(),
            cleanup_hooks: Vec::new(),
        }
    }

    pub fn register_mount(&mut self, f: impl FnOnce() + 'static) {
        self.mount_hooks.push(Box::new(f));
    }

    pub fn register_cleanup(&mut self, f: impl FnOnce() + 'static) {
        self.cleanup_hooks.push(Box::new(f));
    }

    pub fn run_mount_hooks(&mut self) {
        for hook in self.mount_hooks.drain(..) {
            hook();
        }
    }

    pub fn run_cleanup_hooks(&mut self) {
        for hook in self.cleanup_hooks.drain(..) {
            hook();
        }
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

// Free functions that register on thread-local lifecycle manager.
thread_local! {
    static CURRENT_LIFECYCLE: RefCell<Option<LifecycleManager>> = const { RefCell::new(None) };
}

/// Register a mount hook for the currently-rendering component.
pub fn on_mount(f: impl FnOnce() + 'static) {
    CURRENT_LIFECYCLE.with(|lc| {
        if let Some(ref mut mgr) = *lc.borrow_mut() {
            mgr.register_mount(f);
        }
    });
}

/// Register a cleanup hook for the currently-rendering component.
pub fn on_cleanup(f: impl FnOnce() + 'static) {
    CURRENT_LIFECYCLE.with(|lc| {
        if let Some(ref mut mgr) = *lc.borrow_mut() {
            mgr.register_cleanup(f);
        }
    });
}

/// Render a component with lifecycle tracking. Returns `(VNode, LifecycleManager)`.
pub fn render_component(component: &dyn Component) -> (VNode, LifecycleManager) {
    let mgr = LifecycleManager::new();
    CURRENT_LIFECYCLE.with(|lc| {
        *lc.borrow_mut() = Some(mgr);
    });

    let vnode = component.render();

    let mgr = CURRENT_LIFECYCLE.with(|lc| {
        lc.borrow_mut().take().unwrap_or_else(LifecycleManager::new)
    });

    (vnode, mgr)
}

// ── Context ─────────────────────────────────────────────────────

thread_local! {
    static CONTEXT_STACK: RefCell<Vec<HashMap<TypeId, Box<dyn Any>>>> = const { RefCell::new(Vec::new()) };
}

/// Push a new context scope (called when entering a component subtree).
pub fn push_context_scope() {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().push(HashMap::new());
    });
}

/// Pop the current context scope.
pub fn pop_context_scope() {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

/// Make a value available to descendant components.
pub fn provide_context<T: Any + Clone + 'static>(value: T) {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.is_empty() {
            stack.push(HashMap::new());
        }
        let scope = stack.last_mut().expect("context stack empty");
        scope.insert(TypeId::of::<T>(), Box::new(value));
    });
}

/// Retrieve a context value from the nearest ancestor scope.
pub fn use_context<T: Any + Clone + 'static>() -> Option<T> {
    CONTEXT_STACK.with(|stack| {
        let stack = stack.borrow();
        for scope in stack.iter().rev() {
            if let Some(val) = scope.get(&TypeId::of::<T>()) {
                return val.downcast_ref::<T>().cloned();
            }
        }
        None
    })
}

// ── Error Boundary ──────────────────────────────────────────────

/// Catches panics during render and shows a fallback `VNode`.
pub struct ErrorBoundary {
    fallback: VNode,
}

impl ErrorBoundary {
    pub fn new(fallback: VNode) -> Self {
        Self { fallback }
    }

    /// Try to render `f`. If it panics, return the fallback.
    pub fn try_render(self, f: impl FnOnce() -> VNode) -> VNode {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(node) => node,
            Err(_) => self.fallback,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdom::VNode;
    use std::cell::Cell;
    use std::rc::Rc;

    fn reset_context() {
        CONTEXT_STACK.with(|stack| {
            stack.borrow_mut().clear();
        });
    }

    struct Greeting {
        name: String,
    }

    impl Component for Greeting {
        fn render(&self) -> VNode {
            VNode::element("span")
                .child(VNode::text(&format!("Hello, {}!", self.name)))
        }
    }

    #[test]
    fn component_renders_vnode() {
        let comp = Greeting { name: "World".into() };
        let node = comp.render();
        match node {
            VNode::Element { tag, children, .. } => {
                assert_eq!(tag, "span");
                assert_eq!(children.len(), 1);
            }
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn props_passed_correctly() {
        let def = ComponentDef::new("my-props".to_string())
            .key("k1");
        assert_eq!(def.props, "my-props");
        assert_eq!(def.key.as_deref(), Some("k1"));
    }

    #[test]
    fn lifecycle_hooks_fire() {
        let mount_ran = Rc::new(Cell::new(false));
        let cleanup_ran = Rc::new(Cell::new(false));
        let mr = mount_ran.clone();
        let cr = cleanup_ran.clone();

        let mut mgr = LifecycleManager::new();
        mgr.register_mount(move || mr.set(true));
        mgr.register_cleanup(move || cr.set(true));

        mgr.run_mount_hooks();
        assert!(mount_ran.get());
        assert!(!cleanup_ran.get());

        mgr.run_cleanup_hooks();
        assert!(cleanup_ran.get());
    }

    #[test]
    fn context_provide_use() {
        reset_context();
        push_context_scope();
        provide_context(42u32);
        assert_eq!(use_context::<u32>(), Some(42));
        pop_context_scope();
    }

    #[test]
    fn nested_context_override() {
        reset_context();
        push_context_scope();
        provide_context(1u32);

        push_context_scope();
        provide_context(2u32);
        assert_eq!(use_context::<u32>(), Some(2));
        pop_context_scope();

        assert_eq!(use_context::<u32>(), Some(1));
        pop_context_scope();
    }

    #[test]
    fn error_boundary_catches_panic() {
        let fallback = VNode::text("Error occurred");
        let boundary = ErrorBoundary::new(fallback);
        let result = boundary.try_render(|| {
            panic!("render failure");
        });
        match result {
            VNode::Text(s) => assert_eq!(s, "Error occurred"),
            _ => panic!("expected fallback text"),
        }
    }

    #[test]
    fn error_boundary_passes_through() {
        let fallback = VNode::text("Error");
        let boundary = ErrorBoundary::new(fallback);
        let result = boundary.try_render(|| VNode::text("OK"));
        match result {
            VNode::Text(s) => assert_eq!(s, "OK"),
            _ => panic!("expected OK text"),
        }
    }

    #[test]
    fn children_rendered() {
        let def = ComponentDef::new(())
            .child(VNode::text("a"))
            .child(VNode::text("b"));
        assert_eq!(def.children.len(), 2);
    }

    #[test]
    fn component_with_key() {
        let def = ComponentDef::new(()).key("my-key");
        assert_eq!(def.key.as_deref(), Some("my-key"));
    }

    #[test]
    fn render_component_with_lifecycle() {
        struct MountComp;
        impl Component for MountComp {
            fn render(&self) -> VNode {
                on_mount(|| { /* would register */ });
                VNode::text("mounted")
            }
        }

        let (node, mut mgr) = render_component(&MountComp);
        match node {
            VNode::Text(s) => assert_eq!(s, "mounted"),
            _ => panic!("expected text"),
        }
        mgr.run_mount_hooks();
    }
}
