//! Simple type checker — type environment with bindings, primitive types
//! (int/float/bool/string/void), function types, type inference
//! (Hindley-Milner lite), unification, occurs check, type error reporting.

use std::collections::HashMap;
use std::fmt;

// ── Type representation ─────────────────────────────────────────────────────

/// A type in the type system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Integer type.
    Int,
    /// Floating-point type.
    Float,
    /// Boolean type.
    Bool,
    /// String type.
    Str,
    /// Void / unit type.
    Void,
    /// Function type: param types -> return type.
    Func(Vec<Type>, Box<Type>),
    /// Type variable (for inference), identified by an integer.
    Var(u32),
    /// Named / user-defined type.
    Named(String),
    /// Array / list of element type.
    Array(Box<Type>),
    /// Tuple of types.
    Tuple(Vec<Type>),
    /// Optional type.
    Option(Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Float => write!(f, "float"),
            Self::Bool => write!(f, "bool"),
            Self::Str => write!(f, "string"),
            Self::Void => write!(f, "void"),
            Self::Func(params, ret) => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {ret}")
            }
            Self::Var(id) => write!(f, "?{id}"),
            Self::Named(n) => write!(f, "{n}"),
            Self::Array(elem) => write!(f, "[{elem}]"),
            Self::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            Self::Option(inner) => write!(f, "{inner}?"),
        }
    }
}

// ── Type errors ─────────────────────────────────────────────────────────────

/// Errors from type checking / inference.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeError {
    /// Mismatch: expected vs actual.
    Mismatch(Type, Type),
    /// Undefined variable.
    Undefined(String),
    /// Redefinition of a binding.
    Redefined(String),
    /// Wrong number of arguments.
    ArityMismatch { expected: usize, actual: usize },
    /// Tried to call a non-function.
    NotCallable(Type),
    /// Occurs check failed (infinite type).
    OccursCheck(u32, Type),
    /// Generic message.
    Other(String),
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch(exp, act) => write!(f, "type mismatch: expected {exp}, got {act}"),
            Self::Undefined(name) => write!(f, "undefined variable: {name}"),
            Self::Redefined(name) => write!(f, "redefinition of: {name}"),
            Self::ArityMismatch { expected, actual } => {
                write!(f, "arity mismatch: expected {expected} args, got {actual}")
            }
            Self::NotCallable(ty) => write!(f, "type {ty} is not callable"),
            Self::OccursCheck(var, ty) => {
                write!(f, "infinite type: ?{var} occurs in {ty}")
            }
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

// ── Substitution (union-find style) ─────────────────────────────────────────

/// A substitution mapping type variables to their resolved types.
#[derive(Debug, Clone)]
pub struct Substitution {
    bindings: HashMap<u32, Type>,
}

impl Substitution {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Bind a type variable to a type.
    pub fn bind(&mut self, var: u32, ty: Type) {
        self.bindings.insert(var, ty);
    }

    /// Look up the type bound to a variable, chasing redirects.
    pub fn resolve(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(bound) = self.bindings.get(id) {
                    self.resolve(bound)
                } else {
                    ty.clone()
                }
            }
            Type::Func(params, ret) => {
                let params: Vec<Type> = params.iter().map(|p| self.resolve(p)).collect();
                let ret = self.resolve(ret);
                Type::Func(params, Box::new(ret))
            }
            Type::Array(elem) => Type::Array(Box::new(self.resolve(elem))),
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.resolve(e)).collect())
            }
            Type::Option(inner) => Type::Option(Box::new(self.resolve(inner))),
            _ => ty.clone(),
        }
    }

    /// Check if `var` occurs anywhere in `ty` (occurs check).
    fn occurs_in(&self, var: u32, ty: &Type) -> bool {
        let resolved = self.resolve(ty);
        match resolved {
            Type::Var(id) => id == var,
            Type::Func(params, ret) => {
                params.iter().any(|p| self.occurs_in(var, p))
                    || self.occurs_in(var, &ret)
            }
            Type::Array(elem) => self.occurs_in(var, &elem),
            Type::Tuple(elems) => elems.iter().any(|e| self.occurs_in(var, e)),
            Type::Option(inner) => self.occurs_in(var, &inner),
            _ => false,
        }
    }

    /// Unify two types, extending the substitution as needed.
    pub fn unify(&mut self, a: &Type, b: &Type) -> Result<(), TypeError> {
        let a = self.resolve(a);
        let b = self.resolve(b);

        if a == b {
            return Ok(());
        }

        match (&a, &b) {
            (Type::Var(id), _) => {
                let id_val = *id;
                if self.occurs_in(id_val, &b) {
                    return Err(TypeError::OccursCheck(id_val, b));
                }
                self.bind(id_val, b);
                Ok(())
            }
            (_, Type::Var(id)) => {
                let id_val = *id;
                if self.occurs_in(id_val, &a) {
                    return Err(TypeError::OccursCheck(id_val, a));
                }
                self.bind(id_val, a);
                Ok(())
            }
            (Type::Func(p1, r1), Type::Func(p2, r2)) => {
                if p1.len() != p2.len() {
                    return Err(TypeError::ArityMismatch {
                        expected: p1.len(),
                        actual: p2.len(),
                    });
                }
                for (pa, pb) in p1.iter().zip(p2.iter()) {
                    self.unify(pa, pb)?;
                }
                self.unify(r1, r2)
            }
            (Type::Array(e1), Type::Array(e2)) => self.unify(e1, e2),
            (Type::Tuple(es1), Type::Tuple(es2)) => {
                if es1.len() != es2.len() {
                    return Err(TypeError::Mismatch(a, b));
                }
                for (ea, eb) in es1.iter().zip(es2.iter()) {
                    self.unify(ea, eb)?;
                }
                Ok(())
            }
            (Type::Option(i1), Type::Option(i2)) => self.unify(i1, i2),
            // Int can promote to Float
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => Ok(()),
            _ => Err(TypeError::Mismatch(a, b)),
        }
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

// ── Type environment ────────────────────────────────────────────────────────

/// A type environment (context) that maps names to types.
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Stack of scopes, innermost last.
    scopes: Vec<HashMap<String, Type>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    /// Push a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the innermost scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Bind a name in the current scope.
    pub fn bind(&mut self, name: &str, ty: Type) -> Result<(), TypeError> {
        let scope = self.scopes.last_mut().unwrap();
        if scope.contains_key(name) {
            return Err(TypeError::Redefined(name.into()));
        }
        scope.insert(name.into(), ty);
        Ok(())
    }

    /// Overwrite a binding in the current scope (no redefinition error).
    pub fn rebind(&mut self, name: &str, ty: Type) {
        let scope = self.scopes.last_mut().unwrap();
        scope.insert(name.into(), ty);
    }

    /// Look up a name, searching from innermost to outermost scope.
    pub fn lookup(&self, name: &str) -> Result<Type, TypeError> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Ok(ty.clone());
            }
        }
        Err(TypeError::Undefined(name.into()))
    }

    /// Return all bindings in all scopes (for diagnostics).
    pub fn all_bindings(&self) -> Vec<(String, Type)> {
        let mut out = Vec::new();
        for scope in &self.scopes {
            for (k, v) in scope {
                out.push((k.clone(), v.clone()));
            }
        }
        out
    }

    /// Number of active scopes.
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ── Type checker ────────────────────────────────────────────────────────────

/// Expressions the type checker can check.
#[derive(Debug, Clone)]
pub enum TExpr {
    /// Integer literal.
    IntLit(i64),
    /// Float literal.
    FloatLit(f64),
    /// Bool literal.
    BoolLit(bool),
    /// String literal.
    StrLit(String),
    /// Variable reference.
    Var(String),
    /// Binary operation.
    BinOp(Box<TExpr>, BinOp, Box<TExpr>),
    /// Unary operation.
    UnaryOp(UnaryOp, Box<TExpr>),
    /// Let binding: `let name = expr`.
    Let(String, Box<TExpr>),
    /// Function call.
    Call(String, Vec<TExpr>),
    /// If expression.
    If(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    /// Lambda: params and body.
    Lambda(Vec<String>, Box<TExpr>),
    /// Block of expressions, value is last.
    Block(Vec<TExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// The type checker.
pub struct TypeChecker {
    env: TypeEnv,
    subst: Substitution,
    next_var: u32,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            subst: Substitution::new(),
            next_var: 0,
        }
    }

    /// Create a type checker with some pre-defined bindings.
    pub fn with_env(env: TypeEnv) -> Self {
        Self {
            env,
            subst: Substitution::new(),
            next_var: 0,
        }
    }

    /// Generate a fresh type variable.
    pub fn fresh_var(&mut self) -> Type {
        let v = self.next_var;
        self.next_var += 1;
        Type::Var(v)
    }

    /// Access the underlying substitution.
    pub fn substitution(&self) -> &Substitution {
        &self.subst
    }

    /// Access the type environment.
    pub fn env(&self) -> &TypeEnv {
        &self.env
    }

    /// Resolve a type through the current substitution.
    pub fn resolve(&self, ty: &Type) -> Type {
        self.subst.resolve(ty)
    }

    /// Infer the type of an expression.
    pub fn infer(&mut self, expr: &TExpr) -> Result<Type, TypeError> {
        match expr {
            TExpr::IntLit(_) => Ok(Type::Int),
            TExpr::FloatLit(_) => Ok(Type::Float),
            TExpr::BoolLit(_) => Ok(Type::Bool),
            TExpr::StrLit(_) => Ok(Type::Str),
            TExpr::Var(name) => self.env.lookup(name),
            TExpr::BinOp(left, op, right) => {
                let lt = self.infer(left)?;
                let rt = self.infer(right)?;
                self.check_binop(&lt, *op, &rt)
            }
            TExpr::UnaryOp(op, operand) => {
                let ot = self.infer(operand)?;
                self.check_unaryop(*op, &ot)
            }
            TExpr::Let(name, expr) => {
                let ty = self.infer(expr)?;
                let resolved = self.resolve(&ty);
                self.env.bind(name, resolved)?;
                Ok(Type::Void)
            }
            TExpr::Call(name, args) => {
                let ft = self.env.lookup(name)?;
                let ft_resolved = self.resolve(&ft);
                match ft_resolved {
                    Type::Func(params, ret) => {
                        if params.len() != args.len() {
                            return Err(TypeError::ArityMismatch {
                                expected: params.len(),
                                actual: args.len(),
                            });
                        }
                        for (param_ty, arg_expr) in params.iter().zip(args.iter()) {
                            let at = self.infer(arg_expr)?;
                            self.subst.unify(param_ty, &at)?;
                        }
                        Ok(*ret)
                    }
                    other => Err(TypeError::NotCallable(other)),
                }
            }
            TExpr::If(cond, then_br, else_br) => {
                let ct = self.infer(cond)?;
                self.subst.unify(&ct, &Type::Bool)?;
                let tt = self.infer(then_br)?;
                let et = self.infer(else_br)?;
                self.subst.unify(&tt, &et)?;
                Ok(self.resolve(&tt))
            }
            TExpr::Lambda(params, body) => {
                self.env.push_scope();
                let mut param_types = Vec::new();
                for p in params {
                    let tv = self.fresh_var();
                    self.env.bind(p, tv.clone())?;
                    param_types.push(tv);
                }
                let ret = self.infer(body)?;
                self.env.pop_scope();
                Ok(Type::Func(param_types, Box::new(ret)))
            }
            TExpr::Block(exprs) => {
                let mut ty = Type::Void;
                for e in exprs {
                    ty = self.infer(e)?;
                }
                Ok(ty)
            }
        }
    }

    fn check_binop(&mut self, lt: &Type, op: BinOp, rt: &Type) -> Result<Type, TypeError> {
        let lt = self.resolve(lt);
        let rt = self.resolve(rt);
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                self.subst.unify(&lt, &rt)?;
                let result = self.resolve(&lt);
                match result {
                    Type::Int | Type::Float => Ok(result),
                    Type::Str if matches!(op, BinOp::Add) => Ok(Type::Str),
                    Type::Var(_) => Ok(result),
                    _ => Err(TypeError::Mismatch(Type::Int, result)),
                }
            }
            BinOp::Eq | BinOp::Ne => {
                self.subst.unify(&lt, &rt)?;
                Ok(Type::Bool)
            }
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                self.subst.unify(&lt, &rt)?;
                Ok(Type::Bool)
            }
            BinOp::And | BinOp::Or => {
                self.subst.unify(&lt, &Type::Bool)?;
                self.subst.unify(&rt, &Type::Bool)?;
                Ok(Type::Bool)
            }
        }
    }

    fn check_unaryop(&mut self, op: UnaryOp, ty: &Type) -> Result<Type, TypeError> {
        let ty = self.resolve(ty);
        match op {
            UnaryOp::Neg => match ty {
                Type::Int | Type::Float | Type::Var(_) => Ok(ty),
                _ => Err(TypeError::Mismatch(Type::Int, ty)),
            },
            UnaryOp::Not => {
                self.subst.unify(&ty, &Type::Bool)?;
                Ok(Type::Bool)
            }
        }
    }

    /// Define a function in the environment.
    pub fn define_function(
        &mut self,
        name: &str,
        params: Vec<Type>,
        ret: Type,
    ) -> Result<(), TypeError> {
        let ft = Type::Func(params, Box::new(ret));
        self.env.bind(name, ft)
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_literal() {
        let mut tc = TypeChecker::new();
        let ty = tc.infer(&TExpr::IntLit(42)).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_float_literal() {
        let mut tc = TypeChecker::new();
        let ty = tc.infer(&TExpr::FloatLit(3.14)).unwrap();
        assert_eq!(ty, Type::Float);
    }

    #[test]
    fn test_bool_literal() {
        let mut tc = TypeChecker::new();
        let ty = tc.infer(&TExpr::BoolLit(true)).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_string_literal() {
        let mut tc = TypeChecker::new();
        let ty = tc.infer(&TExpr::StrLit("hi".into())).unwrap();
        assert_eq!(ty, Type::Str);
    }

    #[test]
    fn test_variable_lookup() {
        let mut tc = TypeChecker::new();
        tc.env.bind("x", Type::Int).unwrap();
        let ty = tc.infer(&TExpr::Var("x".into())).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_undefined_variable() {
        let mut tc = TypeChecker::new();
        let err = tc.infer(&TExpr::Var("y".into())).unwrap_err();
        assert!(matches!(err, TypeError::Undefined(ref name) if name == "y"));
    }

    #[test]
    fn test_int_addition() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::BinOp(
            Box::new(TExpr::IntLit(1)),
            BinOp::Add,
            Box::new(TExpr::IntLit(2)),
        );
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_comparison_returns_bool() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::BinOp(
            Box::new(TExpr::IntLit(1)),
            BinOp::Lt,
            Box::new(TExpr::IntLit(2)),
        );
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_logical_and() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::BinOp(
            Box::new(TExpr::BoolLit(true)),
            BinOp::And,
            Box::new(TExpr::BoolLit(false)),
        );
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_negation() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::UnaryOp(UnaryOp::Neg, Box::new(TExpr::IntLit(5)));
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_not() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::UnaryOp(UnaryOp::Not, Box::new(TExpr::BoolLit(true)));
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_let_binding() {
        let mut tc = TypeChecker::new();
        tc.infer(&TExpr::Let("x".into(), Box::new(TExpr::IntLit(42)))).unwrap();
        let ty = tc.infer(&TExpr::Var("x".into())).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_function_call() {
        let mut tc = TypeChecker::new();
        tc.define_function("add", vec![Type::Int, Type::Int], Type::Int).unwrap();
        let expr = TExpr::Call("add".into(), vec![TExpr::IntLit(1), TExpr::IntLit(2)]);
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_arity_mismatch() {
        let mut tc = TypeChecker::new();
        tc.define_function("f", vec![Type::Int], Type::Int).unwrap();
        let expr = TExpr::Call("f".into(), vec![TExpr::IntLit(1), TExpr::IntLit(2)]);
        let err = tc.infer(&expr).unwrap_err();
        assert!(matches!(err, TypeError::ArityMismatch { expected: 1, actual: 2 }));
    }

    #[test]
    fn test_not_callable() {
        let mut tc = TypeChecker::new();
        tc.env.bind("x", Type::Int).unwrap();
        let expr = TExpr::Call("x".into(), vec![]);
        let err = tc.infer(&expr).unwrap_err();
        assert!(matches!(err, TypeError::NotCallable(Type::Int)));
    }

    #[test]
    fn test_if_expression() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::If(
            Box::new(TExpr::BoolLit(true)),
            Box::new(TExpr::IntLit(1)),
            Box::new(TExpr::IntLit(2)),
        );
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_if_non_bool_condition() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::If(
            Box::new(TExpr::IntLit(1)),
            Box::new(TExpr::IntLit(2)),
            Box::new(TExpr::IntLit(3)),
        );
        let err = tc.infer(&expr).unwrap_err();
        assert!(matches!(err, TypeError::Mismatch(..)));
    }

    #[test]
    fn test_unification_type_var() {
        let mut subst = Substitution::new();
        subst.unify(&Type::Var(0), &Type::Int).unwrap();
        assert_eq!(subst.resolve(&Type::Var(0)), Type::Int);
    }

    #[test]
    fn test_occurs_check() {
        let mut subst = Substitution::new();
        let err = subst
            .unify(&Type::Var(0), &Type::Array(Box::new(Type::Var(0))))
            .unwrap_err();
        assert!(matches!(err, TypeError::OccursCheck(0, _)));
    }

    #[test]
    fn test_function_unification() {
        let mut subst = Substitution::new();
        let f1 = Type::Func(vec![Type::Var(0)], Box::new(Type::Var(1)));
        let f2 = Type::Func(vec![Type::Int], Box::new(Type::Bool));
        subst.unify(&f1, &f2).unwrap();
        assert_eq!(subst.resolve(&Type::Var(0)), Type::Int);
        assert_eq!(subst.resolve(&Type::Var(1)), Type::Bool);
    }

    #[test]
    fn test_lambda_inference() {
        let mut tc = TypeChecker::new();
        // Lambda: |x| x
        let expr = TExpr::Lambda(vec!["x".into()], Box::new(TExpr::Var("x".into())));
        let ty = tc.infer(&expr).unwrap();
        // Should produce a function type with a type variable
        assert!(matches!(ty, Type::Func(params, _) if params.len() == 1));
    }

    #[test]
    fn test_block() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::Block(vec![
            TExpr::Let("x".into(), Box::new(TExpr::IntLit(1))),
            TExpr::Var("x".into()),
        ]);
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_scope_push_pop() {
        let mut env = TypeEnv::new();
        env.bind("x", Type::Int).unwrap();
        env.push_scope();
        env.bind("x", Type::Bool).unwrap(); // shadows outer
        assert_eq!(env.lookup("x").unwrap(), Type::Bool);
        env.pop_scope();
        assert_eq!(env.lookup("x").unwrap(), Type::Int);
    }

    #[test]
    fn test_redefinition_error() {
        let mut env = TypeEnv::new();
        env.bind("x", Type::Int).unwrap();
        let err = env.bind("x", Type::Bool).unwrap_err();
        assert!(matches!(err, TypeError::Redefined(ref n) if n == "x"));
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", Type::Int), "int");
        assert_eq!(format!("{}", Type::Func(vec![Type::Int], Box::new(Type::Bool))), "(int) -> bool");
        assert_eq!(format!("{}", Type::Array(Box::new(Type::Str))), "[string]");
    }

    #[test]
    fn test_int_float_promotion() {
        let mut subst = Substitution::new();
        // Int and Float should unify (promotion)
        subst.unify(&Type::Int, &Type::Float).unwrap();
    }

    #[test]
    fn test_string_concat() {
        let mut tc = TypeChecker::new();
        let expr = TExpr::BinOp(
            Box::new(TExpr::StrLit("a".into())),
            BinOp::Add,
            Box::new(TExpr::StrLit("b".into())),
        );
        let ty = tc.infer(&expr).unwrap();
        assert_eq!(ty, Type::Str);
    }

    #[test]
    fn test_fresh_var() {
        let mut tc = TypeChecker::new();
        let v1 = tc.fresh_var();
        let v2 = tc.fresh_var();
        assert_ne!(v1, v2);
        assert!(matches!(v1, Type::Var(0)));
        assert!(matches!(v2, Type::Var(1)));
    }

    #[test]
    fn test_all_bindings() {
        let mut env = TypeEnv::new();
        env.bind("a", Type::Int).unwrap();
        env.push_scope();
        env.bind("b", Type::Bool).unwrap();
        let bindings = env.all_bindings();
        assert_eq!(bindings.len(), 2);
    }
}
