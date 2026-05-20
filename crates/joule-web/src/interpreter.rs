//! Interpreter pattern — expression AST with evaluation.
//!
//! Provides an expression tree for arithmetic, boolean, and user-defined
//! function evaluation. Includes a context for variable bindings, AST
//! pretty-printing, and custom function registration.

use std::collections::HashMap;
use std::fmt;

// ── Values ─────────────────────────────────────────────────────────

/// Runtime value produced by expression evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl Val {
    /// Coerce to i64 if possible.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Val::Int(v) => Some(*v),
            Val::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Coerce to f64 if possible.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Val::Float(v) => Some(*v),
            Val::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Coerce to bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Val::Bool(v) => Some(*v),
            Val::Int(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Coerce to string representation.
    pub fn to_display_string(&self) -> String {
        match self {
            Val::Int(v) => v.to_string(),
            Val::Float(v) => v.to_string(),
            Val::Bool(v) => v.to_string(),
            Val::Str(v) => v.clone(),
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

// ── Errors ─────────────────────────────────────────────────────────

/// Evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeError(String),
    DivisionByZero,
    ArityMismatch { expected: usize, got: usize },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            EvalError::UndefinedFunction(name) => write!(f, "undefined function: {name}"),
            EvalError::TypeError(msg) => write!(f, "type error: {msg}"),
            EvalError::DivisionByZero => write!(f, "division by zero"),
            EvalError::ArityMismatch { expected, got } => {
                write!(f, "arity mismatch: expected {expected}, got {got}")
            }
        }
    }
}

// ── Expression AST ─────────────────────────────────────────────────

/// An arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A boolean operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
}

/// An expression node in the AST.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Integer literal.
    IntLit(i64),
    /// Float literal.
    FloatLit(f64),
    /// Boolean literal.
    BoolLit(bool),
    /// String literal.
    StrLit(String),
    /// Variable reference.
    Var(String),
    /// Arithmetic binary expression.
    Arith {
        op: ArithOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Comparison expression.
    Cmp {
        op: CmpOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Boolean binary expression.
    BoolExpr {
        op: BoolOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Boolean negation.
    Not(Box<Expr>),
    /// Unary negation.
    Neg(Box<Expr>),
    /// If-then-else.
    IfElse {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// Function call.
    FnCall {
        name: String,
        args: Vec<Expr>,
    },
}

// ── Constructors ───────────────────────────────────────────────────

impl Expr {
    pub fn int(v: i64) -> Self {
        Expr::IntLit(v)
    }

    pub fn float(v: f64) -> Self {
        Expr::FloatLit(v)
    }

    pub fn bool_val(v: bool) -> Self {
        Expr::BoolLit(v)
    }

    pub fn string(v: impl Into<String>) -> Self {
        Expr::StrLit(v.into())
    }

    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    pub fn add(left: Expr, right: Expr) -> Self {
        Expr::Arith { op: ArithOp::Add, left: Box::new(left), right: Box::new(right) }
    }

    pub fn sub(left: Expr, right: Expr) -> Self {
        Expr::Arith { op: ArithOp::Sub, left: Box::new(left), right: Box::new(right) }
    }

    pub fn mul(left: Expr, right: Expr) -> Self {
        Expr::Arith { op: ArithOp::Mul, left: Box::new(left), right: Box::new(right) }
    }

    pub fn div(left: Expr, right: Expr) -> Self {
        Expr::Arith { op: ArithOp::Div, left: Box::new(left), right: Box::new(right) }
    }

    pub fn modulo(left: Expr, right: Expr) -> Self {
        Expr::Arith { op: ArithOp::Mod, left: Box::new(left), right: Box::new(right) }
    }

    pub fn eq(left: Expr, right: Expr) -> Self {
        Expr::Cmp { op: CmpOp::Eq, left: Box::new(left), right: Box::new(right) }
    }

    pub fn ne(left: Expr, right: Expr) -> Self {
        Expr::Cmp { op: CmpOp::Ne, left: Box::new(left), right: Box::new(right) }
    }

    pub fn lt(left: Expr, right: Expr) -> Self {
        Expr::Cmp { op: CmpOp::Lt, left: Box::new(left), right: Box::new(right) }
    }

    pub fn le(left: Expr, right: Expr) -> Self {
        Expr::Cmp { op: CmpOp::Le, left: Box::new(left), right: Box::new(right) }
    }

    pub fn gt(left: Expr, right: Expr) -> Self {
        Expr::Cmp { op: CmpOp::Gt, left: Box::new(left), right: Box::new(right) }
    }

    pub fn ge(left: Expr, right: Expr) -> Self {
        Expr::Cmp { op: CmpOp::Ge, left: Box::new(left), right: Box::new(right) }
    }

    pub fn and(left: Expr, right: Expr) -> Self {
        Expr::BoolExpr { op: BoolOp::And, left: Box::new(left), right: Box::new(right) }
    }

    pub fn or(left: Expr, right: Expr) -> Self {
        Expr::BoolExpr { op: BoolOp::Or, left: Box::new(left), right: Box::new(right) }
    }

    pub fn not(expr: Expr) -> Self {
        Expr::Not(Box::new(expr))
    }

    pub fn neg(expr: Expr) -> Self {
        Expr::Neg(Box::new(expr))
    }

    pub fn if_else(cond: Expr, then_expr: Expr, else_expr: Expr) -> Self {
        Expr::IfElse {
            cond: Box::new(cond),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::FnCall { name: name.into(), args }
    }
}

// ── Pretty print ───────────────────────────────────────────────────

impl Expr {
    /// Pretty-print the AST with indentation.
    pub fn pretty_print(&self) -> String {
        self.pp_indent(0)
    }

    fn pp_indent(&self, depth: usize) -> String {
        let pad = "  ".repeat(depth);
        match self {
            Expr::IntLit(v) => format!("{pad}{v}"),
            Expr::FloatLit(v) => format!("{pad}{v}"),
            Expr::BoolLit(v) => format!("{pad}{v}"),
            Expr::StrLit(v) => format!("{pad}\"{v}\""),
            Expr::Var(name) => format!("{pad}{name}"),
            Expr::Arith { op, left, right } => {
                let op_str = match op {
                    ArithOp::Add => "+",
                    ArithOp::Sub => "-",
                    ArithOp::Mul => "*",
                    ArithOp::Div => "/",
                    ArithOp::Mod => "%",
                };
                format!(
                    "{pad}({op_str}\n{}\n{})",
                    left.pp_indent(depth + 1),
                    right.pp_indent(depth + 1)
                )
            }
            Expr::Cmp { op, left, right } => {
                let op_str = match op {
                    CmpOp::Eq => "==",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                };
                format!(
                    "{pad}({op_str}\n{}\n{})",
                    left.pp_indent(depth + 1),
                    right.pp_indent(depth + 1)
                )
            }
            Expr::BoolExpr { op, left, right } => {
                let op_str = match op {
                    BoolOp::And => "&&",
                    BoolOp::Or => "||",
                };
                format!(
                    "{pad}({op_str}\n{}\n{})",
                    left.pp_indent(depth + 1),
                    right.pp_indent(depth + 1)
                )
            }
            Expr::Not(inner) => format!("{pad}(!\n{})", inner.pp_indent(depth + 1)),
            Expr::Neg(inner) => format!("{pad}(neg\n{})", inner.pp_indent(depth + 1)),
            Expr::IfElse { cond, then_expr, else_expr } => {
                format!(
                    "{pad}(if\n{}\n{pad}  then\n{}\n{pad}  else\n{})",
                    cond.pp_indent(depth + 1),
                    then_expr.pp_indent(depth + 2),
                    else_expr.pp_indent(depth + 2)
                )
            }
            Expr::FnCall { name, args } => {
                let arg_strs: Vec<String> =
                    args.iter().map(|a| a.pp_indent(depth + 1)).collect();
                if arg_strs.is_empty() {
                    format!("{pad}({name})")
                } else {
                    format!("{pad}({name}\n{})", arg_strs.join("\n"))
                }
            }
        }
    }
}

// ── Context ────────────────────────────────────────────────────────

/// Custom function: takes a Vec of evaluated arguments, returns a Val.
pub type CustomFn = Box<dyn Fn(Vec<Val>) -> Result<Val, EvalError>>;

/// Evaluation context holding variable bindings and custom functions.
pub struct Context {
    vars: HashMap<String, Val>,
    functions: HashMap<String, (usize, CustomFn)>,
}

impl Context {
    /// Create an empty context.
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    /// Set a variable binding.
    pub fn set_var(&mut self, name: impl Into<String>, val: Val) {
        self.vars.insert(name.into(), val);
    }

    /// Get a variable's value.
    pub fn get_var(&self, name: &str) -> Option<&Val> {
        self.vars.get(name)
    }

    /// Remove a variable. Returns the old value if present.
    pub fn remove_var(&mut self, name: &str) -> Option<Val> {
        self.vars.remove(name)
    }

    /// Number of variables.
    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    /// Register a custom function with a known arity.
    pub fn register_fn(
        &mut self,
        name: impl Into<String>,
        arity: usize,
        f: impl Fn(Vec<Val>) -> Result<Val, EvalError> + 'static,
    ) {
        self.functions.insert(name.into(), (arity, Box::new(f)));
    }

    /// Number of registered functions.
    pub fn fn_count(&self) -> usize {
        self.functions.len()
    }

    /// Evaluate an expression in this context.
    pub fn eval(&self, expr: &Expr) -> Result<Val, EvalError> {
        match expr {
            Expr::IntLit(v) => Ok(Val::Int(*v)),
            Expr::FloatLit(v) => Ok(Val::Float(*v)),
            Expr::BoolLit(v) => Ok(Val::Bool(*v)),
            Expr::StrLit(v) => Ok(Val::Str(v.clone())),

            Expr::Var(name) => self
                .vars
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UndefinedVariable(name.clone())),

            Expr::Arith { op, left, right } => {
                let lv = self.eval(left)?;
                let rv = self.eval(right)?;
                self.eval_arith(*op, &lv, &rv)
            }

            Expr::Cmp { op, left, right } => {
                let lv = self.eval(left)?;
                let rv = self.eval(right)?;
                self.eval_cmp(*op, &lv, &rv)
            }

            Expr::BoolExpr { op, left, right } => {
                let lv = self.eval(left)?;
                let lb = lv.as_bool().ok_or_else(|| {
                    EvalError::TypeError("expected bool on left of boolean op".to_string())
                })?;
                match op {
                    BoolOp::And => {
                        if !lb {
                            return Ok(Val::Bool(false));
                        }
                        let rv = self.eval(right)?;
                        rv.as_bool()
                            .map(Val::Bool)
                            .ok_or_else(|| EvalError::TypeError("expected bool".to_string()))
                    }
                    BoolOp::Or => {
                        if lb {
                            return Ok(Val::Bool(true));
                        }
                        let rv = self.eval(right)?;
                        rv.as_bool()
                            .map(Val::Bool)
                            .ok_or_else(|| EvalError::TypeError("expected bool".to_string()))
                    }
                }
            }

            Expr::Not(inner) => {
                let v = self.eval(inner)?;
                let b = v.as_bool().ok_or_else(|| {
                    EvalError::TypeError("expected bool for not".to_string())
                })?;
                Ok(Val::Bool(!b))
            }

            Expr::Neg(inner) => {
                let v = self.eval(inner)?;
                match v {
                    Val::Int(n) => Ok(Val::Int(-n)),
                    Val::Float(n) => Ok(Val::Float(-n)),
                    _ => Err(EvalError::TypeError("expected numeric for negation".to_string())),
                }
            }

            Expr::IfElse { cond, then_expr, else_expr } => {
                let cv = self.eval(cond)?;
                let b = cv.as_bool().ok_or_else(|| {
                    EvalError::TypeError("if condition must be bool".to_string())
                })?;
                if b {
                    self.eval(then_expr)
                } else {
                    self.eval(else_expr)
                }
            }

            Expr::FnCall { name, args } => {
                let (arity, func) = self
                    .functions
                    .get(name)
                    .ok_or_else(|| EvalError::UndefinedFunction(name.clone()))?;
                if args.len() != *arity {
                    return Err(EvalError::ArityMismatch {
                        expected: *arity,
                        got: args.len(),
                    });
                }
                let evaluated: Result<Vec<Val>, EvalError> =
                    args.iter().map(|a| self.eval(a)).collect();
                func(evaluated?)
            }
        }
    }

    fn eval_arith(&self, op: ArithOp, lv: &Val, rv: &Val) -> Result<Val, EvalError> {
        // Promote to float if either operand is float.
        let use_float = matches!(lv, Val::Float(_)) || matches!(rv, Val::Float(_));

        if use_float {
            let a = lv.as_float().ok_or_else(|| {
                EvalError::TypeError("expected numeric".to_string())
            })?;
            let b = rv.as_float().ok_or_else(|| {
                EvalError::TypeError("expected numeric".to_string())
            })?;
            match op {
                ArithOp::Add => Ok(Val::Float(a + b)),
                ArithOp::Sub => Ok(Val::Float(a - b)),
                ArithOp::Mul => Ok(Val::Float(a * b)),
                ArithOp::Div => {
                    if b == 0.0 {
                        Err(EvalError::DivisionByZero)
                    } else {
                        Ok(Val::Float(a / b))
                    }
                }
                ArithOp::Mod => {
                    if b == 0.0 {
                        Err(EvalError::DivisionByZero)
                    } else {
                        Ok(Val::Float(a % b))
                    }
                }
            }
        } else {
            let a = lv.as_int().ok_or_else(|| {
                EvalError::TypeError("expected numeric".to_string())
            })?;
            let b = rv.as_int().ok_or_else(|| {
                EvalError::TypeError("expected numeric".to_string())
            })?;
            match op {
                ArithOp::Add => Ok(Val::Int(a + b)),
                ArithOp::Sub => Ok(Val::Int(a - b)),
                ArithOp::Mul => Ok(Val::Int(a * b)),
                ArithOp::Div => {
                    if b == 0 {
                        Err(EvalError::DivisionByZero)
                    } else {
                        Ok(Val::Int(a / b))
                    }
                }
                ArithOp::Mod => {
                    if b == 0 {
                        Err(EvalError::DivisionByZero)
                    } else {
                        Ok(Val::Int(a % b))
                    }
                }
            }
        }
    }

    fn eval_cmp(&self, op: CmpOp, lv: &Val, rv: &Val) -> Result<Val, EvalError> {
        // Numeric comparison with float promotion.
        if let (Some(a), Some(b)) = (lv.as_float(), rv.as_float()) {
            let result = match op {
                CmpOp::Eq => (a - b).abs() < f64::EPSILON,
                CmpOp::Ne => (a - b).abs() >= f64::EPSILON,
                CmpOp::Lt => a < b,
                CmpOp::Le => a <= b,
                CmpOp::Gt => a > b,
                CmpOp::Ge => a >= b,
            };
            return Ok(Val::Bool(result));
        }

        // String comparison.
        if let (Val::Str(a), Val::Str(b)) = (lv, rv) {
            let result = match op {
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
                CmpOp::Lt => a < b,
                CmpOp::Le => a <= b,
                CmpOp::Gt => a > b,
                CmpOp::Ge => a >= b,
            };
            return Ok(Val::Bool(result));
        }

        // Bool equality.
        if let (Val::Bool(a), Val::Bool(b)) = (lv, rv) {
            let result = match op {
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
                _ => {
                    return Err(EvalError::TypeError(
                        "ordered comparison not supported for booleans".to_string(),
                    ))
                }
            };
            return Ok(Val::Bool(result));
        }

        Err(EvalError::TypeError("incompatible types for comparison".to_string()))
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_int_literal() {
        let ctx = Context::new();
        assert_eq!(ctx.eval(&Expr::int(42)).unwrap(), Val::Int(42));
    }

    #[test]
    fn eval_float_literal() {
        let ctx = Context::new();
        assert_eq!(ctx.eval(&Expr::float(3.14)).unwrap(), Val::Float(3.14));
    }

    #[test]
    fn eval_bool_literal() {
        let ctx = Context::new();
        assert_eq!(ctx.eval(&Expr::bool_val(true)).unwrap(), Val::Bool(true));
    }

    #[test]
    fn eval_string_literal() {
        let ctx = Context::new();
        assert_eq!(
            ctx.eval(&Expr::string("hi")).unwrap(),
            Val::Str("hi".to_string())
        );
    }

    #[test]
    fn eval_variable() {
        let mut ctx = Context::new();
        ctx.set_var("x", Val::Int(10));
        assert_eq!(ctx.eval(&Expr::var("x")).unwrap(), Val::Int(10));
    }

    #[test]
    fn eval_undefined_variable() {
        let ctx = Context::new();
        let err = ctx.eval(&Expr::var("missing")).unwrap_err();
        assert_eq!(err, EvalError::UndefinedVariable("missing".to_string()));
    }

    #[test]
    fn eval_addition() {
        let ctx = Context::new();
        let expr = Expr::add(Expr::int(3), Expr::int(4));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(7));
    }

    #[test]
    fn eval_subtraction() {
        let ctx = Context::new();
        let expr = Expr::sub(Expr::int(10), Expr::int(3));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(7));
    }

    #[test]
    fn eval_multiplication() {
        let ctx = Context::new();
        let expr = Expr::mul(Expr::int(5), Expr::int(6));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(30));
    }

    #[test]
    fn eval_division() {
        let ctx = Context::new();
        let expr = Expr::div(Expr::int(15), Expr::int(3));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(5));
    }

    #[test]
    fn eval_division_by_zero() {
        let ctx = Context::new();
        let expr = Expr::div(Expr::int(1), Expr::int(0));
        assert_eq!(ctx.eval(&expr).unwrap_err(), EvalError::DivisionByZero);
    }

    #[test]
    fn eval_modulo() {
        let ctx = Context::new();
        let expr = Expr::modulo(Expr::int(10), Expr::int(3));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(1));
    }

    #[test]
    fn eval_float_promotion() {
        let ctx = Context::new();
        let expr = Expr::add(Expr::int(1), Expr::float(2.5));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Float(3.5));
    }

    #[test]
    fn eval_comparison() {
        let ctx = Context::new();
        assert_eq!(
            ctx.eval(&Expr::lt(Expr::int(1), Expr::int(2))).unwrap(),
            Val::Bool(true)
        );
        assert_eq!(
            ctx.eval(&Expr::ge(Expr::int(5), Expr::int(5))).unwrap(),
            Val::Bool(true)
        );
        assert_eq!(
            ctx.eval(&Expr::ne(Expr::int(1), Expr::int(2))).unwrap(),
            Val::Bool(true)
        );
    }

    #[test]
    fn eval_string_comparison() {
        let ctx = Context::new();
        let expr = Expr::eq(Expr::string("abc"), Expr::string("abc"));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Bool(true));
    }

    #[test]
    fn eval_boolean_and() {
        let ctx = Context::new();
        let expr = Expr::and(Expr::bool_val(true), Expr::bool_val(false));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Bool(false));
    }

    #[test]
    fn eval_boolean_or() {
        let ctx = Context::new();
        let expr = Expr::or(Expr::bool_val(false), Expr::bool_val(true));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Bool(true));
    }

    #[test]
    fn eval_short_circuit_and() {
        let ctx = Context::new();
        // false && (undefined_var) should short-circuit and not error
        let expr = Expr::and(Expr::bool_val(false), Expr::var("undef"));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Bool(false));
    }

    #[test]
    fn eval_short_circuit_or() {
        let ctx = Context::new();
        let expr = Expr::or(Expr::bool_val(true), Expr::var("undef"));
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Bool(true));
    }

    #[test]
    fn eval_not() {
        let ctx = Context::new();
        assert_eq!(
            ctx.eval(&Expr::not(Expr::bool_val(true))).unwrap(),
            Val::Bool(false)
        );
    }

    #[test]
    fn eval_negation() {
        let ctx = Context::new();
        assert_eq!(
            ctx.eval(&Expr::neg(Expr::int(42))).unwrap(),
            Val::Int(-42)
        );
        assert_eq!(
            ctx.eval(&Expr::neg(Expr::float(1.5))).unwrap(),
            Val::Float(-1.5)
        );
    }

    #[test]
    fn eval_if_else() {
        let ctx = Context::new();
        let expr = Expr::if_else(
            Expr::bool_val(true),
            Expr::int(1),
            Expr::int(2),
        );
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(1));

        let expr2 = Expr::if_else(
            Expr::bool_val(false),
            Expr::int(1),
            Expr::int(2),
        );
        assert_eq!(ctx.eval(&expr2).unwrap(), Val::Int(2));
    }

    #[test]
    fn eval_custom_function() {
        let mut ctx = Context::new();
        ctx.register_fn("double", 1, |args| {
            let v = args[0].as_int().ok_or_else(|| {
                EvalError::TypeError("expected int".to_string())
            })?;
            Ok(Val::Int(v * 2))
        });
        let expr = Expr::call("double", vec![Expr::int(21)]);
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(42));
    }

    #[test]
    fn eval_function_arity_mismatch() {
        let mut ctx = Context::new();
        ctx.register_fn("noop", 0, |_| Ok(Val::Bool(true)));
        let expr = Expr::call("noop", vec![Expr::int(1)]);
        assert_eq!(
            ctx.eval(&expr).unwrap_err(),
            EvalError::ArityMismatch { expected: 0, got: 1 }
        );
    }

    #[test]
    fn eval_undefined_function() {
        let ctx = Context::new();
        let expr = Expr::call("nope", vec![]);
        assert_eq!(
            ctx.eval(&expr).unwrap_err(),
            EvalError::UndefinedFunction("nope".to_string())
        );
    }

    #[test]
    fn complex_expression() {
        let mut ctx = Context::new();
        ctx.set_var("x", Val::Int(10));
        ctx.set_var("y", Val::Int(3));
        // (x + y) * 2 - 1 = 25
        let expr = Expr::sub(
            Expr::mul(
                Expr::add(Expr::var("x"), Expr::var("y")),
                Expr::int(2),
            ),
            Expr::int(1),
        );
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(25));
    }

    #[test]
    fn pretty_print_basic() {
        let expr = Expr::add(Expr::int(1), Expr::var("x"));
        let pp = expr.pretty_print();
        assert!(pp.contains('+'));
        assert!(pp.contains('1'));
        assert!(pp.contains('x'));
    }

    #[test]
    fn pretty_print_function_call() {
        let expr = Expr::call("max", vec![Expr::int(1), Expr::int(2)]);
        let pp = expr.pretty_print();
        assert!(pp.contains("max"));
    }

    #[test]
    fn val_display() {
        assert_eq!(Val::Int(42).to_display_string(), "42");
        assert_eq!(Val::Bool(true).to_display_string(), "true");
        assert_eq!(Val::Str("hi".to_string()).to_display_string(), "hi");
    }

    #[test]
    fn context_var_operations() {
        let mut ctx = Context::new();
        ctx.set_var("a", Val::Int(1));
        assert_eq!(ctx.var_count(), 1);
        assert_eq!(ctx.remove_var("a"), Some(Val::Int(1)));
        assert_eq!(ctx.var_count(), 0);
    }

    #[test]
    fn custom_fn_with_multiple_args() {
        let mut ctx = Context::new();
        ctx.register_fn("add3", 3, |args| {
            let sum: i64 = args.iter().filter_map(|v| v.as_int()).sum();
            Ok(Val::Int(sum))
        });
        let expr = Expr::call("add3", vec![Expr::int(1), Expr::int(2), Expr::int(3)]);
        assert_eq!(ctx.eval(&expr).unwrap(), Val::Int(6));
        assert_eq!(ctx.fn_count(), 1);
    }
}
