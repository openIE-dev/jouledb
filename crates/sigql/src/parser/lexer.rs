//! Lexer types for SigQL
//!
//! Token definitions for the parser.

use smol_str::SmolStr;

/// Token types for SigQL
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    From,
    Where,
    Transform,
    Window,
    Correlate,
    Aggregate,
    Returning,
    Let,
    As,
    And,
    Or,
    Not,
    In,
    Between,

    // Literals
    Integer(i64),
    Float(f64),
    String(SmolStr),
    Bool(bool),

    // Identifiers
    Ident(SmolStr),

    // Operators
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Pipe,
    PipeArrow, // |>

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    DotDot,

    // Units
    Hz,
    Ms,
    S,

    // End of input
    Eof,
}

impl Token {
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Token::From
                | Token::Where
                | Token::Transform
                | Token::Window
                | Token::Correlate
                | Token::Aggregate
                | Token::Returning
                | Token::Let
                | Token::As
                | Token::And
                | Token::Or
                | Token::Not
                | Token::In
                | Token::Between
        )
    }
}
