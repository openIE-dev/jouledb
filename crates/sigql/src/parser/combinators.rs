//! Custom parser combinators for SigQL
//!
//! Additional combinators built on top of nom.

use nom::{
    Parser,
    character::complete::multispace0,
    error::Error,
    sequence::{delimited, preceded, terminated},
};

/// Parse with optional surrounding whitespace
pub fn ws<'a, F, O>(inner: F) -> impl Parser<&'a str, Output = O, Error = Error<&'a str>>
where
    F: Parser<&'a str, Output = O, Error = Error<&'a str>>,
{
    delimited(multispace0, inner, multispace0)
}

/// Parse with leading whitespace
pub fn ws_before<'a, F, O>(inner: F) -> impl Parser<&'a str, Output = O, Error = Error<&'a str>>
where
    F: Parser<&'a str, Output = O, Error = Error<&'a str>>,
{
    preceded(multispace0, inner)
}

/// Parse with trailing whitespace
pub fn ws_after<'a, F, O>(inner: F) -> impl Parser<&'a str, Output = O, Error = Error<&'a str>>
where
    F: Parser<&'a str, Output = O, Error = Error<&'a str>>,
{
    terminated(inner, multispace0)
}
