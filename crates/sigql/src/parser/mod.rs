//! SigQL Parser
//!
//! Parses SigQL query strings into AST using nom combinators.

pub mod combinators;
pub mod error;
pub mod lexer;

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, multispace1},
    combinator::{map, map_res, opt, recognize, value},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
};
use smol_str::SmolStr;

use crate::ast::{expr::*, query::*};
use crate::types::{FrequencyBand, Hertz, Seconds};

pub use error::ParseError;

/// Parse a complete SigQL query
pub fn parse_query(input: &str) -> Result<Query, ParseError> {
    match query(input) {
        Ok(("", query)) => Ok(query),
        Ok((remaining, _)) => Err(ParseError::IncompleteInput(remaining.to_string())),
        Err(e) => Err(ParseError::NomError(format!("{:?}", e))),
    }
}

/// Parse a signal expression (for inline use)
pub fn parse_signal_expr(input: &str) -> Result<SignalExpr, ParseError> {
    match signal_expr(input) {
        Ok(("", expr)) => Ok(expr),
        Ok((remaining, _)) => Err(ParseError::IncompleteInput(remaining.to_string())),
        Err(e) => Err(ParseError::NomError(format!("{:?}", e))),
    }
}

// ============== Core Parsers ==============

fn query(input: &str) -> IResult<&str, Query> {
    let (input, _) = multispace0(input)?;
    let (input, with) = opt(with_clause).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, from) = from_clause(input)?;
    let (input, _) = multispace0(input)?;
    let (input, let_bindings) = many0(let_binding).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, where_clause) = opt(where_clause).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, transforms) = many0(transform_clause).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, window) = opt(window_clause).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, correlate) = opt(correlate_clause).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, aggregate) = opt(aggregate_clause).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, returning) = opt(returning_clause).parse(input)?;
    let (input, _) = multispace0(input)?;

    Ok((
        input,
        Query {
            with: with.unwrap_or_default(),
            from,
            let_bindings,
            where_clause,
            transforms,
            window,
            correlate,
            aggregate,
            interpret: None,
            returning: returning.unwrap_or_default(),
        },
    ))
}

fn with_clause(input: &str) -> IResult<&str, Vec<WithClause>> {
    let (input, _) = tag_no_case("WITH").parse(input)?;
    let (input, _) = multispace1(input)?;
    separated_list1((multispace0, char(','), multispace0), with_item).parse(input)
}

fn with_item(input: &str) -> IResult<&str, WithClause> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("AS").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, expr) = signal_expr(input)?;

    Ok((
        input,
        WithClause {
            name,
            expr,
            materialized: false,
        },
    ))
}

fn from_clause(input: &str) -> IResult<&str, Vec<FromClause>> {
    let (input, _) = tag_no_case("FROM").parse(input)?;
    let (input, _) = multispace1(input)?;
    separated_list1((multispace0, char(','), multispace0), from_item).parse(input)
}

fn from_item(input: &str) -> IResult<&str, FromClause> {
    alt((media_from, map(source_ref, FromClause::Signal), session_from)).parse(input)
}

fn session_from(input: &str) -> IResult<&str, FromClause> {
    let (input, _) = tag_no_case("session").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, session_id) = string_literal(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        FromClause::Session {
            session_id,
            patient: None,
            timestamp: None,
        },
    ))
}

fn source_ref(input: &str) -> IResult<&str, SourceRef> {
    let (input, path) = dotted_identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, alias) = opt(preceded(pair(tag_no_case("AS"), multispace1), identifier)).parse(input)?;

    Ok((
        input,
        SourceRef {
            path,
            alias,
            type_hint: None,
        },
    ))
}

fn let_binding(input: &str) -> IResult<&str, LetBinding> {
    let (input, _) = tag_no_case("LET").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('=').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, expr) = signal_expr(input)?;

    Ok((input, LetBinding { name, expr }))
}

fn where_clause(input: &str) -> IResult<&str, WhereClause> {
    let (input, _) = tag_no_case("WHERE").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, conditions) = separated_list1(
        (multispace0, tag_no_case("AND"), multispace0),
        where_condition,
    ).parse(input)?;

    Ok((
        input,
        WhereClause {
            conditions,
            combinator: LogicalCombinator::And,
        },
    ))
}

fn where_condition(input: &str) -> IResult<&str, WhereCondition> {
    alt((task_phase_condition, time_range_condition, scalar_condition)).parse(input)
}

fn task_phase_condition(input: &str) -> IResult<&str, WhereCondition> {
    let (input, _) = tag_no_case("task_phase").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = alt((tag("="), tag("IN"))).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, phase) = alt((string_literal, identifier)).parse(input)?;

    Ok((
        input,
        WhereCondition::TaskPhase {
            task: None,
            phase: Some(phase),
        },
    ))
}

fn time_range_condition(input: &str) -> IResult<&str, WhereCondition> {
    let (input, _) = tag_no_case("time").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("BETWEEN").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, start) = time_spec(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("AND").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, end) = time_spec(input)?;

    Ok((
        input,
        WhereCondition::TimeRange {
            start: Some(start),
            end: Some(end),
        },
    ))
}

fn time_spec(input: &str) -> IResult<&str, TimeSpec> {
    alt((
        map(duration, TimeSpec::Relative),
        map(identifier, TimeSpec::Named),
    )).parse(input)
}

fn scalar_condition(input: &str) -> IResult<&str, WhereCondition> {
    let (input, expr) = scalar_expr(input)?;
    Ok((input, WhereCondition::Scalar(expr)))
}

fn transform_clause(input: &str) -> IResult<&str, TransformClause> {
    let (input, _) = tag_no_case("TRANSFORM").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, transforms) =
        separated_list1((multispace0, char(','), multispace0), transform_item).parse(input)?;

    Ok((input, TransformClause { transforms }))
}

fn transform_item(input: &str) -> IResult<&str, TransformItem> {
    let (input, op) = transform_op(input)?;
    Ok((input, TransformItem { op, alias: None }))
}

fn transform_op(input: &str) -> IResult<&str, TransformOp> {
    // Split into multiple alt() groups to avoid tuple size limits
    // IMPORTANT: longer keywords must come before shorter prefixes (e.g. log10 before log)
    alt((
        alt((
            bandpass_op,
            lowpass_op,
            highpass_op,
            notch_op,
            median_op,
            fft2d_op,   // fft2d before fft (longer prefix first)
            ifft2d_op,  // ifft2d before ifft
            fft_op,
            ifft_op,
            stft_op,
        )),
        alt((
            wavelet_op,
            hilbert_op,
            resample_op,
            decimate_op,
            interpolate_artifacts_op, // interpolate_artifacts before interpolate
            interpolate_op,
            zscore_op,
            detrend_op,
            baseline_correct_op,
            envelope_op,
        )),
        alt((
            instantaneous_phase_op,
            instantaneous_freq_op,
            dct2d_op,  // dct2d before any short prefix
            idct2d_op,
            mfcc_op,
            perceptual_hash_op,
            histogram_equalize_op,
            edge_detect_op,
            shot_detect_op,
            optical_flow_op,
        )),
        alt((
            reject_op, abs_op, square_op, sqrt_op, log10_op, // log10 before log
            log_op, exp_op, diff_op, cumsum_op, scale_op,
        )),
        offset_op,
    )).parse(input)
}

fn bandpass_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("bandpass").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, low) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, high) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, order) = opt(preceded((char(','), multispace0), integer)).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        TransformOp::Bandpass(FilterParams {
            cutoff_low: Some(low),
            cutoff_high: Some(high),
            order: order.unwrap_or(4) as u8,
            filter_type: FilterType::Butterworth,
        }),
    ))
}

fn lowpass_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("lowpass").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, cutoff) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        TransformOp::Lowpass(FilterParams {
            cutoff_low: None,
            cutoff_high: Some(cutoff),
            order: 4,
            filter_type: FilterType::Butterworth,
        }),
    ))
}

fn highpass_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("highpass").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, cutoff) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        TransformOp::Highpass(FilterParams {
            cutoff_low: Some(cutoff),
            cutoff_high: None,
            order: 4,
            filter_type: FilterType::Butterworth,
        }),
    ))
}

fn fft_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("fft").parse(input)?;
    let (input, params) = opt(delimited(
        (multispace0, char('('), multispace0),
        fft_params,
        (multispace0, char(')')),
    )).parse(input)?;

    Ok((input, TransformOp::Fft(params.unwrap_or_default())))
}

fn fft_params(input: &str) -> IResult<&str, FftParams> {
    // Simplified: just parse window type
    let (input, window) = opt(window_function).parse(input)?;
    Ok((
        input,
        FftParams {
            window: window.unwrap_or(WindowFunction::Hann),
            ..Default::default()
        },
    ))
}

fn window_function(input: &str) -> IResult<&str, WindowFunction> {
    alt((
        value(WindowFunction::Hann, tag_no_case("hann")),
        value(WindowFunction::Hamming, tag_no_case("hamming")),
        value(WindowFunction::Blackman, tag_no_case("blackman")),
        value(WindowFunction::Rectangular, tag_no_case("rectangular")),
    )).parse(input)
}

fn hilbert_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("hilbert").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Hilbert))
}

fn envelope_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("envelope").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Envelope))
}

fn abs_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("abs").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Abs))
}

fn resample_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("resample").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, rate) = integer(input)?;
    let (input, _) = opt(tag_no_case("Hz")).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        TransformOp::Resample(ResampleParams {
            target_rate: crate::types::SampleRate::new(rate as u32),
            method: ResampleMethod::Sinc,
        }),
    ))
}

fn zscore_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("zscore").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((
        input,
        TransformOp::ZScore(ZScoreParams {
            baseline: BaselineReference::Full,
        }),
    ))
}

fn notch_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("notch").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, freq) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((
        input,
        TransformOp::Notch(NotchParams {
            frequency: freq,
            q_factor: 30.0,
        }),
    ))
}

fn median_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("median").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, size) = integer(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((
        input,
        TransformOp::Median(MedianParams {
            kernel_size: size as usize,
        }),
    ))
}

fn ifft_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("ifft").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Ifft))
}

fn stft_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("stft").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Stft(StftParams::default())))
}

fn wavelet_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("wavelet").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((
        input,
        TransformOp::Wavelet(WaveletParams {
            mother: WaveletType::Morlet,
            scales: ScaleSpec::Log {
                start: 1.0,
                end: 100.0,
                count: 32,
            },
        }),
    ))
}

fn decimate_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("decimate").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, factor) = integer(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((
        input,
        TransformOp::Decimate(DecimateParams {
            factor: factor as usize,
            antialias: true,
        }),
    ))
}

fn interpolate_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("interpolate").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, factor) = integer(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((
        input,
        TransformOp::Interpolate(InterpolateParams {
            factor: factor as usize,
            method: ResampleMethod::Sinc,
        }),
    ))
}

fn detrend_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("detrend").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Detrend(DetrendParams { order: 1 })))
}

fn baseline_correct_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("baseline_correct").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((
        input,
        TransformOp::BaselineCorrect(BaselineParams {
            reference: BaselineReference::Full,
            method: BaselineMethod::Subtract,
        }),
    ))
}

fn instantaneous_phase_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("instantaneous_phase").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::InstantaneousPhase))
}

fn instantaneous_freq_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("instantaneous_freq").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::InstantaneousFrequency))
}

fn reject_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("reject").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, threshold) = float(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((
        input,
        TransformOp::Reject(RejectParams {
            conditions: vec![RejectCondition::AmplitudeThreshold { factor: threshold }],
        }),
    ))
}

fn interpolate_artifacts_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("interpolate_artifacts").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((
        input,
        TransformOp::InterpolateArtifacts(ArtifactInterpolateParams {
            method: InterpolateMethod::Linear,
        }),
    ))
}

fn square_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("square").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Square))
}

fn sqrt_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("sqrt").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Sqrt))
}

fn log_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("log").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Log))
}

fn log10_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("log10").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Log10))
}

fn exp_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("exp").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Exp))
}

fn diff_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("diff").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Diff))
}

fn cumsum_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("cumsum").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Cumsum))
}

fn scale_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("scale").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, factor) = float(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((input, TransformOp::Scale(factor)))
}

fn offset_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("offset").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, value) = float(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    Ok((input, TransformOp::Offset(value)))
}

fn window_clause(input: &str) -> IResult<&str, WindowClause> {
    let (input, _) = tag_no_case("WINDOW").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, spec) = window_spec(input)?;

    Ok((
        input,
        WindowClause {
            spec,
            partition_by: Vec::new(),
            order_by: None,
        },
    ))
}

fn window_spec(input: &str) -> IResult<&str, WindowSpec> {
    let (input, kind) = alt((tumbling_window, sliding_window, freq_band_window)).parse(input)?;

    let (input, _) = multispace0(input)?;
    let (input, causality) = opt(causality_spec).parse(input)?;

    Ok((
        input,
        WindowSpec {
            kind,
            causality: causality.unwrap_or(Causality::Causal),
        },
    ))
}

fn tumbling_window(input: &str) -> IResult<&str, WindowKind> {
    let (input, _) = tag_no_case("tumbling").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, duration) = duration(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, WindowKind::Tumbling { duration }))
}

fn sliding_window(input: &str) -> IResult<&str, WindowKind> {
    let (input, _) = tag_no_case("sliding").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, dur) = duration(input)?;
    let (input, _) = multispace0(input)?;
    let (input, step) = opt(preceded((char(','), multispace0), duration)).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        WindowKind::Sliding {
            duration: dur,
            step: step.unwrap_or(Seconds::new(dur.0 / 2.0)),
        },
    ))
}

fn freq_band_window(input: &str) -> IResult<&str, WindowKind> {
    let (input, _) = tag_no_case("freq_band").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, low) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = alt((tag(".."), tag("-"))).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, high) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        WindowKind::FrequencyBand(FrequencyBand::new(low.0, high.0)),
    ))
}

fn causality_spec(input: &str) -> IResult<&str, Causality> {
    alt((
        value(Causality::Causal, tag_no_case("CAUSAL")),
        value(Causality::Acausal, tag_no_case("ACAUSAL")),
    )).parse(input)
}

fn correlate_clause(input: &str) -> IResult<&str, CorrelateClause> {
    let (input, _) = tag_no_case("CORRELATE").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('{').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, operations) =
        separated_list1((multispace0, char(','), multispace0), correlate_item).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('}').parse(input)?;

    Ok((
        input,
        CorrelateClause {
            pairs: Vec::new(),
            operations,
            approximation: None,
        },
    ))
}

fn correlate_item(input: &str) -> IResult<&str, CorrelateItem> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, op) = correlate_op(input)?;

    Ok((input, CorrelateItem { name, op }))
}

fn correlate_op(input: &str) -> IResult<&str, CorrelateOp> {
    alt((
        cross_correlation_op,
        coherence_op,
        granger_causality_op,
        phase_locking_value_op,
        transfer_entropy_op,
        mutual_information_op,
        pearson_op,
        spearman_op,
        custom_correlate_op,
    )).parse(input)
}

fn cross_correlation_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("cross_correlation").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    // Parse signal references
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, max_lag) = opt(preceded(
        (
            char(','),
            multispace0,
            tag_no_case("max_lag"),
            multispace0,
            char(':'),
            multispace0,
        ),
        duration,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, CorrelateOp::CrossCorrelation { max_lag }))
}

fn coherence_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("coherence").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, CorrelateOp::Coherence { band: None }))
}

fn pearson_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("pearson").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, CorrelateOp::Pearson))
}

fn spearman_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("spearman").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, CorrelateOp::Spearman))
}

fn granger_causality_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("granger_causality").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    // Optional order parameter
    let (input, order) = opt(preceded(
        (
            char(','),
            multispace0,
            tag_no_case("order"),
            multispace0,
            char(':'),
            multispace0,
        ),
        integer,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    // Optional direction parameter
    let (input, direction) = opt(preceded(
        (
            char(','),
            multispace0,
            tag_no_case("direction"),
            multispace0,
            char(':'),
            multispace0,
        ),
        causal_direction,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        CorrelateOp::GrangerCausality {
            order: order.unwrap_or(1) as u8,
            direction: direction.unwrap_or(CausalDirection::Bidirectional),
        },
    ))
}

fn causal_direction(input: &str) -> IResult<&str, CausalDirection> {
    alt((
        value(CausalDirection::AtoB, tag_no_case("a_to_b")),
        value(CausalDirection::BtoA, tag_no_case("b_to_a")),
        value(CausalDirection::Bidirectional, tag_no_case("bidirectional")),
    )).parse(input)
}

fn phase_locking_value_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("phase_locking_value").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, band) = frequency_band(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, CorrelateOp::PhaseLockingValue { band }))
}

fn transfer_entropy_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("transfer_entropy").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    // Optional history parameter
    let (input, history) = opt(preceded(
        (
            char(','),
            multispace0,
            tag_no_case("history"),
            multispace0,
            char(':'),
            multispace0,
        ),
        integer,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        CorrelateOp::TransferEntropy {
            history: history.unwrap_or(1) as usize,
        },
    ))
}

fn mutual_information_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("mutual_information").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    // Optional bins parameter
    let (input, bins) = opt(preceded(
        (
            char(','),
            multispace0,
            tag_no_case("bins"),
            multispace0,
            char(':'),
            multispace0,
        ),
        integer,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        CorrelateOp::MutualInformation {
            bins: bins.unwrap_or(10) as usize,
        },
    ))
}

fn custom_correlate_op(input: &str) -> IResult<&str, CorrelateOp> {
    let (input, _) = tag_no_case("custom").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, name) = string_literal(input)?;
    let (input, _) = multispace0(input)?;
    let (input, params) = opt(preceded((char(','), multispace0), param_list)).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        CorrelateOp::Custom {
            name,
            params: params.unwrap_or_default(),
        },
    ))
}

// ============================================================================
// MediaQL Transform Parsers
// ============================================================================

fn fft2d_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("fft2d").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Fft2d(Fft2dParams::default())))
}

fn ifft2d_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("ifft2d").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Ifft2d))
}

fn dct2d_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("dct2d").parse(input)?;
    let (input, params) = opt(delimited(
        (multispace0, char('('), multispace0),
        dct2d_params,
        (multispace0, char(')')),
    )).parse(input)?;
    Ok((input, TransformOp::Dct2d(params.unwrap_or_default())))
}

fn dct2d_params(input: &str) -> IResult<&str, Dct2dParams> {
    // Parse optional quality parameter: dct2d(quality: 85)
    let (input, _) = opt(tag_no_case("quality:")).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, quality) = opt(integer).parse(input)?;
    Ok((
        input,
        Dct2dParams {
            quality: quality.unwrap_or(85) as u8,
            ..Default::default()
        },
    ))
}

fn idct2d_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("idct2d").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::Idct2d))
}

fn mfcc_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("mfcc").parse(input)?;
    let (input, n) = opt(delimited(
        (multispace0, char('('), multispace0),
        integer,
        (multispace0, char(')')),
    )).parse(input)?;
    Ok((
        input,
        TransformOp::Mfcc(MfccParams {
            n_coefficients: n.unwrap_or(13) as usize,
            ..Default::default()
        }),
    ))
}

fn perceptual_hash_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("perceptual_hash").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::PerceptualHash(PHashParams::default())))
}

fn histogram_equalize_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("histogram_equalize").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::HistogramEqualize))
}

fn edge_detect_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("edge_detect").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((
        input,
        TransformOp::EdgeDetect(EdgeParams {
            method: EdgeMethod::Sobel,
            threshold: 0.1,
        }),
    ))
}

fn shot_detect_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("shot_detect").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((input, TransformOp::ShotDetect(ShotDetectParams::default())))
}

fn optical_flow_op(input: &str) -> IResult<&str, TransformOp> {
    let (input, _) = tag_no_case("optical_flow").parse(input)?;
    let (input, _) = opt(pair(char('('), char(')'))).parse(input)?;
    Ok((
        input,
        TransformOp::OpticalFlow(OpticalFlowParams {
            method: FlowMethod::LucasKanade,
        }),
    ))
}

// ============================================================================
// MediaQL FROM Parsers
// ============================================================================

fn media_from(input: &str) -> IResult<&str, FromClause> {
    let (input, _) = tag_no_case("media").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, path) = string_literal(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("AS").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, alias) = identifier(input)?;

    Ok((
        input,
        FromClause::Media {
            source: crate::ast::query::MediaSourceRef::Path(path),
            alias,
        },
    ))
}

// ============================================================================
// MediaQL Aggregate Parsers
// ============================================================================

fn media_aggregate_ops(input: &str) -> IResult<&str, AggregateOp> {
    alt((
        value(AggregateOp::SpatialFrequencyContent, tag_no_case("spatial_frequency_content")),
        value(AggregateOp::TextureEntropy, tag_no_case("texture_entropy")),
        value(AggregateOp::EdgeDensity, tag_no_case("edge_density")),
        value(AggregateOp::PitchContour, tag_no_case("pitch_contour")),
        value(AggregateOp::OnsetStrength, tag_no_case("onset_strength")),
        value(AggregateOp::BeatSpectrum, tag_no_case("beat_spectrum")),
        value(AggregateOp::Loudness, tag_no_case("loudness")),
        value(AggregateOp::SceneChangeCount, tag_no_case("scene_change_count")),
        value(AggregateOp::MotionMagnitude, tag_no_case("motion_magnitude")),
        value(AggregateOp::FlickerMetric, tag_no_case("flicker_metric")),
    )).parse(input)
}

fn aggregate_clause(input: &str) -> IResult<&str, AggregateClause> {
    let (input, _) = tag_no_case("AGGREGATE").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('{').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, aggregations) =
        separated_list1((multispace0, char(','), multispace0), aggregate_item).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('}').parse(input)?;

    Ok((input, AggregateClause { aggregations }))
}

fn aggregate_item(input: &str) -> IResult<&str, AggregateItem> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, op) = aggregate_op(input)?;

    Ok((
        input,
        AggregateItem {
            name,
            op,
            input: None,
        },
    ))
}

fn aggregate_op(input: &str) -> IResult<&str, AggregateOp> {
    // Split into multiple alt() groups to avoid tuple size limits
    // IMPORTANT: longer keywords must come before shorter prefixes (e.g. peak_to_peak before peak)
    alt((
        alt((
            // Time domain - simple (order matters: longer prefixes first)
            value(AggregateOp::PeakToPeak, tag_no_case("peak_to_peak")),
            value(AggregateOp::ZeroCrossings, tag_no_case("zero_crossings")),
            value(AggregateOp::Mean, tag_no_case("mean")),
            value(AggregateOp::Std, tag_no_case("std")),
            value(AggregateOp::Var, tag_no_case("var")),
            value(AggregateOp::Rms, tag_no_case("rms")),
            value(AggregateOp::Peak, tag_no_case("peak")),
            value(AggregateOp::Trough, tag_no_case("trough")),
            value(AggregateOp::Slope, tag_no_case("slope")),
        )),
        alt((
            // Frequency domain - simple
            value(
                AggregateOp::DominantFrequency,
                tag_no_case("dominant_frequency"),
            ),
            value(
                AggregateOp::SpectralEntropy,
                tag_no_case("spectral_entropy"),
            ),
            value(
                AggregateOp::SpectralCentroid,
                tag_no_case("spectral_centroid"),
            ),
            value(
                AggregateOp::SpectralFlatness,
                tag_no_case("spectral_flatness"),
            ),
        )),
        alt((
            // Statistical - simple
            value(AggregateOp::Kurtosis, tag_no_case("kurtosis")),
            value(AggregateOp::Skewness, tag_no_case("skewness")),
            value(AggregateOp::HurstExponent, tag_no_case("hurst_exponent")),
            value(
                AggregateOp::LyapunovExponent,
                tag_no_case("lyapunov_exponent"),
            ),
        )),
        alt((
            // Parameterized aggregates
            percentile_op,
            band_power_op,
            frequency_ratio_op,
            sample_entropy_op,
            tremor_severity_op,
            reaction_time_op,
            movement_smoothness_op,
            custom_aggregate_op,
        )),
        // MediaQL aggregates
        media_aggregate_ops,
    )).parse(input)
}

fn band_power_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("band_power").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, low) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = alt((tag(".."), tag("-"))).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, high) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        AggregateOp::BandPower(FrequencyBand::new(low.0, high.0)),
    ))
}

fn percentile_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("percentile").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, p) = float(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, AggregateOp::Percentile(p)))
}

fn frequency_ratio_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("frequency_ratio").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, low_band) = frequency_band(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, high_band) = frequency_band(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        AggregateOp::FrequencyRatio {
            low: low_band,
            high: high_band,
        },
    ))
}

fn frequency_band(input: &str) -> IResult<&str, FrequencyBand> {
    let (input, low) = frequency(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = alt((tag(".."), tag("-"))).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, high) = frequency(input)?;

    Ok((input, FrequencyBand::new(low.0, high.0)))
}

fn sample_entropy_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("sample_entropy").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    // m parameter (embedding dimension)
    let (input, m) = opt(preceded(
        (tag_no_case("m"), multispace0, char(':'), multispace0),
        integer,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    // r parameter (tolerance)
    let (input, r) = opt(preceded(
        (
            opt(char(',')),
            multispace0,
            tag_no_case("r"),
            multispace0,
            char(':'),
            multispace0,
        ),
        float,
    )).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        AggregateOp::SampleEntropy {
            m: m.unwrap_or(2) as usize,
            r: r.unwrap_or(0.2),
        },
    ))
}

fn tremor_severity_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("tremor_severity").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, scale) = opt(clinical_scale).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        AggregateOp::TremorSeverity {
            scale: scale.unwrap_or(ClinicalScale::Updrs),
        },
    ))
}

fn clinical_scale(input: &str) -> IResult<&str, ClinicalScale> {
    alt((
        value(ClinicalScale::Updrs, tag_no_case("updrs")),
        value(ClinicalScale::Fahn, tag_no_case("fahn")),
        value(ClinicalScale::Bain, tag_no_case("bain")),
        value(ClinicalScale::Custom, tag_no_case("custom")),
    )).parse(input)
}

fn reaction_time_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("reaction_time").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    // stimulus event
    let (input, _) = tag_no_case("stimulus").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, stimulus) = event_ref(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, _) = multispace0(input)?;
    // response event
    let (input, _) = tag_no_case("response").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, response) = event_ref(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((input, AggregateOp::ReactionTime { stimulus, response }))
}

fn event_ref(input: &str) -> IResult<&str, EventRef> {
    let (input, source) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, offset) = opt(preceded((char('+'), multispace0), duration)).parse(input)?;

    Ok((input, EventRef { source, offset }))
}

fn movement_smoothness_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("movement_smoothness").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, method) = opt(smoothness_method).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        AggregateOp::MovementSmoothness {
            method: method.unwrap_or(SmoothnessMethod::Sparc),
        },
    ))
}

fn smoothness_method(input: &str) -> IResult<&str, SmoothnessMethod> {
    alt((
        value(SmoothnessMethod::Sparc, tag_no_case("sparc")),
        value(SmoothnessMethod::Ldlj, tag_no_case("ldlj")),
        value(SmoothnessMethod::Jerk, tag_no_case("jerk")),
    )).parse(input)
}

fn custom_aggregate_op(input: &str) -> IResult<&str, AggregateOp> {
    let (input, _) = tag_no_case("custom").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, name) = string_literal(input)?;
    let (input, _) = multispace0(input)?;
    let (input, params) = opt(preceded((char(','), multispace0), param_list)).parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')').parse(input)?;

    Ok((
        input,
        AggregateOp::Custom {
            name,
            params: params.unwrap_or_default(),
        },
    ))
}

fn param_list(input: &str) -> IResult<&str, Vec<(SmolStr, Literal)>> {
    separated_list0((multispace0, char(','), multispace0), param_pair).parse(input)
}

fn param_pair(input: &str) -> IResult<&str, (SmolStr, Literal)> {
    let (input, name) = identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, val) = literal(input)?;

    Ok((input, (name, val)))
}

fn returning_clause(input: &str) -> IResult<&str, ReturningClause> {
    let (input, _) = tag_no_case("RETURNING").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, confidence) = opt(preceded(
        (
            tag_no_case("confidence"),
            multispace0,
            char('('),
            multispace0,
        ),
        terminated(float, (multispace0, char(')'))),
    )).parse(input)?;

    Ok((
        input,
        ReturningClause {
            confidence: confidence.unwrap_or(0.95),
            ..Default::default()
        },
    ))
}

// ============== Signal Expression Parser ==============

fn signal_expr(input: &str) -> IResult<&str, SignalExpr> {
    alt((pipeline_expr, transform_expr, source_expr, var_expr)).parse(input)
}

fn pipeline_expr(input: &str) -> IResult<&str, SignalExpr> {
    let (input, first) = alt((source_expr, var_expr)).parse(input)?;
    let (input, rest) = many0(preceded(
        (multispace0, tag("|>"), multispace0),
        transform_stage,
    )).parse(input)?;

    if rest.is_empty() {
        Ok((input, first))
    } else {
        // Build pipeline from transforms
        let mut current = first;
        for op in rest {
            current = SignalExpr::Transform {
                input: Box::new(current),
                op,
            };
        }
        Ok((input, current))
    }
}

fn transform_stage(input: &str) -> IResult<&str, TransformOp> {
    transform_op(input)
}

fn transform_expr(input: &str) -> IResult<&str, SignalExpr> {
    let (input, op) = transform_op(input)?;
    let (input, _) = multispace0(input)?;
    let (input, inner) = delimited(
        char('('),
        preceded(multispace0, signal_expr),
        preceded(multispace0, char(')')),
    ).parse(input)?;

    Ok((
        input,
        SignalExpr::Transform {
            input: Box::new(inner),
            op,
        },
    ))
}

fn source_expr(input: &str) -> IResult<&str, SignalExpr> {
    let (input, source) = source_ref(input)?;
    Ok((input, SignalExpr::Source(source)))
}

fn var_expr(input: &str) -> IResult<&str, SignalExpr> {
    let (input, name) = identifier(input)?;
    Ok((input, SignalExpr::Var(name)))
}

// ============== Scalar Expression Parser ==============

fn scalar_expr(input: &str) -> IResult<&str, ScalarExpr> {
    comparison_expr(input)
}

fn comparison_expr(input: &str) -> IResult<&str, ScalarExpr> {
    let (input, left) = scalar_term(input)?;
    let (input, _) = multispace0(input)?;
    let (input, rest) = opt((comparison_op, multispace0, scalar_term)).parse(input)?;

    match rest {
        Some((op, _, right)) => Ok((
            input,
            ScalarExpr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        )),
        None => Ok((input, left)),
    }
}

fn comparison_op(input: &str) -> IResult<&str, BinaryOp> {
    alt((
        value(BinaryOp::Eq, tag("=")),
        value(BinaryOp::Ne, tag("!=")),
        value(BinaryOp::Le, tag("<=")),
        value(BinaryOp::Ge, tag(">=")),
        value(BinaryOp::Lt, tag("<")),
        value(BinaryOp::Gt, tag(">")),
    )).parse(input)
}

fn scalar_term(input: &str) -> IResult<&str, ScalarExpr> {
    alt((
        map(literal, ScalarExpr::Literal),
        map(identifier, ScalarExpr::Var),
    )).parse(input)
}

// ============== Primitive Parsers ==============

fn identifier(input: &str) -> IResult<&str, SmolStr> {
    let (input, s) = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    )).parse(input)?;
    Ok((input, SmolStr::new(s)))
}

fn dotted_identifier(input: &str) -> IResult<&str, SmolStr> {
    let (input, parts) = separated_list1(char('.'), identifier).parse(input)?;
    Ok((input, SmolStr::new(parts.join("."))))
}

fn string_literal(input: &str) -> IResult<&str, SmolStr> {
    let (input, _) = char('\'').parse(input)?;
    let (input, s) = take_while1(|c| c != '\'').parse(input)?;
    let (input, _) = char('\'').parse(input)?;
    Ok((input, SmolStr::new(s)))
}

fn integer(input: &str) -> IResult<&str, i64> {
    map_res(recognize(pair(opt(char('-')), digit1)), |s: &str| {
        s.parse::<i64>()
    }).parse(input)
}

fn float(input: &str) -> IResult<&str, f64> {
    map_res(
        recognize((
            opt(char('-')),
            digit1,
            opt(pair(char('.'), digit1)),
        )),
        |s: &str| s.parse::<f64>(),
    ).parse(input)
}

fn frequency(input: &str) -> IResult<&str, Hertz> {
    let (input, value) = float(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = opt(tag_no_case("Hz")).parse(input)?;
    Ok((input, Hertz::new(value)))
}

fn duration(input: &str) -> IResult<&str, Seconds> {
    let (input, value) = float(input)?;
    let (input, _) = multispace0(input)?;
    let (input, unit) = opt(alt((tag_no_case("ms"), tag_no_case("s"), tag_no_case("m")))).parse(input)?;

    let seconds = match unit {
        Some("ms") | Some("MS") => value / 1000.0,
        Some("m") | Some("M") => value * 60.0,
        _ => value,
    };

    Ok((input, Seconds::new(seconds)))
}

fn literal(input: &str) -> IResult<&str, Literal> {
    alt((
        map(float, Literal::Float),
        map(integer, Literal::Int),
        map(string_literal, Literal::String),
        value(Literal::Bool(true), tag_no_case("true")),
        value(Literal::Bool(false), tag_no_case("false")),
    )).parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_query() {
        let query_str = "FROM controller.imu.accel";
        let result = parse_query(query_str);
        assert!(result.is_ok());
        let query = result.unwrap();
        assert_eq!(query.from.len(), 1);
    }

    #[test]
    fn test_parse_transform() {
        let query_str = "FROM sensor.data TRANSFORM bandpass(4Hz, 12Hz)";
        let result = parse_query(query_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_pipeline() {
        let expr_str = "sensor.data |> bandpass(4Hz, 12Hz) |> hilbert |> envelope";
        let result = parse_signal_expr(expr_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_window() {
        let query_str = "FROM sensor.data WINDOW sliding(2s, 500ms)";
        let result = parse_query(query_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_aggregate() {
        let query_str =
            "FROM sensor.data AGGREGATE { power: band_power(4Hz..12Hz), freq: dominant_frequency }";
        let result = parse_query(query_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_all_simple_aggregates() {
        // Test all simple aggregate ops
        let ops = vec![
            "mean",
            "std",
            "var",
            "rms",
            "peak",
            "trough",
            "peak_to_peak",
            "zero_crossings",
            "slope",
            "dominant_frequency",
            "spectral_entropy",
            "spectral_centroid",
            "spectral_flatness",
            "kurtosis",
            "skewness",
            "hurst_exponent",
            "lyapunov_exponent",
        ];
        for op in ops {
            let query_str = format!("FROM sensor.data AGGREGATE {{ result: {} }}", op);
            let result = parse_query(&query_str);
            assert!(result.is_ok(), "Failed to parse aggregate op: {}", op);
        }
    }

    #[test]
    fn test_parse_parameterized_aggregates() {
        // Test percentile
        let result = parse_query("FROM sensor.data AGGREGATE { p50: percentile(50.0) }");
        assert!(result.is_ok());

        // Test frequency_ratio
        let result = parse_query(
            "FROM sensor.data AGGREGATE { ratio: frequency_ratio(4Hz..8Hz, 8Hz..13Hz) }",
        );
        assert!(result.is_ok());

        // Test sample_entropy
        let result =
            parse_query("FROM sensor.data AGGREGATE { entropy: sample_entropy(m: 2, r: 0.2) }");
        assert!(result.is_ok());

        // Test tremor_severity with scale
        let result = parse_query("FROM sensor.data AGGREGATE { severity: tremor_severity(updrs) }");
        assert!(result.is_ok());

        // Test movement_smoothness with method
        let result =
            parse_query("FROM sensor.data AGGREGATE { smooth: movement_smoothness(sparc) }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_all_correlate_ops() {
        // Test simple correlate ops
        let result = parse_query("FROM sensor.data CORRELATE { r: pearson(a, b) }");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data CORRELATE { r: spearman(a, b) }");
        assert!(result.is_ok());

        // Test cross_correlation with max_lag
        let result = parse_query(
            "FROM sensor.data CORRELATE { xcorr: cross_correlation(a, b, max_lag: 100ms) }",
        );
        assert!(result.is_ok());

        // Test coherence
        let result = parse_query("FROM sensor.data CORRELATE { coh: coherence(a, b) }");
        assert!(result.is_ok());

        // Test granger_causality with order
        let result =
            parse_query("FROM sensor.data CORRELATE { gc: granger_causality(a, b, order: 5) }");
        assert!(result.is_ok());

        // Test phase_locking_value with band
        let result =
            parse_query("FROM sensor.data CORRELATE { plv: phase_locking_value(a, b, 8Hz..13Hz) }");
        assert!(result.is_ok());

        // Test transfer_entropy with history
        let result =
            parse_query("FROM sensor.data CORRELATE { te: transfer_entropy(a, b, history: 3) }");
        assert!(result.is_ok());

        // Test mutual_information with bins
        let result =
            parse_query("FROM sensor.data CORRELATE { mi: mutual_information(a, b, bins: 20) }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_all_transform_ops() {
        // Test all simple transform ops
        let simple_ops = vec![
            "hilbert",
            "envelope",
            "abs",
            "square",
            "sqrt",
            "log",
            "log10",
            "exp",
            "diff",
            "cumsum",
            "ifft",
            "stft",
            "wavelet",
            "detrend",
            "baseline_correct",
            "instantaneous_phase",
            "instantaneous_freq",
            "interpolate_artifacts",
            "zscore",
            "fft",
        ];
        for op in simple_ops {
            let query_str = format!("FROM sensor.data TRANSFORM {}", op);
            let result = parse_query(&query_str);
            assert!(result.is_ok(), "Failed to parse transform op: {}", op);
        }

        // Test parameterized transform ops
        let result = parse_query("FROM sensor.data TRANSFORM notch(60Hz)");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data TRANSFORM median(5)");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data TRANSFORM decimate(4)");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data TRANSFORM interpolate(2)");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data TRANSFORM scale(2.5)");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data TRANSFORM offset(-1.0)");
        assert!(result.is_ok());

        let result = parse_query("FROM sensor.data TRANSFORM reject(3.0)");
        assert!(result.is_ok());
    }

    // ====== MediaQL Parser Tests ======

    #[test]
    fn test_parse_media_from() {
        let query_str = "FROM media('photo.jpg') AS img";
        let result = parse_query(query_str);
        assert!(result.is_ok(), "Failed to parse media FROM: {:?}", result.err());
        let query = result.unwrap();
        assert_eq!(query.from.len(), 1);
        match &query.from[0] {
            FromClause::Media { source, alias } => {
                assert_eq!(alias.as_str(), "img");
                match source {
                    crate::ast::query::MediaSourceRef::Path(p) => assert_eq!(p.as_str(), "photo.jpg"),
                    _ => panic!("Expected Path source"),
                }
            }
            _ => panic!("Expected Media FromClause"),
        }
    }

    #[test]
    fn test_parse_media_transforms() {
        // 2D DCT
        let result = parse_query("FROM media('img.png') AS img TRANSFORM dct2d");
        assert!(result.is_ok(), "dct2d: {:?}", result.err());

        // 2D DCT with quality
        let result = parse_query("FROM media('img.png') AS img TRANSFORM dct2d(quality: 85)");
        assert!(result.is_ok(), "dct2d(quality): {:?}", result.err());

        // IDCT
        let result = parse_query("FROM media('img.png') AS img TRANSFORM idct2d");
        assert!(result.is_ok(), "idct2d: {:?}", result.err());

        // FFT2D
        let result = parse_query("FROM media('img.png') AS img TRANSFORM fft2d");
        assert!(result.is_ok(), "fft2d: {:?}", result.err());

        // MFCC
        let result = parse_query("FROM media('audio.wav') AS audio TRANSFORM mfcc(13)");
        assert!(result.is_ok(), "mfcc: {:?}", result.err());

        // MFCC default
        let result = parse_query("FROM media('audio.wav') AS audio TRANSFORM mfcc");
        assert!(result.is_ok(), "mfcc default: {:?}", result.err());

        // Perceptual hash
        let result = parse_query("FROM media('img.png') AS img TRANSFORM perceptual_hash");
        assert!(result.is_ok(), "perceptual_hash: {:?}", result.err());

        // Edge detect
        let result = parse_query("FROM media('img.png') AS img TRANSFORM edge_detect");
        assert!(result.is_ok(), "edge_detect: {:?}", result.err());

        // Histogram equalize
        let result = parse_query("FROM media('img.png') AS img TRANSFORM histogram_equalize");
        assert!(result.is_ok(), "histogram_equalize: {:?}", result.err());

        // Shot detect
        let result = parse_query("FROM media('video.mp4') AS v TRANSFORM shot_detect");
        assert!(result.is_ok(), "shot_detect: {:?}", result.err());

        // Optical flow
        let result = parse_query("FROM media('video.mp4') AS v TRANSFORM optical_flow");
        assert!(result.is_ok(), "optical_flow: {:?}", result.err());
    }

    #[test]
    fn test_parse_media_aggregates() {
        let result = parse_query(
            "FROM media('img.png') AS img AGGREGATE { tex: texture_entropy, edge: edge_density }",
        );
        assert!(result.is_ok(), "media aggregates: {:?}", result.err());
    }

    #[test]
    fn test_parse_full_mediaql_query() {
        // A complete MediaQL query: ingest → transform → aggregate
        let query_str = "FROM media('photo.jpg') AS img TRANSFORM dct2d(quality: 85) AGGREGATE { tex: texture_entropy }";
        let result = parse_query(query_str);
        assert!(result.is_ok(), "Full MediaQL: {:?}", result.err());

        let query = result.unwrap();
        assert_eq!(query.from.len(), 1);
        assert_eq!(query.transforms.len(), 1);
        assert!(query.aggregate.is_some());
    }
}
