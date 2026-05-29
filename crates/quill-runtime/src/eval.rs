use std::sync::Arc;

use quill_plan::{JitBinaryOp, JitError, JitExpr, JitResult, JitScalar, JitType};

use super::array::BatchView;
use super::value::{option_zip, type_mismatch_types, Scalar};

pub(super) fn eval_expr(expr: &JitExpr, view: &BatchView<'_>, row: usize) -> JitResult<Scalar> {
    match expr {
        JitExpr::Column { index, .. } => view.value(*index, row),
        JitExpr::Literal(value) => Ok(eval_literal(value)),
        JitExpr::Binary {
            op, left, right, ..
        } => eval_binary(
            *op,
            eval_expr(left, view, row)?,
            eval_expr(right, view, row)?,
        ),
        JitExpr::IsNull(arg) => Ok(Scalar::Bool(Some(eval_expr(arg, view, row)?.is_null()))),
    }
}

pub(super) fn ensure_supported_expr(expr: &JitExpr) -> JitResult<()> {
    match expr {
        JitExpr::Column { .. } | JitExpr::Literal(_) => Ok(()),
        JitExpr::IsNull(arg) => ensure_supported_expr(arg),
        JitExpr::Binary {
            op, left, right, ..
        } => {
            if matches!(op, JitBinaryOp::Div) {
                return Err(JitError::UnsupportedExpr(
                    "division is not yet supported by the fixed-width kernel".to_string(),
                ));
            }
            ensure_supported_expr(left)?;
            ensure_supported_expr(right)
        }
    }
}

fn eval_literal(value: &JitScalar) -> Scalar {
    match value {
        JitScalar::Null(ty) => match ty {
            JitType::Bool => Scalar::Bool(None),
            JitType::Date32 => Scalar::Date32(None),
            JitType::Int32 => Scalar::Int32(None),
            JitType::Int64 => Scalar::Int64(None),
            JitType::Float64 => Scalar::Float64(None),
            JitType::Utf8 => Scalar::Utf8(None),
            JitType::Decimal128 { precision, scale } => Scalar::Decimal128 {
                value: None,
                precision: *precision,
                scale: *scale,
            },
        },
        JitScalar::Bool(value) => Scalar::Bool(Some(*value)),
        JitScalar::Date32(value) => Scalar::Date32(Some(*value)),
        JitScalar::Int32(value) => Scalar::Int32(Some(*value)),
        JitScalar::Int64(value) => Scalar::Int64(Some(*value)),
        JitScalar::Float64(value) => Scalar::Float64(Some(*value)),
        JitScalar::Utf8(value) => Scalar::Utf8(Some(Arc::from(value.as_str()))),
        JitScalar::Decimal128 {
            value,
            precision,
            scale,
        } => Scalar::Decimal128 {
            value: Some(*value),
            precision: *precision,
            scale: *scale,
        },
    }
}

fn eval_binary(op: JitBinaryOp, lhs: Scalar, rhs: Scalar) -> JitResult<Scalar> {
    match op {
        JitBinaryOp::Add | JitBinaryOp::Sub | JitBinaryOp::Mul => eval_arithmetic(op, lhs, rhs),
        JitBinaryOp::Eq
        | JitBinaryOp::NotEq
        | JitBinaryOp::Lt
        | JitBinaryOp::LtEq
        | JitBinaryOp::Gt
        | JitBinaryOp::GtEq => eval_comparison(op, lhs, rhs),
        JitBinaryOp::And | JitBinaryOp::Or => eval_boolean(op, lhs, rhs),
        JitBinaryOp::Div => Err(JitError::UnsupportedExpr(
            "division is not yet supported by the fixed-width kernel".to_string(),
        )),
    }
}

fn eval_arithmetic(op: JitBinaryOp, lhs: Scalar, rhs: Scalar) -> JitResult<Scalar> {
    let lhs_ty = lhs.ty();
    let rhs_ty = rhs.ty();
    match (lhs, rhs) {
        (Scalar::Int32(lhs), Scalar::Int32(rhs)) => Ok(Scalar::Int32(option_zip(lhs, rhs).map(
            |(lhs, rhs)| match op {
                JitBinaryOp::Add => lhs + rhs,
                JitBinaryOp::Sub => lhs - rhs,
                JitBinaryOp::Mul => lhs * rhs,
                _ => unreachable!(),
            },
        ))),
        (Scalar::Int64(lhs), Scalar::Int64(rhs)) => Ok(Scalar::Int64(option_zip(lhs, rhs).map(
            |(lhs, rhs)| match op {
                JitBinaryOp::Add => lhs + rhs,
                JitBinaryOp::Sub => lhs - rhs,
                JitBinaryOp::Mul => lhs * rhs,
                _ => unreachable!(),
            },
        ))),
        (Scalar::Float64(lhs), Scalar::Float64(rhs)) => Ok(Scalar::Float64(
            option_zip(lhs, rhs).map(|(lhs, rhs)| match op {
                JitBinaryOp::Add => lhs + rhs,
                JitBinaryOp::Sub => lhs - rhs,
                JitBinaryOp::Mul => lhs * rhs,
                _ => unreachable!(),
            }),
        )),
        (
            Scalar::Decimal128 {
                value: lhs,
                precision: lhs_precision,
                scale: lhs_scale,
            },
            Scalar::Decimal128 {
                value: rhs,
                precision: rhs_precision,
                scale: rhs_scale,
            },
        ) => eval_decimal_arithmetic(
            op,
            lhs,
            lhs_precision,
            lhs_scale,
            rhs,
            rhs_precision,
            rhs_scale,
        ),
        _ => Err(type_mismatch_types(lhs_ty, rhs_ty)),
    }
}

fn eval_comparison(op: JitBinaryOp, lhs: Scalar, rhs: Scalar) -> JitResult<Scalar> {
    let lhs_ty = lhs.ty();
    let rhs_ty = rhs.ty();
    let value = match (lhs, rhs) {
        (Scalar::Bool(lhs), Scalar::Bool(rhs))
            if matches!(op, JitBinaryOp::Eq | JitBinaryOp::NotEq) =>
        {
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_bool(op, lhs, rhs))
        }
        (Scalar::Date32(lhs), Scalar::Date32(rhs)) => {
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_ord(op, lhs, rhs))
        }
        (Scalar::Int32(lhs), Scalar::Int32(rhs)) => {
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_ord(op, lhs, rhs))
        }
        (Scalar::Int64(lhs), Scalar::Int64(rhs)) => {
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_ord(op, lhs, rhs))
        }
        (Scalar::Float64(lhs), Scalar::Float64(rhs)) => {
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_ord(op, lhs, rhs))
        }
        (Scalar::Utf8(lhs), Scalar::Utf8(rhs)) => {
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_ord(op, lhs.as_ref(), rhs.as_ref()))
        }
        (
            Scalar::Decimal128 {
                value: lhs,
                scale: lhs_scale,
                ..
            },
            Scalar::Decimal128 {
                value: rhs,
                scale: rhs_scale,
                ..
            },
        ) => {
            if lhs_scale != rhs_scale {
                return Err(JitError::UnsupportedExpr(format!(
                    "decimal comparison requires matching scale, got {lhs_scale} and {rhs_scale}"
                )));
            }
            option_zip(lhs, rhs).map(|(lhs, rhs)| compare_ord(op, lhs, rhs))
        }
        _ => return Err(type_mismatch_types(lhs_ty, rhs_ty)),
    };
    Ok(Scalar::Bool(value))
}

fn eval_boolean(op: JitBinaryOp, lhs: Scalar, rhs: Scalar) -> JitResult<Scalar> {
    let lhs_ty = lhs.ty();
    let rhs_ty = rhs.ty();
    let (Scalar::Bool(lhs), Scalar::Bool(rhs)) = (lhs, rhs) else {
        return Err(type_mismatch_types(lhs_ty, rhs_ty));
    };
    let value = match op {
        JitBinaryOp::And => match (lhs, rhs) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        },
        JitBinaryOp::Or => match (lhs, rhs) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
        _ => unreachable!(),
    };
    Ok(Scalar::Bool(value))
}

fn compare_bool(op: JitBinaryOp, lhs: bool, rhs: bool) -> bool {
    match op {
        JitBinaryOp::Eq => lhs == rhs,
        JitBinaryOp::NotEq => lhs != rhs,
        _ => unreachable!(),
    }
}

fn eval_decimal_arithmetic(
    op: JitBinaryOp,
    lhs: Option<i128>,
    lhs_precision: u8,
    lhs_scale: i8,
    rhs: Option<i128>,
    rhs_precision: u8,
    rhs_scale: i8,
) -> JitResult<Scalar> {
    match op {
        JitBinaryOp::Add | JitBinaryOp::Sub => {
            if lhs_scale != rhs_scale {
                return Err(JitError::UnsupportedExpr(format!(
                    "decimal {} requires matching scale, got {} and {}",
                    format_decimal_op(op),
                    lhs_scale,
                    rhs_scale
                )));
            }
            let value = option_zip(lhs, rhs).map(|(lhs, rhs)| match op {
                JitBinaryOp::Add => lhs + rhs,
                JitBinaryOp::Sub => lhs - rhs,
                _ => unreachable!(),
            });
            Ok(Scalar::Decimal128 {
                value,
                precision: lhs_precision.max(rhs_precision).saturating_add(1).min(38),
                scale: lhs_scale,
            })
        }
        JitBinaryOp::Mul => Ok(Scalar::Decimal128 {
            value: option_zip(lhs, rhs).map(|(lhs, rhs)| lhs * rhs),
            precision: lhs_precision.saturating_add(rhs_precision).min(38),
            scale: lhs_scale.saturating_add(rhs_scale),
        }),
        _ => Err(JitError::UnsupportedExpr(format!(
            "decimal operator {} is not supported",
            format_decimal_op(op)
        ))),
    }
}

fn format_decimal_op(op: JitBinaryOp) -> &'static str {
    match op {
        JitBinaryOp::Add => "+",
        JitBinaryOp::Sub => "-",
        JitBinaryOp::Mul => "*",
        JitBinaryOp::Div => "/",
        JitBinaryOp::Eq => "==",
        JitBinaryOp::NotEq => "!=",
        JitBinaryOp::Lt => "<",
        JitBinaryOp::LtEq => "<=",
        JitBinaryOp::Gt => ">",
        JitBinaryOp::GtEq => ">=",
        JitBinaryOp::And => "and",
        JitBinaryOp::Or => "or",
    }
}

fn compare_ord<T: PartialOrd + PartialEq>(op: JitBinaryOp, lhs: T, rhs: T) -> bool {
    match op {
        JitBinaryOp::Eq => lhs == rhs,
        JitBinaryOp::NotEq => lhs != rhs,
        JitBinaryOp::Lt => lhs < rhs,
        JitBinaryOp::LtEq => lhs <= rhs,
        JitBinaryOp::Gt => lhs > rhs,
        JitBinaryOp::GtEq => lhs >= rhs,
        _ => unreachable!(),
    }
}
