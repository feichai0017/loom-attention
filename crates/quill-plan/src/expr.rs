use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitType {
    Bool,
    Date32,
    Int32,
    Int64,
    Float64,
    Utf8,
    Decimal128 { precision: u8, scale: i8 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum JitScalar {
    Null(JitType),
    Bool(bool),
    Date32(i32),
    Int32(i32),
    Int64(i64),
    Float64(f64),
    Utf8(String),
    Decimal128 {
        value: i128,
        precision: u8,
        scale: i8,
    },
}

impl JitScalar {
    pub fn ty(&self) -> JitType {
        match self {
            Self::Null(ty) => *ty,
            Self::Bool(_) => JitType::Bool,
            Self::Date32(_) => JitType::Date32,
            Self::Int32(_) => JitType::Int32,
            Self::Int64(_) => JitType::Int64,
            Self::Float64(_) => JitType::Float64,
            Self::Utf8(_) => JitType::Utf8,
            Self::Decimal128 {
                precision, scale, ..
            } => JitType::Decimal128 {
                precision: *precision,
                scale: *scale,
            },
        }
    }
}

impl Display for JitScalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null(ty) => write!(f, "null:{ty}"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Date32(value) => write!(f, "{value}:date32"),
            Self::Int32(value) => write!(f, "{value}:i32"),
            Self::Int64(value) => write!(f, "{value}:i64"),
            Self::Float64(value) => write!(f, "{value}:f64"),
            Self::Utf8(value) => write!(f, "{value:?}:utf8"),
            Self::Decimal128 {
                value,
                precision,
                scale,
            } => write!(f, "{value}:decimal128({precision},{scale})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

impl Display for JitBinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let op = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Eq => "==",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::And => "and",
            Self::Or => "or",
        };
        f.write_str(op)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum JitExpr {
    Column {
        index: usize,
        name: String,
        ty: JitType,
        nullable: bool,
    },
    Literal(JitScalar),
    Binary {
        op: JitBinaryOp,
        left: Box<JitExpr>,
        right: Box<JitExpr>,
        ty: JitType,
        nullable: bool,
    },
    IsNull(Box<JitExpr>),
}

impl JitExpr {
    pub fn ty(&self) -> JitType {
        match self {
            Self::Column { ty, .. } => *ty,
            Self::Literal(value) => value.ty(),
            Self::Binary { ty, .. } => *ty,
            Self::IsNull(_) => JitType::Bool,
        }
    }

    pub fn nullable(&self) -> bool {
        match self {
            Self::Column { nullable, .. } => *nullable,
            Self::Literal(JitScalar::Null(_)) => true,
            Self::Literal(_) => false,
            Self::Binary { nullable, .. } => *nullable,
            Self::IsNull(_) => false,
        }
    }
}

impl Display for JitExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column {
                index,
                name,
                ty,
                nullable,
            } => write!(f, "col({index}, {name}, {ty}, nullable={nullable})"),
            Self::Literal(value) => write!(f, "{value}"),
            Self::Binary {
                op, left, right, ..
            } => write!(f, "({left} {op} {right})"),
            Self::IsNull(arg) => write!(f, "is_null({arg})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JitProjection {
    pub expr: JitExpr,
    pub alias: String,
}

impl JitProjection {
    pub fn new(expr: JitExpr, alias: impl Into<String>) -> Self {
        Self {
            expr,
            alias: alias.into(),
        }
    }
}

impl Display for JitType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let ty = match self {
            Self::Bool => "i1",
            Self::Date32 => "date32",
            Self::Int32 => "i32",
            Self::Int64 => "i64",
            Self::Float64 => "f64",
            Self::Utf8 => "utf8",
            Self::Decimal128 { .. } => "decimal128",
        };
        f.write_str(ty)
    }
}

#[cfg(test)]
mod tests {
    use super::{JitBinaryOp, JitExpr, JitScalar, JitType};

    #[test]
    fn formats_jit_expression_for_mlir_text() {
        let expr = JitExpr::Binary {
            op: JitBinaryOp::Gt,
            left: Box::new(JitExpr::Column {
                index: 0,
                name: "a".to_string(),
                ty: JitType::Int64,
                nullable: true,
            }),
            right: Box::new(JitExpr::Literal(JitScalar::Int64(10))),
            ty: JitType::Bool,
            nullable: true,
        };

        assert_eq!(expr.to_string(), "(col(0, a, i64, nullable=true) > 10:i64)");
    }
}
