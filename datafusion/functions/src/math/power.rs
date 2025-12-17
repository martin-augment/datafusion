// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Math function: `power()`.
use std::any::Any;

use super::log::LogFunc;

use crate::utils::{calculate_binary_decimal_math, calculate_binary_math};
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{
    ArrowNativeTypeOp, DataType, Decimal32Type, Decimal64Type, Decimal128Type,
    Decimal256Type, Float64Type, Int64Type,
};
use arrow::error::ArrowError;
use datafusion_common::types::{NativeType, logical_float64, logical_int64};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyInfo};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, TypeSignature, TypeSignatureClass, Volatility, lit,
};
use datafusion_macros::user_doc;

#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns a base expression raised to the power of an exponent.",
    syntax_example = "power(base, exponent)",
    sql_example = r#"```sql
> SELECT power(2, 3);
+-------------+
| power(2,3)  |
+-------------+
| 8           |
+-------------+
```"#,
    standard_argument(name = "base", prefix = "Numeric"),
    standard_argument(name = "exponent", prefix = "Exponent numeric")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PowerFunc {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for PowerFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerFunc {
    pub fn new() -> Self {
        let integer = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int64()),
            vec![TypeSignatureClass::Integer],
            NativeType::Int64,
        );
        let decimal = Coercion::new_exact(TypeSignatureClass::Decimal);
        let float = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_float64()),
            vec![TypeSignatureClass::Numeric],
            NativeType::Float64,
        );
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Coercible(vec![decimal.clone(), integer]),
                    TypeSignature::Coercible(vec![decimal.clone(), float.clone()]),
                    TypeSignature::Coercible(vec![float; 2]),
                ],
                Volatility::Immutable,
            ),
            aliases: vec![String::from("pow")],
        }
    }
}

/// Binary function to calculate a math power to integer exponent
/// for scaled integer types.
///
/// Formula
/// The power for a scaled integer `b` is
///
/// ```text
/// (b * 10^(-s)) ^ e
/// ```
/// However, the result should be scaled back from scale 0 to scale `s`,
/// which is done by multiplying by `10^s`.
/// At the end, the formula is:
///
/// ```text
///   b^e * 10^(-s * e) * 10^s = b^e / 10^(s * (e-1))
/// ```
/// Example of 2.5 ^ 4 = 39:
///   2.5 is represented as 25 with scale 1
///   The unscaled result is 25^4 = 390625
///   Scale it back to 1: 390625 / 10^4 = 39
fn pow_decimal_int<T>(base: T, scale: i8, exp: i64) -> Result<T, ArrowError>
where
    T: From<i32> + ArrowNativeTypeOp,
{
    if exp < 0 {
        return pow_decimal_float(base, scale, exp as f64);
    }

    let scale: u32 = scale.try_into().map_err(|_| {
        ArrowError::NotYetImplemented(format!(
            "Negative scale is not yet supported value: {scale}"
        ))
    })?;
    if exp == 0 {
        // Edge case to provide 1 as result (10^s with scale)
        let result: T = T::from(10).pow_checked(scale).map_err(|_| {
            ArrowError::ArithmeticOverflow(format!(
                "Cannot make unscale factor for {scale} and {exp}"
            ))
        })?;
        return Ok(result);
    }
    let exp: u32 = exp.try_into().map_err(|_| {
        ArrowError::ArithmeticOverflow(format!("Unsupported exp value: {exp}"))
    })?;
    let powered: T = base.pow_checked(exp).map_err(|_| {
        ArrowError::ArithmeticOverflow(format!("Cannot raise base {base:?} to exp {exp}"))
    })?;
    let unscale_factor: T = T::from(10).pow_checked(scale * (exp - 1)).map_err(|_| {
        ArrowError::ArithmeticOverflow(format!(
            "Cannot make unscale factor for {scale} and {exp}"
        ))
    })?;

    powered.div_checked(unscale_factor)
}

/// Binary function to calculate a math power to float exponent
/// for scaled integer types.
fn pow_decimal_float<T>(base: T, scale: i8, exp: f64) -> Result<T, ArrowError>
where
    T: From<i32> + ArrowNativeTypeOp,
{
    if exp.is_finite() && exp.trunc() == exp && exp >= 0f64 && exp < u32::MAX as f64 {
        return pow_decimal_int(base, scale, exp as i64);
    }

    if !exp.is_finite() {
        return Err(ArrowError::ComputeError(format!(
            "Cannot use non-finite exp: {exp}"
        )));
    }

    pow_decimal_float_fallback(base, scale, exp)
}

/// Fallback implementation using f64 for negative or non-integer exponents.
/// This handles cases that cannot be computed using integer arithmetic.
fn pow_decimal_float_fallback<T>(base: T, scale: i8, exp: f64) -> Result<T, ArrowError>
where
    T: From<i32> + ArrowNativeTypeOp,
{
    let scale_factor = 10f64.powi(scale as i32);
    let base_f64 = format!("{base:?}")
        .parse::<f64>()
        .map(|v| v / scale_factor)
        .map_err(|_| {
            ArrowError::ComputeError(format!("Cannot convert base {base:?} to f64"))
        })?;

    let result_f64 = base_f64.powf(exp);

    if !result_f64.is_finite() {
        return Err(ArrowError::ArithmeticOverflow(format!(
            "Result of {base_f64}^{exp} is not finite"
        )));
    }

    let result_scaled = result_f64 * scale_factor;
    let result_rounded = result_scaled.round();

    if result_rounded.abs() > i128::MAX as f64 {
        return Err(ArrowError::ArithmeticOverflow(format!(
            "Result {result_rounded} is too large for the target decimal type"
        )));
    }

    decimal_from_i128::<T>(result_rounded as i128)
}

fn decimal_from_i128<T>(value: i128) -> Result<T, ArrowError>
where
    T: From<i32> + ArrowNativeTypeOp,
{
    if value == 0 {
        return Ok(T::from(0));
    }

    if value >= i32::MIN as i128 && value <= i32::MAX as i128 {
        return Ok(T::from(value as i32));
    }

    let is_negative = value < 0;
    let abs_value = value.unsigned_abs();

    let billion = 1_000_000_000u128;
    let mut result = T::from(0);
    let mut multiplier = T::from(1);
    let billion_t = T::from(1_000_000_000);

    let mut remaining = abs_value;
    while remaining > 0 {
        let chunk = (remaining % billion) as i32;
        remaining /= billion;

        let chunk_value = T::from(chunk).mul_checked(multiplier).map_err(|_| {
            ArrowError::ArithmeticOverflow(format!(
                "Overflow while converting {value} to decimal type"
            ))
        })?;

        result = result.add_checked(chunk_value).map_err(|_| {
            ArrowError::ArithmeticOverflow(format!(
                "Overflow while converting {value} to decimal type"
            ))
        })?;

        if remaining > 0 {
            multiplier = multiplier.mul_checked(billion_t).map_err(|_| {
                ArrowError::ArithmeticOverflow(format!(
                    "Overflow while converting {value} to decimal type"
                ))
            })?;
        }
    }

    if is_negative {
        result = T::from(0).sub_checked(result).map_err(|_| {
            ArrowError::ArithmeticOverflow(format!(
                "Overflow while negating {value} in decimal type"
            ))
        })?;
    }

    Ok(result)
}

impl ScalarUDFImpl for PowerFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "power"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types[0].is_null() {
            Ok(DataType::Float64)
        } else {
            Ok(arg_types[0].clone())
        }
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [base, exponent] = take_function_args(self.name(), &args.args)?;
        let base = base.to_array(args.number_rows)?;

        let arr: ArrayRef = match (base.data_type(), exponent.data_type()) {
            (DataType::Float64, DataType::Float64) => {
                calculate_binary_math::<Float64Type, Float64Type, Float64Type, _>(
                    &base,
                    exponent,
                    |b, e| Ok(f64::powf(b, e)),
                )?
            }
            (DataType::Decimal32(precision, scale), DataType::Int64) => {
                calculate_binary_decimal_math::<Decimal32Type, Int64Type, Decimal32Type, _>(
                    &base,
                    exponent,
                    |b, e| pow_decimal_int(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal32(precision, scale), DataType::Float64) => {
                calculate_binary_decimal_math::<
                    Decimal32Type,
                    Float64Type,
                    Decimal32Type,
                    _,
                >(
                    &base,
                    exponent,
                    |b, e| pow_decimal_float(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal64(precision, scale), DataType::Int64) => {
                calculate_binary_decimal_math::<Decimal64Type, Int64Type, Decimal64Type, _>(
                    &base,
                    exponent,
                    |b, e| pow_decimal_int(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal64(precision, scale), DataType::Float64) => {
                calculate_binary_decimal_math::<
                    Decimal64Type,
                    Float64Type,
                    Decimal64Type,
                    _,
                >(
                    &base,
                    exponent,
                    |b, e| pow_decimal_float(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal128(precision, scale), DataType::Int64) => {
                calculate_binary_decimal_math::<
                    Decimal128Type,
                    Int64Type,
                    Decimal128Type,
                    _,
                >(
                    &base,
                    exponent,
                    |b, e| pow_decimal_int(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal128(precision, scale), DataType::Float64) => {
                calculate_binary_decimal_math::<
                    Decimal128Type,
                    Float64Type,
                    Decimal128Type,
                    _,
                >(
                    &base,
                    exponent,
                    |b, e| pow_decimal_float(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal256(precision, scale), DataType::Int64) => {
                calculate_binary_decimal_math::<
                    Decimal256Type,
                    Int64Type,
                    Decimal256Type,
                    _,
                >(
                    &base,
                    exponent,
                    |b, e| pow_decimal_int(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (DataType::Decimal256(precision, scale), DataType::Float64) => {
                calculate_binary_decimal_math::<
                    Decimal256Type,
                    Float64Type,
                    Decimal256Type,
                    _,
                >(
                    &base,
                    exponent,
                    |b, e| pow_decimal_float(b, *scale, e),
                    *precision,
                    *scale,
                )?
            }
            (base_type, exp_type) => {
                return internal_err!(
                    "Unsupported data types for base {base_type:?} and exponent {exp_type:?} for power"
                );
            }
        };
        Ok(ColumnarValue::Array(arr))
    }

    /// Simplify the `power` function by the relevant rules:
    /// 1. Power(a, 0) ===> 1
    /// 2. Power(a, 1) ===> a
    /// 3. Power(a, Log(a, b)) ===> b
    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &dyn SimplifyInfo,
    ) -> Result<ExprSimplifyResult> {
        let [base, exponent] = take_function_args("power", args)?;
        let base_type = info.get_data_type(&base)?;
        let exponent_type = info.get_data_type(&exponent)?;

        // Null propagation
        if base_type.is_null() || exponent_type.is_null() {
            let return_type = self.return_type(&[base_type, exponent_type])?;
            return Ok(ExprSimplifyResult::Simplified(lit(
                ScalarValue::Null.cast_to(&return_type)?
            )));
        }

        match exponent {
            Expr::Literal(value, _)
                if value == ScalarValue::new_zero(&exponent_type)? =>
            {
                Ok(ExprSimplifyResult::Simplified(lit(ScalarValue::new_one(
                    &base_type,
                )?)))
            }
            Expr::Literal(value, _) if value == ScalarValue::new_one(&exponent_type)? => {
                Ok(ExprSimplifyResult::Simplified(base))
            }
            Expr::ScalarFunction(ScalarFunction { func, mut args })
                if is_log(&func) && args.len() == 2 && base == args[0] =>
            {
                let b = args.pop().unwrap(); // length checked above
                Ok(ExprSimplifyResult::Simplified(b))
            }
            _ => Ok(ExprSimplifyResult::Original(vec![base, exponent])),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Return true if this function call is a call to `Log`
fn is_log(func: &ScalarUDF) -> bool {
    func.inner().as_any().downcast_ref::<LogFunc>().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_decimal128_helper() {
        // Expression: 2.5 ^ 4 = 39.0625
        assert_eq!(pow_decimal_int(25, 1, 4).unwrap(), i128::from(390));
        assert_eq!(pow_decimal_int(2500, 3, 4).unwrap(), i128::from(39062));
        assert_eq!(pow_decimal_int(25000, 4, 4).unwrap(), i128::from(390625));

        // Expression: 25 ^ 4 = 390625
        assert_eq!(pow_decimal_int(25, 0, 4).unwrap(), i128::from(390625));

        // Expressions for edge cases
        assert_eq!(pow_decimal_int(25, 1, 1).unwrap(), i128::from(25));
        assert_eq!(pow_decimal_int(25, 0, 1).unwrap(), i128::from(25));
        assert_eq!(pow_decimal_int(25, 0, 0).unwrap(), i128::from(1));
        assert_eq!(pow_decimal_int(25, 1, 0).unwrap(), i128::from(10));

        assert_eq!(
            pow_decimal_int(25, -1, 4).unwrap_err().to_string(),
            "Not yet implemented: Negative scale is not yet supported value: -1"
        );
    }

    #[test]
    fn test_pow_decimal_float_fallback() {
        // Test negative exponent: 4^(-1) = 0.25
        // 4 with scale 2 = 400, result should be 25 (0.25 with scale 2)
        let result: i128 = pow_decimal_float(400i128, 2, -1.0).unwrap();
        assert_eq!(result, 25);

        // Test non-integer exponent: 4^0.5 = 2
        // 4 with scale 2 = 400, result should be 200 (2.0 with scale 2)
        let result: i128 = pow_decimal_float(400i128, 2, 0.5).unwrap();
        assert_eq!(result, 200);

        // Test 8^(1/3) = 2 (cube root)
        // 8 with scale 1 = 80, result should be 20 (2.0 with scale 1)
        let result: i128 = pow_decimal_float(80i128, 1, 1.0 / 3.0).unwrap();
        assert_eq!(result, 20);

        // Test negative base with integer exponent still works
        // (-2)^3 = -8
        // -2 with scale 1 = -20, result should be -80 (-8.0 with scale 1)
        let result: i128 = pow_decimal_float(-20i128, 1, 3.0).unwrap();
        assert_eq!(result, -80);

        // Test positive integer exponent goes through fast path
        // 2.5^4 = 39.0625
        // 25 with scale 1, result should be 390 (39.0 with scale 1) - truncated
        let result: i128 = pow_decimal_float(25i128, 1, 4.0).unwrap();
        assert_eq!(result, 390); // Uses integer path

        // Test non-finite exponent returns error
        assert!(pow_decimal_float(100i128, 2, f64::NAN).is_err());
        assert!(pow_decimal_float(100i128, 2, f64::INFINITY).is_err());
    }
}
