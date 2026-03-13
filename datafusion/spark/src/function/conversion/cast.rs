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

use arrow::array::{Array, ArrayRef, AsArray, TimestampMicrosecondBuilder};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Int8Type, Int16Type, Int32Type, Int64Type, TimeUnit,
};
use datafusion_common::{Result as DataFusionResult, ScalarValue, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use std::any::Any;
use std::sync::Arc;
const MICROS_PER_SECOND: i64 = 1_000_000;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Cast {
    signature: Signature,
}
impl Default for Cast {
    fn default() -> Self {
        Self::new()
    }
}

impl Cast {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

fn cast_int_to_timestamp<T: ArrowPrimitiveType>(
    array: &ArrayRef,
) -> DataFusionResult<ArrayRef>
where
    T::Native: Into<i64>,
{
    let arr = array.as_primitive::<T>();
    let mut builder = TimestampMicrosecondBuilder::with_capacity(arr.len());

    for i in 0..arr.len() {
        if arr.is_null(i) {
            builder.append_null();
        } else {
            let micros = (arr.value(i).into()).saturating_mul(MICROS_PER_SECOND);
            builder.append_value(micros);
        }
    }

    Ok(Arc::new(builder.finish()))
}

impl ScalarUDFImpl for Cast {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "spark_cast"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DataFusionResult<DataType> {
        // for now we will be supporting int -> timestamp and keep adding more spark-compatible spark
        match &arg_types[0] {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                Ok(DataType::Timestamp(TimeUnit::Microsecond, None))
            }
            _ => exec_err!("Unsupported cast from {:?}", arg_types[0]),
        }
    }

    fn invoke_with_args(
        &self,
        args: ScalarFunctionArgs,
    ) -> DataFusionResult<ColumnarValue> {
        let input = &args.args[0];
        match input {
            ColumnarValue::Array(array) => match array.data_type() {
                DataType::Int8 => {
                    let result = cast_int_to_timestamp::<Int8Type>(array)?;
                    Ok(ColumnarValue::Array(result))
                }
                DataType::Int16 => {
                    let result = cast_int_to_timestamp::<Int16Type>(array)?;
                    Ok(ColumnarValue::Array(result))
                }
                DataType::Int32 => {
                    let result = cast_int_to_timestamp::<Int32Type>(array)?;
                    Ok(ColumnarValue::Array(result))
                }
                DataType::Int64 => {
                    let result = cast_int_to_timestamp::<Int64Type>(array)?;
                    Ok(ColumnarValue::Array(result))
                }
                _ => exec_err!(
                    "Unsupported cast from {:?} to timestamp",
                    array.data_type()
                ),
            },
            ColumnarValue::Scalar(scalar) => {
                // Handle scalar conversions
                match scalar {
                    ScalarValue::Int8(None)
                    | ScalarValue::Int16(None)
                    | ScalarValue::Int32(None)
                    | ScalarValue::Int64(None) => Ok(ColumnarValue::Scalar(
                        ScalarValue::TimestampMicrosecond(None, None),
                    )),
                    ScalarValue::Int8(Some(v)) => {
                        let micros = (*v as i64).saturating_mul(MICROS_PER_SECOND);
                        Ok(ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                            Some(micros),
                            None,
                        )))
                    }
                    ScalarValue::Int16(Some(v)) => {
                        let micros = (*v as i64).saturating_mul(MICROS_PER_SECOND);
                        Ok(ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                            Some(micros),
                            None,
                        )))
                    }
                    ScalarValue::Int32(Some(v)) => {
                        let micros = (*v as i64).saturating_mul(MICROS_PER_SECOND);
                        Ok(ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                            Some(micros),
                            None,
                        )))
                    }
                    ScalarValue::Int64(Some(v)) => {
                        let micros = (*v).saturating_mul(MICROS_PER_SECOND);
                        Ok(ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                            Some(micros),
                            None,
                        )))
                    }
                    _ => exec_err!("Unsupported cast from {:?} to timestamp", scalar),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int8Array, Int16Array, Int32Array, Int64Array};
    use arrow::datatypes::{Field, TimestampMicrosecondType};
    use datafusion_expr::ScalarFunctionArgs;

    fn make_args(input: ColumnarValue) -> ScalarFunctionArgs {
        let return_field = Arc::new(Field::new(
            "result",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ));
        ScalarFunctionArgs {
            args: vec![input],
            arg_fields: vec![],
            number_rows: 0,
            return_field,
            config_options: Arc::new(Default::default()),
        }
    }

    fn assert_scalar_timestamp(result: ColumnarValue, expected: i64) {
        match result {
            ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(Some(val), None)) => {
                assert_eq!(val, expected);
            }
            _ => panic!("Expected scalar timestamp with value {expected}"),
        }
    }

    fn assert_scalar_null(result: ColumnarValue) {
        assert!(matches!(
            result,
            ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(None, None))
        ));
    }

    #[test]
    fn test_cast_int8_array_to_timestamp() {
        let array: ArrayRef = Arc::new(Int8Array::from(vec![
            Some(0),
            Some(1),
            Some(-1),
            Some(127),
            Some(-128),
            None,
        ]));

        let cast = Cast::new();
        let args = make_args(ColumnarValue::Array(array));
        let result = cast.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Array(result_array) => {
                let ts_array = result_array.as_primitive::<TimestampMicrosecondType>();
                assert_eq!(ts_array.value(0), 0);
                assert_eq!(ts_array.value(1), 1_000_000);
                assert_eq!(ts_array.value(2), -1_000_000);
                assert_eq!(ts_array.value(3), 127_000_000);
                assert_eq!(ts_array.value(4), -128_000_000);
                assert!(ts_array.is_null(5));
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_cast_int16_array_to_timestamp() {
        let array: ArrayRef = Arc::new(Int16Array::from(vec![
            Some(0),
            Some(32767),
            Some(-32768),
            None,
        ]));

        let cast = Cast::new();
        let args = make_args(ColumnarValue::Array(array));
        let result = cast.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Array(result_array) => {
                let ts_array = result_array.as_primitive::<TimestampMicrosecondType>();
                assert_eq!(ts_array.value(0), 0);
                assert_eq!(ts_array.value(1), 32_767_000_000);
                assert_eq!(ts_array.value(2), -32_768_000_000);
                assert!(ts_array.is_null(3));
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_cast_int32_array_to_timestamp() {
        let array: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(0), Some(1704067200), None]));

        let cast = Cast::new();
        let args = make_args(ColumnarValue::Array(array));
        let result = cast.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Array(result_array) => {
                let ts_array = result_array.as_primitive::<TimestampMicrosecondType>();
                assert_eq!(ts_array.value(0), 0);
                assert_eq!(ts_array.value(1), 1_704_067_200_000_000);
                assert!(ts_array.is_null(2));
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_cast_int64_array_overflow() {
        let array: ArrayRef =
            Arc::new(Int64Array::from(vec![Some(i64::MAX), Some(i64::MIN)]));

        let cast = Cast::new();
        let args = make_args(ColumnarValue::Array(array));
        let result = cast.invoke_with_args(args).unwrap();

        match result {
            ColumnarValue::Array(result_array) => {
                let ts_array = result_array.as_primitive::<TimestampMicrosecondType>();
                assert_eq!(ts_array.value(0), i64::MAX);
                assert_eq!(ts_array.value(1), i64::MIN);
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_cast_scalar_int8() {
        let cast = Cast::new();
        let args = make_args(ColumnarValue::Scalar(ScalarValue::Int8(Some(100))));
        let result = cast.invoke_with_args(args).unwrap();
        assert_scalar_timestamp(result, 100_000_000);
    }

    #[test]
    fn test_cast_scalar_int32() {
        let cast = Cast::new();
        let args = make_args(ColumnarValue::Scalar(ScalarValue::Int32(Some(1704067200))));
        let result = cast.invoke_with_args(args).unwrap();
        assert_scalar_timestamp(result, 1_704_067_200_000_000);
    }

    #[test]
    fn test_cast_scalar_null() {
        let cast = Cast::new();
        let args = make_args(ColumnarValue::Scalar(ScalarValue::Int64(None)));
        let result = cast.invoke_with_args(args).unwrap();
        assert_scalar_null(result);
    }

    #[test]
    fn test_cast_scalar_int64_overflow() {
        let cast = Cast::new();
        let args = make_args(ColumnarValue::Scalar(ScalarValue::Int64(Some(i64::MAX))));
        let result = cast.invoke_with_args(args).unwrap();
        assert_scalar_timestamp(result, i64::MAX);
    }

    #[test]
    fn test_unsupported_scalar_type() {
        let cast = Cast::new();
        let args = make_args(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
            "2024-01-01".to_string(),
        ))));
        let result = cast.invoke_with_args(args);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported cast from")
        );
    }

    #[test]
    fn test_unsupported_array_type() {
        let cast = Cast::new();
        let array: ArrayRef =
            Arc::new(arrow::array::Float32Array::from(vec![1.0, 2.0, 3.0]));
        let args = make_args(ColumnarValue::Array(array));
        let result = cast.invoke_with_args(args);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported cast from")
        );
    }
}
