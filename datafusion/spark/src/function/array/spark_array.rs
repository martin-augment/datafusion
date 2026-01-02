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

use std::{any::Any, sync::Arc};

use arrow::array::{Array, ArrayRef, new_null_array};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::utils::SingleRowListArrayBuilder;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_functions_nested::make_array::{array_array, coerce_types_inner};

use crate::function::functions_nested_utils::make_scalar_function;

const ARRAY_FIELD_DEFAULT_NAME: &str = "element";

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArray {
    signature: Signature,
}

impl Default for SparkArray {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArray {
    /// Creates a new SparkArray UDF configured as user-defined and immutable.
    ///
    /// # Examples
    ///
    /// ```
    /// let udf = SparkArray::new();
    /// assert_eq!(udf.name(), "array");
    /// ```
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_types = args
            .arg_fields
            .iter()
            .map(|f| f.data_type())
            .cloned()
            .collect::<Vec<_>>();

        let mut expr_type = DataType::Null;
        for arg_type in &data_types {
            if !arg_type.equals_datatype(&DataType::Null) {
                expr_type = arg_type.clone();
                break;
            }
        }

        let return_type = DataType::List(Arc::new(Field::new(
            ARRAY_FIELD_DEFAULT_NAME,
            expr_type,
            true,
        )));

        Ok(Arc::new(Field::new(
            "this_field_name_is_irrelevant",
            return_type,
            false,
        )))
    }

    /// Constructs an array value from the provided scalar-function arguments.
    ///
    /// # Returns
    ///
    /// A `ColumnarValue` containing the produced array (a List array) built from the arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// // Create an array from zero or more ArrayRef arguments using the same logic
    /// // that `invoke_with_args` delegates to.
    /// use std::sync::Arc;
    /// use datafusion_common::Result;
    /// use datafusion_expr::ColumnarValue;
    ///
    /// // `make_array_inner` is the core builder used by `invoke_with_args`.
    /// // Here we call it directly with an empty slice to demonstrate usage.
    /// let result: Result<Arc<dyn std::any::Any>> = (|| {
    ///     // This mirrors: make_scalar_function(make_array_inner)(args)
    ///     let arrays: &[std::sync::Arc<dyn arrow::array::Array>] = &[];
    ///     let array_ref = crate::function::array::spark_array::make_array_inner(arrays)?;
    ///     // wrap as ColumnarValue::Array
    ///     let cv = ColumnarValue::Array(array_ref);
    ///     // Return as any just to satisfy type expectations in this minimal example.
    ///     Ok(Arc::new(cv) as Arc<dyn std::any::Any>)
    /// })();
    /// ```
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        make_scalar_function(make_array_inner)(args.as_slice())
    }

    /// Determines the target data types to which each argument should be coerced for the `array` function.
    ///
    /// If `arg_types` is empty, returns an empty vector. Otherwise returns a vector with the coerced
    /// element data type for each input so that the resulting array has a single consistent element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow::datatypes::DataType;
    /// use crate::SparkArray;
    ///
    /// let udf = SparkArray::new();
    /// let coerced = udf.coerce_types(&[]).unwrap();
    /// assert_eq!(coerced, Vec::<DataType>::new());
    /// ```
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.is_empty() {
            Ok(vec![])
        } else {
            coerce_types_inner(arg_types, self.name())
        }
    }
}

/// `make_array_inner` is the implementation of the `make_array` function.
/// Constructs an array using the input `data` as `ArrayRef`.
/// Returns a reference-counted `Array` instance result.
pub fn make_array_inner(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let mut data_type = DataType::Null;
    for arg in arrays {
        let arg_data_type = arg.data_type();
        if !arg_data_type.equals_datatype(&DataType::Null) {
            data_type = arg_data_type.clone();
            break;
        }
    }

    match data_type {
        // Either an empty array or all nulls:
        DataType::Null => {
            let length = arrays.iter().map(|a| a.len()).sum();
            // By default Int32
            let array = new_null_array(&DataType::Null, length);
            Ok(Arc::new(
                SingleRowListArrayBuilder::new(array)
                    .with_nullable(true)
                    .with_field_name(Some(ARRAY_FIELD_DEFAULT_NAME.to_string()))
                    .build_list_array(),
            ))
        }
        _ => array_array::<i32>(arrays, data_type, ARRAY_FIELD_DEFAULT_NAME),
    }
}