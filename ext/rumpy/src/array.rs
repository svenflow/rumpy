//! Core NDArray type and operations.
//!
//! # Behavior Differences from NumPy
//!
//! ## Data Type
//! RumPy only supports float64 (f64) arrays. Integer and complex types are not supported.
//! All numeric inputs are converted to f64.
//!
//! ## NaN Handling
//! - Comparisons: NaN values compare as expected (NaN != NaN is true)
//! - Sorting: NaN values sort to the end (greater than all other values)
//! - Logical operations: NaN is truthy (non-zero)
//! - Reductions: Most functions propagate NaN; use nan* variants for NaN-ignoring behavior
//!
//! ## Broadcasting
//! Full NumPy-style broadcasting is supported for binary operations and functions
//! like where(), arctan2(), and hypot().
//!
//! ## Memory Layout
//! Arrays use row-major (C-style) memory layout, matching NumPy's default.

use magnus::{
    class, define_class, exception, function, method, prelude::*, typed_data::Obj,
    Error, IntoValue, RArray, Ruby, Symbol, TryConvert, Value,
};
use ndarray::{Array, ArrayD, Axis, IxDyn, ShapeBuilder, Slice};
use std::cell::RefCell;
use std::fmt;

/// The main N-dimensional array type
#[magnus::wrap(class = "RumPy::NDArray")]
pub struct NDArray {
    data: RefCell<ArrayD<f64>>,
    dtype: String,
}

impl NDArray {
    /// Create a new NDArray from an ndarray ArrayD
    pub fn new(arr: ArrayD<f64>) -> Self {
        NDArray {
            data: RefCell::new(arr),
            dtype: "float64".to_string(),
        }
    }

    /// Create from a Ruby array (nested arrays supported)
    pub fn from_ruby_array(value: Value) -> Result<Self, Error> {
        let (data, shape) = flatten_nested_array(value)?;
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| Error::new(exception::arg_error(), format!("Invalid shape: {}", e)))?;
        Ok(NDArray::new(arr))
    }

    /// Get the shape as a Ruby array
    pub fn shape(&self) -> Vec<usize> {
        self.data.borrow().shape().to_vec()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.borrow().ndim()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.data.borrow().len()
    }

    /// Get the data type
    pub fn dtype(&self) -> String {
        self.dtype.clone()
    }

    /// Convert to nested Ruby arrays
    pub fn to_a(&self) -> Result<Value, Error> {
        let ruby = Ruby::get().unwrap();
        array_to_ruby_nested(&ruby, &self.data.borrow())
    }

    /// String representation
    pub fn to_s(&self) -> String {
        format!("{}", self.data.borrow())
    }

    /// Inspect representation
    pub fn inspect(&self) -> String {
        format!(
            "RumPy::NDArray(shape={:?}, dtype={})\n{}",
            self.shape(),
            self.dtype,
            self.data.borrow()
        )
    }

    /// Get element at index (supports integers, arrays, ranges, and slicing)
    pub fn get(&self, index: Value) -> Result<Value, Error> {
        let ruby = Ruby::get().unwrap();
        let data = self.data.borrow();

        // Handle integer index
        if let Ok(i) = i64::try_convert(index) {
            let idx = normalize_index(i, data.shape()[0])?;
            if data.ndim() == 1 {
                return Ok(data[[idx]].into_value_with(&ruby));
            }
            // For multi-dim, return a slice
            let slice = data.index_axis(Axis(0), idx);
            return Ok(Obj::wrap(NDArray::new(slice.to_owned())).into_value_with(&ruby));
        }

        // Handle Range (for slicing like arr[1:3] which Ruby represents as arr[1..2] or arr[1...3])
        if is_range(&ruby, index) {
            let (start, end, exclusive) = extract_range(&ruby, index)?;
            let len = data.shape()[0];

            let start_idx = match start {
                Some(s) => normalize_index(s, len)?,
                None => 0,
            };
            let end_idx = match end {
                Some(e) => {
                    let normalized = if e < 0 { (len as i64 + e) as usize } else { e as usize };
                    if exclusive { normalized } else { (normalized + 1).min(len) }
                }
                None => len,
            };

            if start_idx > end_idx || start_idx >= len {
                // Return empty array
                let new_shape = std::iter::once(0)
                    .chain(data.shape()[1..].iter().cloned())
                    .collect::<Vec<_>>();
                return Ok(Obj::wrap(NDArray::new(
                    ArrayD::from_shape_vec(IxDyn(&new_shape), vec![]).unwrap()
                )).into_value_with(&ruby));
            }

            let slice = data.slice_axis(Axis(0), Slice::from(start_idx..end_idx));
            return Ok(Obj::wrap(NDArray::new(slice.to_owned())).into_value_with(&ruby));
        }

        // Handle array of indices (for multi-dimensional indexing)
        if let Ok(indices) = RArray::try_convert(index) {
            // Check if this is advanced slicing (array contains ranges or nil)
            let has_slicing = indices.into_iter().any(|v| is_range(&ruby, v) || v.is_nil());

            if has_slicing {
                return self.advanced_slice(&ruby, &data, indices);
            }

            // Simple integer indexing
            let idx_vec: Vec<usize> = indices
                .into_iter()
                .enumerate()
                .map(|(axis, v)| {
                    let i = i64::try_convert(v)?;
                    normalize_index(i, data.shape()[axis])
                })
                .collect::<Result<Vec<_>, _>>()?;

            let idx = IxDyn(&idx_vec);
            return Ok(data[&idx].into_value_with(&ruby));
        }

        Err(Error::new(exception::type_error(), "Invalid index type"))
    }

    /// Advanced slicing with ranges and nil (for selecting entire axes)
    fn advanced_slice(&self, ruby: &Ruby, data: &std::cell::Ref<ArrayD<f64>>, indices: RArray) -> Result<Value, Error> {
        let ndim = data.ndim();
        let idx_len = indices.len();

        if idx_len > ndim {
            return Err(Error::new(exception::index_error(), "too many indices for array"));
        }

        // Build slice info for each axis
        let mut slices: Vec<(usize, usize, bool)> = Vec::with_capacity(ndim); // (start, end, is_slice)

        for (axis, v) in indices.into_iter().enumerate() {
            let axis_len = data.shape()[axis];

            if v.is_nil() {
                // nil means select all (like : in NumPy)
                slices.push((0, axis_len, true));
            } else if is_range(ruby, v) {
                let (start, end, exclusive) = extract_range(ruby, v)?;
                let start_idx = match start {
                    Some(s) => normalize_index(s, axis_len)?,
                    None => 0,
                };
                let end_idx = match end {
                    Some(e) => {
                        let normalized = if e < 0 { (axis_len as i64 + e) as usize } else { e as usize };
                        if exclusive { normalized } else { (normalized + 1).min(axis_len) }
                    }
                    None => axis_len,
                };
                slices.push((start_idx, end_idx, true));
            } else if let Ok(i) = i64::try_convert(v) {
                let idx = normalize_index(i, axis_len)?;
                slices.push((idx, idx + 1, false)); // Single index, will be squeezed
            } else {
                return Err(Error::new(exception::type_error(), "Invalid index type in array"));
            }
        }

        // Fill remaining axes with full selection
        for axis in idx_len..ndim {
            slices.push((0, data.shape()[axis], true));
        }

        // Build result shape (excluding squeezed dimensions)
        let mut result_shape: Vec<usize> = Vec::new();
        for (axis, &(start, end, is_slice)) in slices.iter().enumerate() {
            if is_slice {
                result_shape.push(end - start);
            }
            // Single indices are squeezed out
        }

        // Extract data
        let mut result_data: Vec<f64> = Vec::new();
        let shape = data.shape();

        // Recursive extraction
        fn extract_recursive(
            data: &ArrayD<f64>,
            slices: &[(usize, usize, bool)],
            shape: &[usize],
            current_idx: &mut Vec<usize>,
            axis: usize,
            result: &mut Vec<f64>,
        ) {
            if axis == slices.len() {
                result.push(data[IxDyn(current_idx)]);
                return;
            }

            let (start, end, _) = slices[axis];
            for i in start..end {
                current_idx.push(i);
                extract_recursive(data, slices, shape, current_idx, axis + 1, result);
                current_idx.pop();
            }
        }

        let mut current_idx = Vec::new();
        extract_recursive(&data, &slices, shape, &mut current_idx, 0, &mut result_data);

        if result_shape.is_empty() {
            // Scalar result
            if result_data.is_empty() {
                return Err(Error::new(exception::index_error(), "index out of bounds"));
            }
            return Ok(result_data[0].into_value_with(ruby));
        }

        let result = ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
            .map_err(|e| Error::new(exception::arg_error(), format!("Failed to create slice: {}", e)))?;

        Ok(Obj::wrap(NDArray::new(result)).into_value_with(ruby))
    }

    /// Set element at index
    pub fn set(&self, index: Value, value: Value) -> Result<Value, Error> {
        let ruby = Ruby::get().unwrap();
        let mut data = self.data.borrow_mut();
        let val = f64::try_convert(value)?;

        if let Ok(i) = i64::try_convert(index) {
            let idx = normalize_index(i, data.len())?;
            if data.ndim() == 1 {
                data[[idx]] = val;
                return Ok(val.into_value_with(&ruby));
            }
        }

        if let Ok(indices) = RArray::try_convert(index) {
            let idx_vec: Vec<usize> = indices
                .into_iter()
                .enumerate()
                .map(|(axis, v)| {
                    let i = i64::try_convert(v)?;
                    normalize_index(i, data.shape()[axis])
                })
                .collect::<Result<Vec<_>, _>>()?;

            let idx = IxDyn(&idx_vec);
            data[&idx] = val;
            return Ok(val.into_value_with(&ruby));
        }

        Err(Error::new(exception::type_error(), "Invalid index type"))
    }

    /// Reshape the array
    pub fn reshape(&self, shape: RArray) -> Result<Obj<NDArray>, Error> {
        let shape_vec: Vec<usize> = shape
            .into_iter()
            .map(|v| usize::try_convert(v))
            .collect::<Result<Vec<_>, _>>()?;

        let data = self.data.borrow();
        let reshaped = data
            .clone()
            .into_shape(IxDyn(&shape_vec))
            .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape: {}", e)))?;

        Ok(Obj::wrap(NDArray::new(reshaped)))
    }

    /// Flatten to 1D
    pub fn flatten(&self) -> Obj<NDArray> {
        let data = self.data.borrow();
        let flat: Vec<f64> = data.iter().cloned().collect();
        Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap()))
    }

    /// Ravel (flatten, potentially returning a view)
    pub fn ravel(&self) -> Obj<NDArray> {
        self.flatten()
    }

    /// Transpose the array
    pub fn transpose(&self) -> Obj<NDArray> {
        let data = self.data.borrow();
        Obj::wrap(NDArray::new(data.t().to_owned()))
    }

    /// Create a copy
    pub fn copy(&self) -> Obj<NDArray> {
        let data = self.data.borrow();
        Obj::wrap(NDArray::new(data.clone()))
    }

    /// Convert to different dtype.
    ///
    /// # NotImplementedError
    /// Currently only float64 is supported. This method raises NotImplementedError
    /// for any dtype conversion other than "float64".
    pub fn astype(&self, dtype: String) -> Result<Obj<NDArray>, Error> {
        if dtype == "float64" || dtype == "f64" || dtype == "double" {
            Ok(self.copy())
        } else {
            Err(Error::new(
                exception::not_imp_error(),
                format!("dtype conversion to '{}' not implemented - only float64 is currently supported", dtype)
            ))
        }
    }

    // Arithmetic operations
    pub fn add(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        binary_op(&self.data.borrow(), other, |a, b| a + b)
    }

    pub fn sub(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        binary_op(&self.data.borrow(), other, |a, b| a - b)
    }

    pub fn mul(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        binary_op(&self.data.borrow(), other, |a, b| a * b)
    }

    pub fn div(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        binary_op(&self.data.borrow(), other, |a, b| a / b)
    }

    pub fn pow(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        binary_op(&self.data.borrow(), other, |a, b| a.powf(b))
    }

    pub fn modulo(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        binary_op(&self.data.borrow(), other, |a, b| a % b)
    }

    pub fn neg(&self) -> Obj<NDArray> {
        let data = self.data.borrow();
        Obj::wrap(NDArray::new(data.mapv(|x| -x)))
    }

    // Comparison operations
    pub fn eq(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        comparison_op(&self.data.borrow(), other, |a, b| if a == b { 1.0 } else { 0.0 })
    }

    pub fn ne(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        comparison_op(&self.data.borrow(), other, |a, b| if a != b { 1.0 } else { 0.0 })
    }

    pub fn lt(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        comparison_op(&self.data.borrow(), other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    pub fn le(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        comparison_op(&self.data.borrow(), other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }

    pub fn gt(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        comparison_op(&self.data.borrow(), other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    pub fn ge(&self, other: Value) -> Result<Obj<NDArray>, Error> {
        comparison_op(&self.data.borrow(), other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    // Aggregation methods
    pub fn sum(&self) -> f64 {
        self.data.borrow().sum()
    }

    pub fn prod(&self) -> f64 {
        self.data.borrow().product()
    }

    pub fn mean(&self) -> f64 {
        let data = self.data.borrow();
        if data.is_empty() {
            return f64::NAN;
        }
        data.sum() / data.len() as f64
    }

    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    pub fn var(&self) -> f64 {
        let data = self.data.borrow();
        let mean = data.sum() / data.len() as f64;
        data.mapv(|x| (x - mean).powi(2)).sum() / data.len() as f64
    }

    pub fn min(&self) -> Result<f64, Error> {
        let data = self.data.borrow();
        if data.is_empty() {
            return Err(Error::new(exception::arg_error(), "zero-size array to reduction operation minimum which has no identity"));
        }
        Ok(data.iter().cloned().fold(f64::INFINITY, f64::min))
    }

    pub fn max(&self) -> Result<f64, Error> {
        let data = self.data.borrow();
        if data.is_empty() {
            return Err(Error::new(exception::arg_error(), "zero-size array to reduction operation maximum which has no identity"));
        }
        Ok(data.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }

    pub fn argmin(&self) -> Result<usize, Error> {
        let data = self.data.borrow();
        if data.is_empty() {
            return Err(Error::new(exception::arg_error(), "attempt to get argmin of an empty sequence"));
        }
        Ok(data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap())
    }

    pub fn argmax(&self) -> Result<usize, Error> {
        let data = self.data.borrow();
        if data.is_empty() {
            return Err(Error::new(exception::arg_error(), "attempt to get argmax of an empty sequence"));
        }
        Ok(data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap())
    }

    pub fn all(&self) -> bool {
        // NumPy treats NaN as truthy (non-zero)
        // We explicitly check is_nan() to handle NaN correctly even though
        // NaN != 0.0 would also be true in IEEE 754 semantics
        self.data.borrow().iter().all(|&x| x != 0.0 || x.is_nan())
    }

    pub fn any(&self) -> bool {
        // NumPy treats NaN as truthy (non-zero)
        // We explicitly check is_nan() to handle NaN correctly even though
        // NaN != 0.0 would also be true in IEEE 754 semantics
        self.data.borrow().iter().any(|&x| x != 0.0 || x.is_nan())
    }

    /// Get internal data reference (for other modules)
    pub fn get_data(&self) -> std::cell::Ref<ArrayD<f64>> {
        self.data.borrow()
    }

    /// Get mutable internal data reference
    pub fn get_data_mut(&self) -> std::cell::RefMut<ArrayD<f64>> {
        self.data.borrow_mut()
    }
}

// Helper functions

fn normalize_index(i: i64, len: usize) -> Result<usize, Error> {
    let idx = if i < 0 {
        (len as i64 + i) as usize
    } else {
        i as usize
    };
    if idx >= len {
        return Err(Error::new(exception::index_error(), format!("Index {} out of bounds", i)));
    }
    Ok(idx)
}

/// Check if a Ruby value is a Range
fn is_range(ruby: &Ruby, value: Value) -> bool {
    // Try to call is_a? to check if it's a Range
    let result: Result<bool, _> = value.funcall("is_a?", (ruby.class_range(),));
    result.unwrap_or(false)
}

/// Extract start, end, and exclusive flag from a Ruby Range
fn extract_range(ruby: &Ruby, value: Value) -> Result<(Option<i64>, Option<i64>, bool), Error> {
    // Get begin (start)
    let begin_val: Value = value.funcall("begin", ())?;
    let start = if begin_val.is_nil() {
        None
    } else {
        Some(i64::try_convert(begin_val)?)
    };

    // Get end
    let end_val: Value = value.funcall("end", ())?;
    let end = if end_val.is_nil() {
        None
    } else {
        Some(i64::try_convert(end_val)?)
    };

    // Check if exclusive (... vs ..)
    let exclusive: bool = value.funcall("exclude_end?", ())?;

    Ok((start, end, exclusive))
}

fn flatten_nested_array(value: Value) -> Result<(Vec<f64>, Vec<usize>), Error> {
    fn recursive_flatten(value: Value, depth: usize, shapes: &mut Vec<usize>) -> Result<Vec<f64>, Error> {
        if let Ok(arr) = RArray::try_convert(value) {
            let len = arr.len();
            if depth >= shapes.len() {
                shapes.push(len);
            } else if shapes[depth] != len {
                return Err(Error::new(exception::arg_error(), "Ragged arrays not supported"));
            }

            let mut result = Vec::new();
            for item in arr.into_iter() {
                result.extend(recursive_flatten(item, depth + 1, shapes)?);
            }
            Ok(result)
        } else if let Ok(f) = f64::try_convert(value) {
            Ok(vec![f])
        } else if let Ok(i) = i64::try_convert(value) {
            Ok(vec![i as f64])
        } else {
            Err(Error::new(exception::type_error(), "Array elements must be numeric"))
        }
    }

    let mut shapes = Vec::new();
    let data = recursive_flatten(value, 0, &mut shapes)?;
    Ok((data, shapes))
}

fn array_to_ruby_nested(ruby: &Ruby, arr: &ArrayD<f64>) -> Result<Value, Error> {
    if arr.ndim() == 0 {
        return Ok(arr.iter().next().unwrap_or(&0.0).into_value_with(ruby));
    }
    if arr.ndim() == 1 {
        let result = RArray::new();
        for &x in arr.iter() {
            result.push(x)?;
        }
        return Ok(result.into_value_with(ruby));
    }

    let result = RArray::new();
    for i in 0..arr.shape()[0] {
        let slice = arr.index_axis(Axis(0), i);
        result.push(array_to_ruby_nested(ruby, &slice.to_owned())?)?;
    }
    Ok(result.into_value_with(ruby))
}

/// Compute broadcast shape from two input shapes (NumPy broadcasting rules)
fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, Error> {
    let ndim = shape1.len().max(shape2.len());
    let mut result = vec![0; ndim];

    for i in 0..ndim {
        let dim1 = if i < ndim - shape1.len() { 1 } else { shape1[shape1.len() - (ndim - i)] };
        let dim2 = if i < ndim - shape2.len() { 1 } else { shape2[shape2.len() - (ndim - i)] };

        if dim1 == dim2 {
            result[i] = dim1;
        } else if dim1 == 1 {
            result[i] = dim2;
        } else if dim2 == 1 {
            result[i] = dim1;
        } else {
            return Err(Error::new(
                exception::arg_error(),
                format!("operands could not be broadcast together with shapes {:?} {:?}", shape1, shape2)
            ));
        }
    }

    Ok(result)
}

/// Get the broadcast index for a given output index
fn broadcast_index(out_idx: &[usize], shape: &[usize], ndim: usize) -> Vec<usize> {
    let offset = ndim - shape.len();
    shape.iter().enumerate().map(|(i, &dim)| {
        if dim == 1 { 0 } else { out_idx[offset + i] }
    }).collect()
}

fn binary_op<F>(data: &ArrayD<f64>, other: Value, op: F) -> Result<Obj<NDArray>, Error>
where
    F: Fn(f64, f64) -> f64,
{
    // Try scalar
    if let Ok(scalar) = f64::try_convert(other) {
        return Ok(Obj::wrap(NDArray::new(data.mapv(|x| op(x, scalar)))));
    }
    if let Ok(scalar) = i64::try_convert(other) {
        return Ok(Obj::wrap(NDArray::new(data.mapv(|x| op(x, scalar as f64)))));
    }

    // Try NDArray
    if let Ok(other_arr) = <&NDArray>::try_convert(other) {
        let other_data = other_arr.get_data();

        // Simple case: same shape
        if data.shape() == other_data.shape() {
            let result = ndarray::Zip::from(data)
                .and(&*other_data)
                .map_collect(|&a, &b| op(a, b));
            return Ok(Obj::wrap(NDArray::new(result)));
        }

        // NumPy-style broadcasting
        let out_shape = broadcast_shape(data.shape(), other_data.shape())?;
        let out_ndim = out_shape.len();
        let out_size: usize = out_shape.iter().product();

        let mut result_data = Vec::with_capacity(out_size);

        // Iterate over all output indices
        let mut out_idx = vec![0usize; out_ndim];
        for _ in 0..out_size {
            let idx1 = broadcast_index(&out_idx, data.shape(), out_ndim);
            let idx2 = broadcast_index(&out_idx, other_data.shape(), out_ndim);

            let val1 = data[IxDyn(&idx1)];
            let val2 = other_data[IxDyn(&idx2)];
            result_data.push(op(val1, val2));

            // Increment output index
            for d in (0..out_ndim).rev() {
                out_idx[d] += 1;
                if out_idx[d] < out_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        let result = ArrayD::from_shape_vec(IxDyn(&out_shape), result_data)
            .map_err(|e| Error::new(exception::arg_error(), format!("Failed to create result: {}", e)))?;
        return Ok(Obj::wrap(NDArray::new(result)));
    }

    Err(Error::new(exception::type_error(), "Operand must be numeric or NDArray"))
}

fn comparison_op<F>(data: &ArrayD<f64>, other: Value, op: F) -> Result<Obj<NDArray>, Error>
where
    F: Fn(f64, f64) -> f64,
{
    binary_op(data, other, op)
}

// Module-level array manipulation functions

pub fn concatenate(arrays: RArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let axis = axis.unwrap_or(0);
    let mut arr_vec: Vec<ArrayD<f64>> = Vec::new();

    for item in arrays.into_iter() {
        let arr = <&NDArray>::try_convert(item)?;
        arr_vec.push(arr.get_data().clone());
    }

    if arr_vec.is_empty() {
        return Err(Error::new(exception::arg_error(), "Need at least one array"));
    }

    // Use ndarray concatenation
    let axis_usize = if axis < 0 {
        (arr_vec[0].ndim() as i64 + axis) as usize
    } else {
        axis as usize
    };

    let views: Vec<_> = arr_vec.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis_usize), &views)
        .map_err(|e| Error::new(exception::arg_error(), format!("Cannot concatenate: {}", e)))?;

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn vstack(arrays: RArray) -> Result<Obj<NDArray>, Error> {
    concatenate(arrays, Some(0))
}

pub fn hstack(arrays: RArray) -> Result<Obj<NDArray>, Error> {
    // NumPy hstack: concatenate along axis 1 for 2D+, axis 0 for 1D
    // Check first array's dimensions
    let first_val = arrays.entry::<Value>(0)
        .map_err(|_| Error::new(exception::arg_error(), "Need at least one array"))?;
    let first: &NDArray = <&NDArray>::try_convert(first_val)
        .map_err(|_| Error::new(exception::type_error(), "Expected NDArray"))?;

    if first.get_data().ndim() == 1 {
        concatenate(arrays, Some(0))
    } else {
        concatenate(arrays, Some(1))
    }
}

pub fn dstack(arrays: RArray) -> Result<Obj<NDArray>, Error> {
    // NumPy dstack: stack arrays in sequence depth-wise (along third axis)
    // 1D arrays are promoted to shape (1, N, 1)
    // 2D arrays are promoted to shape (M, N, 1)
    let mut arr_vec: Vec<ArrayD<f64>> = Vec::new();

    for item in arrays.into_iter() {
        let arr = <&NDArray>::try_convert(item)?;
        let data = arr.get_data();
        let shape = data.shape().to_vec();

        let promoted = if shape.len() == 1 {
            // 1D -> (1, N, 1) - construct manually to avoid memory layout issues
            let n = shape[0];
            let flat: Vec<f64> = data.iter().cloned().collect();
            ArrayD::from_shape_vec(IxDyn(&[1, n, 1]), flat)
                .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape: {}", e)))?
        } else if shape.len() == 2 {
            // 2D -> (M, N, 1) - construct manually to avoid memory layout issues
            let m = shape[0];
            let n = shape[1];
            let flat: Vec<f64> = data.iter().cloned().collect();
            ArrayD::from_shape_vec(IxDyn(&[m, n, 1]), flat)
                .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape: {}", e)))?
        } else {
            data.clone()
        };
        arr_vec.push(promoted);
    }

    if arr_vec.is_empty() {
        return Err(Error::new(exception::arg_error(), "Need at least one array"));
    }

    let views: Vec<_> = arr_vec.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(2), &views)
        .map_err(|e| Error::new(exception::arg_error(), format!("Cannot dstack: {}", e)))?;

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn stack(arrays: RArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let axis = axis.unwrap_or(0);
    let mut arr_vec: Vec<ArrayD<f64>> = Vec::new();

    for item in arrays.into_iter() {
        let arr = <&NDArray>::try_convert(item)?;
        // Add new axis
        let mut shape = arr.shape();
        let axis_usize = if axis < 0 {
            (shape.len() as i64 + axis + 1) as usize
        } else {
            axis as usize
        };
        shape.insert(axis_usize, 1);
        let expanded = arr.get_data().clone().into_shape(IxDyn(&shape))
            .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape: {}", e)))?;
        arr_vec.push(expanded);
    }

    if arr_vec.is_empty() {
        return Err(Error::new(exception::arg_error(), "Need at least one array"));
    }

    let axis_usize = if axis < 0 {
        (arr_vec[0].ndim() as i64 + axis) as usize
    } else {
        axis as usize
    };

    let views: Vec<_> = arr_vec.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis_usize), &views)
        .map_err(|e| Error::new(exception::arg_error(), format!("Cannot stack: {}", e)))?;

    Ok(Obj::wrap(NDArray::new(result)))
}

/// Split an array into multiple sub-arrays.
///
/// Currently only supports splitting into equal parts (indices as integer count).
/// TODO: Support array of indices for splitting at specific positions.
pub fn split(array: &NDArray, indices: i64) -> Result<RArray, Error> {
    let ruby = Ruby::get().unwrap();
    let data = array.get_data();
    let n = indices as usize;
    let len = data.shape()[0];
    let chunk_size = len / n;

    let result = RArray::new();
    for i in 0..n {
        let start = i * chunk_size;
        let end = if i == n - 1 { len } else { (i + 1) * chunk_size };
        let slice = data.slice_axis(Axis(0), Slice::from(start..end));
        result.push(Obj::wrap(NDArray::new(slice.to_owned())).into_value_with(&ruby))?;
    }
    Ok(result)
}

pub fn vsplit(array: &NDArray, indices: i64) -> Result<RArray, Error> {
    split(array, indices)
}

pub fn hsplit(array: &NDArray, indices: i64) -> Result<RArray, Error> {
    let ruby = Ruby::get().unwrap();
    let data = array.get_data();
    let n = indices as usize;
    let len = data.shape().last().cloned().unwrap_or(1);
    let chunk_size = len / n;

    let result = RArray::new();
    let axis = data.ndim() - 1;
    for i in 0..n {
        let start = i * chunk_size;
        let end = if i == n - 1 { len } else { (i + 1) * chunk_size };
        let slice = data.slice_axis(Axis(axis), Slice::from(start..end));
        result.push(Obj::wrap(NDArray::new(slice.to_owned())).into_value_with(&ruby))?;
    }
    Ok(result)
}

/// Tile an array by repeating it along each axis.
///
/// If the number of repetitions is greater than the array's dimensions,
/// the array is first promoted by prepending new axes of size 1.
pub fn tile(array: &NDArray, reps: RArray) -> Result<Obj<NDArray>, Error> {
    let reps_vec: Vec<usize> = reps
        .into_iter()
        .map(|v| usize::try_convert(v))
        .collect::<Result<Vec<_>, _>>()?;

    let data = array.get_data();
    let mut shape = data.shape().to_vec();
    let ndim = shape.len();
    let reps_ndim = reps_vec.len();

    // If reps has more dimensions than the array, prepend axes of size 1
    if reps_ndim > ndim {
        let extra = reps_ndim - ndim;
        let mut new_shape = vec![1usize; extra];
        new_shape.extend(shape.iter());
        shape = new_shape;
    }

    // Pad reps with 1s at the front if needed
    let mut full_reps = vec![1usize; shape.len()];
    let offset = shape.len().saturating_sub(reps_vec.len());
    for (i, &rep) in reps_vec.iter().enumerate() {
        full_reps[offset + i] = rep;
    }

    // Reshape the data to the new shape (with prepended 1s if needed)
    let flat: Vec<f64> = data.iter().cloned().collect();
    let mut result = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape for tile: {}", e)))?;

    // Now tile along each axis
    for (axis, &rep) in full_reps.iter().enumerate() {
        if rep > 1 {
            let views: Vec<_> = (0..rep).map(|_| result.view()).collect();
            result = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| Error::new(exception::arg_error(), format!("Cannot tile: {}", e)))?;
        }
    }

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn repeat(array: &NDArray, repeats: i64) -> Result<Obj<NDArray>, Error> {
    repeat_axis(array, repeats, None)
}

/// Repeat elements along an axis.
///
/// axis=None: repeat flattened array (1D result)
/// axis=i: repeat along axis i (preserves dimensionality)
///
/// # Behavior Difference from NumPy
/// When axis=None, NumPy returns a flattened 1D array. This implementation
/// matches that behavior. When axis is specified, elements are repeated
/// in-place along that axis.
pub fn repeat_axis(array: &NDArray, repeats: i64, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let repeats = repeats as usize;

    match axis {
        None => {
            // Repeat flattened
            let flat: Vec<f64> = data
                .iter()
                .flat_map(|&x| std::iter::repeat(x).take(repeats))
                .collect();
            Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap())))
        }
        Some(axis_i) => {
            let ndim = data.ndim();
            let axis = if axis_i < 0 {
                (ndim as i64 + axis_i) as usize
            } else {
                axis_i as usize
            };

            if axis >= ndim {
                return Err(Error::new(exception::arg_error(), format!("axis {} is out of bounds", axis_i)));
            }

            let shape = data.shape().to_vec();
            let mut new_shape = shape.clone();
            new_shape[axis] *= repeats;

            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = Vec::with_capacity(flat.len() * repeats);

            for o in 0..outer_size.max(1) {
                for a in 0..axis_len {
                    for _ in 0..repeats {
                        for i in 0..inner_size.max(1) {
                            let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                            result_data.push(flat[flat_idx]);
                        }
                    }
                }
            }

            Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap())))
        }
    }
}

pub fn flip(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    // NumPy flip: reverse along all axes
    let mut result = data.clone();
    for axis in 0..result.ndim() {
        result.invert_axis(Axis(axis));
    }
    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn fliplr(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    if data.ndim() < 2 {
        return Err(Error::new(exception::arg_error(), "Array must be 2D or higher"));
    }
    let mut result = data.clone();
    result.invert_axis(Axis(1));
    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn flipud(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    if data.ndim() < 1 {
        return Err(Error::new(exception::arg_error(), "Array must be 1D or higher"));
    }
    let mut result = data.clone();
    result.invert_axis(Axis(0));
    Ok(Obj::wrap(NDArray::new(result)))
}

/// Roll array elements along an axis.
///
/// Elements that roll beyond the last position are re-introduced at the first.
pub fn roll(array: &NDArray, shift: i64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let n = flat.len();

    // Early return if shift is 0 or array is empty
    if shift == 0 || n == 0 {
        return Ok(Obj::wrap(NDArray::new(data.clone())));
    }

    let shift = ((shift % n as i64) + n as i64) as usize % n;

    // Early return if effective shift is 0
    if shift == 0 {
        return Ok(Obj::wrap(NDArray::new(data.clone())));
    }

    let mut result = vec![0.0; n];
    for i in 0..n {
        result[(i + shift) % n] = flat[i];
    }

    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(data.raw_dim(), result).unwrap())))
}

pub fn rot90(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    rot90_k(array, Some(1))
}

/// Rotate array by 90 degrees k times (counter-clockwise)
pub fn rot90_k(array: &NDArray, k: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    if data.ndim() < 2 {
        return Err(Error::new(exception::arg_error(), "Array must be 2D or higher"));
    }

    // Normalize k to 0-3 range
    let k = k.unwrap_or(1);
    let k = ((k % 4) + 4) % 4;

    let mut result = data.clone();
    for _ in 0..k {
        // Transpose and flip for one 90-degree rotation
        let transposed = result.t().to_owned();
        result = transposed;
        result.invert_axis(Axis(0));
    }

    Ok(Obj::wrap(NDArray::new(result)))
}

/// Helper for NaN-safe comparison (NaN sorts to end, like NumPy)
fn nan_safe_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,  // NaN goes to end
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

pub fn sort(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    sort_axis(array, None)
}

/// Sort with optional axis parameter
/// axis=None (default): sort flattened array
/// axis=i: sort along axis i
pub fn sort_axis(array: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();

    match axis {
        None => {
            // Sort flattened array
            let mut flat: Vec<f64> = data.iter().cloned().collect();
            flat.sort_by(nan_safe_cmp);
            Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap())))
        }
        Some(axis_i) => {
            let ndim = data.ndim();
            let axis = if axis_i < 0 {
                (ndim as i64 + axis_i) as usize
            } else {
                axis_i as usize
            };

            if axis >= ndim {
                return Err(Error::new(exception::arg_error(), format!("axis {} is out of bounds", axis_i)));
            }

            let shape = data.shape().to_vec();
            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = vec![0.0; flat.len()];

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    // Extract slice along axis
                    let mut slice: Vec<f64> = Vec::with_capacity(axis_len);
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        slice.push(flat[flat_idx]);
                    }

                    // Sort the slice
                    slice.sort_by(nan_safe_cmp);

                    // Put back
                    for (a, &val) in slice.iter().enumerate() {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        result_data[flat_idx] = val;
                    }
                }
            }

            Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(data.raw_dim(), result_data).unwrap())))
        }
    }
}

pub fn argsort(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    argsort_axis(array, None)
}

/// Argsort with optional axis parameter
pub fn argsort_axis(array: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();

    match axis {
        None => {
            // Argsort flattened array
            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut indices: Vec<usize> = (0..flat.len()).collect();
            indices.sort_by(|&a, &b| nan_safe_cmp(&flat[a], &flat[b]));
            let result: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
            Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap())))
        }
        Some(axis_i) => {
            let ndim = data.ndim();
            let axis = if axis_i < 0 {
                (ndim as i64 + axis_i) as usize
            } else {
                axis_i as usize
            };

            if axis >= ndim {
                return Err(Error::new(exception::arg_error(), format!("axis {} is out of bounds", axis_i)));
            }

            let shape = data.shape().to_vec();
            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = vec![0.0; flat.len()];

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    // Extract slice along axis
                    let slice: Vec<f64> = (0..axis_len)
                        .map(|a| {
                            let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                            flat[flat_idx]
                        })
                        .collect();

                    // Get argsort indices
                    let mut indices: Vec<usize> = (0..axis_len).collect();
                    indices.sort_by(|&a, &b| nan_safe_cmp(&slice[a], &slice[b]));

                    // Put back
                    for (a, &idx) in indices.iter().enumerate() {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        result_data[flat_idx] = idx as f64;
                    }
                }
            }

            Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(data.raw_dim(), result_data).unwrap())))
        }
    }
}

pub fn searchsorted(array: &NDArray, value: f64) -> Result<i64, Error> {
    searchsorted_side(array, value, None)
}

/// Search for insertion point in a sorted array.
///
/// side="left" (default): a[i-1] < v <= a[i]
/// side="right": a[i-1] <= v < a[i]
///
/// # Important
/// The input array MUST be sorted in ascending order. If the array is not sorted,
/// the results are undefined. Use `RumPy.sort()` first if needed.
///
/// # NaN Handling
/// NaN values in the array are sorted to the end (as if greater than all other values).
/// When searching for NaN, it will be placed at or after all NaN values in the array.
/// This uses the same NaN-safe comparison as `sort()`, where NaN > any finite number.
pub fn searchsorted_side(array: &NDArray, value: f64, side: Option<String>) -> Result<i64, Error> {
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let side = side.unwrap_or_else(|| "left".to_string());

    if side == "right" {
        // Find rightmost position where value could be inserted
        // This is equivalent to finding first element > value
        let mut lo = 0;
        let mut hi = flat.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if nan_safe_cmp(&flat[mid], &value) == std::cmp::Ordering::Greater {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo as i64)
    } else {
        // "left" - find leftmost position (first element >= value)
        match flat.binary_search_by(|x| nan_safe_cmp(x, &value)) {
            Ok(i) => {
                // Found exact match, go to leftmost occurrence
                let mut idx = i;
                while idx > 0 && nan_safe_cmp(&flat[idx - 1], &value) == std::cmp::Ordering::Equal {
                    idx -= 1;
                }
                Ok(idx as i64)
            }
            Err(i) => Ok(i as i64),
        }
    }
}

pub fn unique(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let mut flat: Vec<f64> = data.iter().cloned().collect();
    // NaN-safe sort
    flat.sort_by(nan_safe_cmp);
    // NaN-safe dedup (treat all NaN as equal)
    flat.dedup_by(|a, b| {
        if a.is_nan() && b.is_nan() { true }
        else { a == b }
    });
    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap())))
}

/// Squeeze: remove axes of length 1
pub fn squeeze(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    squeeze_axis(array, None)
}

/// Squeeze with optional axis parameter
/// axis=None: remove all axes of length 1
/// axis=i: remove axis i only (must have length 1)
pub fn squeeze_axis(array: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let shape = data.shape();
    let ndim = shape.len();

    match axis {
        None => {
            let new_shape: Vec<usize> = shape.iter().filter(|&&s| s != 1).cloned().collect();

            if new_shape.is_empty() {
                // Scalar case
                let val = data.iter().next().cloned().unwrap_or(0.0);
                return Ok(Obj::wrap(NDArray::new(
                    ArrayD::from_shape_vec(IxDyn(&[]), vec![val]).unwrap()
                )));
            }

            let flat: Vec<f64> = data.iter().cloned().collect();
            Ok(Obj::wrap(NDArray::new(
                ArrayD::from_shape_vec(IxDyn(&new_shape), flat).unwrap()
            )))
        }
        Some(axis_i) => {
            let axis = if axis_i < 0 {
                (ndim as i64 + axis_i) as usize
            } else {
                axis_i as usize
            };

            if axis >= ndim {
                return Err(Error::new(exception::arg_error(), format!("axis {} is out of bounds", axis_i)));
            }

            if shape[axis] != 1 {
                return Err(Error::new(exception::arg_error(),
                    format!("cannot select an axis to squeeze out which has size not equal to one, got shape[{}] = {}", axis, shape[axis])));
            }

            let new_shape: Vec<usize> = shape.iter().enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect();

            if new_shape.is_empty() {
                let val = data.iter().next().cloned().unwrap_or(0.0);
                return Ok(Obj::wrap(NDArray::new(
                    ArrayD::from_shape_vec(IxDyn(&[]), vec![val]).unwrap()
                )));
            }

            let flat: Vec<f64> = data.iter().cloned().collect();
            Ok(Obj::wrap(NDArray::new(
                ArrayD::from_shape_vec(IxDyn(&new_shape), flat).unwrap()
            )))
        }
    }
}

/// Take elements from array along an axis.
///
/// # Errors
/// Raises an error if any index is out of bounds. Use mode='clip' or mode='wrap'
/// in NumPy for alternative behaviors (not yet implemented here).
pub fn take(array: &NDArray, indices: &NDArray) -> Result<Obj<NDArray>, Error> {
    take_axis(array, indices, None)
}

/// Take elements from array along a specified axis.
///
/// # Errors
/// Raises an IndexError if any index is out of bounds. This matches NumPy's
/// default mode='raise' behavior.
pub fn take_axis(array: &NDArray, indices: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let idx_data = indices.get_data();
    let idx_vec: Vec<usize> = idx_data.iter().map(|&x| x as usize).collect();

    match axis {
        None => {
            // Take from flattened array
            let flat: Vec<f64> = data.iter().cloned().collect();

            // Validate all indices are in bounds
            for (i, &idx) in idx_vec.iter().enumerate() {
                if idx >= flat.len() {
                    return Err(Error::new(
                        exception::index_error(),
                        format!("index {} is out of bounds for axis with size {}", idx, flat.len())
                    ));
                }
            }

            let result: Vec<f64> = idx_vec.iter().map(|&i| flat[i]).collect();
            Ok(Obj::wrap(NDArray::new(
                ArrayD::from_shape_vec(idx_data.raw_dim(), result).unwrap()
            )))
        }
        Some(axis_i) => {
            let ndim = data.ndim();
            let axis = if axis_i < 0 {
                (ndim as i64 + axis_i) as usize
            } else {
                axis_i as usize
            };

            if axis >= ndim {
                return Err(Error::new(exception::arg_error(), format!("axis {} is out of bounds", axis_i)));
            }

            let shape = data.shape().to_vec();
            let axis_len = shape[axis];

            // Validate all indices are in bounds for this axis
            for &idx in &idx_vec {
                if idx >= axis_len {
                    return Err(Error::new(
                        exception::index_error(),
                        format!("index {} is out of bounds for axis {} with size {}", idx, axis, axis_len)
                    ));
                }
            }

            let num_indices = idx_vec.len();

            let mut new_shape = shape.clone();
            new_shape[axis] = num_indices;

            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = Vec::with_capacity(outer_size.max(1) * num_indices * inner_size.max(1));

            for o in 0..outer_size.max(1) {
                for &idx in &idx_vec {
                    for i in 0..inner_size.max(1) {
                        let flat_idx = o * axis_len * inner_size + idx * inner_size + i;
                        result_data.push(flat[flat_idx]);
                    }
                }
            }

            Ok(Obj::wrap(NDArray::new(
                ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap()
            )))
        }
    }
}

/// Put values into array at specified indices
pub fn put(array: &NDArray, indices: &NDArray, values: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let idx_data = indices.get_data();
    let val_data = values.get_data();

    let mut flat: Vec<f64> = data.iter().cloned().collect();
    let vals: Vec<f64> = val_data.iter().cloned().collect();

    for (i, &idx) in idx_data.iter().enumerate() {
        let idx = idx as usize;
        if idx < flat.len() {
            flat[idx] = vals[i % vals.len()];
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), flat).unwrap()
    )))
}

/// Pad array with constant values.
///
/// Currently only supports 'constant' padding mode.
/// TODO: edge, wrap, and reflect modes are not yet implemented.
pub fn pad(array: &NDArray, pad_width: usize, constant_value: f64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let shape = data.shape().to_vec();
    let ndim = shape.len();

    // Compute new shape
    let new_shape: Vec<usize> = shape.iter().map(|&s| s + 2 * pad_width).collect();
    let new_size: usize = new_shape.iter().product();

    // Initialize result with constant value
    let mut result = vec![constant_value; new_size];

    // Helper to convert multi-dim index to flat index
    let flat_index = |idx: &[usize], shape: &[usize]| -> usize {
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            flat += idx[i] * stride;
            stride *= shape[i];
        }
        flat
    };

    // Iterate over original array and place values in padded array
    let old_flat: Vec<f64> = data.iter().cloned().collect();
    let old_size = old_flat.len();

    // Convert flat index to multi-dimensional index
    for old_flat_idx in 0..old_size {
        // Convert flat index to old multi-dim index
        let mut remaining = old_flat_idx;
        let mut old_idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            old_idx[d] = remaining % shape[d];
            remaining /= shape[d];
        }

        // Compute new index (offset by pad_width in each dimension)
        let new_idx: Vec<usize> = old_idx.iter().map(|&i| i + pad_width).collect();

        // Get new flat index
        let new_flat_idx = flat_index(&new_idx, &new_shape);
        result[new_flat_idx] = old_flat[old_flat_idx];
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap()
    )))
}

/// Expand the shape of an array by inserting a new axis at the specified position.
///
/// # Arguments
/// * `axis` - Position where the new axis should be inserted. Can be 0 to ndim
///            (inclusive). Negative values count from the end: -1 inserts before
///            the last axis, etc.
///
/// # Errors
/// Returns an error if axis is out of the valid range [-ndim-1, ndim].
pub fn expand_dims(array: &NDArray, axis: i64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let shape = data.shape();
    let ndim = shape.len();

    // Normalize axis (can be -ndim-1 to ndim inclusive for insertion)
    // After insertion, the new axis will be at position `axis`
    let axis_normalized = if axis < 0 {
        let adjusted = ndim as i64 + 1 + axis;
        if adjusted < 0 {
            return Err(Error::new(exception::arg_error(),
                format!("axis {} is out of bounds for array of dimension {} (valid range: {} to {})",
                    axis, ndim, -(ndim as i64 + 1), ndim)));
        }
        adjusted as usize
    } else {
        axis as usize
    };

    if axis_normalized > ndim {
        return Err(Error::new(exception::arg_error(),
            format!("axis {} is out of bounds for array of dimension {} (valid range: {} to {})",
                axis, ndim, -(ndim as i64 + 1), ndim)));
    }

    let axis = axis_normalized;

    let mut new_shape = shape.to_vec();
    new_shape.insert(axis, 1);

    let flat: Vec<f64> = data.iter().cloned().collect();
    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&new_shape), flat).unwrap()
    )))
}

/// Interchange two axes of an array.
///
/// # Arguments
/// * `axis1` - First axis. Negative values count from the end.
/// * `axis2` - Second axis. Negative values count from the end.
///
/// # Errors
/// Returns an error if either axis is out of bounds.
pub fn swapaxes(array: &NDArray, axis1: i64, axis2: i64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let shape = data.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::new(exception::arg_error(), "swapaxes requires array with at least 1 dimension"));
    }

    // Normalize axes
    let axis1_norm = if axis1 < 0 { (ndim as i64 + axis1) as usize } else { axis1 as usize };
    let axis2_norm = if axis2 < 0 { (ndim as i64 + axis2) as usize } else { axis2 as usize };

    if axis1_norm >= ndim {
        return Err(Error::new(exception::arg_error(),
            format!("axis1 {} is out of bounds for array of dimension {}", axis1, ndim)));
    }
    if axis2_norm >= ndim {
        return Err(Error::new(exception::arg_error(),
            format!("axis2 {} is out of bounds for array of dimension {}", axis2, ndim)));
    }

    // Reassign to use normalized values
    let axis1 = axis1_norm;
    let axis2 = axis2_norm;

    if axis1 == axis2 {
        return Ok(Obj::wrap(NDArray::new(data.clone())));
    }

    // Create permutation array
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.swap(axis1, axis2);

    // Compute new shape
    let mut new_shape = shape.to_vec();
    new_shape.swap(axis1, axis2);

    // Reorder data
    let flat: Vec<f64> = data.iter().cloned().collect();
    let total_size = flat.len();
    let mut result = vec![0.0; total_size];

    // For each output position, compute the input position
    for out_flat_idx in 0..total_size {
        // Convert flat index to multi-dim index in output shape
        let mut remaining = out_flat_idx;
        let mut out_idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            out_idx[d] = remaining % new_shape[d];
            remaining /= new_shape[d];
        }

        // Apply inverse permutation to get input index
        let mut in_idx = vec![0usize; ndim];
        for d in 0..ndim {
            in_idx[perm[d]] = out_idx[d];
        }

        // Convert input index to flat
        let mut in_flat_idx = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            in_flat_idx += in_idx[d] * stride;
            stride *= shape[d];
        }

        result[out_flat_idx] = flat[in_flat_idx];
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap()
    )))
}

/// Move axes of an array to new positions.
///
/// # Arguments
/// * `source` - Original position of the axis to move. Negative values count from the end.
/// * `destination` - Destination position for the axis. Negative values count from the end.
///
/// # Errors
/// Returns an error if source or destination axis is out of bounds.
pub fn moveaxis(array: &NDArray, source: i64, destination: i64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let shape = data.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::new(exception::arg_error(), "moveaxis requires array with at least 1 dimension"));
    }

    // Normalize axes
    let src = if source < 0 { (ndim as i64 + source) as usize } else { source as usize };
    let dst = if destination < 0 { (ndim as i64 + destination) as usize } else { destination as usize };

    if src >= ndim {
        return Err(Error::new(exception::arg_error(),
            format!("source axis {} is out of bounds for array of dimension {}", source, ndim)));
    }
    if dst >= ndim {
        return Err(Error::new(exception::arg_error(),
            format!("destination axis {} is out of bounds for array of dimension {}", destination, ndim)));
    }

    if src == dst {
        return Ok(Obj::wrap(NDArray::new(data.clone())));
    }

    // Build permutation: remove src from its position and insert at dst
    let mut perm: Vec<usize> = (0..ndim).collect();
    let removed = perm.remove(src);
    perm.insert(dst, removed);

    // Compute new shape
    let new_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();

    // Compute inverse permutation for data reordering
    let mut inv_perm = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    // Reorder data
    let flat: Vec<f64> = data.iter().cloned().collect();
    let total_size = flat.len();
    let mut result = vec![0.0; total_size];

    for out_flat_idx in 0..total_size {
        // Convert flat index to multi-dim index in output shape
        let mut remaining = out_flat_idx;
        let mut out_idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            out_idx[d] = remaining % new_shape[d];
            remaining /= new_shape[d];
        }

        // Apply inverse permutation to get input index
        let mut in_idx = vec![0usize; ndim];
        for d in 0..ndim {
            in_idx[perm[d]] = out_idx[d];
        }

        // Convert input index to flat
        let mut in_flat_idx = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            in_flat_idx += in_idx[d] * stride;
            stride *= shape[d];
        }

        result[out_flat_idx] = flat[in_flat_idx];
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap()
    )))
}
