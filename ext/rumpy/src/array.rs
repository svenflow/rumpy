//! Core NDArray type and operations

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

    /// Convert to different dtype (placeholder - only float64 for now)
    pub fn astype(&self, _dtype: String) -> Obj<NDArray> {
        self.copy()
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
        self.data.borrow().iter().all(|&x| x != 0.0)
    }

    pub fn any(&self) -> bool {
        self.data.borrow().iter().any(|&x| x != 0.0)
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
        let data = arr.get_data().clone();
        let shape = data.shape().to_vec();

        let promoted = if shape.len() == 1 {
            // 1D -> (1, N, 1)
            let n = shape[0];
            data.into_shape(IxDyn(&[1, n, 1]))
                .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape: {}", e)))?
        } else if shape.len() == 2 {
            // 2D -> (M, N, 1)
            let m = shape[0];
            let n = shape[1];
            data.into_shape(IxDyn(&[m, n, 1]))
                .map_err(|e| Error::new(exception::arg_error(), format!("Cannot reshape: {}", e)))?
        } else {
            data
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

pub fn tile(array: &NDArray, reps: RArray) -> Result<Obj<NDArray>, Error> {
    let reps_vec: Vec<usize> = reps
        .into_iter()
        .map(|v| usize::try_convert(v))
        .collect::<Result<Vec<_>, _>>()?;

    let data = array.get_data();
    let mut result = data.clone();

    for (axis, &rep) in reps_vec.iter().enumerate() {
        if axis < result.ndim() && rep > 1 {
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

/// Repeat elements along an axis
/// axis=None: repeat flattened array
/// axis=i: repeat along axis i
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

pub fn roll(array: &NDArray, shift: i64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let n = flat.len();
    let shift = ((shift % n as i64) + n as i64) as usize % n;

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
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();

    // NaN-safe binary search
    match flat.binary_search_by(|x| nan_safe_cmp(x, &value)) {
        Ok(i) => Ok(i as i64),
        Err(i) => Ok(i as i64),
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
    let data = array.get_data();
    let new_shape: Vec<usize> = data.shape().iter().filter(|&&s| s != 1).cloned().collect();

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

/// Take elements from array along an axis
pub fn take(array: &NDArray, indices: &NDArray) -> Result<Obj<NDArray>, Error> {
    take_axis(array, indices, None)
}

/// Take elements from array along a specified axis
pub fn take_axis(array: &NDArray, indices: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let idx_data = indices.get_data();
    let idx_vec: Vec<usize> = idx_data.iter().map(|&x| x as usize).collect();

    match axis {
        None => {
            // Take from flattened array
            let flat: Vec<f64> = data.iter().cloned().collect();
            let result: Vec<f64> = idx_vec.iter().map(|&i| {
                if i < flat.len() { flat[i] } else { f64::NAN }
            }).collect();
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
            let num_indices = idx_vec.len();

            let mut new_shape = shape.clone();
            new_shape[axis] = num_indices;

            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();
            let axis_len = shape[axis];

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = Vec::with_capacity(outer_size.max(1) * num_indices * inner_size.max(1));

            for o in 0..outer_size.max(1) {
                for &idx in &idx_vec {
                    for i in 0..inner_size.max(1) {
                        let flat_idx = o * axis_len * inner_size + idx * inner_size + i;
                        result_data.push(if flat_idx < flat.len() { flat[flat_idx] } else { f64::NAN });
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

/// Pad array with constant values
pub fn pad(array: &NDArray, pad_width: usize, constant_value: f64) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let shape = data.shape();

    // Simple case: 1D padding
    if data.ndim() == 1 {
        let n = shape[0];
        let new_len = n + 2 * pad_width;
        let mut result = vec![constant_value; new_len];
        for (i, &val) in data.iter().enumerate() {
            result[pad_width + i] = val;
        }
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[new_len]), result).unwrap()
        )));
    }

    // 2D padding
    if data.ndim() == 2 {
        let (h, w) = (shape[0], shape[1]);
        let new_h = h + 2 * pad_width;
        let new_w = w + 2 * pad_width;
        let mut result = vec![constant_value; new_h * new_w];

        for i in 0..h {
            for j in 0..w {
                result[(pad_width + i) * new_w + (pad_width + j)] = data[[i, j]];
            }
        }

        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[new_h, new_w]), result).unwrap()
        )));
    }

    Err(Error::new(exception::arg_error(), "pad only supports 1D and 2D arrays"))
}
