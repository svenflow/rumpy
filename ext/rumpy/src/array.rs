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

    /// Get element at index
    pub fn get(&self, index: Value) -> Result<Value, Error> {
        let ruby = Ruby::get().unwrap();
        let data = self.data.borrow();

        // Handle integer index
        if let Ok(i) = i64::try_convert(index) {
            let idx = normalize_index(i, data.len())?;
            if data.ndim() == 1 {
                return Ok(data[[idx]].into_value_with(&ruby));
            }
            // For multi-dim, return a slice
            let slice = data.index_axis(Axis(0), idx);
            return Ok(Obj::wrap(NDArray::new(slice.to_owned())).into_value_with(&ruby));
        }

        // Handle array of indices
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
            return Ok(data[&idx].into_value_with(&ruby));
        }

        Err(Error::new(exception::type_error(), "Invalid index type"))
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

    pub fn min(&self) -> f64 {
        self.data.borrow().iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.data.borrow().iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn argmin(&self) -> usize {
        let data = self.data.borrow();
        data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn argmax(&self) -> usize {
        let data = self.data.borrow();
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
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

        // Broadcasting (simplified - only handles common cases)
        // TODO: Full numpy-style broadcasting
        return Err(Error::new(exception::arg_error(), "Shape mismatch (broadcasting not fully implemented)"));
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
    concatenate(arrays, Some(-1))
}

pub fn dstack(arrays: RArray) -> Result<Obj<NDArray>, Error> {
    concatenate(arrays, Some(2))
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
    let data = array.get_data();
    let flat: Vec<f64> = data
        .iter()
        .flat_map(|&x| std::iter::repeat(x).take(repeats as usize))
        .collect();
    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap())))
}

pub fn flip(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().rev().collect();
    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(data.raw_dim(), flat).unwrap())))
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
    let data = array.get_data();
    if data.ndim() < 2 {
        return Err(Error::new(exception::arg_error(), "Array must be 2D or higher"));
    }

    // Transpose and flip
    let transposed = data.t().to_owned();
    let mut result = transposed;
    result.invert_axis(Axis(0));

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn sort(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let mut flat: Vec<f64> = data.iter().cloned().collect();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(data.raw_dim(), flat).unwrap())))
}

pub fn argsort(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let mut indices: Vec<usize> = (0..flat.len()).collect();
    indices.sort_by(|&a, &b| flat[a].partial_cmp(&flat[b]).unwrap());
    let result: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(data.raw_dim(), result).unwrap())))
}

pub fn searchsorted(array: &NDArray, value: f64) -> Result<i64, Error> {
    let data = array.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();

    match flat.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
        Ok(i) => Ok(i as i64),
        Err(i) => Ok(i as i64),
    }
}

pub fn unique(array: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = array.get_data();
    let mut flat: Vec<f64> = data.iter().cloned().collect();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    flat.dedup();
    Ok(Obj::wrap(NDArray::new(ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap())))
}
