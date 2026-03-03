//! Statistical functions for numerical arrays.
//!
//! # NaN Handling
//!
//! Different functions handle NaN values differently:
//!
//! ## NaN-propagating functions
//! - `mean`, `std`, `var`: Return NaN if any element is NaN
//! - `min`, `max`: Return NaN if any element is NaN
//! - `median`, `percentile`: NaN values sort to the end, may affect result
//!
//! ## NaN-ignoring functions (use these for data with missing values)
//! - `nansum`, `nanmean`, `nanstd`, `nanvar`: Ignore NaN values
//! - `nanmin`, `nanmax`: Ignore NaN values
//! - `nanmedian`: Filter out NaN before computing median
//! - `nanargmin`, `nanargmax`: Return index of min/max among non-NaN values
//!
//! ## Special cases
//! - `quantile`: Filters out NaN values before computing
//! - `corrcoef`, `cov`: NaN in input may produce NaN in output

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, IntoValue, RArray, Value, TryConvert};
use ndarray::{ArrayD, Axis, IxDyn};

// Helper to normalize axis (handle negative indices)
fn normalize_axis(axis: i64, ndim: usize) -> Result<usize, Error> {
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    if axis >= ndim {
        return Err(Error::new(exception::arg_error(), format!("axis {} is out of bounds for array of dimension {}", axis, ndim)));
    }
    Ok(axis)
}

pub fn sum(arr: &NDArray) -> f64 {
    arr.get_data().sum()
}

/// Sum with optional axis parameter
pub fn sum_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(data.sum().into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;
            let result = data.sum_axis(Axis(axis));
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn prod(arr: &NDArray) -> f64 {
    arr.get_data().product()
}

/// Prod with optional axis parameter
pub fn prod_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(data.product().into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;
            let shape = data.shape();
            let mut new_shape: Vec<usize> = shape.iter().enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect();
            if new_shape.is_empty() {
                new_shape = vec![];
            }

            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let mut result_data = Vec::with_capacity(outer_size * inner_size);
            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut prod = 1.0;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        prod *= data.iter().nth(flat_idx).cloned().unwrap_or(1.0);
                    }
                    result_data.push(prod);
                }
            }

            let result = ArrayD::from_shape_vec(IxDyn(&new_shape), result_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn mean(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    if data.is_empty() {
        return f64::NAN;
    }
    data.sum() / data.len() as f64
}

/// Mean with optional axis parameter
pub fn mean_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(mean(arr).into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;
            let result = data.mean_axis(Axis(axis));
            match result {
                Some(r) => Ok(Obj::wrap(NDArray::new(r)).into_value_with(&ruby)),
                None => Err(Error::new(exception::arg_error(), "Cannot compute mean of empty slice"))
            }
        }
    }
}

pub fn std(arr: &NDArray) -> f64 {
    var(arr).sqrt()
}

/// Standard deviation with ddof (delta degrees of freedom) parameter
pub fn std_ddof(arr: &NDArray, ddof: Option<i64>) -> f64 {
    var_ddof(arr, ddof).sqrt()
}

/// Std with optional axis parameter
pub fn std_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(std(arr).into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;
            let var_result = data.var_axis(Axis(axis), 0.0);
            let result = var_result.mapv(|x| x.sqrt());
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn var(arr: &NDArray) -> f64 {
    var_ddof(arr, Some(0))
}

/// Variance with ddof (delta degrees of freedom) parameter
/// ddof=0 for population variance, ddof=1 for sample variance
pub fn var_ddof(arr: &NDArray, ddof: Option<i64>) -> f64 {
    let data = arr.get_data();
    let n = data.len();
    let ddof = ddof.unwrap_or(0) as usize;
    if n <= ddof {
        return f64::NAN;
    }
    let m = data.sum() / n as f64;
    data.mapv(|x| (x - m).powi(2)).sum() / (n - ddof) as f64
}

/// Var with optional axis parameter
pub fn var_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(var(arr).into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;
            let result = data.var_axis(Axis(axis), 0.0);
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn min(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();
    if data.is_empty() {
        return Err(Error::new(exception::arg_error(), "zero-size array to reduction operation minimum which has no identity"));
    }
    Ok(data.iter().cloned().fold(f64::INFINITY, f64::min))
}

/// Min with optional axis parameter
pub fn min_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(min(arr)?.into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;

            let shape = data.shape();
            let mut new_shape: Vec<usize> = shape.iter().enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect();
            if new_shape.is_empty() {
                new_shape = vec![];
            }

            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let mut result_data = Vec::with_capacity(outer_size * inner_size);
            let flat: Vec<f64> = data.iter().cloned().collect();

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut min_val = f64::INFINITY;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        if flat_idx < flat.len() {
                            min_val = min_val.min(flat[flat_idx]);
                        }
                    }
                    result_data.push(min_val);
                }
            }

            let result = ArrayD::from_shape_vec(IxDyn(&new_shape), result_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn max(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();
    if data.is_empty() {
        return Err(Error::new(exception::arg_error(), "zero-size array to reduction operation maximum which has no identity"));
    }
    Ok(data.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}

/// Max with optional axis parameter
pub fn max_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok(max(arr)?.into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;

            let shape = data.shape();
            let mut new_shape: Vec<usize> = shape.iter().enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect();
            if new_shape.is_empty() {
                new_shape = vec![];
            }

            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let mut result_data = Vec::with_capacity(outer_size * inner_size);
            let flat: Vec<f64> = data.iter().cloned().collect();

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut max_val = f64::NEG_INFINITY;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        if flat_idx < flat.len() {
                            max_val = max_val.max(flat[flat_idx]);
                        }
                    }
                    result_data.push(max_val);
                }
            }

            let result = ArrayD::from_shape_vec(IxDyn(&new_shape), result_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn argmin(arr: &NDArray) -> Result<usize, Error> {
    let data = arr.get_data();
    if data.is_empty() {
        return Err(Error::new(exception::arg_error(), "attempt to get argmin of an empty sequence"));
    }
    Ok(data.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap())
}

/// Argmin with optional axis parameter (returns integer indices)
pub fn argmin_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok((argmin(arr)? as i64).into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;

            let shape = data.shape();
            let mut new_shape: Vec<usize> = shape.iter().enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect();
            if new_shape.is_empty() {
                new_shape = vec![];
            }

            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let mut result_data: Vec<i64> = Vec::with_capacity(outer_size * inner_size);
            let flat: Vec<f64> = data.iter().cloned().collect();

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut min_val = f64::INFINITY;
                    let mut min_idx = 0i64;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        if flat_idx < flat.len() && flat[flat_idx] < min_val {
                            min_val = flat[flat_idx];
                            min_idx = a as i64;
                        }
                    }
                    result_data.push(min_idx);
                }
            }

            // Return NDArray with integer values stored as f64 for consistency
            let float_data: Vec<f64> = result_data.iter().map(|&x| x as f64).collect();
            let result = ArrayD::from_shape_vec(IxDyn(&new_shape), float_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn argmax(arr: &NDArray) -> Result<usize, Error> {
    let data = arr.get_data();
    if data.is_empty() {
        return Err(Error::new(exception::arg_error(), "attempt to get argmax of an empty sequence"));
    }
    Ok(data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap())
}

/// Argmax with optional axis parameter (returns integer indices)
pub fn argmax_axis(arr: &NDArray, axis: Option<i64>) -> Result<Value, Error> {
    let ruby = magnus::Ruby::get().unwrap();
    let data = arr.get_data();

    match axis {
        None => Ok((argmax(arr)? as i64).into_value_with(&ruby)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;

            let shape = data.shape();
            let mut new_shape: Vec<usize> = shape.iter().enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect();
            if new_shape.is_empty() {
                new_shape = vec![];
            }

            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let mut result_data = Vec::with_capacity(outer_size * inner_size);
            let flat: Vec<f64> = data.iter().cloned().collect();

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        if flat_idx < flat.len() && flat[flat_idx] > max_val {
                            max_val = flat[flat_idx];
                            max_idx = a;
                        }
                    }
                    result_data.push(max_idx as f64);
                }
            }

            let result = ArrayD::from_shape_vec(IxDyn(&new_shape), result_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)).into_value_with(&ruby))
        }
    }
}

pub fn cumsum(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    let mut sum = 0.0;
    let result: Vec<f64> = data
        .iter()
        .map(|&x| {
            sum += x;
            sum
        })
        .collect();
    Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    ))
}

/// Cumsum with optional axis parameter
pub fn cumsum_axis(arr: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    match axis {
        None => Ok(cumsum(arr)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;

            let shape = data.shape();
            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = vec![0.0; flat.len()];

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut sum = 0.0;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        if flat_idx < flat.len() {
                            sum += flat[flat_idx];
                            result_data[flat_idx] = sum;
                        }
                    }
                }
            }

            let result = ArrayD::from_shape_vec(data.raw_dim(), result_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)))
        }
    }
}

pub fn cumprod(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    let mut prod = 1.0;
    let result: Vec<f64> = data
        .iter()
        .map(|&x| {
            prod *= x;
            prod
        })
        .collect();
    Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    ))
}

/// Cumprod with optional axis parameter
pub fn cumprod_axis(arr: &NDArray, axis: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    match axis {
        None => Ok(cumprod(arr)),
        Some(axis_i) => {
            let axis = normalize_axis(axis_i, data.ndim())?;

            let shape = data.shape();
            let axis_len = shape[axis];
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis+1..].iter().product();

            let flat: Vec<f64> = data.iter().cloned().collect();
            let mut result_data = vec![1.0; flat.len()];

            for o in 0..outer_size.max(1) {
                for i in 0..inner_size.max(1) {
                    let mut prod = 1.0;
                    for a in 0..axis_len {
                        let flat_idx = o * axis_len * inner_size + a * inner_size + i;
                        if flat_idx < flat.len() {
                            prod *= flat[flat_idx];
                            result_data[flat_idx] = prod;
                        }
                    }
                }
            }

            let result = ArrayD::from_shape_vec(data.raw_dim(), result_data)
                .map_err(|e| Error::new(exception::arg_error(), format!("{}", e)))?;
            Ok(Obj::wrap(NDArray::new(result)))
        }
    }
}

/// Helper for NaN-safe comparison (NaN sorts to end)
fn nan_safe_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

pub fn median(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let mut sorted: Vec<f64> = data.iter().cloned().collect();
    sorted.sort_by(nan_safe_cmp);

    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 0 {
        let a = sorted[n / 2 - 1];
        let b = sorted[n / 2];
        if a.is_nan() || b.is_nan() { f64::NAN } else { (a + b) / 2.0 }
    } else {
        sorted[n / 2]
    }
}

pub fn percentile(arr: &NDArray, q: f64) -> f64 {
    quantile(arr, q / 100.0)
}

/// Compute the q-th quantile of the array.
///
/// # NaN Handling
/// NaN values are filtered out before computing the quantile. If all values
/// are NaN, returns NaN. This matches NumPy's nanquantile behavior.
pub fn quantile(arr: &NDArray, q: f64) -> f64 {
    let data = arr.get_data();
    // Filter out NaN values before computing quantile
    let mut sorted: Vec<f64> = data.iter().cloned().filter(|x| !x.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return f64::NAN;
    }

    let n = sorted.len();
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;

    if lo >= n {
        sorted[n - 1]
    } else if hi >= n {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

pub fn histogram(arr: &NDArray, bins: usize) -> Result<RArray, Error> {
    let data = arr.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();

    if flat.is_empty() {
        return Err(Error::new(exception::arg_error(), "Cannot compute histogram of empty array"));
    }

    let min_val = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    // Handle single value / uniform data case
    let (bin_width, adjusted_min, adjusted_max) = if range == 0.0 || range.abs() < 1e-14 {
        // All values are the same - create bins centered around the value
        // NumPy uses a small range around the value
        let center = min_val;
        let half_width = if center == 0.0 { 0.5 } else { center.abs() * 0.5 };
        (half_width * 2.0 / bins as f64, center - half_width, center + half_width)
    } else {
        (range / bins as f64, min_val, max_val)
    };

    let mut counts = vec![0.0; bins];
    let mut edges = Vec::with_capacity(bins + 1);

    for i in 0..=bins {
        edges.push(adjusted_min + i as f64 * bin_width);
    }

    for &x in &flat {
        if range == 0.0 || range.abs() < 1e-14 {
            // All values go in the middle bin
            counts[bins / 2] += 1.0;
        } else {
            let bin = ((x - adjusted_min) / bin_width).floor() as usize;
            let bin = bin.min(bins - 1);
            counts[bin] += 1.0;
        }
    }

    let _ruby = magnus::Ruby::get().unwrap();
    let result = RArray::new();

    let counts_arr = NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[counts.len()]), counts).unwrap(),
    );
    let edges_arr = NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[edges.len()]), edges).unwrap(),
    );

    result.push(Obj::wrap(counts_arr))?;
    result.push(Obj::wrap(edges_arr))?;

    Ok(result)
}

/// Compute correlation coefficient matrix.
///
/// # Edge Cases
/// - For single observation (n_obs=1), correlation is undefined and returns NaN
/// - For zero variance variables, correlation is undefined and returns NaN/1.0
/// - Diagonal elements are always 1.0 (self-correlation)
pub fn corrcoef(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "corrcoef requires 2D array"));
    }

    let shape = data.shape();
    let n_vars = shape[0];
    let n_obs = shape[1];

    // Edge case: single observation means correlation is undefined
    if n_obs < 2 {
        // Return matrix of NaN except diagonal which is 1.0
        let mut corr = vec![f64::NAN; n_vars * n_vars];
        for i in 0..n_vars {
            corr[i * n_vars + i] = 1.0;
        }
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[n_vars, n_vars]), corr).unwrap(),
        )));
    }

    let means: Vec<f64> = (0..n_vars)
        .map(|i| {
            (0..n_obs).map(|j| data[[i, j]]).sum::<f64>() / n_obs as f64
        })
        .collect();

    // Use n-1 divisor (Bessel's correction) for consistency with cov() and NumPy
    let divisor = (n_obs - 1).max(1) as f64;
    let mut cov = vec![0.0; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut sum = 0.0;
            for k in 0..n_obs {
                sum += (data[[i, k]] - means[i]) * (data[[j, k]] - means[j]);
            }
            cov[i * n_vars + j] = sum / divisor;
        }
    }

    let mut corr = vec![0.0; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let denom = (cov[i * n_vars + i] * cov[j * n_vars + j]).sqrt();
            corr[i * n_vars + j] = if denom > 0.0 {
                cov[i * n_vars + j] / denom
            } else {
                if i == j { 1.0 } else { 0.0 }
            };
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n_vars, n_vars]), corr).unwrap(),
    )))
}

pub fn cov(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "cov requires 2D array"));
    }

    let shape = data.shape();
    let n_vars = shape[0];
    let n_obs = shape[1];

    let means: Vec<f64> = (0..n_vars)
        .map(|i| {
            (0..n_obs).map(|j| data[[i, j]]).sum::<f64>() / n_obs as f64
        })
        .collect();

    let mut cov = vec![0.0; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut sum = 0.0;
            for k in 0..n_obs {
                sum += (data[[i, k]] - means[i]) * (data[[j, k]] - means[j]);
            }
            cov[i * n_vars + j] = sum / (n_obs - 1).max(1) as f64;
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n_vars, n_vars]), cov).unwrap(),
    )))
}

// NaN-ignoring aggregation functions

pub fn nansum(arr: &NDArray) -> f64 {
    arr.get_data().iter().filter(|x| !x.is_nan()).sum()
}

pub fn nanmean(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let non_nan: Vec<f64> = data.iter().filter(|x| !x.is_nan()).cloned().collect();
    if non_nan.is_empty() {
        f64::NAN
    } else {
        non_nan.iter().sum::<f64>() / non_nan.len() as f64
    }
}

pub fn nanvar(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let non_nan: Vec<f64> = data.iter().filter(|x| !x.is_nan()).cloned().collect();
    if non_nan.is_empty() {
        return f64::NAN;
    }
    let m = non_nan.iter().sum::<f64>() / non_nan.len() as f64;
    non_nan.iter().map(|x| (x - m).powi(2)).sum::<f64>() / non_nan.len() as f64
}

pub fn nanstd(arr: &NDArray) -> f64 {
    nanvar(arr).sqrt()
}

pub fn nanmin(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let non_nan: Vec<f64> = data.iter().filter(|x| !x.is_nan()).cloned().collect();
    if non_nan.is_empty() {
        // NumPy returns NaN for all-NaN arrays with a warning
        return f64::NAN;
    }
    non_nan.iter().cloned().fold(f64::INFINITY, f64::min)
}

pub fn nanmax(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let non_nan: Vec<f64> = data.iter().filter(|x| !x.is_nan()).cloned().collect();
    if non_nan.is_empty() {
        // NumPy returns NaN for all-NaN arrays with a warning
        return f64::NAN;
    }
    non_nan.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

pub fn nanargmin(arr: &NDArray) -> Result<usize, Error> {
    let data = arr.get_data();
    let result = data.iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    match result {
        Some((i, _)) => Ok(i),
        None => Err(Error::new(exception::arg_error(), "All-NaN slice encountered")),
    }
}

pub fn nanargmax(arr: &NDArray) -> Result<usize, Error> {
    let data = arr.get_data();
    let result = data.iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    match result {
        Some((i, _)) => Ok(i),
        None => Err(Error::new(exception::arg_error(), "All-NaN slice encountered")),
    }
}

pub fn nanprod(arr: &NDArray) -> f64 {
    arr.get_data().iter().filter(|x| !x.is_nan()).product()
}

pub fn nancumsum(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    let mut sum = 0.0;
    let result: Vec<f64> = data
        .iter()
        .map(|&x| {
            if !x.is_nan() {
                sum += x;
            }
            sum
        })
        .collect();
    Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    ))
}

pub fn nancumprod(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    let mut prod = 1.0;
    let result: Vec<f64> = data
        .iter()
        .map(|&x| {
            if !x.is_nan() {
                prod *= x;
            }
            prod
        })
        .collect();
    Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    ))
}

pub fn nanmedian(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let mut sorted: Vec<f64> = data.iter().filter(|x| !x.is_nan()).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

pub fn trapz(y: &NDArray, dx: f64) -> f64 {
    let data = y.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let n = flat.len();

    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..n-1 {
        sum += (flat[i] + flat[i+1]) / 2.0 * dx;
    }
    sum
}

/// Trapezoidal integration with x array (non-uniform spacing)
pub fn trapz_x(y: &NDArray, x: &NDArray) -> Result<f64, Error> {
    let y_data = y.get_data();
    let x_data = x.get_data();

    let y_flat: Vec<f64> = y_data.iter().cloned().collect();
    let x_flat: Vec<f64> = x_data.iter().cloned().collect();

    if y_flat.len() != x_flat.len() {
        return Err(Error::new(exception::arg_error(), "x and y must have the same length"));
    }

    let n = y_flat.len();
    if n < 2 {
        return Ok(0.0);
    }

    let mut sum = 0.0;
    for i in 0..n-1 {
        let dx = x_flat[i+1] - x_flat[i];
        sum += (y_flat[i] + y_flat[i+1]) / 2.0 * dx;
    }
    Ok(sum)
}

pub fn polyfit(x: &NDArray, y: &NDArray, deg: usize) -> Result<Obj<NDArray>, Error> {
    let x_data = x.get_data();
    let y_data = y.get_data();

    if x_data.len() != y_data.len() {
        return Err(Error::new(exception::arg_error(), "x and y must have the same length"));
    }

    let n = x_data.len();
    let m = deg + 1;

    if n < m {
        return Err(Error::new(exception::arg_error(), "Not enough data points for polynomial degree"));
    }

    let mut vandermonde = vec![0.0; n * m];
    for i in 0..n {
        let xi = x_data.iter().nth(i).cloned().unwrap();
        for j in 0..m {
            vandermonde[i * m + j] = xi.powi((m - 1 - j) as i32);
        }
    }

    let mut ata = vec![0.0; m * m];
    let mut atb = vec![0.0; m];

    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vandermonde[k * m + i] * vandermonde[k * m + j];
            }
            ata[i * m + j] = sum;
        }
        let mut sum = 0.0;
        for k in 0..n {
            sum += vandermonde[k * m + i] * y_data.iter().nth(k).cloned().unwrap();
        }
        atb[i] = sum;
    }

    let mut coeffs = atb.clone();
    for i in 0..m {
        let mut max_row = i;
        for k in i+1..m {
            if ata[k * m + i].abs() > ata[max_row * m + i].abs() {
                max_row = k;
            }
        }
        for j in 0..m {
            let tmp = ata[i * m + j];
            ata[i * m + j] = ata[max_row * m + j];
            ata[max_row * m + j] = tmp;
        }
        let tmp = coeffs[i];
        coeffs[i] = coeffs[max_row];
        coeffs[max_row] = tmp;

        // Check pivot tolerance before division to avoid numerical instability
        let pivot = ata[i * m + i];
        if pivot.abs() < 1e-14 {
            return Err(Error::new(exception::runtime_error(),
                "Polyfit failed: matrix is singular or nearly singular. Data may be collinear."));
        }

        for k in i+1..m {
            let factor = ata[k * m + i] / pivot;
            for j in i..m {
                ata[k * m + j] -= factor * ata[i * m + j];
            }
            coeffs[k] -= factor * coeffs[i];
        }
    }

    for i in (0..m).rev() {
        for j in i+1..m {
            coeffs[i] -= ata[i * m + j] * coeffs[j];
        }
        // Check pivot tolerance before final division
        if ata[i * m + i].abs() < 1e-14 {
            return Err(Error::new(exception::runtime_error(),
                "Polyfit failed: matrix is singular or nearly singular. Data may be collinear."));
        }
        coeffs[i] /= ata[i * m + i];
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m]), coeffs).unwrap(),
    )))
}

pub fn polyval(p: &NDArray, x: &NDArray) -> Result<Obj<NDArray>, Error> {
    let p_data = p.get_data();
    let x_data = x.get_data();

    let coeffs: Vec<f64> = p_data.iter().cloned().collect();
    let result: Vec<f64> = x_data.iter().map(|&xi| {
        let mut val = 0.0;
        for &c in coeffs.iter() {
            val = val * xi + c;
        }
        val
    }).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(x_data.raw_dim(), result).unwrap(),
    )))
}

pub fn digitize(x: &NDArray, bins: &NDArray) -> Result<Obj<NDArray>, Error> {
    digitize_right(x, bins, Some(false))
}

/// Digitize with right parameter.
///
/// Returns bin indices for each value in x. Uses binary search internally
/// for O(log n) performance per element.
///
/// right=False (default): bins[i-1] <= x < bins[i]
/// right=True: bins[i-1] < x <= bins[i]
///
/// Note: This is equivalent to searchsorted with appropriate side parameter.
pub fn digitize_right(x: &NDArray, bins: &NDArray, right: Option<bool>) -> Result<Obj<NDArray>, Error> {
    let x_data = x.get_data();
    let bins_data = bins.get_data();

    let bins_vec: Vec<f64> = bins_data.iter().cloned().collect();
    let right = right.unwrap_or(false);

    // Use binary search (same as searchsorted) for O(log n) performance
    let result: Vec<f64> = x_data.iter().map(|&xi| {
        let mut lo = 0;
        let mut hi = bins_vec.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let cmp = if right {
                xi > bins_vec[mid]
            } else {
                xi >= bins_vec[mid]
            };
            if cmp {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as f64
    }).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(x_data.raw_dim(), result).unwrap(),
    )))
}
