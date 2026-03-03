//! Mathematical functions

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, TryConvert, Value};
use ndarray::{ArrayD, IxDyn};

// Element-wise unary operations

pub fn sin(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.sin())))
}

pub fn cos(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.cos())))
}

pub fn tan(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.tan())))
}

pub fn arcsin(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.asin())))
}

pub fn arccos(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.acos())))
}

pub fn arctan(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.atan())))
}

pub fn sinh(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.sinh())))
}

pub fn cosh(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.cosh())))
}

pub fn tanh(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.tanh())))
}

pub fn exp(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.exp())))
}

pub fn log(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.ln())))
}

pub fn log10(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.log10())))
}

pub fn log2(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.log2())))
}

pub fn sqrt(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.sqrt())))
}

pub fn cbrt(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.cbrt())))
}

pub fn abs(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.abs())))
}

pub fn sign(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    // NumPy's sign: returns sign of number, preserving -0.0
    // sign(NaN) = NaN, sign(-0.0) = -0.0, sign(0.0) = 0.0
    Obj::wrap(NDArray::new(data.mapv(|x| {
        if x.is_nan() {
            f64::NAN
        } else if x == 0.0 {
            // Preserve sign of zero: -0.0 stays -0.0, 0.0 stays 0.0
            x
        } else {
            x.signum()
        }
    })))
}

pub fn floor(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.floor())))
}

pub fn ceil(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.ceil())))
}

pub fn round(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.round())))
}

// Numerical stability functions (MODERATE audit issues)

pub fn expm1(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.exp_m1())))
}

pub fn log1p(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.ln_1p())))
}

pub fn rint(arr: &NDArray) -> Obj<NDArray> {
    // Banker's rounding (round to nearest even)
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| {
        let floored = x.floor();
        let frac = x - floored;
        if frac == 0.5 {
            // Round to even
            if floored as i64 % 2 == 0 {
                floored
            } else {
                floored + 1.0
            }
        } else {
            x.round()
        }
    })))
}

pub fn arctan2(y: &NDArray, x: &NDArray) -> Result<Obj<NDArray>, Error> {
    let y_data = y.get_data();
    let x_data = x.get_data();

    if y_data.shape() != x_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    let result = ndarray::Zip::from(&*y_data)
        .and(&*x_data)
        .map_collect(|&y_val, &x_val| y_val.atan2(x_val));

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn deg2rad(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.to_radians())))
}

pub fn rad2deg(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.to_degrees())))
}

pub fn hypot(x: &NDArray, y: &NDArray) -> Result<Obj<NDArray>, Error> {
    let x_data = x.get_data();
    let y_data = y.get_data();

    if x_data.shape() != y_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    let result = ndarray::Zip::from(&*x_data)
        .and(&*y_data)
        .map_collect(|&x_val, &y_val| x_val.hypot(y_val));

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn diff(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();

    if flat.len() < 2 {
        return Ok(Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0])))));
    }

    let result: Vec<f64> = flat.windows(2).map(|w| w[1] - w[0]).collect();
    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap(),
    )))
}

pub fn gradient(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let n = flat.len();

    if n == 0 {
        return Ok(Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0])))));
    }
    if n == 1 {
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        )));
    }

    let mut result = vec![0.0; n];

    // First element: forward difference
    result[0] = flat[1] - flat[0];

    // Interior elements: central difference
    for i in 1..n-1 {
        result[i] = (flat[i+1] - flat[i-1]) / 2.0;
    }

    // Last element: backward difference
    result[n-1] = flat[n-1] - flat[n-2];

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    )))
}

// Modulo/remainder operations

pub fn mod_fn(x: &NDArray, y: &NDArray) -> Result<Obj<NDArray>, Error> {
    let x_data = x.get_data();
    let y_data = y.get_data();

    if x_data.shape() != y_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    // NumPy mod: same sign as divisor
    let result = ndarray::Zip::from(&*x_data)
        .and(&*y_data)
        .map_collect(|&x_val, &y_val| {
            let r = x_val % y_val;
            if r != 0.0 && (r < 0.0) != (y_val < 0.0) {
                r + y_val
            } else {
                r
            }
        });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn fmod(x: &NDArray, y: &NDArray) -> Result<Obj<NDArray>, Error> {
    let x_data = x.get_data();
    let y_data = y.get_data();

    if x_data.shape() != y_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    // C-style fmod: same sign as dividend
    let result = ndarray::Zip::from(&*x_data)
        .and(&*y_data)
        .map_collect(|&x_val, &y_val| x_val % y_val);

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn remainder(x: &NDArray, y: &NDArray) -> Result<Obj<NDArray>, Error> {
    // remainder is same as mod in NumPy
    mod_fn(x, y)
}

// Maximum/minimum with NaN handling (fmax/fmin ignore NaN)

pub fn maximum(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    // maximum propagates NaN
    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| {
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                x.max(y)
            }
        });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn minimum(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    // minimum propagates NaN
    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| {
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                x.min(y)
            }
        });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn fmax(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    // fmax ignores NaN
    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| {
            if x.is_nan() { y }
            else if y.is_nan() { x }
            else { x.max(y) }
        });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn fmin(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    // fmin ignores NaN
    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| {
            if x.is_nan() { y }
            else if y.is_nan() { x }
            else { x.min(y) }
        });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn clip(arr: &NDArray, min_val: f64, max_val: f64) -> Result<Obj<NDArray>, Error> {
    // NumPy semantics for clip with NaN bounds:
    // - NaN min: only apply max bound
    // - NaN max: only apply min bound
    // - Both NaN: return copy of input
    // - min > max: ValueError in NumPy (we'll allow but clamp to empty range)
    let data = arr.get_data();

    let min_is_nan = min_val.is_nan();
    let max_is_nan = max_val.is_nan();

    let result = data.mapv(|x| {
        if x.is_nan() {
            x
        } else if min_is_nan && max_is_nan {
            x
        } else if min_is_nan {
            x.min(max_val)
        } else if max_is_nan {
            x.max(min_val)
        } else {
            x.max(min_val).min(max_val)
        }
    });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn square(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x * x)))
}

pub fn reciprocal(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| 1.0 / x)))
}

pub fn power(arr: &NDArray, exponent: Value) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if let Ok(exp) = f64::try_convert(exponent) {
        return Ok(Obj::wrap(NDArray::new(data.mapv(|x| x.powf(exp)))));
    }
    if let Ok(exp_arr) = <&NDArray>::try_convert(exponent) {
        let exp_data = exp_arr.get_data();
        if data.shape() != exp_data.shape() {
            return Err(Error::new(exception::arg_error(), "Shape mismatch"));
        }
        let result = ndarray::Zip::from(&*data)
            .and(&*exp_data)
            .map_collect(|&a, &b| a.powf(b));
        return Ok(Obj::wrap(NDArray::new(result)));
    }

    Err(Error::new(exception::type_error(), "Invalid exponent type"))
}

// Matrix operations

pub fn dot(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    // 1D dot product
    if a_data.ndim() == 1 && b_data.ndim() == 1 {
        if a_data.len() != b_data.len() {
            return Err(Error::new(exception::arg_error(), "Length mismatch for dot product"));
        }
        let sum: f64 = a_data.iter().zip(b_data.iter()).map(|(&x, &y)| x * y).sum();
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[]), vec![sum]).unwrap(),
        )));
    }

    matmul(a, b)
}

pub fn matmul(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.ndim() != 2 || b_data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "matmul requires 2D arrays"));
    }

    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    if a_shape[1] != b_shape[0] {
        return Err(Error::new(
            exception::arg_error(),
            format!("Cannot multiply {}x{} by {}x{}", a_shape[0], a_shape[1], b_shape[0], b_shape[1]),
        ));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[[i, l]] * b_data[[l, j]];
            }
            result[i * n + j] = sum;
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap(),
    )))
}

pub fn inner(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    // 1D case: standard inner product
    if a_data.ndim() == 1 && b_data.ndim() == 1 {
        return dot(a, b);
    }

    // For higher-dimensional arrays, NumPy's inner sums over the last axis of both arrays
    // Result shape: a.shape[:-1] + b.shape[:-1]
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    // The last dimensions must match
    if a_shape[a_ndim - 1] != b_shape[b_ndim - 1] {
        return Err(Error::new(
            exception::arg_error(),
            format!("shape mismatch: last dimensions {} and {} must match",
                    a_shape[a_ndim - 1], b_shape[b_ndim - 1]),
        ));
    }

    let contract_dim = a_shape[a_ndim - 1];

    // Build result shape
    let mut result_shape: Vec<usize> = Vec::new();
    for i in 0..(a_ndim - 1) {
        result_shape.push(a_shape[i]);
    }
    for i in 0..(b_ndim - 1) {
        result_shape.push(b_shape[i]);
    }

    // Compute sizes
    let a_outer_size: usize = a_shape[..a_ndim - 1].iter().product::<usize>().max(1);
    let b_outer_size: usize = b_shape[..b_ndim - 1].iter().product::<usize>().max(1);
    let result_size = (a_outer_size * b_outer_size).max(1);

    let a_flat: Vec<f64> = a_data.iter().cloned().collect();
    let b_flat: Vec<f64> = b_data.iter().cloned().collect();

    let mut result_data = vec![0.0; result_size];

    for a_outer in 0..a_outer_size {
        for b_outer in 0..b_outer_size {
            let mut sum = 0.0;
            for k in 0..contract_dim {
                let a_idx = a_outer * contract_dim + k;
                let b_idx = b_outer * contract_dim + k;
                if a_idx < a_flat.len() && b_idx < b_flat.len() {
                    sum += a_flat[a_idx] * b_flat[b_idx];
                }
            }
            let result_idx = a_outer * b_outer_size + b_outer;
            if result_idx < result_data.len() {
                result_data[result_idx] = sum;
            }
        }
    }

    // Handle scalar result
    if result_shape.is_empty() {
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[]), result_data).unwrap(),
        )));
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
            .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?,
    )))
}

pub fn outer(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    let m = a_data.len();
    let n = b_data.len();

    let mut result = Vec::with_capacity(m * n);
    for &ai in a_data.iter() {
        for &bi in b_data.iter() {
            result.push(ai * bi);
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap(),
    )))
}

pub fn cross(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    let a_vec: Vec<f64> = a_data.iter().cloned().collect();
    let b_vec: Vec<f64> = b_data.iter().cloned().collect();

    // Handle 2D and 3D vectors
    if a_vec.len() == 2 && b_vec.len() == 2 {
        // 2D cross product returns scalar (z-component of 3D cross)
        let result = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0];
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[]), vec![result]).unwrap(),
        )));
    }

    if a_vec.len() == 3 && b_vec.len() == 3 {
        // 3D cross product
        let result = vec![
            a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
            a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
            a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
        ];
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), result).unwrap(),
        )));
    }

    // Mixed 2D/3D: extend 2D to 3D with z=0
    if (a_vec.len() == 2 || a_vec.len() == 3) && (b_vec.len() == 2 || b_vec.len() == 3) {
        let a3 = if a_vec.len() == 2 { vec![a_vec[0], a_vec[1], 0.0] } else { a_vec };
        let b3 = if b_vec.len() == 2 { vec![b_vec[0], b_vec[1], 0.0] } else { b_vec };

        let result = vec![
            a3[1] * b3[2] - a3[2] * b3[1],
            a3[2] * b3[0] - a3[0] * b3[2],
            a3[0] * b3[1] - a3[1] * b3[0],
        ];
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), result).unwrap(),
        )));
    }

    Err(Error::new(exception::arg_error(), "Cross product requires 2D or 3D vectors"))
}

pub fn tensordot(a: &NDArray, b: &NDArray, axes: i64) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    // Handle axes=0: outer product
    if axes == 0 {
        return outer(a, b);
    }

    // Handle axes=1: simple matrix multiplication case
    if axes == 1 && a_ndim == 2 && b_ndim == 2 {
        return matmul(a, b);
    }

    let axes = axes as usize;

    // Validate axes
    if axes > a_ndim || axes > b_ndim {
        return Err(Error::new(
            exception::arg_error(),
            format!("axes {} is too large for arrays with {} and {} dimensions", axes, a_ndim, b_ndim),
        ));
    }

    // Check that the contracting dimensions match
    for i in 0..axes {
        let a_dim = a_shape[a_ndim - axes + i];
        let b_dim = b_shape[i];
        if a_dim != b_dim {
            return Err(Error::new(
                exception::arg_error(),
                format!("shape mismatch: {} vs {} for contraction axis {}", a_dim, b_dim, i),
            ));
        }
    }

    // Compute result shape: a_shape[:-axes] + b_shape[axes:]
    let mut result_shape: Vec<usize> = Vec::new();
    for i in 0..(a_ndim - axes) {
        result_shape.push(a_shape[i]);
    }
    for i in axes..b_ndim {
        result_shape.push(b_shape[i]);
    }

    // Size of the contracting dimensions
    let contract_size: usize = (0..axes).map(|i| b_shape[i]).product();

    // Size of the outer dimensions for a and b
    let a_outer_size: usize = (0..(a_ndim - axes)).map(|i| a_shape[i]).product();
    let b_outer_size: usize = (axes..b_ndim).map(|i| b_shape[i]).product();

    // Flatten arrays for computation
    let a_flat: Vec<f64> = a_data.iter().cloned().collect();
    let b_flat: Vec<f64> = b_data.iter().cloned().collect();

    let mut result_data = vec![0.0; a_outer_size * b_outer_size];

    // Compute tensordot by iterating over all combinations
    // This is a general but not optimized implementation
    for a_outer in 0..a_outer_size.max(1) {
        for b_outer in 0..b_outer_size.max(1) {
            let mut sum = 0.0;
            for contract in 0..contract_size.max(1) {
                // Map a_outer and contract to flat index in a
                let a_idx = a_outer * contract_size + contract;
                // Map contract and b_outer to flat index in b
                let b_idx = contract * b_outer_size + b_outer;

                if a_idx < a_flat.len() && b_idx < b_flat.len() {
                    sum += a_flat[a_idx] * b_flat[b_idx];
                }
            }
            let result_idx = a_outer * b_outer_size + b_outer;
            if result_idx < result_data.len() {
                result_data[result_idx] = sum;
            }
        }
    }

    // Handle scalar result
    if result_shape.is_empty() {
        result_shape.push(1);
        if result_data.is_empty() {
            result_data.push(0.0);
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
            .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?,
    )))
}

// Logical operations

pub fn logical_and(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn logical_or(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn logical_not(arr: &NDArray) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| if x == 0.0 { 1.0 } else { 0.0 })))
}

pub fn logical_xor(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.shape() != b_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&x, &y| {
            let a_bool = x != 0.0;
            let b_bool = y != 0.0;
            if a_bool ^ b_bool { 1.0 } else { 0.0 }
        });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn where_fn(condition: &NDArray, x: &NDArray, y: &NDArray) -> Result<Obj<NDArray>, Error> {
    let cond = condition.get_data();
    let x_data = x.get_data();
    let y_data = y.get_data();

    if cond.shape() != x_data.shape() || cond.shape() != y_data.shape() {
        return Err(Error::new(exception::arg_error(), "Shape mismatch"));
    }

    let result = ndarray::Zip::from(&*cond)
        .and(&*x_data)
        .and(&*y_data)
        .map_collect(|&c, &xv, &yv| if c != 0.0 { xv } else { yv });

    Ok(Obj::wrap(NDArray::new(result)))
}

pub fn nonzero(arr: &NDArray) -> Result<magnus::RArray, Error> {
    use magnus::RArray;

    let data = arr.get_data();
    let shape = data.shape();
    let ndim = data.ndim();

    // Find all nonzero element positions
    let mut positions: Vec<Vec<usize>> = vec![Vec::new(); ndim];

    // Iterate over all elements with their multi-dimensional indices
    let flat: Vec<f64> = data.iter().cloned().collect();
    for (flat_idx, &val) in flat.iter().enumerate() {
        if val != 0.0 {
            // Convert flat index to multi-dimensional index
            let mut remaining = flat_idx;
            let mut multi_idx = vec![0usize; ndim];
            for d in (0..ndim).rev() {
                let dim_size = shape[d];
                multi_idx[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            for d in 0..ndim {
                positions[d].push(multi_idx[d]);
            }
        }
    }

    // Return tuple of arrays (one per dimension)
    let result = RArray::new();
    for d in 0..ndim {
        let indices: Vec<f64> = positions[d].iter().map(|&i| i as f64).collect();
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices).unwrap()
        );
        result.push(Obj::wrap(arr))?;
    }

    Ok(result)
}

pub fn flatnonzero(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let indices: Vec<f64> = data
        .iter()
        .enumerate()
        .filter(|(_, &x)| x != 0.0)
        .map(|(i, _)| i as f64)
        .collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices).unwrap(),
    )))
}

// FFT - Not implemented (requires complex number support and rustfft crate)
// These functions now return errors instead of silently returning incorrect results

pub fn fft(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "FFT not implemented - requires complex number support"))
}

pub fn ifft(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "IFFT not implemented - requires complex number support"))
}

pub fn fft2(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "FFT2 not implemented - requires complex number support"))
}

pub fn ifft2(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "IFFT2 not implemented - requires complex number support"))
}

pub fn fftn(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "FFTN not implemented - requires complex number support"))
}

pub fn ifftn(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "IFFTN not implemented - requires complex number support"))
}

pub fn rfft(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "RFFT not implemented - requires complex number support"))
}

pub fn irfft(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(), "IRFFT not implemented - requires complex number support"))
}

pub fn fftfreq(n: usize, d: f64) -> Result<Obj<NDArray>, Error> {
    let mut result = vec![0.0; n];
    let n_f = n as f64;
    for i in 0..n / 2 {
        result[i] = i as f64 / (n_f * d);
    }
    for i in n / 2..n {
        result[i] = (i as f64 - n_f) / (n_f * d);
    }
    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), result).unwrap(),
    )))
}

pub fn rfftfreq(n: usize, d: f64) -> Result<Obj<NDArray>, Error> {
    let m = n / 2 + 1;
    let n_f = n as f64;
    let result: Vec<f64> = (0..m).map(|i| i as f64 / (n_f * d)).collect();
    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m]), result).unwrap(),
    )))
}

pub fn fftshift(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let n = data.len();
    let shift = n / 2;

    let flat: Vec<f64> = data.iter().cloned().collect();
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[(i + shift) % n] = flat[i];
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    )))
}

pub fn ifftshift(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let n = data.len();
    let shift = (n + 1) / 2;

    let flat: Vec<f64> = data.iter().cloned().collect();
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[(i + shift) % n] = flat[i];
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), result).unwrap(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_cos() {
        // Test sin(0) = 0, cos(0) = 1
        let arr = NDArray::new(ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap());
        let sin_result = sin(&arr);
        let cos_result = cos(&arr);
        assert!((sin_result.get_data()[[0]]).abs() < 1e-10);
        assert!((cos_result.get_data()[[0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_outer_product() {
        let a = NDArray::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap());
        let b = NDArray::new(ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 5.0]).unwrap());
        let result = outer(&a, &b).unwrap();
        let data = result.get_data();
        assert_eq!(data.shape(), &[3, 2]);
        assert_eq!(data[[0, 0]], 4.0);
        assert_eq!(data[[0, 1]], 5.0);
        assert_eq!(data[[2, 1]], 15.0);
    }
}
