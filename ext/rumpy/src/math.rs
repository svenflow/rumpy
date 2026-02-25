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
    // NumPy's sign returns 0.0 for 0.0, unlike Rust's signum which returns 1.0
    Obj::wrap(NDArray::new(data.mapv(|x| {
        if x == 0.0 { 0.0 } else { x.signum() }
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

pub fn clip(arr: &NDArray, min_val: f64, max_val: f64) -> Obj<NDArray> {
    let data = arr.get_data();
    Obj::wrap(NDArray::new(data.mapv(|x| x.max(min_val).min(max_val))))
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
    dot(a, b)
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

    if a_data.len() != 3 || b_data.len() != 3 {
        return Err(Error::new(exception::arg_error(), "Cross product requires 3D vectors"));
    }

    let a_vec: Vec<f64> = a_data.iter().cloned().collect();
    let b_vec: Vec<f64> = b_data.iter().cloned().collect();

    let result = vec![
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
    ];

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), result).unwrap(),
    )))
}

pub fn tensordot(a: &NDArray, b: &NDArray, axes: i64) -> Result<Obj<NDArray>, Error> {
    // Simplified tensordot - just handles the common cases
    if axes == 1 {
        return matmul(a, b);
    }
    if axes == 0 {
        return outer(a, b);
    }

    Err(Error::new(
        exception::arg_error(),
        "tensordot only supports axes=0 or axes=1",
    ))
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

pub fn nonzero(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
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

pub fn flatnonzero(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    nonzero(arr)
}

// FFT (placeholder implementations - would need rustfft for real implementation)

pub fn fft(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    // Placeholder - returns input (would need complex number support)
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn ifft(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn fft2(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn ifft2(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn fftn(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn ifftn(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn rfft(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
}

pub fn irfft(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::new(arr.get_data().clone())))
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
