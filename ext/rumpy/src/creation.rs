//! Array creation functions

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, RArray, TryConvert, Value};
use ndarray::{ArrayD, IxDyn};

/// Create an array from a Ruby array (nested arrays supported)
pub fn array(value: Value) -> Result<Obj<NDArray>, Error> {
    Ok(Obj::wrap(NDArray::from_ruby_array(value)?))
}

/// Create an array of zeros with given shape
pub fn zeros(shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec: Vec<usize> = shape
        .into_iter()
        .map(|v| usize::try_convert(v))
        .collect::<Result<Vec<_>, _>>()?;

    let arr = ArrayD::zeros(IxDyn(&shape_vec));
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an array of ones with given shape
pub fn ones(shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec: Vec<usize> = shape
        .into_iter()
        .map(|v| usize::try_convert(v))
        .collect::<Result<Vec<_>, _>>()?;

    let arr = ArrayD::ones(IxDyn(&shape_vec));
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an array filled with a given value
pub fn full(shape: RArray, value: f64) -> Result<Obj<NDArray>, Error> {
    let shape_vec: Vec<usize> = shape
        .into_iter()
        .map(|v| usize::try_convert(v))
        .collect::<Result<Vec<_>, _>>()?;

    let arr = ArrayD::from_elem(IxDyn(&shape_vec), value);
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an empty (uninitialized) array with given shape
/// Note: In Rust we initialize to zero for safety
pub fn empty(shape: RArray) -> Result<Obj<NDArray>, Error> {
    zeros(shape)
}

/// Create an array with evenly spaced values within a given interval
pub fn arange(start: f64, stop: f64, step: f64) -> Result<Obj<NDArray>, Error> {
    if step == 0.0 {
        return Err(Error::new(exception::arg_error(), "Step cannot be zero"));
    }

    let mut values = Vec::new();
    let mut current = start;

    if step > 0.0 {
        while current < stop {
            values.push(current);
            current += step;
        }
    } else {
        while current > stop {
            values.push(current);
            current += step;
        }
    }

    let arr = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create evenly spaced numbers over a specified interval
pub fn linspace(start: f64, stop: f64, num: usize) -> Result<Obj<NDArray>, Error> {
    if num == 0 {
        return Ok(Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0])))));
    }
    if num == 1 {
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![start]).unwrap(),
        )));
    }

    let step = (stop - start) / (num - 1) as f64;
    let values: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();

    let arr = ArrayD::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create numbers spaced evenly on a log scale (base 10)
pub fn logspace(start: f64, stop: f64, num: usize) -> Result<Obj<NDArray>, Error> {
    logspace_base(start, stop, num, Some(10.0))
}

/// Create numbers spaced evenly on a log scale with custom base
pub fn logspace_base(start: f64, stop: f64, num: usize, base: Option<f64>) -> Result<Obj<NDArray>, Error> {
    let base = base.unwrap_or(10.0);

    if num == 0 {
        return Ok(Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0])))));
    }
    if num == 1 {
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![base.powf(start)]).unwrap(),
        )));
    }

    let step = (stop - start) / (num - 1) as f64;
    let values: Vec<f64> = (0..num)
        .map(|i| base.powf(start + step * i as f64))
        .collect();

    let arr = ArrayD::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create numbers spaced evenly on a geometric scale
pub fn geomspace(start: f64, stop: f64, num: usize) -> Result<Obj<NDArray>, Error> {
    if start == 0.0 || stop == 0.0 {
        return Err(Error::new(exception::arg_error(), "Geometric sequence cannot include zero"));
    }
    if (start < 0.0) != (stop < 0.0) {
        return Err(Error::new(exception::arg_error(), "Geometric sequence cannot include sign change"));
    }

    if num == 0 {
        return Ok(Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0])))));
    }
    if num == 1 {
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![start]).unwrap(),
        )));
    }

    // Use logarithmic spacing
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();
    let sign = if start < 0.0 { -1.0 } else { 1.0 };

    let log_step = (log_stop - log_start) / (num - 1) as f64;
    let values: Vec<f64> = (0..num)
        .map(|i| sign * (log_start + log_step * i as f64).exp())
        .collect();

    let arr = ArrayD::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an identity matrix
pub fn eye(n: usize) -> Result<Obj<NDArray>, Error> {
    eye_k(n, Some(n), Some(0))
}

/// Create an identity-like matrix with optional M (rows) and k (diagonal offset)
/// k > 0: diagonal above main, k < 0: diagonal below main
pub fn eye_k(n: usize, m: Option<usize>, k: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let m = m.unwrap_or(n);
    let k = k.unwrap_or(0);

    let mut data = vec![0.0; n * m];
    for i in 0..n {
        // Handle negative k without overflow
        let j_signed = i as i64 + k;
        if j_signed >= 0 && (j_signed as usize) < m {
            let j = j_signed as usize;
            data[i * m + j] = 1.0;
        }
    }

    let arr = ArrayD::from_shape_vec(IxDyn(&[n, m]), data)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an identity matrix (alias for eye)
pub fn identity(n: usize) -> Result<Obj<NDArray>, Error> {
    eye(n)
}

/// Extract a diagonal or construct a diagonal array
pub fn diag(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    diag_k(arr, Some(0))
}

/// Extract a diagonal or construct a diagonal array with k offset
/// k > 0: diagonal above main, k < 0: diagonal below main
pub fn diag_k(arr: &NDArray, k: Option<i64>) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let shape = data.shape();
    let k = k.unwrap_or(0);

    if shape.len() == 1 {
        // 1D -> create diagonal matrix with offset k
        let n = shape[0];
        let matrix_size = n + k.unsigned_abs() as usize;
        let mut result = vec![0.0; matrix_size * matrix_size];
        for (i, &val) in data.iter().enumerate() {
            let row = if k >= 0 { i } else { i + (-k) as usize };
            let col = if k >= 0 { i + k as usize } else { i };
            if row < matrix_size && col < matrix_size {
                result[row * matrix_size + col] = val;
            }
        }
        Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[matrix_size, matrix_size]), result).unwrap(),
        )))
    } else if shape.len() == 2 {
        // 2D -> extract diagonal with offset k
        let m = shape[0];
        let n = shape[1];
        let start_row = if k < 0 { (-k) as usize } else { 0 };
        let start_col = if k > 0 { k as usize } else { 0 };

        let diag_len = if k >= 0 {
            m.min(n.saturating_sub(k as usize))
        } else {
            n.min(m.saturating_sub((-k) as usize))
        };

        let mut result = vec![0.0; diag_len];
        for i in 0..diag_len {
            let row = start_row + i;
            let col = start_col + i;
            if row < m && col < n {
                result[i] = data[[row, col]];
            }
        }
        Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[diag_len]), result).unwrap(),
        )))
    } else {
        Err(Error::new(
            exception::arg_error(),
            "diag requires 1D or 2D array",
        ))
    }
}

/// Create a zeros array with the same shape as another array
pub fn zeros_like(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let shape = arr.shape();
    let result = ArrayD::zeros(IxDyn(&shape));
    Ok(Obj::wrap(NDArray::new(result)))
}

/// Create a ones array with the same shape as another array
pub fn ones_like(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let shape = arr.shape();
    let result = ArrayD::ones(IxDyn(&shape));
    Ok(Obj::wrap(NDArray::new(result)))
}

/// Create coordinate matrices from coordinate vectors (meshgrid)
/// Uses 'xy' indexing by default (Cartesian)
pub fn meshgrid(x: &NDArray, y: &NDArray) -> Result<RArray, Error> {
    meshgrid_indexing(x, y, None)
}

/// Create coordinate matrices with indexing parameter
/// indexing='xy' (default): Cartesian indexing (X varies along columns, Y along rows)
/// indexing='ij': Matrix indexing (X varies along rows, Y along columns)
pub fn meshgrid_indexing(x: &NDArray, y: &NDArray, indexing: Option<String>) -> Result<RArray, Error> {
    let x_data = x.get_data();
    let y_data = y.get_data();

    let nx = x_data.len();
    let ny = y_data.len();

    let indexing = indexing.unwrap_or_else(|| "xy".to_string());

    let (xx, yy, shape) = if indexing == "ij" {
        // Matrix indexing: X varies along rows (axis 0), Y along columns (axis 1)
        // Shape is (nx, ny)
        let mut xx = Vec::with_capacity(nx * ny);
        let mut yy = Vec::with_capacity(nx * ny);

        for &xv in x_data.iter() {
            for &yv in y_data.iter() {
                xx.push(xv);
                yy.push(yv);
            }
        }

        (xx, yy, vec![nx, ny])
    } else {
        // Cartesian indexing (default 'xy'): X varies along columns, Y along rows
        // Shape is (ny, nx)
        let mut xx = Vec::with_capacity(ny * nx);
        let mut yy = Vec::with_capacity(ny * nx);

        for &yv in y_data.iter() {
            for &xv in x_data.iter() {
                xx.push(xv);
                yy.push(yv);
            }
        }

        (xx, yy, vec![ny, nx])
    };

    let xx_arr = NDArray::new(ArrayD::from_shape_vec(IxDyn(&shape), xx).unwrap());
    let yy_arr = NDArray::new(ArrayD::from_shape_vec(IxDyn(&shape), yy).unwrap());

    let result = RArray::new();
    result.push(Obj::wrap(xx_arr))?;
    result.push(Obj::wrap(yy_arr))?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace() {
        // Mirrors numpy's test_linspace
        let arr = linspace(0.0, 10.0, 11).unwrap();
        let data = arr.get_data();
        assert_eq!(data.len(), 11);
        assert!((data[[0]] - 0.0).abs() < 1e-10);
        assert!((data[[10]] - 10.0).abs() < 1e-10);
        assert!((data[[5]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_arange() {
        let arr = arange(0.0, 5.0, 1.0).unwrap();
        let data = arr.get_data();
        assert_eq!(data.len(), 5);
        for i in 0..5 {
            assert!((data[[i]] - i as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_eye() {
        let arr = eye(3).unwrap();
        let data = arr.get_data();
        assert_eq!(data.shape(), &[3, 3]);
        assert_eq!(data[[0, 0]], 1.0);
        assert_eq!(data[[1, 1]], 1.0);
        assert_eq!(data[[2, 2]], 1.0);
        assert_eq!(data[[0, 1]], 0.0);
    }
}
