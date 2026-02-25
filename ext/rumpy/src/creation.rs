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

/// Create numbers spaced evenly on a log scale
pub fn logspace(start: f64, stop: f64, num: usize) -> Result<Obj<NDArray>, Error> {
    if num == 0 {
        return Ok(Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0])))));
    }
    if num == 1 {
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0_f64.powf(start)]).unwrap(),
        )));
    }

    let step = (stop - start) / (num - 1) as f64;
    let values: Vec<f64> = (0..num)
        .map(|i| 10.0_f64.powf(start + step * i as f64))
        .collect();

    let arr = ArrayD::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an identity matrix
pub fn eye(n: usize) -> Result<Obj<NDArray>, Error> {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }

    let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), data)
        .map_err(|e| Error::new(exception::runtime_error(), format!("{}", e)))?;
    Ok(Obj::wrap(NDArray::new(arr)))
}

/// Create an identity matrix (alias for eye)
pub fn identity(n: usize) -> Result<Obj<NDArray>, Error> {
    eye(n)
}

/// Extract a diagonal or construct a diagonal array
pub fn diag(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let shape = data.shape();

    if shape.len() == 1 {
        // 1D -> create diagonal matrix
        let n = shape[0];
        let mut result = vec![0.0; n * n];
        for (i, &val) in data.iter().enumerate() {
            result[i * n + i] = val;
        }
        Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[n, n]), result).unwrap(),
        )))
    } else if shape.len() == 2 {
        // 2D -> extract diagonal
        let n = shape[0].min(shape[1]);
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = data[[i, i]];
        }
        Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), result).unwrap(),
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
