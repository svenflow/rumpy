//! Statistical functions

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, RArray};
use ndarray::{ArrayD, IxDyn};

pub fn sum(arr: &NDArray) -> f64 {
    arr.get_data().sum()
}

pub fn prod(arr: &NDArray) -> f64 {
    arr.get_data().product()
}

pub fn mean(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    data.sum() / data.len() as f64
}

pub fn std(arr: &NDArray) -> f64 {
    var(arr).sqrt()
}

pub fn var(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let m = data.sum() / data.len() as f64;
    data.mapv(|x| (x - m).powi(2)).sum() / data.len() as f64
}

pub fn min(arr: &NDArray) -> f64 {
    arr.get_data().iter().cloned().fold(f64::INFINITY, f64::min)
}

pub fn max(arr: &NDArray) -> f64 {
    arr.get_data().iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

pub fn argmin(arr: &NDArray) -> usize {
    let data = arr.get_data();
    data.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn argmax(arr: &NDArray) -> usize {
    let data = arr.get_data();
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
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

pub fn median(arr: &NDArray) -> f64 {
    let data = arr.get_data();
    let mut sorted: Vec<f64> = data.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

pub fn percentile(arr: &NDArray, q: f64) -> f64 {
    quantile(arr, q / 100.0)
}

pub fn quantile(arr: &NDArray, q: f64) -> f64 {
    let data = arr.get_data();
    let mut sorted: Vec<f64> = data.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

    let min_val = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range == 0.0 {
        return Err(Error::new(exception::arg_error(), "All values are the same"));
    }

    let bin_width = range / bins as f64;
    let mut counts = vec![0.0; bins];
    let mut edges = Vec::with_capacity(bins + 1);

    for i in 0..=bins {
        edges.push(min_val + i as f64 * bin_width);
    }

    for &x in &flat {
        let bin = ((x - min_val) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1.0;
    }

    let ruby = magnus::Ruby::get().unwrap();
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

pub fn corrcoef(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "corrcoef requires 2D array"));
    }

    let shape = data.shape();
    let n_vars = shape[0];
    let n_obs = shape[1];

    // Calculate means
    let means: Vec<f64> = (0..n_vars)
        .map(|i| {
            (0..n_obs).map(|j| data[[i, j]]).sum::<f64>() / n_obs as f64
        })
        .collect();

    // Calculate covariance matrix
    let mut cov = vec![0.0; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut sum = 0.0;
            for k in 0..n_obs {
                sum += (data[[i, k]] - means[i]) * (data[[j, k]] - means[j]);
            }
            cov[i * n_vars + j] = sum / n_obs as f64;
        }
    }

    // Convert to correlation
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

    // Calculate means
    let means: Vec<f64> = (0..n_vars)
        .map(|i| {
            (0..n_obs).map(|j| data[[i, j]]).sum::<f64>() / n_obs as f64
        })
        .collect();

    // Calculate covariance matrix (with Bessel's correction)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        );
        assert!((mean(&arr) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_var() {
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 4.0, 4.0, 4.0]).unwrap(),
        );
        // Mean is 3.5, variance is ((2-3.5)^2 + 3*(4-3.5)^2)/4 = (2.25 + 0.75)/4 = 0.75
        let v = var(&arr);
        assert!((v - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_median() {
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, 3.0, 2.0, 5.0, 4.0]).unwrap(),
        );
        assert!((median(&arr) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumsum() {
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let result = cumsum(&arr);
        let data = result.get_data();
        assert_eq!(data[[0]], 1.0);
        assert_eq!(data[[1]], 3.0);
        assert_eq!(data[[2]], 6.0);
        assert_eq!(data[[3]], 10.0);
    }
}
