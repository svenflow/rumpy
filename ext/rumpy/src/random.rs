//! Random number generation

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, RArray, TryConvert};
use ndarray::{ArrayD, IxDyn};
use rand::prelude::*;
use rand_distr::{Binomial, Exp, Normal, Poisson, Uniform};
use std::sync::Mutex;

// Global RNG state
lazy_static::lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::from_os_rng());
}

fn get_shape(shape: RArray) -> Result<Vec<usize>, Error> {
    shape
        .into_iter()
        .map(|v| usize::try_convert(v))
        .collect::<Result<Vec<_>, _>>()
}

/// Uniform random values in [0, 1)
pub fn rand(shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng)).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Standard normal random values (mean=0, std=1)
pub fn randn(shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng)).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Random integers in [low, high)
pub fn randint(low: i64, high: i64, shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Uniform::new(low, high).unwrap();
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng) as f64).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Uniform distribution in [low, high)
pub fn uniform(low: f64, high: f64, shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Uniform::new(low, high).unwrap();
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng)).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Normal (Gaussian) distribution
pub fn normal(loc: f64, scale: f64, shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Normal::new(loc, scale).map_err(|e| {
        Error::new(exception::arg_error(), format!("Invalid parameters: {}", e))
    })?;
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng)).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Standard normal distribution
pub fn standard_normal(shape: RArray) -> Result<Obj<NDArray>, Error> {
    randn(shape)
}

/// Binomial distribution
pub fn binomial(n: u64, p: f64, shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Binomial::new(n, p).map_err(|e| {
        Error::new(exception::arg_error(), format!("Invalid parameters: {}", e))
    })?;
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng) as f64).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Poisson distribution
pub fn poisson(lam: f64, shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Poisson::new(lam).map_err(|e| {
        Error::new(exception::arg_error(), format!("Invalid parameters: {}", e))
    })?;
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng) as f64).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Exponential distribution
pub fn exponential(scale: f64, shape: RArray) -> Result<Obj<NDArray>, Error> {
    let shape_vec = get_shape(shape)?;
    let size: usize = shape_vec.iter().product();

    let mut rng = RNG.lock().unwrap();
    let dist = Exp::new(1.0 / scale).map_err(|e| {
        Error::new(exception::arg_error(), format!("Invalid parameters: {}", e))
    })?;
    let values: Vec<f64> = (0..size).map(|_| dist.sample(&mut *rng)).collect();

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&shape_vec), values).unwrap(),
    )))
}

/// Random choice from array (with replacement by default)
pub fn choice(arr: &NDArray, size: usize) -> Result<Obj<NDArray>, Error> {
    choice_full(arr, size, Some(true), None)
}

/// Random choice with replace and p (probability) parameters
/// replace=True: sample with replacement (default)
/// replace=False: sample without replacement (size must be <= array length)
/// p: probability array (must sum to 1.0, same length as arr)
pub fn choice_full(arr: &NDArray, size: usize, replace: Option<bool>, p: Option<&NDArray>) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let n = data.len();
    let replace = replace.unwrap_or(true);

    if n == 0 {
        return Err(Error::new(exception::arg_error(), "Array is empty"));
    }

    if !replace && size > n {
        return Err(Error::new(exception::arg_error(),
            format!("Cannot take {} samples without replacement from array of size {}", size, n)));
    }

    let flat: Vec<f64> = data.iter().cloned().collect();
    let mut rng = RNG.lock().unwrap();

    // Get probabilities
    let probs: Option<Vec<f64>> = p.map(|p_arr| p_arr.get_data().iter().cloned().collect());

    let values: Vec<f64> = if let Some(probs) = probs {
        // Validate probabilities
        if probs.len() != n {
            return Err(Error::new(exception::arg_error(), "p must have the same length as the array"));
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(Error::new(exception::arg_error(), format!("p must sum to 1.0, got {}", sum)));
        }

        // Sample with probabilities using cumulative distribution
        let mut cumsum = vec![0.0; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + probs[i];
        }

        if replace {
            // With replacement
            (0..size)
                .map(|_| {
                    let r: f64 = rng.random();
                    // Binary search in cumsum
                    let mut lo = 0;
                    let mut hi = n;
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if r >= cumsum[mid + 1] {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    flat[lo.min(n - 1)]
                })
                .collect()
        } else {
            // Without replacement - use reservoir-like approach
            let mut available: Vec<usize> = (0..n).collect();
            let mut result = Vec::with_capacity(size);

            for _ in 0..size {
                // Recalculate probabilities for remaining elements
                let remaining_sum: f64 = available.iter().map(|&i| probs[i]).sum();
                let r: f64 = rng.random::<f64>() * remaining_sum;

                let mut cumsum = 0.0;
                let mut chosen_idx = 0;
                for (j, &i) in available.iter().enumerate() {
                    cumsum += probs[i];
                    if r <= cumsum {
                        chosen_idx = j;
                        break;
                    }
                }

                result.push(flat[available[chosen_idx]]);
                available.remove(chosen_idx);
            }
            result
        }
    } else {
        // Uniform probabilities
        if replace {
            let dist = Uniform::new(0, n).unwrap();
            (0..size)
                .map(|_| flat[dist.sample(&mut *rng)])
                .collect()
        } else {
            // Fisher-Yates shuffle for without replacement
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..size {
                let remaining = n - i;
                let dist = Uniform::new(0, remaining).unwrap();
                let j = i + dist.sample(&mut *rng);
                indices.swap(i, j);
            }
            indices[..size].iter().map(|&i| flat[i]).collect()
        }
    };

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[size]), values).unwrap(),
    )))
}

/// Shuffle array in place (returns shuffled copy)
pub fn shuffle(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let mut values: Vec<f64> = data.iter().cloned().collect();

    let mut rng = RNG.lock().unwrap();
    values.shuffle(&mut *rng);

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(data.raw_dim(), values).unwrap(),
    )))
}

/// Random permutation
pub fn permutation(n: i64) -> Result<Obj<NDArray>, Error> {
    let n = n as usize;
    let mut values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let mut rng = RNG.lock().unwrap();
    values.shuffle(&mut *rng);

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), values).unwrap(),
    )))
}

/// Set random seed
pub fn seed(s: u64) -> Result<(), Error> {
    let mut rng = RNG.lock().unwrap();
    *rng = StdRng::seed_from_u64(s);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_reproducibility() {
        seed(42).unwrap();
        let arr = magnus::Ruby::get().map(|_| {
            // Can't test without Ruby runtime
        });
        // In actual tests, verify that same seed produces same sequence
    }
}
