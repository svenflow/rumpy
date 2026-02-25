//! Random number generation

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, RArray, TryConvert};
use ndarray::{ArrayD, IxDyn};
use rand::prelude::*;
use rand_distr::{Binomial, Exp, Normal, Poisson, Uniform};
use std::sync::Mutex;

// Global RNG state
lazy_static::lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::from_entropy());
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
    let dist = Uniform::new(0.0, 1.0);
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
    let dist = Uniform::new(low, high);
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
    let dist = Uniform::new(low, high);
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

/// Random choice from array
pub fn choice(arr: &NDArray, size: usize) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();
    let n = data.len();

    if n == 0 {
        return Err(Error::new(exception::arg_error(), "Array is empty"));
    }

    let mut rng = RNG.lock().unwrap();
    let dist = Uniform::new(0, n);
    let values: Vec<f64> = (0..size)
        .map(|_| {
            let idx = dist.sample(&mut *rng);
            data.iter().nth(idx).cloned().unwrap_or(0.0)
        })
        .collect();

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
