//! RumPy - NumPy for Ruby, powered by Rust
//!
//! This crate provides NumPy-like array operations for Ruby using the ndarray crate.

mod array;
mod creation;
mod linalg;
mod math;
mod random;
mod stats;

use magnus::{define_module, function, method, prelude::*, Error, Ruby};

/// Initialize the RumPy Ruby extension
#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let rumpy = define_module("RumPy")?;

    // Register the NDArray class
    let ndarray_class = rumpy.define_class("NDArray", ruby.class_object())?;

    // Array creation methods on module
    rumpy.define_singleton_method("array", function!(creation::array, 1))?;
    rumpy.define_singleton_method("zeros", function!(creation::zeros, 1))?;
    rumpy.define_singleton_method("ones", function!(creation::ones, 1))?;
    rumpy.define_singleton_method("full", function!(creation::full, 2))?;
    rumpy.define_singleton_method("empty", function!(creation::empty, 1))?;
    rumpy.define_singleton_method("arange", function!(creation::arange, 3))?;
    rumpy.define_singleton_method("linspace", function!(creation::linspace, 3))?;
    rumpy.define_singleton_method("logspace", function!(creation::logspace, 3))?;
    rumpy.define_singleton_method("eye", function!(creation::eye, 1))?;
    rumpy.define_singleton_method("identity", function!(creation::identity, 1))?;
    rumpy.define_singleton_method("diag", function!(creation::diag, 1))?;
    rumpy.define_singleton_method("zeros_like", function!(creation::zeros_like, 1))?;
    rumpy.define_singleton_method("ones_like", function!(creation::ones_like, 1))?;

    // NDArray instance methods
    ndarray_class.define_method("shape", method!(array::NDArray::shape, 0))?;
    ndarray_class.define_method("ndim", method!(array::NDArray::ndim, 0))?;
    ndarray_class.define_method("size", method!(array::NDArray::size, 0))?;
    ndarray_class.define_method("dtype", method!(array::NDArray::dtype, 0))?;
    ndarray_class.define_method("to_a", method!(array::NDArray::to_a, 0))?;
    ndarray_class.define_method("to_s", method!(array::NDArray::to_s, 0))?;
    ndarray_class.define_method("inspect", method!(array::NDArray::inspect, 0))?;
    ndarray_class.define_method("[]", method!(array::NDArray::get, 1))?;
    ndarray_class.define_method("[]=", method!(array::NDArray::set, 2))?;
    ndarray_class.define_method("reshape", method!(array::NDArray::reshape, 1))?;
    ndarray_class.define_method("flatten", method!(array::NDArray::flatten, 0))?;
    ndarray_class.define_method("ravel", method!(array::NDArray::ravel, 0))?;
    ndarray_class.define_method("transpose", method!(array::NDArray::transpose, 0))?;
    ndarray_class.define_method("T", method!(array::NDArray::transpose, 0))?;
    ndarray_class.define_method("copy", method!(array::NDArray::copy, 0))?;
    ndarray_class.define_method("astype", method!(array::NDArray::astype, 1))?;

    // Arithmetic operators
    ndarray_class.define_method("+", method!(array::NDArray::add, 1))?;
    ndarray_class.define_method("-", method!(array::NDArray::sub, 1))?;
    ndarray_class.define_method("*", method!(array::NDArray::mul, 1))?;
    ndarray_class.define_method("/", method!(array::NDArray::div, 1))?;
    ndarray_class.define_method("**", method!(array::NDArray::pow, 1))?;
    ndarray_class.define_method("%", method!(array::NDArray::modulo, 1))?;
    ndarray_class.define_method("-@", method!(array::NDArray::neg, 0))?;

    // Comparison operators
    ndarray_class.define_method("==", method!(array::NDArray::eq, 1))?;
    ndarray_class.define_method("!=", method!(array::NDArray::ne, 1))?;
    ndarray_class.define_method("<", method!(array::NDArray::lt, 1))?;
    ndarray_class.define_method("<=", method!(array::NDArray::le, 1))?;
    ndarray_class.define_method(">", method!(array::NDArray::gt, 1))?;
    ndarray_class.define_method(">=", method!(array::NDArray::ge, 1))?;

    // Mathematical functions on module
    rumpy.define_singleton_method("sin", function!(math::sin, 1))?;
    rumpy.define_singleton_method("cos", function!(math::cos, 1))?;
    rumpy.define_singleton_method("tan", function!(math::tan, 1))?;
    rumpy.define_singleton_method("arcsin", function!(math::arcsin, 1))?;
    rumpy.define_singleton_method("arccos", function!(math::arccos, 1))?;
    rumpy.define_singleton_method("arctan", function!(math::arctan, 1))?;
    rumpy.define_singleton_method("sinh", function!(math::sinh, 1))?;
    rumpy.define_singleton_method("cosh", function!(math::cosh, 1))?;
    rumpy.define_singleton_method("tanh", function!(math::tanh, 1))?;
    rumpy.define_singleton_method("exp", function!(math::exp, 1))?;
    rumpy.define_singleton_method("log", function!(math::log, 1))?;
    rumpy.define_singleton_method("log10", function!(math::log10, 1))?;
    rumpy.define_singleton_method("log2", function!(math::log2, 1))?;
    rumpy.define_singleton_method("sqrt", function!(math::sqrt, 1))?;
    rumpy.define_singleton_method("cbrt", function!(math::cbrt, 1))?;
    rumpy.define_singleton_method("abs", function!(math::abs, 1))?;
    rumpy.define_singleton_method("sign", function!(math::sign, 1))?;
    rumpy.define_singleton_method("floor", function!(math::floor, 1))?;
    rumpy.define_singleton_method("ceil", function!(math::ceil, 1))?;
    rumpy.define_singleton_method("round", function!(math::round, 1))?;
    rumpy.define_singleton_method("clip", function!(math::clip, 3))?;
    rumpy.define_singleton_method("power", function!(math::power, 2))?;
    rumpy.define_singleton_method("square", function!(math::square, 1))?;
    rumpy.define_singleton_method("reciprocal", function!(math::reciprocal, 1))?;

    // Array operations
    rumpy.define_singleton_method("dot", function!(math::dot, 2))?;
    rumpy.define_singleton_method("matmul", function!(math::matmul, 2))?;
    rumpy.define_singleton_method("inner", function!(math::inner, 2))?;
    rumpy.define_singleton_method("outer", function!(math::outer, 2))?;
    rumpy.define_singleton_method("cross", function!(math::cross, 2))?;
    rumpy.define_singleton_method("tensordot", function!(math::tensordot, 3))?;

    // Aggregation on module
    rumpy.define_singleton_method("sum", function!(stats::sum, 1))?;
    rumpy.define_singleton_method("prod", function!(stats::prod, 1))?;
    rumpy.define_singleton_method("mean", function!(stats::mean, 1))?;
    rumpy.define_singleton_method("std", function!(stats::std, 1))?;
    rumpy.define_singleton_method("var", function!(stats::var, 1))?;
    rumpy.define_singleton_method("min", function!(stats::min, 1))?;
    rumpy.define_singleton_method("max", function!(stats::max, 1))?;
    rumpy.define_singleton_method("argmin", function!(stats::argmin, 1))?;
    rumpy.define_singleton_method("argmax", function!(stats::argmax, 1))?;
    rumpy.define_singleton_method("cumsum", function!(stats::cumsum, 1))?;
    rumpy.define_singleton_method("cumprod", function!(stats::cumprod, 1))?;
    rumpy.define_singleton_method("median", function!(stats::median, 1))?;
    rumpy.define_singleton_method("percentile", function!(stats::percentile, 2))?;
    rumpy.define_singleton_method("quantile", function!(stats::quantile, 2))?;
    rumpy.define_singleton_method("histogram", function!(stats::histogram, 2))?;
    rumpy.define_singleton_method("corrcoef", function!(stats::corrcoef, 1))?;
    rumpy.define_singleton_method("cov", function!(stats::cov, 1))?;

    // NDArray aggregation methods
    ndarray_class.define_method("sum", method!(array::NDArray::sum, 0))?;
    ndarray_class.define_method("prod", method!(array::NDArray::prod, 0))?;
    ndarray_class.define_method("mean", method!(array::NDArray::mean, 0))?;
    ndarray_class.define_method("std", method!(array::NDArray::std, 0))?;
    ndarray_class.define_method("var", method!(array::NDArray::var, 0))?;
    ndarray_class.define_method("min", method!(array::NDArray::min, 0))?;
    ndarray_class.define_method("max", method!(array::NDArray::max, 0))?;
    ndarray_class.define_method("argmin", method!(array::NDArray::argmin, 0))?;
    ndarray_class.define_method("argmax", method!(array::NDArray::argmax, 0))?;

    // Boolean operations
    ndarray_class.define_method("all", method!(array::NDArray::all, 0))?;
    ndarray_class.define_method("any", method!(array::NDArray::any, 0))?;

    // Linear algebra submodule
    let linalg_mod = rumpy.define_module("Linalg")?;
    linalg_mod.define_singleton_method("dot", function!(linalg::dot, 2))?;
    linalg_mod.define_singleton_method("matmul", function!(linalg::matmul, 2))?;
    linalg_mod.define_singleton_method("inv", function!(linalg::inv, 1))?;
    linalg_mod.define_singleton_method("pinv", function!(linalg::pinv, 1))?;
    linalg_mod.define_singleton_method("det", function!(linalg::det, 1))?;
    linalg_mod.define_singleton_method("trace", function!(linalg::trace, 1))?;
    linalg_mod.define_singleton_method("rank", function!(linalg::rank, 1))?;
    linalg_mod.define_singleton_method("norm", function!(linalg::norm, 1))?;
    linalg_mod.define_singleton_method("cond", function!(linalg::cond, 1))?;
    linalg_mod.define_singleton_method("eig", function!(linalg::eig, 1))?;
    linalg_mod.define_singleton_method("eigvals", function!(linalg::eigvals, 1))?;
    linalg_mod.define_singleton_method("eigh", function!(linalg::eigh, 1))?;
    linalg_mod.define_singleton_method("eigvalsh", function!(linalg::eigvalsh, 1))?;
    linalg_mod.define_singleton_method("svd", function!(linalg::svd, 1))?;
    linalg_mod.define_singleton_method("qr", function!(linalg::qr, 1))?;
    linalg_mod.define_singleton_method("cholesky", function!(linalg::cholesky, 1))?;
    linalg_mod.define_singleton_method("lu", function!(linalg::lu, 1))?;
    linalg_mod.define_singleton_method("solve", function!(linalg::solve, 2))?;
    linalg_mod.define_singleton_method("lstsq", function!(linalg::lstsq, 2))?;

    // Random submodule
    let random_mod = rumpy.define_module("Random")?;
    random_mod.define_singleton_method("rand", function!(random::rand, 1))?;
    random_mod.define_singleton_method("randn", function!(random::randn, 1))?;
    random_mod.define_singleton_method("randint", function!(random::randint, 3))?;
    random_mod.define_singleton_method("uniform", function!(random::uniform, 3))?;
    random_mod.define_singleton_method("normal", function!(random::normal, 3))?;
    random_mod.define_singleton_method("standard_normal", function!(random::standard_normal, 1))?;
    random_mod.define_singleton_method("binomial", function!(random::binomial, 3))?;
    random_mod.define_singleton_method("poisson", function!(random::poisson, 2))?;
    random_mod.define_singleton_method("exponential", function!(random::exponential, 2))?;
    random_mod.define_singleton_method("choice", function!(random::choice, 2))?;
    random_mod.define_singleton_method("shuffle", function!(random::shuffle, 1))?;
    random_mod.define_singleton_method("permutation", function!(random::permutation, 1))?;
    random_mod.define_singleton_method("seed", function!(random::seed, 1))?;

    // Array manipulation functions
    rumpy.define_singleton_method("concatenate", function!(array::concatenate, 2))?;
    rumpy.define_singleton_method("vstack", function!(array::vstack, 1))?;
    rumpy.define_singleton_method("hstack", function!(array::hstack, 1))?;
    rumpy.define_singleton_method("dstack", function!(array::dstack, 1))?;
    rumpy.define_singleton_method("stack", function!(array::stack, 2))?;
    rumpy.define_singleton_method("split", function!(array::split, 2))?;
    rumpy.define_singleton_method("vsplit", function!(array::vsplit, 2))?;
    rumpy.define_singleton_method("hsplit", function!(array::hsplit, 2))?;
    rumpy.define_singleton_method("tile", function!(array::tile, 2))?;
    rumpy.define_singleton_method("repeat", function!(array::repeat, 2))?;
    rumpy.define_singleton_method("flip", function!(array::flip, 1))?;
    rumpy.define_singleton_method("fliplr", function!(array::fliplr, 1))?;
    rumpy.define_singleton_method("flipud", function!(array::flipud, 1))?;
    rumpy.define_singleton_method("roll", function!(array::roll, 2))?;
    rumpy.define_singleton_method("rot90", function!(array::rot90, 1))?;

    // Logical operations
    rumpy.define_singleton_method("logical_and", function!(math::logical_and, 2))?;
    rumpy.define_singleton_method("logical_or", function!(math::logical_or, 2))?;
    rumpy.define_singleton_method("logical_not", function!(math::logical_not, 1))?;
    rumpy.define_singleton_method("logical_xor", function!(math::logical_xor, 2))?;
    rumpy.define_singleton_method("where", function!(math::where_fn, 3))?;
    rumpy.define_singleton_method("nonzero", function!(math::nonzero, 1))?;
    rumpy.define_singleton_method("flatnonzero", function!(math::flatnonzero, 1))?;

    // Sorting and searching
    rumpy.define_singleton_method("sort", function!(array::sort, 1))?;
    rumpy.define_singleton_method("argsort", function!(array::argsort, 1))?;
    rumpy.define_singleton_method("searchsorted", function!(array::searchsorted, 2))?;
    rumpy.define_singleton_method("unique", function!(array::unique, 1))?;

    // FFT submodule
    let fft_mod = rumpy.define_module("FFT")?;
    fft_mod.define_singleton_method("fft", function!(math::fft, 1))?;
    fft_mod.define_singleton_method("ifft", function!(math::ifft, 1))?;
    fft_mod.define_singleton_method("fft2", function!(math::fft2, 1))?;
    fft_mod.define_singleton_method("ifft2", function!(math::ifft2, 1))?;
    fft_mod.define_singleton_method("fftn", function!(math::fftn, 1))?;
    fft_mod.define_singleton_method("ifftn", function!(math::ifftn, 1))?;
    fft_mod.define_singleton_method("rfft", function!(math::rfft, 1))?;
    fft_mod.define_singleton_method("irfft", function!(math::irfft, 1))?;
    fft_mod.define_singleton_method("fftfreq", function!(math::fftfreq, 2))?;
    fft_mod.define_singleton_method("rfftfreq", function!(math::rfftfreq, 2))?;
    fft_mod.define_singleton_method("fftshift", function!(math::fftshift, 1))?;
    fft_mod.define_singleton_method("ifftshift", function!(math::ifftshift, 1))?;

    Ok(())
}
