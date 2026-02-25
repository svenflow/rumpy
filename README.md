# RumPy

**NumPy for Ruby, powered by Rust.**

RumPy is a Ruby gem that provides NumPy-like array operations, implemented in Rust for performance. It uses the [ndarray](https://github.com/rust-ndarray/ndarray) crate for efficient array computations and [Magnus](https://github.com/matsadler/magnus) for Ruby bindings.

## Features

- N-dimensional arrays with broadcasting
- Element-wise mathematical operations
- Linear algebra operations (matmul, inv, det, SVD)
- Statistical functions (mean, std, var, sum)
- Array creation routines (zeros, ones, linspace, arange)
- Random number generation
- Boolean indexing and fancy indexing
- Memory-efficient views and slicing

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rumpy'
```

And then execute:

```bash
$ bundle install
```

Or install it yourself as:

```bash
$ gem install rumpy
```

## Usage

```ruby
require 'rumpy'

# Create arrays
a = RumPy.array([1, 2, 3, 4, 5])
b = RumPy.zeros([3, 3])
c = RumPy.ones([2, 4])
d = RumPy.linspace(0, 10, 100)

# Mathematical operations
result = a + b
result = RumPy.sin(a)
result = RumPy.dot(a, b)

# Linear algebra
inv = RumPy.linalg.inv(matrix)
det = RumPy.linalg.det(matrix)
u, s, vt = RumPy.linalg.svd(matrix)

# Statistics
mean = a.mean
std = a.std
sum = a.sum(axis: 0)

# Indexing
a[0]           # First element
a[1..3]        # Slice
a[a > 2]       # Boolean indexing
```

## Development

After checking out the repo, run:

```bash
$ bundle install
$ rake compile
$ rake spec
```

To build and install:

```bash
$ gem build rumpy.gemspec
$ gem install ./rumpy-0.1.0.gem
```

## Architecture

RumPy is structured as:

- `ext/rumpy/` - Rust extension using Magnus for Ruby bindings
- `lib/rumpy/` - Ruby wrapper classes and syntactic sugar
- `spec/` - RSpec tests mirroring NumPy's test suite

The Rust implementation uses:
- `ndarray` - N-dimensional array library
- `ndarray-linalg` - Linear algebra operations
- `rand` - Random number generation
- `magnus` - Ruby bindings

## License

The gem is available as open source under the terms of the MIT License.
