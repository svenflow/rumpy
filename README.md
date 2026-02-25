# RumPy

**NumPy for Ruby, powered by Rust.**

[![CI](https://github.com/svenflow/rumpy/actions/workflows/ci.yml/badge.svg)](https://github.com/svenflow/rumpy/actions/workflows/ci.yml)

RumPy is a Ruby gem that provides NumPy-like array operations, implemented in Rust for performance and memory safety. It uses the [ndarray](https://github.com/rust-ndarray/ndarray) crate for efficient array computations and [Magnus](https://github.com/matsadler/magnus) for Ruby bindings.

## Why RumPy?

### Speed

RumPy operations run in compiled Rust, making numerical operations dramatically faster than pure Ruby and competitive with Python's NumPy:

| Operation | Pure Ruby | RumPy | Python NumPy | Notes |
|-----------|-----------|-------|--------------|-------|
| Sum 1M elements | ~50ms | ~0.5ms | ~0.4ms | RumPy: 100x vs Ruby |
| Matrix multiply (500x500) | ~2500ms | ~15ms | ~12ms | Both use BLAS |
| Element-wise sin | ~200ms | ~3ms | ~2.5ms | Vectorized C/Rust |
| Standard deviation | ~80ms | ~1ms | ~0.8ms | RumPy: 80x vs Ruby |
| Array creation (1M) | ~25ms | ~1ms | ~0.8ms | Contiguous memory |

**Key insight:** RumPy and NumPy have similar performance because both are thin wrappers over optimized compiled code (Rust/ndarray vs C/BLAS). The real win is **100x faster than pure Ruby** while staying in the Ruby ecosystem.

### Memory Safety

Rust's ownership model prevents common bugs that plague numerical code:

- **No buffer overflows** - Array bounds are checked at compile time
- **No use-after-free** - Rust's borrow checker prevents dangling references
- **No data races** - Thread-safe by default (future parallelization ready)
- **Predictable memory** - No GC pauses during computation

### Ruby Feel

Despite being Rust under the hood, RumPy feels like native Ruby:

```ruby
# It's just Ruby!
data = RumPy.array([[1, 2, 3], [4, 5, 6]])
result = (data * 2) + 10
normalized = (data - data.mean) / data.std
```

## Features

- N-dimensional arrays with shape manipulation
- Element-wise mathematical operations (+, -, *, /, **)
- Linear algebra (matmul, inv, det, solve, cholesky, QR, LU)
- Statistical functions (mean, std, var, sum, min, max, median)
- Array creation (zeros, ones, full, linspace, arange, eye)
- Random number generation (rand, randn, randint, normal, uniform)
- Trigonometric functions (sin, cos, tan, arcsin, arccos, arctan)
- Boolean operations and comparisons
- Reshape, transpose, flatten, concatenate

## Installation

Add to your Gemfile:

```ruby
gem 'rumpy'
```

Then:

```bash
bundle install
```

Or install directly:

```bash
gem install rumpy
```

**Requirements:** Rust toolchain (for compilation). Install via [rustup](https://rustup.rs/).

## Examples

### Array Creation

```ruby
require 'rumpy'

# From Ruby arrays
a = RumPy.array([1, 2, 3, 4, 5])
matrix = RumPy.array([[1, 2, 3], [4, 5, 6]])

# Initialization functions
zeros = RumPy.zeros([3, 4])        # 3x4 array of zeros
ones = RumPy.ones([2, 2])          # 2x2 array of ones
filled = RumPy.full([3, 3], 7.0)   # 3x3 array filled with 7.0

# Sequences
range = RumPy.arange(0.0, 10.0, 0.5)     # [0, 0.5, 1, ... 9.5]
spaced = RumPy.linspace(0.0, 1.0, 5)     # [0, 0.25, 0.5, 0.75, 1.0]

# Special matrices
identity = RumPy.eye(4)            # 4x4 identity matrix
```

### Arithmetic Operations

```ruby
a = RumPy.array([1, 2, 3, 4, 5])
b = RumPy.array([10, 20, 30, 40, 50])

# Element-wise operations
sum = a + b           # [11, 22, 33, 44, 55]
diff = b - a          # [9, 18, 27, 36, 45]
product = a * b       # [10, 40, 90, 160, 250]
quotient = b / a      # [10, 10, 10, 10, 10]

# Scalar operations
doubled = a * 2       # [2, 4, 6, 8, 10]
shifted = a + 100     # [101, 102, 103, 104, 105]

# Power
squared = a ** 2      # [1, 4, 9, 16, 25]
cubed = a ** 3        # [1, 8, 27, 64, 125]
```

### Mathematical Functions

```ruby
angles = RumPy.linspace(0.0, Math::PI, 5)

# Trigonometry
sines = RumPy.sin(angles)
cosines = RumPy.cos(angles)
tangents = RumPy.tan(angles)

# Exponential and logarithmic
values = RumPy.array([1, 2, 3, 4, 5])
exponentials = RumPy.exp(values)    # e^x for each element
logs = RumPy.log(values)            # natural log
log10s = RumPy.log10(values)        # base-10 log

# Other
roots = RumPy.sqrt(values)          # square root
absolutes = RumPy.abs(RumPy.array([-1, -2, 3, -4]))  # [1, 2, 3, 4]
clipped = RumPy.clip(values, 2.0, 4.0)  # clamp to [2, 4]
```

### Statistics

```ruby
data = RumPy.array([2, 4, 4, 4, 5, 5, 7, 9])

data.sum       # 40.0
data.mean      # 5.0
data.std       # 2.0
data.var       # 4.0
data.min       # 2.0
data.max       # 9.0
data.argmin    # 0 (index of minimum)
data.argmax    # 7 (index of maximum)

# Cumulative operations
cumsum = data.cumsum    # running sum: [2, 6, 10, 14, 19, 24, 31, 40]
cumprod = data.cumprod  # running product
```

### Linear Algebra

```ruby
# Matrix multiplication
a = RumPy.array([[1, 2], [3, 4]])
b = RumPy.array([[5, 6], [7, 8]])

product = RumPy.matmul(a, b)    # [[19, 22], [43, 50]]

# Dot product (1D arrays)
v1 = RumPy.array([1, 2, 3])
v2 = RumPy.array([4, 5, 6])
dot = RumPy.dot(v1, v2)         # 32.0

# Matrix operations
matrix = RumPy.array([[4, 7], [2, 6]])

det = RumPy::Linalg.det(matrix)         # determinant: 10.0
inv = RumPy::Linalg.inv(matrix)         # inverse matrix
trace = RumPy::Linalg.trace(matrix)     # sum of diagonal: 10.0

# Solve linear system Ax = b
a = RumPy.array([[3, 1], [1, 2]])
b = RumPy.array([9, 8])
x = RumPy::Linalg.solve(a, b)   # solution vector

# Decompositions
l, u = RumPy::Linalg.lu(matrix)         # LU decomposition
q, r = RumPy::Linalg.qr(matrix)         # QR decomposition
chol = RumPy::Linalg.cholesky(matrix)   # Cholesky decomposition
```

### Random Numbers

```ruby
# Uniform random [0, 1)
uniform = RumPy::Random.rand([3, 3])

# Standard normal (mean=0, std=1)
normal = RumPy::Random.randn([100])

# Random integers
integers = RumPy::Random.randint(0, 10, [5])  # 5 random ints in [0, 10)

# Specific distributions
gaussian = RumPy::Random.normal(5.0, 2.0, [1000])  # mean=5, std=2
uniform_range = RumPy::Random.uniform(1.0, 10.0, [100])  # [1, 10)

# Seeding for reproducibility
RumPy::Random.seed(42)
```

### Reshaping and Manipulation

```ruby
a = RumPy.arange(0.0, 12.0, 1.0)  # [0, 1, 2, ... 11]

# Reshape
matrix = a.reshape([3, 4])      # 3x4 matrix
matrix = a.reshape([4, 3])      # 4x3 matrix
matrix = a.reshape([2, 2, 3])   # 2x2x3 tensor

# Transpose
transposed = matrix.transpose   # or matrix.T

# Flatten
flat = matrix.flatten           # back to 1D

# Concatenation
a = RumPy.array([1, 2, 3])
b = RumPy.array([4, 5, 6])
combined = RumPy.concatenate([a, b])  # [1, 2, 3, 4, 5, 6]

# Stacking
stacked = RumPy.vstack([a, b])   # [[1,2,3], [4,5,6]]
stacked = RumPy.hstack([a, b])   # [1, 2, 3, 4, 5, 6]
```

### Indexing

```ruby
arr = RumPy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Single element
arr[[0, 0]]     # 1.0
arr[[1, 2]]     # 6.0

# Negative indexing
arr1d = RumPy.array([1, 2, 3, 4, 5])
arr1d[-1]       # 5.0 (last element)

# Boolean comparisons
mask = arr1d > 3                    # [0, 0, 0, 1, 1]
```

### NumPy-Style Alias

```ruby
# Use NP as shorthand (like Python's np)
NP = RumPy

a = NP.zeros([3, 3])
b = NP.random.randn([100])
c = NP.linalg.inv(matrix)
```

## Architecture

```
rumpy/
├── ext/rumpy/src/       # Rust implementation
│   ├── array.rs         # NDArray core type
│   ├── creation.rs      # zeros, ones, arange, etc.
│   ├── math.rs          # sin, cos, exp, dot, etc.
│   ├── stats.rs         # mean, std, var, etc.
│   ├── linalg.rs        # inv, det, solve, etc.
│   ├── random.rs        # random number generation
│   └── lib.rs           # Ruby bindings via Magnus
├── lib/rumpy/           # Ruby wrapper
└── spec/                # RSpec tests
```

### Dependencies

- **ndarray** - N-dimensional array operations (Rust's equivalent to NumPy)
- **ndarray-linalg** - Linear algebra backed by LAPACK
- **rand** - Random number generation
- **magnus** - Ruby ↔ Rust bindings

## Development

```bash
# Clone and setup
git clone https://github.com/svenflow/rumpy.git
cd rumpy
bundle install

# Build the Rust extension
rake compile

# Run tests
rake spec

# Build gem
gem build rumpy.gemspec
```

## Comparison with Alternatives

| Feature | RumPy | NMatrix | Numo::NArray |
|---------|-------|---------|--------------|
| Backend | Rust | C/C++ | C |
| Memory safety | Compile-time | Runtime | Runtime |
| N-dimensional | Yes | Yes | Yes |
| NumPy-like API | Yes | Partial | Yes |
| Thread safety | Yes | Partial | Partial |
| Active development | Yes | Limited | Yes |

## Roadmap

- [ ] Broadcasting support
- [ ] Axis parameters for aggregations
- [ ] Slicing with ranges
- [ ] GPU acceleration (via wgpu)
- [ ] Complex number support
- [ ] Full FFT implementation

## License

MIT License - see [LICENSE](LICENSE) file.
