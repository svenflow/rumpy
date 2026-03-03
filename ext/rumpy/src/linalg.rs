//! Linear algebra functions

use crate::array::NDArray;
use magnus::{exception, typed_data::Obj, Error, RArray};
use ndarray::{ArrayD, IxDyn};

pub fn dot(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    crate::math::dot(a, b)
}

pub fn matmul(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    crate::math::matmul(a, b)
}

/// Matrix inverse using Gauss-Jordan elimination
pub fn inv(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "inv requires 2D array"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "Matrix must be square"));
    }

    let n = shape[0];
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| data[[i, j]]).collect())
        .collect();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }

        if a[max_row][col].abs() < 1e-10 {
            return Err(Error::new(exception::runtime_error(), "Matrix is singular"));
        }

        // Swap rows
        a.swap(col, max_row);
        inv.swap(col, max_row);

        // Scale pivot row
        let pivot = a[col][col];
        for j in 0..n {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = a[row][col];
                for j in 0..n {
                    a[row][j] -= factor * a[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }

    let result: Vec<f64> = inv.into_iter().flatten().collect();
    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), result).unwrap(),
    )))
}

/// Moore-Penrose pseudo-inverse using regularized least squares
/// Handles rank-deficient matrices by using Tikhonov regularization
pub fn pinv(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "pinv requires 2D array"));
    }

    let shape = data.shape();
    let m = shape[0];
    let n = shape[1];

    let at = Obj::wrap(NDArray::new(data.t().to_owned()));

    // Try the normal equations first: A+ = (A^T A)^-1 A^T
    let ata = matmul(&at, arr)?;

    // Check if A^T A is well-conditioned by checking its determinant
    let d = det(&ata)?;

    if d.abs() > 1e-10 {
        // Matrix is well-conditioned, use normal equations
        let ata_inv = inv(&ata)?;
        return matmul(&ata_inv, &at);
    }

    // Rank-deficient case: use Tikhonov regularization (ridge regression)
    // A+ ≈ (A^T A + λI)^-1 A^T where λ is a small regularization parameter
    let lambda = 1e-10;

    // Add λI to A^T A
    let mut ata_reg_data = ata.get_data().to_owned();
    let ata_shape = ata_reg_data.shape();
    let nn = ata_shape[0];
    for i in 0..nn {
        ata_reg_data[[i, i]] += lambda;
    }

    let ata_reg = Obj::wrap(NDArray::new(ata_reg_data));

    match inv(&ata_reg) {
        Ok(ata_reg_inv) => matmul(&ata_reg_inv, &at),
        Err(_) => {
            // If still fails, use A^T (A A^T + λI)^-1 for m < n case
            let aat = matmul(arr, &at)?;
            let mut aat_reg_data = aat.get_data().to_owned();
            let mm = aat_reg_data.shape()[0];
            for i in 0..mm {
                aat_reg_data[[i, i]] += lambda;
            }
            let aat_reg = Obj::wrap(NDArray::new(aat_reg_data));
            let aat_reg_inv = inv(&aat_reg)?;
            matmul(&at, &aat_reg_inv)
        }
    }
}

/// Matrix determinant using LU decomposition
pub fn det(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "det requires 2D array"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "Matrix must be square"));
    }

    let n = shape[0];

    if n == 1 {
        return Ok(data[[0, 0]]);
    }
    if n == 2 {
        return Ok(data[[0, 0]] * data[[1, 1]] - data[[0, 1]] * data[[1, 0]]);
    }

    // LU decomposition for larger matrices
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| data[[i, j]]).collect())
        .collect();

    let mut det = 1.0;
    let mut sign = 1.0;

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }

        if a[max_row][col].abs() < 1e-15 {
            return Ok(0.0);
        }

        if max_row != col {
            a.swap(col, max_row);
            sign = -sign;
        }

        det *= a[col][col];

        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
        }
    }

    Ok(det * sign)
}

/// Matrix trace
pub fn trace(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "trace requires 2D array"));
    }

    let shape = data.shape();
    let n = shape[0].min(shape[1]);

    let mut sum = 0.0;
    for i in 0..n {
        sum += data[[i, i]];
    }
    Ok(sum)
}

/// Matrix rank (approximate, using SVD concept)
pub fn rank(arr: &NDArray) -> Result<i64, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "rank requires 2D array"));
    }

    // Simplified: use row echelon form to count non-zero rows
    let shape = data.shape();
    let m = shape[0];
    let n = shape[1];

    let mut a: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..n).map(|j| data[[i, j]]).collect())
        .collect();

    let mut rank = 0i64;
    let mut col = 0;

    for row in 0..m {
        if col >= n {
            break;
        }

        // Find pivot
        let mut pivot_row = row;
        while pivot_row < m && a[pivot_row][col].abs() < 1e-10 {
            pivot_row += 1;
        }

        if pivot_row == m {
            col += 1;
            continue;
        }

        a.swap(row, pivot_row);

        // Eliminate below
        for i in (row + 1)..m {
            if a[row][col].abs() > 1e-10 {
                let factor = a[i][col] / a[row][col];
                for j in col..n {
                    a[i][j] -= factor * a[row][j];
                }
            }
        }

        rank += 1;
        col += 1;
    }

    Ok(rank)
}

/// Frobenius norm (default)
pub fn norm(arr: &NDArray) -> Result<f64, Error> {
    norm_ord(arr, None)
}

/// Matrix/vector norm with ord parameter
/// ord=None or ord=2: Frobenius norm for matrices, 2-norm for vectors
/// ord=1: max column sum for matrices, 1-norm for vectors
/// ord=-1: min column sum for matrices
/// ord="inf": max row sum for matrices, infinity norm for vectors
/// ord=-"inf": min row sum for matrices
pub fn norm_ord(arr: &NDArray, ord: Option<f64>) -> Result<f64, Error> {
    let data = arr.get_data();

    // Handle 1D (vector norm)
    if data.ndim() == 1 {
        let flat: Vec<f64> = data.iter().cloned().collect();
        return match ord {
            None | Some(2.0) => {
                let sum_sq: f64 = flat.iter().map(|&x| x * x).sum();
                Ok(sum_sq.sqrt())
            }
            Some(1.0) => {
                Ok(flat.iter().map(|&x| x.abs()).sum())
            }
            Some(p) if p == f64::INFINITY => {
                Ok(flat.iter().map(|&x| x.abs()).fold(0.0, f64::max))
            }
            Some(p) if p == f64::NEG_INFINITY => {
                Ok(flat.iter().map(|&x| x.abs()).fold(f64::INFINITY, f64::min))
            }
            Some(0.0) => {
                // 0-norm = count of nonzero elements
                Ok(flat.iter().filter(|&&x| x != 0.0).count() as f64)
            }
            Some(p) => {
                // p-norm
                let sum: f64 = flat.iter().map(|&x| x.abs().powf(p)).sum();
                Ok(sum.powf(1.0 / p))
            }
        };
    }

    // Handle 2D (matrix norm)
    if data.ndim() == 2 {
        let shape = data.shape();
        let m = shape[0];
        let n = shape[1];

        return match ord {
            None => {
                // Frobenius norm
                let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
                Ok(sum_sq.sqrt())
            }
            Some(2.0) => {
                // Frobenius norm
                let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
                Ok(sum_sq.sqrt())
            }
            Some(1.0) => {
                // Max absolute column sum
                let mut max_sum: f64 = 0.0;
                for j in 0..n {
                    let col_sum: f64 = (0..m).map(|i| data[[i, j]].abs()).sum();
                    max_sum = max_sum.max(col_sum);
                }
                Ok(max_sum)
            }
            Some(p) if p == f64::INFINITY => {
                // Max absolute row sum
                let mut max_sum: f64 = 0.0;
                for i in 0..m {
                    let row_sum: f64 = (0..n).map(|j| data[[i, j]].abs()).sum();
                    max_sum = max_sum.max(row_sum);
                }
                Ok(max_sum)
            }
            Some(p) if p == f64::NEG_INFINITY => {
                // Min absolute row sum
                let mut min_sum: f64 = f64::INFINITY;
                for i in 0..m {
                    let row_sum: f64 = (0..n).map(|j| data[[i, j]].abs()).sum();
                    min_sum = min_sum.min(row_sum);
                }
                Ok(min_sum)
            }
            Some(-1.0) => {
                // Min absolute column sum
                let mut min_sum: f64 = f64::INFINITY;
                for j in 0..n {
                    let col_sum: f64 = (0..m).map(|i| data[[i, j]].abs()).sum();
                    min_sum = min_sum.min(col_sum);
                }
                Ok(min_sum)
            }
            _ => {
                // Default to Frobenius
                let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
                Ok(sum_sq.sqrt())
            }
        };
    }

    // General case: Frobenius norm
    let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
    Ok(sum_sq.sqrt())
}

/// Condition number (ratio of largest to smallest singular value)
/// Uses norm(A) * norm(A^-1) with fallback for singular matrices
pub fn cond(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "cond requires 2D array"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "cond requires square matrix"));
    }

    // Check if matrix is singular
    let d = det(arr)?;
    if d.abs() < 1e-14 {
        // Singular matrix has infinite condition number
        return Ok(f64::INFINITY);
    }

    // Use norm(A) * norm(A^-1)
    let norm_a = norm(arr)?;
    match inv(arr) {
        Ok(inv_a) => {
            let norm_inv = norm(&inv_a)?;
            Ok(norm_a * norm_inv)
        }
        Err(_) => {
            // Inverse failed, matrix is effectively singular
            Ok(f64::INFINITY)
        }
    }
}

/// Eigenvalues and eigenvectors
/// Note: Full implementation requires QR algorithm or external LAPACK
pub fn eig(_arr: &NDArray) -> Result<RArray, Error> {
    Err(Error::new(exception::not_imp_error(),
        "eig not fully implemented - requires LAPACK or similar library for accurate eigendecomposition"))
}

/// Eigenvalues only
pub fn eigvals(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(),
        "eigvals not fully implemented - requires LAPACK or similar library"))
}

/// Eigenvalues/vectors for symmetric matrix
pub fn eigh(_arr: &NDArray) -> Result<RArray, Error> {
    Err(Error::new(exception::not_imp_error(),
        "eigh not fully implemented - requires LAPACK or similar library"))
}

/// Eigenvalues for symmetric matrix
pub fn eigvalsh(_arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    Err(Error::new(exception::not_imp_error(),
        "eigvalsh not fully implemented - requires LAPACK or similar library"))
}

/// Singular Value Decomposition
/// Note: Full implementation requires iterative algorithms or LAPACK
pub fn svd(_arr: &NDArray) -> Result<RArray, Error> {
    Err(Error::new(exception::not_imp_error(),
        "svd not fully implemented - requires LAPACK or similar library for accurate SVD"))
}

/// QR decomposition using Modified Gram-Schmidt (numerically stable)
pub fn qr(arr: &NDArray) -> Result<RArray, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "qr requires 2D array"));
    }

    let shape = data.shape();
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n); // Number of columns in Q

    // Modified Gram-Schmidt - more numerically stable than classical GS
    // We work with columns directly and orthogonalize against previous columns
    let mut q_cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| data[[i, j]]).collect())
        .collect();

    let mut r = vec![0.0; k * n];

    for j in 0..k {
        // Compute norm of column j
        let norm_j: f64 = q_cols[j].iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm_j > 1e-14 {
            r[j * n + j] = norm_j;
            // Normalize column j
            for x in &mut q_cols[j] {
                *x /= norm_j;
            }
        } else {
            r[j * n + j] = 0.0;
            // Column is zero, leave as is
            continue;
        }

        // Modified GS: orthogonalize remaining columns against column j
        for i in (j + 1)..n {
            // Compute projection of column i onto column j
            let proj: f64 = q_cols[i].iter()
                .zip(q_cols[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            r[j * n + i] = proj;

            // Subtract projection
            for row in 0..m {
                q_cols[i][row] -= proj * q_cols[j][row];
            }
        }
    }

    // Build Q matrix (m x k)
    let mut q = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = q_cols[j][i];
        }
    }

    // Build R matrix (k x n)
    let result = RArray::new();
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m, k]), q).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[k, n]), r).unwrap(),
    )))?;

    Ok(result)
}

/// Cholesky decomposition
pub fn cholesky(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "cholesky requires 2D array"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "Matrix must be square"));
    }

    let n = shape[0];
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            if i == j {
                for k in 0..j {
                    sum += l[j * n + k] * l[j * n + k];
                }
                let val = data[[j, j]] - sum;
                if val < 0.0 {
                    return Err(Error::new(
                        exception::runtime_error(),
                        "Matrix is not positive definite",
                    ));
                }
                l[j * n + j] = val.sqrt();
            } else {
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if l[j * n + j].abs() < 1e-10 {
                    return Err(Error::new(
                        exception::runtime_error(),
                        "Matrix is not positive definite",
                    ));
                }
                l[i * n + j] = (data[[i, j]] - sum) / l[j * n + j];
            }
        }
    }

    Ok(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), l).unwrap(),
    )))
}

/// LU decomposition with partial pivoting
/// Returns (P, L, U) where PA = LU
pub fn lu(arr: &NDArray) -> Result<RArray, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "lu requires 2D array"));
    }

    let shape = data.shape();
    let n = shape[0];
    let m = shape[1];

    // Initialize P as identity, L as zeros (will fill lower triangular), U as copy of A
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        p[i * n + i] = 1.0;
    }
    let mut l = vec![0.0; n * n];
    let mut u: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[[i, j]]).collect())
        .collect();

    // Track row permutations
    let mut perm: Vec<usize> = (0..n).collect();

    for col in 0..n.min(m) {
        // Find pivot (largest absolute value in column)
        let mut max_row = col;
        let mut max_val = u[col][col].abs();
        for row in (col + 1)..n {
            if u[row][col].abs() > max_val {
                max_row = row;
                max_val = u[row][col].abs();
            }
        }

        // Swap rows in U and track in perm
        if max_row != col {
            u.swap(col, max_row);
            perm.swap(col, max_row);
            // Also swap L rows (for columns already processed)
            for j in 0..col {
                let tmp = l[col * n + j];
                l[col * n + j] = l[max_row * n + j];
                l[max_row * n + j] = tmp;
            }
        }

        // Set L diagonal
        l[col * n + col] = 1.0;

        if u[col][col].abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = u[row][col] / u[col][col];
            l[row * n + col] = factor;
            for j in col..m {
                u[row][j] -= factor * u[col][j];
            }
        }
    }

    // Build P matrix from permutation
    let mut p_matrix = vec![0.0; n * n];
    for i in 0..n {
        p_matrix[i * n + perm[i]] = 1.0;
    }

    let u_flat: Vec<f64> = u.into_iter().flatten().collect();

    let result = RArray::new();
    // Return (P, L, U)
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), p_matrix).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), l).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, m]), u_flat).unwrap(),
    )))?;

    Ok(result)
}

/// Solve linear system Ax = b
/// Checks for singular matrices before attempting to solve
pub fn solve(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = a.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "solve requires 2D array for A"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "Matrix A must be square"));
    }

    // Check if matrix is singular by computing determinant
    let d = det(a)?;
    if d.abs() < 1e-14 {
        return Err(Error::new(exception::runtime_error(), "Matrix is singular and cannot be solved"));
    }

    let a_inv = inv(a)?;
    matmul(&a_inv, b)
}

/// Least squares solution - returns (x, residuals, rank, singular_values)
/// For compatibility, we return an RArray with 4 elements
/// Note: singular_values is approximated since we don't have full SVD
pub fn lstsq(a: &NDArray, b: &NDArray) -> Result<RArray, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    if a_data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "lstsq requires 2D array for A"));
    }

    let shape = a_data.shape();
    let _m = shape[0];
    let _n = shape[1];

    // Compute solution using pseudo-inverse
    let a_pinv = pinv(a)?;
    let x = matmul(&a_pinv, b)?;

    // Compute residuals: ||b - Ax||^2
    let ax = matmul(a, &x)?;
    let ax_data = ax.get_data();
    let residuals: f64 = b_data.iter()
        .zip(ax_data.iter())
        .map(|(&bi, &axi)| (bi - axi).powi(2))
        .sum();
    let residuals_arr = Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![residuals]).unwrap()
    ));

    // Compute rank
    let r = rank(a)?;

    // Approximate singular values using eigenvalues of A^T A
    // sqrt of eigenvalues of A^T A would be singular values
    // Since we don't have eig, we'll return an empty array for singular values
    let s = Obj::wrap(NDArray::new(ArrayD::zeros(IxDyn(&[0]))));

    let result = RArray::new();
    result.push(x)?;  // x is already Obj<NDArray>
    result.push(residuals_arr)?;
    result.push(r)?;
    result.push(s)?;

    Ok(result)
}

/// Matrix power (raise square matrix to integer power)
pub fn matrix_power(arr: &NDArray, n: i64) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "matrix_power requires 2D array"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "Matrix must be square"));
    }

    let size = shape[0];

    // Handle special cases
    if n == 0 {
        // Return identity matrix
        let mut result = vec![0.0; size * size];
        for i in 0..size {
            result[i * size + i] = 1.0;
        }
        return Ok(Obj::wrap(NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[size, size]), result).unwrap(),
        )));
    }

    if n == 1 {
        return Ok(Obj::wrap(NDArray::new(data.clone())));
    }

    if n < 0 {
        // For negative powers, compute inverse first
        let arr_inv = inv(arr)?;
        return matrix_power(&arr_inv, -n);
    }

    // For positive n, use repeated squaring (exponentiation by squaring)
    let mut result_data = vec![0.0; size * size];
    for i in 0..size {
        result_data[i * size + i] = 1.0;
    }
    let mut result = Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[size, size]), result_data).unwrap(),
    ));

    let mut base = Obj::wrap(NDArray::new(data.clone()));
    let mut exp = n;

    while exp > 0 {
        if exp % 2 == 1 {
            result = matmul(&result, &base)?;
        }
        base = matmul(&base, &base)?;
        exp /= 2;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_2x2() {
        // [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let d = det(&arr).unwrap();
        assert!((d - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]).unwrap(),
        );
        let t = trace(&arr).unwrap();
        assert!((t - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_2x2() {
        // [[4, 7], [2, 6]] -> inv = [[0.6, -0.7], [-0.2, 0.4]]
        let arr = NDArray::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![4.0, 7.0, 2.0, 6.0]).unwrap(),
        );
        let inverse = inv(&arr).unwrap();
        let data = inverse.get_data();

        assert!((data[[0, 0]] - 0.6).abs() < 1e-10);
        assert!((data[[0, 1]] - (-0.7)).abs() < 1e-10);
        assert!((data[[1, 0]] - (-0.2)).abs() < 1e-10);
        assert!((data[[1, 1]] - 0.4).abs() < 1e-10);
    }
}
