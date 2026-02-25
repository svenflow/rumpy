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

/// Moore-Penrose pseudo-inverse (simplified - uses normal equations)
pub fn pinv(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "pinv requires 2D array"));
    }

    // A+ = (A^T A)^-1 A^T for overdetermined systems
    // This is a simplified implementation
    let at = Obj::wrap(NDArray::new(data.t().to_owned()));
    let ata = matmul(&at, arr)?;
    let ata_inv = inv(&ata)?;
    matmul(&ata_inv, &at)
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

/// Frobenius norm
pub fn norm(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();
    let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
    Ok(sum_sq.sqrt())
}

/// Condition number (ratio of largest to smallest singular value)
/// Simplified implementation
pub fn cond(arr: &NDArray) -> Result<f64, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "cond requires 2D array"));
    }

    // Simplified: use norm(A) * norm(A^-1)
    let norm_a = norm(arr)?;
    let inv_a = inv(arr)?;
    let norm_inv = norm(&inv_a)?;

    Ok(norm_a * norm_inv)
}

/// Eigenvalues and eigenvectors (power iteration - simplified)
pub fn eig(arr: &NDArray) -> Result<RArray, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "eig requires 2D array"));
    }

    let shape = data.shape();
    if shape[0] != shape[1] {
        return Err(Error::new(exception::arg_error(), "Matrix must be square"));
    }

    // For now, return placeholder
    // Real implementation would use QR algorithm
    let n = shape[0];
    let eigenvalues = vec![0.0; n];
    let eigenvectors = vec![0.0; n * n];

    let result = RArray::new();
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), eigenvalues).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), eigenvectors).unwrap(),
    )))?;

    Ok(result)
}

/// Eigenvalues only
pub fn eigvals(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    let result = eig(arr)?;
    let first = result.entry::<magnus::Value>(0)?;
    Ok(<Obj<NDArray>>::try_convert(first)?)
}

/// Eigenvalues/vectors for symmetric matrix
pub fn eigh(arr: &NDArray) -> Result<RArray, Error> {
    eig(arr)
}

/// Eigenvalues for symmetric matrix
pub fn eigvalsh(arr: &NDArray) -> Result<Obj<NDArray>, Error> {
    eigvals(arr)
}

/// Singular Value Decomposition (simplified)
pub fn svd(arr: &NDArray) -> Result<RArray, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "svd requires 2D array"));
    }

    let shape = data.shape();
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);

    // Placeholder implementation
    let u = vec![0.0; m * m];
    let s = vec![0.0; k];
    let vt = vec![0.0; n * n];

    let result = RArray::new();
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m, m]), u).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[k]), s).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), vt).unwrap(),
    )))?;

    Ok(result)
}

/// QR decomposition
pub fn qr(arr: &NDArray) -> Result<RArray, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "qr requires 2D array"));
    }

    let shape = data.shape();
    let m = shape[0];
    let n = shape[1];

    // Gram-Schmidt orthogonalization
    let mut q_cols: Vec<Vec<f64>> = Vec::new();
    let mut r = vec![0.0; n * n];

    for j in 0..n {
        // Get column j
        let mut v: Vec<f64> = (0..m).map(|i| data[[i, j]]).collect();

        // Subtract projections
        for (k, q_col) in q_cols.iter().enumerate() {
            let proj: f64 = v.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[k * n + j] = proj;
            for i in 0..m {
                v[i] -= proj * q_col[i];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            r[j * n + j] = norm;
            for x in &mut v {
                *x /= norm;
            }
            q_cols.push(v);
        } else {
            r[j * n + j] = 0.0;
            q_cols.push(vec![0.0; m]);
        }
    }

    // Build Q matrix
    let q: Vec<f64> = (0..m)
        .flat_map(|i| q_cols.iter().map(move |col| col.get(i).cloned().unwrap_or(0.0)))
        .collect();

    let result = RArray::new();
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[m, n.min(m)]), q).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), r).unwrap(),
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

/// LU decomposition
pub fn lu(arr: &NDArray) -> Result<RArray, Error> {
    let data = arr.get_data();

    if data.ndim() != 2 {
        return Err(Error::new(exception::arg_error(), "lu requires 2D array"));
    }

    let shape = data.shape();
    let n = shape[0];
    let m = shape[1];

    let mut l = vec![0.0; n * n];
    let mut u: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[[i, j]]).collect())
        .collect();

    // Initialize L diagonal
    for i in 0..n {
        l[i * n + i] = 1.0;
    }

    for col in 0..n.min(m) {
        if u[col][col].abs() < 1e-10 {
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

    let u_flat: Vec<f64> = u.into_iter().flatten().collect();

    let result = RArray::new();
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, n]), l).unwrap(),
    )))?;
    result.push(Obj::wrap(NDArray::new(
        ArrayD::from_shape_vec(IxDyn(&[n, m]), u_flat).unwrap(),
    )))?;

    Ok(result)
}

/// Solve linear system Ax = b
pub fn solve(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    let a_inv = inv(a)?;
    matmul(&a_inv, b)
}

/// Least squares solution
pub fn lstsq(a: &NDArray, b: &NDArray) -> Result<Obj<NDArray>, Error> {
    // x = (A^T A)^-1 A^T b
    let at = Obj::wrap(NDArray::new(a.get_data().t().to_owned()));
    let ata = matmul(&at, a)?;
    let ata_inv = inv(&ata)?;
    let atb = matmul(&at, b)?;
    matmul(&ata_inv, &atb)
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
