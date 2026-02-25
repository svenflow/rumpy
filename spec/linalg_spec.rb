# frozen_string_literal: true

require "spec_helper"

RSpec.describe "RumPy::Linalg" do
  describe "det" do
    it "computes determinant of 2x2 matrix" do
      # [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
      arr = RumPy.array([[1, 2], [3, 4]])
      expect(RumPy::Linalg.det(arr)).to be_within(1e-10).of(-2.0)
    end

    it "computes determinant of 3x3 matrix" do
      # Identity matrix has det = 1
      arr = RumPy.eye(3)
      expect(RumPy::Linalg.det(arr)).to be_within(1e-10).of(1.0)
    end

    it "returns 0 for singular matrix" do
      arr = RumPy.array([[1, 2], [2, 4]])
      expect(RumPy::Linalg.det(arr)).to be_within(1e-10).of(0.0)
    end
  end

  describe "trace" do
    it "computes trace of matrix" do
      arr = RumPy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      expect(RumPy::Linalg.trace(arr)).to eq(15.0)  # 1 + 5 + 9
    end
  end

  describe "inv" do
    it "computes inverse of 2x2 matrix" do
      arr = RumPy.array([[4, 7], [2, 6]])
      inv = RumPy::Linalg.inv(arr)

      # Check A * A^-1 = I
      product = RumPy.matmul(arr, inv)
      expect(product[[0, 0]]).to be_within(1e-10).of(1.0)
      expect(product[[0, 1]]).to be_within(1e-10).of(0.0)
      expect(product[[1, 0]]).to be_within(1e-10).of(0.0)
      expect(product[[1, 1]]).to be_within(1e-10).of(1.0)
    end

    it "computes inverse of identity" do
      arr = RumPy.eye(3)
      inv = RumPy::Linalg.inv(arr)
      # Inverse of identity is identity
      expect(inv[[0, 0]]).to be_within(1e-10).of(1.0)
      expect(inv[[1, 1]]).to be_within(1e-10).of(1.0)
      expect(inv[[0, 1]]).to be_within(1e-10).of(0.0)
    end

    it "raises error for singular matrix" do
      arr = RumPy.array([[1, 2], [2, 4]])
      expect { RumPy::Linalg.inv(arr) }.to raise_error(RuntimeError, /singular/i)
    end
  end

  describe "norm" do
    it "computes Frobenius norm" do
      arr = RumPy.array([[1, 2], [3, 4]])
      # sqrt(1 + 4 + 9 + 16) = sqrt(30)
      expect(RumPy::Linalg.norm(arr)).to be_within(1e-10).of(Math.sqrt(30))
    end
  end

  describe "rank" do
    it "computes rank of full rank matrix" do
      arr = RumPy.eye(3)
      expect(RumPy::Linalg.rank(arr)).to eq(3)
    end

    it "computes rank of rank-deficient matrix" do
      arr = RumPy.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
      # Row 2 is 2 * Row 1, so rank should be 2
      expect(RumPy::Linalg.rank(arr)).to eq(2)
    end
  end

  describe "solve" do
    it "solves linear system" do
      # Solve 2x + y = 5, x + 3y = 5
      # Solution: x = 2, y = 1
      a = RumPy.array([[2, 1], [1, 3]])
      b = RumPy.array([[5], [5]])
      x = RumPy::Linalg.solve(a, b)

      expect(x[[0, 0]]).to be_within(1e-10).of(2.0)
      expect(x[[1, 0]]).to be_within(1e-10).of(1.0)
    end
  end

  describe "cholesky" do
    it "computes Cholesky decomposition" do
      # Positive definite matrix
      arr = RumPy.array([[4, 2], [2, 5]])
      l = RumPy::Linalg.cholesky(arr)

      # L * L^T should equal original
      lt = l.transpose
      product = RumPy.matmul(l, lt)
      expect(product[[0, 0]]).to be_within(1e-10).of(4.0)
      expect(product[[0, 1]]).to be_within(1e-10).of(2.0)
    end

    it "raises error for non-positive-definite matrix" do
      arr = RumPy.array([[1, 2], [2, 1]])  # Not positive definite
      expect { RumPy::Linalg.cholesky(arr) }.to raise_error(RuntimeError)
    end
  end

  describe "qr" do
    it "computes QR decomposition" do
      arr = RumPy.array([[1, 2], [3, 4], [5, 6]])
      q, r = RumPy::Linalg.qr(arr)

      # Q should be orthogonal (Q^T * Q = I)
      # R should be upper triangular
      expect(q.shape[0]).to eq(3)
      expect(r[[1, 0]]).to be_within(1e-10).of(0.0)  # Below diagonal should be 0
    end
  end

  describe "lu" do
    it "computes LU decomposition" do
      arr = RumPy.array([[2, 1], [4, 3]])
      l, u = RumPy::Linalg.lu(arr)

      # L should be lower triangular with 1s on diagonal
      expect(l[[0, 0]]).to be_within(1e-10).of(1.0)
      expect(l[[1, 1]]).to be_within(1e-10).of(1.0)

      # L * U should equal original
      product = RumPy.matmul(l, u)
      expect(product[[0, 0]]).to be_within(1e-10).of(2.0)
      expect(product[[1, 1]]).to be_within(1e-10).of(3.0)
    end
  end
end
