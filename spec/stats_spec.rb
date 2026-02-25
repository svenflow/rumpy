# frozen_string_literal: true

require "spec_helper"

RSpec.describe "RumPy statistics" do
  describe "basic statistics" do
    let(:arr) { RumPy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) }

    it "computes sum" do
      expect(RumPy.sum(arr)).to eq(55.0)
    end

    it "computes prod" do
      small = RumPy.array([1, 2, 3, 4])
      expect(RumPy.prod(small)).to eq(24.0)
    end

    it "computes mean" do
      expect(RumPy.mean(arr)).to eq(5.5)
    end

    it "computes min" do
      expect(RumPy.min(arr)).to eq(1.0)
    end

    it "computes max" do
      expect(RumPy.max(arr)).to eq(10.0)
    end
  end

  describe "variance and standard deviation" do
    it "computes variance" do
      arr = RumPy.array([2, 4, 4, 4, 5, 5, 7, 9])
      # Population variance: mean = 5, variance = 4
      expect(RumPy.var(arr)).to be_within(0.01).of(4.0)
    end

    it "computes standard deviation" do
      arr = RumPy.array([2, 4, 4, 4, 5, 5, 7, 9])
      expect(RumPy.std(arr)).to be_within(0.01).of(2.0)
    end
  end

  describe "median and percentiles" do
    it "computes median of odd-length array" do
      arr = RumPy.array([1, 3, 2, 5, 4])
      expect(RumPy.median(arr)).to eq(3.0)
    end

    it "computes median of even-length array" do
      arr = RumPy.array([1, 2, 3, 4])
      expect(RumPy.median(arr)).to eq(2.5)
    end

    it "computes percentile" do
      arr = RumPy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      expect(RumPy.percentile(arr, 50.0)).to be_within(0.1).of(5.5)
      expect(RumPy.percentile(arr, 0.0)).to eq(1.0)
      expect(RumPy.percentile(arr, 100.0)).to eq(10.0)
    end

    it "computes quantile" do
      arr = RumPy.array([1, 2, 3, 4, 5])
      expect(RumPy.quantile(arr, 0.5)).to eq(3.0)
      expect(RumPy.quantile(arr, 0.0)).to eq(1.0)
      expect(RumPy.quantile(arr, 1.0)).to eq(5.0)
    end
  end

  describe "cumulative operations" do
    it "computes cumsum" do
      arr = RumPy.array([1, 2, 3, 4])
      result = RumPy.cumsum(arr)
      expect(result.to_a).to eq([1.0, 3.0, 6.0, 10.0])
    end

    it "computes cumprod" do
      arr = RumPy.array([1, 2, 3, 4])
      result = RumPy.cumprod(arr)
      expect(result.to_a).to eq([1.0, 2.0, 6.0, 24.0])
    end
  end

  describe "argmin and argmax" do
    it "finds argmin" do
      arr = RumPy.array([3, 1, 4, 1, 5])
      expect(RumPy.argmin(arr)).to eq(1)  # First occurrence of 1
    end

    it "finds argmax" do
      arr = RumPy.array([3, 1, 4, 1, 5])
      expect(RumPy.argmax(arr)).to eq(4)  # Index of 5
    end
  end

  describe "histogram" do
    it "computes histogram" do
      arr = RumPy.array([1, 2, 1, 3, 2, 2, 3, 3, 3])
      counts, edges = RumPy.histogram(arr, 3)

      expect(counts.shape[0]).to eq(3)
      expect(edges.shape[0]).to eq(4)  # bins + 1 edges
    end
  end

  describe "covariance and correlation" do
    it "computes covariance matrix" do
      # 2 variables, 4 observations
      arr = RumPy.array([[1, 2, 3, 4], [4, 3, 2, 1]])
      cov = RumPy.cov(arr)

      expect(cov.shape).to eq([2, 2])
      # Variance should be positive
      expect(cov[[0, 0]]).to be > 0
      # Off-diagonal should be negative (variables are inversely related)
      expect(cov[[0, 1]]).to be < 0
    end

    it "computes correlation coefficient" do
      arr = RumPy.array([[1, 2, 3, 4], [4, 3, 2, 1]])
      corr = RumPy.corrcoef(arr)

      expect(corr.shape).to eq([2, 2])
      # Diagonal should be 1
      expect(corr[[0, 0]]).to be_within(1e-10).of(1.0)
      expect(corr[[1, 1]]).to be_within(1e-10).of(1.0)
      # Off-diagonal should be -1 (perfect negative correlation)
      expect(corr[[0, 1]]).to be_within(1e-10).of(-1.0)
    end
  end
end
