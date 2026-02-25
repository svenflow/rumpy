# frozen_string_literal: true

require "spec_helper"

RSpec.describe "RumPy math functions" do
  describe "trigonometric functions" do
    it "computes sin" do
      arr = RumPy.array([0, Math::PI / 2, Math::PI])
      result = RumPy.sin(arr)
      expect(result[0]).to be_within(1e-10).of(0.0)
      expect(result[1]).to be_within(1e-10).of(1.0)
      expect(result[2]).to be_within(1e-10).of(0.0)
    end

    it "computes cos" do
      arr = RumPy.array([0, Math::PI / 2, Math::PI])
      result = RumPy.cos(arr)
      expect(result[0]).to be_within(1e-10).of(1.0)
      expect(result[1]).to be_within(1e-10).of(0.0)
      expect(result[2]).to be_within(1e-10).of(-1.0)
    end

    it "computes tan" do
      arr = RumPy.array([0, Math::PI / 4])
      result = RumPy.tan(arr)
      expect(result[0]).to be_within(1e-10).of(0.0)
      expect(result[1]).to be_within(1e-10).of(1.0)
    end
  end

  describe "exponential and logarithmic" do
    it "computes exp" do
      arr = RumPy.array([0, 1, 2])
      result = RumPy.exp(arr)
      expect(result[0]).to be_within(1e-10).of(1.0)
      expect(result[1]).to be_within(1e-10).of(Math::E)
      expect(result[2]).to be_within(1e-10).of(Math::E ** 2)
    end

    it "computes log (natural)" do
      arr = RumPy.array([1, Math::E, Math::E ** 2])
      result = RumPy.log(arr)
      expect(result[0]).to be_within(1e-10).of(0.0)
      expect(result[1]).to be_within(1e-10).of(1.0)
      expect(result[2]).to be_within(1e-10).of(2.0)
    end

    it "computes log10" do
      arr = RumPy.array([1, 10, 100])
      result = RumPy.log10(arr)
      expect(result[0]).to be_within(1e-10).of(0.0)
      expect(result[1]).to be_within(1e-10).of(1.0)
      expect(result[2]).to be_within(1e-10).of(2.0)
    end

    it "computes sqrt" do
      arr = RumPy.array([0, 1, 4, 9, 16])
      result = RumPy.sqrt(arr)
      expect(result.to_a).to eq([0.0, 1.0, 2.0, 3.0, 4.0])
    end
  end

  describe "rounding" do
    it "computes floor" do
      arr = RumPy.array([1.7, 2.3, -0.5])
      result = RumPy.floor(arr)
      expect(result.to_a).to eq([1.0, 2.0, -1.0])
    end

    it "computes ceil" do
      arr = RumPy.array([1.1, 2.9, -0.5])
      result = RumPy.ceil(arr)
      expect(result.to_a).to eq([2.0, 3.0, 0.0])
    end

    it "computes round" do
      arr = RumPy.array([1.4, 1.5, 1.6])
      result = RumPy.round(arr)
      expect(result.to_a).to eq([1.0, 2.0, 2.0])
    end
  end

  describe "clip" do
    it "clips values to range" do
      arr = RumPy.array([1, 5, 10, 15, 20])
      result = RumPy.clip(arr, 5.0, 15.0)
      expect(result.to_a).to eq([5.0, 5.0, 10.0, 15.0, 15.0])
    end
  end

  describe "abs and sign" do
    it "computes absolute value" do
      arr = RumPy.array([-2, -1, 0, 1, 2])
      result = RumPy.abs(arr)
      expect(result.to_a).to eq([2.0, 1.0, 0.0, 1.0, 2.0])
    end

    it "computes sign" do
      arr = RumPy.array([-5, 0, 5])
      result = RumPy.sign(arr)
      expect(result.to_a).to eq([-1.0, 0.0, 1.0])
    end
  end

  describe "matrix operations" do
    it "computes dot product of 1D arrays" do
      a = RumPy.array([1, 2, 3])
      b = RumPy.array([4, 5, 6])
      result = RumPy.dot(a, b)
      # NumPy returns a scalar for 1D dot product
      expect(result.to_a).to eq(32.0)  # 1*4 + 2*5 + 3*6 = 32
    end

    it "computes matrix multiplication" do
      a = RumPy.array([[1, 2], [3, 4]])
      b = RumPy.array([[5, 6], [7, 8]])
      result = RumPy.matmul(a, b)
      expect(result.shape).to eq([2, 2])
      expect(result[[0, 0]]).to eq(19.0)  # 1*5 + 2*7
      expect(result[[1, 1]]).to eq(50.0)  # 3*6 + 4*8
    end

    it "computes outer product" do
      a = RumPy.array([1, 2, 3])
      b = RumPy.array([4, 5])
      result = RumPy.outer(a, b)
      expect(result.shape).to eq([3, 2])
      expect(result[[0, 0]]).to eq(4.0)
      expect(result[[2, 1]]).to eq(15.0)
    end

    it "computes cross product" do
      a = RumPy.array([1, 0, 0])
      b = RumPy.array([0, 1, 0])
      result = RumPy.cross(a, b)
      expect(result.to_a).to eq([0.0, 0.0, 1.0])
    end
  end

  describe "logical operations" do
    it "computes logical_and" do
      a = RumPy.array([1, 0, 1, 0])
      b = RumPy.array([1, 1, 0, 0])
      result = RumPy.logical_and(a, b)
      expect(result.to_a).to eq([1.0, 0.0, 0.0, 0.0])
    end

    it "computes logical_or" do
      a = RumPy.array([1, 0, 1, 0])
      b = RumPy.array([1, 1, 0, 0])
      result = RumPy.logical_or(a, b)
      expect(result.to_a).to eq([1.0, 1.0, 1.0, 0.0])
    end

    it "computes logical_not" do
      arr = RumPy.array([1, 0, 1, 0])
      result = RumPy.logical_not(arr)
      expect(result.to_a).to eq([0.0, 1.0, 0.0, 1.0])
    end

    it "computes where" do
      cond = RumPy.array([1, 0, 1, 0])
      x = RumPy.array([1, 2, 3, 4])
      y = RumPy.array([5, 6, 7, 8])
      result = RumPy.where(cond, x, y)
      expect(result.to_a).to eq([1.0, 6.0, 3.0, 8.0])
    end
  end
end
