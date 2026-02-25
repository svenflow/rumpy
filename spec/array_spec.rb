# frozen_string_literal: true

require "spec_helper"

RSpec.describe RumPy::NDArray do
  describe "array creation" do
    it "creates array from nested Ruby arrays" do
      arr = RumPy.array([[1, 2, 3], [4, 5, 6]])
      expect(arr.shape).to eq([2, 3])
      expect(arr.ndim).to eq(2)
      expect(arr.size).to eq(6)
    end

    it "creates 1D array from flat array" do
      arr = RumPy.array([1, 2, 3, 4, 5])
      expect(arr.shape).to eq([5])
      expect(arr.ndim).to eq(1)
    end

    it "creates array from integers" do
      arr = RumPy.array([1, 2, 3])
      expect(arr[0]).to eq(1.0)
    end
  end

  describe "zeros and ones" do
    it "creates zeros array" do
      arr = RumPy.zeros([3, 4])
      expect(arr.shape).to eq([3, 4])
      expect(arr.sum).to eq(0.0)
    end

    it "creates ones array" do
      arr = RumPy.ones([2, 3])
      expect(arr.shape).to eq([2, 3])
      expect(arr.sum).to eq(6.0)
    end

    it "creates full array" do
      arr = RumPy.full([2, 2], 7.0)
      expect(arr.sum).to eq(28.0)
    end
  end

  describe "arange and linspace" do
    it "creates arange" do
      arr = RumPy.arange(0.0, 5.0, 1.0)
      expect(arr.to_a).to eq([0.0, 1.0, 2.0, 3.0, 4.0])
    end

    it "creates linspace" do
      arr = RumPy.linspace(0.0, 10.0, 5)
      expect(arr.to_a).to eq([0.0, 2.5, 5.0, 7.5, 10.0])
    end

    it "creates linspace with single point" do
      arr = RumPy.linspace(5.0, 5.0, 1)
      expect(arr.to_a).to eq([5.0])
    end
  end

  describe "eye and identity" do
    it "creates identity matrix" do
      arr = RumPy.eye(3)
      expect(arr.shape).to eq([3, 3])
      expect(arr[0, 0]).to eq(1.0)
      expect(arr[0, 1]).to eq(0.0)
      expect(arr[1, 1]).to eq(1.0)
    end
  end

  describe "indexing" do
    let(:arr) { RumPy.array([[1, 2, 3], [4, 5, 6]]) }

    it "gets element by index" do
      expect(arr[[0, 0]]).to eq(1.0)
      expect(arr[[1, 2]]).to eq(6.0)
    end

    it "sets element by index" do
      arr[[0, 0]] = 99.0
      expect(arr[[0, 0]]).to eq(99.0)
    end

    it "handles negative indices" do
      arr1d = RumPy.array([1, 2, 3, 4, 5])
      expect(arr1d[-1]).to eq(5.0)
    end
  end

  describe "reshaping" do
    it "reshapes array" do
      arr = RumPy.arange(0.0, 6.0, 1.0)
      reshaped = arr.reshape([2, 3])
      expect(reshaped.shape).to eq([2, 3])
      expect(reshaped[[1, 2]]).to eq(5.0)
    end

    it "flattens array" do
      arr = RumPy.array([[1, 2], [3, 4]])
      flat = arr.flatten
      expect(flat.shape).to eq([4])
      expect(flat.to_a).to eq([1.0, 2.0, 3.0, 4.0])
    end

    it "transposes array" do
      arr = RumPy.array([[1, 2, 3], [4, 5, 6]])
      t = arr.transpose
      expect(t.shape).to eq([3, 2])
      expect(t[[0, 1]]).to eq(4.0)
    end
  end

  describe "arithmetic operations" do
    let(:a) { RumPy.array([1, 2, 3]) }
    let(:b) { RumPy.array([4, 5, 6]) }

    it "adds arrays" do
      result = a + b
      expect(result.to_a).to eq([5.0, 7.0, 9.0])
    end

    it "subtracts arrays" do
      result = b - a
      expect(result.to_a).to eq([3.0, 3.0, 3.0])
    end

    it "multiplies arrays element-wise" do
      result = a * b
      expect(result.to_a).to eq([4.0, 10.0, 18.0])
    end

    it "divides arrays element-wise" do
      result = b / a
      expect(result.to_a).to eq([4.0, 2.5, 2.0])
    end

    it "adds scalar" do
      result = a + 10
      expect(result.to_a).to eq([11.0, 12.0, 13.0])
    end

    it "multiplies by scalar" do
      result = a * 2
      expect(result.to_a).to eq([2.0, 4.0, 6.0])
    end

    it "negates array" do
      result = -a
      expect(result.to_a).to eq([-1.0, -2.0, -3.0])
    end

    it "raises to power" do
      result = a ** 2
      expect(result.to_a).to eq([1.0, 4.0, 9.0])
    end
  end

  describe "comparison operations" do
    let(:arr) { RumPy.array([1, 2, 3, 4, 5]) }

    it "compares greater than" do
      result = arr > 3
      expect(result.to_a).to eq([0.0, 0.0, 0.0, 1.0, 1.0])
    end

    it "compares equal" do
      result = arr == 3
      expect(result.to_a).to eq([0.0, 0.0, 1.0, 0.0, 0.0])
    end
  end

  describe "aggregations" do
    let(:arr) { RumPy.array([1, 2, 3, 4, 5]) }

    it "computes sum" do
      expect(arr.sum).to eq(15.0)
    end

    it "computes mean" do
      expect(arr.mean).to eq(3.0)
    end

    it "computes min/max" do
      expect(arr.min).to eq(1.0)
      expect(arr.max).to eq(5.0)
    end

    it "computes argmin/argmax" do
      expect(arr.argmin).to eq(0)
      expect(arr.argmax).to eq(4)
    end

    it "computes std and var" do
      arr2 = RumPy.array([2, 4, 4, 4, 5, 5, 7, 9])
      expect(arr2.var).to be_within(0.1).of(4.0)
      expect(arr2.std).to be_within(0.1).of(2.0)
    end
  end

  describe "boolean operations" do
    it "checks all" do
      expect(RumPy.array([1, 1, 1]).all).to be true
      expect(RumPy.array([1, 0, 1]).all).to be false
    end

    it "checks any" do
      expect(RumPy.array([0, 0, 1]).any).to be true
      expect(RumPy.array([0, 0, 0]).any).to be false
    end
  end
end
