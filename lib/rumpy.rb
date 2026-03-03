# frozen_string_literal: true

require_relative "rumpy/rumpy"
require_relative "rumpy/version"

module RumPy
  class Error < StandardError; end

  # Convenience aliases matching numpy conventions
  class << self
    # Array creation shortcuts
    def asarray(obj)
      array(obj)
    end

    def empty_like(arr)
      zeros_like(arr)
    end

    def full_like(arr, value)
      arr = array(arr) unless arr.is_a?(NDArray)
      full(arr.shape, value)
    end

    # Axis-aware aggregations with proper axis/keepdims support
    def sum(arr, axis: nil, keepdims: false)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        arr.sum
      else
        result = sum_axis(arr, axis)
        keepdims ? _keepdims_reshape(result, arr.shape, axis) : result
      end
    end

    def mean(arr, axis: nil, keepdims: false)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        arr.mean
      else
        result = mean_axis(arr, axis)
        keepdims ? _keepdims_reshape(result, arr.shape, axis) : result
      end
    end

    def std(arr, axis: nil, ddof: 0, keepdims: false)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        std_ddof(arr, ddof)
      else
        result = std_axis(arr, axis)
        keepdims ? _keepdims_reshape(result, arr.shape, axis) : result
      end
    end

    def var(arr, axis: nil, ddof: 0, keepdims: false)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        var_ddof(arr, ddof)
      else
        result = var_axis(arr, axis)
        keepdims ? _keepdims_reshape(result, arr.shape, axis) : result
      end
    end

    def min(arr, axis: nil, keepdims: false)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        arr.min
      else
        result = min_axis(arr, axis)
        keepdims ? _keepdims_reshape(result, arr.shape, axis) : result
      end
    end

    def max(arr, axis: nil, keepdims: false)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        arr.max
      else
        result = max_axis(arr, axis)
        keepdims ? _keepdims_reshape(result, arr.shape, axis) : result
      end
    end

    def argmin(arr, axis: nil)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        arr.argmin
      else
        argmin_axis(arr, axis)
      end
    end

    def argmax(arr, axis: nil)
      arr = array(arr) unless arr.is_a?(NDArray)
      if axis.nil?
        arr.argmax
      else
        argmax_axis(arr, axis)
      end
    end

    # Helper to restore reduced dimension for keepdims
    def _keepdims_reshape(result, original_shape, axis)
      return result unless result.is_a?(NDArray)
      new_shape = original_shape.dup
      new_shape[axis] = 1
      result.reshape(new_shape)
    end

    # Reshape convenience
    def reshape(arr, shape)
      arr.reshape(shape)
    end

    # Transpose convenience
    def transpose(arr)
      arr.transpose
    end

    alias_method :T, :transpose
  end

  # Linalg module aliases
  module Linalg
    class << self
      # Common aliases
      alias_method :matrix_rank, :rank
    end
  end

  # Random module with Generator class
  module Random
    class Generator
      def initialize(seed = nil)
        Random.seed(seed) if seed
      end

      def random(shape)
        Random.rand(shape)
      end

      def integers(low, high, shape)
        Random.randint(low, high, shape)
      end

      def standard_normal(shape)
        Random.randn(shape)
      end
    end

    class << self
      def default_rng(seed = nil)
        Generator.new(seed)
      end
    end
  end
end

# Top-level alias for numpy-like usage
NP = RumPy
