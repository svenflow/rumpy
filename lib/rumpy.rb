# frozen_string_literal: true

require_relative "rumpy/rumpy"
require_relative "rumpy/version"

module RumPy
  class Error < StandardError; end

  # Convenience aliases matching numpy conventions
  class << self
    alias_method :np, :RumPy

    # Array creation shortcuts
    def asarray(obj)
      array(obj)
    end

    def empty_like(arr)
      zeros_like(arr)
    end

    def full_like(arr, value)
      result = zeros_like(arr)
      # Would need to implement fill
      result
    end

    # Axis-aware aggregations (simplified - ignores axis for now)
    def sum(arr, axis: nil, keepdims: false)
      if arr.is_a?(NDArray)
        arr.sum
      else
        array(arr).sum
      end
    end

    def mean(arr, axis: nil, keepdims: false)
      if arr.is_a?(NDArray)
        arr.mean
      else
        array(arr).mean
      end
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
      alias_method :matrix_power, :matmul
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
