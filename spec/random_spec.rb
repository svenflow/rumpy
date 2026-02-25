# frozen_string_literal: true

require "spec_helper"

RSpec.describe "RumPy::Random" do
  describe "basic random functions" do
    it "generates uniform random in [0, 1)" do
      arr = RumPy::Random.rand([100])
      expect(arr.min).to be >= 0.0
      expect(arr.max).to be < 1.0
    end

    it "generates standard normal" do
      arr = RumPy::Random.randn([1000])
      # Mean should be close to 0, std close to 1
      expect(arr.mean).to be_within(0.2).of(0.0)
      expect(arr.std).to be_within(0.2).of(1.0)
    end

    it "generates random integers" do
      arr = RumPy::Random.randint(0, 10, [100])
      expect(arr.min).to be >= 0.0
      expect(arr.max).to be < 10.0
      # Should be integers
      arr.to_a.each do |val|
        expect(val).to eq(val.floor)
      end
    end
  end

  describe "distribution sampling" do
    it "generates uniform distribution" do
      arr = RumPy::Random.uniform(5.0, 10.0, [100])
      expect(arr.min).to be >= 5.0
      expect(arr.max).to be < 10.0
    end

    it "generates normal distribution" do
      arr = RumPy::Random.normal(100.0, 15.0, [1000])
      expect(arr.mean).to be_within(5.0).of(100.0)
      expect(arr.std).to be_within(5.0).of(15.0)
    end

    it "generates binomial distribution" do
      arr = RumPy::Random.binomial(10, 0.5, [100])
      # Should be integers between 0 and 10
      expect(arr.min).to be >= 0.0
      expect(arr.max).to be <= 10.0
    end

    it "generates poisson distribution" do
      arr = RumPy::Random.poisson(5.0, [1000])
      expect(arr.min).to be >= 0.0
      expect(arr.mean).to be_within(1.0).of(5.0)
    end

    it "generates exponential distribution" do
      arr = RumPy::Random.exponential(2.0, [1000])
      expect(arr.min).to be >= 0.0
      expect(arr.mean).to be_within(1.0).of(2.0)
    end
  end

  describe "permutations and choices" do
    it "shuffles array" do
      arr = RumPy.array([1, 2, 3, 4, 5])
      shuffled = RumPy::Random.shuffle(arr)
      # Same elements, different order (usually)
      expect(shuffled.sum).to eq(arr.sum)
    end

    it "generates permutation" do
      perm = RumPy::Random.permutation(5)
      sorted = perm.to_a.sort
      expect(sorted).to eq([0.0, 1.0, 2.0, 3.0, 4.0])
    end

    it "makes random choices" do
      arr = RumPy.array([10, 20, 30, 40, 50])
      choices = RumPy::Random.choice(arr, 100)
      expect(choices.size).to eq(100)
      # All choices should be from original array
      choices.to_a.each do |val|
        expect([10.0, 20.0, 30.0, 40.0, 50.0]).to include(val)
      end
    end
  end

  describe "seed reproducibility" do
    it "produces same sequence with same seed" do
      RumPy::Random.seed(42)
      first = RumPy::Random.rand([5]).to_a

      RumPy::Random.seed(42)
      second = RumPy::Random.rand([5]).to_a

      expect(first).to eq(second)
    end

    it "produces different sequences with different seeds" do
      RumPy::Random.seed(42)
      first = RumPy::Random.rand([5]).to_a

      RumPy::Random.seed(43)
      second = RumPy::Random.rand([5]).to_a

      expect(first).not_to eq(second)
    end
  end

  describe "Generator class" do
    it "creates generator with seed" do
      gen = RumPy::Random.default_rng(42)
      arr = gen.random([5])
      expect(arr.size).to eq(5)
    end
  end
end
