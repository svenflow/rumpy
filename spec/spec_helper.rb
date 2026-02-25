# frozen_string_literal: true

require "rumpy"

RSpec.configure do |config|
  config.expect_with :rspec do |expectations|
    expectations.include_chain_clauses_in_custom_matcher_descriptions = true
  end

  config.mock_with :rspec do |mocks|
    mocks.verify_partial_doubles = true
  end

  config.shared_context_metadata_behavior = :apply_to_host_groups
end

# Custom matchers for array comparisons
RSpec::Matchers.define :be_close_to do |expected, tolerance|
  tolerance ||= 1e-10

  match do |actual|
    if actual.is_a?(RumPy::NDArray) && expected.is_a?(RumPy::NDArray)
      actual_arr = actual.to_a.flatten
      expected_arr = expected.to_a.flatten
      actual_arr.zip(expected_arr).all? { |a, e| (a - e).abs < tolerance }
    elsif actual.is_a?(Numeric) && expected.is_a?(Numeric)
      (actual - expected).abs < tolerance
    else
      false
    end
  end
end
