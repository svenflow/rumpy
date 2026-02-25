# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name = "rumpy"
  spec.version = "0.1.0"
  spec.authors = ["Sam McGrail"]
  spec.email = ["sam@example.com"]

  spec.summary = "NumPy for Ruby, powered by Rust"
  spec.description = "A Ruby gem providing NumPy-like array operations, implemented in Rust for performance using ndarray and Magnus."
  spec.homepage = "https://github.com/sammcgrail/RumPy"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.0.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) ||
        f.start_with?(*%w[bin/ test/ spec/ features/ .git .github appveyor Gemfile])
    end
  end

  spec.require_paths = ["lib"]
  spec.extensions = ["ext/rumpy/Cargo.toml"]

  spec.add_dependency "rb_sys", "~> 0.9"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "rb_sys", "~> 0.9"
  spec.add_development_dependency "rspec", "~> 3.0"
end
