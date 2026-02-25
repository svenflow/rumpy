# frozen_string_literal: true

require "bundler/gem_tasks"
require "rspec/core/rake_task"
require "rb_sys/extensiontask"

RSpec::Core::RakeTask.new(:spec)

RbSys::ExtensionTask.new("rumpy") do |ext|
  ext.lib_dir = "lib/rumpy"
end

task default: %i[compile spec]
