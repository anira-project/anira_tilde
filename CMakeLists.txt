cmake_minimum_required(VERSION 3.19)
project(anira_tilde VERSION 0.0.1)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ==============================================================================
# Build options
# ==============================================================================

option(ANIRA_TILDE_WITH_TESTS "Build unit tests" OFF)
option(ANIRA_TILDE_WITH_ASAN  "Enable AddressSanitizer" OFF)
option(ANIRA_TILDE_WITH_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ANIRA_TILDE_WITH_TSAN  "Enable ThreadSanitizer" OFF)
option(ANIRA_TILDE_WITH_RTSAN "Enable RealtimeSanitizer (requires Clang 20+)" OFF)

# ==============================================================================
# Toolchain helpers: macOS arch selection, C++ standard, sanitizers.
# anira_tilde_apply_sanitizers() must run BEFORE any add_library / add_executable
# so the sanitizer flags are picked up globally (anira's dylib included).
# ==============================================================================

include(macos-arch)
include(cxx-standard)
include(sanitizers)
anira_tilde_apply_sanitizers()

# ==============================================================================
# External dependencies (Min-API, min-lib, anira) brought in as build-tree
# subprojects. Sets up cache variables and defines anira::anira.
# ==============================================================================

include(setup-dependencies)

# Example models: downloaded next to their JSON configs so the examples work
# without any path editing. -DANIRA_TILDE_DOWNLOAD_EXAMPLE_MODELS=OFF to skip.
include(setup-example-models)

# ==============================================================================
# Targets
#   - anira_tilde_core : pure C++ library, no Max dependency
#   - anira_tilde      : Max/MSP external, links anira_tilde_core
# ==============================================================================

include(setup-target-anira-tilde-core)
include(setup-target-anira-tilde)

# ==============================================================================
# Tests (linked against anira_tilde_core only — no Max needed)
# ==============================================================================

if(ANIRA_TILDE_WITH_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
