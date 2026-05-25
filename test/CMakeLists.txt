cmake_minimum_required(VERSION 3.19)
project(anira_tilde_tests)

include(setup-googletest)
include(copy-windows-runtime-dlls)

# Each test JSON has a per-build baseline max_inference_time. Slow build
# flavours (sanitizers, x86_64 emulated under Rosetta) can add a fixed
# overhead via ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD without
# touching the source JSONs. Default overhead is 0.
set(ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD 0 CACHE STRING
    "Extra ms added to every test JSON's max_inference_time at configure time")

# name -> base max_inference_time (ms)
set(_test_json_bases
    downsample_test         40
    rate_adapt_test         40
    upsample_state_test     40
    upsample_multistate_test 40
    sine_oscillator_test    10
)
# Absolute path to the test models, baked into each generated JSON so
# anira's relative-path resolution finds them from the binary dir.
set(MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")

list(LENGTH _test_json_bases _n)
math(EXPR _last "${_n} - 1")
foreach(_i RANGE 0 ${_last} 2)
    math(EXPR _j "${_i} + 1")
    list(GET _test_json_bases ${_i} _name)
    list(GET _test_json_bases ${_j} _base)
    math(EXPR _v "${_base} + ${ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD}")
    set(${_name}_max_inference_time "${_v}.0")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${_name}.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/${_name}.json
        @ONLY
    )
endforeach()

add_executable(anira_tilde_tests
    test_state_pair_parser.cpp
    test_state_passing.cpp
    test_anira_processor.cpp
    test_rate_adaptation.cpp
    test_effective_buffer_size.cpp
    test_engine_reload.cpp
)

target_compile_definitions(anira_tilde_tests PRIVATE
    ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD_MS=${ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD}
    STATE_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/test/models/state_accumulator.pt"
    SINE_OSC_MODEL_PATH="${CMAKE_SOURCE_DIR}/test/models/sine_oscillator.pt"
    SINE_OSC_JSON_PATH="${CMAKE_CURRENT_BINARY_DIR}/sine_oscillator_test.json"
    SINE_OSC_RELATIVE_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sine_oscillator_relative_test.json"
    RATE_ADAPT_JSON_PATH="${CMAKE_CURRENT_BINARY_DIR}/rate_adapt_test.json"
    UPSAMPLE_STATE_JSON_PATH="${CMAKE_CURRENT_BINARY_DIR}/upsample_state_test.json"
    UPSAMPLE_MULTISTATE_JSON_PATH="${CMAKE_CURRENT_BINARY_DIR}/upsample_multistate_test.json"
    DOWNSAMPLE_JSON_PATH="${CMAKE_CURRENT_BINARY_DIR}/downsample_test.json"
)

target_link_libraries(anira_tilde_tests
    gtest_main
    anira_tilde_core
)

anira_tilde_apply_cxx_standard(anira_tilde_tests)

# PRE_TEST defers discovery to ctest run time, which is required when
# cross-compiling (e.g. building macOS x86_64 on an arm64 runner) since
# the binary can't necessarily be executed at configure time.
gtest_discover_tests(anira_tilde_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 90
)

anira_tilde_copy_windows_runtime_dlls(anira_tilde_tests)
