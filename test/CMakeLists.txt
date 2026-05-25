cmake_minimum_required(VERSION 3.19)
project(anira_tilde_tests)

include(setup-googletest)
include(copy-windows-runtime-dlls)

add_executable(anira_tilde_tests
    test_state_pair_parser.cpp
    test_state_passing.cpp
    test_anira_processor.cpp
    test_rate_adaptation.cpp
    test_effective_buffer_size.cpp
    ${CMAKE_SOURCE_DIR}/source/dsp/StatePairParser.cpp
    ${CMAKE_SOURCE_DIR}/source/dsp/AniraProcessor.cpp
)

target_include_directories(anira_tilde_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/source/dsp
)

target_compile_definitions(anira_tilde_tests PRIVATE
    STATE_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/test/models/state_accumulator.pt"
    SINE_OSC_MODEL_PATH="${CMAKE_SOURCE_DIR}/test/models/sine_oscillator.pt"
    SINE_OSC_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sine_oscillator_test.json"
    SINE_OSC_RELATIVE_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sine_oscillator_relative_test.json"
    RATE_ADAPT_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/rate_adapt_test.json"
    UPSAMPLE_STATE_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/upsample_state_test.json"
    UPSAMPLE_MULTISTATE_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/upsample_multistate_test.json"
    DOWNSAMPLE_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/downsample_test.json"
)

target_link_libraries(anira_tilde_tests
    gtest_main
    anira::anira
)

include(cxx-standard)

# PRE_TEST defers discovery to ctest run time, which is required when
# cross-compiling (e.g. building macOS x86_64 on an arm64 runner) since
# the binary can't necessarily be executed at configure time.
gtest_discover_tests(anira_tilde_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 90
)

anira_tilde_copy_windows_runtime_dlls(anira_tilde_tests)
