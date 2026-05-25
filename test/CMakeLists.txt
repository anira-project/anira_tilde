cmake_minimum_required(VERSION 3.19)
project(anira_tilde_tests)

include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_SHALLOW TRUE
    GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)


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

target_compile_features(anira_tilde_tests PRIVATE cxx_std_20)

include(GoogleTest)
# PRE_TEST defers discovery to ctest run time, which is required when
# cross-compiling (e.g. building macOS x86_64 on an arm64 runner) since
# the binary can't necessarily be executed at configure time.
gtest_discover_tests(anira_tilde_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 90
)

# On Windows, copy anira's runtime DLLs (anira + libtorch/onnx/tflite backends)
# next to the test executable so tests can locate them at runtime.
if(WIN32)
    add_custom_command(TARGET anira_tilde_tests PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ANIRA_SHARED_LIBS_WIN}
            $<TARGET_FILE_DIR:anira_tilde_tests>
        COMMAND_EXPAND_LISTS
    )
    add_custom_command(TARGET anira_tilde_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:anira_tilde_tests>
            $<TARGET_FILE_DIR:anira_tilde_tests>
        COMMAND_EXPAND_LISTS
    )
endif()
