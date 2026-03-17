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
    DOWNSAMPLE_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/downsample_test.json"
)

target_link_libraries(anira_tilde_tests
    gtest_main
    anira::anira
)

target_compile_features(anira_tilde_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(anira_tilde_tests)
