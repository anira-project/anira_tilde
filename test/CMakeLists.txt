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
    ${CMAKE_SOURCE_DIR}/source/dsp/StatePairParser.cpp
)

target_include_directories(anira_tilde_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/source/dsp
)

target_link_libraries(anira_tilde_tests
    gtest_main
    nlohmann_json::nlohmann_json
)

target_compile_features(anira_tilde_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(anira_tilde_tests)
