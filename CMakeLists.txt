cmake_minimum_required(VERSION 3.19)
project(anira_tilde VERSION 0.0.1)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(macos-arch)

set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/externals")

# Misc setup and subroutines
set(C74_MIN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/min-api/script)
include(${C74_MIN_SCRIPT_DIR}/min-package.cmake)
include(${C74_MIN_SCRIPT_DIR}/min-pretarget.cmake)

option(ANIRA_TILDE_WITH_TESTS "Build unit tests" OFF)
option(ANIRA_TILDE_WITH_ASAN  "Enable AddressSanitizer" OFF)
option(ANIRA_TILDE_WITH_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ANIRA_TILDE_WITH_TSAN  "Enable ThreadSanitizer" OFF)
option(ANIRA_TILDE_WITH_RTSAN "Enable RealtimeSanitizer (requires Clang 20+)" OFF)

# Sanitizers must be applied globally so anira's dylib is also instrumented;
# otherwise dyld loads it before ASan can install its interceptors.
include(sanitizers)
anira_tilde_apply_sanitizers()

# Add the Min-API & the Anira library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/modules/min-lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/modules/anira)

#############################################################
# ANIRA MAX EXTERNAL
#############################################################

add_library(${PROJECT_NAME} MODULE
    source/AniraTilde.cpp
    source/dsp/AniraProcessor.cpp
    source/dsp/StatePairParser.cpp
    source/utils/Mixer.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${C74_INCLUDES}"
)

include(cxx-standard)
target_link_libraries(${PROJECT_NAME} PRIVATE anira::anira)
include(${C74_MIN_SCRIPT_DIR}/min-posttarget.cmake)

if(ANIRA_TILDE_WITH_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
