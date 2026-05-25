cmake_minimum_required(VERSION 3.19)
project(anira_tilde VERSION 0.0.1)

if (APPLE)
    if (${CMAKE_GENERATOR} MATCHES "Xcode")
            if (${XCODE_VERSION} VERSION_LESS 10)
                message(STATUS "Xcode 10 or higher is required. Please install from the Mac App Store.")
                return ()
            elseif(${XCODE_VERSION} VERSION_GREATER_EQUAL 12)
                set(C74_BUILD_FAT YES)
            endif ()
    endif ()

    if (NOT CMAKE_OSX_ARCHITECTURES)
        if(C74_BUILD_FAT)
            set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architecture" FORCE)
        else()
            set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "macOS architecture" FORCE)
        endif()
        message("CMAKE_OSX_ARCHITECTURES set to ${CMAKE_OSX_ARCHITECTURES}")
    endif()
endif()

set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/externals")

# Misc setup and subroutines
set(C74_MIN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/min-api/script)
include(${C74_MIN_SCRIPT_DIR}/min-package.cmake)
include(${C74_MIN_SCRIPT_DIR}/min-pretarget.cmake)

# Add the Min-API & the Anira library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/modules/min-lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/modules/anira)

#############################################################
# ANIRA MAX EXTERNAL
#############################################################

add_library(${PROJECT_NAME} MODULE
    source/AniraTilde.cpp
    source/dsp/AniraProcessor.cpp
    source/utils/Mixer.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${C74_INCLUDES}"
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE anira::anira)
include(${C74_MIN_SCRIPT_DIR}/min-posttarget.cmake)
