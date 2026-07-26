# Brings in min-api, min-lib and anira as build-tree subprojects.
#
# Order matters: min-pretarget forces CMAKE_OSX_DEPLOYMENT_TARGET to 10.11 and
# we raise it back to 11.0 below (anira >= 2.1 respects a pre-set value instead
# of overriding it). Targets created after these includes pick up 11.0.

# Where shared libs (the anira~ external, plus its libanira/gtest deps) land.
# Defaults to the source-tree externals/ that Max loads. Overridable from the
# command line / a preset so dev + test builds can target their own build tree
# instead of clobbering the installed external's libanira.* at runtime.
if(NOT DEFINED C74_LIBRARY_OUTPUT_DIRECTORY)
    set(C74_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/externals")
endif()
set(C74_MIN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/min-api/script)
include(${C74_MIN_SCRIPT_DIR}/min-package.cmake)
include(${C74_MIN_SCRIPT_DIR}/min-pretarget.cmake)

if(MSVC)
    # max-pretarget forces the static CRT (/MT), the Max SDK default. Every
    # static backend archive from anira-project/backends (ONNX Runtime, LiteRT,
    # ExecuTorch) is built with the MSVC default dynamic CRT (/MD), and mixing
    # the two is a hard LNK2038 error. Flip the whole build to /MD — Max
    # ships the VC redistributable, so a dynamic-CRT external loads fine.
    foreach(_lang C CXX)
        foreach(_cfg DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
            string(REPLACE "/MTd" "/MDd" CMAKE_${_lang}_FLAGS_${_cfg} "${CMAKE_${_lang}_FLAGS_${_cfg}}")
            string(REPLACE "/MT" "/MD" CMAKE_${_lang}_FLAGS_${_cfg} "${CMAKE_${_lang}_FLAGS_${_cfg}}")
        endforeach()
    endforeach()
    # max-pretarget also injects /MT via add_compile_options generator
    # expressions; rewrite those on the directory property.
    get_property(_anira_tilde_opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    string(REPLACE "/MTd" "/MDd" _anira_tilde_opts "${_anira_tilde_opts}")
    string(REPLACE "/MT" "/MD" _anira_tilde_opts "${_anira_tilde_opts}")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${_anira_tilde_opts}")
endif()

if(APPLE)
    # min/max-pretarget force 10.11. anira_tilde_core needs std::filesystem
    # (macOS 10.15+) and anira prefers 11.0 for C++20 <semaphore>; older anira
    # versions forced 11.0 themselves, current anira respects whatever is set,
    # so raise it here.
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

add_subdirectory(modules/min-lib)

# ------------------------------------------------------------------------------
# libsamplerate: real-time sample-rate conversion between the host rate and the
# model's native rate (resampler_config.model_sample_rate in the JSON configs).
# Static, linked into anira_tilde_core; src_process() allocates nothing after
# src_new(), so the audio-thread path stays real-time safe.
# ------------------------------------------------------------------------------
include(FetchContent)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(LIBSAMPLERATE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LIBSAMPLERATE_INSTALL OFF CACHE BOOL "" FORCE)
# The 0.2.2 release's cmake_minimum_required predates CMake 4's compatibility
# floor; opt its subproject into the modern policy baseline.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(
    libsamplerate
    GIT_REPOSITORY https://github.com/libsndfile/libsamplerate.git
    GIT_TAG 0.2.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libsamplerate)
unset(CMAKE_POLICY_VERSION_MINIMUM)
# Linked into shared impl dylibs / .mxo bundles on every platform.
set_target_properties(samplerate PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------
# anira backend selection and linkage. Everything is bundled INSIDE the
# externals: BUILD_SHARED_LIBS=OFF makes libanira static, and the ONNX
# Runtime / LiteRT / ExecuTorch backends follow it into the same archive
# (ExecuTorch is static-only regardless). The shipped package is then just
# anira~.mxo / mc.anira~.mxe64 with no runtime libraries next to it —
# shipping shared libs is a pain, especially on Windows.
#
# LibTorch is the exception on both counts and is therefore OFF by default:
#   - it only ships as shared libraries (anira forces shared linkage), so it
#     can never be bundled inside — enabling it brings the torch dylib/DLL
#     set (hundreds of MB) back into the package;
#   - macOS: dyld identifies dylibs by install name, so our libtorch collides
#     with any other libtorch-based external (e.g. nn~) loaded into the same
#     Max process. Working around that requires the loader-shim split in
#     setup-target-anira-tilde.cmake plus extra signing/notarization.
# TorchScript models (.pt/.ts) will NOT load without it. To build with
# LibTorch anyway, configure with:
#     cmake -DANIRA_WITH_LIBTORCH=ON ...
# (or set it in a CMake preset). ExecuTorch covers torch-family models via
# ahead-of-time exported .pte files instead.
# ------------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build anira as a shared library")
set(ANIRA_WITH_LIBTORCH OFF CACHE BOOL "Build with the LibTorch backend")

add_subdirectory(modules/anira)
