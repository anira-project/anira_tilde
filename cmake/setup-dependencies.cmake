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

if(APPLE)
    # min/max-pretarget force 10.11. anira_tilde_core needs std::filesystem
    # (macOS 10.15+) and anira prefers 11.0 for C++20 <semaphore>; older anira
    # versions forced 11.0 themselves, current anira respects whatever is set,
    # so raise it here.
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

add_subdirectory(modules/min-lib)

# ------------------------------------------------------------------------------
# anira backend selection. anira defaults every backend to ON; we pre-seed
# LibTorch to OFF because shipping it alongside the external is painful:
#   - macOS: dyld identifies dylibs by install name, so our libtorch collides
#     with any other libtorch-based external (e.g. nn~) loaded into the same
#     Max process. Working around that requires the loader-shim split in
#     setup-target-anira-tilde.cmake plus extra signing/notarization.
#   - Windows: the torch DLL set is hundreds of MB next to the external.
# TorchScript models (.pt/.ts) will NOT load without it. To build with
# LibTorch anyway, configure with:
#     cmake -DANIRA_WITH_LIBTORCH=ON ...
# (or set it in a CMake preset). The ONNX Runtime, LiteRT and ExecuTorch
# backends remain enabled; ExecuTorch covers torch-family models via
# ahead-of-time exported .pte files and links statically, so it adds no
# runtime libraries to ship.
# ------------------------------------------------------------------------------
set(ANIRA_WITH_LIBTORCH OFF CACHE BOOL "Build with the LibTorch backend")

add_subdirectory(modules/anira)
