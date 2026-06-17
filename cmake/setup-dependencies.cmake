# Brings in min-api, min-lib and anira as build-tree subprojects.
#
# Order matters: min-pretarget forces CMAKE_OSX_DEPLOYMENT_TARGET to 10.11;
# anira's CMakeLists then raises it to 11.0. Targets created AFTER this
# include picks up the final 11.0 value.

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

add_subdirectory(modules/min-lib)
add_subdirectory(modules/anira)
