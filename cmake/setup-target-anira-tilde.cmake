# The libtorch-collision guard is macOS-specific: it relies on dyld install
# names, dlopen, mach-o image enumeration and a separately signed .dylib (see
# Guard.cpp). On macOS we therefore split the external in two so we can warn
# instead of crash when a conflicting libtorch is already loaded:
#
#   anira_tilde       -> anira~.mxo, a libtorch-free loader shim Max loads.
#   anira_tilde_impl  -> anira_tilde_impl.dylib, the real external (links
#                        libtorch via anira), dlopen'd by the shim only when
#                        no foreign libtorch is present.
#
# On other platforms we build the real external directly as a single Min
# target and let max-posttarget link the audio/jitter libraries.

set(_max_dir ${CMAKE_CURRENT_SOURCE_DIR}/targets/anira_tilde)

if(APPLE)
    # -----------------------------------------------------------------------
    # Implementation dylib: the actual Min external, carrying the libtorch
    # link. Built as a plain shared library (not a .mxo) and dropped in
    # externals/ next to the shim bundle. The global Max linker flags
    # (-Wl,-U,_*) already allow the host-provided Max symbols to be undefined
    # at link time.
    # -----------------------------------------------------------------------

    add_library(anira_tilde_impl SHARED ${_max_dir}/src/AniraTilde.cpp)

    target_include_directories(anira_tilde_impl PRIVATE
        "${C74_INCLUDES}"
        ${_max_dir}/src
    )

    target_link_libraries(anira_tilde_impl PRIVATE anira_tilde_core)
    anira_tilde_apply_cxx_standard(anira_tilde_impl)

    # Audio/Jitter frameworks are normally linked by max-posttarget; this
    # target bypasses that machinery, so link them explicitly.
    find_library(MAX_AUDIO_API_LIBRARY "MaxAudioAPI"
        REQUIRED PATHS "${MAX_SDK_MSP_INCLUDES}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(MAX_JITTER_API_LIBRARY "JitterAPI"
        REQUIRED PATHS "${MAX_SDK_JIT_INCLUDES}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    target_link_libraries(anira_tilde_impl PRIVATE
        ${MAX_AUDIO_API_LIBRARY} ${MAX_JITTER_API_LIBRARY})

    # Sits next to the anira~.mxo shim (which min-posttarget drops in
    # CMAKE_LIBRARY_OUTPUT_DIRECTORY) so Guard.cpp can dlopen it by relative
    # path. Follows C74_LIBRARY_OUTPUT_DIRECTORY so dev/test builds keep it out
    # of the source externals/ that Max loads (see setup-dependencies.cmake).
    set_target_properties(anira_tilde_impl PROPERTIES
        PREFIX ""
        OUTPUT_NAME "anira_tilde_impl"
        LIBRARY_OUTPUT_DIRECTORY "${C74_LIBRARY_OUTPUT_DIRECTORY}"
    )

    # Ad-hoc sign so the dylib can be dlopen'd on Apple Silicon.
    add_custom_command(TARGET anira_tilde_impl POST_BUILD
        COMMAND codesign --force --sign - "$<TARGET_FILE:anira_tilde_impl>"
        VERBATIM
        COMMENT "Ad-hoc signing anira_tilde_impl.dylib"
    )

    # -----------------------------------------------------------------------
    # Loader shim: the bundle Max actually loads. No libtorch, so its ext_main
    # always runs and can report a conflict. Goes through the normal Min/Max
    # post-target machinery to produce anira~.mxo.
    # -----------------------------------------------------------------------

    add_library(${PROJECT_NAME} MODULE
        ${_max_dir}/src/Guard.cpp
    )

    target_include_directories(${PROJECT_NAME} PRIVATE
        "${C74_INCLUDES}"
    )

    anira_tilde_apply_cxx_standard(${PROJECT_NAME})

    add_dependencies(${PROJECT_NAME} anira_tilde_impl)

    include(${C74_MIN_SCRIPT_DIR}/min-posttarget.cmake)
else()
    # -----------------------------------------------------------------------
    # Single-target external: no collision guard, so the real Min external
    # ships directly. max-posttarget links the audio/jitter libraries and
    # produces the platform external.
    # -----------------------------------------------------------------------

    add_library(${PROJECT_NAME} MODULE
        ${_max_dir}/src/AniraTilde.cpp
    )

    target_include_directories(${PROJECT_NAME} PRIVATE
        "${C74_INCLUDES}"
        ${_max_dir}/src
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE anira_tilde_core)
    anira_tilde_apply_cxx_standard(${PROJECT_NAME})

    include(${C74_MIN_SCRIPT_DIR}/min-posttarget.cmake)
endif()
