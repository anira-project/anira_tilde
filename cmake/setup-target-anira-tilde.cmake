# anira~ and mc.anira~ are two Max objects built from this project. min-api keeps
# the registered class in a per-translation-unit `this_class` global and supports
# only one class per TU, so each object is its own translation unit and its own
# binary (AniraTilde.cpp -> anira~, McAniraTilde.cpp -> mc.anira~).
#
# The libtorch-collision guard is macOS-specific: it relies on dyld install
# names, dlopen, mach-o image enumeration and a separately signed .dylib (see
# Guard.cpp). It is only needed when the LibTorch backend is compiled in
# (ANIRA_WITH_LIBTORCH=ON, see setup-dependencies.cmake); on macOS we then
# split each object in two:
#
#   <obj>             -> <obj>.mxo, a libtorch-free loader shim Max loads.
#   <obj>_impl.dylib  -> the real external (links libtorch via anira), dlopen'd
#                        by the shim only when no foreign libtorch is present.
#
# A single shim binary serves both objects: it derives its identity (anira~ vs
# mc.anira~) from its own bundle path at load time and dlopen's the matching
# impl dylib / entry point. anira~.mxo is built by the Min machinery; mc.anira~.mxo
# is a signed clone of it.
#
# Without LibTorch (the default), and on all other platforms, we build each
# object directly as its own Min target and let max-posttarget link the
# audio/jitter libraries.

set(_max_dir ${CMAKE_CURRENT_SOURCE_DIR}/targets/anira_tilde)

if(APPLE AND ANIRA_WITH_LIBTORCH)
    # Audio/Jitter frameworks are normally linked by max-posttarget; the impl
    # dylibs bypass that machinery, so locate them once and link explicitly.
    find_library(MAX_AUDIO_API_LIBRARY "MaxAudioAPI"
        REQUIRED PATHS "${MAX_SDK_MSP_INCLUDES}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(MAX_JITTER_API_LIBRARY "JitterAPI"
        REQUIRED PATHS "${MAX_SDK_JIT_INCLUDES}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    # -----------------------------------------------------------------------
    # Implementation dylib: the actual Min external for one object, carrying the
    # libtorch link. Built as a plain shared library (not a .mxo) and dropped in
    # externals/ next to the shim bundle. The global Max linker flags (-Wl,-U,_*)
    # already allow the host-provided Max symbols to be undefined at link time.
    # The output name doubles as the leaf the loader shim dlopen's.
    # -----------------------------------------------------------------------
    function(anira_tilde_add_impl target src)
        add_library(${target} SHARED "${src}")
        target_include_directories(${target} PRIVATE "${C74_INCLUDES}" ${_max_dir}/src)
        target_link_libraries(${target} PRIVATE
            anira_tilde_core ${MAX_AUDIO_API_LIBRARY} ${MAX_JITTER_API_LIBRARY})
        # Export the shim entry point (anira_tilde_impl_main) instead of ext_main;
        # the sources switch on this define.
        target_compile_definitions(${target} PRIVATE ANIRA_TILDE_LIBTORCH_GUARD)
        anira_tilde_apply_cxx_standard(${target})

        # Sits next to the .mxo shims (which min-posttarget drops in
        # CMAKE_LIBRARY_OUTPUT_DIRECTORY) so Guard.cpp can dlopen it by relative
        # path. Follows C74_LIBRARY_OUTPUT_DIRECTORY so dev/test builds keep it
        # out of the source externals/ that Max loads (see setup-dependencies).
        set_target_properties(${target} PROPERTIES
            PREFIX ""
            OUTPUT_NAME "${target}"
            LIBRARY_OUTPUT_DIRECTORY "${C74_LIBRARY_OUTPUT_DIRECTORY}"
        )

        # Ad-hoc sign so the dylib can be dlopen'd on Apple Silicon.
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND codesign --force --sign - "$<TARGET_FILE:${target}>"
            VERBATIM
            COMMENT "Ad-hoc signing ${target}.dylib"
        )
    endfunction()

    anira_tilde_add_impl(anira_tilde_impl    ${_max_dir}/src/AniraTilde.cpp)
    anira_tilde_add_impl(mc_anira_tilde_impl ${_max_dir}/src/McAniraTilde.cpp)

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

    add_dependencies(${PROJECT_NAME} anira_tilde_impl mc_anira_tilde_impl)

    include(${C74_MIN_SCRIPT_DIR}/min-posttarget.cmake)

    # -----------------------------------------------------------------------
    # mc.anira~ bundle. Max discovers externals by bundle name, so mc.anira~
    # needs its own loadable .mxo. It uses the very same loader-shim binary as
    # anira~ (the shim picks its object identity from its bundle path at load
    # time and dlopen's the matching impl dylib), so we clone the freshly built
    # anira~.mxo and re-sign. This POST_BUILD runs after min-posttarget's own
    # signing step because it is added later on the same target.
    # -----------------------------------------------------------------------
    set(_mc_bundle "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mc.anira~.mxo")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_mc_bundle}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:${PROJECT_NAME}>" "${_mc_bundle}"
        COMMAND codesign --force --sign - "${_mc_bundle}"
        VERBATIM
        COMMENT "Cloning anira~.mxo loader shim into mc.anira~.mxo"
    )
else()
    # -----------------------------------------------------------------------
    # Direct externals: no collision guard needed (no LibTorch on macOS, or a
    # non-Apple platform), so each real Min external ships directly.
    # max-posttarget links the audio/jitter libraries and produces the platform
    # external (anira~ from ${PROJECT_NAME}).
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

    # -----------------------------------------------------------------------
    # mc.anira~ external, its own translation unit / binary. min-posttarget is
    # bound to ${PROJECT_NAME}, so set the bundle naming/properties to match what
    # it produced for the anira~ target.
    # -----------------------------------------------------------------------
    add_library(mc_anira_tilde MODULE
        ${_max_dir}/src/McAniraTilde.cpp
    )
    target_include_directories(mc_anira_tilde PRIVATE
        "${C74_INCLUDES}"
        ${_max_dir}/src
    )
    target_link_libraries(mc_anira_tilde PRIVATE anira_tilde_core)
    anira_tilde_apply_cxx_standard(mc_anira_tilde)

    # min-posttarget (via max-posttarget) links the Max import libraries into
    # ${PROJECT_NAME} only. Windows resolves all symbols at link time, so
    # mc_anira_tilde needs them too; macOS links the MaxAudioAPI/JitterAPI
    # stubs the same way max-posttarget did for ${PROJECT_NAME} (other Max
    # symbols stay undefined for the host to provide at load time).
    if(WIN32)
        target_link_libraries(mc_anira_tilde PRIVATE
            ${MaxAPI_LIB} ${MaxAudio_LIB} ${Jitter_LIB})
    endif()

    set_target_properties(mc_anira_tilde PROPERTIES
        PREFIX ""
        OUTPUT_NAME "mc.anira~"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )

    if(APPLE)
        # The min/max-posttarget bundle machinery is bound to ${PROJECT_NAME},
        # so replicate it here: .mxo bundle, Info.plist/PkgInfo, audio/jitter
        # stubs (MSP_LIBRARY/JITTER_LIBRARY were located by max-posttarget) and
        # an ad-hoc signature so the bundle loads on Apple Silicon.
        target_link_libraries(mc_anira_tilde PRIVATE
            ${MSP_LIBRARY} ${JITTER_LIBRARY})
        # Ninja rejects two bundle targets sharing one MACOSX_BUNDLE_INFO_PLIST
        # template (the file is emitted twice as a re-configure output), and
        # ${PROJECT_NAME} already uses the SDK template, so point mc.anira~ at
        # its own verbatim copy.
        set(_mc_plist "${CMAKE_CURRENT_BINARY_DIR}/mc_anira_tilde-Info.plist.in")
        configure_file("${C74_MAX_SDK_DIR}/script/Info.plist.in" "${_mc_plist}" COPYONLY)
        set_target_properties(mc_anira_tilde PROPERTIES
            BUNDLE TRUE
            BUNDLE_EXTENSION "mxo"
            MACOSX_BUNDLE_BUNDLE_VERSION "${GIT_VERSION_TAG}"
            MACOSX_BUNDLE_INFO_PLIST "${_mc_plist}"
        )
        add_custom_command(TARGET mc_anira_tilde POST_BUILD
            COMMAND cp "${C74_MAX_SDK_DIR}/script/PkgInfo"
                "$<TARGET_BUNDLE_DIR:mc_anira_tilde>/Contents/PkgInfo"
            COMMAND codesign --force --sign - "$<TARGET_BUNDLE_DIR:mc_anira_tilde>"
            VERBATIM
            COMMENT "Finalizing mc.anira~.mxo bundle (PkgInfo + ad-hoc signature)"
        )
    else()
        get_target_property(_ext_suffix ${PROJECT_NAME} SUFFIX)
        set_target_properties(mc_anira_tilde PROPERTIES SUFFIX "${_ext_suffix}")
    endif()
endif()
