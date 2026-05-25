# Sanitizer setup. Applies flags GLOBALLY (via add_compile_options /
# add_link_options) so dependency libraries (e.g. anira) are instrumented
# too — required on macOS where dyld will otherwise load the uninstrumented
# anira.dylib before ASan's runtime can install its interceptors.
#
# Call anira_tilde_apply_sanitizers() once at the top level, BEFORE
# add_subdirectory() for any target you want instrumented.

function(anira_tilde_apply_sanitizers)
    set(_sanitizers)
    if(ANIRA_TILDE_WITH_ASAN)
        list(APPEND _sanitizers address)
        add_compile_definitions(ANIRA_TILDE_WITH_ASAN)
    endif()
    if(ANIRA_TILDE_WITH_UBSAN)
        list(APPEND _sanitizers undefined)
        add_compile_definitions(ANIRA_TILDE_WITH_UBSAN)
    endif()
    if(ANIRA_TILDE_WITH_TSAN)
        list(APPEND _sanitizers thread)
        add_compile_definitions(ANIRA_TILDE_WITH_TSAN)
    endif()
    if(ANIRA_TILDE_WITH_RTSAN)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "RTSan (-fsanitize=realtime) requires Clang. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
        endif()
        list(APPEND _sanitizers realtime)
        add_compile_definitions(ANIRA_TILDE_WITH_RTSAN)
    endif()

    if(_sanitizers)
        list(JOIN _sanitizers "," _sanitizers_csv)
        add_compile_options(-fsanitize=${_sanitizers_csv} -fno-omit-frame-pointer)
        add_link_options(-fsanitize=${_sanitizers_csv})
    endif()
endfunction()
