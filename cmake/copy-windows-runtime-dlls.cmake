# Copy runtime DLLs next to a Windows executable target so it can locate them.
# - ANIRA_SHARED_LIBS_WIN (exported by anira's msvc-support.cmake) covers
#   anira.dll and any shared backend DLLs (empty in the default all-static
#   build; carries the torch DLLs when ANIRA_WITH_LIBTORCH=ON).
# - TARGET_RUNTIME_DLLS picks up direct link-deps (gtest, etc.).
function(anira_tilde_copy_windows_runtime_dlls target)
    if(NOT WIN32)
        return()
    endif()
    if(ANIRA_SHARED_LIBS_WIN)
        add_custom_command(TARGET ${target} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ANIRA_SHARED_LIBS_WIN}
                $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS
        )
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${target}>
            $<TARGET_FILE_DIR:${target}>
        COMMAND_EXPAND_LISTS
    )
endfunction()
