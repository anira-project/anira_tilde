#pragma once

// Symbol-visibility macro for the anira_tilde_core public API.
//
// Use ANIRA_TILDE_API on every public class / struct / free function that
// crosses the library boundary, so the symbols are exported when the
// library is built as a Windows DLL and imported by consumers.
//
//   - Building the DLL          → ANIRA_TILDE_CORE_EXPORTS defined → dllexport
//   - Consuming the DLL         → ANIRA_TILDE_CORE_SHARED  defined → dllimport
//   - Static lib (Windows)      → neither defined → empty
//   - Non-Windows               → visibility("default")
//
// CMake takes care of defining the right macro on the right side.

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(ANIRA_TILDE_CORE_EXPORTS)
        #define ANIRA_TILDE_API __declspec(dllexport)
    #elif defined(ANIRA_TILDE_CORE_SHARED)
        #define ANIRA_TILDE_API __declspec(dllimport)
    #else
        #define ANIRA_TILDE_API
    #endif
#else
    // macOS/Linux: default visibility is already "default", no annotation needed.
    #define ANIRA_TILDE_API
#endif
