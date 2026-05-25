# Force single-architecture macOS builds. CMAKE_OSX_ARCHITECTURES is honoured
# if set explicitly (e.g. by a CMake preset selecting x86_64); otherwise it
# defaults to the host architecture. Universal binaries are not supported.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "macOS architecture" FORCE)
    message(STATUS "CMAKE_OSX_ARCHITECTURES set to ${CMAKE_OSX_ARCHITECTURES}")
endif()
