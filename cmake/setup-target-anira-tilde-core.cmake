# Pure C++ library that holds the host-agnostic inference engine.

add_library(anira_tilde_core STATIC
    src/Engine.cpp
    src/inference/Session.cpp
    src/mixing/Mixer.cpp
    src/rate_adaptation/RateAdaptor.cpp
    src/state_passing/StatePairParser.cpp
)

target_include_directories(anira_tilde_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(anira_tilde_core PUBLIC anira::anira)
target_compile_definitions(anira_tilde_core PRIVATE ANIRA_TILDE_CORE_EXPORTS)

anira_tilde_apply_cxx_standard(anira_tilde_core)
