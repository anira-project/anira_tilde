# Pure C++ library that holds the host-agnostic inference engine.

set(_core_dir ${CMAKE_CURRENT_SOURCE_DIR}/targets/anira_tilde_core)

add_library(anira_tilde_core STATIC
    ${_core_dir}/src/Engine.cpp
    ${_core_dir}/src/inference/Session.cpp
    ${_core_dir}/src/inference/TensorLayout.cpp
    ${_core_dir}/src/mixing/Mixer.cpp
    ${_core_dir}/src/rate_adaptation/RateAdaptor.cpp
    ${_core_dir}/src/resampling/Resampler.cpp
    ${_core_dir}/src/resampling/ResamplerConfigParser.cpp
    ${_core_dir}/src/state_passing/StatePairParser.cpp
)

target_include_directories(anira_tilde_core PUBLIC
    ${_core_dir}/include
)

target_link_libraries(anira_tilde_core PUBLIC anira::anira PRIVATE samplerate)
target_compile_definitions(anira_tilde_core PRIVATE ANIRA_TILDE_CORE_EXPORTS)

anira_tilde_apply_cxx_standard(anira_tilde_core)
