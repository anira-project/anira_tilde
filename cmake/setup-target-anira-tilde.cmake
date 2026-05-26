# Thin Min-API wrapper that ships as the Max/MSP external.

set(_max_dir ${CMAKE_CURRENT_SOURCE_DIR}/targets/anira_tilde)

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
