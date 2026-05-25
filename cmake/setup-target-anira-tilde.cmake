# Thin Min-API wrapper that ships as the Max/MSP external.

add_library(${PROJECT_NAME} MODULE
    max/AniraTilde.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${C74_INCLUDES}"
)

target_link_libraries(${PROJECT_NAME} PRIVATE anira_tilde_core)
anira_tilde_apply_cxx_standard(${PROJECT_NAME})

include(${C74_MIN_SCRIPT_DIR}/min-posttarget.cmake)
