# Apply the project-wide C++ standard to a target. Centralised so we
# don't drift between targets.
function(anira_tilde_apply_cxx_standard target)
    target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()
