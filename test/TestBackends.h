// Backend parameterization for the integration tests.
//
// Every test model ships in three formats (test/models/): .pte (ExecuTorch),
// .onnx (ONNX Runtime) and .pt (TorchScript/LibTorch), and every fixture JSON
// is configured once per enabled backend as <base>_<token>.json (see
// test/CMakeLists.txt). The list below contains exactly the backends compiled
// into this build — LibTorch entries appear only when the build opted in via
// ANIRA_WITH_LIBTORCH, so its tests run only if enabled.

#pragma once

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <anira/anira.h>

#ifndef ANIRA_TILDE_TEST_JSON_DIR
#error "ANIRA_TILDE_TEST_JSON_DIR must be defined via CMake"
#endif
#ifndef ANIRA_TILDE_TEST_MODELS_DIR
#error "ANIRA_TILDE_TEST_MODELS_DIR must be defined via CMake"
#endif

namespace anira_tilde_test {

struct Backend {
    anira::InferenceBackend backend;
    const char* token;  // lowercase: fixture-JSON suffix and gtest param name
    const char* ext;    // model file extension
};

inline const std::vector<Backend>& backends() {
    static const std::vector<Backend> list = [] {
        std::vector<Backend> b;
#ifdef USE_EXECUTORCH
        b.push_back({anira::InferenceBackend::EXECUTORCH, "executorch", "pte"});
#endif
#ifdef USE_ONNXRUNTIME
        b.push_back({anira::InferenceBackend::ONNX, "onnx", "onnx"});
#endif
#ifdef USE_LIBTORCH
        b.push_back({anira::InferenceBackend::LIBTORCH, "libtorch", "pt"});
#endif
        return b;
    }();
    return list;
}

inline std::string json_path(const std::string& base, const Backend& b) {
    return std::string(ANIRA_TILDE_TEST_JSON_DIR) + "/" + base + "_" + b.token + ".json";
}

inline std::string model_path(const std::string& base, const Backend& b) {
    return std::string(ANIRA_TILDE_TEST_MODELS_DIR) + "/" + base + "." + b.ext;
}

inline std::string param_name(const testing::TestParamInfo<Backend>& info) {
    return info.param.token;
}

}  // namespace anira_tilde_test
