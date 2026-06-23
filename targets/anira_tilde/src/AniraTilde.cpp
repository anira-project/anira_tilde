#include "AniraTilde.h"

AniraTilde::AniraTilde(const c74::min::atoms& args) : AniraTildeBase("anira~") {
    initialize(args);
}

void AniraTilde::create_signal_ports(const std::vector<size_t>& sig_inputs,
                                     const std::vector<size_t>& sig_outputs) {
    // Signal inlets: one per channel of each signal input tensor.
    for (size_t t = 0; t < sig_inputs.size(); ++t) {
        for (size_t c = 0; c < sig_inputs[t]; ++c) {
            Input in;
            in.inlet        = std::make_unique<c74::min::inlet<>>(this, channel_label("signal", t, c), "signal");
            in.type         = MaxType::SIGNAL;
            in.num_channels = 1;
            in.tensor_index = t;
            m_sig_inlets.push_back(std::move(in));
        }
    }

    // Signal outlets: same shape.
    for (size_t t = 0; t < sig_outputs.size(); ++t) {
        for (size_t c = 0; c < sig_outputs[t]; ++c) {
            Output out;
            out.outlet       = std::make_unique<c74::min::outlet<>>(this, channel_label("signal", t, c), "signal");
            out.type         = MaxType::SIGNAL;
            out.num_channels = 1;
            out.tensor_index = t;
            m_sig_outlets.push_back(std::move(out));
        }
    }
}

// Registration. anira~ and mc.anira~ are separate translation units / binaries
// because min-api keeps the registered class in a per-TU `this_class` global and
// only supports one class per TU; mc.anira~ has the symmetric entry in
// McAniraTilde.cpp.
namespace {
    void register_anira(void* r) {
        c74::min::wrap_as_max_external<AniraTilde>("AniraTilde", "anira~", r);
    }
}

#ifdef __APPLE__
// macOS: the libtorch-collision guard lives in a separate shim bundle that
// dlopens this dylib and calls this C-linkage entry once it has cleared the
// guard (see Guard.cpp / setup-target-anira-tilde.cmake).
extern "C" __attribute__((visibility("default"))) void anira_tilde_impl_main(void* r) {
    register_anira(r);
}
#else
// Other platforms load this module (anira~) directly.
void ext_main(void* r) {
    register_anira(r);
}
#endif
