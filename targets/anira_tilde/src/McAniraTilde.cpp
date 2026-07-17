#include "McAniraTilde.h"

McAniraTilde::McAniraTilde(const c74::min::atoms& args) :
    AniraTildeBase("mc.anira~"),
    maxclass_setup(this, "maxclass_setup",
        MIN_FUNCTION {
            // min-api has no multichannel-output support, so register the two
            // A_CANT methods Max needs to negotiate mc outlets ourselves. The
            // class pointer is handed to us as the first argument.
            c74::max::t_class* c = args[0];
            c74::max::class_addmethod(c, reinterpret_cast<c74::max::method>(mc_multichanneloutputs),
                                      "multichanneloutputs", c74::max::A_CANT, 0);
            c74::max::class_addmethod(c, reinterpret_cast<c74::max::method>(mc_inputchanged),
                                      "inputchanged", c74::max::A_CANT, 0);
            return {};
        }
    )
{
    initialize(args);
}

void McAniraTilde::create_signal_ports(const std::vector<size_t>& sig_inputs,
                                       const std::vector<size_t>& sig_outputs) {
    // Signal inlets: one multichannel inlet per signal input tensor, carrying
    // all of that tensor's channels on a single mc signal.
    for (size_t t = 0; t < sig_inputs.size(); ++t) {
        Input in;
        in.inlet        = std::make_unique<c74::min::inlet<>>(this, tensor_label("mc signal", t), "signal");
        in.type         = MaxType::SIGNAL;
        in.num_channels = sig_inputs[t];
        in.tensor_index = t;
        m_sig_inlets.push_back(std::move(in));
    }

    // Signal outlets: one multichannel outlet per signal output tensor.
    for (size_t t = 0; t < sig_outputs.size(); ++t) {
        Output out;
        out.outlet       = std::make_unique<c74::min::outlet<>>(this, tensor_label("mc signal", t), "multichannelsignal");
        out.type         = MaxType::SIGNAL;
        out.num_channels = sig_outputs[t];
        out.tensor_index = t;
        m_sig_outlets.push_back(std::move(out));
    }
}

// Report the channel count for a given signal outlet. Signal outlets are created
// first, so the index Max passes maps directly onto m_sig_outlets.
long McAniraTilde::mc_multichanneloutputs(c74::max::t_object* x, long index) {
    auto& self = reinterpret_cast<c74::min::minwrap<McAniraTilde>*>(x)->m_min_object;
    if (index >= 0 && static_cast<size_t>(index) < self.m_sig_outlets.size())
        return static_cast<long>(self.m_sig_outlets[index].num_channels);
    return 0;
}

// Notified when the channel count arriving on a signal inlet changes. Our output
// channel counts are fixed by the model config and never depend on the input, so
// we return false (no DSP-graph re-derivation needed). We only use this to warn
// the user when the connection doesn't match what the model expects.
long McAniraTilde::mc_inputchanged(c74::max::t_object* x, long index, long count) {
    auto& self = reinterpret_cast<c74::min::minwrap<McAniraTilde>*>(x)->m_min_object;
    if (index >= 0 && static_cast<size_t>(index) < self.m_sig_inlets.size()) {
        const size_t expected = self.m_sig_inlets[index].num_channels;
        if (count > 0 && static_cast<size_t>(count) != expected) {
            c74::max::object_warn(x,
                "mc.anira~: signal inlet %ld has %ld channel(s) but the model expects %zu; "
                "extra channels are ignored and missing channels are zero-padded.",
                index + 1, count, expected);
        }
    }
    return 0;
}

// Registration. See AniraTilde.cpp for why anira~ and mc.anira~ are separate
// translation units / binaries.
namespace {
    void register_mc_anira(void* r) {
        c74::min::wrap_as_max_external<McAniraTilde>("McAniraTilde", "mc.anira~", r);
    }
}

#ifdef __APPLE__
// macOS: built as mc_anira_tilde_impl.dylib, dlopen'd by the mc.anira~ loader
// shim, which calls this C-linkage entry once it has cleared the libtorch guard.
extern "C" __attribute__((visibility("default"))) void mc_anira_tilde_impl_main(void* r) {
    register_mc_anira(r);
}
#else
// Other platforms load this module (mc.anira~) directly.
void ext_main(void* r) {
    register_mc_anira(r);
}
#endif
