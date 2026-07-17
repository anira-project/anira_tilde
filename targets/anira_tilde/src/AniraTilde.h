#pragma once

#include "AniraTildeBase.h"

// anira~ — one mono signal inlet/outlet per channel of each signal tensor.
// All behaviour is shared with mc.anira~ via AniraTildeBase; only the signal
// port layout differs (see create_signal_ports).
class AniraTilde : public AniraTildeBase<AniraTilde, c74::min::vector_operator<>> {
public:
    explicit AniraTilde(const c74::min::atoms& args = {});
    ~AniraTilde() override = default;

    MIN_DESCRIPTION {
        "Neural network inference wrapper for Max. "
        "The anira~ external integrates the <a href='https://github.com/anira-project/anira'>anira</a> library to offer neural network inference inside Max. "
        "Inlets and outlets are dynamically configured during object initialization streamable (signal) and non-streamable (message) data. "
        "The last outlet always outputs the processing latency."
    };
    MIN_TAGS    { "audio, ML, inference" };
    MIN_AUTHOR  { "Valentin Ackva, Fares Schulz, Konstantin Fontaine" };
    MIN_RELATED { "mc.anira~, nn~" };

protected:
    void create_signal_ports(const std::vector<size_t>& sig_inputs,
                             const std::vector<size_t>& sig_outputs) override;
};
