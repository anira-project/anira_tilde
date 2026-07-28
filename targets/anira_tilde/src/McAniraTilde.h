#pragma once

#include "AniraTildeBase.h"

// mc.anira~ — each signal tensor is presented as one Max multichannel (mc) signal
// rather than one mono inlet/outlet per channel. Deriving from mc_operator<> sets
// the Z_MC_INLETS flag so the signal inlets accept packed multichannel signals;
// the mc outputs are declared via the multichanneloutputs/inputchanged methods
// registered in maxclass_setup (min-api does not provide these). All other
// behaviour is shared with anira~ via AniraTildeBase.
class McAniraTilde : public AniraTildeBase<McAniraTilde, c74::min::mc_operator<>> {
public:
    explicit McAniraTilde(const c74::min::atoms& args = {});
    ~McAniraTilde() override = default;

    MIN_DESCRIPTION{
        "Multichannel neural network inference wrapper for Max. "
        "Like <o>anira~</o>, but each signal tensor flows on a single multichannel "
        "(mc) signal cord rather than one mono inlet/outlet per channel. "
        "Inlets and outlets are dynamically configured during object initialization "
        "for streamable (signal) and non-streamable (message) data. "
        "The last outlet always outputs the processing latency."};
    MIN_TAGS{"audio, ML, inference, multichannel"};
    MIN_AUTHOR{"Valentin Ackva, Fares Schulz, Konstantin Fontaine"};
    MIN_RELATED{"anira~, nn~"};

    // Registers the A_CANT methods Max needs to negotiate mc outlets.
    c74::min::message<> m_maxclass_setup;

    // multichanneloutputs reports how many channels a given signal outlet emits;
    // inputchanged is notified when an inlet's incoming channel count changes.
    static long mc_multichanneloutputs(c74::max::t_object* x, long index);
    static long mc_inputchanged(c74::max::t_object* x, long index, long count);

protected:
    void create_signal_ports(const std::vector<size_t>& sig_inputs,
                             const std::vector<size_t>& sig_outputs) override;
};
