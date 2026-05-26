# Rate Adaptation Design

## Problem

anira's inference scheduling is input-driven: it fires one inference whenever
`preprocess_input_size` samples accumulate in the send buffer. For models where
`postprocess_output_size ≠ preprocess_input_size`, this produces output at the
wrong rate, under- or overflowing the receive ring buffer.

## Config Interpretation

The ratio between input and output sample counts is fully determined by the
existing `processing_spec` fields:

```
N = max(output_size, input_size) / min(output_size, input_size)
```

N must be a positive integer (`max` divisible by `min`); validated at initialisation time.
Non-integer ratios are not supported (incoherent in a fixed-size real-time audio I/O context).

For the sashimi decoder example (examples/sashimi_tiny_latent_to_audio/sashimi_tiny_decode.json)
(`preprocess_input_size=1, postprocess_output_size=2048`): N = 2048 — one 64-channel latent frame
in, 2048 mono audio samples out.

Non-streamable tensors (`size=0`) are state or control-rate and are unaffected.
No new config fields are needed; the ratio is derived, not declared.

## Solution

Rate adaptation is handled entirely in `AniraProcessor` (source/dsp/AniraProcessor.cpp), which sits
at the boundary between the Max audio graph (always audio-rate) and the anira library. The
`PrePostProcessor` (source/dsp/StatePassingPrePostProcessor.h) is unchanged — it packs/unpacks
tensors per inference and is agnostic to rate.

Inference fires as soon as the minimum required input is available, with no additional delay.

### input_size < output_size (e.g. latent → audio)

- Within each callback, identify any sample at position `i` where `i % output_size == 0`.
- Push that group of `input_size` samples to anira; inference fires immediately.
- Call `pop_data` each callback to drain the output.
- anira buffers the output across callbacks as needed.

### input_size > output_size (e.g. audio → latent)

- Push all samples to anira each callback; anira accumulates across callbacks and fires
  inference when `input_size` samples have been received.
- Hold (sample-and-hold) the output value for `input_size` samples until the next inference.

### input_size == output_size (audio-rate)

No rate adaptation required; existing behaviour is correct.

## Out of Scope

- Latent-to-latent models (both input and output sub-audio-rate): the config
  scheme cannot represent these unambiguously. An
  explicit `inference_period_samples` field (or equivalent) would be required.
