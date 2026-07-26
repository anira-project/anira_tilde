"""
Generate a traced TorchScript sine oscillator model for state-passing tests.

Tensor layout
-------------
Inputs:
  0  freq      [1, 1, 512]   streamable  — frequency in Hz per sample (drives inference)
  1  phase_in  [1, 1]        non-streamable — current phase in radians (state)

Outputs:
  0  audio_out [1, 1, 512]   streamable  — sine wave samples
  1  phase_out [1, 1]        non-streamable — updated phase (state)

State pair: output_tensor=1 → input_tensor=1

The frequency tensor drives inference timing: once 512 frequency samples have
accumulated, anira triggers one inference that produces 512 audio samples.
Frequency can vary per-sample within the block (useful for FM/glide), or be
held constant for a plain oscillator.  Phase state is passed back so the sine
wave is continuous across block boundaries.
"""

import math
import os
import torch

from export_util import export_all

SIGNAL_SIZE = 512
SAMPLE_RATE = 44100.0


class SineOscillator(torch.nn.Module):
    def forward(
        self,
        freq: torch.Tensor,       # [1, 1, SIGNAL_SIZE] — Hz per sample
        phase_in: torch.Tensor,   # [1, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_increments = 2.0 * math.pi * freq / SAMPLE_RATE    # [1, 1, 512]
        cumulative = torch.cumsum(phase_increments, dim=2)         # [1, 1, 512]
        phases = phase_in.unsqueeze(-1) + cumulative               # [1, 1, 512]
        audio_out = torch.sin(phases)
        phase_out = phases[..., -1] % (2.0 * math.pi)             # [1, 1]
        return audio_out, phase_out


def export_tflite() -> None:
    """TensorFlow twin of the same math, exported as sine_oscillator.tflite for
    the LiteRT backend (there is no torch->tflite path for this graph; the ops
    are trivial, so the duplication is a few lines). Skipped when tensorflow is
    not installed — the committed .tflite only needs regenerating when the
    model itself changes."""
    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow not installed - skipping sine_oscillator.tflite")
        return

    class SineOscillatorTF(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([1, 1, SIGNAL_SIZE], tf.float32),  # freq
            tf.TensorSpec([1, 1], tf.float32),               # phase_in
        ])
        def __call__(self, freq, phase_in):
            phase_increments = 2.0 * math.pi * freq / SAMPLE_RATE
            cumulative = tf.cumsum(phase_increments, axis=2)
            phases = tf.expand_dims(phase_in, -1) + cumulative
            audio_out = tf.sin(phases)
            phase_out = tf.math.floormod(phases[..., -1], 2.0 * math.pi)
            # Named outputs with ordering prefixes: both the TFLite signature
            # (alphabetical) and the graph's output tensor order follow these
            # keys, and anira maps tensors by index — audio must stay tensor 0.
            return {"out_0_audio": audio_out, "out_1_phase": phase_out}

    module = SineOscillatorTF()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.__call__.get_concrete_function()], module)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sine_oscillator.tflite")
    with open(out, "wb") as f:
        f.write(converter.convert())
    print(f"Saved {out}")


def main() -> None:
    model = SineOscillator()
    freq     = torch.full((1, 1, SIGNAL_SIZE), 440.0)
    phase_in = torch.zeros(1, 1)

    export_all(model, (freq, phase_in), "sine_oscillator")
    export_tflite()


if __name__ == "__main__":
    main()
