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


def main() -> None:
    model = SineOscillator()
    model.eval()

    freq     = torch.full((1, 1, SIGNAL_SIZE), 440.0)
    phase_in = torch.zeros(1, 1)

    traced = torch.jit.trace(model, (freq, phase_in))

    out_path = os.path.join(os.path.dirname(__file__), "sine_oscillator.pt")
    traced.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
