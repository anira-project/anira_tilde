"""
Generate a traced TorchScript model with a single multi-channel output tensor.

Model signature:
    forward(x: Tensor[1, 1, 1]) -> Tensor[1, 4, 1]

Behaviour:
    output = input broadcast across 4 output channels (one output tensor,
    4 channels, block size 1 — no rate adaptation).

This reproduces the encode-style RAVE shape (one output tensor carrying many
channels, e.g. [1, 16, 1] latents) that crashed the dry/wet Mixer: the Mixer's
per-channel state was sized to the number of output *tensors* (1) but indexed
by the number of output *channels* (4), so channels >= 1 wrote out of bounds.
"""

import os
import torch

from export_util import export_all

CHANNELS = 4


class MultiChannelOut(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 1, 1] -> [1, CHANNELS, 1]
        return x.expand(-1, CHANNELS, -1).contiguous()


def main() -> None:
    model = MultiChannelOut()
    example = torch.zeros(1, 1, 1)
    export_all(model, (example,), "multichannel_out")


if __name__ == "__main__":
    main()
