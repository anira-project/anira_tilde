"""
Generate a traced TorchScript model for downsampler rate-adaptation tests.

Model signature:
    forward(x: Tensor[1, 1, 32]) -> Tensor[1, 1, 1]

Behaviour:
    output = mean of the 32 input samples

With input_size=32, output_size=1 (N=32), this exercises the
input_size > output_size code path: anira accumulates 32 input samples
before firing one inference that produces 1 output sample.
The design specifies that this single output is sample-and-held across
all 32 output positions until the next inference fires.
"""

import os
import torch

from export_util import export_all


class Downsampler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, C, 32] -> [1, C, 1]
        return x.mean(dim=-1, keepdim=True)


def main() -> None:
    model = Downsampler()
    example = torch.zeros(1, 1, 32)
    export_all(model, (example,), "downsampler_x32")


if __name__ == "__main__":
    main()
