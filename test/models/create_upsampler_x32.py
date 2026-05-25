"""
Generate a minimal traced TorchScript model for rate-adaptation integration tests.

Model signature:
    forward(x: Tensor[1, 1, 1]) -> Tensor[1, 1, 32]

Behaviour:
    output = input repeated 32 times (expand along time dimension)

This makes it easy to verify rate adaptation: input value 1.0 should produce
32 output samples all equal to 1.0.  The model has input_size=1, output_size=32
so N=32: one inference every 32 audio output samples.
"""

import os
import torch


class Upsampler(torch.nn.Module):
    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, C, 1] -> [1, C, factor]
        return x.expand(-1, -1, self.factor).contiguous()


def main() -> None:
    factor = 32
    model = Upsampler(factor)
    model.eval()

    example = torch.zeros(1, 1, 1)
    traced = torch.jit.trace(model, (example,))

    out_path = os.path.join(os.path.dirname(__file__), "upsampler_x32.pt")
    traced.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
