"""
Generate a minimal traced TorchScript model for state-passing integration tests.

Model signature:
    forward(audio_in: Tensor[1, 1, 128], state_in: Tensor[1, 4])
        -> (audio_out: Tensor[1, 1, 128], state_out: Tensor[1, 4])

Behaviour:
    audio_out = audio_in          (passthrough)
    state_out = state_in + 1.0   (state accumulates by 1 each inference)

This makes it easy to verify state-passing: after N inferences starting from
all-zeros state, state_in should equal N.
"""

import os
import torch


class StateAccumulator(torch.nn.Module):
    def forward(
        self, audio_in: torch.Tensor, state_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return audio_in, state_in + 1.0


def main() -> None:
    model = StateAccumulator()
    model.eval()

    audio_in = torch.zeros(1, 1, 128)
    state_in = torch.zeros(1, 4)

    traced = torch.jit.trace(model, (audio_in, state_in))

    out_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "state_accumulator.pt")
    traced.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
