"""
Generate a TorchScript model for testing upsample + multiple state tensors.
    - tensor 0 (streamable): input_size=1 latent frame -> output_size=16 audio samples
    - tensor 1 (state):      non-streamable, passed back between inferences
    - tensor 2 (state):      non-streamable, passed back between inferences

Model signature:
    forward(latent: Tensor[1, 1, 1], state1: Tensor[1, 4], state2: Tensor[1, 4])
        -> (audio: Tensor[1, 1, 16], state1: Tensor[1, 4], state2: Tensor[1, 4])

Behaviour:
    audio  = (latent + state1[:, :1].unsqueeze(-1)).expand(-1, -1, 16)
    state1 = state1_in + 1.0   (accumulates each inference)
    state2 = state2_in + 1.0   (accumulates each inference)

With latent=0, audio equals the pre-inference state1[0] value (same as
upsample_state.pt) so the same correctness assertion applies.  The second
state tensor is present purely to expose the out-of-bounds adj-array access
that the single-state-tensor model avoids by accident.
"""

import os
import torch


class UpsampleWithMultiState(torch.nn.Module):
    def forward(
        self,
        latent: torch.Tensor,
        state1: torch.Tensor,
        state2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio = (latent + state1[:, :1].unsqueeze(-1)).expand(-1, -1, 16).contiguous()
        return audio, state1 + 1.0, state2 + 1.0


def main() -> None:
    model = UpsampleWithMultiState()
    model.eval()

    latent = torch.zeros(1, 1, 1)
    state1 = torch.zeros(1, 4)
    state2 = torch.zeros(1, 4)

    traced = torch.jit.trace(model, (latent, state1, state2))

    out_path = os.path.join(os.path.dirname(__file__), "upsample_multistate.pt")
    traced.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
