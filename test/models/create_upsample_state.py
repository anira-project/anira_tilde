"""
Generate a TorchScript model for rate-adaptation + state-passing integration tests.

This matches the sashimi decoder pattern:
    - tensor 0 (streamable): input_size=1 latent frame → output_size=16 audio samples
    - tensor 1 (state):      non-streamable, passed back between inferences

Model signature:
    forward(latent: Tensor[1, 1, 1], state: Tensor[1, 4])
        -> (audio: Tensor[1, 1, 16], state: Tensor[1, 4])

Behaviour:
    audio  = (latent + state[:, :1, None]).expand(-1, -1, 16)
             (output = latent value + first state element, repeated 16 times)
    state  = state_in + 1.0              (accumulates each inference)

The audio output depends on the state so that state passing can be verified
purely from the audio output: with latent=0, audio equals the pre-inference
state value and increases by 1 each inference.
"""

import os
import torch

from export_util import export_all


class UpsampleWithState(torch.nn.Module):
    def forward(
        self, latent: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state[:, :1] has shape [1, 1]; unsqueeze(-1) → [1, 1, 1] for broadcast
        audio = (latent + state[:, :1].unsqueeze(-1)).expand(-1, -1, 16).contiguous()
        return audio, state + 1.0


def main() -> None:
    model = UpsampleWithState()
    latent = torch.zeros(1, 1, 1)
    state  = torch.zeros(1, 4)

    export_all(model, (latent, state), "upsample_state")


if __name__ == "__main__":
    main()
