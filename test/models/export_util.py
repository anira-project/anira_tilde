"""Shared export helper for the test models.

Every test model is exported in three formats so the test suite can run on
whichever anira backends are compiled in:

  <name>.pt    TorchScript (trace)      — LIBTORCH backend (opt-in builds)
  <name>.onnx  ONNX                     — ONNX Runtime backend
  <name>.pte   ExecuTorch program       — EXECUTORCH backend

All models are fixed-shape, so no dynamic axes are involved. Requires the
`torch`, `onnx` and `executorch` packages (see requirements below); regenerate
with any Python >= 3.10:

    python -m venv .venv && .venv/bin/pip install torch executorch onnx
    for f in create_*.py; do .venv/bin/python "$f"; done
"""

import os

import torch


def export_all(model: torch.nn.Module, example_inputs: tuple, name: str) -> None:
    """Export `model` as <name>.pt, <name>.onnx and <name>.pte next to this file."""
    model.eval()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    traced = torch.jit.trace(model, example_inputs)
    traced.save(base + ".pt")
    print(f"Saved {base}.pt")

    torch.onnx.export(
        model,
        example_inputs,
        base + ".onnx",
        dynamo=False,  # legacy TorchScript exporter: no extra deps, fine for fixed shapes
    )
    print(f"Saved {base}.onnx")

    from executorch.exir import to_edge

    exported = torch.export.export(model, example_inputs)
    program = to_edge(exported).to_executorch()
    with open(base + ".pte", "wb") as f:
        f.write(program.buffer)
    print(f"Saved {base}.pte")
