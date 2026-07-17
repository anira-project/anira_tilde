# RAVE Djembe (ONNX, split encoder / decoder)

Two `anira~` instances replicate the classic nn~ RAVE patch — audio →
`encode` → latent manipulation → `decode` → audio — using ONNX models with
**explicit streaming state**. The state feedback is wired internally via
`state_config`; nothing to patch.

Models come from
[example-models/RaveDjembe](https://github.com/anira-project/example-models/tree/main/third-party/ircam-acids/RAVE/RaveDjembe)
(fetched at configure time). 44.1 kHz mono, 2048 samples per latent frame.

## Patch layout

```
[anira~ rave_djembe_encoder.json]   audio inlet → 4 latent signal outlets
        |  (latents are sample-and-hold signals; manipulate with +~ / *~)
[anira~ rave_djembe_decoder.json]   4 latent inlets + 1 noise inlet → audio outlet
```

## The decoder's noise inlet

The model's noise generator takes its uniform noise as an *input* (that is
what makes the graph deterministic and portable). The third decoder inlet is
a host-rate signal inlet — exactly one noise value per output sample:

- **unconnected** → zeros → the noise branch is silent (slightly drier
  sound; the branch sits ~34 dB below the signal)
- **`noise~`** → matches the original TorchScript model's internal
  `torch.rand` behavior
- anything else (`pink~`, enveloped noise, `*~` for amount) → creative
  control over the noise texture that the original model doesn't expose
