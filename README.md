![anira Logo](/.github/assets/anira-tilde-logo.png)

---

## Description

The `anira~` and `mc.anira~` externals integrate the [anira](https://github.com/anira-project/anira) library to offer neural network inference inside Max/MSP. `ONNX Runtime`, `LiteRT` (TensorFlow Lite), and `ExecuTorch` are statically bundled inside the external — nothing to install, no shared libraries to ship. `LibTorch` is available as an opt-in build (`-DANIRA_WITH_LIBTORCH=ON`, shared library).

The external is initialized dynamically from a JSON configuration file passed as the first argument. The config determines the number of streamable (signal) and non-streamable (message) inlets and outlets the object exposes.

Minimal example:

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "models/my_model.onnx", "inference_backend": "ONNX" }
        ],
        "tensor_shape": [
            {
                "input_shape":  [[1, 1, 512]],
                "output_shape": [[1, 1, 512]]
            }
        ],
        "max_inference_time": 5.0
    }
}
```

## Documentation

| Doc | Topic |
|---|---|
| [JSON config reference](.github/docs/json-config-reference.md) | Full schema for `inference_config` + `state_config` + `resampler_config`, every field, worked examples. |
| [Rate adaptation](.github/docs/rate-adaptation.md) | How `anira~` handles latent ↔ audio block-size mismatches, classification rules, multi-frame blocks, latency. |
| [Build instructions](.github/docs/build.md) | CMake presets, project layout, packaging, sanitizers. |

State-passing (RNN/LSTM models), rate adaptation, sample-rate conversion, and relative model paths are documented in detail in the JSON config reference. The
[anira project documentation](https://anira-project.github.io/anira/) covers the underlying inference library.

## Examples

The [`examples/`](examples) directory ships working configs for every bundled backend — amp emulation (CNN and RNN), a sine oscillator, and the RAVE djembe model (forward, encoder, decoder). The model files are downloaded automatically next to their configs at CMake configure time, so every example loads without editing any path (disable with `-DANIRA_TILDE_DOWNLOAD_EXAMPLE_MODELS=OFF` for offline builds).

## Build

```bash
git clone --recurse-submodules https://github.com/anira-project/anira_tilde.git
cd anira_tilde
cmake --preset desktop-release
cmake --build --preset desktop-release
```

The built external lands at `externals/anira~.mxo` (macOS) or `externals/anira~.mxe64` (Windows). Drop the repo into `~/Documents/Max 8/Packages/` to use it from Max.

See [`.github/docs/build.md`](.github/docs/build.md) for the full reference: presets, Windows specifics, tests, sanitizers, packaging.

## License

This project is licensed under [Apache 2.0](LICENSE).
