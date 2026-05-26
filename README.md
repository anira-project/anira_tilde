![anira Logo](/.github/assets/anira-tilde-logo.png)

---

## Description

The `anira~` external integrates the [anira](https://github.com/anira-project/anira) library to offer neural network inference inside Max/MSP. It supports the `LibTorch`, `ONNXRuntime`, and `TensorFlow Lite` inference engines.

The external is initialized dynamically from a JSON configuration file passed as the first argument. The config determines the number of streamable (signal) and non-streamable (message) inlets and outlets the object exposes.

Minimal example:

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "models/my_model.pt", "inference_backend": "LIBTORCH" }
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
| [JSON config reference](.github/docs/json-config-reference.md) | Full schema for `inference_config` + `state_config`, every field, worked examples. |
| [Rate adaptation](.github/docs/rate-adaptation.md) | How `anira~` handles latent ↔ audio (different model and host sample rates), classification rules, latency. |
| [Build instructions](.github/docs/build.md) | CMake presets, project layout, packaging, sanitizers. |

State-passing (RNN/LSTM models), rate adaptation, and relative model paths are documented in detail in the JSON config reference. The
[anira project documentation](https://anira-project.github.io/anira/) covers the underlying inference library.

## Examples

The [`examples/`](examples) directory ships a handful of working configs. Each contains a `.json` config and a `.txt` with the download URL for the corresponding model — download the model and update `model_path` in the JSON before loading.

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
