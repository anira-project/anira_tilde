![anira Logo](/docs/img/anira-tilde-logo.png)

---

## Description

The `anira~` external integrates the [anira](https://github.com/anira-project/anira) library to offer neural network inference inside Max. It currently supports the following inference engines: `LibTorch`, `ONNXRuntime`, and `TensorFlow Lite`.

The external is initialized dynamically based on a JSON configuration file provided as argument, which determines the number of streamable (signal) and non-streamable (message) inlets and outlets of the object.

The configuration file has a top-level `inference_config` object with the following fields:

| Field (inside `inference_config`) | Required | Description |
|------------------------------------|----------|-------------|
| `model_data` | yes | Array of objects, each with `model_path` (path to `.onnx`, `.ts`, `.tflite`, or `.pt`) and `inference_backend` (`LIBTORCH`, `ONNX`, or `TFLITE`). |
| `tensor_shape` | yes | Array of objects, each with `input_shape` and `output_shape` — arrays of per-tensor shapes (each shape is itself an array). |
| `max_inference_time` | yes | Maximum inference time in milliseconds. If unsure, start high (higher latency) and reduce gradually. See [anira's benchmarking](https://github.com/anira-project/anira/blob/main/docs/benchmark-usage.md). |
| `processing_spec` | no | Object with `preprocess_input_channels`, `preprocess_input_size`, `postprocess_output_channels`, and `postprocess_output_size` — one entry per tensor. Values > 0 in `*_size` define streamable signal inlets/outlets; 0 defines non-streamable message inlets/outlets. Auto-computed from tensor shapes when omitted (all tensors treated as single-channel signals). |

The corresponding `.json` file for the model should look like this:

Example configuration structure:
```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "/path/to/model.pt", "inference_backend": "LIBTORCH" }
        ],
        "tensor_shape": [
            {
                "input_shape": [[1, 1, 512]],
                "output_shape": [[1, 1, 512]]
            }
        ],
        "max_inference_time": 5.0
    }
}
```

`processing_spec` is optional for simple models — the library auto-computes it from the tensor shapes, treating all tensors as single-channel signal tensors. It is required when you have non-streamable (message) tensors mixed with signal tensors.
*(Note: For comprehensive documentation on the anira library and configuration file structure, please visit: [https://anira-project.github.io/anira/](https://anira-project.github.io/anira/))*

## State-passing models

Models that pass internal state between inferences (e.g. RNNs/LSTMs) can be configured with an optional `state_config` block. This tells `anira~` which output tensor holds the new state and which input tensor expects the previous state. After each inference the state output is automatically fed back as the state input — no manual routing in Max is needed.

State tensor indices are **global**: they index into the full `input_shape` / `output_shape` arrays (0-based, counting both signal and non-signal tensors). State tensors must have `preprocess_input_size` / `postprocess_output_size` of `0` (non-streamable). They do not appear as Max inlets or outlets. The initial state on the first inference is zeros.

```json
{
    "inference_config": {
        "model_data": [{ "model_path": "/path/to/model.pt", "inference_backend": "LIBTORCH" }],
        "tensor_shape": [{
            "input_shape":  [[1, 1, 512], [1, 32]],
            "output_shape": [[1, 1, 512], [1, 32]]
        }],
        "processing_spec": {
            "preprocess_input_channels":   [1, 1],
            "postprocess_output_channels": [1, 1],
            "preprocess_input_size":       [512, 0],
            "postprocess_output_size":     [512, 0]
        },
        "max_inference_time": 10.0
    },
    "state_config": {
        "state_pairs": [
            { "output_tensor": 1, "input_tensor": 1 }
        ]
    }
}
```

In this example the model takes `(audio[512], state[32])` and returns `(audio[512], state[32])`. Output tensor 1 (the state) is fed back as input tensor 1 automatically. Multiple state pairs can be listed for models with more than one state tensor.

> **Note:** state tensors should appear after any non-state message tensors in the shape arrays. If a state tensor appears before a non-state message tensor, the message inlet/outlet index mapping will be incorrect.

## Examples

To get started with `anira~`, examples are provided in the `examples` directory. Each example contains a config file and a `.txt` with the download url for the model. **Note:** Before you can run it, you'll need to download the model and change the path in the config file.

### Build in examples

...

## Build instructions

### Clone repository and load submodules

```bash
git clone https://github.com/anira-project/anira_tilde
cd anira_tilde
git submodule update --init --recursive
```
### Debug build

```bash
cmake . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug --config Debug --target anira_tilde
```

### Release build

```bash
cmake . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release --target anira_tilde
```

### Build options

By default, all three inference engines are installed. You can disable specific backends as needed:

- LibTorch: `-DANIRA_WITH_LIBTORCH=OFF`
- OnnxRuntime: `-DANIRA_WITH_ONNXRUNTIME=OFF`
- TensorFlow Lite: `-DANIRA_WITH_TFLITE=OFF`

More build options are documented in the [anira repository](https://github.com/anira-project/anira).

## License

This project is licensed under [Apache 2.0](LICENSE).