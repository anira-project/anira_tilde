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