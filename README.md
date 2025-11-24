![anira Logo](/docs/img/anira-tilde-logo.png)

---

## Description

The `anira~` external integrates the [anira](https://github.com/anira-project/anira) library to offer neural network inference inside Max. It currently supports the following inference engines: `LibTorch`, `ONNXRuntime`, and `TensorFlow Lite`.

The external is initialized dynamically based on a JSON configuration file provided as argument, which determines the number of streamable (signal) and non-streamable (message) inlets and outlets of the object.

The configuration file contains the following fields:

| Primary Fields   | Description                                                                                                                                                                                                                                                                                                                                                                                 |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_path          | Path to the corresponding neural network model file (`.onnx`, `.ts`, `.tflite`, `.pt`). <br/> - Windows: `C:/Users/user/Documents/model.tflite` <br/> - macOS: `/Users/user/Documents/model.tflite`                                                                                                                                                                                             |
| inference_backend   | Inference engine to use. Currently the following inference engines are available:<br/> - LibTorch: `LIBTORCH` <br/> - ONNX Runtime: `ONNX` <br/> - TensorFlow Lite: `TFLITE`                                                                                                                                                                                                                |
| input_shape        | Shape of the input tensor, including the number of input samples and batch size. <br/> - Example: `[1, 2048, 1]`                                                                                                                                                                                                                                                                            |
| output_shape       | Shape of the output tensor, including the number of output samples and batch size. <br/> - Example: `[1, 2048, 1]`                                                                                                                                                                                                                                                                          |
| preprocess_input_size | Defines the buffer size for input tensors. Values > 0 define streamable signals and values of 0 define non-streamable messages. |
| postprocess_output_size | Defines the buffer size for output tensors. Values > 0 define streamable signals and values of 0 define non-streamable messages. |
| max_inference_time | The maximum time (in milliseconds) the inference needs to process the data with the given input and output shapes. If unsure, start with a high value (which results in higher latency) and gradually reduce it. Alternatively, use [anira's benchmarking](https://github.com/anira-project/anira/blob/main/docs/benchmark-usage.md) to automatically evaluate this. <br/>- Example: `63.1` |

The corresponding `.json` file for the model should look like this:

Example configuration structure:
```json
{
    "model_path": "/path/to/model.pt",
    "inference_backend": "LIBTORCH",
    "max_inference_time": 5.0,
    "tensor_shape": [
      {
        "input_shape": [[1, 1, 512],[1]],
        "output_shape": [[1, 1, 512],[1]]
      }
    ],
    "processing_spec": {
        "preprocess_input_channels": [1, 1],
        "postprocess_output_channels": [1, 1],
        "preprocess_input_size": [512, 0],
        "postprocess_output_size": [512, 0]
    },
}
```
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