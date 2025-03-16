![anira Logo](/docs/img/anira-tilde-logo.png)

---

## Description

The `anira~` external integrates the [anira](https://github.com/anira-project/anira) library to offer neural network inference inside max msp. It currently supports the following inference engines: `LibTorch`, `ONNXRuntime`, and `TensorFlow Lite`. At runtime a configuration file can be submitted to the external to load a model.

The configuration file is a JSON file that contains the following fields:

| Necessary Fields   | Description                                                                                                                                                                                                                                                                                                                                                                                 |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| modelpath          | Path to the corresponding neural network model file (`.onnx`, `.ts`, `.tflite`, `.pt`). <br/> - Windows: `C:/Users/user/Documents/model.tflite` <br/> - macOS: `/Users/user/Documents/model.tflite`                                                                                                                                                                                             |
| backend            | Inference engine to use. Currently the following inference engines are available:<br/> - LibTorch: `LIBTORCH` <br/> - ONNX Runtime: `ONNX` <br/> - TensorFlow Lite: `TFLITE`                                                                                                                                                                                                                |
| input_shape        | Shape of the input tensor, including the number of input samples and batch size. <br/> - Example: `[1, 2048, 1]`                                                                                                                                                                                                                                                                            |
| output_shape       | Shape of the output tensor, including the number of output samples and batch size. <br/> - Example: `[1, 2048, 1]`                                                                                                                                                                                                                                                                          |
| max_inference_time | The maximum time (in milliseconds) the inference needs to process the data with the given input and output shapes. If unsure, start with a high value (which results in higher latency) and gradually reduce it. Alternatively, use [anira's benchmarking](https://github.com/anira-project/anira/blob/main/docs/benchmark-usage.md) to automatically evaluate this. <br/>- Example: `63.1` |

The corresponding `.json` file for the model should look like this:
```json
{
  "modelpath": "C:/Users/user/Documents/model.tflite",
  "backend": "TFLITE",
  "input_shape": [ 1, 2048, 1 ],
  "output_shape": [ 1, 2048, 1 ],
  "max_inference_time": 63.1
}
```

## Supported Neural Network Models

The external currently supports all neural network models that are capable of real-time processing and are configured to receive and output audio data.

Out of the box, `anira~` supports networks that process the same number of input samples as output samples. If the model requires more input samples than output samples (e.g., due to a receptive field), the source code of the external must be modified, and a custom `anira::PrePostProcessor` class must be implemented.

For detailed instructions on how to write a custom `anira::PrePostProcessor` class, please refer to the [anira user guide](https://github.com/anira-project/anira/blob/main/docs/anira-usage.md#step-2-create-a-prepostprocessor-instance).

## Examples

To get started with `anira~`, three examples are provided in the `examples` directory. Each example contains a config file and a `.txt` with the download url for the model. **Note:** Before you can run it, you'll need to download the model and change the path in the config file.

### Build in examples

- [RAVE Darbouka (LibTorch)](examples/rave_darbouka_libtorch/): Config for loading a RAVE model with the `LibTorch` inference engine.
- [RNN Amp Emulation (TFLite)](examples/anira_rnn_amp_emulation_tflite/): Config for loading a RNN model with the `TFLite` inference engine.
- [CNN Amp Emulation (ONNX)](examples/anira_cnn_amp_emulation_onnx/): Config for loading a CNN model with the `ONNX` inference engine. **Note:** This model requires a custom `anira::PrePostProcessor`, which we have to add at compile time. The implementation is provided in the example.

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