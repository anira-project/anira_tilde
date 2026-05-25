# `anira~` JSON Config Reference

`anira~` loads a single JSON file that describes the model, its tensor I/O
shape, and optional state-passing wiring. The top-level structure is:

```jsonc
{
    "inference_config": { … },   // required — what to run and how
    "state_config":     { … }    // optional — only for stateful models
}
```

---

## `inference_config`

Required. Tells anira which model file(s) to load and how to feed them.

| Field | Required | Type | Description |
|---|---|---|---|
| [`model_data`](#model_data) | yes | array of object | One entry per backend / model file. |
| [`tensor_shape`](#tensor_shape) | yes | array of object | Per-backend tensor shape descriptions. |
| [`max_inference_time`](#max_inference_time) | yes | number (ms) | Hard latency budget for one inference. |
| [`processing_spec`](#processing_spec) | no | object | Per-tensor I/O semantics (streamable vs. non-streamable, channel counts). Auto-derived when omitted. |
| [`model_function`](#model_function) | no | string | LibTorch / scripted models can expose alternative entry points. |
| [`num_parallel_processors`](#num_parallel_processors) | no | integer | Override anira's worker pool size. Forced to `1` automatically when `state_config` is present. |

### `model_data`

Array of objects. Each entry binds a model file to a backend:

```jsonc
"model_data": [
    { "model_path": "models/my_model.pt", "inference_backend": "LIBTORCH" },
    { "model_path": "models/my_model.onnx", "inference_backend": "ONNX" }
]
```

| Field | Required | Description |
|---|---|---|
| `model_path` | yes | Path to the model file (`.pt` / `.ts` for LibTorch, `.onnx` for ONNX, `.tflite` for TensorFlow Lite). Relative paths are resolved against the JSON file's directory, not the working directory. |
| `inference_backend` | yes | One of `LIBTORCH`, `ONNX`, `TFLITE`. |

The first matching entry whose backend is supported in the current build is loaded.

### `tensor_shape`

Array of objects, one per backend you want to ship with. Each describes the
shape of every input and output tensor:

```jsonc
"tensor_shape": [
    {
        "backend":     "LIBTORCH",                                 // optional
        "input_shape":  [[1, 1, 512], [1, 32]],                    // 2 input tensors
        "output_shape": [[1, 1, 512], [1, 32]]                     // 2 output tensors
    }
]
```

- `input_shape` / `output_shape` are arrays of arrays. The outer array is per
  tensor; the inner array is the shape of that tensor.
- `backend` is optional. If a single shape entry covers every backend (the
  common case), omit it; otherwise tag each entry.

### `max_inference_time`

A number in **milliseconds**. anira treats this as the *budget per inference*
and uses it to compute scheduling latency. If your inference actually exceeds
this on a given block anira reports "missing samples" and emits zeros for
that block.

If you don't know the right value, **start high (e.g. 40 ms)** and reduce
it once measurements stabilize. See
[anira's benchmark usage](https://github.com/anira-project/anira/blob/main/docs/benchmark-usage.md).

### `processing_spec`

Optional. Describes how each tensor is presented to the Max patch:

```jsonc
"processing_spec": {
    "preprocess_input_channels":   [1, 1],   // per input tensor
    "preprocess_input_size":       [512, 0], // per input tensor; 0 → message inlet
    "postprocess_output_channels": [1, 1],
    "postprocess_output_size":     [512, 0]
}
```

| Field | Description |
|---|---|
| `preprocess_input_channels`  | Number of channels presented as inlets per input tensor. |
| `preprocess_input_size`      | Block size per input tensor. `> 0` → signal (audio) inlet. `0` → message inlet (single value or list per element). |
| `postprocess_output_channels` | Same, for outputs. |
| `postprocess_output_size`     | Same, for outputs. |

When `processing_spec` is omitted, anira auto-computes it from the tensor
shapes, treating every tensor as a single-channel signal at host rate.

### `model_function`

Optional string. Some LibTorch scripted models expose multiple methods
(e.g. `forward`, `encode`, `decode`). Set `model_function` to the method
name to call something other than `forward`.

### `num_parallel_processors`

Optional integer. Lets anira parallelise inferences across multiple worker
threads. Defaults to anira's own heuristic. **Forced to `1`** automatically
when `state_config` is present (state passing requires sequential
inferences).

---

## `state_config` (optional)

Use this for stateful models (RNNs, LSTMs, anything that threads a state
tensor from one inference to the next). With `state_config`, `anira~`
feeds the named output tensor back into the named input tensor between
inferences — no patching required in Max.

```jsonc
"state_config": {
    "state_pairs": [
        { "output_tensor": 1, "input_tensor": 1 }
    ]
}
```

| Field | Description |
|---|---|
| `state_pairs` | Array of `{ "output_tensor": N, "input_tensor": M }` mappings. After each inference, output tensor `N` is copied into input tensor `M` so the next inference reads it. |

**Rules / caveats:**

- State tensor indices are **global**, indexing into the full `input_shape`
  / `output_shape` arrays (0-based, counting both signal and non-signal
  tensors).
- A state tensor must be **non-streamable**: its
  `preprocess_input_size` / `postprocess_output_size` must be `0`.
- State tensors are **not** exposed as Max inlets / outlets — they're
  handled internally.
- Initial state on the first inference is zeros.
- State tensors should be declared *after* any non-state message tensors in
  the shape arrays — putting them earlier shifts the message inlet/outlet
  index mapping.

---

## Rate-adapting (latent ↔ audio) models

There is **no extra config field** for rate adaptation. `anira~` detects it
automatically from the per-tensor sizes:

- `preprocess_input_size[i] < postprocess_output_size[i]`
  → **upsample** (latent → audio). One inference fires per
  `postprocess_output_size` boundary; the larger output drains across
  subsequent callbacks.
- `preprocess_input_size[i] > postprocess_output_size[i]`
  → **downsample** (audio → latent). `anira~` accumulates input samples,
  fires inference, and sample-and-holds the single output value across the
  host buffer.
- `preprocess_input_size[i] == postprocess_output_size[i]` → host rate, no
  adaptation.

The ratio `N = max / min` must be a positive integer. Non-integer ratios
are not supported.

---

## Worked examples

### Mono in / mono out, single tensor

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "models/passthrough.pt", "inference_backend": "LIBTORCH" }
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

### Stateful (RNN-style), 1 audio + 1 state tensor

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "models/rnn.pt", "inference_backend": "LIBTORCH" }
        ],
        "tensor_shape": [
            {
                "input_shape":  [[1, 1, 512], [1, 32]],
                "output_shape": [[1, 1, 512], [1, 32]]
            }
        ],
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

### Latent → audio decoder (rate adaptation)

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "models/decoder.pt", "inference_backend": "LIBTORCH" }
        ],
        "tensor_shape": [
            {
                "input_shape":  [[1, 64, 1]],
                "output_shape": [[1, 1, 2048]]
            }
        ],
        "processing_spec": {
            "preprocess_input_channels":   [64],
            "postprocess_output_channels": [1],
            "preprocess_input_size":       [1],
            "postprocess_output_size":     [2048]
        },
        "max_inference_time": 40.0
    }
}
```

---

## See also

- [anira library docs](https://anira-project.github.io/anira/) — full
  reference for backends, scheduling, and benchmarking.
- [`docs/anira~.maxref.xml`](anira~.maxref.xml) — Max-side help reference.
