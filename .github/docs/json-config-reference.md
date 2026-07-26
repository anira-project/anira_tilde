# `anira~` JSON Config Reference

`anira~` loads a single JSON file that describes the model, its tensor I/O
shape, optional state-passing wiring, and an optional model sample rate.
The top-level structure is:

```jsonc
{
    "inference_config": { … },   // required — what to run and how
    "state_config":     { … },   // optional — only for stateful models
    "resampler_config": { … }    // optional — model trained at a fixed sample rate
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
| [`model_function`](#model_function) | no | string (per `model_data` entry) | LibTorch and ExecuTorch models can expose alternative entry points. |
| [`num_parallel_processors`](#num_parallel_processors) | no | integer | Override anira's worker pool size. Forced to `1` automatically when `state_config` is present. |

### `model_data`

Array of objects. Each entry binds a model file to a backend:

```jsonc
"model_data": [
    { "model_path": "models/my_model.onnx", "inference_backend": "ONNX" },
    { "model_path": "models/my_model.pte", "inference_backend": "EXECUTORCH" }
]
```

| Field | Required | Description |
|---|---|---|
| `model_path` | yes | Path to the model file (`.onnx` for ONNX Runtime, `.tflite` for LiteRT, `.pte` for ExecuTorch, `.pt` / `.ts` for LibTorch). Relative paths are resolved against the JSON file's directory, not the working directory. |
| `inference_backend` | yes | One of `ONNX`, `LITERT`, `EXECUTORCH`, `LIBTORCH` (`TFLITE` is accepted as a legacy alias for the TensorFlow Lite path). |
| `model_function` | no | Named entry point to call instead of `forward` — supported for LibTorch scripted models and multi-method ExecuTorch programs. |

The first matching entry whose backend is supported in the current build is
loaded. `ONNX`, `LITERT`, and `EXECUTORCH` are always compiled in (statically
bundled); `LIBTORCH` requires a build with `-DANIRA_WITH_LIBTORCH=ON`.

### `tensor_shape`

Array of objects, one per backend you want to ship with. Each describes the
shape of every input and output tensor:

```jsonc
"tensor_shape": [
    {
        "inference_backend": "ONNX",                               // optional
        "input_shape":  [[1, 1, 512], [1, 32]],                    // 2 input tensors
        "output_shape": [[1, 1, 512], [1, 32]]                     // 2 output tensors
    }
]
```

- `input_shape` / `output_shape` are arrays of arrays. The outer array is per
  tensor; the inner array is the shape of that tensor.
- `inference_backend` is optional. If a single shape entry covers every
  backend (the common case), omit it; otherwise tag each entry — useful when
  one backend's export declares a tensor differently (e.g. a scalar as
  `[1, 1, 1]` instead of `[1]`).

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

Optional string, set per `model_data` entry. Some models expose multiple
entry points (e.g. `forward`, `encode`, `decode`): LibTorch scripted
modules and multi-method ExecuTorch `.pte` programs both support this. Set
`model_function` to the method name to call something other than `forward`.

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
  `postprocess_output_size` boundary — several per callback when the host
  buffer is larger than the model's output block.
- `preprocess_input_size[i] > postprocess_output_size[i]`
  → **downsample** (audio → latent). `anira~` accumulates input samples and
  pops one output block per `preprocess_input_size` boundary; each popped
  frame is sample-and-held across its own sub-segment of the host stream.
- `preprocess_input_size[i] == postprocess_output_size[i]` → host rate, no
  adaptation.

The ratio `N = max / min` must be a positive integer. Non-integer ratios
are not supported.

A block may carry **several frames** of the slower stream (e.g. the RAVE
encoder emits `[1, 2, 8]` — 8 latent frames per 1024-sample call, one per
128 samples). Frame timing is preserved: each frame occupies its own
`1024/8 = 128`-sample segment of the latent signal, on both the gather
(decoder input) and hold (encoder output) side. See
[rate-adaptation.md](rate-adaptation.md) for the mechanics.

---

## `resampler_config` (optional)

For models trained at a fixed sample rate. When the block is present **and**
the host runs at a different rate, `anira~` builds a host ↔ model
sample-rate conversion shim (libsamplerate) around the whole pipeline;
when the host rate matches `model_sample_rate` (or the block is absent),
the shim is bypassed entirely — no resampler objects, no added latency.
There is no on/off switch beyond this: presence of the block plus an
actual rate mismatch is what enables it.

```jsonc
"resampler_config": {
    "model_sample_rate": 44100,
    "quality": "sinc_fastest",          // sinc_best | sinc_medium | sinc_fastest | linear | hold
    "input_quality":  { "0": "hold" },  // optional per-tensor overrides (tensor index → quality)
    "output_quality": { "0": "hold" }
}
```

| Field | Required | Description |
|---|---|---|
| `model_sample_rate` | yes | The rate the model was trained at. `<= 0` (or absent) disables resampling. |
| `quality` | no | Converter for all streamable tensors. Default `sinc_fastest`. |
| `input_quality` / `output_quality` | no | Per-tensor overrides, keyed by tensor index. Use `hold` (zero-order hold) for latent tensors — sinc interpolation is meaningless for frame-paced control signals. |

State and message tensors are never resampled. The resampler's priming
delay is measured exactly at `prepare()` time and folded into the latency
reported by the external, so delay compensation stays sample-accurate.
The bundled `rave_djembe` examples use this to run a 44.1 kHz model at any
host rate.

---

## Worked examples

### Mono in / mono out, single tensor

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "models/passthrough.onnx", "inference_backend": "ONNX" }
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
            { "model_path": "models/rnn.onnx", "inference_backend": "ONNX" }
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

### Latent → audio decoder (rate adaptation + resampling)

Two latent channels, 8 frames per 1024-sample block, a state tensor fed
back between inferences, and a model rate of 44.1 kHz (see the bundled
`examples/rave_djembe/` configs for the full family):

```json
{
    "inference_config": {
        "model_data": [
            { "model_path": "rave_decoder.onnx", "inference_backend": "ONNX" }
        ],
        "tensor_shape": [
            {
                "input_shape":  [[1, 2, 8], [1, 154464]],
                "output_shape": [[1, 1, 1024], [1, 154464]]
            }
        ],
        "processing_spec": {
            "preprocess_input_channels":   [2, 1],
            "postprocess_output_channels": [1, 1],
            "preprocess_input_size":       [8, 0],
            "postprocess_output_size":     [1024, 0]
        },
        "max_inference_time": 12.0
    },
    "state_config": {
        "state_pairs": [
            { "output_tensor": 1, "input_tensor": 1 }
        ]
    },
    "resampler_config": {
        "model_sample_rate": 44100,
        "quality": "sinc_fastest",
        "input_quality": { "0": "hold" }
    }
}
```

---

## See also

- [anira library docs](https://anira-project.github.io/anira/) — full
  reference for backends, scheduling, and benchmarking.
- [`docs/anira~.maxref.xml`](anira~.maxref.xml) — Max-side help reference.
