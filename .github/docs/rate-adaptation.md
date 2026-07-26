# Rate adaptation in `anira~`

Many ML audio models don't run at the host's audio rate. A latent decoder
might consume 1 frame of latent data and produce 2048 audio samples; a
feature encoder might do the opposite. `anira~` handles both cases
transparently — no extra JSON field, no extra patching in Max.

## How it's detected

The ratio is derived from the per-tensor sizes in
[`processing_spec`](json-config-reference.md#processing_spec):

```
N = max(postprocess_output_size, preprocess_input_size)
  / min(postprocess_output_size, preprocess_input_size)
```

For every signal tensor, `anira~` classifies the relationship between
`preprocess_input_size` and `postprocess_output_size` exactly once at
`prepare()` time:

| Relationship | Classification | Behaviour |
|---|---|---|
| `input_size == output_size` | **Equal** | tensor runs at host rate; pass straight through. |
| `input_size <  output_size` | **Upsample** | latent → audio decoder; one inference fires per `output_size` boundary of host time — zero, one, or several per callback, depending on how the host buffer compares to the model's output block. |
| `input_size >  output_size` | **Downsample** | audio → latent encoder; anira accumulates `input_size` host samples per inference, and one output block is popped per `input_size` boundary; each popped frame is sample-and-held across its own sub-segment of the host stream. |

**Constraints:**

- `N` must be a positive integer. Non-integer ratios are not supported.
- Sizes of `0` mean "non-streamable" (message or state tensor); rate
  adaptation doesn't apply to those.
- Latent-to-latent models (both `input_size` and `output_size` below host
  rate) can't be represented purely by the size fields; they'd need an
  explicit `inference_period_samples` field which isn't currently
  implemented.

## Multi-frame blocks

A model block may carry several frames of the slower stream. The RAVE
exports are the canonical case: the decoder consumes `[1, 2, 8]` — 8
latent frames per 1024 output samples, one frame per `1024/8 = 128`
samples of host time. The adaptor preserves that per-frame timing:

- **Upsample gather** — frame `j` of an input block is picked from the
  host signal at `boundary + j * output_size / input_size`, i.e. at the
  frame stride, not consecutively. Consecutive picking would hand the
  model 8 copies of one latent value.
- **Downsample hold** — frame `j` of a popped output block is held across
  host samples `[j, j+1) * input_size / output_size` after its boundary.
  A phase counter carries a partially-played block across callback
  boundaries, so segment alignment survives any host buffer size.

With one frame per block both rules reduce to the plain
gather-at-boundary / sample-and-hold behaviour.

Rate adaptation is about **block sizes**, not sample rates: a model
trained at a fixed sample rate additionally declares
[`resampler_config`](json-config-reference.md#resampler_config-optional),
which wraps the whole pipeline (rate adaptor included) in a host ↔ model
sample-rate conversion.

## Why this matters

Without rate adaptation, anira's scheduler fires one inference per host
buffer regardless of the model's actual block size, which under- or
overflows the internal ring buffers. The result is silence, dropouts, or
the wrong number of samples making it to the output. The rate adaptor
sits between Max's audio thread and anira's `push_data` / `pop_data`,
re-presenting the host buffer in the shape anira's scheduler actually
wants this block.

## Where it lives in the code

`anira_tilde::RateAdaptor` (`include/anira_tilde/rate_adaptation/RateAdaptor.h`)
owns:

- The per-tensor `Kind` enum (precomputed at `prepare()` time).
- The "view" arrays that re-present the host's pointers to anira each block.
- The gather scratch (upsample) and the pop scratch + currently-playing
  block + phase counter (downsample).

`Session::process` drives it in three phases:

```cpp
m_rate_adaptor.pre_dispatch(layout, in, in_n, out, out_n);   // build views
m_inference_handler.push_data / pop_data(...);                // anira call
m_rate_adaptor.post_dispatch(layout, out, out_n);             // S&H fixup
```

When every tensor is `Kind::Equal` the adaptor degenerates to a plain
pass-through dispatch with no extra work beyond the pointer copies.

## Latency

For upsample tensors anira reports a non-trivial latency (a full
`output_size` window plus safety margin). The reported value is
forwarded through the latency outlet of the external — patch-side, you
can use it for delay compensation.

## Validating your config

The repo's test suite (`test_rate_adaptation.cpp`) covers both
directions. Two of the bundled examples exercise rate adaptation
directly:

- `test/rate_adapt_test.json` — upsampler x32
- `test/downsample_test.json` — downsampler x32

See [examples in the build dir](build.md#tests) for how to run them.
