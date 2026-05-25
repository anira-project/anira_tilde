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
| `input_size <  output_size` | **Upsample** | latent → audio decoder; one inference fires per output-size boundary, anira drains the larger output across following callbacks. |
| `input_size >  output_size` | **Downsample** | audio → latent encoder; anira accumulates `input_size` host samples, pops one output value, sample-and-holds it across the host buffer. |

**Constraints:**

- `N` must be a positive integer. Non-integer ratios are not supported.
- Sizes of `0` mean "non-streamable" (message or state tensor); rate
  adaptation doesn't apply to those.
- Latent-to-latent models (both `input_size` and `output_size` below host
  rate) can't be represented purely by the size fields; they'd need an
  explicit `inference_period_samples` field which isn't currently
  implemented.

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
- The 1-sample scratch + S&H state used by the downsample path.

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
