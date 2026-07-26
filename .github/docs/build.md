# Building `anira~` from source

## Quick build (macOS / Linux)

```bash
git clone --recurse-submodules https://github.com/anira-project/anira_tilde.git
cd anira_tilde
cmake --preset desktop-release
cmake --build --preset desktop-release
```

The Max external lands at `externals/anira~.mxo` (macOS) or
`externals/anira~.mxe64` (Windows). Drop the whole repo (or the staged
package — see [Packaging](#packaging) below) into
`~/Documents/Max 8/Packages/` to use it from Max.

## Quick build (Windows)

```powershell
git clone --recurse-submodules https://github.com/anira-project/anira_tilde.git
cd anira_tilde
cmake --preset windows-release
cmake --build --preset windows-release
```

## Requirements

| Tool | Min version | Notes |
|---|---|---|
| CMake | 3.19 | 3.25+ recommended (for preset features) |
| C++ compiler | C++20 capable | Apple Clang ≥ 13, MSVC 2022, GCC 11, Clang 14 |
| Ninja | any | Used by all desktop presets |
| git | any | Submodules are fetched recursively |

macOS additionally needs Xcode command-line tools.

## Presets

Defined in [`CMakePresets.json`](../../CMakePresets.json):

| Preset | What it does |
|---|---|
| `desktop-release` | Host-native release build (used for macOS arm64, Linux) |
| `macos-x64-release` | macOS Intel cross-compile (single-arch x86_64) |
| `windows-release` | Windows x64 MSVC release |
| `desktop-debug` | Host-native debug build with tests on |
| `desktop-debug-asan` | Debug + AddressSanitizer + UBSan (enum sub-check off) |
| `desktop-debug-tsan` | Debug + ThreadSanitizer |
| `desktop-debug-rtsan` | Debug + RealtimeSanitizer (needs Clang 20 — `brew install llvm@20`) |

Each preset has matching build and test presets:

```bash
cmake --preset desktop-debug
cmake --build --preset desktop-debug
ctest --preset desktop-debug
```

## Build options

| Option | Default | Effect |
|---|---|---|
| `ANIRA_WITH_LIBTORCH` | `OFF` | Adds the LibTorch backend. Unlike the always-on backends (ONNX Runtime, LiteRT, ExecuTorch — statically bundled inside the external), LibTorch is a large shared library that must ship next to the external; it stays opt-in because distributing it on macOS/Windows is painful. |
| `ANIRA_TILDE_DOWNLOAD_EXAMPLE_MODELS` | `ON` | Downloads every example model next to its JSON config at configure time (versioned — stale models from an older manifest are re-fetched). Turn off for offline builds; everything except the examples works without them. |
| `ANIRA_TILDE_WITH_TESTS` | `OFF` | Builds the test suite (on in the debug/sanitizer presets). |

## Tests

Disabled by default. Enable with the `ANIRA_TILDE_WITH_TESTS` option (already
set ON in the debug + sanitizer presets):

```bash
cmake --preset desktop-debug                              # configure (tests on)
cmake --build --preset desktop-debug --target anira_tilde_tests  # build just the tests
ctest --preset desktop-debug                             # run tests
```

The test target links only against `anira_tilde_core` (the host-agnostic
library), so the Min-API isn't needed for running tests. Passing
`--target anira_tilde_tests` also skips relinking the Max external.

### Debug builds don't disturb an installed external

The debug and sanitizer presets redirect `C74_LIBRARY_OUTPUT_DIRECTORY` into
their own build tree (e.g. `build/desktop/Debug/externals/`), so the
`libanira.*` and external they produce never overwrite the **release** builds in
`externals/`. (The external links `libanira` from its
own directory at runtime, so without this a debug — or sanitizer-instrumented —
`libanira` would silently replace the release build.)

Only `desktop-release` / `*-release` presets write to the source `externals/`.
To point any build at a custom location, override the variable:

```bash
cmake --preset desktop-debug -DC74_LIBRARY_OUTPUT_DIRECTORY=/some/where/externals
```

### Slow build flavours (sanitizers, Rosetta-emulated x86_64)

Under sanitizers anira's inference can run several times slower than
release. Per-test `max_inference_time` is generated at configure time from
a base + overhead:

```bash
cmake --preset desktop-debug-asan -DANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD=500
```

The matching CI presets bake `500` in already.

## Project layout

```
include/anira_tilde/
├── Engine.h                     entry point — host plugs in here
├── Exports.h                    Windows DLL export macros
├── anira_tilde.h                umbrella header (includes everything below)
├── inference/
│   ├── Session.h                anira pipeline + state passing wrapper
│   └── TensorLayout.h           per-model channel/size metadata + lookups
├── mixing/
│   └── Mixer.h                  dry/wet mixer with latency-aligned delay
├── rate_adaptation/
│   ├── EffectiveBufferSize.h    derive the right block size for prepare()
│   └── RateAdaptor.h            up/downsample views around anira's I/O
└── state_passing/
    ├── StatePairParser.h        JSON → state pair list
    └── StatePassingPrePostProcessor.h
                                 feeds state outputs back as state inputs

src/         …                   matching .cpp files (Engine.cpp at the top)
max/         AniraTilde.{h,cpp}  Max/MSP wrapper (Min-API). The only
                                 directory that touches Max types.
cmake/       *.cmake             toolchain helpers + per-target setup
test/        *.cpp + .json.in    gtest suite, exercises the core library
.github/     workflows/, docs/   CI + non-shipping documentation
```

## CMake targets

- **`anira_tilde_core`** (STATIC) — pure C++ library. No Max dependency.
  Tests and any future host (e.g. JUCE plugin) link this.
- **`anira_tilde`** (MODULE / Max external) — Min-API wrapper around
  `anira_tilde_core`. Casts double↔float at the host boundary, owns
  inlets/outlets, forwards everything else to `Engine`.

## Packaging

CI builds produce per-platform Max packages (`anira_tilde-<platform>.zip`)
containing the layout Max expects:

```
anira_tilde/
├── package-info.json
├── icon.png
├── README.md
├── LICENSE
├── externals/anira~.mxo  (or anira~.mxe64)
├── help/anira~.maxhelp
├── docs/anira~.maxref.xml
└── examples/
```

See [`.github/workflows/build.yml`](../workflows/build.yml) for the
staging step.

## Sanitizers in CI

Two top-level workflow jobs:

- **`build`** — release builds across all three platforms.
- **`sanitizers`** — ASan/UBSan, TSan, RTSan on macos-latest. RTSan
  installs `llvm@20` via Homebrew because Apple Clang ships an older
  toolchain that doesn't include the realtime sanitizer.
