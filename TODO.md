# TODO

## Features

- [ ] **Internal sample-rate conversion (resampler)** — anira treats the model's
  expected sample rate as fixed, so a model trained at e.g. 44.1 kHz misbehaves
  when Max's audio engine runs at 48 kHz / 96 kHz. Add a resampler in
  `AniraProcessor` (or `Mixer`) that converts host → model rate on the way in
  and model → host rate on the way out. The model's expected sample rate could
  be declared in the JSON config (`processing_spec.model_sample_rate` or
  similar). Need to decide on a library (libsamplerate, r8brain, or a small
  hand-rolled polyphase FIR) — must be real-time safe.

## Distribution

- [ ] **Codesigning & notarization (macOS)** — the `.mxo` is currently ad-hoc signed only.
  For distribution outside the Package Manager, sign with a Developer ID Application
  certificate and notarize via `notarytool`. Needs Apple Developer account secrets
  (`APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD`) wired into the CI workflow,
  plus the signing identity imported into the runner's keychain. Windows external
  may also benefit from an Authenticode signature.

- [ ] **Tag-push release automation** — add a `release` job to `.github/workflows/build.yml`
  gated on `startsWith(github.ref, 'refs/tags/v')`. It should download the three
  platform package artifacts and publish them as assets on a GitHub Release (draft
  by default), and sync `package-info.json` `version` with the tag.

- [ ] **Cycling '74 Package Manager submission** — once the project is stable and a
  signed/notarized release is available, submit through the Cycling '74 portal so
  `anira~` appears in Max's built-in Package Manager. One-time review process.

- [ ] **Windows: ship backend DLLs separately + side-load at runtime** — the libtorch /
  onnxruntime / tflite DLLs are large (libtorch alone is hundreds of MB). Bundling
  them inside the package's `externals/` directory bloats every distribution and
  risks DLL hell when multiple Max packages ship overlapping copies. Instead build
  a Windows installer (e.g. WiX / Inno Setup) that places the backend DLLs in a
  fixed shared location (e.g. `%ProgramFiles%\anira\bin\` or `%ProgramData%\anira\bin\`),
  and have `anira~.mxe64` add that directory to the DLL search path at load time
  via `AddDllDirectory()` + `LoadLibraryEx(..., LOAD_LIBRARY_SEARCH_USER_DIRS)`,
  or register the path via an "App Paths" registry key. This is the standard
  "side-loading" pattern Windows audio plugins use for heavy shared dependencies.

## Upstream anira issues to follow up

- [x] **UBSan: `InferenceConfig::get_tensor_shape` reads uninitialized `m_backend`** —
  fixed upstream on `feat/executorch-support`: `TensorShape::m_backend` is now
  default-initialized (the universal constructor used to leave it uninitialized).
  The `-fno-sanitize=enum` workaround is removed from `cmake/sanitizers.cmake`.

## Housekeeping

- [ ] **`package-info.json` `homepatcher`** — currently points at
  `"C++ Object Development Kit.maxpat"`, which does not exist in the repo. Either
  ship an actual homepatcher (a `.maxpat` at the package root or under `extras/`)
  or remove the field so the Package Manager doesn't fail to open it.
