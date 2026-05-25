# TODO

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
