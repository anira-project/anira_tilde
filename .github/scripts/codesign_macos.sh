#!/usr/bin/env bash
#
# Codesign the macOS anira~ externals with a Developer ID Application identity
# and the hardened runtime. Signs every mach-o dylib and every .mxo bundle
# found in the given externals directory, inside-out (loose dylibs first, then
# the bundle). Notarization + stapling happen afterwards in the release
# workflow — signing must run first (and after any lipo, which invalidates
# signatures).
#
# Required environment:
#   SIGN_IDENTITY   Developer ID Application identity, e.g.
#                   "Developer ID Application: Anira Project (TEAMID)"
#
# Usage: codesign_macos.sh <externals-dir>

set -euo pipefail

EXTERNALS_DIR="${1:?usage: codesign_macos.sh <externals-dir>}"
: "${SIGN_IDENTITY:?SIGN_IDENTITY (Developer ID Application) must be set}"

if [[ ! -d "$EXTERNALS_DIR" ]]; then
  echo "error: '$EXTERNALS_DIR' is not a directory" >&2
  exit 1
fi

sign() {
  codesign --force --timestamp --options runtime --sign "$SIGN_IDENTITY" "$1"
}

echo "==> Codesigning mach-o in '$EXTERNALS_DIR' as: $SIGN_IDENTITY"

# 1. Loose dylibs first. -type f skips the version symlinks so each real file is
#    signed exactly once.
while IFS= read -r -d '' lib; do
  echo "    dylib  $lib"
  sign "$lib"
done < <(find "$EXTERNALS_DIR" -type f -name '*.dylib' -print0)

# 2. Then the .mxo bundle(s), so their sealed contents are already signed.
while IFS= read -r -d '' bundle; do
  echo "    bundle $bundle"
  sign "$bundle"
done < <(find "$EXTERNALS_DIR" -maxdepth 1 -type d -name '*.mxo' -print0)

echo "==> Verifying signatures"
while IFS= read -r -d '' bundle; do
  codesign --verify --deep --strict --verbose=2 "$bundle"
done < <(find "$EXTERNALS_DIR" -maxdepth 1 -type d -name '*.mxo' -print0)

echo "==> macOS codesigning complete."
