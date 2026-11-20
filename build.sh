#!/bin/sh
# Build the gpuatlas wgpu viewer (web) and the native benchmark.
#
# If a workspace-local toolchain exists at ../toolchain (rustup + cargo +
# wasm-pack, as in the author's development tree), it is used; otherwise
# the system rustup/cargo and a wasm-pack found on PATH are used.
set -e
cd "$(dirname "$0")"

if [ -d ../toolchain ]; then
  export RUSTUP_HOME="$PWD/../toolchain/rustup"
  export CARGO_HOME="$PWD/../toolchain/cargo"
  export PATH="$CARGO_HOME/bin:$PWD/../toolchain/wasm-pack/wasm-pack-v0.13.1-x86_64-unknown-linux-musl:$PATH"
fi

# Release builds must not leak local absolute paths into the shipped
# .wasm (dependency panic-location strings). Remap the checkout directory
# and the cargo home to neutral placeholders. Both prefixes must be
# canonicalized first (pwd -P): rustc records canonical paths, so a
# prefix containing ".." (e.g. "$PWD/../toolchain/cargo") never matches.
ROOT_DIR=$(pwd -P)
REMAP="--remap-path-prefix $ROOT_DIR=/gpuatlas"
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
if [ -d "$CARGO_HOME_DIR" ]; then
  CARGO_HOME_DIR=$(cd "$CARGO_HOME_DIR" && pwd -P)
  REMAP="$REMAP --remap-path-prefix $CARGO_HOME_DIR=/cargo"
fi
export RUSTFLAGS="$REMAP"

# Web build -> pkg/ (.wasm + JS glue), loaded by index.html.
wasm-pack build --release --target web --out-dir pkg

# Native headless benchmark binary (target/release/bench).
cargo build --release --bin bench

# Report the wasm size, raw and gzipped.
WASM=pkg/gpuatlas_bg.wasm
echo "wasm raw : $(wc -c < "$WASM") bytes"
echo "wasm gzip: $(gzip -c "$WASM" | wc -c) bytes"
