set windows-shell := ["powershell.exe", "-Command"]
set export := true
ORT_DYLIB_PATH := "C:/Apps/onnxruntime/lib/onnxruntime.dll"

# --- Lints:

check: fmt clippy test

fmt:
    cargo fmt --all

clippy:
    cargo clippy --no-deps --all-features --tests --benches -- \
        -D clippy::all \
        -D clippy::pedantic \
        -D clippy::nursery

# --- Misc:

clean:
    cargo clean

setup:
    uv run py-yolo/export_onnx.py

# --- Execution:

test:
    cargo test --profile release

bench:
    cargo bench

run:
    cargo run --bin run_onnx_v2 --profile release
