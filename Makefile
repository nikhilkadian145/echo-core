.PHONY: dev test build pipeline lint

dev:
	tauri dev

test:
	cargo test --manifest-path src-tauri/Cargo.toml & pytest pipeline/tests/

build:
	tauri build

pipeline:
	bash pipeline/venv_setup.sh

lint:
	cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
	ruff check pipeline/
