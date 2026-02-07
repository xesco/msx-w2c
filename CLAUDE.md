# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**wav2cas** — An MSX WAV-to-CAS converter written in Rust. It reads audio recordings of MSX cassette tapes (WAV files) and produces `.cas` files (the standard MSX cassette image format). The tool uses auto-adaptive signal analysis to handle varying recording quality.

## Build & Run

```bash
cargo build                          # debug build
cargo build --release                # release build
cargo run -- input.wav               # convert input.wav → input.cas
cargo run -- input.wav -o out.cas    # specify output
cargo run -- input.wav --lenient     # relaxed thresholds for poor recordings
cargo run -- input.wav --debug       # per-bit diagnostic output
cargo test                           # run tests (currently in dsp.rs)
```

## Architecture

Three-module pipeline processing WAV samples into CAS bytes:

### `main.rs` — Orchestration & CLI
- CLI parsing (clap), WAV loading (hound), signal pre-scan, and the main decode loop
- **Pre-scan analysis**: Measures peak amplitude, noise floor, SNR over configurable duration. All downstream parameters (silence thresholds, phase coefficients, etc.) are adapted from SNR via linear interpolation (`lerp`)
- **Pilot detection**: Runs a preliminary header frequency detection to calibrate baud range and quality thresholds before the main loop
- **Main loop**: Repeats header-detect → bit-read → byte-decode cycle for each block in the tape

### `dsp.rs` — Signal Processing
- **GoertzelDetector**: Sliding-window single-frequency power measurement (translated from C `lib/audio/frequency_detector.c`). Used both for header detection and bit reading
- **BitReader**: Dual-Goertzel + PLL bit recovery. Runs two detectors at 1200Hz (bit-0) and 2400Hz (bit-1). Three phases: init fill → header-end detection (waits for 1200Hz) → PLL-based bit reading with adaptive silence detection and power hold to prevent ratchet-down

### `decoder.rs` — Byte Framing & Protocol
- **Decoder state machine**: `ExpectFirstStart → ExpectStart → ReadByte → ExpectStop1 → ExpectStop2` (11-bit UART: 1 start, 8 data LSB-first, 2 stop)
- **Header detection** (`detect_header_frequency`): Goertzel-based iterative frequency narrowing — scans blocks, refines center frequency via weighted power average, resets on poor-quality blocks, requires `header_length` seconds of consecutive good signal
- **File type detection**: Identifies MSX tape blocks (ASCII `0xEA×10`, Binary `0xD0×10`, Basic `0xD3×10`) and extracts 6-char filenames
- **CAS header**: 8-byte magic `[1F A6 DE BA CC 13 7D 74]` written before each block

## Key Design Decisions

- All processing happens on fully-loaded mono f64 samples (no streaming) — allows pre-scan and multiple passes
- Parameters adapt to recording quality: SNR factor (0.0–1.0) interpolates between "poor signal" and "clean signal" defaults
- `--lenient` mode halves silence_ratio and header_end_sensitivity, and fills small gaps with 0x00 bytes
- Leader zone detection uses consecutive-ones counting in `ExpectStart` (>10 ones = back in header tone = end of block)
- Binary blocks track 6 address bytes (load/end/exec LE u16s) to know exact expected data size for truncation
