mod decoder;
mod dsp;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use hound::WavReader;

use decoder::{Action, CAS_HEADER, Decoder, HeaderConfig, detect_header_frequency};
use dsp::{Bit, BitReader, BitReaderConfig};

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "wav2cas", about = "MSX WAV to CAS converter with auto-adaptive signal analysis")]
struct Cli {
    /// Input WAV file
    input: PathBuf,

    /// Output CAS file (default: input with .cas extension)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Force baud rate (skip header auto-detection)
    #[arg(long)]
    baud: Option<f64>,

    /// Try harder: lower thresholds, ignore framing errors, fill gaps
    #[arg(long)]
    lenient: bool,

    /// Pre-scan duration for signal analysis in seconds (default: 60.0)
    #[arg(long, default_value_t = 60.0)]
    scan: f64,

    /// Manual volume override (disables auto-gain)
    #[arg(short = 'v', long = "volume")]
    volume: Option<f64>,

    /// Show per-bit decisions with power levels and confidence
    #[arg(long)]
    debug: bool,

    /// Minimal output (errors only)
    #[arg(long)]
    quiet: bool,
}

// =============================================================================
// Pre-Scan Signal Analysis
// =============================================================================

struct SignalAnalysis {
    peak_amplitude: f64,
    noise_floor: f64,
    snr_db: f64,
    auto_gain: f64,
}

fn prescan_signal(samples: &[f64], sample_rate: u32, scan_seconds: f64) -> SignalAnalysis {
    let scan_count = ((sample_rate as f64 * scan_seconds) as usize).min(samples.len());
    let scan = &samples[..scan_count];

    // Peak amplitude
    let peak = scan.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);

    // Noise floor: RMS of quietest 10% of 50ms blocks
    let block_size = (sample_rate as f64 * 0.05) as usize;
    let block_size = block_size.max(1);
    let mut block_rms_values: Vec<f64> = Vec::new();

    let mut i = 0;
    while i + block_size <= scan_count {
        let block = &scan[i..i + block_size];
        let rms = (block.iter().map(|s| s * s).sum::<f64>() / block.len() as f64).sqrt();
        block_rms_values.push(rms);
        i += block_size;
    }

    block_rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quietest_count = (block_rms_values.len() as f64 * 0.1).ceil() as usize;
    let quietest_count = quietest_count.max(1).min(block_rms_values.len());
    let noise_floor =
        block_rms_values[..quietest_count].iter().sum::<f64>() / quietest_count as f64;

    let snr_db = if noise_floor > 0.0 {
        20.0 * (peak / noise_floor).log10()
    } else {
        60.0 // effectively infinite SNR
    };

    let auto_gain = if peak > 0.0 {
        (2.0 / peak).min(20.0)
    } else {
        1.0
    };

    SignalAnalysis {
        peak_amplitude: peak,
        noise_floor,
        snr_db,
        auto_gain,
    }
}

fn snr_quality(snr: f64) -> &'static str {
    if snr >= 30.0 {
        "excellent"
    } else if snr >= 20.0 {
        "good"
    } else if snr >= 10.0 {
        "fair"
    } else {
        "poor"
    }
}

// =============================================================================
// Time Formatting
// =============================================================================

fn format_time(sample_pos: usize, sample_rate: u32) -> String {
    let seconds = sample_pos as f64 / sample_rate as f64;
    let minutes = (seconds / 60.0) as u32;
    let secs = seconds - minutes as f64 * 60.0;
    format!("{:2}:{:07.4}", minutes, secs)
}

// =============================================================================
// Block Decode
// =============================================================================

fn decode_block(
    processed: &[f64],
    pos: usize,
    sample_rate: u32,
    header_freq: f64,
    bit_config: &BitReaderConfig,
    dec: &mut Decoder,
    out: &mut BufWriter<File>,
    total_bytes_written: &mut usize,
    verbose: bool,
    debug: bool,
    lenient: bool,
) -> usize {
    let mut pos = pos;
    let mut bit_reader = BitReader::new(sample_rate, header_freq, bit_config.clone());
    dec.reset_for_block();

    let mut block = BlockState::new(pos);

    while pos < processed.len() {
        let sample = processed[pos];
        pos += 1;

        if let Some(decision) = bit_reader.process_sample(sample) {
            if debug {
                let bit_label = match decision.bit {
                    Bit::Zero => "0",
                    Bit::One => "1",
                    Bit::Silence => "S",
                };
                eprintln!(
                    "[{}] BIT={} p0={:.1} p1={:.1} conf={:.2}",
                    format_time(pos, sample_rate),
                    bit_label,
                    decision.power_zero,
                    decision.power_one,
                    decision.confidence,
                );
            }

            let action = dec.process_bit(&decision);

            match action {
                Action::DataFound => {
                    if verbose && !block.data_found_printed {
                        eprintln!("[{}] Data found", format_time(pos, sample_rate));
                        block.data_found_printed = true;
                    }
                    block.consecutive_silences = 0;
                }
                Action::ByteComplete(byte) => {
                    block.buffer.push(byte);
                    block.consecutive_silences = 0;

                    if !block.file_type_printed {
                        if let Some((ft, name)) = dec.take_file_type() {
                            if verbose {
                                eprintln!(
                                    "[{}] {} \"{}\"",
                                    format_time(pos, sample_rate),
                                    ft.name(),
                                    name
                                );
                            }
                            block.file_type_printed = true;
                        }
                    }

                    if !block.binary_info_printed {
                        if let Some(info) = dec.check_binary_info() {
                            if verbose {
                                eprintln!(
                                    "[{}]   Load: 0x{:04X}, End: 0x{:04X}, Exec: 0x{:04X}",
                                    format_time(pos, sample_rate),
                                    info.load,
                                    info.end,
                                    info.exec
                                );
                            }
                            if info.end >= info.load {
                                block.expected_data_size =
                                    Some(6 + (info.end as usize - info.load as usize + 1));
                            }
                            block.binary_info_printed = true;
                        }
                    }

                    if debug && block.buffer.len() % 50 == 0 {
                        eprintln!(
                            "[{}] BYTE #{:04}: 0x{:02X}",
                            format_time(pos, sample_rate),
                            block.buffer.len(),
                            byte
                        );
                    }
                }
                Action::SilenceDetected => {
                    if lenient {
                        block.consecutive_silences += 1;
                        if block.consecutive_silences <= 3 {
                            block.buffer.push(0x00);
                            dec.reset_for_block();
                            continue;
                        }
                    }

                    flush_block(out, &mut block, total_bytes_written, pos, sample_rate, verbose);
                    if verbose {
                        eprintln!("[{}] Silence detected", format_time(pos, sample_rate));
                        eprintln!();
                    }
                    break;
                }
                Action::Done => {
                    flush_block(out, &mut block, total_bytes_written, pos, sample_rate, verbose);
                    break;
                }
                Action::Continue => {
                    block.consecutive_silences = 0;
                }
            }
        }
    }

    // Flush any remaining buffered bytes (block terminated by EOF)
    if !block.buffer.is_empty() {
        flush_block(out, &mut block, total_bytes_written, pos, sample_rate, verbose);
    }

    pos
}

// =============================================================================
// Parameter Adaptation
// =============================================================================

fn adapt_parameters(
    snr_db: f64,
    detected_baud: Option<f64>,
    pilot_quality: Option<f64>,
    lenient: bool,
) -> (BitReaderConfig, HeaderConfig) {
    let snr_factor = ((snr_db - 10.0) / 20.0).clamp(0.0, 1.0);

    let mut bit_config = BitReaderConfig {
        silence_ratio: lerp(0.03, 0.01, snr_factor),
        silence_threshold: lerp(6.0, 3.0, snr_factor).round() as usize,
        header_end_sensitivity: lerp(0.25, 0.125, snr_factor),
        phase_adjust_coeff: lerp(0.3, 0.5, snr_factor),
        power_hold_ratio: lerp(0.10, 0.25, snr_factor),
    };

    let mut header_config = HeaderConfig::default();

    if let Some(baud) = detected_baud {
        header_config.min_bauds = baud * 0.9;
        header_config.max_bauds = baud * 1.1;
    }
    if let Some(quality) = pilot_quality {
        header_config.quality_threshold = quality * 0.4;
    }

    if lenient {
        bit_config = bit_config.with_lenient();
        header_config.quality_threshold *= 0.5;
    }

    (bit_config, header_config)
}

fn print_analysis_summary(
    analysis: &SignalAnalysis,
    scan_seconds: f64,
    total_duration: f64,
    gain: f64,
    manual_volume: bool,
    detected_baud: Option<f64>,
    pilot_quality: Option<f64>,
    bit_config: &BitReaderConfig,
    header_config: &HeaderConfig,
) {
    let defaults_bit = BitReaderConfig::default();
    let defaults_hdr = HeaderConfig::default();

    let mark = |val: f64, def: f64| -> &'static str {
        if (val - def).abs() > 1e-6 { " *" } else { "" }
    };
    let mark_usize = |val: usize, def: usize| -> &'static str {
        if val != def { " *" } else { "" }
    };

    eprintln!();
    eprintln!(
        "Signal analysis ({:.1}s pre-scan):",
        scan_seconds.min(total_duration)
    );
    eprintln!("  Peak amplitude:    {:.2}", analysis.peak_amplitude);
    eprintln!("  Noise floor:       {:.2} (RMS)", analysis.noise_floor);
    eprintln!(
        "  SNR:               {:.0} dB ({})",
        analysis.snr_db,
        snr_quality(analysis.snr_db)
    );
    if manual_volume {
        eprintln!("  Gain:              {:.1}x (manual)", gain);
    } else {
        eprintln!("  Gain:              {:.1}x *", gain);
    }
    if let Some(baud) = detected_baud {
        eprintln!("  Detected baud:     {:.0} *", baud);
    }
    if let Some(quality) = pilot_quality {
        eprintln!("  Pilot quality:     {:.1} *", quality);
    }
    eprintln!("  Min bauds:         {:.0}{}", header_config.min_bauds,
        mark(header_config.min_bauds, defaults_hdr.min_bauds));
    eprintln!("  Max bauds:         {:.0}{}", header_config.max_bauds,
        mark(header_config.max_bauds, defaults_hdr.max_bauds));
    eprintln!("  Quality threshold: {:.1}{}", header_config.quality_threshold,
        mark(header_config.quality_threshold, defaults_hdr.quality_threshold));
    eprintln!("  Silence ratio:     {:.3}{}", bit_config.silence_ratio,
        mark(bit_config.silence_ratio, defaults_bit.silence_ratio));
    eprintln!("  Silence tolerance: {}{}", bit_config.silence_threshold,
        mark_usize(bit_config.silence_threshold, defaults_bit.silence_threshold));
    eprintln!("  Header end sens:   {:.3}{}", bit_config.header_end_sensitivity,
        mark(bit_config.header_end_sensitivity, defaults_bit.header_end_sensitivity));
    eprintln!("  Phase adjust:      {:.2}{}", bit_config.phase_adjust_coeff,
        mark(bit_config.phase_adjust_coeff, defaults_bit.phase_adjust_coeff));
    eprintln!("  Power hold:        {:.2}{}", bit_config.power_hold_ratio,
        mark(bit_config.power_hold_ratio, defaults_bit.power_hold_ratio));

    let any_changed = !manual_volume
        || detected_baud.is_some()
        || (header_config.quality_threshold - defaults_hdr.quality_threshold).abs() > 1e-6
        || (bit_config.silence_ratio - defaults_bit.silence_ratio).abs() > 1e-6
        || bit_config.silence_threshold != defaults_bit.silence_threshold
        || (bit_config.header_end_sensitivity - defaults_bit.header_end_sensitivity).abs() > 1e-6
        || (bit_config.phase_adjust_coeff - defaults_bit.phase_adjust_coeff).abs() > 1e-6
        || (bit_config.power_hold_ratio - defaults_bit.power_hold_ratio).abs() > 1e-6;
    if any_changed {
        eprintln!("  * adapted from pre-scan");
    }
    eprintln!();
}

// =============================================================================
// WAV Loading
// =============================================================================

fn load_wav(path: &std::path::Path, verbose: bool) -> (Vec<f64>, u32) {
    let reader = match WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Cannot open {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    let bits = spec.bits_per_sample;

    if verbose {
        eprintln!(
            "WAV: {} Hz, {}-bit, {}",
            sample_rate,
            bits,
            if channels == 1 { "mono" } else { "stereo" }
        );
    }

    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (bits - 1)) as f64;
            let raw: Vec<i32> = reader.into_samples::<i32>().filter_map(|s| s.ok()).collect();
            downmix_to_mono(&raw, channels, |s| *s as f64 / max_val)
        }
        hound::SampleFormat::Float => {
            let raw: Vec<f32> = reader.into_samples::<f32>().filter_map(|s| s.ok()).collect();
            downmix_to_mono(&raw, channels, |s| *s as f64)
        }
    };

    if samples.is_empty() {
        eprintln!("Error: WAV file contains no samples");
        std::process::exit(1);
    }

    (samples, sample_rate)
}

// =============================================================================
// Main Decode
// =============================================================================

fn main() {
    let cli = Cli::parse();

    let output_path = cli.output.clone().unwrap_or_else(|| {
        let mut p = cli.input.clone();
        p.set_extension("cas");
        p
    });

    let verbose = !cli.quiet;

    if verbose {
        eprintln!("wav2cas - MSX WAV to CAS converter");
    }

    // Load WAV file
    let (all_samples, sample_rate) = load_wav(&cli.input, verbose);

    // Pre-scan signal analysis
    let analysis = prescan_signal(&all_samples, sample_rate, cli.scan);
    let gain = cli.volume.unwrap_or(analysis.auto_gain);

    // Apply gain (no bandpass filter — matches C reference behavior)
    let processed: Vec<f64> = all_samples
        .iter()
        .map(|&s| s * gain)
        .collect();

    // Track max volume after processing
    let max_volume = processed.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);

    // Pilot header detection on processed samples to calibrate parameters
    let pilot_config = HeaderConfig {
        quality_threshold: 1.0,
        ..HeaderConfig::default()
    };
    let mut pilot_iter = processed.iter().copied();
    let pilot_result = detect_header_frequency(&mut pilot_iter, sample_rate, &pilot_config);

    let detected_baud = pilot_result.as_ref().map(|r| r.frequency * 0.5);
    let pilot_quality = pilot_result.as_ref().map(|r| r.quality);

    // Adapt all parameters from pre-scan
    let (bit_config, header_config) = adapt_parameters(
        analysis.snr_db, detected_baud, pilot_quality, cli.lenient,
    );

    if verbose {
        let total_duration = all_samples.len() as f64 / sample_rate as f64;
        print_analysis_summary(
            &analysis, cli.scan, total_duration,
            gain, cli.volume.is_some(),
            detected_baud, pilot_quality,
            &bit_config, &header_config,
        );
    }

    // Open output file
    let out_file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: Cannot create {}: {}", output_path.display(), e);
            std::process::exit(1);
        }
    };
    let mut out = BufWriter::new(out_file);

    // Main decode loop
    let mut pos = 0_usize;
    let mut total_bytes_written = 0_usize;
    let mut dec = Decoder::new();

    loop {
        if pos >= processed.len() {
            break;
        }

        // Phase 1: Header detection
        let header_result = if let Some(forced_baud) = cli.baud {
            // Skip header detection; consume some samples to sync
            let freq = forced_baud * 2.0;
            let skip = (sample_rate as f64 * 0.1) as usize;
            pos += skip.min(processed.len() - pos);
            Some((freq, 0.0, pos))
        } else {
            let mut sample_iter = processed[pos..].iter().copied();
            match detect_header_frequency(&mut sample_iter, sample_rate, &header_config) {
                Some(result) => {
                    let consumed = result.samples_consumed;
                    pos += consumed;
                    Some((result.frequency, result.quality, pos))
                }
                None => {
                    // No more headers found - EOF
                    pos = processed.len();
                    None
                }
            }
        };

        let (header_freq, quality, _) = match header_result {
            Some(r) => r,
            None => break,
        };

        let baud_rate = header_freq * 0.5;

        if verbose {
            eprintln!(
                "[{}] Header detected ({:.2}Bd, quality: {:.2})",
                format_time(pos, sample_rate),
                baud_rate,
                quality
            );
        }

        // Write CAS header
        if out.write_all(&CAS_HEADER).is_err() {
            eprintln!("Error: Failed to write CAS header");
            std::process::exit(1);
        }
        total_bytes_written += CAS_HEADER.len();

        // Phase 2: Bit reading + decoding
        pos = decode_block(
            &processed, pos, sample_rate, header_freq, &bit_config,
            &mut dec, &mut out, &mut total_bytes_written,
            verbose, cli.debug, cli.lenient,
        );
    }

    // Flush output
    if out.flush().is_err() {
        eprintln!("Error: Failed to flush output");
        std::process::exit(1);
    }

    if verbose {
        eprintln!("[{}] End of file", format_time(processed.len(), sample_rate));
        eprintln!("Max volume: {:.6}", max_volume);
        eprintln!(
            "Output: {} ({} bytes)",
            output_path.display(),
            total_bytes_written
        );
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn lerp(poor: f64, clean: f64, factor: f64) -> f64 {
    poor + (clean - poor) * factor
}

struct BlockState {
    buffer: Vec<u8>,
    start_pos: usize,
    data_found_printed: bool,
    file_type_printed: bool,
    binary_info_printed: bool,
    consecutive_silences: usize,
    expected_data_size: Option<usize>,
}

impl BlockState {
    fn new(start_pos: usize) -> Self {
        Self {
            buffer: Vec::new(),
            start_pos,
            data_found_printed: false,
            file_type_printed: false,
            binary_info_printed: false,
            consecutive_silences: 0,
            expected_data_size: None,
        }
    }
}

/// Truncate buffer if expected size is known, write to output, print stats.
/// Returns number of data bytes written.
fn flush_block(
    out: &mut BufWriter<File>,
    block: &mut BlockState,
    total_bytes_written: &mut usize,
    pos: usize,
    sample_rate: u32,
    verbose: bool,
) -> usize {
    if let Some(expected) = block.expected_data_size {
        block.buffer.truncate(block.buffer.len().min(expected));
    }
    let n = block.buffer.len();
    if n > 0 {
        if out.write_all(&block.buffer).is_err() {
            eprintln!("Error: Failed to write data");
            std::process::exit(1);
        }
        *total_bytes_written += n;

        if verbose {
            let elapsed_samples = pos - block.start_pos;
            let elapsed = elapsed_samples as f64 / sample_rate as f64;
            let rate = if elapsed > 0.0 { n as f64 / elapsed } else { 0.0 };
            eprintln!(
                "[{}] Reading data ({} bytes) [{:.1} B/s]",
                format_time(pos, sample_rate),
                n,
                rate
            );
        }
    }
    block.buffer.clear();
    n
}



fn downmix_to_mono<T, F>(raw: &[T], channels: usize, to_f64: F) -> Vec<f64>
where
    F: Fn(&T) -> f64,
{
    if channels == 1 {
        raw.iter().map(&to_f64).collect()
    } else {
        raw.chunks(channels)
            .map(|frame| {
                let sum: f64 = frame.iter().map(&to_f64).sum();
                sum / channels as f64
            })
            .collect()
    }
}
