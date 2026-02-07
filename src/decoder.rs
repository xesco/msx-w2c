use crate::dsp::{Bit, BitDecision, GoertzelDetector};

// =============================================================================
// CAS Constants
// =============================================================================

pub const CAS_HEADER: [u8; 8] = [0x1F, 0xA6, 0xDE, 0xBA, 0xCC, 0x13, 0x7D, 0x74];

const MSX_ASCII_ID: [u8; 10] = [0xEA; 10];
const MSX_BINARY_ID: [u8; 10] = [0xD0; 10];
const MSX_BASIC_ID: [u8; 10] = [0xD3; 10];

const HEADER_ANALYZE_BYTES: usize = 16;
const FILENAME_OFFSET: usize = 10;
const FILENAME_LENGTH: usize = 6;
const BINARY_ADDR_BYTES: usize = 6;

// Maximum consecutive 1-bits in ExpectStart before declaring leader zone.
// In valid framing, ExpectStart sees 0 ones (start bit comes immediately after
// stop2). PLL jitter adds at most 1-2. Leader zone produces many more.
const MAX_ONES_IN_EXPECT_START: usize = 10;

// Header detection constants
const BAUD_TO_FREQ_MULTIPLIER: f64 = 2.0;
const MIN_RANGE_THRESHOLD: f64 = 1.0;

// =============================================================================
// File Type Detection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileType {
    Ascii,
    Binary,
    Basic,
    Custom,
}

impl FileType {
    pub fn name(&self) -> &'static str {
        match self {
            FileType::Ascii => "ASCII",
            FileType::Binary => "BINARY",
            FileType::Basic => "BASIC",
            FileType::Custom => "CUSTOM",
        }
    }
}

// =============================================================================
// Decoder Actions (returned to caller for output/control)
// =============================================================================

#[derive(Debug)]
pub enum Action {
    Continue,
    ByteComplete(u8),
    SilenceDetected,
    DataFound,
    Done,
}

// =============================================================================
// Binary address info (returned by check_binary_info)
// =============================================================================

#[derive(Debug)]
pub struct BinaryInfo {
    pub load: u16,
    pub end: u16,
    pub exec: u16,
}

// =============================================================================
// Decoder State Machine
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    ExpectFirstStart,
    ExpectStart,
    ReadByte { bits_remaining: u8, value: u8 },
    ExpectStop1,
    ExpectStop2,
    Done,
}

pub struct Decoder {
    state: State,
    byte_count: usize,
    header_buffer: Vec<u8>,
    // Binary address tracking
    binary_addr_count: i32, // -1 = not tracking
    binary_addr_buffer: [u8; BINARY_ADDR_BYTES],
    last_file_type: Option<FileType>,
    current_file_type: Option<FileType>,
    pending_file_type: Option<(FileType, String)>,
    // Consecutive 1-bits seen in ExpectStart state only.
    // Used to detect leader zone (>10 means pure header tone).
    consecutive_ones: usize,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            state: State::ExpectFirstStart,
            byte_count: 0,
            header_buffer: Vec::with_capacity(HEADER_ANALYZE_BYTES),
            binary_addr_count: -1,
            binary_addr_buffer: [0u8; BINARY_ADDR_BYTES],
            last_file_type: None,
            current_file_type: None,
            pending_file_type: None,
            consecutive_ones: 0,
        }
    }

    /// Reset for a new data block (after header detection)
    pub fn reset_for_block(&mut self) {
        self.last_file_type = self.current_file_type;
        self.current_file_type = None;
        self.state = State::ExpectFirstStart;
        self.byte_count = 0;
        self.header_buffer.clear();
        self.consecutive_ones = 0;

        // Only track binary addresses for the data block right after a BINARY header
        if self.last_file_type == Some(FileType::Binary) {
            self.binary_addr_count = 0;
        } else {
            self.binary_addr_count = -1;
        }
    }

    /// Process a bit decision from the BitReader. Returns an Action.
    pub fn process_bit(&mut self, decision: &BitDecision) -> Action {
        match decision.bit {
            Bit::Silence => {
                self.state = State::Done;
                Action::SilenceDetected
            }
            Bit::Zero | Bit::One => self.process_data_bit(decision),
        }
    }

    /// Check if a file type was just detected. Returns once then clears.
    pub fn take_file_type(&mut self) -> Option<(FileType, String)> {
        self.pending_file_type.take()
    }

    /// Check if binary address info is complete.
    pub fn check_binary_info(&self) -> Option<BinaryInfo> {
        if self.binary_addr_count as usize == BINARY_ADDR_BYTES {
            Some(BinaryInfo {
                load: u16::from_le_bytes([self.binary_addr_buffer[0], self.binary_addr_buffer[1]]),
                end: u16::from_le_bytes([self.binary_addr_buffer[2], self.binary_addr_buffer[3]]),
                exec: u16::from_le_bytes([self.binary_addr_buffer[4], self.binary_addr_buffer[5]]),
            })
        } else {
            None
        }
    }

    fn process_data_bit(&mut self, decision: &BitDecision) -> Action {
        let bit = decision.bit;

        match self.state {
            // After header detection we're still in the header tone (all 1s).
            // Skip leading 1-bits until we see the first 0 (start bit).
            State::ExpectFirstStart => {
                if bit == Bit::One {
                    return Action::Continue; // still in header tone, skip
                }
                // Got the first 0 - this is the start bit, begin data framing
                self.state = State::ReadByte {
                    bits_remaining: 8,
                    value: 0,
                };
                return Action::DataFound;
            }

            State::ExpectStart => {
                if bit == Bit::Zero {
                    // Valid start bit — begin reading byte
                    self.consecutive_ones = 0;
                    self.state = State::ReadByte {
                        bits_remaining: 8,
                        value: 0,
                    };
                } else {
                    // Count consecutive 1-bits in ExpectStart.
                    // In valid framing this should be 0 (PLL jitter: 1-2).
                    // Many more means we're in leader tone — end of block.
                    self.consecutive_ones += 1;
                    if self.consecutive_ones > MAX_ONES_IN_EXPECT_START {
                        self.state = State::Done;
                        return Action::SilenceDetected;
                    }
                }
                Action::Continue
            }

            State::ReadByte {
                bits_remaining,
                value,
            } => {
                // LSB first
                let new_value = (value >> 1) | if bit == Bit::One { 0x80 } else { 0 };
                let remaining = bits_remaining - 1;

                if remaining == 0 {
                    self.state = State::ExpectStop1;
                    self.on_byte_complete(new_value)
                } else {
                    self.state = State::ReadByte {
                        bits_remaining: remaining,
                        value: new_value,
                    };
                    Action::Continue
                }
            }

            State::ExpectStop1 => {
                if bit == Bit::One {
                    self.state = State::ExpectStop2;
                }
                // Wrong bit: stay in ExpectStop1, retry next bit (self-healing)
                Action::Continue
            }

            State::ExpectStop2 => {
                if bit == Bit::One {
                    self.state = State::ExpectStart;
                }
                // Wrong bit: stay in ExpectStop2, retry next bit (self-healing)
                Action::Continue
            }

            State::Done => Action::Done,
        }
    }

    fn on_byte_complete(&mut self, byte: u8) -> Action {
        // Store in header buffer
        if self.byte_count < HEADER_ANALYZE_BYTES {
            self.header_buffer.push(byte);
        }

        // Track binary addresses
        if self.binary_addr_count >= 0 && (self.binary_addr_count as usize) < BINARY_ADDR_BYTES {
            self.binary_addr_buffer[self.binary_addr_count as usize] = byte;
            self.binary_addr_count += 1;
        }

        self.byte_count += 1;

        // Detect file type after collecting enough bytes
        if self.byte_count == HEADER_ANALYZE_BYTES {
            self.detect_file_type();
        }

        Action::ByteComplete(byte)
    }

    fn detect_file_type(&mut self) {
        let buf = &self.header_buffer;
        if buf.len() < HEADER_ANALYZE_BYTES {
            return;
        }

        let (file_type, descriptor) = if buf[..10] == MSX_ASCII_ID {
            let name = extract_filename(&buf[FILENAME_OFFSET..]);
            (FileType::Ascii, name)
        } else if buf[..10] == MSX_BINARY_ID {
            let name = extract_filename(&buf[FILENAME_OFFSET..]);
            (FileType::Binary, name)
        } else if buf[..10] == MSX_BASIC_ID {
            let name = extract_filename(&buf[FILENAME_OFFSET..]);
            (FileType::Basic, name)
        } else {
            (FileType::Custom, "(non-standard header)".into())
        };

        self.current_file_type = Some(file_type);

        // Don't report CUSTOM for data continuation blocks
        // (blocks following an ASCII/BINARY/BASIC type header block)
        if file_type == FileType::Custom
            && matches!(
                self.last_file_type,
                Some(FileType::Ascii | FileType::Binary | FileType::Basic)
            )
        {
            return;
        }

        self.pending_file_type = Some((file_type, descriptor));
    }
}

fn extract_filename(data: &[u8]) -> String {
    let end = data.len().min(FILENAME_LENGTH);
    let bytes = &data[..end];
    String::from_utf8_lossy(bytes).trim_end().to_string()
}

// =============================================================================
// Header Frequency Detection (Goertzel-based iterative narrowing)
// Translated from wav_decoder.c:555-652
// =============================================================================

pub struct HeaderConfig {
    pub block_length: f64,
    pub header_length: f64,
    pub min_bauds: f64,
    pub max_bauds: f64,
    pub scan_resolution: f64,
    pub scan_ratio: f64,
    pub quality_threshold: f64,
}

impl Default for HeaderConfig {
    fn default() -> Self {
        Self {
            block_length: 0.2,
            header_length: 0.5,
            min_bauds: 500.0,
            max_bauds: 4000.0,
            scan_resolution: 10.0,
            scan_ratio: 0.9,
            quality_threshold: 100.0,
        }
    }
}

pub struct HeaderResult {
    pub frequency: f64,
    pub quality: f64,
    pub samples_consumed: usize,
}

/// Detect the header frequency from a stream of samples.
///
/// Returns None if no header is found (EOF or insufficient quality).
/// Keeps scanning (resetting on poor blocks) until header_length seconds
/// of consecutive good signal, or EOF.
pub fn detect_header_frequency(
    samples: &mut dyn Iterator<Item = f64>,
    sample_rate: u32,
    config: &HeaderConfig,
) -> Option<HeaderResult> {
    let block_size = (sample_rate as f64 * config.block_length + 0.5) as usize;
    let total_header_samples = (sample_rate as f64 * config.header_length + 0.5) as usize;

    let freq_min = config.min_bauds * BAUD_TO_FREQ_MULTIPLIER;
    let freq_max = config.max_bauds * BAUD_TO_FREQ_MULTIPLIER;

    let mut detector = GoertzelDetector::new(sample_rate, 0.0, block_size);

    let mut weighted_freq_sum = 0.0;
    let mut power_sum = 0.0;
    let mut total_power = 0.0;
    let mut total_consumed = 0_usize;
    let mut header_start = 0_usize;

    while (total_consumed - header_start) < total_header_samples {
        // Fill detector with one block of samples
        detector.reset();
        let mut block_read = 0;
        for _ in 0..block_size {
            match samples.next() {
                Some(s) => {
                    detector.update(s);
                    block_read += 1;
                }
                None => return None, // EOF
            }
        }
        total_consumed += block_read;

        // Iteratively narrow frequency range
        let mut range = (freq_max - freq_min) * 0.5;
        let mut center_freq = (freq_max + freq_min) * 0.5;

        while range > MIN_RANGE_THRESHOLD {
            let mut inner_power_sum = 0.0;
            let mut inner_weighted_freq = 0.0;

            let mut freq = center_freq - range * 0.5;
            while freq < center_freq + range * 0.5 {
                detector.set_freq(freq);
                let power = detector.power(true);
                inner_weighted_freq += power * freq;
                inner_power_sum += power;
                freq += range / config.scan_resolution;
            }

            if inner_power_sum > 0.0 {
                center_freq = inner_weighted_freq / inner_power_sum;
            }

            center_freq = center_freq.clamp(freq_min, freq_max);
            range *= config.scan_ratio;
        }

        // Measure final power
        detector.set_freq(center_freq);
        let power = detector.power(true);

        if power > config.quality_threshold {
            weighted_freq_sum += center_freq * power;
            power_sum += power;
            total_power = power;
        } else {
            // Poor quality: reset and start fresh
            header_start = total_consumed;
            weighted_freq_sum = 0.0;
            power_sum = 0.0;
            total_power = 0.0;
        }
    }

    if power_sum > 0.0 {
        Some(HeaderResult {
            frequency: weighted_freq_sum / power_sum,
            quality: total_power,
            samples_consumed: total_consumed,
        })
    } else {
        None
    }
}
