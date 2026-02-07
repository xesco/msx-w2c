use std::f64::consts::PI;

// =============================================================================
// Goertzel Detector (sliding-window frequency power measurement)
// Translated from lib/audio/frequency_detector.c
// =============================================================================

pub struct GoertzelDetector {
    sampling_frequency: u32,
    frequency_to_detect: f64,
    samples: Vec<f64>,
    current_sample: usize,
    sample_sum: f64,
    coeff: f64,
}

impl GoertzelDetector {
    pub fn new(sample_rate: u32, freq: f64, window_size: usize) -> Self {
        let coeff = 2.0 * (2.0 * PI * freq / sample_rate as f64).cos();
        Self {
            sampling_frequency: sample_rate,
            frequency_to_detect: freq,
            samples: vec![0.0; window_size],
            current_sample: 0,
            sample_sum: 0.0,
            coeff,
        }
    }

    pub fn reset(&mut self) {
        self.current_sample = 0;
        self.sample_sum = 0.0;
        self.samples.fill(0.0);
    }

    pub fn set_freq(&mut self, freq: f64) {
        self.frequency_to_detect = freq;
        self.coeff = 2.0 * (2.0 * PI * freq / self.sampling_frequency as f64).cos();
    }

    pub fn update(&mut self, sample: f64) {
        self.sample_sum -= self.samples[self.current_sample];
        self.samples[self.current_sample] = sample;
        self.sample_sum += sample;

        self.current_sample += 1;
        if self.current_sample >= self.samples.len() {
            self.current_sample = 0;
        }
    }

    pub fn power(&self, normalize: bool) -> f64 {
        let n = self.samples.len();
        let average = self.sample_sum / n as f64;

        let mut previous = 0.0_f64;
        let mut previous_previous = 0.0_f64;
        let mut idx = self.current_sample;

        for _ in 0..n {
            let mut s = self.samples[idx];
            if normalize {
                s -= average;
            }
            s += self.coeff * previous - previous_previous;

            idx += 1;
            if idx >= n {
                idx = 0;
            }

            previous_previous = previous;
            previous = s;
        }

        let result = previous_previous * previous_previous
            + previous * previous
            - self.coeff * previous * previous_previous;

        result / n as f64
    }
}

// =============================================================================
// BitReader (dual Goertzel + PLL, translated from lib/audio/bit_reader.c)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bit {
    Zero,
    One,
    Silence,
}

#[derive(Debug, Clone)]
pub struct BitDecision {
    pub bit: Bit,
    pub power_zero: f64,
    pub power_one: f64,
    pub confidence: f64,
}

/// Configuration for BitReader (matches wav_decoder_types.h defaults)
#[derive(Debug, Clone)]
pub struct BitReaderConfig {
    pub silence_ratio: f64,
    pub header_end_sensitivity: f64,
    pub phase_adjust_coeff: f64,
    /// Number of consecutive below-threshold bit periods before declaring silence.
    /// Provides dropout tolerance for real-world recordings.
    pub silence_threshold: usize,
    /// Minimum fraction of current power reference for update.
    /// Prevents ratchet-down during signal decay at block boundaries.
    pub power_hold_ratio: f64,
}

impl Default for BitReaderConfig {
    fn default() -> Self {
        Self {
            silence_ratio: 0.1,
            header_end_sensitivity: 0.125,
            phase_adjust_coeff: 0.5,
            silence_threshold: 3,
            power_hold_ratio: 0.25,
        }
    }
}

impl BitReaderConfig {
    pub fn with_lenient(mut self) -> Self {
        self.silence_ratio *= 0.5;
        self.header_end_sensitivity *= 0.5;
        self
    }
}

pub struct BitReader {
    // Dual Goertzel detectors
    fd_zero: GoertzelDetector, // bit 0 frequency (1200Hz)
    fd_one: GoertzelDetector,  // bit 1 frequency (2400Hz)
    // Phase-lock loop (phase in seconds, matching C bit_reader.c)
    phase: f64,
    period: f64,     // 1/sample_rate
    bit_period: f64, // 1/baud_rate
    // Configuration
    config: BitReaderConfig,
    // Bit detection state
    current_bit: bool, // true = 2400Hz dominant (bit 1)
    previous_bit: bool,
    phase_defined: bool,
    // Power tracking for adaptive silence
    last_power_zero: f64,
    last_power_one: f64,
    // Consecutive silence tolerance
    consecutive_silence_count: usize,
    // Initialization
    init_samples: usize,
    samples_fed: usize,
}

impl BitReader {
    pub fn new(sample_rate: u32, header_freq: f64, config: BitReaderConfig) -> Self {
        let baud_rate = header_freq * 0.5;
        let period = 1.0 / sample_rate as f64;
        let bit_period = 1.0 / baud_rate;

        // Window = 2 cycles of lower frequency = 1 bit period
        let window_size = (sample_rate as f64 * 2.0 / header_freq + 0.5) as usize;

        let fd_zero = GoertzelDetector::new(sample_rate, header_freq * 0.5, window_size);
        let fd_one = GoertzelDetector::new(sample_rate, header_freq, window_size);

        // Initialize with one bit period of samples
        let init_samples = (sample_rate as f64 * bit_period + 0.5) as usize;

        Self {
            fd_zero,
            fd_one,
            phase: 0.0,
            period,
            bit_period,
            config,
            current_bit: true, // Start assuming bit 1 (header tone)
            previous_bit: true,
            phase_defined: false,
            last_power_zero: 0.0,
            last_power_one: 0.0,
            consecutive_silence_count: 0,
            init_samples,
            samples_fed: 0,
        }
    }

    /// Process a single sample. Returns Some(BitDecision) when a full bit period completes.
    pub fn process_sample(&mut self, sample: f64) -> Option<BitDecision> {
        // Feed both Goertzel detectors
        self.fd_zero.update(sample);
        self.fd_one.update(sample);
        self.samples_fed += 1;

        // Phase 1: Fill detectors with one bit period before starting
        if self.samples_fed <= self.init_samples {
            if self.samples_fed == self.init_samples {
                self.last_power_one = self.fd_one.power(false);
                self.last_power_zero = self.fd_zero.power(false);
            }
            return None;
        }

        // Get current power at both frequencies
        let power_zero = self.fd_zero.power(false);
        let power_one = self.fd_one.power(false);

        // Determine which frequency dominates
        self.previous_bit = self.current_bit;
        self.current_bit = power_one > power_zero;

        // Phase 2: Header end detection (wait for 1200Hz to appear)
        if !self.phase_defined {
            if power_zero > self.last_power_one * self.config.header_end_sensitivity {
                self.phase = 0.5 * self.bit_period;
                self.phase_defined = true;
            }
            self.phase += self.period;
            return None;
        }

        // Phase 3: PLL-based bit reading

        // Adjust phase on bit transitions (blend toward ideal mid-bit)
        if self.previous_bit != self.current_bit {
            let ideal_phase = 0.5 * self.bit_period;
            self.phase = ideal_phase * self.config.phase_adjust_coeff
                + self.phase * (1.0 - self.config.phase_adjust_coeff);
        }

        self.phase += self.period;

        // Check if bit period complete
        if self.phase > self.bit_period {
            self.phase -= self.bit_period;

            // Determine bit value with adaptive silence detection
            let is_below_threshold = if self.current_bit {
                power_one <= self.last_power_one * self.config.silence_ratio
            } else {
                power_zero <= self.last_power_zero * self.config.silence_ratio
            };

            if is_below_threshold {
                self.consecutive_silence_count += 1;
                if self.consecutive_silence_count >= self.config.silence_threshold {
                    return Some(BitDecision {
                        bit: Bit::Silence,
                        power_zero,
                        power_one,
                        confidence: 0.0,
                    });
                }
                // Grace period: swallow the weak bit, don't emit to decoder
                return None;
            }

            self.consecutive_silence_count = 0;
            let bit = if self.current_bit {
                // Only update reference when signal is stable or rising,
                // not during decay — prevents ratchet-down that delays silence detection
                if power_one >= self.last_power_one * self.config.power_hold_ratio {
                    self.last_power_one = power_one;
                }
                Bit::One
            } else {
                if power_zero >= self.last_power_zero * self.config.power_hold_ratio {
                    self.last_power_zero = power_zero;
                }
                Bit::Zero
            };

            let ratio = if self.current_bit {
                power_one / power_zero.max(1e-10)
            } else {
                power_zero / power_one.max(1e-10)
            };

            Some(BitDecision {
                bit,
                power_zero,
                power_one,
                confidence: (ratio / 10.0).min(1.0),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goertzel_detects_known_frequency() {
        let sample_rate = 44100;
        let freq = 1200.0;
        let window = (sample_rate as f64 / freq * 2.0) as usize;
        let mut det = GoertzelDetector::new(sample_rate, freq, window);

        // Feed a pure 1200Hz sine wave
        for i in 0..window {
            let s = (2.0 * PI * freq * i as f64 / sample_rate as f64).sin();
            det.update(s);
        }

        let power_at_freq = det.power(true);

        // Now check power at a different frequency
        det.set_freq(2400.0);
        let power_off_freq = det.power(true);

        assert!(
            power_at_freq > power_off_freq * 5.0,
            "Should detect 1200Hz strongly"
        );
    }

}
