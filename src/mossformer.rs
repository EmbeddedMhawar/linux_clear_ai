//! MossFormer2 ONNX inference module for speech enhancement
//!
//! This module provides a wrapper around the MossFormer2 ONNX model
//! for real-time speech enhancement with optional GPU acceleration.

use anyhow::{anyhow, Result};
use ndarray::{ArrayView2, ArrayViewMut2};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::execution_providers::{CUDAExecutionProvider};
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use rustfft::num_complex::Complex;
use std::path::Path;
use std::sync::Arc;

/// FFT size for STFT processing (48kHz / ~21ms frame)
const FFT_SIZE: usize = 1920;
/// Hop size (50% overlap)
const HOP_SIZE: usize = 384;
/// Number of frequency bins (FFT_SIZE / 2 + 1)
const NUM_BINS: usize = FFT_SIZE / 2 + 1; // 961
const MEL_BINS: usize = 180;
const SAMPLE_RATE: usize = 48000;

/// MossFormer2 processor for speech enhancement via ONNX Runtime
pub struct MossFormer {
    session: Session,
    sample_rate: usize,
    
    // FFT processing
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    
    // Processing buffers
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    ifft_input: Vec<Complex<f32>>,
    ifft_output: Vec<f32>,
    window: Vec<f32>,
    
    // Overlap-add state
    overlap_buffer: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
    
    // Processing settings
    use_gpu: bool,
}

impl MossFormer {
    /// Create a new MossFormer2 processor
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `use_gpu` - Whether to use GPU acceleration (CUDA)
    pub fn new<P: AsRef<Path>>(model_path: P, use_gpu: bool) -> Result<Self> {
        // Load ONNX model
        let environment = ort::init()
            .with_name("MossFormer2")
            .commit()?;
            
        let mut session_builder = Session::builder()?;
        session_builder = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?; // Use 4 threads for inference

        if use_gpu {
            // Try CUDA first
            #[cfg(target_os = "linux")]
            {
                if let Err(e) = session_builder.clone().with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build()
                ]) {
                     eprintln!("Failed to load CUDA provider: {:?}", e);
                }
            }
        }

        let session = session_builder.commit_from_file(model_path)?;
            
        // DEBUG LOGGING
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/linux_clear_ai_debug.log") 
        {
            use std::io::Write;
            let _ = writeln!(file, "MossFormer2 Initialized");
            
            // Log Inputs
            for (i, input) in session.inputs.iter().enumerate() {
                let _ = writeln!(file, "Input {}: Name={}, Type={:?}", i, input.name, input.input_type);
            }
            
            // Log Outputs
            for (i, output) in session.outputs.iter().enumerate() {
                let _ = writeln!(file, "Output {}: Name={}, Type={:?}", i, output.name, output.output_type);
            }
            
            // Log Metadata
            if let Ok(meta) = session.metadata() {
                let _ = writeln!(file, "Metadata:");
                let _ = writeln!(file, "  Description: {}", meta.description().unwrap_or("None".to_string()));
                let _ = writeln!(file, "  Domain: {}", meta.domain().unwrap_or("None".to_string()));
                let _ = writeln!(file, "  Producer: {}", meta.producer().unwrap_or("None".to_string()));
                let _ = writeln!(file, "  Version: {:?}", meta.version());
            }
        }
        
        // Initialize FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(FFT_SIZE);
        let fft_inverse = planner.plan_fft_inverse(FFT_SIZE);
        
        // Create Hann window
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos())
            })
            .collect();
        
        // Initialize Mel Filterbank
        let mel_filters = Self::create_mel_filterbank();

        let input_buffer = vec![0.0; FFT_SIZE];
        let output_buffer = vec![0.0; FFT_SIZE];
        let fft_input = vec![0.0; FFT_SIZE];
        let fft_output = vec![Complex::default(); NUM_BINS];
        let ifft_input = vec![Complex::default(); NUM_BINS];
        let ifft_output = vec![0.0; FFT_SIZE];
        let overlap_buffer = vec![0.0; FFT_SIZE]; // Size = FFT_SIZE for Shift-Accumulate

        Ok(Self {
            session,
            sample_rate: SAMPLE_RATE,
            fft_forward,
            fft_inverse,
            input_buffer,
            output_buffer,
            fft_input,
            fft_output,
            ifft_input,
            ifft_output,
            window,
            overlap_buffer,
            mel_filters,
            use_gpu,
        })
    }

    fn create_mel_filterbank() -> Vec<Vec<f32>> {
        let num_fft_bins = NUM_BINS;
        let num_mel_bins = MEL_BINS;
        let sample_rate = SAMPLE_RATE as f32;
        let low_freq = 0.0;
        let high_freq = sample_rate / 2.0;
        
        // Mel scale conversion functions
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
        
        let min_mel = hz_to_mel(low_freq);
        let max_mel = hz_to_mel(high_freq);
        
        // Create Mel points
        let mut mel_points = Vec::with_capacity(num_mel_bins + 2);
        let step = (max_mel - min_mel) / (num_mel_bins + 1) as f32;
        for i in 0..=num_mel_bins + 1 {
            mel_points.push(mel_to_hz(min_mel + i as f32 * step));
        }
        
        // Create filterbank matrix
        let mut filters = vec![vec![0.0; num_fft_bins]; num_mel_bins];
        let fft_freqs: Vec<f32> = (0..num_fft_bins)
            .map(|i| i as f32 * sample_rate / FFT_SIZE as f32)
            .collect();
            
        for i in 0..num_mel_bins {
            let left = mel_points[i];
            let center = mel_points[i+1];
            let right = mel_points[i+2];
            
            for j in 0..num_fft_bins {
                let freq = fft_freqs[j];
                if freq >= left && freq <= center {
                    filters[i][j] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    filters[i][j] = (right - freq) / (right - center);
                }
            }
        }
        
        filters
    }
    
    /// Get the hop size for this processor
    pub fn hop_size(&self) -> usize {
        HOP_SIZE
    }
    
    /// Get the sample rate this processor expects
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    
    /// Process a single channel of audio
    ///
    /// # Arguments
    /// * `input` - Input samples (hop_size samples)
    /// * `output` - Output buffer (hop_size samples)
    ///
    /// Returns Ok(()) on success
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != HOP_SIZE || output.len() != HOP_SIZE {
            return Err(anyhow!(
                "Input/output must be {} samples, got {}/{}",
                HOP_SIZE,
                input.len(),
                output.len()
            ));
        }
        
        // Shift input buffer and add new samples
        self.input_buffer.copy_within(HOP_SIZE.., 0);
        let start_idx = FFT_SIZE - HOP_SIZE;
        self.input_buffer[start_idx..].copy_from_slice(input);
        
        // Apply window
        for (i, sample) in self.input_buffer.iter().enumerate() {
            self.fft_input[i] = sample * self.window[i];
        }
        
        // Forward FFT
        self.fft_forward
            .process(&mut self.fft_input, &mut self.fft_output)
            .map_err(|e| anyhow!("FFT error: {:?}", e))?;
        
        // Prepare input tensor for ONNX model
        // MossFormer2 expects: [batch=1, time_frames, freq_bins, 2] (real/imag)
        let batch_size = 1;
        let num_frames = 1;
        
        // Create flat input: real and imaginary parts interleaved
        let mut model_input: Vec<f32> = Vec::with_capacity(NUM_BINS * 2);
        for bin in &self.fft_output {
            model_input.push(bin.re);
            model_input.push(bin.im);
        }
        
        // 3. Compute Power Spectrum
        let mut power_spec = vec![0.0; NUM_BINS];
        for (i, c) in self.fft_output.iter().enumerate() {
            power_spec[i] = c.norm_sqr();
        }
        
        // 4. Compute Mel Spectrogram
        let mut mel_spec = vec![0.0; MEL_BINS];
        for i in 0..MEL_BINS {
            let mut sum = 0.0;
            for j in 0..NUM_BINS {
                sum += power_spec[j] * self.mel_filters[i][j];
            }
            // Log compression (log10(x + 1e-6))
            mel_spec[i] = (sum + 1e-6).log10();
        }
        
        // 5. Prepare ONNX Input [1, 1, 180]
        let shape = [1usize, 1, MEL_BINS];
        let input_value = ort::value::Tensor::<f32>::from_array((shape, mel_spec.clone()))?;
        
        // 6. Run Inference
        let outputs = self.session.run(ort::inputs![input_value]).map_err(|e| {
             if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/linux_clear_ai_debug.log") 
            {
                use std::io::Write;
                let _ = writeln!(file, "Inference Error: {:?}", e);
            }
            e
        })?;
        
        // 7. Get Output Mask [1, 1, 961]
        let first_output = outputs.iter().next()
            .ok_or_else(|| anyhow!("No output from model"))?;
        let (_, output_value) = first_output;
        let (_output_shape, mask_data) = output_value.try_extract_tensor::<f32>()?;
        
        // DEBUG: Log values once per second (approx every 100 frames)
        // We can use a static counter or just random sampling
        if rand::random::<f32>() < 0.01 {
             if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/linux_clear_ai_values.log") 
            {
                use std::io::Write;
                let input_max = self.input_buffer.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let mel_max = mel_spec.iter().fold(-100.0f32, |a, &b| a.max(b));
                let mask_min = mask_data.iter().fold(1.0f32, |a, &b| a.min(b));
                let mask_max = mask_data.iter().fold(0.0f32, |a, &b| a.max(b));
                let _ = writeln!(file, "InputMax: {:.4}, MelMax: {:.4}, MaskMin: {:.4}, MaskMax: {:.4}", 
                    input_max, mel_max, mask_min, mask_max);
            }
        }

        // 8. Apply Mask to Spectrum
        // Mask is likely Magnitude Mask. 
        // Clean = Noisy * Mask
        for i in 0..NUM_BINS {
            if i < mask_data.len() {
                let mask = mask_data[i];
                // Apply mask to complex spectrum (preserves phase)
                self.ifft_input[i] = self.fft_output[i] * mask;
            } else {
                self.ifft_input[i] = self.fft_output[i]; // Fallback
            }
        }
        
        // 9. Inverse FFT
        self.fft_inverse
            .process(&mut self.ifft_input, &mut self.ifft_output)
            .map_err(|e| anyhow!("IFFT error: {:?}", e))?;
            
        // Normalize (IFFT scaling)
        let norm = 1.0 / FFT_SIZE as f32;
        for sample in self.ifft_output.iter_mut() {
            *sample *= norm;
        }
            
        // 10. Overlap-Add (Shift-Accumulate)
        // Add current frame to overlap buffer
        for i in 0..FFT_SIZE {
            // Apply window and accumulate
            self.overlap_buffer[i] += self.ifft_output[i] * self.window[i]; 
        }
        
        // Write to output (HOP_SIZE)
        for i in 0..HOP_SIZE {
            output[i] = self.overlap_buffer[i];
        }
        
        // Shift overlap buffer left by HOP_SIZE
        let remaining = FFT_SIZE - HOP_SIZE;
        for i in 0..remaining {
            self.overlap_buffer[i] = self.overlap_buffer[i + HOP_SIZE];
        }
        
        // Zero the tail
        for i in remaining..FFT_SIZE {
            self.overlap_buffer[i] = 0.0;
        }
        
        Ok(())
    }
    
    /// Process stereo audio (2 channels)
    pub fn process_stereo(
        &mut self,
        input: ArrayView2<f32>,
        mut output: ArrayViewMut2<f32>,
    ) -> Result<()> {
        let num_channels = input.nrows();
        let num_samples = input.ncols();
        
        if num_samples != HOP_SIZE {
            return Err(anyhow!("Expected {} samples, got {}", HOP_SIZE, num_samples));
        }
        
        // For now, process each channel independently
        // A more sophisticated approach would process them together
        for ch in 0..num_channels.min(2) {
            let ch_input: Vec<f32> = input.row(ch).iter().copied().collect();
            let mut ch_output = vec![0.0f32; HOP_SIZE];
            
            let start = std::time::Instant::now();
            self.process(&ch_input, &mut ch_output)?;
            let duration = start.elapsed();
            
            // Log if processing is too slow (> 8ms)
            if duration.as_millis() > 8 {
                 if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/linux_clear_ai_perf.log") 
                {
                    use std::io::Write;
                    let _ = writeln!(file, "Slow processing: {:?} (Budget: 8ms)", duration);
                }
            }
            
            // Write output
            for (i, &sample) in ch_output.iter().enumerate() {
                output[[ch, i]] = sample;
            }
        }
        
        Ok(())
    }
    
    /// Check if GPU is being used
    pub fn is_using_gpu(&self) -> bool {
        self.use_gpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hop_size() {
        assert_eq!(HOP_SIZE, 512);
    }
    
    #[test]
    fn test_num_bins() {
        assert_eq!(NUM_BINS, 513);
    }
}
