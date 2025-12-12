use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, EguiState, widgets, egui, egui::Widget};
use std::sync::Arc;
use ringbuf::{HeapRb, Producer, Consumer};
use std::thread;
use std::sync::mpsc;

mod mossformer;
use mossformer::MossFormer;

// Type aliases for cleaner code
type RbProd = Producer<f32, Arc<HeapRb<f32>>>;
type RbCons = Consumer<f32, Arc<HeapRb<f32>>>;

/// Message sent to worker thread for parameter updates
#[derive(Clone)]
struct WorkerParams {
    voice_gain: f32,
    ambiance_gain: f32,
    reverb_gain: f32,
    use_gpu: bool,
}

struct LinuxClearAi {
    params: Arc<LinuxClearAiParams>,
    
    // Host -> Plugin
    input_producers: Vec<RbProd>,
    // Plugin -> Host
    output_consumers: Vec<RbCons>,
    
    hop_size: usize,
    channel_count: usize,
    
    // Worker handle (optional, to keep it alive)
    worker_handle: Option<thread::JoinHandle<()>>,
    
    // Channel to send parameters to worker
    param_tx: Option<mpsc::Sender<WorkerParams>>,

    // Editor state
    editor_state: Arc<EguiState>,
}

#[derive(Params)]
struct LinuxClearAiParams {
    #[id = "voice_gain"]
    pub voice_gain: FloatParam,
    
    #[id = "ambiance_gain"]
    pub ambiance_gain: FloatParam,
    
    #[id = "reverb_gain"]
    pub reverb_gain: FloatParam,

    #[id = "gain"]
    pub gain: FloatParam,
    
    #[id = "use_gpu"]
    pub use_gpu: BoolParam,
}

impl Default for LinuxClearAi {
    fn default() -> Self {
        Self {
            params: Arc::new(LinuxClearAiParams::default()),
            input_producers: Vec::new(),
            output_consumers: Vec::new(),
            hop_size: 480,
            channel_count: 0,
            worker_handle: None,
            param_tx: None,
            editor_state: EguiState::from_size(400, 450),
        }
    }
}

impl Default for LinuxClearAiParams {
    fn default() -> Self {
        Self {
            voice_gain: FloatParam::new(
                "Voice",
                0.0,
                FloatRange::Linear { min: -60.0, max: 12.0 },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            
            ambiance_gain: FloatParam::new(
                "Ambiance",
                -60.0,  // Muted by default (noise reduction)
                FloatRange::Linear { min: -60.0, max: 12.0 },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            
            reverb_gain: FloatParam::new(
                "Reverb",
                -60.0,  // Muted by default (coupled with ambiance for now)
                FloatRange::Linear { min: -60.0, max: 12.0 },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            
            gain: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear { min: -12.0, max: 12.0 },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            
            use_gpu: BoolParam::new("Use GPU", false)
                .with_value_to_string(Arc::new(|v| {
                    if v { "CUDA".to_string() } else { "CPU".to_string() }
                })),
        }
    }
}

impl Plugin for LinuxClearAi {
    const NAME: &'static str = "Linux Clear AI";
    const VENDOR: &'static str = "Mhawar";
    const URL: &'static str = "https://embeddedmhawar.wordpress.com";
    const EMAIL: &'static str = "abderrahmanemhawar@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        create_egui_editor(
            self.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                egui::CentralPanel::default().show(egui_ctx, |ui| {
                    ui.heading("Linux Clear AI");
                    ui.separator();
                    
                    ui.label("Processing (MossFormer2)");
                    let mut use_gpu = params.use_gpu.value();
                    if ui.checkbox(&mut use_gpu, "Use GPU (CUDA)").changed() {
                        setter.begin_set_parameter(&params.use_gpu);
                        setter.set_parameter(&params.use_gpu, use_gpu);
                        setter.end_set_parameter(&params.use_gpu);
                    }
                    
                    ui.add_space(10.0);
                    
                    ui.label("Stem Controls");
                    ui.label("Voice");
                    widgets::ParamSlider::for_param(&params.voice_gain, setter).ui(ui);
                    
                    ui.label("Ambiance (Background)");
                    widgets::ParamSlider::for_param(&params.ambiance_gain, setter).ui(ui);
                    
                    ui.label("Reverb");
                    widgets::ParamSlider::for_param(&params.reverb_gain, setter).ui(ui);
                    
                    ui.add_space(10.0);
                    
                    ui.label("Output");
                    widgets::ParamSlider::for_param(&params.gain, setter).ui(ui);
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let channels = _audio_io_layout.main_output_channels.unwrap().get() as usize;
        self.channel_count = channels;

        let sr = buffer_config.sample_rate as usize;
        
        // Channels for handshake and params
        let (tx_init, rx_init) = mpsc::channel::<Result<(usize, usize), String>>();
        let (tx_bufs, rx_bufs) = mpsc::channel::<(Vec<RbCons>, Vec<RbProd>)>();
        let (tx_params, rx_params) = mpsc::channel::<WorkerParams>();
        
        self.param_tx = Some(tx_params);

        // MossFormer2 hop size (384 for 48k model)
        let mf_hop_size = 384;

        // Spawn worker thread
        self.worker_handle = Some(thread::spawn(move || {
            // Initialize MossFormer2
            let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join("model.onnx");
            
            // Try to initialize MossFormer2 - for now start with CPU
            let mut mossformer = match MossFormer::new(&model_path, false) {
                Ok(mf) => mf,
                Err(e) => {
                    let _ = tx_init.send(Err(format!("Failed to load MossFormer2: {:?}", e)));
                    return;
                }
            };
            
            // Send hop_size and sr back
            let _ = tx_init.send(Ok((mf_hop_size, 48000usize)));
            
            // Wait for buffers
            let (mut input_consumers, mut output_producers) = match rx_bufs.recv() {
                Ok(bufs) => bufs,
                Err(_) => return, // Init failed or dropped
            };
            
            // Processing loop
            use ndarray::Array2;
            let mut mf_in_buf = Array2::<f32>::zeros((channels, mf_hop_size));
            let mut mf_out_buf = Array2::<f32>::zeros((channels, mf_hop_size));
            
            // Stem gain state (linear scale, default: voice=1.0, ambiance=0.001)
            let mut current_voice_gain: f32 = 1.0;      // 0 dB
            let mut current_ambiance_gain: f32 = 0.001; // -60 dB
            let mut debug_phase: f32 = 0.0;             // For tone generation test
            
            loop {
                // Check for parameter updates
                if let Ok(params) = rx_params.try_recv() {
                    // Convert dB gains to linear
                    let voice_gain_linear = 10f32.powf(params.voice_gain / 20.0);
                    let ambiance_gain_linear = 10f32.powf(params.ambiance_gain / 20.0);
                    
                    current_voice_gain = voice_gain_linear;
                    current_ambiance_gain = ambiance_gain_linear;
                    
                    // TODO: Handle use_gpu change (would require reinitializing MossFormer)
                }

                // Check input availability
                let mut enough_data = true;
                for cons in &input_consumers {
                    if cons.len() < mf_hop_size {
                        enough_data = false;
                        break;
                    }
                }

                if !enough_data {
                    thread::yield_now();
                    continue;
                }

                // Read into MossFormer buffer (with safety check)
                for (ch_idx, cons) in input_consumers.iter_mut().enumerate() {
                    for i in 0..mf_hop_size {
                        mf_in_buf[[ch_idx, i]] = cons.pop().unwrap_or(0.0);
                    }
                }
                
                // Process with MossFormer2 (outputs clean speech)
                if let Err(_e) = mossformer.process_stereo(mf_in_buf.view(), mf_out_buf.view_mut()) {
                    // DEBUG: On error, output a 440Hz sine wave tone
                    // This confirms if the model is failing silently
                    for i in 0..mf_hop_size {
                        let sample = (debug_phase * 2.0 * std::f32::consts::PI).sin() * 0.5;
                        debug_phase += 440.0 / 48000.0;
                        if debug_phase > 1.0 { debug_phase -= 1.0; }
                        
                        for ch in 0..channels {
                            mf_out_buf[[ch, i]] = sample;
                        }
                    }
                }
                
                // Apply stem mixing: output = clean * voice_gain + residual * ambiance_gain
                // "clean" = MossFormer2 output (voice separated from noise)
                // "residual" = input - clean (the background/noise)
                for (ch_idx, prod) in output_producers.iter_mut().enumerate() {
                    for i in 0..mf_hop_size {
                        let input_sample = mf_in_buf[[ch_idx, i]];
                        let clean_sample = mf_out_buf[[ch_idx, i]];
                        let residual_sample = input_sample - clean_sample;
                        
                        let mixed = (clean_sample * current_voice_gain) 
                                  + (residual_sample * current_ambiance_gain);
                        let _ = prod.push(mixed);
                    }
                }
            }
        }));
        
        // Wait for initialization result
        match rx_init.recv() {
            Ok(Ok((hop_size, model_sr))) => {
                if model_sr != sr {
                    nih_log!("Warning: Host sample rate {} != Model sample rate {}", sr, model_sr);
                }
                self.hop_size = hop_size;
                
                self.input_producers.clear();
                self.output_consumers.clear();
                
                let mut input_consumers = Vec::new();
                let mut output_producers = Vec::new();

                for _ in 0..channels {
                    let rb_in = HeapRb::<f32>::new(self.hop_size * 16);
                    let (prod_in, cons_in) = rb_in.split();
                    self.input_producers.push(prod_in);
                    input_consumers.push(cons_in);

                    let rb_out = HeapRb::<f32>::new(self.hop_size * 16);
                    let (prod_out, cons_out) = rb_out.split();
                    output_producers.push(prod_out);
                    self.output_consumers.push(cons_out);
                }
                
                // Latency compensation
                for prod in &mut output_producers {
                    for _ in 0..self.hop_size * 4 {
                        let _ = prod.push(0.0);
                    }
                }
                
                // Send buffers to worker
                let _ = tx_bufs.send((input_consumers, output_producers));
                
                true
            }
            Ok(Err(e)) => {
                nih_log!("Failed to initialize MossFormer2: {:?}", e);
                false
            }
            Err(_) => {
                nih_log!("Worker thread panicked during initialization");
                false
            }
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Convert output gain from dB to linear (0 dB = 1.0 linear)
        let gain_db = self.params.gain.smoothed.next();
        let gain_linear = 10f32.powf(gain_db / 20.0);
        let voice_gain = self.params.voice_gain.value();
        let ambiance_gain = self.params.ambiance_gain.value();
        let reverb_gain = self.params.reverb_gain.value();
        let use_gpu = self.params.use_gpu.value();
        
        // Send parameter update
        if let Some(tx) = &self.param_tx {
            let _ = tx.send(WorkerParams {
                voice_gain,
                ambiance_gain,
                reverb_gain,
                use_gpu,
            });
        }

        // Push input samples to ring buffers (iterate time -> channels)
        let mut max_input = 0.0f32;
        for mut channel_samples in buffer.iter_samples() {
            for (ch_idx, sample) in channel_samples.iter_mut().enumerate() {
                if sample.abs() > max_input { max_input = sample.abs(); }
                if ch_idx < self.input_producers.len() {
                    if let Err(_) = self.input_producers[ch_idx].push(*sample) {
                         // Buffer full?
                    }
                }
            }
        }
        
        // Debug: Print max input once in a while to FILE
        if max_input > 0.01 && rand::random::<f32>() < 0.001 {
             if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/linux_clear_ai_values.log") 
            {
                use std::io::Write;
                let _ = writeln!(file, "AudioThread_In_Max: {:.4}", max_input);
            }
        }

        // Pop output samples from ring buffers (iterate time -> channels)
        for mut channel_samples in buffer.iter_samples() {
            for (ch_idx, sample) in channel_samples.iter_mut().enumerate() {
                if ch_idx < self.output_consumers.len() {
                    if let Some(out_sample) = self.output_consumers[ch_idx].pop() {
                        *sample = out_sample * gain_linear;
                    } else {
                        *sample = 0.0; // Underrun
                    }
                }
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for LinuxClearAi {
    const CLAP_ID: &'static str = "com.linux-clear-ai";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A Linux-native AI noise reduction plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for LinuxClearAi {
    const VST3_CLASS_ID: [u8; 16] = *b"LinuxClearAiPlug";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Restoration,
        Vst3SubCategory::Tools,
    ];
}

nih_export_clap!(LinuxClearAi);
nih_export_vst3!(LinuxClearAi);
