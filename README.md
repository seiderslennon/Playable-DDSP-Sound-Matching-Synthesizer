# VCV Synth Optimizer

A Python project that uses evolutionary algorithms (CMA-ES) to optimize synthesizer parameters to match target audio files, then sends the optimized parameters to VCV Rack via OSC.

This project bridges machine learning and modular synthesis by:
- Using CMA-ES to find optimal synthesizer parameters that recreate the sound
- Sending control voltage (CV) parameters to VCV Rack for hardware/software synthesis

![Alt text](path/to/your/demo.gif)

## Features

- **Audio-to-Synth Matching**: Train a synthesizer to match any target audio file
- **JAX-based Synthesis**: Fast, GPU-accelerated audio synthesis using Synthax
- **VCV Rack Integration**: Sends optimized parameters to VCV Rack via OSC
- **Modular Components**: VCO (square/saw), VCA, LPF, and ADSR envelope

## Usage

- Python 3.8+
- VCV Rack (for real-time synthesis)

### Train synthesizer parameters from target audio and send optimized parameters to VCV Rack:

```bash
python main.py --audio audio_targets/lah.wav
```

Options:
- `--audio`: Path to target audio file (default: `audio_targets/lah.wav`)
- `--generations`: Number of CMA-ES generations (default: 400)
- `--population`: Population size for evolution (default: 20)
- `--sigma`: Initial step size (default: 0.1)
- `--seed`: Random seed (default: 0)

Ensure VCV Rack is running with OSC input enabled on port 7001.

## How It Works

1. **Load Target**: Loads and analyzes the target audio file
2. **Initialize Synth**: Creates a MinimalSquareSynth with VCO, filter, and envelope
3. **Optimize**: Uses CMA-ES to evolve synth parameters that minimize spectral distance
4. **Export**: Saves synthesized audio and sends CV values to VCV Rack


