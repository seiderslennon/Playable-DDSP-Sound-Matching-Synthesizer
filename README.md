A Python project that uses an evolutionary algorithm (CMA-ES) to optimize synthesizer parameters (LFO, VCA, VCF, ASDR) to match target audio files, then sends the optimized parameters to VCV Rack via OSC. Uses Jax for fast audio synthesis. Works best for 1-4s audio clips with a well-defined envelope.

How it works:
1. **Load Target**: Loads and analyzes the target audio file
2. **Initialize Synth**: Creates a MinimalSquareSynth with VCO, filter, and envelope
3. **Optimize**: Uses CMA-ES to evolve synth parameters that minimize spectral distance
4. **Export**: Saves synthesized audio and sends CV values to VCV Rack

<!-- ![Alt text](path/to/your/demo.gif) -->

Usage:
- Python 3.8+
- VCV Rack (for real-time synthesis)

```bash
python main.py --audio audio_path.wav
```

- `--audio`: Path to target audio file
- `--generations`: Number of CMA-ES generations (default: 400)
- `--population`: Population size for evolution (default: 20)
- `--sigma`: Initial step size (default: 0.1)
- `--seed`: Random seed (default: 0)

Ensure VCV Rack is running with OSC input enabled on port 7001.




