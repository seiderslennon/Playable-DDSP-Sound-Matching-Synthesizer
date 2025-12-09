import argparse
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import librosa
import soundfile as sf
from pprint import pprint
from evosax.algorithms import CMA_ES
from flax import linen as nn

from synthax.config import SynthConfig
from synthax.modules.filters import LPF
from synthax.modules.envelopes import ADSR
from synthax.modules.control import ControlRateUpsample
from synthax.modules.amplifiers import VCA
from synthax.modules.oscillators import SquareSawVCO


class MinimalSquareSynth(nn.Module):
    config: SynthConfig
    min_cutoff: float = 120.0
    max_cutoff: float = 20000.0

    def setup(self):
        self.vco = SquareSawVCO(config=self.config)
        self.upsample = ControlRateUpsample(config=self.config)
        self.vca = VCA(config=self.config)
        self.lpf = LPF(config=self.config)

        attack = jnp.array([0.1])
        decay = jnp.array([0.1])
        sustain = jnp.array([0.8])
        release = jnp.array([0.3])
        alpha = jnp.array([3.0])

        self.amp_env = ADSR(
            config=self.config,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            alpha=alpha
        )
        self.cutoff = self.param(
            "cutoff",
            lambda rng: jax.random.uniform(
                rng, (self.config.batch_size,), minval=800.0, maxval=6000.0
            ),
        )
        self.midi_f0 = self.param(
            "midi_f0",
            lambda rng: jax.random.uniform(
                rng, (self.config.batch_size,), minval=36.0, maxval=84.0
            ),
        )

    @nn.compact
    def __call__(self):
        midi_f0 = jnp.clip(self.midi_f0, 24.0, 96.0)
        note_on_duration = jnp.ones((self.config.batch_size,)) * float(
            self.config.buffer_size_seconds
        )

        env_ctrl = self.amp_env(note_on_duration)
        env_audio = self.upsample(env_ctrl)

        osc = self.vco(midi_f0, None)
        amp = self.vca(osc, env_audio)

        # Build cutoff control at control rate, then upsample
        cutoff_hz = jnp.clip(self.cutoff, self.min_cutoff, self.max_cutoff)
        cutoff_ctrl = jnp.broadcast_to(
            cutoff_hz[:, None], (self.config.batch_size, self.config.control_buffer_size)
        )
        cutoff_audio = self.upsample(cutoff_ctrl)

        filtered = self.lpf(amp, cutoff_audio)
        return filtered


def load_target(audio_path: str):
    audio_path = "audio_targets/" + audio_path
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    f0 = jnp.max(librosa.pyin(y=audio, fmin=32, fmax=5000)[0])
    midi_f0 = 69.0 + 12.0 * jnp.log2(jnp.maximum(f0, 1e-12) / 440.0)
    print('est. f0:', f0, 'midi note:', midi_f0)
    target = jnp.asarray(audio, dtype=jnp.float32)
    target = target / (jnp.max(jnp.abs(target)) + 1e-6)

    buffer_seconds = target.shape[0] / sr
    config = SynthConfig(batch_size=1, sample_rate=sr, buffer_size_seconds=buffer_seconds)
    buffer_size = int(config.buffer_size)

    if target.shape[0] < buffer_size:
        target = jnp.pad(target, (0, buffer_size - target.shape[0]))
    else:
        target = target[:buffer_size]
    target = target[None, :]
    
    return target, sr, config, midi_f0


def build_loss(model, target):
    def loss_fn(params):
        pred = model.apply(params)
        pred = jnp.nan_to_num(pred, nan=0.0, posinf=2.0, neginf=-2.0)
        pred = jnp.clip(pred, -2.0, 2.0)
        pred_fft = jnp.abs(jnp.fft.rfft(pred))
        tgt_fft = jnp.abs(jnp.fft.rfft(target))
        pred_fft = jnp.nan_to_num(pred_fft, nan=0.0, posinf=0.0, neginf=0.0)
        spectral = jnp.mean(jnp.abs(pred_fft - tgt_fft))
        return jnp.nan_to_num(spectral, nan=1e6, posinf=1e6, neginf=1e6)

    return jax.jit(loss_fn)


def optimize_with_cma(audio_path: str, generations: int = 40, population: int = 16, sigma: float = 2.0, seed: int = 0):
    target, sr, config, midi_f0 = load_target(audio_path)
    model = MinimalSquareSynth(config=config)

    key = jax.random.PRNGKey(seed)
    params0 = model.init(key)
    pprint(params0['params'], width=120)

    loss_fn = build_loss(model, target)
    flat0, unravel_fn = ravel_pytree(params0)

    es = CMA_ES(population_size=population, solution=flat0)
    es_params = es.default_params.replace(std_init=sigma)
    key, key_init = jax.random.split(key)
    es_state = es.init(key_init, flat0, es_params)

    loss_from_vec = jax.jit(lambda vec: loss_fn(unravel_fn(vec)))
    batched_loss = jax.jit(jax.vmap(loss_from_vec))

    for gen in range(generations):
        key, key_ask = jax.random.split(key)
        population_vecs, es_state = es.ask(key_ask, es_state, es_params)
        fitness = batched_loss(population_vecs)
        key, key_tell = jax.random.split(key)
        es_state, _ = es.tell(key_tell, population_vecs, fitness, es_state, es_params)

        if gen % 10 == 0:
            best_loss = float(jnp.min(fitness))
            print(f"gen {gen}: best_loss={best_loss:.4f}")

    best_vec = es_state.best_solution if es_state.best_solution is not None else flat0
    best_params = unravel_fn(best_vec)

    raw_audio = jnp.squeeze(model.apply(best_params))
    raw_audio = jnp.nan_to_num(raw_audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = jnp.max(jnp.abs(raw_audio))
    if peak > 0:
        raw_audio = (raw_audio / peak) * 0.99

    out = jax.device_get(raw_audio)
    sf.write("trained_cma_es.wav", out, sr)

    pprint(best_params['params'], width=120)

    print(midi_f0)
    if not jnp.isnan(midi_f0):
        best_params['params']['midi_f0'] = jnp.array([midi_f0])

    return best_params['params']


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize MinimalSquareSynth with CMA-ES.")
    parser.add_argument("--audio", type=str, default="audio_targets/lah.wav", help="Path to target mono wav")
    parser.add_argument("--generations", type=int, default=400, help="Number of CMA-ES generations")
    parser.add_argument("--population", type=int, default=20, help="Population size for CMA-ES")
    parser.add_argument("--sigma", type=float, default=0.1, help="Initial sigma (step size) for CMA-ES")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    optimize_with_cma(args.audio, generations=args.generations, population=args.population, sigma=args.sigma, seed=args.seed)