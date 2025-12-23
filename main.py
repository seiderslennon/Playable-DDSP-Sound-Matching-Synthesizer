import numpy as np
from train_synth import optimize_with_cma
from pprint import pprint
import math
from pythonosc.udp_client import SimpleUDPClient
import argparse
import jax.numpy as jnp

client = SimpleUDPClient("127.0.0.1", 7001)

CH1 = 0x90
MAX_V = 127
CUTOFF_MIN_HZ = 20.0
CUTOFF_MAX_HZ = 8000.0
LOG_CUTOFF_MIN = jnp.log(CUTOFF_MIN_HZ)
LOG_CUTOFF_MAX = jnp.log(CUTOFF_MAX_HZ)

MIDI_MIN = 24.0
MIDI_MAX = 96.0

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize MinimalSquareSynth with CMA-ES.")
    parser.add_argument("--audio", type=str, default="audio_targets/lah.wav", help="Path to target mono wav")
    parser.add_argument("--generations", type=int, default=400, help="Number of CMA-ES generations")
    parser.add_argument("--population", type=int, default=20, help="Population size for CMA-ES")
    parser.add_argument("--sigma", type=float, default=0.2, help="Initial sigma (step size) for CMA-ES")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

def square_saw_cvs(m, Vmax=10.0):
    # Equal-power crossfade
    square_gain = np.cos(0.5 * math.pi * m)
    saw_gain    = np.sin(0.5 * math.pi * m)

    CV_square = square_gain * Vmax
    CV_saw    = saw_gain  * Vmax

    return float(CV_square), float(CV_saw)

def f0_to_cv(f_target_hz, f0=261.63):
    return math.log(f_target_hz / f0, 2)

import math

def seconds_to_cv(attack_time_sec, slider=0.5, cv_depth=1.0):
    MIN_TIME = 0.001       # seconds
    LAMBDA_BASE = 10000.0

    ln6 = math.log(6.0)
    tau = attack_time_sec / ln6

    p = math.log(tau / MIN_TIME, LAMBDA_BASE)

    V = 10.0 * (p - slider) / cv_depth

    V = max(-5.0, min(5.0, V))

    return V


def main():
    args = parse_args()

    params = optimize_with_cma(args.audio, generations=args.generations, population=args.population, sigma=args.sigma, seed=args.seed)
    pprint(params, width=120)

    attack = np.clip(params['amp_env']['attack'], 0.0, 2.0)[0]
    attack_cv = seconds_to_cv(attack)

    decay = np.clip(params['amp_env']['decay'], 0.0, 2.0)[0]
    decay_cv = seconds_to_cv(decay)

    sustain = np.clip(params['amp_env']['sustain'], 0.0, 1.0)[0]
    sustain_cv = (float(sustain) * 10.0) - 5.0

    release = np.clip(params['amp_env']['release'], 0.0, 5.0)[0]
    release_cv = seconds_to_cv(release)

    cutoff = params['cutoff'][0]
    cutoff_cv = f0_to_cv(cutoff)

    midi_f0 = params['midi_f0'][0]
    print(midi_f0)
    f0 = 440.0 * 2 ** ((midi_f0 - 69) / 12)
    f0_cv = f0_to_cv(f0)

    shape = np.clip(params['vco']['shape'], 0.0, 1.0)[0]
    cv_sq, cv_saw = square_saw_cvs(shape)

    values =    [attack,    decay,    sustain,    release,    cutoff,    midi_f0,    shape]
    cvs =       [attack_cv, decay_cv, sustain_cv, release_cv, cutoff_cv,   f0_cv, cv_sq, cv_saw]
    print(values)
    print(cvs)

    client.send_message("/ch/1", f0_cv)
    client.send_message("/ch/2", cv_saw)
    client.send_message("/ch/3", cv_sq)
    client.send_message("/ch/4", cutoff_cv)
    client.send_message("/ch/5", attack_cv)
    client.send_message("/ch/6", decay_cv)
    client.send_message("/ch/7", sustain_cv)
    client.send_message("/ch/8", release_cv)


if __name__ == "__main__":
    main()