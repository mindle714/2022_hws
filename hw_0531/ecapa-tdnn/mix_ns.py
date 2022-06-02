import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eval-list", type=str, required=True) 
parser.add_argument("--noise-list", type=str, required=True) 
parser.add_argument("--min-snr", type=int, required=False, default=10)
parser.add_argument("--max-snr", type=int, required=False, default=10)
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import os
import sys

os.makedirs(args.output, exist_ok=True)

eval_list = [e.strip() for e in open(args.eval_list).readlines()]
noise_list = [e.strip() for e in open(args.noise_list).readlines()]
assert len(eval_list) <= len(noise_list)

import warnings
import soundfile
import tqdm
import numpy as np

wavs = []
for idx, _list in tqdm.tqdm(enumerate(eval_list), total=len(eval_list)):
  if len(_list.split()) < 3:
    warnings.warn("failed to parse {} at line {}".format(_list, idx))
    continue

  _lines = _list.split()
  fa = _lines[0]; ta = _lines[1]; enrolls = _lines[2:]

  def add_noise(wav):
    pcm, sr = soundfile.read(wav)
    snr_db = np.random.uniform(args.min_snr, args.max_snr)
    noise, _ = soundfile.read(noise_list[idx])

    noise = np.repeat(noise, (pcm.shape[0]//noise.shape[0]+1))
    noise = noise[:pcm.shape[0]]

    pcm_en = np.mean(pcm**2)
    noise_en = np.maximum(np.mean(noise**2), 1e-9)
    snr_en = 10.**(snr_db/10.)

    noise *= np.sqrt(pcm_en / (snr_en * noise_en))
    pcm += noise
    noise_pcm_en = np.maximum(np.mean(pcm**2), 1e-9)
    pcm *= np.sqrt(pcm_en / noise_pcm_en)

    return pcm, sr

  fa_ns, sr = add_noise(fa)
  ta_ns, sr = add_noise(ta)

  fa_path = os.path.join(args.output, "eval-fa-{}.wav".format(idx))
  fa_path = os.path.abspath(fa_path)
  soundfile.write(fa_path, fa_ns, sr)
  
  ta_path = os.path.join(args.output, "eval-ta-{}.wav".format(idx))
  ta_path = os.path.abspath(ta_path)
  soundfile.write(ta_path, ta_ns, sr)

  wavs.append([fa_path, ta_path] + enrolls)

with open(os.path.join(args.output, "eval.list"), "w") as f:
  for _wavs in wavs:
    f.write("{}\n".format(" ".join(_wavs)))
