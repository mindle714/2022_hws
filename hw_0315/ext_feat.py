import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, required=True)
parser.add_argument("--frame-sec", type=float, required=False, default=0.02)
parser.add_argument("--shift-sec", type=float, required=False, default=0.01)
parser.add_argument("--num-mel", type=int, required=False, default=128)
parser.add_argument("--num-mfcc", type=int, required=False, default=20)
parser.add_argument("--feat-type", type=str, required=False,
  default="logmel", choices=["logmel", "mfcc"])
args = parser.parse_args()

import librosa
import scipy.fftpack
import soundfile
import numpy as np
import matplotlib.pyplot as plt

wave = args.wav
basename = '.'.join(wave.split('/')[-1].split('.')[:-1])

pcm, sr = soundfile.read(wave)
  
f = np.abs(librosa.stft(pcm, n_fft=int(sr*args.frame_sec), 
  hop_length=int(sr*args.shift_sec), window='hamming'))**2.
mel = librosa.filters.mel(sr, int(sr*args.frame_sec), n_mels=args.num_mel)
f = np.dot(mel, f)

if args.feat_type == "mfcc":
  db = 10. * np.log10(np.maximum(f, 1e-10))
  f = scipy.fftpack.dct(db, axis=0, type=2, norm='ortho')
  f = f[:args.num_mfcc]
  
np.save('{}_{}'.format(basename, args.feat_type), f)
