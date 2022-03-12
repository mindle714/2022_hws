import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, required=True,
  help="wave input to be processed")
parser.add_argument("--frame-sec", type=float, required=False, default=0.02,
  help="width of frame in seconds")
parser.add_argument("--shift-sec", type=float, required=False, default=0.01,
  help="distance between adjacent frames in seconds")
parser.add_argument("--num-mel", type=int, required=False, default=40,
  help="number of mel bands")
parser.add_argument("--num-mfcc", type=int, required=False, default=20,
  help="number of mfcc")
parser.add_argument("--feat-type", type=str, required=False,
  default="logmel", choices=["logmel", "mfcc"],
  help="type of features to be extracted")
args = parser.parse_args()

import librosa
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

  # f = scipy.fftpack.dct(db, axis=0, type=2, norm='ortho')
  def ortho_dct(e):
    N = e.shape[-1]

    dct_W = np.zeros([N, N], dtype='float')
    for n in range(N):
      for k in range(N):
        dct_W[n][k] = np.sqrt(2./N) * np.cos(np.pi/N*(n+.5)*k)

    res = np.matmul(e, dct_W)
    res[:,0] /= np.sqrt(2.)
    return res

  f = np.transpose(ortho_dct(np.transpose(db)))
  f = f[:args.num_mfcc]
  
np.save('{}_{}'.format(basename, args.feat_type), f)
