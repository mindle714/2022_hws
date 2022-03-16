import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target-dir", type=str, required=True)
parser.add_argument("--num-chunk", type=int, required=False, default=10)
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

import os
import glob
wavs = glob.glob(os.path.join(args.target_dir, "*.wav"))
if len(wavs) == 0:
  import sys
  sys.exit("target directory[{}] needs at least 1 wav".format(args.target_dir))

import librosa
import soundfile
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt

def ext_feat(wave):
  pcm, sr = soundfile.read(wave)
  
  f = np.abs(librosa.stft(pcm, n_fft=int(sr*args.frame_sec), 
    hop_length=int(sr*args.shift_sec), window='hamming'))**2.
  mel = librosa.filters.mel(sr=sr, n_fft=int(sr*args.frame_sec), n_mels=args.num_mel)
  f = np.dot(mel, f)

  if args.feat_type == "mfcc":
    db = 10. * np.log10(np.maximum(f, 1e-10))

    f = scipy.fftpack.dct(db, axis=0, type=2, norm='ortho')
    def ortho_dct(e):
      N = e.shape[-1]

      dct_W = np.zeros([N, N], dtype='float')
      for n in range(N):
        for k in range(N):
          dct_W[n][k] = np.sqrt(2./N) * np.cos(np.pi/N*(n+.5)*k)

      res = np.matmul(e, dct_W)
      res[:,0] /= np.sqrt(2.)
      return res

    # f = np.transpose(ortho_dct(np.transpose(db)))
    f = f[:args.num_mfcc]
  return f

out_dim = args.num_mel if args.feat_type == "logmel" else args.num_mfcc
feat_dir = os.path.join(args.target_dir, '{}-{}'.format(args.feat_type, out_dim))
os.makedirs(feat_dir, exist_ok=True)

if args.num_chunk == 0:
  for wav in wavs:
    bname = '.'.join(os.path.basename(wav).split('.')[:-1])
    f = ext_feat(wav)
    np.save(os.path.join(feat_dir, bname), f)

else:
  chunks = [None for _ in range(args.num_chunk)]
  for _i, wav in enumerate(wavs):
    f = ext_feat(wav)
    i = _i % 10
    if chunks[i] is None:
      chunks[i] = f
    else:
      chunks[i] = np.concatenate([chunks[i], f], axis=-1)

  for i, chunk in enumerate(chunks):
    if chunk is not None:
      np.save(os.path.join(feat_dir, 'feat-{}'.format(i)), chunk)
