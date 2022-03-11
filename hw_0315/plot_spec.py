import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, required=True)
parser.add_argument("--frame-sec", type=float, required=False, default=0.02)
args = parser.parse_args()

import librosa
import librosa.display
import soundfile
import numpy as np
import matplotlib.pyplot as plt

wave = args.wav

pcm, sr = soundfile.read(wave)
f = librosa.stft(pcm, n_fft=int(sr*args.frame_sec))
db = librosa.amplitude_to_db(np.abs(f), ref=np.max) 

fig = plt.figure(figsize=(19.2, 4.8))
librosa.display.specshow(db, x_axis='time', y_axis='linear',
  sr=sr, hop_length=int(sr*args.frame_sec)//4)
plt.colorbar()

basename = '.'.join(wave.split('/')[-1].split('.')[:-1])
suffix = int(args.frame_sec*1000)
plt.savefig('{}_{}ms.png'.format(basename, suffix))
