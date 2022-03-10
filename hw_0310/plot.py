import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, required=True)
parser.add_argument("--frame-sec", type=float, required=False, default=0.02)
parser.add_argument("--snapshot", action="store_true")
args = parser.parse_args()

import soundfile
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

wave = args.wav
frame_sec = args.frame_sec

pcm, sr = soundfile.read(wave)
frame_samp = int(sr * frame_sec)
window = scipy.signal.get_window('hanning', frame_samp)

def magnitude(e): return np.abs(e) / np.abs(window).sum()
def phase(e): return np.arctan2(e.imag, e.real) #np.angle(e) 

fig = plt.figure()
ax_mag = fig.add_subplot(2, 1, 1)
ax_mag.set_title("magnitude")
ax_phase = fig.add_subplot(2, 1, 2)
ax_phase.set_title("phase")

frames = []
offsets = np.arange(0, pcm.shape[0]-frame_samp, frame_samp)
if args.snapshot:
  offsets = [(pcm.shape[0]-frame_samp)//2]

for offset in offsets:
  segment = pcm[offset:offset+frame_samp]
  segment *= window
  frames.append(
    ax_mag.plot(np.arange(frame_samp//2+1), magnitude(np.fft.rfft(segment)), animated=True, color='blue') + \
    ax_phase.plot(np.arange(frame_samp//2+1), phase(np.fft.rfft(segment)), animated=True, color='blue'))

basename = '.'.join(wave.split('/')[-1].split('.')[:-1])
if args.snapshot:
  plt.savefig('{}.png'.format(basename))
else:
  ani = animation.ArtistAnimation(fig, frames,
    interval=frame_sec*1000, repeat=False)
  ani.save('{}.mp4'.format(basename))
