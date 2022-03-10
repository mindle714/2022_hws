import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, required=True)
parser.add_argument("--frame-sec", type=float, required=False, default=0.02)
parser.add_argument("--plot-type", type=str, required=False,
  choices=["mag", "phase"], default="mag")
parser.add_argument("--snapshot", action="store_true")
args = parser.parse_args()

import soundfile
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

wave = args.wav
frame_sec = args.frame_sec
plot_mag = True if args.plot_type == "mag" else False

pcm, sr = soundfile.read(wave)
frame_samp = int(sr * frame_sec)
window = scipy.signal.get_window('hanning', frame_samp)

def magnitude(e): return np.abs(e)
def phase(e): return np.arctan2(e.imag, e.real) #np.angle(e) 
func = magnitude if plot_mag else phase

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log')
ax.get_xaxis().set_visible(False)

frames = []
offsets = np.arange(0, pcm.shape[0]-frame_samp, frame_samp)
if args.snapshot:
  offsets = [(pcm.shape[0]-frame_samp)//2]

for offset in offsets:
  segment = pcm[offset:offset+frame_samp]
  segment *= window
  frames.append(ax.plot(
    np.arange(frame_samp//2+1), func(np.fft.rfft(segment)),
    animated=True, color='blue'))

basename = '.'.join(wave.split('/')[-1].split('.')[:-1])
suf = 'mag' if plot_mag else 'phase'
if args.snapshot:
  plt.savefig('{}_{}.png'.format(basename, suf))
else:
  ani = animation.ArtistAnimation(fig, frames,
    interval=frame_sec*1000, repeat=False)
  ani.save('{}_{}.mp4'.format(basename, suf))
