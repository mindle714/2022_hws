import soundfile
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import playsound

frame_sec = 0.02
plot_mag = True

pcm, sr = soundfile.read("hejung.wav")
frame_samp = int(sr * frame_sec)
window = scipy.signal.get_window('hanning', frame_samp)

def magnitude(e): return np.abs(e)
def phase(e): return np.angle(e) 
func = magnitude if plot_mag else phase

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log')
ax.get_xaxis().set_visible(False)

frames = []
for offset in [7938]:
#for offset in np.arange(0, pcm.shape[0], frame_samp):
  segment = pcm[offset:offset+frame_samp]
  segment *= window
  frames.append(ax.plot(
    np.arange(frame_samp//2+1), func(np.fft.rfft(segment)),
    animated=True, color='blue'))
  plt.savefig('hejung_{}.png'.format(offset))

ani = animation.ArtistAnimation(fig, frames,
  interval=frame_sec*1000*20, repeat=False)
plt.show()
