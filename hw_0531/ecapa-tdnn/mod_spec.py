import librosa
import scipy.fftpack
import numpy as np

def magnitude(e): return np.abs(e)
def phase(e): return np.arctan2(e.imag, e.real)
def polar(mag, phase):
  return mag * (np.cos(phase) + np.sin(phase) * 1j)

def mps(pcm, axis=0, cnt=1):
  f = librosa.stft(pcm, n_fft=512, hop_length=64)
  f_mag = magnitude(f)
  f_ph = phase(f)

  f2_mag = scipy.fftpack.fft2(f_mag)
  f2_mag_shift = scipy.fftpack.fftshift(f2_mag)

  mag = librosa.amplitude_to_db(magnitude(f2_mag_shift))

  if axis == 0:
    idxs = []
    while len(idxs) < cnt:  
      idx = np.random.randint(0, mag.shape[0])
      if idx not in idxs:
        idxs.append(idx)

    for idx in idxs:
      mag[idx, :] = 0.
      mag[-idx-1, :] = 0.

  else:
    assert axis == 1
    idxs = []
    while len(idxs) < cnt:  
      idx = np.random.randint(0, mag.shape[1])
      if idx not in idxs:
        idxs.append(idx)

    for idx in idxs:
      mag[:, idx] = 0.
      mag[:, -idx-1] = 0.

  ph = phase(f2_mag_shift)

  f_rev = scipy.fftpack.ifft2(scipy.fftpack.ifftshift(
      polar(librosa.db_to_amplitude(mag), ph)), shape=f.shape)
  pcm_rev = librosa.istft(polar(f_rev, f_ph), n_fft=512, hop_length=64, length=pcm.shape[0])

  return pcm_rev
