# modulation power spectrum
import soundfile
import librosa
import numpy as np
import scipy.fftpack

def magnitude(e): return np.abs(e)
def phase(e): return np.arctan2(e.imag, e.real)
def polar(mag, phase):
  return mag * (np.cos(phase) + np.sin(phase) * 1j)

def mps(pcm):
  f = librosa.stft(pcm, n_fft=512, hop_length=64)
  print(f.shape)
  f_mag = magnitude(f)
  f_ph = phase(f)

  f2_mag = scipy.fftpack.fft2(f_mag)
  #f2_mag_shift = scipy.fftpack.fftshift(f2_mag, axes=(1,))
  f2_mag_shift = scipy.fftpack.fftshift(f2_mag)

  mag = librosa.amplitude_to_db(magnitude(f2_mag_shift))
  #mag[150,:] *= 0. #v2sym
  #mag[-151,:] *= 0. #v2sym
  mag[:,150] *= 0. #v3sym
  mag[:,-151] *= 0. #v3sym
  #mag[:20,:20] = np.mean(mag[:20,:20])
  ph = phase(f2_mag_shift)

  #f_rev = scipy.fftpack.ifft2(scipy.fftpack.ifftshift(
  #    polar(librosa.db_to_amplitude(mag), ph), axes=(1,)), shape=f.shape)
  #f_rev = scipy.fftpack.ifft2(
  #    polar(librosa.db_to_amplitude(mag), ph), shape=f.shape)
  f_rev = scipy.fftpack.ifft2(scipy.fftpack.ifftshift(
      polar(librosa.db_to_amplitude(mag), ph)), shape=f.shape)
  pcm_rev = librosa.istft(polar(f_rev, f_ph), n_fft=512, hop_length=64, length=pcm.shape[0])

  return mag, pcm_rev

idx = 0
for wav in [
        "/home/hejung/vctk/wav16/p225/p225_296_mic1.wav",
        "/home/hejung/vctk/wav16/p226/p226_288_mic1.wav",
        "/home/hejung/vctk/wav16/p227/p227_074_mic1.wav"]:

  pcm, sr = soundfile.read(wav)
  pcm = pcm[sr:2*sr]

  pcm_min = np.min(pcm); pcm_max = np.max(pcm)
  pcm_ns = pcm + np.random.uniform(pcm_min/100, pcm_max/100, pcm.shape)
  soundfile.write("mps_ns_{}.wav".format(idx), pcm, sr)

  import matplotlib.pyplot as plt
  fig = plt.figure()

  f, pcm_rev = mps(pcm_ns)
  plt.imshow(f)
  plt.savefig('mps_ns_db_{}_v3sym.png'.format(idx))
  soundfile.write("mps_ns_db_{}_v3sym.wav".format(idx), pcm_rev, sr)
  
  f, pcm_rev = mps(pcm)
  plt.imshow(f)
  plt.savefig('mps_db_{}_v3sym.png'.format(idx))
  soundfile.write("mps_db_{}_v3sym.wav".format(idx), pcm_rev, sr)
  
  #f = mps(pcm_ns) - mps(pcm)
  #plt.imshow(f)
  #plt.savefig('mps_ns_sub_db_{}.png'.format(idx))

  idx += 1
