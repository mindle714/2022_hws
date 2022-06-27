import soundfile
import librosa
import numpy as np

wav1 = "/home/hejung/vctk/wav16/p225/p225_296_mic1.wav"
wav2 = "/home/hejung/vctk/wav16/p226/p226_288_mic1.wav"

pcm1, sr = soundfile.read(wav1)
pcm2, sr = soundfile.read(wav2)

pcm3_len = min(pcm1.shape[0], pcm2.shape[0])
pcm1 = pcm1[:pcm3_len]
pcm2 = pcm2[:pcm3_len]

pcm3 = pcm1 * 0.6 + pcm2 * 0.4
soundfile.write("mix_p225_296_mic1_p226_288_mic1.wav", pcm3, sr)

def magnitude(e): return np.abs(e)
def phase(e): return np.arctan2(e.imag, e.real)
def polar(mag, phase):
  return mag * (np.cos(phase) + np.sin(phase) * 1j)

f1 = librosa.stft(pcm1)
f2 = librosa.stft(pcm2)

f1_mag = librosa.amplitude_to_db(magnitude(f1))
f2_mag = librosa.amplitude_to_db(magnitude(f2))

f3_mag = f1_mag * 0.6 + f2_mag * 0.4
f3 = polar(librosa.db_to_amplitude(f3_mag), phase(f1))
pcm3 = librosa.istft(f3, length=pcm3_len)
soundfile.write("mixmag_p225_296_mic1_p226_288_mic1.wav", pcm3, sr)
