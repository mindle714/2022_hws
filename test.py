import librosa

pcm, _ = librosa.load("test.wav")
print(pcm.shape)
