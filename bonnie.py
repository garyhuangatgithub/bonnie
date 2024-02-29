# Bravo Octave Neural Network Intelligence Engine
# Filename bonnie.py
# Prerequisites
# c:\>python -m pip install librosa
# DOS Command Line Execution
# Execution
# c:\>python bonnie.py

# Open bonnie-vocal.wav and plot in time domain
import librosa
audio_path = 'bonnie-vocal.wav'         # Vocal Waveform
vocal, sr = librosa.load(audio_path)
audio_path = 'bonnie-midi-vocal.wav'    # MIDI Waveform
midi, sr = librosa.load(audio_path)
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(12, 5))
librosa.display.waveshow(midi, sr=sr)   # Plot MIDI Waveform
librosa.display.waveshow(vocal, sr=sr)  # Plot Vocal Waveform

# Spectral Analysis with Spectral f0
import numpy as np
y, sr = librosa.load('bonnie-vocal.wav')
f0, voiced_flag, voiced_probs = librosa.pyin(y,fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0)
import matplotlib.pyplot as plt
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')

# Spectral f0 MIDI
y, sr = librosa.load('bonnie-midi-vocal.wav')
f0midi, voiced_flag, voiced_probs = librosa.pyin(y,fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0midi)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Pitch Shifting
f0Avg = np.nanmean(f0)
f0midiAvg = np.nanmean(f0midi)
print("f0 = ", f0Avg)
print("f0midi = ", f0midiAvg)
f0Octave = librosa.hz_to_octs(f0Avg)
f0midiOctave = librosa.hz_to_octs(f0midiAvg)
print("f0 Octave = ", f0Octave)
print("f0midi Octave = ", f0midiOctave)
keyshift = 12 * (f0Octave - f0midiOctave)
print("Key Shift = ", keyshift)
f0Note = librosa.hz_to_note(f0Avg)
f0midiNote = librosa.hz_to_note(f0midiAvg)
print("f0 Note = ", f0Note)
print("f0midi Note = ", f0midiNote)
yTune = librosa.effects.pitch_shift(y, sr=sr, n_steps=keyshift)
f0midi, voiced_flag, voiced_probs = librosa.pyin(yTune,fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0midi)
D = librosa.amplitude_to_db(np.abs(librosa.stft(yTune)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0midi, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')

# Scoring the Performance
f0diff = 0
f0deviation = 50  # Judgement Tuning Threshold
f0ceiling = 1
fmin=librosa.note_to_hz('C2')
fmax=librosa.note_to_hz('C7')

for t in range(0, np.size(f0midi)) :
  if ((f0midi[t] >= fmin and f0midi[t] <= fmax) and (f0[t] >= fmin and f0[t] <= fmax)) :
    f0diff = f0diff + np.abs(f0midi[t] - f0[t])
    f0ceiling = f0ceiling + f0deviation

f0score = (f0ceiling - f0diff) / f0ceiling
f0score = f0score * 100  # Scale the score to percentage 
print(f0score)

# Plot Vocal Against MIDI
fig, ax = plt.subplots()
times = librosa.times_like(f0)
ax.plot(times, f0, label='f0', color='r')
times = librosa.times_like(f0midi)
ax.plot(times, f0midi, label='f0 MIDI', color='b')
scorestring = str(f0score)
titlescore = 'Score ' + scorestring
ax.set(title=titlescore)
plt.show()  # Display Plot
