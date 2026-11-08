import librosa
import numpy as np

def room_level_db(path):
    data, sr = librosa.load(path, sr=None, mono=True)
    rms = np.sqrt(np.mean(data ** 2))
    return int(20 * np.log10(rms) if rms > 0 else float("-inf"))

print(f"{room_level_db('/home/csaliba/Documents/Room-Capacity-Sensor/data/sample-speech-1m.mp3')} dBFS")