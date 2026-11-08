import librosa
import numpy as np
import subprocess

def room_level_db(path):
    data, sr = librosa.load(path, sr=None, mono=True)
    rms = np.sqrt(np.mean(data ** 2))
    return int(20 * np.log10(rms) if rms > 0 else float("-inf"))



subprocess.run(["ffmpeg", "-y", "-i", "/home/csaliba/my_video-2.mkv", "-vn", "/home/csaliba/my_video-2.wav"], check=True)

print(f"{room_level_db('/home/csaliba/my_video-2.wav')} dBFS")