from typing import Any
from numpy.typing import NDArray
from sympy import NDimArray
import numpy as np
import whisper
import soundcard as sc
import librosa
SAMPLERATE=16000
SECONDS=10

def get_loopback_device():
    loopback_devices = sc.all_microphones(include_loopback=True)
    print([x.name for x in loopback_devices])
    device = loopback_devices[2]
    print(f"Using device {device.name}")
    mic = device.recorder(samplerate=SAMPLERATE)

    return mic

model = whisper.load_model("base")
loopback = get_loopback_device()
speaker = sc.default_speaker().player(samplerate=SAMPLERATE)



def get_audio_data() -> np.ndarray:
    print("START AUDIO DATA COLLECTION")
    data = loopback.record(numframes=SAMPLERATE*SECONDS)
    data = data.mean(axis=1)
    data = data.astype("float32")
    print(f"Got {len(data)} samples")

    return data

def transcribe(audio: np.ndarray):
    audio = whisper.pad_or_trim(audio) # type: ignore
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}") # pyright: ignore[reportAttributeAccessIssue]
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text # pyright: ignore[reportAttributeAccessIssue]


def main():
    print(loopback)
    while (True):
        with loopback, speaker:
            data = get_audio_data()
            text = transcribe(data)
            print(f"Got transcription: {text}")





main()