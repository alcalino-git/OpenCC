import numpy as np
import whisper
from typing import Any
from numpy.typing import NDArray
from sympy import NDimArray
import numpy as np
import whisper
import soundcard as sc
import librosa
import asyncio
import threading

SAMPLERATE=16000
SECONDS=30

class GlobalState:
    data: np.ndarray
    text: str
    model: whisper.Whisper = whisper.load_model("base")
    speaker = sc.default_speaker().player(samplerate=SAMPLERATE)


    def __init__(self) -> None:
        self.data = np.array([],  dtype=np.float32)
        self.text = "..."
        self.loopback = GlobalState.get_loopback_device()

    @classmethod
    def get_loopback_device(cls):
        loopback_devices = sc.all_microphones(include_loopback=True)
        print([x.name for x in loopback_devices])
        device = loopback_devices[2]
        print(f"Using device {device.name}")
        mic = device.recorder(samplerate=SAMPLERATE)

        return mic


    def get_audio_data(self) -> np.ndarray:
        while True:
            #print("START AUDIO DATA COLLECTION")
            data_next: np.ndarray = self.loopback.record(numframes=None)
            data_next = data_next.mean(axis=1)
            data_next = data_next.astype("float32")
            #print(f"Got {len(data_next)} samples")
            
            self.data = np.concatenate((self.data, data_next), casting="unsafe")
            if (len(self.data) > SAMPLERATE*SECONDS):
                print(f"RESETTING DATA BUFFER AFTER {SECONDS}s")
                self.data = data_next #No need to endlessly accumulate data

    def transcribe(self):
        while True:
            #print("Beginning transcription")
            audio = np.copy(self.data)
            audio = audio.astype("float32")
            audio = whisper.pad_or_trim(audio) # type: ignore
            mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            #print(f"Detected language: {max(probs, key=probs.get)}") # pyright: ignore[reportAttributeAccessIssue]
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(self.model, mel, options)

            self.text = result.text # type: ignore
            print(self.text)