import math
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
import time
import shutil
import os

SAMPLERATE=16000
SECONDS=30

class SubtitleGenerator:
    data: np.ndarray
    text: str
    model: whisper.Whisper
    speaker = sc.default_speaker().player(samplerate=SAMPLERATE)


    def __init__(self, model_name: str, translate_lang: str | None) -> None:
        self.data = np.array([],  dtype=np.float32)
        self.text = ""
        self.loopback = SubtitleGenerator.get_loopback_device()
        self.model = whisper.load_model(model_name)
        self.translate_lang = translate_lang

    def display_text(self):
        last_len = 0
        
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(self.text)
            time.sleep(0.1)
            # cols = shutil.get_terminal_size().columns
            # rows_used = math.ceil(last_len / cols)

            # for i in range(rows_used):
            #     print("\033[F", end="")
            #     print("\033[2K", end="")

            # text = self.text.replace("\n", " ")
            # text = f"\33[2K\rTranscription: {text}"
            # print(text, end="", flush=True)
            # last_len = len(text)
            # time.sleep(0.01)
            
        

    @classmethod
    def get_loopback_device(cls):
        loopback_devices = sc.all_microphones(include_loopback=True)
        print([x.name for x in loopback_devices])
        device = loopback_devices[2]
        print(f"Using device {device.name}")
        mic = device.recorder(samplerate=SAMPLERATE)

        return mic

    #Runs in a cycle and writes audio data into `self.data` buffer
    def start_audio_data_stream(self):
        while True:
            #print("START AUDIO DATA COLLECTION")
            data_next: np.ndarray = self.loopback.record(numframes=None)
            data_next = data_next.mean(axis=1)
            data_next = data_next.astype("float32")
            #print(f"Got {len(data_next)} samples")
            
            self.data = np.concatenate((self.data, data_next), casting="unsafe")
            if (len(self.data) > SAMPLERATE*SECONDS):
                self.data = self.data[ len(self.data)-(SAMPLERATE*SECONDS)  :]
                #self.data = data_next #No need to endlessly accumulate data
            #TODO: FIND A WAY TO KEEP "THE LAST 30 SECONDS" RATHER THAN JUST DELETING THE DATA

    #Runs in a cycle and transcribes text from `self.data` buffer
    def start_transcription_routine(self):
        last_length = 0

        while True:
            if (last_length == len(self.data)):
                #continue #Skip if no data has been added to toe buffer
                pass
            else:
                last_length = len(self.data)
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
            #print(self.text)