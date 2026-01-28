from typing import Any
from numpy.typing import NDArray
from sympy import NDimArray
import numpy as np
import whisper
import soundcard as sc
import librosa
import asyncio
import threading
import global_state
SAMPLERATE=16000
SECONDS=10






def main():
    state = global_state.GlobalState()

    print(state.loopback)
    with state.loopback, state.speaker:
        t1 = threading.Thread(target=lambda: state.get_audio_data())
        t2 = threading.Thread(target=lambda: state.transcribe())

        t1.start()
        t2.start()

        t1.join()
        t2.join()

#TODO: GET `get_audio_data()` and `transcribe` to execute in parallel




main()