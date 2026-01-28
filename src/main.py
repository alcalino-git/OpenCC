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
        t1 = threading.Thread(target=lambda: state.start_audio_data_stream())
        t2 = threading.Thread(target=lambda: state.start_transcription_routine())
        t3 = threading.Thread(target=lambda: state.display_text())

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()
    

    

#TODO: FIND A WAY TO NOT MAKE OLD TRANSCRIPTIONS OVERRIDE NEW ONES
#MAYBE USE TIME STAMPS????


main()