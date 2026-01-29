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
import argparse
SAMPLERATE=16000
SECONDS=10






def main():
    parser = argparse.ArgumentParser(description="Simple, universal subtitiles, anywhere")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        default="base",
        required=False,
        help="Choose OpenAI Whisper model to use"
    )
    parser.add_argument(
        "--translate",
        type=str,
        default=None,
        required=False,
        help="Choose language to translate subtitules into"
    )

    args = parser.parse_args()
    model_name: str = args.model
    translate: str = args.translate



    state = global_state.SubtitleGenerator(model_name, translate_lang=translate)

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