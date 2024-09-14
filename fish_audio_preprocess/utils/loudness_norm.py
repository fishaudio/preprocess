import traceback
from pathlib import Path
from typing import Union

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from loguru import logger


def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)

    Returns:
        loudness normalized audio
    """

    # peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    return pyln.normalize.loudness(audio, _loudness, loudness)


def loudness_norm_file(
    input_file, output_file, peak=-1.0, loudness=-23.0, block_size=0.400
):
    try:
        audio, rate = sf.read(str(input_file))
        meter = pyln.Meter(rate)
        loudness_pre = meter.integrated_loudness(audio)

        audio = pyln.normalize.loudness(audio, loudness_pre, loudness)

        audio = pyln.normalize.peak(audio, peak)

        # ファイルを小さなチャンクに分割して書き込む
        chunk_size = 100000  # チャンクサイズを調整する必要があるかもしれません
        with sf.SoundFile(
            output_file,
            "w",
            samplerate=rate,
            channels=audio.shape[1] if len(audio.shape) > 1 else 1,
        ) as f:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                f.write(chunk)

    except Exception as e:
        print(f"Error in loudness_norm_file: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        raise
