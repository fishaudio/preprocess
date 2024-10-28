# This file is edited from Anya

from pathlib import Path
from typing import Iterable, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

from fish_audio_preprocess.utils.slice_audio import slice_by_max_duration
from fish_audio_preprocess.utils.slice_audio_v2 import merge_short_chunks


def slice_audio_v3(
    audio: np.ndarray,
    rate: int,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    min_silence_duration: float = 0.3,
    speech_pad_s: float = 0.1,
    merge_short: bool = False,
) -> Iterable[np.ndarray]:
    """Slice audio by silence

    Args:
        audio: audio data, in shape (samples, channels)
        rate: sample rate
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        min_silence_duration: minimum duration of silence
        speech_pad_s: final speech chunks are padded by speech_pad_s each side
        merge_short: merge short slices automatically

    Returns:
        Iterable of sliced audio
    """

    if len(audio) / rate < min_duration:
        sliced_by_max_duration_chunk = slice_by_max_duration(audio, max_duration, rate)
        yield from (
            merge_short_chunks(sliced_by_max_duration_chunk, max_duration, rate)
            if merge_short
            else sliced_by_max_duration_chunk
        )
        return

    vad_model = load_silero_vad()

    wav = torch.from_numpy(audio)
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)

    sr = 16000
    if sr != rate:
        transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        wav = transform(wav)

    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        min_silence_duration_ms=int(min_silence_duration * 1000),
        min_speech_duration_ms=int(min_duration * 1000),
        speech_pad_ms=int(speech_pad_s * 1000),
        max_speech_duration_s=max_duration,
        return_seconds=True,
    )

    sliced_audio = [
        audio[int(timestamp["start"] * rate) : int(timestamp["end"] * rate)]
        for timestamp in speech_timestamps
    ]

    if merge_short:
        sliced_audio = merge_short_chunks(sliced_audio, max_duration, rate)

    for chunk in sliced_audio:
        sliced_by_max_duration_chunk = slice_by_max_duration(chunk, max_duration, rate)
        yield from sliced_by_max_duration_chunk


def slice_audio_file_v3(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    min_silence_duration: float = 0.1,
    speech_pad_s: float = 0.1,
    flat_layout: bool = False,
    merge_short: bool = False,
) -> None:
    """
    Slice audio by silence and save to output folder

    Args:
        input_file: input audio file
        output_dir: output folder
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        min_silence_duration: minimum duration of silence
        speech_pad_s: final speech chunks are padded by speech_pad_s each side
        flat_layout: use flat directory structure
        merge_short: merge short slices automatically
    """

    output_dir = Path(output_dir)
    audio, rate = librosa.load(str(input_file), sr=None, mono=True)
    for idx, sliced in enumerate(
        slice_audio_v3(
            audio,
            rate,
            min_duration=min_duration,
            max_duration=max_duration,
            speech_pad_s=speech_pad_s,
            min_silence_duration=min_silence_duration,
            merge_short=merge_short,
        )
    ):
        if flat_layout:
            sf.write(str(output_dir) + f"_{idx:04d}.wav", sliced, rate)
        else:
            sf.write(str(output_dir / f"{idx:04d}.wav"), sliced, rate)
