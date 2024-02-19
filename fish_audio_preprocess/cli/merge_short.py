from pathlib import Path

import click
import librosa
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files
from fish_audio_preprocess.utils.slice_audio_v2 import merge_short_chunks


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--max-duration",
    help="Maximum duration of each file",
    default=20,
    show_default=True,
    type=int,
)
def merge_short(input_dir: str, output_dir: str, recursive: bool, max_duration: int):
    """Merge short audio chunks into longer ones. Caution: This tool will scramble the filenames and this tool need files has same sample rate."""

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files")

    audios = []

    rate = 0
    # 遍历每个音频确定采样率是否一致
    for file in tqdm(files, desc="Checking file"):
        audio, sr = librosa.load(str(file), sr=None, mono=False)
        if rate == 0:
            rate = sr
        if rate != sr:
            raise ValueError(f"Sample rate of {file} is {sr}, not {rate}")
        audios.append(audio)

    logger.info("Start merging")
    res = merge_short_chunks(audios, max_duration, rate)

    for i, audio in enumerate(res):
        sf.write(output_dir / f"{i}.wav", audio, rate)
