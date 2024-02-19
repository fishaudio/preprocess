from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files


def process_one(file, input_dir):
    import soundfile as sf
    sound = sf.SoundFile(str(file))
    return (
        len(sound),
        sound.samplerate,
        len(sound) / sound.samplerate,
        file.relative_to(input_dir),
    )


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--visualize/--no-visualize", default=False, help="Visualize the distribution"
)
@click.option(
    "-l", "--long-threshold", default=None, type=float, help="Threshold for long files"
)
@click.option(
    "-s",
    "--short-threshold",
    default=None,
    type=float,
    help="Threshold for short files",
)
def length(
    input_dir: str,
    recursive: bool,
    visualize: bool,
    long_threshold: Optional[float],
    short_threshold: Optional[float],
):
    """
    Get the length of all audio files in a directory
    """
    import soundfile as sf
    from matplotlib import pyplot as plt
    input_dir = Path(input_dir)
    files = list_files(input_dir, AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, calculating length")

    infos = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        tasks = []
        for file in tqdm(files, desc="Preparing"):
            tasks.append(executor.submit(process_one, file, input_dir))
        for task in tqdm(tasks, desc="Processing"):
            infos.append(task.result())

    # Duration
    total_duration = sum(i[2] for i in infos)
    avg_duration = total_duration / len(infos)
    logger.info(f"Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"Average duration: {avg_duration:.2f} seconds")
    logger.info(f"Max duration: {max(i[2] for i in infos):.2f} seconds")
    logger.info(f"Min duration: {min(i[2] for i in infos):.2f} seconds")

    # Too Long
    if long_threshold is not None:
        long_files = [i for i in infos if i[2] > float(long_threshold)]

        # sort by duration
        if long_files:
            long_files = sorted(long_files, key=lambda x: x[2], reverse=True)
            logger.warning(
                f"Found {len(long_files)} files longer than {long_threshold} seconds"
            )
            for i in [f"{i[3]}: {i[2]:.2f}" for i in long_files]:
                logger.warning(f"    {i}")

    # Too Short
    if short_threshold is not None:
        short_files = [i for i in infos if i[2] < float(short_threshold)]

        if short_files:
            short_files = sorted(short_files, key=lambda x: x[2], reverse=False)
            logger.warning(
                f"Found {len(short_files)} files shorter than {short_threshold} seconds"
            )
            for i in [f"{i[3]}: {i[2]:.2f}" for i in short_files]:
                logger.warning(f"    {i}")

    # Sample Rate
    total_samplerate = sum(i[1] for i in infos)
    avg_samplerate = total_samplerate / len(infos)
    logger.info(f"Average samplerate: {avg_samplerate:.2f}")

    if not visualize:
        return

    # Visualize
    plt.hist([i[2] for i in infos], bins=100)
    plt.title(
        f"Distribution of audio lengths (Total: {len(infos)} files, {total_duration / 3600:.2f} hours)"
    )
    plt.xlabel("Length (seconds)")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    length()
