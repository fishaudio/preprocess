import subprocess as sp
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    list_files,
    make_dirs,
)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--overwrite/--no-overwrite", default=False, help="Overwrite existing files"
)
@click.option(
    "--clean/--no-clean", default=False, help="Clean output directory before processing"
)
@click.option(
    "--segment",
    help="Maximum segment length in seconds, use 0 to disable",
    default=60 * 30,
    show_default=True,
)
def to_wav(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    segment: int,
):
    """Converts all audio and video files in input_dir to wav files in output_dir."""

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(
        input_dir, extensions=VIDEO_EXTENSIONS | AUDIO_EXTENSIONS, recursive=recursive
    )
    logger.info(f"Found {len(files)} files, converting to wav")

    skipped = 0
    for file in tqdm(files):
        # Get relative path to input_dir
        relative_path = file.relative_to(input_dir)
        new_file = (
            output_dir
            / relative_path.parent
            / relative_path.name.replace(
                file.suffix, "_%04d.wav" if segment > 0 else ".wav"
            )
        )

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)

        check_path = (
            (new_file.parent / (new_file.name % 0)) if segment > 0 else new_file
        )
        if check_path.exists() and not overwrite:
            skipped += 1
            continue

        command = ["ffmpeg", "-y", "-nostats", "-loglevel", "error", "-i", str(file)]

        if segment > 0:
            command.extend(["-f", "segment", "-segment_time", str(segment)])

        command.append(str(new_file))

        sp.check_call(command)

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    to_wav()
