import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs


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
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=os.cpu_count(),
    show_default=True,
    type=int,
)
@click.option(
    "--min-duration",
    help="Minimum duration of each slice",
    default=6.0,
    show_default=True,
    type=float,
)
@click.option(
    "--max-duration",
    help="Maximum duration of each slice",
    default=30.0,
    show_default=True,
    type=float,
)
@click.option(
    "--pad-silence",
    help="Pad silence between each non-silent slice",
    default=0.4,
    show_default=True,
    type=float,
)
@click.option(
    "--top-db",
    help="top_db of librosa.effects.split",
    default=60,
    show_default=True,
    type=int,
)
@click.option(
    "--frame-length",
    help="frame_length of librosa.effects.split",
    default=2048,
    show_default=True,
    type=int,
)
@click.option(
    "--hop-length",
    help="hop_length of librosa.effects.split",
    default=512,
    show_default=True,
    type=int,
)
def slice_audio(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    num_workers: int,
    min_duration: float,
    max_duration: float,
    pad_silence: float,
    top_db: int,
    frame_length: int,
    hop_length: int,
):
    """Slice audio files into smaller chunks by silence."""

    from fish_audio_preprocess.utils.slice_audio import slice_audio_file

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")

    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_dir)
            save_path = output_dir / relative_path.parent / relative_path.stem

            if save_path.exists() and not overwrite:
                skipped += 1
                continue

            if save_path.exists() is False:
                save_path.mkdir(parents=True)

            tasks.append(
                executor.submit(
                    slice_audio_file,
                    input_file=str(file),
                    output_dir=save_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    pad_silence=pad_silence,
                    top_db=top_db,
                    frame_length=frame_length,
                    hop_length=hop_length,
                )
            )

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


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
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=os.cpu_count(),
    show_default=True,
    type=int,
)
@click.option(
    "--min-duration",
    help="Minimum duration of each slice",
    default=5.0,
    show_default=True,
    type=float,
)
@click.option(
    "--max-duration",
    help="Maximum duration of each slice",
    default=30.0,
    show_default=True,
    type=float,
)
@click.option(
    "--min-silence-duration",
    help="Minimum duration of each slice",
    default=0.3,
    show_default=True,
    type=float,
)
@click.option(
    "--top-db",
    help="top_db of librosa.effects.split",
    default=-40,
    show_default=True,
    type=int,
)
@click.option(
    "--hop-length",
    help="hop_length of librosa.effects.split",
    default=10,
    show_default=True,
    type=int,
)
@click.option(
    "--max-silence-kept",
    help="Maximum duration of each slice",
    default=0.5,
    show_default=True,
    type=float,
)
@click.option(
    "--flat-layout/--no-flat-layout", default=False, help="Use flat directory structure"
)
@click.option(
    "--merge-short/--no-merge-short",
    default=False,
    help="Merge short slices automatically",
)
def slice_audio_v2(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    num_workers: int,
    min_duration: float,
    max_duration: float,
    min_silence_duration: float,
    top_db: int,
    hop_length: int,
    max_silence_kept: float,
    flat_layout: bool,
    merge_short: bool,
):
    """(OpenVPI version) Slice audio files into smaller chunks by silence."""

    from fish_audio_preprocess.utils.slice_audio_v2 import slice_audio_file_v2

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if flat_layout:
        logger.info("Using flat directory structure")

    if merge_short:
        logger.info("Merging short slices automatically")

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")

    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_dir)
            save_path = output_dir / relative_path.parent / relative_path.stem

            if save_path.exists() and not overwrite:
                skipped += 1
                continue

            if (
                output_dir / relative_path.parent / relative_path.stem
                if not flat_layout
                else output_dir / relative_path.parent
            ).exists() is False:
                (
                    output_dir / relative_path.parent / relative_path.stem
                    if not flat_layout
                    else output_dir / relative_path.parent
                ).mkdir(parents=True)

            tasks.append(
                executor.submit(
                    slice_audio_file_v2,
                    input_file=str(file),
                    output_dir=save_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_silence_duration=min_silence_duration,
                    top_db=top_db,
                    hop_length=hop_length,
                    max_silence_kept=max_silence_kept,
                    flat_layout=flat_layout,
                    merge_short=merge_short,
                )
            )

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    slice_audio()
