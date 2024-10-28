import click
import richuru
from loguru import logger

from fish_audio_preprocess.cli.merge_lab import merge_lab

from .convert_to_wav import to_wav
from .frequency import frequency
from .length import length
from .loudness_norm import loudness_norm
from .merge_short import merge_short
from .resample import resample
from .separate_audio import separate
from .slice_audio import slice_audio, slice_audio_v2
from .transcribe import transcribe


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug: bool):
    """An audio preprocessing CLI."""

    if debug:
        richuru.install()
        logger.info("Debug mode is on")


# Register subcommands
cli.add_command(length)
cli.add_command(frequency)

cli.add_command(to_wav)
cli.add_command(separate)
cli.add_command(loudness_norm)
cli.add_command(slice_audio)
cli.add_command(slice_audio_v2)
cli.add_command(resample)
cli.add_command(transcribe)
cli.add_command(merge_short)
cli.add_command(merge_lab)


if __name__ == "__main__":
    to_wav()
