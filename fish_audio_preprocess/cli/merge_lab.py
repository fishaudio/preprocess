import os

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.cli.transcribe import replace_lastest
from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_file", type=click.Path(exists=False))
@click.argument("template", type=str)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Search recursively",
)
def merge_lab(
    input_dir: str,
    output_file: str,
    template: str,
    recursive: bool,
):
    audio_files = list_files(input_dir, recursive=recursive)
    audio_files = [str(file) for file in audio_files if file.suffix in AUDIO_EXTENSIONS]
    results = []
    for audio_file in tqdm(audio_files):
        # logger.info(f"Processing {audio_file}")
        lab_file = replace_lastest(audio_file, ".wav", ".lab")
        if not os.path.exists(lab_file) or not os.path.isfile(lab_file):
            logger.warning(f"lab file not found for {audio_file}")

        try:
            lab_content = open(lab_file, "r", encoding="utf-8").read()
        except UnicodeDecodeError:
            lab_content = open(lab_file, "r", encoding="gbk").read()
        except Exception as e:
            logger.error(f"Error reading lab file {lab_file}: {e}")
            continue
        results.append(
            template.replace("{TEXT}", lab_content).replace("{PATH}", audio_file)
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
