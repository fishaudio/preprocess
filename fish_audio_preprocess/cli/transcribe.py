import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import click
import torch
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, split_list
from fish_audio_preprocess.utils.transcribe import batch_transcribe


def replace_lastest(string, old, new):
    return string[::-1].replace(old[::-1], new[::-1], 1)[::-1]


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--num-workers",
    help="Number of workers to use for processing, defaults to 2",
    default=2,
    show_default=True,
    type=int,
)
@click.option(
    "--lang",
    help="language",
    default="zh",
    show_default=True,
    type=str,
)
@click.option(
    "--model-size",
    help="whisper model size",
    default="tiny",
    show_default=True,
    type=str,
)
def transcribe(input_dir, num_workers, lang, model_size):
    """
    Transcribe audio files in a directory.
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available, using CPU. This will be slow and even this script can not work. "
            "To speed up, use a GPU enabled machine or install torch with cuda builtin."
        )

    logger.info(f"Transcribing audio files in {input_dir}")
    # 扫描出所有的音频文件
    audio_files = list_files(input_dir)
    audio_files = [str(file) for file in audio_files if file.suffix in AUDIO_EXTENSIONS]

    if len(audio_files) == 0:
        logger.error(f"No audio files found in {input_dir}.")
        return

    # 按照 num workers 切块
    chunks = split_list(audio_files, num_workers)

    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        tasks = []
        for chunk in chunks:
            tasks.append(
                executor.submit(
                    batch_transcribe,
                    files=chunk,
                    model_size=model_size,
                    lang=lang,
                    pos=len(tasks),
                )
            )
        results = {}
        for task in tasks:
            ret = task.result()
            for res in ret.keys():
                results[res] = ret[res]

        logger.info("Output to .lab file")
        for file in tqdm(results.keys()):
            path = replace_lastest(file, ".wav", ".lab")
            # logger.info(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(results[file])
