import os
import shutil
from pathlib import Path
from typing import Union

from loguru import logger

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".flv",
    ".mov",
    ".wmv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def list_files(
    path: Union[Path, str],
    extensions: set[str] = None,
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Extensions to filter. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    files = (
        [
            Path(os.path.join(root, filename))
            for root, _, filenames in os.walk(path, followlinks=True)
            for filename in filenames
            if Path(os.path.join(root, filename)).is_file()
        ]
        if recursive
        else [f for f in path.glob("*") if f.is_file()]
    )

    if extensions is not None:
        files = [f for f in files if f.suffix in extensions]

    if sort:
        files = sorted(files)

    return files


def make_dirs(path: Union[Path, str], clean: bool = False):
    """Make directories.

    Args:
        path (Union[Path, str]): Path to the directory.
        clean (bool, optional): Whether to clean the directory. Defaults to False.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists():
        if clean:
            logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
        else:
            logger.info(f"Output directory already exists: {path}")

    path.mkdir(parents=True, exist_ok=True)
