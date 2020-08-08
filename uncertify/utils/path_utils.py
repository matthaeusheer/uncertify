import os
from pathlib import Path

from typing import List


def listdir_full_path(dir_path: Path) -> List[Path]:
    """os.listdir(path) only gives us the file names while this function gives us the absolute paths of all items."""
    return [dir_path / item_name for item_name in os.listdir(str(dir_path))]


def get_current_file_name() -> str:
    print(Path(__file__))
    return ''
