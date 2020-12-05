from pathlib import Path
import json


def store_dict(dictionary: dict, dir_path: Path, file_name: str) -> None:
    """Stores a dictionary to disk in json format."""
    with (dir_path / file_name).open('w') as outfile:
        json.dump(dictionary, outfile, indent=4)


def load_dict(file_path: Path) -> dict:
    """Loads a json file into a dictionary."""
    with file_path.open('r') as infile:
        return json.load(infile)
