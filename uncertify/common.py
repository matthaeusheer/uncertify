"""
Helper file to setup common structure in the repo such as default PATHS or other global variables which are
used throughout the project.
"""

import os
from pathlib import Path

DATA_FOLDER_NAME = 'data'
PROJECT_ROOT_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR_PATH = PROJECT_ROOT_PATH / DATA_FOLDER_NAME
