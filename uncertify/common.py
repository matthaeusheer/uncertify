"""
Helper file to setup common structure in the repo such as default PATHS or other global variables which are
used throughout the project.
"""

import os
from pathlib import Path

PROJECT_ROOT_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
DATA_FOLDER_NAME = 'data'
DATA_DIR_PATH = PROJECT_ROOT_PATH / DATA_FOLDER_NAME
HD_DATA_PATH = Path('/mnt/2TB_internal_HD/datasets')
