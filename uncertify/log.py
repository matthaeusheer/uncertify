import os
import logging.config
import yaml
from pathlib import Path

from uncertify.common import PROJECT_ROOT_PATH
from uncertify.utils.path_utils import listdir_full_path

LOGS_DIR = PROJECT_ROOT_PATH / 'logs'
LOGGING_CFG_PATH = PROJECT_ROOT_PATH / 'logging_config.yaml'
LOGGER = logging.getLogger(__name__)


def setup_logging(cfg_path: Path = LOGGING_CFG_PATH, default_level: int = logging.INFO) -> None:
    """Load logging config, setup log output dir, eventually clean logs."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    if cfg_path.exists():
        with cfg_path.open() as cfg_obj:
            config = yaml.safe_load(cfg_obj.read())

        # make sure the log directory is always at the project root and not relative to the config call location
        for handler_name, handler_dict in config['handlers'].items():
            if 'filename' in handler_dict.keys():
                config['handlers'][handler_name]['filename'] = os.path.join(LOGS_DIR, handler_dict['filename'])
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def clean_logs(log_dir_path: Path) -> None:
    """Erases all content in the log files. Since the logging setup has been done at this point already, we cannot
    simply delete the files but rather have to delete the contents."""
    # Delete back counted log files (e.g. error2.log which was created when error1.log reached max capacity)
    for file_path in listdir_full_path(log_dir_path):
        file_name, suffix = str(file_path).split('.')
        if file_name[-1].isdigit():
            os.remove(str(file_path))
    # Empty contents of current logfiles
    for file_path in listdir_full_path(log_dir_path):
        if os.path.exists(str(file_path)):
            with open(str(file_path), "w"):
                pass  # will empty the file :-)
    LOGGER.info(f'Cleaned log dir {LOGS_DIR}')
