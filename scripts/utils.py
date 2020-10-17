import argparse
from pathlib import Path

class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """Shows default arguments when doing --help."""
    def add_argument(self, *args, help=None, default=None, **kwargs):
        if help is not None:
            kwargs['help'] = help
        if default is not None and args[0] != '-h':
            kwargs['default'] = default
            if help is not None:
                kwargs['help'] += ' Default: {}'.format(default)
        super().add_argument(*args, **kwargs)
