"""
This script servers as a template for all subsequent scripts.
"""
import argparse

import add_uncertify_to_path  # makes sure we can use the uncertify-ai library
import uncertify
from uncertify.log import setup_logging


def parse_args() -> argparse.Namespace:
    """Use argparse to parse command line arguments and pass it on to the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """"Main entry point for our program."""
    # Magic happens here...
    print(f'Successfully loaded {uncertify}')


if __name__ == '__main__':
    setup_logging()
    main(parse_args())
