"""Shared helpers for coughkit command-line entry points."""

import argparse
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=None)
def get_version():
    """Return the installed coughkit version, with a source-tree fallback."""
    try:
        return version("coughkit")
    except PackageNotFoundError:
        from coughkit import __version__

        return __version__


def add_version_argument(parser):
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )


def build_parser(add_arguments, description, prog=None):
    """Build an ArgumentParser with the standard ``--version`` flag and arguments.

    ``add_arguments`` is a callable that registers a command's options on the
    parser; ``description`` is shown in ``--help`` (typically the module docstring).
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    add_version_argument(parser)
    add_arguments(parser)
    return parser
