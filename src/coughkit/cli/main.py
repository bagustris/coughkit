#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coughkit command-line interface."""

import argparse

from coughkit.cli import count as count_cli
from coughkit.cli import detect as detect_cli
from coughkit.cli import segment as segment_cli
from coughkit.cli.common import add_version_argument


def _run_detect(args):
    detect_cli.detect(args.input)


def _run_segment(args):
    segment_cli.segment(args.input_file, args.output_dir, args.fs_out)


def _run_count(args):
    count_cli.count(input_file=args.input, use_mic=args.mic,
                    duration=args.duration, fs_out=args.fs_out,
                    threshold=args.threshold, verbose=args.verbose)


# (name, module, runner, help) for each subcommand.
_SUBCOMMANDS = [
    ("detect", detect_cli, _run_detect,
     "Detect whether an audio file contains a cough."),
    ("segment", segment_cli, _run_segment,
     "Segment a recording into individual cough WAV files."),
    ("count", count_cli, _run_count,
     "Count cough events in an audio file or microphone recording."),
]


def build_parser(prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_version_argument(parser)

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND",
                                       required=True)
    for name, module, runner, help_text in _SUBCOMMANDS:
        sub = subparsers.add_parser(name, help=help_text,
                                    description=module.__doc__)
        add_version_argument(sub)
        module.add_arguments(sub)
        sub.set_defaults(func=runner)

    return parser


def main(argv=None, prog=None):
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
