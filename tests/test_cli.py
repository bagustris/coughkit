import pytest

from coughkit.cli import count as count_cli
from coughkit.cli import detect as detect_cli
from coughkit.cli import main as main_cli
from coughkit.cli import segment as segment_cli


def test_top_level_version_flag(capsys):
    parser = main_cli.build_parser(prog="coughkit")

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("coughkit ")


def test_subcommand_version_flag(capsys):
    parser = main_cli.build_parser(prog="coughkit")

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["detect", "-V"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("coughkit detect ")


@pytest.mark.parametrize(
    "build_parser,prog",
    [
        (detect_cli.build_parser, "cough-detect"),
        (segment_cli.build_parser, "cough-segment"),
        (count_cli.build_parser, "cough-count"),
    ],
)
def test_legacy_commands_accept_version_flag(build_parser, prog, capsys):
    parser = build_parser(prog=prog)

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith(f"{prog} ")


def test_detect_subcommand_dispatches_to_existing_cli(monkeypatch):
    called = {}

    def fake_detect(input_file):
        called["input_file"] = input_file

    monkeypatch.setattr(detect_cli, "detect", fake_detect)

    main_cli.main(["detect", "-i", "input.wav"])

    assert called == {"input_file": "input.wav"}


def test_segment_subcommand_dispatches_to_existing_cli(monkeypatch):
    called = {}

    def fake_segment(input_file, dir_output="./", fs_out=16000):
        called["input_file"] = input_file
        called["dir_output"] = dir_output
        called["fs_out"] = fs_out

    monkeypatch.setattr(segment_cli, "segment", fake_segment)

    main_cli.main(["segment", "-i", "input.wav", "-o", "segments", "-fs", "8000"])

    assert called == {
        "input_file": "input.wav",
        "dir_output": "segments",
        "fs_out": 8000,
    }


def test_count_subcommand_dispatches_to_existing_cli(monkeypatch):
    called = {}

    def fake_count(input_file=None, use_mic=False, duration=None, fs_out=16000,
                   threshold=count_cli.DEFAULT_THRESHOLD, verbose=False):
        called["input_file"] = input_file
        called["use_mic"] = use_mic
        called["duration"] = duration
        called["fs_out"] = fs_out
        called["threshold"] = threshold
        called["verbose"] = verbose

    monkeypatch.setattr(count_cli, "count", fake_count)

    main_cli.main(["count", "-i", "input.wav", "-fs", "8000", "-t", "0.25", "-v"])

    assert called == {
        "input_file": "input.wav",
        "use_mic": False,
        "duration": None,
        "fs_out": 8000,
        "threshold": 0.25,
        "verbose": True,
    }
