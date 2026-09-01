from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .telemetry import compare_developmental_runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare developmental EMC telemetry runs"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="repeat NAME=CHECKPOINT_DIRECTORY",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    runs: dict[str, Path] = {}
    for value in args.run:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--run must use NAME=CHECKPOINT_DIRECTORY")
        runs[name] = Path(path)
    compare_developmental_runs(runs, args.output_dir)
    print(f"wrote developmental comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
