from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .research_config import ExperimentConfig, research_schema
from .research_runner import estimate_experiment, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rayvan EMC Research Console runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a saved ExperimentConfig")
    run.add_argument("config")
    run.add_argument("--runs-dir", default=os.environ.get("RAYVAN_RUNS_DIR", "runs"))
    run.add_argument("--run-id")
    validate = subparsers.add_parser("validate", help="validate and normalize a config")
    validate.add_argument("config")
    estimate = subparsers.add_parser("estimate", help="estimate parameters and compute")
    estimate.add_argument("config")
    subparsers.add_parser("schema", help="print UI/backend capability metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "schema":
        print(json.dumps(research_schema(), allow_nan=False))
        return
    path = Path(args.config)
    config = ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if args.command == "validate":
        print(json.dumps(config.to_dict(), indent=2, allow_nan=False))
        return
    if args.command == "estimate":
        print(json.dumps(estimate_experiment(config), allow_nan=False))
        return
    run_experiment(config, runs_directory=args.runs_dir, run_id=args.run_id)


if __name__ == "__main__":
    main()
