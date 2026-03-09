"""Run repository formatting hooks for InstinctMJ."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run InstinctMJ formatting hooks via pre-commit.",
  )
  parser.add_argument(
    "paths",
    nargs="*",
    help="Optional files or directories relative to the InstinctMJ repository root.",
  )
  return parser


def main() -> int:
  parser = _build_arg_parser()
  args = parser.parse_args()

  pre_commit = shutil.which("pre-commit")
  if pre_commit is None:
    print(
      "pre-commit is not installed. Install it with `pip install pre-commit` and try again.",
      file=sys.stderr,
    )
    return 1

  command = [pre_commit, "run"]
  if args.paths:
    command.extend(["--files", *args.paths])
  else:
    command.append("--all-files")

  completed = subprocess.run(command, cwd=_REPO_ROOT, check=False)
  return int(completed.returncode)


if __name__ == "__main__":
  raise SystemExit(main())
