"""Command line entry point: configuration check and plain-text scan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from tribunal.errors import ConfigurationError
from tribunal.runtime_config import load_runtime_config
from tribunal.settings import load_settings
from tribunal.textpolicy import TextPolicy

SCAN_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".ts", ".tsx", ".html", ".css", ".sh", ".ini", ".mako"}
SKIP_DIRS = {"__pycache__", ".venv", "node_modules"}


def _check_config() -> int:
    try:
        snapshot = load_runtime_config(load_settings().runtime_config)
    except ConfigurationError as exc:
        for problem in exc.problems:
            print(f"- {problem}", file=sys.stderr)
        return 2
    print(f"OK {snapshot.snapshot_id}")
    return 0


def _db_upgrade() -> int:
    from tribunal.db.migrate import upgrade

    upgrade(load_settings().database_url)
    print("database at head")
    return 0


def _worker() -> int:
    from tribunal.context import build_context
    from tribunal.jobs.worker import run_forever

    try:
        ctx = build_context(load_settings())
    except ConfigurationError as exc:
        for problem in exc.problems:
            print(f"- {problem}", file=sys.stderr)
        return 2
    run_forever(ctx)
    return 0


def _files(paths: list[str], errors: list[str]):
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            errors.append(f"{raw}: path does not exist")
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for p in candidates:
            if p.is_file() and p.suffix in SCAN_SUFFIXES and not SKIP_DIRS.intersection(p.parts):
                yield p


def _scan_text(config: str, paths: list[str]) -> int:
    # Read the preserve list without interpolation so no environment is needed.
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    policy = TextPolicy(raw["text_policy"]["preserve_codepoints"])
    found = 0
    errors: list[str] = []
    for path in _files(paths, errors):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            errors.append(f"{path}: not scanned ({exc})")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for finding in policy.findings(line):
                found += 1
                print(f"{path}:{lineno}:{finding.index + 1} {finding.codepoints}")
    for message in errors:
        print(message, file=sys.stderr)
    # 0 clean, 1 findings, 2 a path missing or a file unreadable (a partial scan is never a pass)
    return 2 if errors else (1 if found else 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tribunal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config", help="validate settings and runtime configuration")
    sub.add_parser("db-upgrade", help="apply database migrations")
    sub.add_parser("worker", help="run the background job worker")
    scan = sub.add_parser("scan-text", help="report emoji in text files")
    scan.add_argument("--config", required=True)
    scan.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config()
    if args.command == "db-upgrade":
        return _db_upgrade()
    if args.command == "worker":
        return _worker()
    return _scan_text(args.config, args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
