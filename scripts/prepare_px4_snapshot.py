#!/usr/bin/env python3
"""Create independent local Git metadata required by the copied PX4 build.

Original commit IDs are preserved separately in docs/provenance. These local
commits describe copied working files, including their uncommitted changes.
No original repository metadata, credentials, or object alternates are used.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def initialize_snapshot(directory: Path, version_tag: str | None = None) -> None:
    """Initialize a local snapshot repository if metadata does not exist.

    Args:
        directory: Copied source directory to initialize.
        version_tag: Optional synthetic version tag for the top-level PX4 tree.
    """
    if (directory / ".git").exists():
        print(f"Keep existing Git metadata: {directory}", flush=True)
        return
    commands = [
        ["git", "init", "-q", "-b", "codex/wind-field-snapshot"],
        ["git", "add", "--all", "."],
        [
            "git", "-c", "user.name=Wind field source snapshot",
            "-c", "user.email=snapshot@localhost", "-c", "commit.gpgsign=false",
            "-c", "core.hooksPath=/dev/null", "commit", "-q", "--allow-empty",
            "-m", "Snapshot copied working files; original provenance is recorded separately",
        ],
    ]
    if version_tag:
        commands.append(["git", "tag", version_tag])
    for command in commands:
        subprocess.run(command, cwd=directory, check=True)
    print(f"Initialized local snapshot: {directory}", flush=True)


def main() -> int:
    """Initialize checked-out submodules from deepest to shallowest, then PX4."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    project = arguments.project_root.resolve()
    px4 = project / "third_party/PX4-Autopilot"
    lines = (project / "docs/provenance/px4-submodules.txt").read_text().splitlines()
    submodules = [line.split()[1] for line in lines if line and not line.startswith("-")]
    for relative in sorted(submodules, key=lambda value: value.count("/"), reverse=True):
        directory = px4 / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Copied submodule missing: {directory}")
        initialize_snapshot(directory)
    initialize_snapshot(px4, version_tag="v1.17.0-0.1.0-dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
