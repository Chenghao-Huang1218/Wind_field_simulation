#!/usr/bin/env python3
"""Generate only the separately versioned 3000-pair nominal pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training.dataset_v1 import generate_dataset  # noqa: E402


def main() -> None:
    """Parse explicit input/output paths and print the resulting manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate_dataset(args.config, args.output_dir))


if __name__ == "__main__":
    main()
