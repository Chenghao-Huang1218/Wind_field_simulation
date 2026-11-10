#!/usr/bin/env python3
"""Export the offline X4-950 initial-pose reference without launching SITL."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training.xml_parameters import (  # noqa: E402
    DEFAULT_ASSUMPTIONS,
    DEFAULT_SCENE,
    PROJECT_ROOT,
    export_reference_bundle,
)


def main() -> int:
    """Write the three reference artifacts and print provenance/counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--config", type=Path, default=DEFAULT_ASSUMPTIONS)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/pinn/x4_950_wind_v1")
    args = parser.parse_args()
    reference = export_reference_bundle(args.xml_path, args.config, args.output_dir)
    print(f"mass={reference['mass_total_kg']:.7f} kg; "
          f"proxies={len(reference['proxies'])}")
    print(f"reference_sha256={reference['reference_sha256']}")
    print(f"Offline reference artifacts: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
