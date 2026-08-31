from __future__ import annotations

import argparse
import sys
from pathlib import Path

import research.mnq_opportunity_target_matrix as matrix

CONFIGS = {
    "h6_vol05": (6, 0.5),
    "h6_vol10": (6, 1.0),
    "h12_vol05": (12, 0.5),
    "h12_vol10": (12, 1.0),
    "h24_vol05": (24, 0.5),
    "h24_vol10": (24, 1.0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--deep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    horizon, multiplier = CONFIGS[args.config_key]
    matrix.HORIZONS = (horizon,)
    matrix.VOL_MULTIPLIERS = (multiplier,)
    sys.argv = [
        "mnq_opportunity_target_matrix",
        "--deep-root", str(args.deep_root),
        "--output", str(args.output),
    ]
    return matrix.main()


if __name__ == "__main__":
    raise SystemExit(main())
