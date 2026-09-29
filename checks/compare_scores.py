"""Compare same-protocol base and adapter metrics after actual runs."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    trained = json.loads(args.adapter.read_text(encoding="utf-8"))
    for metric in ("score", "acc_main", "acc_p1", "acc_p2"):
        before, after = base[metric], trained[metric]
        print(f"{metric:10s} base={before:.4f} adapter={after:.4f} delta={after-before:+.4f}")


if __name__ == "__main__":
    main()
