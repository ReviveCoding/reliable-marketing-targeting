from __future__ import annotations

from evaluate_from_artifacts import build_parser, run


if __name__ == "__main__":
    run(build_parser().parse_args())
