from __future__ import annotations

import argparse
from pathlib import Path

from quadratic_regression import (
    FitConfig,
    LeastSquaresQuadraticFittingStrategy,
    QuadraticRegressionService,
    plot_regression,
    read_points,
    write_snapshots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming quadratic regression (Python).")
    parser.add_argument("--input", type=Path, default=Path("input.txt"), help="Input TSV path")
    parser.add_argument("--output", type=Path, default=Path("output.txt"), help="Output TSV path")
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--max-failed-predictions", type=int, default=4)
    parser.add_argument(
        "--validation-mode",
        choices=["avg", "envelope", "inner"],
        default="avg",
        help="avg: distancia media; envelope: precio dentro de la parábola; inner: precio fuera",
    )
    parser.add_argument("--significant-digits", type=int, default=6)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("regression_plot.png"),
        help="Ruta PNG de la gráfica generada con matplotlib",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = FitConfig(
        tolerance=args.tolerance,
        max_failed_predictions=args.max_failed_predictions,
        validation_mode=args.validation_mode,
        significant_digits=args.significant_digits,
        min_points=args.min_points,
    )
    service = QuadraticRegressionService(config=config, strategy=LeastSquaresQuadraticFittingStrategy())
    points = read_points(args.input)
    snapshots = service.run(points)
    write_snapshots(args.output, snapshots)
    plot_regression(args.plot_output, points, snapshots)

    print(f"Snapshots generated: {len(snapshots)}")
    print(f"Output file: {args.output}")
    print(f"Plot file: {args.plot_output}")


if __name__ == "__main__":
    main()
