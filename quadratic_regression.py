"""Quadratic regression engine for streaming points.

This module intentionally keeps a clean, Python-first architecture with:
- OOP via dataclasses and service objects.
- AOP-style cross-cutting concerns using decorators.
- Strategy pattern for interchangeable fitting implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Literal, Protocol


ValidationMode = Literal["avg", "envelope", "inner"]


# ---------- AOP utility ----------
def timed(metric_name: str) -> Callable:
    """AOP-like decorator to capture execution time in milliseconds."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (perf_counter() - start) * 1000
            wrapper.last_metric = (metric_name, elapsed_ms)
            return result

        wrapper.last_metric = (metric_name, 0.0)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# ---------- Domain ----------
@dataclass(frozen=True)
class DataPoint:
    x: float
    y: float


@dataclass(frozen=True)
class FitConfig:
    tolerance: float
    max_failed_predictions: int
    validation_mode: ValidationMode = "avg"
    significant_digits: int = 6
    min_points: int = 5
    smoothing_alpha: float = 0.35
    prediction_horizon: int = 1


@dataclass(frozen=True)
class FitCoefficients:
    a: float
    b: float
    c: float

    def predict(self, x: float) -> float:
        return self.a * (x**2) + self.b * x + self.c

    def slope(self, x: float) -> float:
        return 2 * self.a * x + self.b

    def side_of_parabola(self, x: float) -> int:
        vertex_x = -self.b / (2 * self.a) if self.a != 0 else x
        if x < vertex_x:
            return -1
        if x > vertex_x:
            return 1
        return 0


@dataclass(frozen=True)
class FitSnapshot:
    x_start: float
    x_end: float
    a: float
    b: float
    c: float
    y_calc: float
    y_pred: float
    slope: float
    parabola_side: int


@dataclass
class RunningMoments:
    """Sufficient statistics for quadratic least squares.

    Maintaining these values incrementally makes fitting O(1) with respect
    to the current window size.
    """

    n: float = 0.0
    sx: float = 0.0
    sx2: float = 0.0
    sx3: float = 0.0
    sx4: float = 0.0
    sy: float = 0.0
    sxy: float = 0.0
    sx2y: float = 0.0

    def add(self, point: DataPoint) -> None:
        x = point.x
        y = point.y
        x2 = x * x

        self.n += 1.0
        self.sx += x
        self.sx2 += x2
        self.sx3 += x2 * x
        self.sx4 += x2 * x2
        self.sy += y
        self.sxy += x * y
        self.sx2y += x2 * y

    def clear(self) -> None:
        self.n = 0.0
        self.sx = 0.0
        self.sx2 = 0.0
        self.sx3 = 0.0
        self.sx4 = 0.0
        self.sy = 0.0
        self.sxy = 0.0
        self.sx2y = 0.0


# ---------- Strategy ----------
class QuadraticFittingStrategy(Protocol):
    def fit(self, points: list[DataPoint]) -> FitCoefficients:
        ...


class LeastSquaresQuadraticFittingStrategy:
    """Least-squares quadratic fitting without external dependencies."""

    @timed("least_squares_fit")
    def fit(self, points: list[DataPoint]) -> FitCoefficients:
        moments = RunningMoments()
        for point in points:
            moments.add(point)
        return self.fit_from_moments(moments)

    def fit_from_moments(self, moments: RunningMoments) -> FitCoefficients:
        matrix = [
            [moments.sx4, moments.sx3, moments.sx2, moments.sx2y],
            [moments.sx3, moments.sx2, moments.sx, moments.sxy],
            [moments.sx2, moments.sx, moments.n, moments.sy],
        ]

        try:
            a, b, c = self._solve_3x3([row[:] for row in matrix])
        except ValueError:
            # Small regularization to keep the stream stable on near-singular windows.
            lam = 1e-8
            matrix[0][0] += lam
            matrix[1][1] += lam
            matrix[2][2] += lam
            a, b, c = self._solve_3x3([row[:] for row in matrix])
        return FitCoefficients(a=a, b=b, c=c)

    @staticmethod
    def _solve_3x3(aug: list[list[float]]) -> tuple[float, float, float]:
        # Gaussian elimination (KISS, deterministic, no hidden magic).
        for col in range(3):
            pivot = max(range(col, 3), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]

            pivot_val = aug[col][col]
            if abs(pivot_val) < 1e-12:
                raise ValueError("Cannot fit quadratic: singular matrix")

            for j in range(col, 4):
                aug[col][j] /= pivot_val

            for row in range(3):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(col, 4):
                    aug[row][j] -= factor * aug[col][j]

        return aug[0][3], aug[1][3], aug[2][3]


# ---------- Application service ----------
class QuadraticRegressionService:
    def __init__(self, config: FitConfig, strategy: QuadraticFittingStrategy):
        self._config = config
        self._strategy = strategy
        self._window: list[DataPoint] = []
        self._moments = RunningMoments()
        self._snapshots: list[FitSnapshot] = []
        self._smoothed_coeffs: FitCoefficients | None = None

    @property
    def snapshots(self) -> list[FitSnapshot]:
        return list(self._snapshots)

    @timed("process_point")
    def process_point(self, point: DataPoint) -> FitSnapshot | None:
        self._window.append(point)
        self._moments.add(point)
        if len(self._window) < self._config.min_points:
            return None

        if isinstance(self._strategy, LeastSquaresQuadraticFittingStrategy):
            coeffs = self._strategy.fit_from_moments(self._moments)
        else:
            coeffs = self._strategy.fit(self._window)

        coeffs = self._smooth_coefficients(coeffs)
        failed = self._count_failed_predictions(
            coeffs,
            self._window,
            failure_threshold=self._config.max_failed_predictions,
        )

        if failed > self._config.max_failed_predictions:
            self._window = [point]
            self._moments.clear()
            self._moments.add(point)
            self._smoothed_coeffs = None
            return None

        snapshot = self._to_snapshot(coeffs)
        self._snapshots.append(snapshot)
        return snapshot

    def run(self, points: Iterable[DataPoint]) -> list[FitSnapshot]:
        for point in points:
            self.process_point(point)
        return self.snapshots

    def _count_failed_predictions(
        self,
        coeffs: FitCoefficients,
        points: list[DataPoint],
        failure_threshold: int,
    ) -> int:
        tolerance = self._config.tolerance
        mode = self._config.validation_mode
        failed = 0

        for point in points:
            prediction = coeffs.predict(point.x)
            is_inside = self._point_inside_parabola(coeffs.a, point.y, prediction, tolerance)

            if mode == "avg":
                is_fail = abs(prediction - point.y) > tolerance
            elif mode == "envelope":
                is_fail = not is_inside
            elif mode == "inner":
                is_fail = is_inside
            else:
                raise ValueError(f"Unsupported validation mode: {mode}")

            if is_fail:
                failed += 1
                if failed > failure_threshold:
                    return failed

        return failed

    @staticmethod
    def _point_inside_parabola(a: float, y: float, y_curve: float, tolerance: float) -> bool:
        if a >= 0:
            return y >= (y_curve - tolerance)
        return y <= (y_curve + tolerance)

    def _to_snapshot(self, coeffs: FitCoefficients) -> FitSnapshot:
        x_start = self._window[0].x
        x_end = self._window[-1].x
        y_calc = coeffs.predict(x_end)
        x_future = x_end + self._prediction_step() * max(1, self._config.prediction_horizon)
        y_pred = coeffs.predict(x_future)
        return FitSnapshot(
            x_start=round(x_start, self._config.significant_digits),
            x_end=round(x_end, self._config.significant_digits),
            a=round(coeffs.a, self._config.significant_digits),
            b=round(coeffs.b, self._config.significant_digits),
            c=round(coeffs.c, self._config.significant_digits),
            y_calc=round(y_calc, self._config.significant_digits),
            y_pred=round(y_pred, self._config.significant_digits),
            slope=round(coeffs.slope(x_end), self._config.significant_digits),
            parabola_side=coeffs.side_of_parabola(x_end),
        )

    def _smooth_coefficients(self, coeffs: FitCoefficients) -> FitCoefficients:
        alpha = min(1.0, max(0.0, self._config.smoothing_alpha))
        previous = self._smoothed_coeffs
        if previous is None:
            self._smoothed_coeffs = coeffs
            return coeffs

        smoothed = FitCoefficients(
            a=alpha * coeffs.a + (1.0 - alpha) * previous.a,
            b=alpha * coeffs.b + (1.0 - alpha) * previous.b,
            c=alpha * coeffs.c + (1.0 - alpha) * previous.c,
        )
        self._smoothed_coeffs = smoothed
        return smoothed

    def _prediction_step(self) -> float:
        if len(self._window) >= 2:
            delta = self._window[-1].x - self._window[-2].x
            return delta if delta != 0 else 1.0
        return 1.0


# ---------- I/O ----------
def read_points(input_path: Path) -> list[DataPoint]:
    points: list[DataPoint] = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        x_raw, y_raw = raw.split()
        points.append(DataPoint(x=float(x_raw), y=float(y_raw)))
    return points


def write_snapshots(output_path: Path, snapshots: Iterable[FitSnapshot]) -> None:
    rows = [
        "\t".join(
            map(
                str,
                [
                    s.x_start,
                    s.x_end,
                    s.a,
                    s.b,
                    s.c,
                    s.y_calc,
                    s.y_pred,
                    s.slope,
                    s.parabola_side,
                ],
            )
        )
        for s in snapshots
    ]
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def plot_regression(
    plot_path: Path,
    points: list[DataPoint],
    snapshots: list[FitSnapshot],
) -> None:
    """Genera una gráfica robusta usando matplotlib (sin Highcharts)."""
    import matplotlib.pyplot as plt

    if not points:
        return

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    ax.scatter(xs, ys, label="Puntos observados", color="#1f77b4", alpha=0.8)

    if snapshots:
        final = snapshots[-1]
        curve_x = sorted(xs)
        curve_y = [final.a * (x**2) + final.b * x + final.c for x in curve_x]
        ax.plot(curve_x, curve_y, label="Ajuste cuadrático", color="#d62728", linewidth=2)

    ax.set_title("Regresión cuadrática")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
