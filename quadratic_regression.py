"""Quadratic regression engine for streaming points.

This module intentionally keeps a clean, Python-first architecture with:
- OOP via dataclasses and service objects.
- AOP-style cross-cutting concerns using decorators.
- Strategy pattern for interchangeable fitting implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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
    prediction_horizon: int = 1
    markov_lookback: int = 30
    hurst_min_points: int = 20
    predictive_strength: float = 0.65


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
    """Sufficient statistics for quadratic least squares."""

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


@dataclass
class PredictiveState:
    """Online O(1) predictor state using Markov transitions + persistence proxy.

    Notes:
    - Markov transitions are tracked with exponential forgetting (no window scans).
    - Persistence (Hurst-like) is approximated from directional persistence and
      trend/volatility balance, also in O(1).
    """

    prev_x: float | None = None
    prev_y: float | None = None
    prev_return: float = 0.0
    prev_state: int | None = None
    sample_count: int = 0

    # transition counts (decayed)
    uu: float = 1e-6
    ud: float = 1e-6
    du: float = 1e-6
    dd: float = 1e-6

    # running signal stats
    ret_ema: float = 0.0
    ret_var_ema: float = 0.0
    accel_ema: float = 0.0
    abs_ret_ema: float = 0.0

    def reset(self) -> None:
        self.prev_x = None
        self.prev_y = None
        self.prev_return = 0.0
        self.prev_state = None
        self.sample_count = 0
        self.uu = self.ud = self.du = self.dd = 1e-6
        self.ret_ema = 0.0
        self.ret_var_ema = 0.0
        self.accel_ema = 0.0
        self.abs_ret_ema = 0.0

    def update(self, point: DataPoint, lookback: int) -> None:
        if self.prev_y is None or self.prev_x is None:
            self.prev_x = point.x
            self.prev_y = point.y
            return

        dx = point.x - self.prev_x
        dx = dx if abs(dx) > 1e-12 else 1.0
        ret = (point.y - self.prev_y) / dx
        state = 1 if ret >= 0 else -1

        lb = max(4, lookback)
        alpha = 2.0 / (lb + 1.0)
        decay = 1.0 - alpha

        if self.prev_state is not None:
            self.uu *= decay
            self.ud *= decay
            self.du *= decay
            self.dd *= decay
            if self.prev_state == 1 and state == 1:
                self.uu += 1.0
            elif self.prev_state == 1 and state == -1:
                self.ud += 1.0
            elif self.prev_state == -1 and state == 1:
                self.du += 1.0
            else:
                self.dd += 1.0

        delta = ret - self.ret_ema
        self.ret_ema += alpha * delta
        self.ret_var_ema = (1.0 - alpha) * self.ret_var_ema + alpha * (delta * delta)

        accel = ret - self.prev_return
        self.accel_ema = (1.0 - alpha) * self.accel_ema + alpha * accel
        self.abs_ret_ema = (1.0 - alpha) * self.abs_ret_ema + alpha * abs(ret)

        self.prev_return = ret
        self.prev_state = state
        self.prev_x = point.x
        self.prev_y = point.y
        self.sample_count += 1

    def markov_bias(self) -> float:
        if self.prev_state == 1:
            p_up = self.uu / (self.uu + self.ud)
        elif self.prev_state == -1:
            p_up = self.du / (self.du + self.dd)
        else:
            return 0.0
        return 2.0 * p_up - 1.0

    def persistence_proxy(self, min_points: int) -> float:
        """Returns a Hurst-like persistence score in [0, 1]."""
        if self.sample_count < max(4, min_points):
            return 0.5

        p_stay_up = self.uu / (self.uu + self.ud)
        p_stay_down = self.dd / (self.dd + self.du)
        directional_persistence = 0.5 * (p_stay_up + p_stay_down)

        volatility = sqrt(max(self.ret_var_ema, 1e-12))
        trendiness = abs(self.ret_ema) / (abs(self.ret_ema) + volatility)

        score = 0.5 + (directional_persistence - 0.5) * (0.4 + 0.6 * trendiness)
        return max(0.0, min(1.0, score))

    def volatility_penalty(self) -> float:
        volatility = sqrt(max(self.ret_var_ema, 1e-12))
        baseline = self.abs_ret_ema + 1e-9
        ratio = volatility / (volatility + baseline)
        return max(0.0, min(1.0, ratio))


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
            lam = 1e-8
            matrix[0][0] += lam
            matrix[1][1] += lam
            matrix[2][2] += lam
            a, b, c = self._solve_3x3([row[:] for row in matrix])
        return FitCoefficients(a=a, b=b, c=c)

    @staticmethod
    def _solve_3x3(aug: list[list[float]]) -> tuple[float, float, float]:
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
        self._predictive = PredictiveState()

    @property
    def snapshots(self) -> list[FitSnapshot]:
        return list(self._snapshots)

    @timed("process_point")
    def process_point(self, point: DataPoint) -> FitSnapshot | None:
        self._window.append(point)
        self._moments.add(point)
        self._predictive.update(point, self._config.markov_lookback)

        if len(self._window) < self._config.min_points:
            return None

        if isinstance(self._strategy, LeastSquaresQuadraticFittingStrategy):
            coeffs = self._strategy.fit_from_moments(self._moments)
        else:
            coeffs = self._strategy.fit(self._window)

        failed = self._count_failed_predictions(
            coeffs,
            self._window,
            failure_threshold=self._config.max_failed_predictions,
        )

        if failed > self._config.max_failed_predictions:
            self._window = [point]
            self._moments.clear()
            self._moments.add(point)
            self._predictive.reset()
            self._predictive.update(point, self._config.markov_lookback)
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
        x_future = self._predictive_future_x(x_end, coeffs)
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

    def _predictive_future_x(self, x_end: float, coeffs: FitCoefficients) -> float:
        step = self._prediction_step()
        horizon = step * max(1, self._config.prediction_horizon)

        markov_bias = self._predictive.markov_bias()
        persistence = self._predictive.persistence_proxy(self._config.hurst_min_points)
        memory_regime = (persistence - 0.5) * 2.0

        slope_state = 1.0 if coeffs.slope(x_end) >= 0 else -1.0
        accel_state = 1.0 if self._predictive.accel_ema >= 0 else -1.0

        directional_bias = 0.45 * markov_bias + 0.35 * slope_state + 0.20 * accel_state
        volatility_penalty = self._predictive.volatility_penalty()
        confidence = max(0.1, 1.0 - 0.7 * volatility_penalty)

        # Curvature-aware push: if trend and curvature agree, allow slightly further projection.
        curvature_alignment = 1.0 if (coeffs.a >= 0 and slope_state > 0) or (coeffs.a < 0 and slope_state < 0) else -1.0

        boost = 1.0 + (
            self._config.predictive_strength * directional_bias * memory_regime * confidence
            + 0.10 * curvature_alignment * confidence
        )
        boost = min(2.8, max(0.25, boost))
        return x_end + horizon * boost

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
        trend_up = final.y_pred >= final.y_calc
        trend_color = "#78AFFF" if trend_up else "#FF96D2"
        trend_label = "Ajuste cuadrático (subida)" if trend_up else "Ajuste cuadrático (bajada)"
        ax.plot(curve_x, curve_y, label=trend_label, color=trend_color, linewidth=2)

    ax.set_title("Regresión cuadrática")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
