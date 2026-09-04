from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProjectionFit:
    metric: str
    model_type: str
    parameters: dict[str, float]
    r_squared: float
    measured_points: int
    measured_end: float
    prediction_target: float
    predicted_value: float
    confidence: str
    warning: str | None = None


def fit_projection(
    points: Iterable[tuple[float, float]], target: float, *, metric: str
) -> ProjectionFit | None:
    clean = [(float(x), float(y)) for x, y in points if x > 0 and math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 4 or target <= clean[-1][0]:
        return None
    xs, ys = zip(*clean)
    candidates: list[tuple[str, dict[str, float], list[float]]] = []
    candidates.append(_linear_candidate(xs, ys))
    candidates.append(_transformed_candidate("logarithmic", [math.log(x) for x in xs], ys))
    if all(y > 0 for y in ys):
        candidates.append(_positive_candidate("power_law", [math.log(x) for x in xs], ys))
        candidates.append(_positive_candidate("exponential", list(xs), ys))
    scored = []
    for model_type, parameters, predictions in candidates:
        r2 = _r_squared(ys, predictions)
        if math.isfinite(r2):
            scored.append((r2, model_type, parameters))
    if not scored:
        return None
    r2, model_type, parameters = max(scored, key=lambda row: row[0])
    predicted = _predict(model_type, parameters, target)
    extrapolation = target / max(xs)
    confidence = "high" if r2 >= 0.95 and extrapolation <= 2 else "medium" if r2 >= 0.8 and extrapolation <= 10 else "low"
    warning = None
    if extrapolation > 10:
        confidence = "low"
        warning = f"Target is {extrapolation:.1f}× beyond measured data."
    elif r2 < 0.8:
        warning = "Observed checkpoints do not strongly support any candidate curve."
    return ProjectionFit(
        metric=metric,
        model_type=model_type,
        parameters=parameters,
        r_squared=r2,
        measured_points=len(clean),
        measured_end=max(xs),
        prediction_target=target,
        predicted_value=predicted,
        confidence=confidence,
        warning=warning,
    )


def projection_payload(points: Iterable[tuple[float, float]], targets: Iterable[float], *, metric: str) -> dict[str, object]:
    fits = [fit_projection(points, target, metric=metric) for target in targets]
    return {"metric": metric, "fits": [asdict(fit) for fit in fits if fit is not None]}


def _regression(xs: list[float] | tuple[float, ...], ys: list[float] | tuple[float, ...]) -> tuple[float, float]:
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / max(variance, 1e-18)
    return mean_y - slope * mean_x, slope


def _linear_candidate(xs: tuple[float, ...], ys: tuple[float, ...]):
    a, b = _regression(xs, ys)
    return "linear", {"intercept": a, "slope": b}, [a + b * x for x in xs]


def _transformed_candidate(name: str, transformed: list[float], ys: tuple[float, ...]):
    a, b = _regression(transformed, ys)
    return name, {"intercept": a, "slope": b}, [a + b * x for x in transformed]


def _positive_candidate(name: str, transformed: list[float], ys: tuple[float, ...]):
    a, b = _regression(transformed, [math.log(y) for y in ys])
    return name, {"scale": math.exp(a), "exponent" if name == "power_law" else "rate": b}, [math.exp(a + b * x) for x in transformed]


def _predict(model_type: str, parameters: dict[str, float], x: float) -> float:
    if model_type == "linear":
        return parameters["intercept"] + parameters["slope"] * x
    if model_type == "logarithmic":
        return parameters["intercept"] + parameters["slope"] * math.log(x)
    if model_type == "power_law":
        return parameters["scale"] * x ** parameters["exponent"]
    return parameters["scale"] * math.exp(parameters["rate"] * x)


def _r_squared(actual: tuple[float, ...], predicted: list[float]) -> float:
    mean = sum(actual) / len(actual)
    total = sum((value - mean) ** 2 for value in actual)
    residual = sum((value - fit) ** 2 for value, fit in zip(actual, predicted))
    return 1.0 - residual / max(total, 1e-18)
