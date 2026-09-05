from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


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


@dataclass(frozen=True)
class _Candidate:
    model_type: str
    parameters: dict[str, float]
    predictions: list[float]
    parameter_count: int


def fit_projection(
    points: Iterable[tuple[float, float]], target: float, *, metric: str
) -> ProjectionFit | None:
    """Fit a non-negative, non-increasing loss learning curve.

    Candidate families are power-law decay to a non-negative asymptote,
    exponential decay to a non-negative asymptote, and a bounded constant
    fallback. Unconstrained linear/logarithmic extrapolation is deliberately
    excluded because it can predict impossible negative cross-entropy.
    """

    clean = sorted(
        {
            float(x): float(y)
            for x, y in points
            if x > 0
            and y >= 0
            and math.isfinite(x)
            and math.isfinite(y)
        }.items()
    )
    if len(clean) < 4 or not math.isfinite(target) or target <= clean[-1][0]:
        return None
    xs = tuple(x for x, _ in clean)
    ys = tuple(y for _, y in clean)
    candidates = [_constant_candidate(xs, ys)]
    if min(ys) > 0:
        for family in ("power_law", "exponential"):
            candidate = _best_decay_candidate(family, xs, ys)
            if candidate is not None:
                candidates.append(candidate)

    selected = min(candidates, key=lambda row: _bic(ys, row))
    r_squared = _r_squared(ys, selected.predictions)
    predicted = max(0.0, _predict(selected.model_type, selected.parameters, target))
    if not math.isfinite(predicted):
        predicted = max(0.0, min(ys))
    extrapolation = target / max(xs)
    confidence, warning = _confidence(
        measured_points=len(clean),
        r_squared=r_squared,
        extrapolation=extrapolation,
        model_type=selected.model_type,
    )
    return ProjectionFit(
        metric=metric,
        model_type=selected.model_type,
        parameters=selected.parameters,
        r_squared=r_squared,
        measured_points=len(clean),
        measured_end=max(xs),
        prediction_target=target,
        predicted_value=predicted,
        confidence=confidence,
        warning=warning,
    )


def projection_payload(
    points: Iterable[tuple[float, float]],
    targets: Iterable[float],
    *,
    metric: str,
) -> dict[str, object]:
    fits = [fit_projection(points, target, metric=metric) for target in targets]
    return {
        "schema_version": 2,
        "metric": metric,
        "constraints": {
            "minimum": 0.0,
            "monotonic_extrapolation": "non_increasing",
            "candidate_models": [
                "power_law_decay",
                "power_law_asymptote",
                "exponential_decay",
                "exponential_asymptote",
                "constant_asymptote",
            ],
        },
        "fits": [asdict(fit) for fit in fits if fit is not None],
    }


def perplexity_projection_payload(
    loss_projection: Mapping[str, Any],
) -> dict[str, object]:
    """Derive projected perplexity exclusively as exp(projected loss)."""

    fits: list[dict[str, Any]] = []
    for raw in loss_projection.get("fits", []):
        fit = dict(raw)
        projected_loss = max(0.0, float(fit["predicted_value"]))
        source_model = str(fit["model_type"])
        fit.update(
            metric="perplexity",
            model_type=f"exp_of_{source_model}",
            predicted_value=_safe_exp(projected_loss),
            source_metric="validation_loss",
            source_model_type=source_model,
            source_predicted_loss=projected_loss,
        )
        fits.append(fit)
    return {
        "schema_version": 2,
        "metric": "perplexity",
        "derivation": "exp(projected_validation_loss)",
        "fits": fits,
    }


def _best_decay_candidate(
    family: str,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
) -> _Candidate | None:
    minimum = min(ys)
    floors = [0.0] + [minimum * 0.98 * index / 24 for index in range(1, 25)]
    candidates: list[_Candidate] = []
    transformed_x = (
        [math.log(x) for x in xs] if family == "power_law" else list(xs)
    )
    for floor in floors:
        adjusted = [y - floor for y in ys]
        if any(value <= 0 for value in adjusted):
            continue
        intercept, slope = _regression(
            transformed_x, [math.log(value) for value in adjusted]
        )
        if slope >= -1e-12:
            continue
        scale = math.exp(intercept)
        if not math.isfinite(scale) or scale <= 0:
            continue
        model_type = (
            f"{family}_decay" if floor == 0 else f"{family}_asymptote"
        )
        parameters = {
            "scale": scale,
            "exponent" if family == "power_law" else "rate": slope,
            "asymptote": floor,
        }
        predictions = [_predict(model_type, parameters, x) for x in xs]
        if all(math.isfinite(value) and value >= 0 for value in predictions):
            candidates.append(
                _Candidate(
                    model_type,
                    parameters,
                    predictions,
                    2 if floor == 0 else 3,
                )
            )
    return min(candidates, key=lambda row: _bic(ys, row)) if candidates else None


def _constant_candidate(
    xs: tuple[float, ...], ys: tuple[float, ...]
) -> _Candidate:
    del xs
    asymptote = max(0.0, ys[-1])
    return _Candidate(
        "constant_asymptote",
        {"asymptote": asymptote},
        [asymptote for _ in ys],
        1,
    )


def _regression(
    xs: list[float] | tuple[float, ...],
    ys: list[float] | tuple[float, ...],
) -> tuple[float, float]:
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / max(variance, 1e-18)
    return mean_y - slope * mean_x, slope


def _predict(model_type: str, parameters: dict[str, float], x: float) -> float:
    if model_type == "constant_asymptote":
        return parameters["asymptote"]
    decay = (
        parameters["scale"] * x ** parameters["exponent"]
        if model_type.startswith("power_law")
        else parameters["scale"] * math.exp(parameters["rate"] * x)
    )
    return max(0.0, parameters.get("asymptote", 0.0) + decay)


def _bic(actual: tuple[float, ...], candidate: _Candidate) -> float:
    residual = sum(
        (value - fit) ** 2
        for value, fit in zip(actual, candidate.predictions)
    )
    count = len(actual)
    return count * math.log(max(residual / count, 1e-18)) + (
        candidate.parameter_count * math.log(count)
    )


def _confidence(
    *, measured_points: int, r_squared: float, extrapolation: float, model_type: str
) -> tuple[str, str | None]:
    if (
        measured_points >= 10
        and r_squared >= 0.95
        and extrapolation <= 2
        and model_type != "constant_asymptote"
    ):
        confidence = "high"
    elif (
        measured_points >= 6
        and r_squared >= 0.85
        and extrapolation <= 5
        and model_type != "constant_asymptote"
    ):
        confidence = "medium"
    else:
        confidence = "low"

    warnings: list[str] = []
    if measured_points < 6:
        warnings.append(
            f"Only {measured_points} checkpoints are available; confidence is capped at low."
        )
    if r_squared < 0.8:
        warnings.append("Observed checkpoints weakly support the bounded decay curves.")
    if model_type == "constant_asymptote":
        warnings.append(
            "No supported decreasing trend was found; projection holds the latest loss constant."
        )
    if extrapolation > 10:
        warnings.append(
            f"Target is {extrapolation:.1f}× beyond measured data; treat it as highly speculative."
        )
        confidence = "low"
    elif extrapolation > 5:
        warnings.append(f"Target is {extrapolation:.1f}× beyond measured data.")
        confidence = "low"
    return confidence, " ".join(warnings) or None


def _r_squared(actual: tuple[float, ...], predicted: list[float]) -> float:
    mean = sum(actual) / len(actual)
    total = sum((value - mean) ** 2 for value in actual)
    residual = sum(
        (value - fit) ** 2 for value, fit in zip(actual, predicted)
    )
    return 1.0 - residual / max(total, 1e-18)


def _safe_exp(value: float) -> float:
    return max(
        sys.float_info.min,
        math.exp(min(value, math.log(sys.float_info.max))),
    )
