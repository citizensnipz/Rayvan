from __future__ import annotations

import math
from pathlib import Path

import pytest

from rayvan_emc.projections import (
    fit_projection,
    perplexity_projection_payload,
    projection_payload,
)


def test_unconstrained_linear_negative_regression_is_impossible() -> None:
    points = [(1_000.0, 4.0), (2_000.0, 3.0), (3_000.0, 2.0), (4_000.0, 1.0)]
    fit = fit_projection(points, 1_000_000.0, metric="validation_loss")
    assert fit is not None
    assert fit.model_type not in {"linear", "logarithmic"}
    assert math.isfinite(fit.predicted_value)
    assert fit.predicted_value >= 0


def test_all_loss_and_perplexity_targets_respect_their_bounds() -> None:
    loss_projection = projection_payload(
        [(10.0, 5.0), (20.0, 3.0), (40.0, 2.0), (80.0, 1.0)],
        (160.0, 10_000.0, 1_000_000_000.0),
        metric="validation_loss",
    )
    perplexity_projection = perplexity_projection_payload(loss_projection)
    assert all(fit["predicted_value"] >= 0 for fit in loss_projection["fits"])
    assert all(
        math.isfinite(fit["predicted_value"]) and fit["predicted_value"] > 0
        for fit in perplexity_projection["fits"]
    )


def test_perplexity_is_exactly_exp_of_corresponding_projected_loss() -> None:
    loss_projection = projection_payload(
        [(1.0, 3.0), (2.0, 2.5), (4.0, 2.0), (8.0, 1.7), (16.0, 1.5)],
        (32.0, 64.0),
        metric="validation_loss",
    )
    perplexity_projection = perplexity_projection_payload(loss_projection)
    for loss_fit, perplexity_fit in zip(
        loss_projection["fits"], perplexity_projection["fits"], strict=True
    ):
        assert perplexity_fit["source_metric"] == "validation_loss"
        assert perplexity_fit["source_predicted_loss"] == loss_fit["predicted_value"]
        assert perplexity_fit["predicted_value"] == pytest.approx(
            math.exp(loss_fit["predicted_value"])
        )


def test_four_checkpoint_fit_can_never_claim_high_confidence() -> None:
    fit = fit_projection(
        [(x, 5.0 * x**-0.3) for x in (10.0, 20.0, 40.0, 80.0)],
        120.0,
        metric="validation_loss",
    )
    assert fit is not None
    assert fit.r_squared > 0.99
    assert fit.confidence == "low"
    assert "Only 4 checkpoints" in (fit.warning or "")


def test_far_extrapolation_is_low_confidence_and_warned() -> None:
    fit = fit_projection(
        [(x, 1.0 + 4.0 * x**-0.25) for x in (10, 20, 40, 80, 160, 320)],
        32_000,
        metric="validation_loss",
    )
    assert fit is not None
    assert fit.confidence == "low"
    assert "beyond measured data" in (fit.warning or "")


def test_weak_or_non_decreasing_data_uses_safe_warned_fallback() -> None:
    fit = fit_projection(
        [(1.0, 1.0), (2.0, 1.3), (3.0, 1.1), (4.0, 1.4)],
        1_000.0,
        metric="validation_loss",
    )
    assert fit is not None
    assert fit.model_type == "constant_asymptote"
    assert fit.predicted_value == pytest.approx(1.4)
    assert fit.confidence == "low"
    assert "No supported decreasing trend" in (fit.warning or "")


def test_console_labels_measured_and_projected_segments_explicitly() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "research" / "LiveExperiment.tsx"
    ).read_text(encoding="utf-8")
    assert "Validation loss — measured" in source
    assert "Bounded loss" in source
    assert "PPL = exp(projected loss)" in source
    assert "dashed: true" in source

