from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .architecture import N2_ARCHITECTURES
from .capability_tasks import CAPABILITIES
from .experiments.common import MODEL_PRESET_DIMENSIONS, N1_STAGES


SCHEMA_VERSION = 3
EXPERT_FAMILIES = ("gpt", "ssm", "recurrent", "delta")
ARCHITECTURES = (
    "emc",
    "sequential_module_aware_emc",
    "legacy_parallel_emc",
    "heterogeneous_serial",
    "homogeneous_serial",
    "old_emc",
    *tuple(N2_ARCHITECTURES),
)
SUITES = ("tinystories", "capability_10")


@dataclass(frozen=True)
class RoutingConfig:
    top_k: int | None = None
    cycles: int = 2
    trajectory_steps: int = 3
    router_type: str = "geometric"
    integrator_type: str = "acceptance_gate"
    routing_geometry_dim: int = 32
    competence_prototypes_per_expert: int = 1
    balance_coefficient: float = 0.0
    balance_entropy_floor: float = 0.75
    switch_cost: float = 0.05
    persistence_bonus: float = 0.1
    minimum_lease_chunks: int = 0
    loss_free_balance_enabled: bool = False
    balance_bias_lr: float = 0.01
    balance_bias_limit: float = 0.25
    balance_warmup_chunks: int = 0
    refractory_enabled: bool = True
    refractory_strength: float = 0.15
    refractory_decay: float = 0.35
    counterfactual_calibration_enabled: bool = True
    counterfactual_probe_preset: str = "decaying"
    counterfactual_probe_fixed_rate: float | None = None
    counterfactual_probe_early_rate: float = 0.08
    counterfactual_probe_stable_rate: float = 0.02
    counterfactual_probe_mature_rate: float = 0.01
    counterfactual_uncertainty_enabled: bool = False
    counterfactual_uncertainty_margin: float = 0.05
    counterfactual_max_probes_per_forward: int = 1
    counterfactual_probe_temperature: float = 0.25
    geometry_temperature: float = 0.25
    geometry_calibration_weight: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    preset: str = "quick"
    fairness_mode: str = "custom"
    latent_dim: int = 64
    context_length: int = 256
    attention_heads: int = 4
    module_hidden_dim: int = 128
    integrator_heads: int = 4
    chunk_size: int = 16
    shared_state_slots: int = 4
    n1_depth: int = 3
    tie_embeddings: bool = True


@dataclass(frozen=True)
class ResearchTrainingConfig:
    tokens: int = 50_000
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 42
    gradient_accumulation: int = 1
    precision: str = "auto"
    device: str = "cuda"
    evaluation_interval: int = 25
    evaluation_batches: int = 4
    telemetry_interval: int = 1
    diagnostic_examples_per_capability: int = 20


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = SCHEMA_VERSION
    name: str = ""
    notes: str = ""
    tags: tuple[str, ...] = ()
    projection_targets: tuple[int, ...] = (250_000, 500_000, 1_000_000, 10_000_000)
    suite: str = "tinystories"
    architecture: str = "emc"
    experts: Mapping[str, int] = field(
        default_factory=lambda: {"gpt": 1, "ssm": 1, "recurrent": 1, "delta": 1}
    )
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: ResearchTrainingConfig = field(default_factory=ResearchTrainingConfig)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment schema version: {self.schema_version}")
        if self.suite not in SUITES:
            raise ValueError(f"unknown suite: {self.suite!r}")
        if any(target <= 0 for target in self.projection_targets):
            raise ValueError("projection targets must be positive token counts")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {self.architecture!r}")
        unknown = set(self.experts) - set(EXPERT_FAMILIES)
        if unknown:
            raise ValueError(f"unknown expert families: {sorted(unknown)}")
        if any(not isinstance(count, int) or count < 0 for count in self.experts.values()):
            raise ValueError("expert counts must be non-negative integers")
        total = sum(self.experts.values())
        if self.architecture in {"emc", "sequential_module_aware_emc", "legacy_parallel_emc", "heterogeneous_serial", "old_emc"} and total == 0:
            raise ValueError("the selected architecture requires at least one expert")
        if self.architecture in {"legacy_parallel_emc", "old_emc", *N2_ARCHITECTURES}:
            if self.routing.top_k is None:
                raise ValueError(
                    "top_k is required for legacy parallel and N2 architectures"
                )
            if self.routing.top_k <= 0 or self.routing.top_k > max(total, 1):
                raise ValueError("top_k must be between one and the configured expert count")
        if self.architecture in {"emc", "sequential_module_aware_emc"} and self.routing.top_k is not None:
            raise ValueError(
                "sequential EMC does not accept top_k; use trajectory_steps or "
                "select legacy_parallel_emc"
            )
        if self.routing.cycles <= 0:
            raise ValueError("cycles must be positive")
        if self.routing.trajectory_steps <= 0:
            raise ValueError("trajectory_steps must be positive")
        if self.routing.refractory_strength < 0:
            raise ValueError("refractory_strength cannot be negative")
        if not 0 <= self.routing.refractory_decay <= 1:
            raise ValueError("refractory_decay must be between zero and one")
        if self.routing.router_type not in {"fixed_index", "module_aware", "geometric"}:
            raise ValueError("unsupported router_type")
        if self.routing.integrator_type not in {"weighted_average", "proposal_attention", "acceptance_gate"}:
            raise ValueError("unsupported integrator_type")
        if self.routing.routing_geometry_dim <= 0 or self.routing.competence_prototypes_per_expert <= 0:
            raise ValueError("routing geometry dimensions and prototype counts must be positive")
        if self.routing.counterfactual_probe_preset not in {"decaying", "fixed"}:
            raise ValueError("unsupported counterfactual probe preset")
        rates = (
            self.routing.counterfactual_probe_early_rate,
            self.routing.counterfactual_probe_stable_rate,
            self.routing.counterfactual_probe_mature_rate,
        )
        if any(rate < 0 or rate > 1 for rate in rates):
            raise ValueError("counterfactual probe rates must be between zero and one")
        fixed_rate = self.routing.counterfactual_probe_fixed_rate
        if fixed_rate is not None and not 0 <= fixed_rate <= 1:
            raise ValueError("fixed counterfactual probe rate must be between zero and one")
        if self.routing.counterfactual_max_probes_per_forward < 0:
            raise ValueError("counterfactual probe budget cannot be negative")
        if self.routing.counterfactual_probe_temperature <= 0 or self.routing.geometry_temperature <= 0:
            raise ValueError("routing calibration temperatures must be positive")
        if self.model.preset not in {"quick", "research", "custom"}:
            raise ValueError("preset must be quick, research, or custom")
        if self.model.fairness_mode not in {"custom", "capacity", "compute"}:
            raise ValueError("fairness_mode must be custom, capacity, or compute")
        for name in ("latent_dim", "context_length", "attention_heads", "module_hidden_dim", "integrator_heads", "chunk_size", "shared_state_slots"):
            if getattr(self.model, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.model.latent_dim % self.model.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if self.model.latent_dim % self.model.integrator_heads:
            raise ValueError("latent_dim must be divisible by integrator_heads")
        for name in ("tokens", "batch_size", "gradient_accumulation", "evaluation_interval", "evaluation_batches", "telemetry_interval"):
            if getattr(self.training, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.training.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("unsupported precision")

    @property
    def expert_families(self) -> tuple[str, ...]:
        return tuple(
            family
            for family in EXPERT_FAMILIES
            for _ in range(int(self.experts.get(family, 0)))
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        incoming_version = int(value.get("schema_version", 1))
        if incoming_version == 2:
            value = dict(value)
            if value.get("architecture") == "emc":
                value["architecture"] = "sequential_module_aware_emc"
            value["schema_version"] = SCHEMA_VERSION
            incoming_version = SCHEMA_VERSION
        if incoming_version != SCHEMA_VERSION:
            raise ValueError(
                "experiment schema v1 is not auto-migrated because v1 'emc' "
                "means parallel Top-K. Choose an explicit current architecture."
            )
        return cls(
            schema_version=incoming_version,
            name=str(value.get("name", "")),
            notes=str(value.get("notes", "")),
            tags=tuple(str(tag) for tag in value.get("tags", ())),
            projection_targets=tuple(int(target) for target in value.get("projection_targets", (250_000, 500_000, 1_000_000, 10_000_000))),
            suite=str(value.get("suite", "tinystories")),
            architecture=str(value.get("architecture", "emc")),
            experts={str(key): int(count) for key, count in dict(value.get("experts", {})).items()},
            routing=RoutingConfig(**dict(value.get("routing", {}))),
            model=ModelConfig(**dict(value.get("model", {}))),
            training=ResearchTrainingConfig(**dict(value.get("training", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        routing = payload["routing"]
        if self.architecture in {"emc", "sequential_module_aware_emc"}:
            for key in (
                "top_k",
                "cycles",
                "minimum_lease_chunks",
                "balance_coefficient",
                "balance_entropy_floor",
                "balance_warmup_chunks",
            ):
                routing.pop(key, None)
            if self.architecture == "emc":
                routing.pop("switch_cost", None)
                routing.pop("persistence_bonus", None)
            if self.architecture == "sequential_module_aware_emc":
                for key in (
                    "routing_geometry_dim",
                    "competence_prototypes_per_expert",
                    "counterfactual_calibration_enabled",
                    "counterfactual_probe_preset",
                    "counterfactual_probe_fixed_rate",
                    "counterfactual_probe_early_rate",
                    "counterfactual_probe_stable_rate",
                    "counterfactual_probe_mature_rate",
                    "counterfactual_uncertainty_enabled",
                    "counterfactual_uncertainty_margin",
                    "counterfactual_max_probes_per_forward",
                    "counterfactual_probe_temperature",
                    "geometry_temperature",
                    "geometry_calibration_weight",
                ):
                    routing.pop(key, None)
        elif self.architecture in {"legacy_parallel_emc", "old_emc", *N2_ARCHITECTURES}:
            for key in (
                "trajectory_steps",
                "refractory_enabled",
                "refractory_strength",
                "refractory_decay",
            ):
                routing.pop(key, None)
            if self.architecture == "legacy_parallel_emc":
                routing.pop("cycles", None)
        return payload


def research_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suites": [
            {
                "id": "tinystories",
                "label": "TinyStories / Language Only",
                "description": "The existing GPT-2-tokenized TinyStories causal language-model training path.",
                "tasks": ["language"],
            },
            {
                "id": "capability_10",
                "label": "10-task Mixed Diagnostic",
                "description": "The deterministic mixed computational-capability curriculum and balanced diagnostic evaluator.",
                "tasks": list(CAPABILITIES),
            },
        ],
        "architectures": [
            {"id": "emc", "label": "Sequential EMC — Geometric"},
            {"id": "sequential_module_aware_emc", "label": "Sequential EMC — Legacy Module-Aware"},
            {"id": "legacy_parallel_emc", "label": "Legacy Parallel Top-K EMC"},
            {"id": "heterogeneous_serial", "label": "Heterogeneous Serial"},
            {"id": "homogeneous_serial", "label": "Homogeneous Transformer"},
            {"id": "old_emc", "label": "Legacy Token-routed EMC"},
            *({"id": name, "label": name.replace("_", " ").upper()} for name in N2_ARCHITECTURES),
        ],
        "expert_families": [
            {"id": "gpt", "label": "GPT / Attention"},
            {"id": "ssm", "label": "State Space"},
            {"id": "recurrent", "label": "GRU Recurrent"},
            {"id": "delta", "label": "Gated DeltaNet"},
        ],
        "n1_stages": list(N1_STAGES),
        "presets": {
            "50k": {"label": "50K Smoke Test", "tokens": 50_000},
            "100k": {"label": "100K Test", "tokens": 100_000},
            "1m": {"label": "1M Gate", "tokens": 1_000_000},
        },
        "model_presets": MODEL_PRESET_DIMENSIONS,
        "defaults": ExperimentConfig().to_dict(),
    }
