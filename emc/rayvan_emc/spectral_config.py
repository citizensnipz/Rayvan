"""Explicit configuration for the opt-in spectral-geometric experiment."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class SpectralConfig:
    spectral_neighbourhood_size: int = 16
    spectral_knn_k: int = 4
    spectral_graph_mode: str = 'latent_knn'
    spectral_sequence_edge_weight: float = 0.05
    spectral_modes: int = 8
    spectral_bands: int = 6
    spectral_log_frequency_min: float = 0.01
    spectral_log_frequency_max: float = 2.0
    spectral_filter_width: float = 0.8
    spectral_zero_threshold: float = 1e-5
    spectral_eps: float = 1e-8
    hks_scales: int = 6
    hks_time_min: float = 0.1
    hks_time_max: float = 100.0
    wks_bands: int = 6
    geometry_pca_components: int = 8
    geometry_tangent_rank: int = 2
    geometry_statistics_frozen: bool = False
    geometry_std_floor: float = 1e-3
    basins_per_expert: int = 4
    basin_temperature: float = 0.1
    spectral_router_temperature: float = 0.25
    spectral_target_temperature: float = 0.25
    geometry_score_weight: float = 1.0
    resonance_score_weight: float = 1.0
    basin_variance_floor: float = 0.05
    basin_variance_ceiling: float = 100.0
    transformation_rank: int = 4
    transformation_ridge: float = 1e-5
    transformation_complementarity_threshold: float = 0.9
    transformation_complementarity_weight: float = 0.0
    transformation_diagnostics_enabled: bool = True
    basin_redundancy_weight: float = 0.001
    basin_redundancy_distance: float = 0.1
    basin_redundancy_similarity: float = 0.95
    spectral_route_weight: float = 1.0
    spectral_regret_weight: float = 0.0
    geometry_temporal_delta_enabled: bool = False

    def __post_init__(self):
        nonnegative = {'spectral_sequence_edge_weight', 'geometry_score_weight', 'resonance_score_weight',
                       'transformation_complementarity_weight', 'basin_redundancy_weight',
                       'spectral_route_weight', 'spectral_regret_weight'}
        bounded = {'transformation_complementarity_threshold', 'basin_redundancy_similarity'}
        for f in fields(SpectralConfig):
            v = getattr(self, f.name)
            if f.type is bool:
                if type(v) is not bool:
                    raise ValueError(f'{f.name} must be boolean')
            elif f.type is int:
                if type(v) is not int or v <= 0:
                    raise ValueError(f'{f.name} must be a positive integer')
            elif f.type is float:
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f'{f.name} must be finite')
                if f.name in bounded:
                    valid = -1 <= v <= 1
                else:
                    valid = v >= 0 if f.name in nonnegative else v > 0
                if not valid:
                    raise ValueError(f'{f.name} outside allowed range')
        if self.spectral_graph_mode not in {'latent_knn', 'latent_knn_plus_sequence'}:
            raise ValueError('unsupported spectral graph mode')
        for low, high in [('spectral_log_frequency_min', 'spectral_log_frequency_max'),
                          ('hks_time_min', 'hks_time_max'), ('basin_variance_floor', 'basin_variance_ceiling')]:
            if getattr(self, high) <= getattr(self, low):
                raise ValueError(f'{high} must exceed {low}')


def spectral_config(config):
    return SpectralConfig(**{f.name: getattr(config, f.name) for f in fields(SpectralConfig)})
