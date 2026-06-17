"""Interpretable, steerable generation: TFM-as-EBM + SAE-latent clamping."""
from .energy import EnergyModel, MockEnergyModel
from .sgld import generate, sgld_sample
from .steer import controllability_report, make_steered_energy
from .eval_gen import fidelity_suite

__all__ = [
    "EnergyModel",
    "MockEnergyModel",
    "sgld_sample",
    "generate",
    "make_steered_energy",
    "controllability_report",
    "fidelity_suite",
]
