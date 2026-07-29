"""TPRE-ALNS reproducibility package."""

from .alns import ALNSConfig, TPREALNS
from .entities import (
    Customer,
    EVRPInstance,
    EvalMetrics,
    Scenario,
    Solution,
    Station,
)
from .evaluation import evaluate_solution, weighted_cvar
from .instance import (
    generate_synthetic_instance,
    load_customer_csv,
    load_instance_json,
    save_instance_json,
)
from .planning import certify_and_restore_solution
from .scenarios import (
    generate_scenarios,
    load_scenarios_json,
    save_scenarios_json,
)

__version__ = "1.0.0"

__all__ = [
    "ALNSConfig",
    "Customer",
    "EVRPInstance",
    "EvalMetrics",
    "Scenario",
    "Solution",
    "Station",
    "TPREALNS",
    "certify_and_restore_solution",
    "evaluate_solution",
    "generate_scenarios",
    "generate_synthetic_instance",
    "load_customer_csv",
    "load_instance_json",
    "load_scenarios_json",
    "save_instance_json",
    "save_scenarios_json",
    "weighted_cvar",
]
