#signals/indicators_bank.py
import numpy as np
from indicators.indicators_pool import CANDIDATE_REGISTRY, build_flat_specs, instance_key, describe_spec

# =============================================================================
# SELECTED INDICATORS — manual curation after reading the screening printout.
# =============================================================================
SELECTED_INDICATORS = [
    "donchian_width",
    "ichimoku_cloud_thickness",
    "inertia",
    "ppo",
    "rvi",
    "trend_intensity_index",
   
]

def _compare(value: np.ndarray, op: str, threshold: float) -> np.ndarray:
    return value > threshold if op == ">" else value < threshold

class ConditionBank:

    def __init__(self, arr: dict, ctx: dict = None):
        self.arr = {
            "open":  np.asarray(arr["open"],  dtype=np.float64),
            "high":  np.asarray(arr["high"],  dtype=np.float64),
            "low":   np.asarray(arr["low"],   dtype=np.float64),
            "close": np.asarray(arr["close"], dtype=np.float64),
        }
        self.ctx = ctx or {}
        self.n = len(self.arr["close"])
        self._cache = {}

    def _get_value(self, indicator: str, params: dict) -> np.ndarray:
        key = instance_key(indicator, params)
        if key not in self._cache:
            fn = CANDIDATE_REGISTRY[indicator]["fn"]
            self._cache[key] = fn(self.arr, self.ctx, params)
        return self._cache[key]

    def build_condition_specs(self) -> list:
        return [
            spec for spec in build_flat_specs()
            if spec["indicator"] in SELECTED_INDICATORS
        ]

    def evaluate(self, spec: dict) -> np.ndarray:
        value = self._get_value(spec["indicator"], spec["params"])
        return _compare(value, spec["op"], spec["threshold"])

    def describe(self, spec: dict) -> str:
        return describe_spec(spec)