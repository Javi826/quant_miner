#signals/indicators_bank.py (fx)
import numpy as np
from indicators.indicators_pool import CANDIDATE_REGISTRY, build_flat_specs, instance_key, describe_spec

# =============================================================================
# SELECTED INDICATORS — manual curation after reading the screening printout.
# =============================================================================
SELECTED_INDICATORS_BY_TIMEFRAME = {
    "1H": [
        "ichimoku_price_vs_cloud",
        "close_pos_in_bar",
        "donchian_pos",
        "ma_dist",
        "vortex",
        "bb_pctb",
        "bb_pctb_slope",
        "ppo",
    ],
    "4H": [
        "close_pos_in_bar",
        "bb_pctb",
        "open_close_momentum",
        "inside_outside_ratio",
        "acceleration",
        "donchian_pos",
        "ichimoku_price_vs_cloud",
        "pivot_dist",
    ],
}

def _compare(value: np.ndarray, op: str, threshold: float) -> np.ndarray:
    return value > threshold if op == ">" else value < threshold

class ConditionBank:

    def __init__(self, arr: dict, ctx: dict = None, timeframe: str = None):
        self.arr = arr
        self.ctx = ctx or {}
        self.timeframe = timeframe
        self.n = len(self.arr["close"])
        self._cache = {}

    def _get_value(self, indicator: str, params: dict) -> np.ndarray:
        key = instance_key(indicator, params)
        if key not in self._cache:
            fn = CANDIDATE_REGISTRY[indicator]["fn"]
            self._cache[key] = fn(self.arr, self.ctx, params)
        return self._cache[key]

    def build_condition_specs(self) -> list:
        if self.timeframe not in SELECTED_INDICATORS_BY_TIMEFRAME:
            raise ValueError(
                f"No SELECTED_INDICATORS_BY_TIMEFRAME entry for timeframe: {self.timeframe}"
            )
        selected = SELECTED_INDICATORS_BY_TIMEFRAME[self.timeframe]
        return [
            spec for spec in build_flat_specs()
            if spec["indicator"] in selected
        ]

    def evaluate(self, spec: dict) -> np.ndarray:
        value = self._get_value(spec["indicator"], spec["params"])
        return _compare(value, spec["op"], spec["threshold"])

    def describe(self, spec: dict) -> str:
        return describe_spec(spec)