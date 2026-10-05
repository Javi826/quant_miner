#signals/indicators_bank.py
import numpy as np
from indicators.indicators_pool import CANDIDATE_REGISTRY, instance_key, describe_spec


def _compare(value: np.ndarray, op: str, threshold: float) -> np.ndarray:
    return value > threshold if op == ">" else value < threshold


class ConditionBank:

    def __init__(self, arr: dict, ctx: dict = None):
        self.arr = arr
        self.ctx = ctx or {}
        self.n = len(self.arr["close"])
        self._cache = {}

    def _get_value(self, indicator: str, params: dict) -> np.ndarray:
        key = instance_key(indicator, params)
        if key not in self._cache:
            fn = CANDIDATE_REGISTRY[indicator]["fn"]
            self._cache[key] = fn(self.arr, self.ctx, params)
        return self._cache[key]

    def evaluate(self, spec: dict) -> np.ndarray:
        value = self._get_value(spec["indicator"], spec["params"])
        return _compare(value, spec["op"], spec["threshold"])

    def describe(self, spec: dict) -> str:
        return describe_spec(spec)