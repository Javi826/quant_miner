#signals/signal_builder.py
import numpy as np

from signals.condition_bank import ConditionBank


def describe_rule(bank: ConditionBank, rule_specs: list) -> str:
    return " AND ".join(bank.describe(spec) for spec in rule_specs)


def build_signal_fn(rule_specs: list, side: str):
    def signal_fn(arr: dict, live_trading: bool = True, bank: ConditionBank = None) -> np.ndarray:
        if bank is None:
            bank = ConditionBank(arr)
        mask = np.ones(bank.n, dtype=bool)
        for spec in rule_specs:
            mask &= bank.evaluate(spec)

        signal = np.zeros(bank.n, dtype=np.int32)
        if side == "long":
            signal[mask] = 1
        else:
            signal[mask] = -1
            
        if not live_trading:
            signal = np.roll(signal, 1)
            signal[0] = 0

        return signal

    return signal_fn

