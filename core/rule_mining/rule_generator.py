# core/rule_mining/rule_generator.py
import logging
from itertools import combinations
from signals.condition_bank import ConditionBank
from signals.signal_builder import build_signal_fn, describe_rule

logger = logging.getLogger("BOT_batch.rule_mining.generator")

MAX_DEPTH        = 3
SIDES            = ("long", "short")

def _indicator_key(bank: ConditionBank, spec: dict):
    entry         = bank._REGISTRY_BY_TYPE[spec["type"]]
    identity_keys = entry["identity_keys"]
    return (spec["type"],) + tuple(spec[k] for k in identity_keys)


def _pair_conflicts(bank: ConditionBank, spec_a: dict, spec_b: dict) -> bool:
    entry = bank._REGISTRY_BY_TYPE[spec_a["type"]]

    if not entry["has_threshold"]:

        return True

    if spec_a["op"] == spec_b["op"]:

        return True

    greater_spec = spec_a if spec_a["op"] == ">" else spec_b
    less_spec    = spec_b if spec_a["op"] == ">" else spec_a
    return greater_spec["value"] >= less_spec["value"]


def _has_conflict(bank: ConditionBank, rule_specs: list) -> bool:
    for i in range(len(rule_specs)):
        for j in range(i + 1, len(rule_specs)):
            spec_a, spec_b = rule_specs[i], rule_specs[j]
            if _indicator_key(bank, spec_a) != _indicator_key(bank, spec_b):
                continue
            if _pair_conflicts(bank, spec_a, spec_b):
                return True
    return False


def generate_rule_combinations(bank: ConditionBank, condition_specs: list, max_depth: int = MAX_DEPTH) -> list:
    rules = []
    for depth in range(1, max_depth + 1):
        for combo in combinations(range(len(condition_specs)), depth):
            rule_specs = [condition_specs[i] for i in combo]
            if depth > 1 and _has_conflict(bank, rule_specs):
                label = " AND ".join(bank.describe(spec) for spec in rule_specs)
                logger.debug(f"discarded: {label}")
                continue
            rules.append(rule_specs)
    return rules


def is_side_coherent(spec: dict, side: str) -> bool:

    if spec["type"] == "rsi":
        op    = spec.get("op")
        value = spec["value"]
        if op == "<" and value <= 30:
            return side == "long"
        if op == ">" and value >= 70:
            return side == "short"
    return True


def generate_all_rules(arr_sample: dict, max_depth: int = MAX_DEPTH) -> list:
    bank            = ConditionBank(arr_sample)
    condition_specs = bank.build_condition_specs()
    rule_combos     = generate_rule_combinations(bank, condition_specs, max_depth)
    all_rules = []
    for side in SIDES:
        for rule_specs in rule_combos:
            if not all(is_side_coherent(spec, side) for spec in rule_specs):
                continue
            all_rules.append({
                "side":       side,
                "specs":      rule_specs,
                "label":      describe_rule(bank, rule_specs),
                "signal_fn":  build_signal_fn(rule_specs, side),
            })
    return all_rules
