#core/rule_mining/rule_generator.py
import logging
from indicators.indicators_pool import generate_valid_combos, implied_side
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn, describe_rule

logger = logging.getLogger("BOT_batch.rule_mining.generator")

MAX_DEPTH = 2
SIDES     = ("long", "short")


def generate_rule_combinations(condition_specs: list, max_depth: int = MAX_DEPTH) -> list:
    rules = []
    for depth in range(1, max_depth + 1):
        for members in generate_valid_combos(condition_specs, depth):
            rules.append([condition_specs[i] for i in members])
    return rules


def is_side_coherent(spec: dict, side: str) -> bool:
    implied = implied_side(spec)
    return implied is None or implied == side


def generate_all_rules(arr_sample: dict, max_depth: int = MAX_DEPTH) -> list:
    bank            = ConditionBank(arr_sample)
    condition_specs = bank.build_condition_specs()
    rule_combos     = generate_rule_combinations(condition_specs, max_depth)
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