#core/rule_mining/rule_generator.py
import itertools
import logging
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn, describe_rule

logger = logging.getLogger("BOT_batch.rule_mining.generator")

MAX_DEPTH = 3
SIDES     = ("long", "short")


def generate_valid_combos(specs: list, depth: int, indices: list = None) -> list:
    """Combinations of `depth` conditions, at most one per indicator. No side filtering."""
    candidate_indices = indices if indices is not None else range(len(specs))

    by_indicator = {}
    for i in candidate_indices:
        by_indicator.setdefault(specs[i]["indicator"], []).append(i)

    names  = list(by_indicator.keys())
    combos = []
    for name_combo in itertools.combinations(names, depth):
        pools = [by_indicator[n] for n in name_combo]
        for members in itertools.product(*pools):
            combos.append(tuple(sorted(members)))
    return combos


def generate_rule_combinations(condition_specs: list, max_depth: int = MAX_DEPTH) -> list:
    rules = []
    for depth in range(1, max_depth + 1):
        for members in generate_valid_combos(condition_specs, depth):
            rules.append([condition_specs[i] for i in members])
    return rules


def generate_all_rules(arr_sample: dict, max_depth: int = MAX_DEPTH, timeframe: str = None) -> list:
    bank            = ConditionBank(arr_sample, timeframe=timeframe)
    condition_specs = bank.build_condition_specs()
    rule_combos     = generate_rule_combinations(condition_specs, max_depth)
    all_rules = []
    for side in SIDES:
        for rule_specs in rule_combos:
            all_rules.append({
                "side":       side,
                "specs":      rule_specs,
                "label":      describe_rule(bank, rule_specs),
                "signal_fn":  build_signal_fn(rule_specs, side),
            })
    return all_rules