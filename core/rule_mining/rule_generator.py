#core/rule_mining/rule_generator.py
import itertools
import logging
from indicators.indicators_pool import CANDIDATE_REGISTRY, build_flat_specs
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn, describe_rule

logger = logging.getLogger("BOT_batch.rule_mining.generator")

MAX_DEPTH = 3
SIDES     = ("long", "short")


# =============================================================================
# INDICATOR SELECTION
# =============================================================================
def validate_indicators(indicators) -> list:
    if isinstance(indicators, str) or not isinstance(indicators, (list, tuple)) or not indicators:
        raise ValueError(f"indicators must be a non-empty list of indicator names: {indicators!r}")
    unknown = sorted(set(indicators) - set(CANDIDATE_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown indicators: {unknown}")
    return list(dict.fromkeys(indicators))


def validate_indicators_by_timeframe(indicators_by_timeframe: dict, timeframes) -> None:
    missing = sorted(set(timeframes) - set(indicators_by_timeframe))
    if missing:
        raise ValueError(f"No indicators for timeframes: {missing}")
    for timeframe in set(timeframes):
        try:
            validate_indicators(indicators_by_timeframe[timeframe])
        except ValueError as exc:
            raise ValueError(f"{timeframe}: {exc}") from None


def select_condition_specs(indicators) -> list:
    selected = set(validate_indicators(indicators))
    specs    = [spec for spec in build_flat_specs() if spec["indicator"] in selected]
    if not specs:
        raise ValueError(f"No condition specs for indicators: {sorted(selected)}")
    return specs


# =============================================================================
# RULE GENERATION
# =============================================================================
def generate_valid_combos(specs: list, depth: int, indices: list = None) -> list:
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


def generate_all_rules(arr_sample: dict, *, indicators, max_depth: int = MAX_DEPTH) -> list:
    bank            = ConditionBank(arr_sample)
    condition_specs = select_condition_specs(indicators)
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