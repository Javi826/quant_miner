# core/research/stages/combos.py
import os
import sys
import math
import random
import logging
import itertools
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import partial
import pandas as pd
from tqdm import tqdm
from setup import config_research as cr
from research import artifacts as ra

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = logging.INFO    # INFO: a header, a bar and a summary per timeframe and N. DEBUG: every combo as the backtest
logger    = logging.getLogger("BOT_research.combos")

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_day
from setup.config_backtest import ORDER_AMOUNT
from pipeline import signal_cleaning, backtest_runner, stepM_is
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.stepM_is import pipe_stepm, STEPM_ALPHA
from setup.config_core import settings
# =============================================================================
# MODULE CONFIG
# =============================================================================
RANDOM_SEED       = 42
N_JOBS            = -1
SIGNAL_CLEANING   = True

RANK_TOP_N  = 30
NOISE_PCTL  = 0.90
SPLIT_MAX   = 5   


# =============================================================================
# STAGE CONFIG (set by the orchestrator)
# =============================================================================
@dataclass(frozen=True)
class CombosStageConfig:
    plus_minus:         float = 0.2
    combo_sizes:        list  = field(default_factory=lambda: [1, 2])
    n_samples_per_size: dict  = field(default_factory=lambda: {1: None, 99: 99})    # None = exhaustive

    def __post_init__(self):
        if self.plus_minus < 0:
            raise ValueError(f"plus_minus must be >= 0: {self.plus_minus}")
        if not self.combo_sizes or any(not isinstance(n, int) or n < 1 for n in self.combo_sizes):
            raise ValueError(f"combo_sizes must be integers >= 1: {self.combo_sizes}")
        bad = {n: v for n, v in self.n_samples_per_size.items() if v is not None and (not isinstance(v, int) or v < 1)}
        if bad:
            raise ValueError(f"n_samples_per_size values must be None or integers >= 1: {bad}")


def configure_loggers() -> None:
    logger.setLevel(LOG_LEVEL)
    logging.getLogger("BOT_batch").setLevel(logging.INFO if LOG_LEVEL <= logging.DEBUG else logging.WARNING)
    for noisy_logger in ("joblib", "matplotlib", "numba"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# =============================================================================
# COMBO GENERATION
# =============================================================================
def _generate_combos(pool: list, size: int, n_samples: int | None, seed: int) -> list:
    all_combos = list(itertools.combinations(pool, size))
    if n_samples is None or n_samples >= len(all_combos):
        return all_combos
    rng = random.Random(seed)
    return rng.sample(all_combos, n_samples)

# =============================================================================
# PIPELINE PATCHES — bars of the pipeline modules only in DEBUG; global p-value and k of StepM
# =============================================================================
@contextmanager
def pipeline_bars(show: bool):
    mods = (signal_cleaning, backtest_runner, stepM_is)
    orig = [m.tqdm for m in mods]
    if not show:
        for m, t in zip(mods, orig):
            m.tqdm = partial(t, disable=True)
    try:
        yield
    finally:
        for m, t in zip(mods, orig):
            m.tqdm = t

@contextmanager
def stepm_capture():
    cap = {}
    orig_gp, orig_k = stepM_is.compute_global_pvalue, stepM_is.resolve_k_by_fdp

    def global_pvalue(*args, **kwargs):
        out = orig_gp(*args, **kwargs)
        cap["global_p"] = out["global_p"]
        return out

    def resolve_k(*args, **kwargs):
        out = orig_k(*args, **kwargs)
        cap["k"] = out[0]
        return out

    stepM_is.compute_global_pvalue, stepM_is.resolve_k_by_fdp = global_pvalue, resolve_k
    try:
        yield cap
    finally:
        stepM_is.compute_global_pvalue, stepM_is.resolve_k_by_fdp = orig_gp, orig_k

# =============================================================================
# BACKTEST UNIVERSE — build the raw universe once, hand it to StepM.
# =============================================================================
def _run_backtest_universe(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str,
    name: str,
    bar,
    n_jobs: int = -1,
    apply_signal_cleaning: bool = False,
) -> tuple:
    """Run the brute-force backtest once; shared by any diagnostic that needs the same matrix."""
    if apply_signal_cleaning:
        bar.set_postfix_str(f"{name} ── Jaccard")
        rules = pipe_signal_cleaning_jaccard(
            rules     = rules,
            ohlcv_arr = ohlcv_arr,
            timeframe = timeframe,
        )

    bar.set_postfix_str(f"{name} ── backtest")
    original_n_jobs = backtest_runner.BACKTEST_N_JOBS
    backtest_runner.BACKTEST_N_JOBS = n_jobs
    try:
        return backtest_runner.pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr,
            param_grid   = param_grid,
            order_amount = order_amount,
            timeframe    = timeframe,
        )
    finally:
        backtest_runner.BACKTEST_N_JOBS = original_n_jobs

# =============================================================================
# SINGLE COMBO RUN
# =============================================================================
def _run_combo(combo: tuple, combo_idx: int, ohlcv_arr_pool: dict, rule_templates: list, timeframe: str,
               param_grid: dict, bar, cap: dict) -> dict | None:

    name = "+".join(combo)
    ohlcv_arr_combo = {sym: ohlcv_arr_pool[sym] for sym in combo}

    combo_key = f"{timeframe}_c{combo_idx:02d}"            # as main_back_fx: the index shown in "Testing: ... <i/n>"
    rules = build_rule_dicts(rule_templates, combo_key, timeframe)

    try:
        raw_results, _, matrix_arr, col_names = _run_backtest_universe(
            rules                  = rules,
            ohlcv_arr              = ohlcv_arr_combo,
            param_grid             = param_grid,
            order_amount           = ORDER_AMOUNT,
            timeframe              = timeframe,
            name                   = name,
            bar                    = bar,
            n_jobs                 = N_JOBS,
            apply_signal_cleaning  = SIGNAL_CLEANING,
        )

        bar.set_postfix_str(f"{name} ── StepM")
        cap.clear()
        res = pipe_stepm(raw_results=raw_results, matrix_arr=matrix_arr, col_names=col_names, timeframe=timeframe)
    except ValueError as exc:
        logger.debug(f"SKIP ── {timeframe} ── {combo} ── {exc}")
        return None

    ran = "global_p" in cap and "k" in cap
    if not ran and any(r.get("stepm_p") is not None for r in res):
        raise RuntimeError("StepM ran but its global p-value and k were not captured: update stepm_capture")
    if not ran:
        logger.debug(f"STEPM SKIPPED  {timeframe}: {name} counted as 0 rules")
    return {
        "timeframe": timeframe,
        "n_symbols": len(combo),
        "symbols":   name,
        "rules":     sum(bool(r["passed_mbias"]) for r in res) if ran else 0,
        "global_p":  cap["global_p"] if ran else None,
        "k":         cap["k"] if ran else None,
    }

# =============================================================================
# REPORT HELPERS
# =============================================================================
SYMBOLS_COL_WIDTH  = 20
REPORT_LINE_WIDTH  = 100
HEADER_LABEL_WIDTH = 24
HEADER_INDENT       = " " * (2 + HEADER_LABEL_WIDTH + 3)   # aligns continuation lines under the value


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _header_line(label: str, value: str) -> str:
    """One header row, label padded to HEADER_LABEL_WIDTH; value's own newlines get HEADER_INDENT."""
    value = value.replace("\n", "\n" + HEADER_INDENT)
    return f"  {label:<{HEADER_LABEL_WIDTH}} : {value}"


def _format_param_grid(grid: dict) -> str:
    """PARAM_GRID_BY_TIMEFRAME as a multi-line list for the run header (one timeframe per line)."""
    lines = [f"'{tf}': {params}," for tf, params in grid.items()]
    return "\n".join(lines)


def _format_list(items: list, per_line: int = 5) -> str:
    lines = []
    for i in range(0, len(items), per_line):
        chunk = ", ".join(f"'{s}'" for s in items[i:i + per_line])
        lines.append(chunk + ",")
    return "\n".join(lines)


def _capped(parts: list, sep: str = " | ") -> str:
    if not parts:
        return "-"
    text = sep.join(parts[:SPLIT_MAX])
    return text + (f" +{len(parts) - SPLIT_MAX}" if len(parts) > SPLIT_MAX else "")


def _list(values: list) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"

def _spread(values: list, plus_minus: float) -> list:
    return sorted({round(v + d, 10) for v in values for d in (-plus_minus, 0.0, plus_minus)})
def _ordered_timeframes(timeframes: list) -> list:
    return sorted(timeframes, key=get_bars_per_day, reverse=True)    # smallest timeframe first

# =============================================================================
# EVERY TIMEFRAME AND N: header and summary
# =============================================================================
def _log_block_header(timeframe: str, size: int, k: int, n_sizes: int, n_combos: int, n_symbols: int,
                      n_rules: int) -> None:
    text = (f"  {timeframe} ── N={size} ({k}/{n_sizes}) ── combos: {n_combos} ── symbols: {n_symbols} ── "
            f"rules: {_fmt(n_rules)} per combo")
    logger.info(f"\n{'─' * (len(text) + 2)}\n{text}\n{'─' * (len(text) + 2)}")


def _log_block(timeframe: str, size: int, rows: list, skipped: list) -> None:

    tag = f"N={size}"
    ran = [r for r in rows if r["global_p"] is not None]
    if ran:
        ps, ks = [r["global_p"] for r in ran], [r["k"] for r in ran]
        n_sig  = sum(p <= STEPM_ALPHA for p in ps)
        logger.info(f"{f'WHITE {tag}':<16}{timeframe}: p {min(ps):.4f} to {max(ps):.4f} ── "
                    f"{n_sig}/{len(ran)} with p ≤ {STEPM_ALPHA:g}")
        logger.info(f"{f'STEPM {tag}':<16}{timeframe}: k {_fmt(min(ks))} to {_fmt(max(ks))}")
    else:
        logger.info(f"{f'WHITE {tag}':<16}{timeframe}: -")
        logger.info(f"{f'STEPM {tag}':<16}{timeframe}: -")
    logger.info(f"{f'RULES {tag}':<16}{timeframe}: {_fmt(sum(r['rules'] for r in rows))}")
    if skipped:
        logger.info(f"{f'SKIPPED {tag}':<16}{timeframe}: {len(skipped)} ── {_capped(skipped, ', ')}")

# =============================================================================
# RANKING ── rules passing StepM IS, then global p-value
# =============================================================================
def _build_ranking(subset: pd.DataFrame) -> pd.DataFrame:
    """One timeframe: the combos with rules passing StepM IS, by rules, then global p-value (lowest first)."""
    df = subset[subset["rules"] > 0]
    return df.sort_values(["rules", "global_p"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _binom_ceiling(n: int, p: float, pctl: float) -> int:
    cumulative = 0.0
    for k in range(n + 1):
        cumulative += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                               + k * math.log(p) + (n - k) * math.log1p(-p))
        if cumulative >= pctl:
            return k
    return n


def _summary_row(timeframe: str, ranking: pd.DataFrame, shortlist: pd.DataFrame, n_run: int) -> dict:
    return {
        "timeframe": timeframe,
        "n_run":     n_run,
        "n_with":    len(ranking),
        "ceiling":   _binom_ceiling(n_run, STEPM_ALPHA, NOISE_PCTL) if n_run else None,
        "n_top":     len(shortlist),
        "top_rules": int(shortlist["rules"].sum()),
    }


def _log_top_table(timeframe: str, shortlist: pd.DataFrame) -> None:
    if shortlist.empty or not logger.isEnabledFor(logging.DEBUG):
        return
    table = shortlist[["symbols", "n_symbols", "rules", "global_p"]].assign(
        symbols=shortlist["symbols"].str.ljust(SYMBOLS_COL_WIDTH))
    logger.debug(f"\n{'─' * REPORT_LINE_WIDTH}\n  TOP {len(shortlist)} {timeframe}\n{'─' * REPORT_LINE_WIDTH}")
    logger.debug(table.round(4).to_string(index=False))


def _log_summary(rows: list) -> None:
    ceiling_label = f"P{NOISE_PCTL * 100:g} CEILING"
    logger.info(f"\n{'=' * REPORT_LINE_WIDTH}")
    logger.info(f"  RANKING ── rules passing StepM IS, then global p-value ── TOP {RANK_TOP_N}")
    logger.info(f"{'=' * REPORT_LINE_WIDTH}")
    logger.info(f"  {'TIMEFRAME':<11}{'COMBOS WITH RULES':<20}{'%':>8}{ceiling_label:>14}{'TOP':>6}"
                f"{'TOP RULES':>12}   VERDICT")
    logger.info(f"  {'─' * (REPORT_LINE_WIDTH - 2)}")
    for r in rows:
        if r["n_run"] == 0:
            logger.info(f"  {r['timeframe']:<11}no combo produced a result")
            continue
        combos  = f"{_fmt(r['n_with'])} / {_fmt(r['n_run'])}"
        pct     = f"{r['n_with'] / r['n_run']:.1%}"
        verdict = "✅" if r["n_with"] > r["ceiling"] else "❌"
        logger.info(f"  {r['timeframe']:<11}{combos:<20}{pct:>8}{r['ceiling']:>14}{r['n_top']:>6}"
                    f"{_fmt(r['top_rules']):>12}   {verdict}")
    logger.info(f"{'=' * REPORT_LINE_WIDTH}")


def _backtest_config(shortlists: dict, param_grids: dict, indicators: dict, plus_minus: float) -> dict:
    return {
        timeframe: {
            "combos":     [combo.split("+") for combo in shortlist["symbols"]],
            "param_grid": {"SELL_AFTER": list(param_grids[timeframe]["SELL_AFTER"]),
                           "TP_PCT":     _spread(param_grids[timeframe]["TP_PCT"], plus_minus),
                           "SL_PCT":     _spread(param_grids[timeframe]["SL_PCT"], plus_minus)},
            "indicators": list(indicators[timeframe]),
        }
        for timeframe, shortlist in shortlists.items()
    }


def _log_paste(result: dict) -> None:

    logger.info("\nSYMBOL_COMBOS_BY_TIMEFRAME = {")
    for timeframe, r in result.items():
        if not r["combos"]:
            logger.info(f'    "{timeframe}": [],')
            continue
        logger.info(f'    "{timeframe}": [')
        for combo in r["combos"]:
            logger.info("        [" + ", ".join(f'"{s}"' for s in combo) + "],")
        logger.info("    ],")
    logger.info("}")

    logger.info("\nPARAM_GRID_BY_TIMEFRAME = {")
    for timeframe, r in result.items():
        grid = r["param_grid"]
        logger.info(f'    "{timeframe}": {{')
        logger.info(f'        "SELL_AFTER": {_list(grid["SELL_AFTER"])},')
        logger.info(f'        "TP_PCT":     {_list(grid["TP_PCT"])},')
        logger.info(f'        "SL_PCT":     {_list(grid["SL_PCT"])},')
        logger.info("    },")
    logger.info("}")

    logger.info("\nSELECTED_INDICATORS_BY_TIMEFRAME = {")
    for timeframe, r in result.items():
        logger.info(f'    "{timeframe}": [')
        for ind in r["indicators"]:
            logger.info(f'        "{ind}",')
        logger.info("    ],")
    logger.info("}")

# =============================================================================
# RUN
# =============================================================================
def run(cfg: CombosStageConfig, inputs: dict | None = None) -> dict:
    configure_loggers()
    if inputs is None:
        doc = ra.load_combos_input(settings.BACKTEST_MODE, cr.TIMEFRAMES)
    else:
        doc = ra.check_combos_input(inputs, settings.BACKTEST_MODE, cr.TIMEFRAMES)
    inputs      = {tf: doc["by_timeframe"][tf] for tf in cr.TIMEFRAMES}
    pools       = {tf: e["symbols"] for tf, e in inputs.items()}
    param_grids = {tf: e["param_grid"] for tf, e in inputs.items()}
    indicators  = {tf: e["indicators"] for tf, e in inputs.items()}

    logger.info(f"\n{'─' * 100}")
    logger.info("  COMBOS START")
    logger.info(f"{'─' * 100}")
    logger.info(_header_line("DATASET", f"{cr.DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[cr.DATASET])}"))
    logger.info(_header_line("BACKTEST", str(settings.BACKTEST_MODE)))
    logger.info(_header_line("GRIDS", f"{ra.combos_path()} (created {doc['created']})"))
    for tf in cr.TIMEFRAMES:
        logger.info(_header_line(f"SYMBOL_POOL {tf}", _format_list(pools[tf])))
        logger.info(_header_line(f"INDICATORS {tf}", _format_list(indicators[tf], per_line=3)))
    logger.info(_header_line("COMBO_SIZES", str(cfg.combo_sizes)))
    logger.info(_header_line("MAX_DEPTH", f"{RULE_MAX_DEPTH} (grids: {doc['max_depth']})"))
    logger.info(_header_line("PARAM_GRID_BY_TIMEFRAME", _format_param_grid(param_grids)))
    logger.info(_header_line("N_SAMPLES_PER_SIZE", str(cfg.n_samples_per_size)))
    logger.info(_header_line("PLUS_MINUS", str(cfg.plus_minus)))
    logger.info(_header_line("RANKING", "rules passing StepM IS, then global p-value"))
    logger.info(f"{'─' * 100}")

    ohlcv_data_by_timeframe = build_universe(
        DATA_FOLDER_BY_DATASET[cr.DATASET], {tf: pools[tf] for tf in cr.TIMEFRAMES},
        dataset=cr.DATASET,
    )
    ohlcv_arr_by_timeframe  = {
        timeframe: prepare_ohlcv_arrays(ohlcv_is)
        for timeframe, ohlcv_is in ohlcv_data_by_timeframe.items()
    }

    all_rows = []
    debug    = logger.isEnabledFor(logging.DEBUG)
    with pipeline_bars(show=debug), stepm_capture() as cap:
        for timeframe in cr.TIMEFRAMES:
            pool           = pools[timeframe]
            ohlcv_arr_pool = ohlcv_arr_by_timeframe[timeframe]
            param_grid     = param_grids[timeframe]
            rule_templates = build_rule_templates(ohlcv_arr_pool, indicators=indicators[timeframe],
                                                  max_depth=RULE_MAX_DEPTH)

            for i_size, size in enumerate(cfg.combo_sizes, start=1):
                combos   = _generate_combos(pool, size, cfg.n_samples_per_size.get(size), RANDOM_SEED)
                n_combos = len(combos)
                _log_block_header(timeframe, size, i_size, len(cfg.combo_sizes), n_combos, len(pool),
                                  len(rule_templates))

                rows, skipped = [], []
                bar = tqdm(combos, desc=f"{f'COMBOS {cr.DATASET} N={size}':<16}{timeframe}", dynamic_ncols=True,
                           disable=debug, file=sys.stdout)
                for combo_idx, combo in enumerate(bar, start=1):
                    logger.debug(f"{'-' * 100}")
                    logger.debug(f"Testing: {' + '.join(combo)} <{combo_idx}/{n_combos}>")
                    logger.debug(f"{'-' * 100}")
                    row = _run_combo(combo, combo_idx, ohlcv_arr_pool, rule_templates, timeframe, param_grid, bar,
                                     cap)
                    if row is None or row["global_p"] is None:
                        skipped.append("+".join(combo))
                    else:
                        rows.append(row)
                bar.close()
                _log_block(timeframe, size, rows, skipped)
                all_rows += rows

    columns    = ["timeframe", "n_symbols", "symbols", "rules", "global_p", "k"]
    results_df = pd.DataFrame(all_rows, columns=columns)

    shortlists, summary = {}, []
    for timeframe in _ordered_timeframes(cr.TIMEFRAMES):
        subset    = results_df[results_df["timeframe"] == timeframe]
        ranking   = _build_ranking(subset)
        shortlist = ranking.head(RANK_TOP_N)
        _log_top_table(timeframe, shortlist)
        shortlists[timeframe] = shortlist
        summary.append(_summary_row(timeframe, ranking, shortlist, len(subset)))
    _log_summary(summary)
    result = _backtest_config(shortlists, param_grids, indicators, cfg.plus_minus)
    _log_paste(result)
    return result
