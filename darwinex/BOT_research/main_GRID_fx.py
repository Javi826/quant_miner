#quant_miner/darwinex/BOT_research/main_grid_fx.py (forex)
import os
import sys
import json
import time
import logging
from contextlib import contextmanager
from functools import partial
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from setup.config_core import settings
from utils.ohlcv_utils import prepare_ohlcv_arrays
from signals.indicators_bank import SELECTED_INDICATORS_BY_TIMEFRAME, ConditionBank
from rule_mining.rule_generator import generate_rule_combinations, SIDES
from rule_mining.rule_runner import _build_rule_dicts
from pipeline import signal_cleaning, backtest_runner, stepM_is
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.backtest_runner import pipe_backtesting, _combo_grid, _combo_id
from pipeline.stepM_is import pipe_stepm
# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = logging.INFO    # INFO: a header and a bar per SELL_AFTER. DEBUG: every symbol and terna as the backtest
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.grid")
logger.setLevel(LOG_LEVEL)
logging.getLogger("BOT_batch").setLevel(logging.INFO if LOG_LEVEL <= logging.DEBUG else logging.WARNING)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
# =============================================================================
# CONFIG
# =============================================================================
MAX_DEPTHS = [3]      # one sweep per MAX_DEPTH (rules of 1..MAX_DEPTH conditions per side, as in the backtest), each
                      # with its ranking and winner. The table goes in the order of the first one
N_WINNERS  = 3        # blocks printed per MAX_DEPTH: the first N of its ranking (ternas with 0 rules are skipped)
SPLIT_MAX  = 5        # SPLIT column of the ranking: rules of the first N symbols (most rules first), the rest as +n
TIMEFRAME = "4H"
DATASET   = "IS"

# Input (do not edit): TOP, SYMBOL_POOL and TP_PCT x SL_PCT of every SELL_AFTER, written by main_screen_fx
SCREEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_grids", f"screen_{DATASET}_{TIMEFRAME}.json")

# =============================================================================
# DATA
# =============================================================================
def load_screen() -> dict:
    with open(SCREEN_PATH, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc["dataset"] != DATASET or doc["timeframe"] != TIMEFRAME:
        raise ValueError(f"{SCREEN_PATH} is {doc['dataset']} {doc['timeframe']}, expected {DATASET} {TIMEFRAME}")
    if doc["mode"] != settings.BACKTEST_MODE:
        raise ValueError(f"{SCREEN_PATH} was screened in MODE={doc['mode']} but the backtest runs in "
                         f"BACKTEST_MODE={settings.BACKTEST_MODE} (setup/config_core): they must be the same")
    return doc

def load_symbol(sym: str) -> tuple:
    """One independent universe per symbol, as load_ohlcv_by_combo of main_back_fx: (data, arrays)."""
    data = build_universe(DATA_FOLDER_BY_DATASET[DATASET], {TIMEFRAME: [sym]}, dataset=DATASET)[TIMEFRAME]
    return data, prepare_ohlcv_arrays(data)


def count_rules(arr: dict, max_depth: int) -> int:
    """Rules per symbol of the current TOP, as rule_generator makes them (both sides): the same on every symbol."""
    specs = ConditionBank(arr, timeframe=TIMEFRAME).build_condition_specs()
    return len(generate_rule_combinations(specs, max_depth)) * len(SIDES)

# =============================================================================
# SWEEP
# =============================================================================
@contextmanager
def bank_selection():

    had, orig = TIMEFRAME in SELECTED_INDICATORS_BY_TIMEFRAME, SELECTED_INDICATORS_BY_TIMEFRAME.get(TIMEFRAME)
    try:
        yield
    finally:
        if had:
            SELECTED_INDICATORS_BY_TIMEFRAME[TIMEFRAME] = orig
        else:
            SELECTED_INDICATORS_BY_TIMEFRAME.pop(TIMEFRAME, None)


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


def sweep_symbol(combo_key: str, sym: str, data: dict, arr: dict, param_grid: dict, max_depth: int, bar) -> dict:
    rules = _build_rule_dicts(data, combo_key, TIMEFRAME, max_depth)
    logger.debug(f"\n\033[36m{'─' * 70}")
    logger.debug(f"─ RULE MINING ── {combo_key} {[sym]} ── rules: {_fmt(len(rules))}")
    logger.debug(f"{'─' * 70}\033[0m")

    bar.set_postfix_str(f"{sym} ── Jaccard")
    rules = pipe_signal_cleaning_jaccard(rules=rules, ohlcv_arr=arr, timeframe=TIMEFRAME)
    bar.set_postfix_str(f"{sym} ── backtest")
    raw_results, _, matrix_arr, col_names = pipe_backtesting(
        rules=rules, ohlcv_arr=arr, param_grid=param_grid, order_amount=ORDER_AMOUNT, timeframe=TIMEFRAME,
    )
    suffix = [c.rsplit("__", 1)[1] for c in col_names]
    ternas = _combo_grid(param_grid)
    n_pass, skipped = {}, []
    for k, p in enumerate(ternas, start=1):
        cid = _combo_id(p)
        bar.set_postfix_str(f"{sym} ── StepM terna {k}/{len(ternas)}")
        logger.debug(f"\n── TERNA {k}/{len(ternas)} ── {combo_key} {[sym]} ── "
                     f"SELL_AFTER={p['SELL_AFTER']} TP_PCT={p['TP_PCT']} SL_PCT={p['SL_PCT']}")
        cols = np.asarray([i for i, s in enumerate(suffix) if s == cid], dtype=np.int64)
        sub  = matrix_arr[:, cols]
        sub  = np.ascontiguousarray(sub[np.any(sub != 0, axis=1)])
        res  = pipe_stepm(raw_results=raw_results, matrix_arr=sub, col_names=[col_names[i] for i in cols],
                          timeframe=TIMEFRAME)
        if any(r["passed_mbias"] and r["stepm_p"] is None for r in res):
            n_pass[cid] = 0
            skipped.append(cid)
            logger.warning(f"STEPM SKIPPED  {TIMEFRAME}: {sym} SELL_AFTER={p['SELL_AFTER']} TP_PCT={p['TP_PCT']} "
                           f"SL_PCT={p['SL_PCT']} counted as 0 rules in the sweep")
        else:
            n_pass[cid] = sum(bool(r["passed_mbias"]) for r in res)
    return {"pass": n_pass, "skipped": skipped}

# =============================================================================
# MAIN
# =============================================================================
def main():
    doc   = load_screen()
    by_sa = {int(sa): o for sa, o in doc["by_sell_after"].items()}
    log_run_config(doc, by_sa)
    for sa in sorted(by_sa):
        if not by_sa[sa]["top"] or not by_sa[sa]["symbols"]:
            logger.warning(f"  SELL_AFTER={sa}: empty TOP or SYMBOL_POOL, skipped")
    sas = [sa for sa in sorted(by_sa) if by_sa[sa]["top"] and by_sa[sa]["symbols"]]
    symbols_all = list(dict.fromkeys(sym for sa in sas for sym in by_sa[sa]["symbols"]))
    loaded = {sym: load_symbol(sym) for sym in symbols_all}             # all the data first, as main_back_fx
    grids  = {sa: {"SELL_AFTER": [sa], "TP_PCT": doc["tp_pct"], "SL_PCT": doc["sl_pct"]} for sa in sas}
    rows   = [{"sa": sa, "tp": p["TP_PCT"], "sl": p["SL_PCT"], "cid": _combo_id(p), "top": by_sa[sa]["top"],
               "symbols": by_sa[sa]["symbols"], "total": {}, "n_sym": {}, "n_eff": {}, "split": {}, "with_rules": {}}
              for sa in sas for p in _combo_grid(grids[sa])]
    debug  = logger.isEnabledFor(logging.DEBUG)
    with pipeline_bars(show=debug), bank_selection():
        for depth in MAX_DEPTHS:                                        # the first MAX_DEPTH complete, then the next
            for k, sa in enumerate(sas, start=1):
                top, symbols = by_sa[sa]["top"], by_sa[sa]["symbols"]
                SELECTED_INDICATORS_BY_TIMEFRAME[TIMEFRAME] = list(top)   # the rules of this SELL_AFTER: its own TOP
                n_rules = count_rules(loaded[symbols[0]][1][symbols[0]], depth)
                log_sell_after_header(depth, sa, k, len(sas), len(symbols), len(top), n_rules)
                bar = tqdm(symbols, desc=f"{'SYMBOLS':<16}{TIMEFRAME}", dynamic_ncols=True, disable=debug)
                by_sym = {sym: sweep_symbol(f"{TIMEFRAME}_c{i:02d}", sym, *loaded[sym], grids[sa], depth, bar)
                          for i, sym in enumerate(bar, start=1)}
                bar.close()
                log_sell_after(depth, sa, _combo_grid(grids[sa]), by_sym)
                for r in rows:
                    if r["sa"] == sa:
                        per_sym = [by_sym[sym]["pass"][r["cid"]] for sym in symbols]
                        r["total"][depth] = sum(per_sym)
                        r["n_sym"][depth] = f"{sum(v > 0 for v in per_sym)}/{len(per_sym)}"
                        r["n_eff"][depth] = sum(per_sym) ** 2 / sum(v * v for v in per_sym) if sum(per_sym) else 0.0
                        r["split"][depth] = sorted((v for v in per_sym if v > 0), reverse=True)
                        r["with_rules"][depth] = [sym for sym, v in zip(symbols, per_sym) if v > 0]
    log_ranking(rows)

# =============================================================================
# PRINT
# =============================================================================
SEP = "─" * 115


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _label(tp, sl) -> str:
    return f"{tp}/{sl}"


def log_run_config(doc: dict, by_sa: dict) -> None:
    logger.info(f"\n{SEP}")
    logger.info(f"  GRID SWEEP START")
    logger.info(f"{SEP}")
    logger.info(f"  SCREENING   : {SCREEN_PATH} (created {doc['created']})")
    logger.info(f"  DATASET     : {DATASET} ── {TIMEFRAME}")
    logger.info(f"  MODE        : {doc['mode']} (screening) = {settings.BACKTEST_MODE} (BACKTEST_MODE)")
    logger.info(f"  PARAM GRID  : SELL_AFTER={sorted(by_sa)} TP_PCT={doc['tp_pct']} SL_PCT={doc['sl_pct']}")
    logger.info(f"  SELECTION   : GROUP_N={doc['selection']['group_n']} TOP_I={doc['selection']['top_i']} (screening)")
    logger.info(f"  RULES       : MAX_DEPTHS={MAX_DEPTHS}")
    logger.info(f"{SEP}\n")


def log_sell_after_header(depth: int, sa: int, k: int, n_sa: int, n_sym: int, n_ind: int, n_rules: int) -> None:
    text = (f"  MAX_DEPTH={depth} ── SELL_AFTER={sa} ({k}/{n_sa}) ── symbols: {n_sym} ── indicators: {n_ind} ── "
            f"rules: {_fmt(n_rules)} per symbol")
    logger.info(f"\n{'─' * (len(text) + 2)}\n{text}\n{'─' * (len(text) + 2)}")


def log_sell_after(depth: int, sa: int, ternas: list, by_sym: dict) -> None:
    labels = [_label(p["TP_PCT"], p["SL_PCT"]) for p in ternas]
    cids   = [_combo_id(p) for p in ternas]
    w      = max(9, max(len(s) for s in labels) + 2)
    logger.debug(f"\n{SEP}")
    logger.debug(f"  MAX_DEPTH={depth} ── SELL_AFTER={sa} ── rules passing StepM IS per terna (TP/SL)")
    logger.debug(f"{SEP}")
    logger.debug(f"  {'SYMBOL':<10}" + "".join(f"{s:>{w}}" for s in labels))
    for sym, res in by_sym.items():
        logger.debug(f"  {sym:<10}" + "".join(f"{res['pass'][c]:>{w}}" for c in cids))
    logger.debug(f"  {'TOTAL':<10}" + "".join(f"{sum(r['pass'][c] for r in by_sym.values()):>{w}}" for c in cids))


def _split_text(counts: list) -> str:
    """Rules per symbol with rules, most first, separated by |: the first SPLIT_MAX, the rest as +n."""
    if not counts:
        return "-"
    text = " | ".join(str(v) for v in counts[:SPLIT_MAX])
    return text + (f" +{len(counts) - SPLIT_MAX}" if len(counts) > SPLIT_MAX else "")


def _log_vertical(label: str, items: list) -> None:
    logger.info(f"  {label} ({len(items)}):")
    for it in items:
        logger.info(f'    "{it}",')


def log_ranking(rows: list) -> None:
    """One ranking per MAX_DEPTH (tie: the smaller SELL_AFTER, then grid order), side by side in the order of the
    first one, and the winner of each MAX_DEPTH."""
    for d in MAX_DEPTHS:
        for pos, r in enumerate(sorted(rows, key=lambda r: (-r["total"][d], r["sa"])), start=1):
            r.setdefault("rank", {})[d] = pos
    d0 = MAX_DEPTHS[0]
    rows = sorted(rows, key=lambda r: r["rank"][d0])

    w = {d: max(len("SPLIT"), *(len(_split_text(r["split"][d])) for r in rows)) for d in MAX_DEPTHS}

    def block(r, d):                                                # rank (the first is the row's own), rules, symbols, n_eff, split
        return (f"  {r['total'][d]:>6}  {r['n_sym'][d]:>8}  {r['n_eff'][d]:>6.1f}  {_split_text(r['split'][d]):<{w[d]}}"
                if d == d0 else
                f"  {r['rank'][d]:>4}  {r['total'][d]:>6}  {r['n_sym'][d]:>8}  {r['n_eff'][d]:>6.1f}  "
                f"{_split_text(r['split'][d]):<{w[d]}}")

    lead  = f"  {'#':>3}  {'SA':>5}  {'TP':>6}  {'SL':>6}"
    heads = [f"  {'RULES':>6}  {'SYMBOLS':>8}  {'N_EFF':>6}  {'SPLIT':<{w[d]}}" if d == d0 else
             f"  {'#':>4}  {'RULES':>6}  {'SYMBOLS':>8}  {'N_EFF':>6}  {'SPLIT':<{w[d]}}"
             for d in MAX_DEPTHS]
    logger.info(f"\n{SEP}")
    logger.info(f"  RANKING ── rules passing StepM IS, summed over the SYMBOL_POOL of each SELL_AFTER "
                f"(tie: smaller SELL_AFTER)")
    logger.info(f"{SEP}")
    logger.info(" " * len(lead) + "".join(f"{f'MAX_DEPTH={d}':>{len(h) - w[d] - 2}}{'':{w[d] + 2}}"   # over the numbers
                                          for d, h in zip(MAX_DEPTHS, heads)))
    logger.info(lead + "".join(heads))
    for r in rows:
        logger.info(f"  {r['rank'][d0]:>3}  {r['sa']:>5}  {r['tp']:>6}  {r['sl']:>6}"
                    + "".join(block(r, d) for d in MAX_DEPTHS))
    logger.info(f"  N_EFF: effective number of symbols with rules, (sum of rules)² / sum of (rules per symbol)² "
                f"── 1 = every rule in one symbol")
    logger.info(f"  SPLIT: rules of each symbol with rules, most first (the first {SPLIT_MAX}, the rest as +n)")
    logger.info(f"{SEP}")
    for d in MAX_DEPTHS:
        log_winner(sorted(rows, key=lambda r: r["rank"][d]), d)


def log_winner(rows: list, d: int) -> None:
    """The first N_WINNERS ternas of MAX_DEPTH d, one block each. rows: in the order of its ranking."""
    winners = [r for r in rows[:N_WINNERS] if r["total"][d] > 0]
    if not winners:
        logger.info(f"  WINNERS (MAX_DEPTH={d}): no terna has rules passing StepM IS")
        logger.info(f"{SEP}")
        return
    for r in winners:
        logger.info(f"  #{r['rank'][d]} (MAX_DEPTH={d}): {r['total'][d]} rules ── symbols with rules {r['n_sym'][d]} "
                    f"── N_EFF {r['n_eff'][d]:.1f}")
        logger.info(f"")
        logger.info(f"  PARAM_GRID_BY_TIMEFRAME:")
        logger.info(f'    "{TIMEFRAME}": {{')
        logger.info(f'        "SELL_AFTER": [{r["sa"]}],')
        logger.info(f'        "TP_PCT":     [{r["tp"]}],')
        logger.info(f'        "SL_PCT":     [{r["sl"]}],')
        logger.info(f"    }},")
        logger.info(f"")
        _log_vertical(f"SELECTED_INDICATORS_BY_TIMEFRAME \"{TIMEFRAME}\"", r["top"])
        logger.info(f"")
        _log_vertical(f"SYMBOL_COMBOS_BY_TIMEFRAME       \"{TIMEFRAME}\" SYMBOL_POOL of the json", r["symbols"])
        logger.info(f"")
        _log_vertical(f"SYMBOL_COMBOS_BY_TIMEFRAME       \"{TIMEFRAME}\" with rules in this terna", r["with_rules"][d])
        logger.info(f"{SEP}")

if __name__ == "__main__":
    start = time.time()
    try:
        main()
        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)