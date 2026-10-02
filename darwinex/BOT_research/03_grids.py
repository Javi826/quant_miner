#quant_miner/darwinex/BOT_research/03_grids_fx.py (forex)
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
from signals.indicators_bank import SELECTED_INDICATORS_BY_TIMEFRAME
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from pipeline import signal_cleaning, backtest_runner, stepM_is
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.backtest_runner import pipe_backtesting, _combo_grid, _combo_id
from pipeline.stepM_is import pipe_stepm, STEPM_ALPHA
# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = logging.INFO    # INFO: a header, a bar and a summary per SELL_AFTER. DEBUG: every symbol and terna as the backtest
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_research.grid")
logger.setLevel(LOG_LEVEL)
logging.getLogger("BOT_batch").setLevel(logging.INFO if LOG_LEVEL <= logging.DEBUG else logging.WARNING)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
# =============================================================================
# CONFIG
# =============================================================================
MAX_DEPTHS = [3]        # one sweep per MAX_DEPTH (rules of 1..MAX_DEPTH conditions per side, as in the backtest), each
                        # with its ranking (RANK_BY). The table goes by the worst of each terna's positions
N_WINNERS  = 2          # blocks printed: the first N of the table, each at its DEPTH (ternas with 0 rules are skipped)
RANK_BY    = "SYMBOLS"  # ranking of every MAX_DEPTH: "SYMBOLS" (symbols with rules, tie: rules) or "RULES" (the reverse)
SPLIT_MAX  = 5          # lines of every SELL_AFTER (SKIPPED; TERNAS and SYMBOLS in DEBUG): the first N, the rest as +n
TIMEFRAME = "4H"
DATASET   = "IS"

# Validated here, at import (do not edit)
RANK_BYS = {"SYMBOLS": ("symbols with rules", "rules"), "RULES": ("rules", "symbols with rules")}   # (key, tie)
if RANK_BY not in RANK_BYS:
    raise ValueError(f"RANK_BY must be one of {list(RANK_BYS)}: {RANK_BY}")

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


@contextmanager
def stepm_capture():

    cap = {}
    orig_gp = stepM_is.compute_global_pvalue

    def global_pvalue(*args, **kwargs):
        out = orig_gp(*args, **kwargs)
        cap["global_p"] = out["global_p"]
        return out

    stepM_is.compute_global_pvalue = global_pvalue
    try:
        yield cap
    finally:
        stepM_is.compute_global_pvalue = orig_gp


def sweep_symbol(combo_key: str, sym: str, arr: dict, rule_templates: list, param_grid: dict, bar, cap: dict) -> dict:

    rules = build_rule_dicts(rule_templates, combo_key, TIMEFRAME)
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
    n_pass, p_glob, skipped = {}, {}, []
    for k, p in enumerate(ternas, start=1):
        cid = _combo_id(p)
        bar.set_postfix_str(f"{sym} ── StepM terna {k}/{len(ternas)}")
        logger.debug(f"\n── TERNA {k}/{len(ternas)} ── {combo_key} {[sym]} ── "
                     f"SELL_AFTER={p['SELL_AFTER']} TP_PCT={p['TP_PCT']} SL_PCT={p['SL_PCT']}")
        cols = np.asarray([i for i, s in enumerate(suffix) if s == cid], dtype=np.int64)
        sub  = matrix_arr[:, cols]
        sub  = np.ascontiguousarray(sub[np.any(sub != 0, axis=1)])
        cap.clear()
        res  = pipe_stepm(raw_results=raw_results, matrix_arr=sub, col_names=[col_names[i] for i in cols],
                          timeframe=TIMEFRAME)
        if any(r["passed_mbias"] and r["stepm_p"] is None for r in res):
            n_pass[cid] = 0
            skipped.append(cid)
            logger.debug(f"STEPM SKIPPED  {TIMEFRAME}: {sym} SELL_AFTER={p['SELL_AFTER']} TP_PCT={p['TP_PCT']} "
                         f"SL_PCT={p['SL_PCT']} counted as 0 rules in the sweep")
        else:
            n_pass[cid] = sum(bool(r["passed_mbias"]) for r in res)
        ran = "global_p" in cap
        if not ran and any(r.get("stepm_p") is not None for r in res):
            raise RuntimeError("StepM ran but its global p-value was not captured: update stepm_capture")
        p_glob[cid] = cap["global_p"] if ran else None
    return {"pass": n_pass, "p": p_glob, "skipped": skipped}

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
               "symbols": by_sa[sa]["symbols"], "total": {}, "n_sym": {}, "n_eff": {}, "with_rules": {}}
              for sa in sas for p in _combo_grid(grids[sa])]
    debug  = logger.isEnabledFor(logging.DEBUG)
    with pipeline_bars(show=debug), bank_selection(), stepm_capture() as cap:
        for depth in MAX_DEPTHS:                                        # the first MAX_DEPTH complete, then the next
            for k, sa in enumerate(sas, start=1):
                top, symbols = by_sa[sa]["top"], by_sa[sa]["symbols"]
                SELECTED_INDICATORS_BY_TIMEFRAME[TIMEFRAME] = list(top)   # the rules of this SELL_AFTER: its own TOP
                rule_templates = build_rule_templates(loaded[symbols[0]][1], TIMEFRAME, depth)
                log_sell_after_header(depth, sa, k, len(sas), len(symbols), len(top), len(rule_templates))
                bar = tqdm(symbols, desc=f"{f'GRIDS {DATASET} SA={sa}':<16}{TIMEFRAME}", dynamic_ncols=True,
                           disable=debug, file=sys.stdout)
                by_sym = {sym: sweep_symbol(f"{TIMEFRAME}_c{i:02d}", sym, loaded[sym][1], rule_templates, grids[sa],
                                            bar, cap)
                          for i, sym in enumerate(bar, start=1)}
                bar.close()
                log_sell_after(depth, sa, _combo_grid(grids[sa]), by_sym)
                for r in rows:
                    if r["sa"] == sa:
                        per_sym = [by_sym[sym]["pass"][r["cid"]] for sym in symbols]
                        r["total"][depth] = sum(per_sym)
                        r["n_sym"][depth] = sum(v > 0 for v in per_sym)
                        r["n_eff"][depth] = sum(per_sym) ** 2 / sum(v * v for v in per_sym) if sum(per_sym) else 0.0
                        r["with_rules"][depth] = [sym for sym, v in zip(symbols, per_sym) if v > 0]
    log_ranking(rows)

# =============================================================================
# PRINT
# =============================================================================
SEP = "─" * 115
PASTE_KEY_INDENT  = " " * 4    # paste-ready dicts: the timeframe key
PASTE_ITEM_INDENT = " " * 8    # paste-ready dicts: the items of the timeframe


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
    logger.info(f"  RANKING     : RANK_BY={RANK_BY} ({RANK_BYS[RANK_BY][0]}, tie: {RANK_BYS[RANK_BY][1]})")
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
    by_terna  = {lbl: sum(res["pass"][c] for res in by_sym.values()) for lbl, c in zip(labels, cids)}
    sym_terna = {lbl: sum(res["pass"][c] > 0 for res in by_sym.values()) for lbl, c in zip(labels, cids)}
    by_symbol = {sym: sum(res["pass"].values()) for sym, res in by_sym.items()}
    skipped   = [f"{sym} {lbl}" for sym, res in by_sym.items() for lbl, c in zip(labels, cids) if c in res["skipped"]]
    n_terna   = sum(v > 0 for v in by_terna.values())
    n_symbol  = sum(v > 0 for v in by_symbol.values())
    logger.debug(f"{'TERNAS (TP/SL)':<16}{TIMEFRAME}: {_counts_text(by_terna)} ── {n_terna}/{len(by_terna)} with rules")
    logger.debug(f"{'SYMBOLS':<16}{TIMEFRAME}: {_counts_text(by_symbol)} ── {n_symbol}/{len(by_symbol)} with rules")

    tag = f"SA={sa}"
    ps  = [res["p"][c] for res in by_sym.values() for c in cids if res["p"][c] is not None]
    if ps:
        n_sig = sum(p <= STEPM_ALPHA for p in ps)
        logger.info(f"{f'WHITE {tag}':<16}{TIMEFRAME}: p {min(ps):.4f} to {max(ps):.4f} ── "
                    f"{n_sig}/{len(ps)} with p ≤ {STEPM_ALPHA:g}")
    else:
        logger.info(f"{f'WHITE {tag}':<16}{TIMEFRAME}: -")
    logger.info(f"{f'RULES {tag}':<16}{TIMEFRAME}: {_fmt(sum(by_terna.values()))} in {n_terna}/{len(cids)} ternas ── "
                f"{n_symbol}/{len(by_sym)} symbols")
    if RANK_BY == "SYMBOLS":
        best = min(labels, key=lambda lbl: (-sym_terna[lbl], -by_terna[lbl]))
    else:
        best = min(labels, key=lambda lbl: (-by_terna[lbl], -sym_terna[lbl]))
    if by_terna[best] > 0:
        logger.info(f"{f'BEST {tag}':<16}{TIMEFRAME}: {best} ── {_fmt(by_terna[best])} rules ── "
                    f"{sym_terna[best]}/{len(by_sym)} symbols")
    else:
        logger.info(f"{f'BEST {tag}':<16}{TIMEFRAME}: -")
    if skipped:
        logger.info(f"{f'SKIPPED {tag}':<16}{TIMEFRAME}: {len(skipped)} ── {_capped(skipped, ', ')}")

def _capped(parts: list, sep: str = " | ") -> str:
    if not parts:
        return "-"
    text = sep.join(parts[:SPLIT_MAX])
    return text + (f" +{len(parts) - SPLIT_MAX}" if len(parts) > SPLIT_MAX else "")


def _counts_text(counts: dict) -> str:
    ranked = sorted(((k, v) for k, v in counts.items() if v > 0), key=lambda kv: -kv[1])
    return _capped([f"{k}={v}" for k, v in ranked])

def _log_timeframe_array(name: str, items: list, comment: str = "") -> None:
    if comment:
        logger.info(f"# {comment}")
    logger.info(f"{name} = {{")
    logger.info(f'{PASTE_KEY_INDENT}"{TIMEFRAME}": [')
    for it in items:
        logger.info(f'{PASTE_ITEM_INDENT}"{it}",')
    logger.info(f"{PASTE_KEY_INDENT}],")
    logger.info("}")


def _log_param_grid(r: dict) -> None:
    logger.info("PARAM_GRID_BY_TIMEFRAME = {")
    logger.info(f'{PASTE_KEY_INDENT}"{TIMEFRAME}": {{')
    logger.info(f'{PASTE_ITEM_INDENT}"SELL_AFTER": [{r["sa"]}],')
    logger.info(f'{PASTE_ITEM_INDENT}"TP_PCT":     [{r["tp"]}],')
    logger.info(f'{PASTE_ITEM_INDENT}"SL_PCT":     [{r["sl"]}],')
    logger.info(f"{PASTE_KEY_INDENT}}},")
    logger.info("}")


def _sym_text(r: dict, d: int) -> str:
    return f"{r['n_sym'][d]}/{len(r['symbols'])}"


def _rank_key(r: dict, d: int) -> tuple:
    """Ranking of MAX_DEPTH d by RANK_BY, the other one as tie, then the smaller SELL_AFTER (then grid order)."""
    if RANK_BY == "SYMBOLS":
        return -r["n_sym"][d], -r["total"][d], r["sa"]
    return -r["total"][d], -r["n_sym"][d], r["sa"]


def _best_depth(r: dict) -> int:
    """MAX_DEPTH of a terna: the one with most symbols with rules (tie: the smaller)."""
    return max(sorted(MAX_DEPTHS), key=lambda d: r["n_sym"][d])


def log_ranking(rows: list) -> None:

    for d in MAX_DEPTHS:
        for pos, r in enumerate(sorted(rows, key=lambda r: _rank_key(r, d)), start=1):
            r.setdefault("rank", {})[d] = pos
    rows = sorted(rows, key=lambda r: (max(r["rank"][d] for d in MAX_DEPTHS), min(r["rank"][d] for d in MAX_DEPTHS),
                                       r["sa"]))
    for pos, r in enumerate(rows, start=1):
        r["pos"], r["depth"] = pos, _best_depth(r)

    def block(r, d):                                                # rank, rules, symbols, n_eff
        return f"  │  {r['rank'][d]:>4}  {r['total'][d]:>6}  {_sym_text(r, d):>8}  {r['n_eff'][d]:>6.1f}"

    lead = f"  {'#':>3}  {'SA':>5}  {'TP':>6}  {'SL':>6}  {'DEPTH':>5}"
    head = f"  │  {'#':>4}  {'RULES':>6}  {'SYMBOLS':>8}  {'N_EFF':>6}"
    key, tie = RANK_BYS[RANK_BY]
    logger.info(f"\n{SEP}")
    logger.info(f"  RANKING ── rules passing StepM IS, summed over the SYMBOL_POOL of each SELL_AFTER ── RANK_BY={RANK_BY}")
    logger.info(f"{SEP}")
    logger.info(" " * len(lead) + "".join(f"  │{f'MAX_DEPTH={d}':^{len(head) - 3}}" for d in MAX_DEPTHS))
    logger.info(lead + head * len(MAX_DEPTHS))
    for r in rows:
        logger.info(f"  {r['pos']:>3}  {r['sa']:>5}  {r['tp']:>6}  {r['sl']:>6}  {r['depth']:>5}"
                    + "".join(block(r, d) for d in MAX_DEPTHS))
    logger.debug(f"  #: the worst of the terna's # in each MAX_DEPTH (tie: the best, then smaller SELL_AFTER)")
    logger.debug(f"  # of each MAX_DEPTH: its ranking by {key} (tie: {tie}, then smaller SELL_AFTER)")
    logger.debug(f"  DEPTH: the MAX_DEPTH with most symbols with rules (tie: the smaller), the one of the winners")
    logger.debug(f"  N_EFF: effective number of symbols with rules, (sum of rules)² / sum of (rules per symbol)² "
                 f"── 1 = every rule in one symbol")
    logger.info(f"{SEP}")
    log_winner(rows)


def log_winner(rows: list) -> None:
    """The first N_WINNERS ternas of the table, one block each, at their DEPTH. rows: in the order of the table."""
    winners = [r for r in rows[:N_WINNERS] if r["total"][r["depth"]] > 0]
    if not winners:
        logger.info(f"  WINNERS: no terna has rules passing StepM IS")
        logger.info(f"{SEP}")
        return
    for r in winners:
        d = r["depth"]
        logger.info(f"  #{r['pos']} (MAX_DEPTH={d}): {r['total'][d]} rules ── symbols with rules {_sym_text(r, d)} "
                    f"── N_EFF {r['n_eff'][d]:.1f}")
        logger.info(f"")
        _log_timeframe_array("SYMBOL_POOL_BY_TIMEFRAME", r["symbols"], "SYMBOL_POOL of the json")
        logger.info(f"")
        _log_timeframe_array("SYMBOL_POOL_BY_TIMEFRAME", r["with_rules"][d], "with rules in this terna")
        logger.info(f"")
        _log_param_grid(r)
        logger.info(f"")
        _log_timeframe_array("SELECTED_INDICATORS_BY_TIMEFRAME", r["top"])
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