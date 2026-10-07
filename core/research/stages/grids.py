# core/research/stages/grids.py
import sys
import logging
from contextlib import contextmanager
from functools import partial
import numpy as np
from tqdm import tqdm

from setup import config_research as cr
from research import artifacts as ra
from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from setup.config_core import settings
from utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_day
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from pipeline import signal_cleaning, backtest_runner, stepM_is
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.backtest_runner import pipe_backtesting, _combo_grid, _combo_id
from pipeline.stepM_is import pipe_stepm, STEPM_ALPHA
# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = logging.INFO    # INFO: a header, a bar and a summary per SELL_AFTER. DEBUG: every symbol and terna as the backtest
logger    = logging.getLogger("BOT_research.grid")

# =============================================================================
# MODULE CONFIG
# =============================================================================
MAX_ROWS       = 5
SPLIT_MAX      = 5

RANKING = "rules (tie: symbols with rules, then the smaller SELL_AFTER)"     # the winner terna, the best of all


def configure_loggers() -> None:
    logger.setLevel(LOG_LEVEL)
    logging.getLogger("BOT_batch").setLevel(logging.INFO if LOG_LEVEL <= logging.DEBUG else logging.WARNING)
    for noisy_logger in ("joblib", "matplotlib", "numba"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# =============================================================================
# DATA
# =============================================================================
def load_symbol(timeframe: str, sym: str) -> tuple:
    """One independent universe per symbol, as load_ohlcv_by_combo of main_back_fx: (data, arrays)."""
    data = build_universe(DATA_FOLDER_BY_DATASET[cr.DATASET], {timeframe: [sym]}, dataset=cr.DATASET)[timeframe]
    return data, prepare_ohlcv_arrays(data)

# =============================================================================
# SWEEP
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


def sweep_symbol(timeframe: str, combo_key: str, sym: str, arr: dict, rule_templates: list, param_grid: dict, bar,
                 cap: dict) -> dict:

    rules = build_rule_dicts(rule_templates, combo_key, timeframe)
    logger.debug(f"\n\033[36m{'─' * 70}")
    logger.debug(f"─ RULE MINING ── {combo_key} {[sym]} ── rules: {_fmt(len(rules))}")
    logger.debug(f"{'─' * 70}\033[0m")

    bar.set_postfix_str(f"{sym} ── Jaccard")
    rules = pipe_signal_cleaning_jaccard(rules=rules, ohlcv_arr=arr, timeframe=timeframe)
    bar.set_postfix_str(f"{sym} ── backtest")
    raw_results, _, matrix_arr, col_names = pipe_backtesting(
        rules=rules, ohlcv_arr=arr, param_grid=param_grid, order_amount=ORDER_AMOUNT, timeframe=timeframe,
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
                          timeframe=timeframe)
        if any(r["passed_mbias"] and r["stepm_p"] is None for r in res):
            n_pass[cid] = 0
            skipped.append(cid)
            logger.debug(f"STEPM SKIPPED  {timeframe}: {sym} SELL_AFTER={p['SELL_AFTER']} TP_PCT={p['TP_PCT']} "
                         f"SL_PCT={p['SL_PCT']} counted as 0 rules in the sweep")
        else:
            n_pass[cid] = sum(bool(r["passed_mbias"]) for r in res)
        ran = "global_p" in cap
        if not ran and any(r.get("stepm_p") is not None for r in res):
            raise RuntimeError("StepM ran but its global p-value was not captured: update stepm_capture")
        p_glob[cid] = cap["global_p"] if ran else None
    return {"pass": n_pass, "p": p_glob, "skipped": skipped}

# =============================================================================
# RUN
# =============================================================================
def _grids_inputs(tfs: list, inputs: dict | None) -> dict:
    if inputs is None:
        return {tf: ra.load_grids_input(tf, settings.BACKTEST_MODE) for tf in tfs}
    missing = [tf for tf in tfs if tf not in inputs]
    if missing:
        raise ValueError(f"Screen output has no timeframes: {missing}")
    return {tf: ra.check_grids_input(inputs[tf], tf, settings.BACKTEST_MODE) for tf in tfs}


def run(inputs: dict | None = None) -> dict:
    configure_loggers()
    tfs  = _ordered_timeframes(cr.TIMEFRAMES)
    docs = _grids_inputs(tfs, inputs)                                  # all the screenings first: fail before the run
    log_run_config(docs)
    winners_by_tf = {}
    debug  = logger.isEnabledFor(logging.DEBUG)
    with pipeline_bars(show=debug), stepm_capture() as cap:
        for tf in tfs:
            doc   = docs[tf]
            by_sa = {int(sa): o for sa, o in doc["by_sell_after"].items()}
            for sa in sorted(by_sa):
                if not by_sa[sa]["top"] or not by_sa[sa]["symbols"]:
                    logger.warning(f"  {tf} SELL_AFTER={sa}: empty TOP or SYMBOL_POOL, skipped")
            sas = [sa for sa in sorted(by_sa) if by_sa[sa]["top"] and by_sa[sa]["symbols"]]
            symbols_all = list(dict.fromkeys(sym for sa in sas for sym in by_sa[sa]["symbols"]))
            loaded = {sym: load_symbol(tf, sym) for sym in symbols_all}     # all the data first, as main_back_fx
            grids  = {sa: {"SELL_AFTER": [sa], "TP_PCT": doc["tp_pct"], "SL_PCT": doc["sl_pct"]} for sa in sas}
            rows   = [{"tf": tf, "sa": sa, "tp": p["TP_PCT"], "sl": p["SL_PCT"], "cid": _combo_id(p),
                       "top": by_sa[sa]["top"], "symbols": by_sa[sa]["symbols"], "total": 0, "n_sym": 0,
                       "n_eff": 0.0}
                      for sa in sas for p in _combo_grid(grids[sa])]
            for k, sa in enumerate(sas, start=1):
                top, symbols = by_sa[sa]["top"], by_sa[sa]["symbols"]
                rule_templates = build_rule_templates(loaded[symbols[0]][1], indicators=top, max_depth=RULE_MAX_DEPTH)
                log_sell_after_header(tf, sa, k, len(sas), len(symbols), len(top), len(rule_templates))
                bar = tqdm(symbols, desc=f"{f'GRIDS {cr.DATASET} SA={sa}':<16}{tf}", dynamic_ncols=True,
                           disable=debug, file=sys.stdout)
                by_sym = {sym: sweep_symbol(tf, f"{tf}_c{i:02d}", sym, loaded[sym][1], rule_templates,
                                            grids[sa], bar, cap)
                          for i, sym in enumerate(bar, start=1)}
                bar.close()
                log_sell_after(tf, sa, _combo_grid(grids[sa]), by_sym)
                for r in rows:
                    if r["sa"] == sa:
                        per_sym = [by_sym[sym]["pass"][r["cid"]] for sym in symbols]
                        r["total"] = sum(per_sym)
                        r["n_sym"] = sum(v > 0 for v in per_sym)
                        r["n_eff"] = sum(per_sym) ** 2 / sum(v * v for v in per_sym) if sum(per_sym) else 0.0
            winners_by_tf[tf] = log_ranking(tf, rows)
    return save_output(winners_by_tf)

# =============================================================================
# PRINT
# =============================================================================
SEP = "─" * 115


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _label(tp, sl) -> str:
    return f"{tp}/{sl}"


def _ordered_timeframes(timeframes: list) -> list:
    return sorted(timeframes, key=get_bars_per_day, reverse=True)    # smallest timeframe first


def log_run_config(docs: dict) -> None:
    logger.info(f"\n{SEP}")
    logger.info("  GRIDS START")
    logger.info(f"{SEP}")
    for tf, doc in docs.items():
        by_sa = {int(sa) for sa in doc["by_sell_after"]}
        sel = doc["selection"]
        gn  = " | ".join(f"SA={sa}: " + " ".join(f"{kind} {'-' if g is None else g}" for kind, g in o["group_n"].items())
                         for sa, o in doc["by_sell_after"].items())
        logger.info(f"  {'SCREENING':<12}: {tf} ── {ra.grids_path(tf)} (created {doc['created']})")
        logger.info(f"  {'PARAM GRID':<12}: {tf} ── SELL_AFTER={sorted(by_sa)} TP_PCT={doc['tp_pct']} SL_PCT={doc['sl_pct']}")
        logger.info(f"  {'SELECTION':<12}: {tf} ── LUCK_MAX={sel['luck_max']:.0%} TOP_I={sel['top_i']} ── GROUP_N {gn}")
    first = next(iter(docs.values()))
    logger.info(f"  {'DATASET':<12}: {cr.DATASET} ── {list(docs)}")
    logger.info(f"  {'MODE':<12}: {first['mode']} (screening) = {settings.BACKTEST_MODE} (BACKTEST_MODE)")
    logger.info(f"  {'RULES':<12}: MAX_DEPTH={RULE_MAX_DEPTH}")
    logger.info(f"  {'RANKING':<12}: {RANKING}")
    logger.info(f"{SEP}\n")


def log_sell_after_header(tf: str, sa: int, k: int, n_sa: int, n_sym: int, n_ind: int, n_rules: int) -> None:
    text = (f"  {tf} ── MAX_DEPTH={RULE_MAX_DEPTH} ── SELL_AFTER={sa} ({k}/{n_sa}) ── symbols: {n_sym} ── indicators: {n_ind} ── "
            f"rules: {_fmt(n_rules)} per symbol")
    logger.info(f"\n{'─' * (len(text) + 2)}\n{text}\n{'─' * (len(text) + 2)}")


def log_sell_after(tf: str, sa: int, ternas: list, by_sym: dict) -> None:

    labels = [_label(p["TP_PCT"], p["SL_PCT"]) for p in ternas]
    cids   = [_combo_id(p) for p in ternas]
    w      = max(9, max(len(s) for s in labels) + 2)
    logger.debug(f"\n{SEP}")
    logger.debug(f"  {tf} ── MAX_DEPTH={RULE_MAX_DEPTH} ── SELL_AFTER={sa} ── rules passing StepM IS per terna (TP/SL)")
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
    logger.debug(f"{'TERNAS (TP/SL)':<16}{tf}: {_counts_text(by_terna)} ── {n_terna}/{len(by_terna)} with rules")
    logger.debug(f"{'SYMBOLS':<16}{tf}: {_counts_text(by_symbol)} ── {n_symbol}/{len(by_symbol)} with rules")

    tag = f"SA={sa}"
    ps  = [res["p"][c] for res in by_sym.values() for c in cids if res["p"][c] is not None]
    if ps:
        n_sig = sum(p <= STEPM_ALPHA for p in ps)
        logger.info(f"{f'WHITE {tag}':<16}{tf}: p {min(ps):.4f} to {max(ps):.4f} ── "
                    f"{n_sig}/{len(ps)} with p ≤ {STEPM_ALPHA:g}")
    else:
        logger.info(f"{f'WHITE {tag}':<16}{tf}: -")
    logger.info(f"{f'RULES {tag}':<16}{tf}: {_fmt(sum(by_terna.values()))} in {n_terna}/{len(cids)} ternas ── "
                f"{n_symbol}/{len(by_sym)} symbols")
    best = min(labels, key=lambda lbl: (-by_terna[lbl], -sym_terna[lbl]))      # as _rank_key
    if by_terna[best] > 0:
        logger.info(f"{f'BEST {tag}':<16}{tf}: {best} ── {_fmt(by_terna[best])} rules ── "
                    f"{sym_terna[best]}/{len(by_sym)} symbols")
    else:
        logger.info(f"{f'BEST {tag}':<16}{tf}: -")
    if skipped:
        logger.info(f"{f'SKIPPED {tag}':<16}{tf}: {len(skipped)} ── {_capped(skipped, ', ')}")

def _capped(parts: list, sep: str = " | ") -> str:
    if not parts:
        return "-"
    text = sep.join(parts[:SPLIT_MAX])
    return text + (f" +{len(parts) - SPLIT_MAX}" if len(parts) > SPLIT_MAX else "")


def _counts_text(counts: dict) -> str:
    ranked = sorted(((k, v) for k, v in counts.items() if v > 0), key=lambda kv: -kv[1])
    return _capped([f"{k}={v}" for k, v in ranked])

def _sym_text(r: dict) -> str:
    return f"{r['n_sym']}/{len(r['symbols'])}"


def _rank_key(r: dict) -> tuple:
    """Rules, then symbols with rules, then the smaller SELL_AFTER (stable sort: then grid order)."""
    return -r["total"], -r["n_sym"], r["sa"]


def log_ranking(tf: str, rows: list) -> dict | None:
    """The ternas of every SELL_AFTER ranked by rules. Returns the winner, None if no terna has rules."""
    rows = sorted(rows, key=_rank_key)
    for pos, r in enumerate(rows, start=1):
        r["pos"] = pos

    lead = f"  {'#':>3}  {'SA':>5}  {'TP':>6}  {'SL':>6}"
    head = f"  │{'RULES':>7}{'SYMBOLS':>9}{'N_EFF':>7}"
    logger.info(f"\n{SEP}")
    logger.info(f"  RANKING {tf} ── MAX_DEPTH={RULE_MAX_DEPTH} ── rules passing StepM IS, summed over the SYMBOL_POOL of each SELL_AFTER")
    logger.info(f"{SEP}")
    logger.info(lead + head)
    for r in rows[:MAX_ROWS]:
        logger.info(f"  {r['pos']:>3}  {r['sa']:>5}  {r['tp']:>6}  {r['sl']:>6}"
                    f"  │{r['total']:>7}{_sym_text(r):>9}{r['n_eff']:>7.1f}")
    logger.debug(f"  #: ranking by {RANKING}")
    logger.debug(f"  N_EFF: effective number of symbols with rules, (sum of rules)² / sum of (rules per symbol)² "
                 f"── 1 = every rule in one symbol")
    logger.info(f"{SEP}")
    return log_winner(tf, rows)


def log_winner(tf: str, rows: list) -> dict | None:
    """The first terna of the table, its summary line. rows: in the order of the table. Returns the winner, None
    if it has no rules (then no terna has)."""
    r = rows[0] if rows and rows[0]["total"] > 0 else None
    if r is None:
        logger.info(f"  WINNER {tf}: no terna has rules passing StepM IS")
    else:
        logger.info(f"  {tf} #{r['pos']}: {r['total']} rules ── symbols with rules {_sym_text(r)} "
                    f"── N_EFF {r['n_eff']:.1f}")
    logger.info(f"{SEP}")
    return r


# =============================================================================
# OUTPUT
# =============================================================================
def _winner_entry(r: dict) -> dict:
    return {
        "symbols":    list(r["symbols"]),                                        # the SYMBOL_POOL of its SELL_AFTER
        "param_grid": {"SELL_AFTER": [r["sa"]], "TP_PCT": [r["tp"]], "SL_PCT": [r["sl"]]},
        "indicators": list(r["top"]),
    }


def save_output(winners_by_tf: dict) -> dict:
    by_tf = {tf: _winner_entry(w) for tf, w in winners_by_tf.items() if w}
    meta  = {"mode": settings.BACKTEST_MODE, "max_depth": RULE_MAX_DEPTH}
    doc   = ra.save_combos_input(by_tf, meta)
    logger.info("")
    for tf, e in by_tf.items():
        g = e["param_grid"]
        sym_lbl, ind_lbl = f"SYMBOLS ({len(e['symbols'])})", f"INDICATORS ({len(e['indicators'])})"
        w = max(len(sym_lbl), len(ind_lbl), len("PARAM_GRID"))
        logger.info(f"  {tf:<4}{'PARAM_GRID':<{w}} : SELL_AFTER={g['SELL_AFTER']} TP_PCT={g['TP_PCT']} SL_PCT={g['SL_PCT']}")
        logger.info(f"  {'':<4}{sym_lbl:<{w}} : {', '.join(e['symbols'])}")
        logger.info(f"  {'':<4}{ind_lbl:<{w}} : {', '.join(e['indicators'])}")
    for tf in (tf for tf, win in winners_by_tf.items() if not win):
        logger.warning(f"  {tf:<4}no winner: not saved, the combos stage cannot run this timeframe")
    logger.info(f"{SEP}")
    logger.info(f"✅  GRIDS OUTPUT ── {ra.combos_path()}")
    logger.info(f"{SEP}")
    return doc