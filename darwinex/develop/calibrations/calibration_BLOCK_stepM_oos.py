#research/calibrations/calibration_BLOCK_sM_oos_size.py
"""
Block size of the StepM OOS bootstrap (STEPM_OOS_BLOCK_SIZE), checked by the size of the test, as
calibration_BLOCK_sM_size.py does for StepM IS (Romano & Wolf 2005, section 7, Algorithm 7.1). Synthetic OOS data
where no rule has an edge goes through the real WFO and the real StepM OOS statistics once per candidate block. With a
well sized block both p-values are uniform: about 10% of the matrices at p <= 0.10.
  - rules          : generated and Jaccard-cleaned on IS as production (>= BACKTEST_MIN_TRADES IS signals), sampled
  - signals        : computed once on the real OOS data and replayed, by timestamp, on every synthetic path and window
  - synthetic OOS  : block permutation of the real OOS candles (multiverse.py), one layout shared by the symbols of a
                     group, drift removed so the price is a martingale
  - costs          : COMISION = 0 in every module and worker, so the true Sharpe of every column is exactly 0
  - WFO            : the real pipe_wfo, SELL_AFTER blocked by best_combo_id, TP/SL optimized per window (production)
  - matrix         : WFO test trades of the rules of GROUP_SIZE symbols pooled (production pools every combo of the
                     timeframe), rules with >= STEPM_OOS_MIN_TRADES test trades, same code as stepM_oos
  - tests          : WHITE RC (raw Sharpe max) and EXCEEDANCE COUNT, same computation as stepM_oos
  - null check     : mean t of the columns by side must be ~0 (costs or drift left would show here)
A longer block is expected to be more permissive, so the recommended block is the largest one before the first block
where either test is permissive (blocks scanned from the shortest).
"""
import os
import sys
import math
import time
import logging
from functools import partial
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "core")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.stepm_oos_block_size_check")
for noisy_logger in ("BOT_batch.pipeline.backtest_runner", "BOT_batch.pipeline.stepM_is", "BOT_batch.pipeline.stepM_oos",
                     "BOT_batch.pipeline.wfo", "BOT_batch.engines.wfo_WF", "BOT_batch.pipeline.multiverse",
                     "joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from utils.ohlcv_utils import prepare_ohlcv_arrays
from signals.indicators_bank import ConditionBank
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from pipeline import backtest_runner as backtest_module
from pipeline import wfo as wfo_module
from pipeline import stepM_is as stepm_module
from pipeline.backtest_runner import _combo_id
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.wfo import pipe_wfo
from pipeline.stepM_is import compute_bootstrap_null, compute_global_pvalue
from pipeline.stepM_oos import build_wfo_test_matrix, _raw_deviations, STEPM_OOS_ALPHA, STEPM_OOS_MIN_TRADES
from pipeline.multiverse import _build_path_bundle, _build_synthetic_ohlcv_arr, _COL_LOG_RET_CLOSE

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES = ["4H", "1H"]
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]
SYMBOLS_BY_TIMEFRAME = {
    "4H": SYMBOLS,
    "1H": SYMBOLS,
}
GROUP_SIZE = 5              # symbols pooled in one StepM OOS matrix (production pools every combo of the timeframe)
SELL_AFTER = [20, 40, 100]  # one WFO + one table block per SELL_AFTER (every rule gets it through best_combo_id)
TP_PCT     = [0.5, 1.0, 1.5]
SL_PCT     = [0.5, 1.0, 1.5]

BLOCKS = [1, 2, 3, 5, 10, 15, 20]              # candidate STEPM_OOS_BLOCK_SIZE, in rows of the OOS matrix (days)
# permutation block of the synthetic prices: at least as long as the longest StepM block tested
PATH_BLOCK_DAYS = max(BLOCKS)
CANDLES_PER_DAY = {"4H": 6, "1H": 24}

RULES_PER_SYMBOL = 100      # rules sampled per symbol (a group matrix has up to GROUP_SIZE x this columns)
N_PATHS          = 50       # synthetic paths per group: 4 groups x 50 = 200 p-values per test (run 2 first to time it)
N_BOOTSTRAP      = stepm_module.WHITE_N_BOOTSTRAP
ALPHA            = STEPM_OOS_ALPHA
MIN_TRADES       = STEPM_OOS_MIN_TRADES
TAIL_SIGNIF      = 0.05     # a share is "clearly" off alpha when its binomial p is below this
NULL_T_TH        = 0.05     # |mean t| of a side above this: the true Sharpe is not 0 (costs or drift left)
RANDOM_SEED      = 42

# =============================================================================
# NO COSTS: COMISION = 0 in every loaded module that holds it, here and in every worker (loky initializer)
# =============================================================================
def _zero_costs() -> None:
    import importlib
    from setup.config_core import settings as core_settings
    import setup.config_backtest                                                   # noqa: F401
    importlib.import_module("pipeline.backtest_runner")
    importlib.import_module("pipeline.wfo")
    importlib.import_module(f"backtesters.ZX_compute_BT_{core_settings.BACKTEST_MODE}")
    for module in list(sys.modules.values()):
        attrs = getattr(module, "__dict__", None)
        if attrs is not None and "COMISION" in attrs:
            attrs["COMISION"] = 0.0


def _commissions() -> list:
    return [(name, float(module.__dict__["COMISION"])) for name, module in list(sys.modules.items())
            if getattr(module, "__dict__", None) is not None and "COMISION" in module.__dict__]


def check_zero_costs() -> None:
    found = _commissions() + [item for batch in Parallel(n_jobs=wfo_module.RULES_N_JOBS)(
        delayed(_commissions)() for _ in range(64)) for item in batch]
    bad = sorted({name for name, value in found if value != 0.0})
    if bad:
        raise RuntimeError(f"COMISION is not 0 in {bad}: the true Sharpe would not be 0")
    logger.info(f"COSTS: COMISION = 0 in {sorted({name for name, _ in found})}")

# =============================================================================
# RULES: generated on IS as production, real OOS signals replayed by timestamp (the WFO slices the data in windows)
# =============================================================================
def make_fixed_signal_fn(ts_ns: np.ndarray, signal: np.ndarray):

    def signal_fn(arr: dict, live_trading: bool = True, bank=None, **_) -> np.ndarray:
        ts  = np.asarray(arr["ts"]).astype("datetime64[ns]").astype(np.int64)
        pos = np.searchsorted(ts_ns, ts)
        if pos.size and (pos[-1] >= ts_ns.size or not np.array_equal(ts_ns[pos], ts)):
            raise ValueError("synthetic candles do not match the real OOS timestamps")
        return signal[pos]

    return signal_fn


def load_is_arrays(sym: str, timeframe: str) -> dict:
    data_is = build_universe(DATA_FOLDER_BY_DATASET["IS"], {timeframe: [sym]}, dataset="IS")[timeframe]
    return prepare_ohlcv_arrays(data_is)


def build_rules(rule_templates: list, sym: str, timeframe: str, sym_idx: int) -> tuple:
    combo_key = f"{timeframe}_{sym}"
    arr_is    = load_is_arrays(sym, timeframe)
    rules     = build_rule_dicts(rule_templates, combo_key, timeframe)
    rules     = pipe_signal_cleaning_jaccard(rules=rules, ohlcv_arr=arr_is, timeframe=timeframe)

    bank_is    = ConditionBank(arr_is[sym])
    min_trades = backtest_module.BACKTEST_MIN_TRADES
    eligible   = [r for r in rules
                  if np.count_nonzero(r["signal_fn"](arr_is[sym], live_trading=False, bank=bank_is)) >= min_trades]
    rng  = np.random.default_rng(RANDOM_SEED + sym_idx)
    pick = np.sort(rng.choice(len(eligible), size=min(RULES_PER_SYMBOL, len(eligible)), replace=False))
    return [eligible[i] for i in pick], len(rules), len(eligible)


def fix_oos_signals(rules: list, arr_oos: dict) -> list:
    bank  = ConditionBank(arr_oos)
    ts_ns = np.asarray(arr_oos["ts"]).astype("datetime64[ns]").astype(np.int64)
    fixed = []
    for r in rules:
        signal = np.asarray(r["signal_fn"](arr_oos, live_trading=False, bank=bank), dtype=np.int8)
        fixed.append({**r, "signal_fn": make_fixed_signal_fn(ts_ns, signal)})
    return fixed

# =============================================================================
# SYNTHETIC OOS PRICES: multiverse permutation shared by the group, mean gross return 1 per candle
# =============================================================================
def build_martingale_bundle(data: dict, timeframe: str) -> dict:
    bundle = _build_path_bundle(data, raw_columns=["volume"], timeframe=timeframe)
    for symbol_bundle in bundle["symbols"].values():
        log_ret = symbol_bundle["data_array"][:, _COL_LOG_RET_CLOSE]
        log_ret -= np.log(np.mean(np.exp(log_ret)))
    return bundle


def path_block_candles(timeframe: str) -> int:
    return PATH_BLOCK_DAYS * CANDLES_PER_DAY[timeframe]

# =============================================================================
# STEPM OOS P-VALUES: same computation as stepM_oos._evaluate_timeframe
# =============================================================================
def _build_weight_matrix_fast(
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    n_obs: int,
    n_replicas: int,
) -> np.ndarray:
    # same weights as stepM_is._build_bootstrap_weight_matrix (bincount instead of np.add.at)
    width = n_obs + 1
    base  = (np.arange(n_replicas, dtype=np.int64) * width)[:, None]
    up    = np.concatenate(((base + starts_full).ravel(), base[:, 0] + starts_last))
    down  = np.concatenate(((base + starts_full + block_size).ravel(), base[:, 0] + starts_last + len_last))
    diff  = np.bincount(up, minlength=n_replicas * width) - np.bincount(down, minlength=n_replicas * width)
    return np.cumsum(diff.reshape(n_replicas, width)[:, :n_obs], axis=1).astype(np.float32)


def oos_pvalues(matrix: np.ndarray, col_names: list, block_size: int, seed: int) -> tuple:
    # matrix only has columns with std > 0, so compute_bootstrap_null never compacts it in place: no copy needed
    null = compute_bootstrap_null(
        matrix, col_names, n_bootstrap=N_BOOTSTRAP, block_size=block_size, seed=seed, topm_size=matrix.shape[1],
    )
    if null.n_kept == 0:
        return np.nan, np.nan
    deviations = _raw_deviations(null)
    statistic  = null.real_sharpe

    row_max  = deviations.max(axis=1)                                              # WHITE RC
    white_p  = compute_global_pvalue(row_max, statistic)["global_p"]

    crit        = np.quantile(deviations, 1.0 - ALPHA, axis=0)                     # EXCEEDANCE COUNT
    real_count  = int((statistic > crit).sum())
    null_counts = (deviations > crit[None, :]).sum(axis=1)
    exceed_p    = float(np.mean(null_counts >= real_count))
    return white_p, exceed_p


def evaluate_matrix(wfo_rules: list, seed: int) -> dict:
    out = {"cols": 0, "days": 0, "p": {}, "t_side": {}}
    matrix, col_names = build_wfo_test_matrix(wfo_rules, MIN_TRADES)
    if matrix is None:
        return out
    keep      = matrix.std(axis=0) > 0
    matrix    = np.ascontiguousarray(matrix[:, keep])
    col_names = [c for c, k in zip(col_names, keep) if k]
    out["cols"], out["days"] = len(col_names), matrix.shape[0]
    if len(col_names) == 0:
        return out

    side_by_id = {r["rule_id"]: r["side"] for r in wfo_rules}                      # null check: mean t by side
    t_stat = (matrix.mean(axis=0, dtype=np.float64) / matrix.std(axis=0, ddof=1, dtype=np.float64)
              * math.sqrt(matrix.shape[0]))
    for side in ("long", "short"):
        mask = np.array([side_by_id[c] == side for c in col_names])
        if mask.any():
            out["t_side"][side] = float(t_stat[mask].mean())

    for block in BLOCKS:
        if matrix.shape[0] < 2 * block:                                            # same guard as stepM_oos
            continue
        out["p"][block] = oos_pvalues(matrix, col_names, block, seed)
    return out

# =============================================================================
# ONE PATH: synthetic OOS universe of the group, real WFO per SELL_AFTER, StepM OOS p-values per block
# =============================================================================
def run_path(rules_by_sym: dict, bundle: dict, path_idx: int, base_seed: int, boot_seed: int, timeframe: str) -> tuple:
    param_grid = {"SELL_AFTER": SELL_AFTER, "TP_PCT": TP_PCT, "SL_PCT": SL_PCT}
    synthetic, _layout = _build_synthetic_ohlcv_arr(bundle, path_idx, path_block_candles(timeframe), base_seed)

    out, t_wfo, t_boot = {}, 0.0, 0.0
    for sa in SELL_AFTER:
        t0 = time.time()
        best_combo_id = _combo_id({"SELL_AFTER": sa, "TP_PCT": TP_PCT[0], "SL_PCT": SL_PCT[0]})   # only SA is used
        wfo_rules = []
        for sym, rules in rules_by_sym.items():
            wfo_rules += pipe_wfo(
                rules        = [{**r, "best_combo_id": best_combo_id} for r in rules],
                ohlcv_arr    = {sym: synthetic[sym]},
                param_grid   = param_grid,
                order_amount = ORDER_AMOUNT,
                timeframe    = timeframe,
                combo_key    = f"{timeframe}_{sym}",
                log_level    = logging.WARNING,
            )
        t1 = time.time()
        out[sa] = evaluate_matrix(wfo_rules, boot_seed)
        t_wfo, t_boot = t_wfo + (t1 - t0), t_boot + (time.time() - t1)
    return out, (t_wfo, t_boot)

# =============================================================================
# SIZE OF THE TEST
# =============================================================================
def _binom_pmf(j: int, n: int, p: float) -> float:
    return math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                    + j * math.log(p) + (n - j) * math.log1p(-p))


def size_stats(p_values: np.ndarray) -> dict:
    p_values = p_values[np.isfinite(p_values)]
    n = len(p_values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "le": np.nan, "verdict": "-"}
    k_low = int((p_values <= ALPHA).sum())
    if sum(_binom_pmf(j, n, ALPHA) for j in range(k_low, n + 1)) < TAIL_SIGNIF:
        verdict = "permissive"
    elif sum(_binom_pmf(j, n, ALPHA) for j in range(0, k_low + 1)) < TAIL_SIGNIF:
        verdict = "conservative"
    else:
        verdict = "ok"
    return {"n": n, "mean": float(p_values.mean()), "le": k_low / n, "verdict": verdict}


def recommended_block(rows: list, sa: int):
    # largest block before the first one where either test is permissive, scanning from the shortest
    chosen = None
    for block in sorted(BLOCKS):
        row = next((r for r in rows if r["sa"] == sa and r["block"] == block), None)
        if row is None or row["white"]["n"] == 0 or "permissive" in (row["white"]["verdict"], row["exceed"]["verdict"]):
            break
        chosen = block
    return chosen

# =============================================================================
# ONE TIMEFRAME
# =============================================================================
def symbol_groups(symbols: list) -> list:
    return [symbols[i:i + GROUP_SIZE] for i in range(0, len(symbols), GROUP_SIZE)]


def run_timeframe(timeframe: str) -> None:
    symbols = SYMBOLS_BY_TIMEFRAME[timeframe]
    white   = {(sa, b): [] for sa in SELL_AFTER for b in BLOCKS}
    exceed  = {(sa, b): [] for sa in SELL_AFTER for b in BLOCKS}
    n_cols  = {sa: [] for sa in SELL_AFTER}
    n_days  = {sa: [] for sa in SELL_AFTER}
    t_side  = {(sa, side): [] for sa in SELL_AFTER for side in ("long", "short")}
    rule_templates = build_rule_templates(load_is_arrays(symbols[0], timeframe), timeframe, RULE_MAX_DEPTH)

    for g, group in enumerate(symbol_groups(symbols), start=1):
        data_oos, rules_by_sym = {}, {}
        for sym in group:
            rules, n_jaccard, n_eligible = build_rules(rule_templates, sym, timeframe, symbols.index(sym))
            data_oos[sym] = build_universe(DATA_FOLDER_BY_DATASET["OOS"], {timeframe: [sym]}, dataset="OOS")[timeframe][sym]
            arr_oos       = prepare_ohlcv_arrays({sym: data_oos[sym]})[sym]
            if rules:
                rules_by_sym[sym] = fix_oos_signals(rules, arr_oos)
            logger.info(f"{timeframe} G{g} {sym}: {len(rules)} rules (sampled from {n_eligible} with >= "
                        f"{backtest_module.BACKTEST_MIN_TRADES} IS signals, {n_jaccard} after Jaccard)")
        if not rules_by_sym:
            continue

        bundle    = build_martingale_bundle(data_oos, timeframe)
        base_seed = RANDOM_SEED + 1_000_003 * g
        n_blocks  = bundle["n_ref_rows"] / path_block_candles(timeframe)
        if n_blocks < 20:
            logger.warning(f"{timeframe} G{g}: only {n_blocks:.0f} permutation blocks: the synthetic paths will look alike")

        with tqdm(range(N_PATHS), desc=f"PATHS {timeframe} G{g} {'+'.join(group)}", dynamic_ncols=True) as bar:
            for path_idx in bar:
                out, (t_wfo, t_boot) = run_path(rules_by_sym, bundle, path_idx, base_seed,
                                                RANDOM_SEED + 1_000 * g + path_idx, timeframe)
                bar.set_postfix_str(f"wfo {t_wfo:.0f}s ── bootstrap {t_boot:.1f}s")
                for sa, res in out.items():
                    n_cols[sa].append(res["cols"])
                    n_days[sa].append(res["days"])
                    for side, t in res["t_side"].items():
                        t_side[(sa, side)].append(t)
                    for block, (p_white, p_exceed) in res["p"].items():
                        white[(sa, block)].append(p_white)
                        exceed[(sa, block)].append(p_exceed)

    rows = [{"sa": sa, "block": b,
             "cols": float(np.mean(n_cols[sa])) if n_cols[sa] else 0.0,
             "days": float(np.mean(n_days[sa])) if n_days[sa] else 0.0,
             "white":  size_stats(np.array(white[(sa, b)], dtype=float)),
             "exceed": size_stats(np.array(exceed[(sa, b)], dtype=float))}
            for sa in SELL_AFTER for b in BLOCKS]
    log_table(timeframe, rows, t_side)


def log_table(timeframe: str, rows: list, t_side: dict) -> None:
    width = 104
    logger.info(f"\n{'─' * width}")
    logger.info(f"  {timeframe} ── size of StepM OOS with no-edge rules (valid: ~{ALPHA:.0%} of the matrices at p <= {ALPHA:g})")
    logger.info(f"{'─' * width}")
    logger.info(f"  {'':>4}  {'':>5}  {'':>6}  {'':>5}  {'':>5}  {'WHITE RC':<30}  {'EXCEEDANCE COUNT':<30}")
    logger.info(f"  {'SA':>4}  {'BLOCK':>5}  {'COLS':>6}  {'DAYS':>5}  {'N':>5}  "
                f"{'MEAN_P':>7} {f'P<={ALPHA:g}':>8} {'VERDICT':<13}  {'MEAN_P':>7} {f'P<={ALPHA:g}':>8} {'VERDICT':<13}")
    previous = None
    for r in rows:
        if previous is not None and r["sa"] != previous:
            logger.info("")
        previous = r["sa"]
        w, e = r["white"], r["exceed"]
        logger.info(f"  {r['sa']:>4}  {r['block']:>5}  {r['cols']:>6.0f}  {r['days']:>5.0f}  {w['n']:>5}  "
                    f"{w['mean']:>7.3f} {w['le']:>8.1%} {w['verdict']:<13}  "
                    f"{e['mean']:>7.3f} {e['le']:>8.1%} {e['verdict']:<13}")
    logger.info(f"{'─' * width}")
    for sa in SELL_AFTER:
        parts = []
        for side in ("long", "short"):
            values = np.asarray(t_side[(sa, side)], dtype=float)
            if values.size:
                se   = values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else np.nan
                flag = "  <── not 0: costs or drift left" if abs(values.mean()) >= NULL_T_TH else ""
                parts.append(f"{side} {values.mean():+.3f} (SE {se:.3f}){flag}")
        logger.info(f"  NULL CHECK SA{sa} ── mean t of the columns: " + " ── ".join(parts))
    logger.info(f"{'─' * width}")
    parts = []
    for sa in SELL_AFTER:
        block = recommended_block(rows, sa)
        parts.append(f"SA{sa}: {block if block is not None else '-'}")
    logger.info(f"  RECOMMENDED {timeframe} ── largest block before the first permissive one (either test) ── "
                + " | ".join(parts))
    logger.info(f"{'─' * width}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    wfo_module.tqdm   = partial(tqdm, disable=True)     # one bar per group, not per WFO / bootstrap
    stepm_module.tqdm = partial(tqdm, disable=True)
    stepm_module._build_bootstrap_weight_matrix = _build_weight_matrix_fast
    _zero_costs()

    logger.info(f"DATA: rules on IS ({os.path.basename(DATA_FOLDER_BY_DATASET['IS'])}) ── WFO and StepM OOS on OOS "
                f"({os.path.basename(DATA_FOLDER_BY_DATASET['OOS'])})")
    logger.info(f"GRID: SA {SELL_AFTER} (blocked per rule) x TP {TP_PCT} x SL {SL_PCT} (optimized per WFO window)")
    path_blocks = {tf: path_block_candles(tf) for tf in TIMEFRAMES}
    logger.info(f"BLOCKS: {BLOCKS} ── GROUPS of {GROUP_SIZE} symbols x {RULES_PER_SYMBOL} rules ── PATHS: {N_PATHS} "
                f"per group (permutation block {PATH_BLOCK_DAYS} days = {path_blocks} candles) ── "
                f"BOOTSTRAP: {N_BOOTSTRAP} ── ALPHA: {ALPHA:g} ── MIN WFO TEST TRADES: {MIN_TRADES}")

    with parallel_config(backend="loky", initializer=_zero_costs):
        check_zero_costs()
        for tf in TIMEFRAMES:
            run_timeframe(tf)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL ── {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")