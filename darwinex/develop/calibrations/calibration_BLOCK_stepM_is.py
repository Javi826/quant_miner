#research/calibrations/calibration_BLOCK_sM_size.py
"""
Block size of the StepM IS bootstrap (WHITE_BLOCK_SIZE), checked by the size of the test: calibration of Romano &
Wolf (2005), section 7, Algorithm 7.1. Synthetic data where no rule has an edge goes through the real backtest and the
real StepM bootstrap once per candidate block. With a well sized block the global White p-value (studentized) is
uniform: about 10% of the paths at p <= 0.10, which is the chance that StepM approves at least one rule when none has
an edge.
  - synthetic prices: block permutation of the real candles (multiverse.py), drift removed so the price is a martingale
  - signals        : computed once on the real data and replayed on every synthetic path (they know nothing of its future)
  - costs          : COMISION = 0, so the true Sharpe of every column is exactly 0
  - matrix         : one SELL_AFTER with its whole TP/SL grid (as main_back_fx), one test per SELL_AFTER
  - permissive     : clearly more than alpha at p <= alpha (StepM approves rules without edge)
  - conservative   : clearly less than alpha at p <= alpha (StepM loses power)
A longer block approves more rules, so the recommended block is the largest one before the first permissive one
(blocks scanned from the shortest).
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
logger = logging.getLogger("BOT_batch.stepm_block_size_check")
for noisy_logger in ("BOT_batch.pipeline.backtest_runner", "BOT_batch.pipeline.stepM_is",
                     "BOT_batch.pipeline.multiverse", "joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from utils.ohlcv_utils import prepare_ohlcv_arrays
from signals.indicators_bank import ConditionBank
from rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from pipeline import backtest_runner as backtest_module
from pipeline import stepM_is as stepm_module
from pipeline.backtest_runner import _combo_id
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.multiverse import _build_path_bundle, _build_synthetic_ohlcv_arr, _COL_LOG_RET_CLOSE

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET    = "IS"           # StepM runs on the IS data
TIMEFRAMES = ["4H", "1H"]
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]
SYMBOLS_BY_TIMEFRAME = {    # one combo per symbol
    "4H": SYMBOLS,
    "1H": SYMBOLS,
}
SELL_AFTER  = [20, 40, 100]
TP_PCT      = [0.5, 1.0, 1.5]
SL_PCT      = [0.5, 1.0, 1.5]

BLOCKS = [1, 2, 3, 5, 10, 20, 30, 50, 80]     # candidate WHITE_BLOCK_SIZE, in rows of the StepM matrix (days); 1 = no blocks
# permutation block of the synthetic prices: the synthetic data only keeps the dependence inside a block, so it must
# be at least as long as the longest StepM block tested (otherwise the calibration can never ask for a longer one)
PATH_BLOCK_DAYS = max(BLOCKS)
CANDLES_PER_DAY = {"4H": 6, "1H": 24}

MAX_RULES   = 2000          # rules sampled per symbol after Jaccard (only rules with enough signals)
N_PATHS     = 40            # synthetic paths per symbol: 20 symbols x 50 = 1000 p-values per test (run 2 first to time it)
N_BOOTSTRAP = stepm_module.WHITE_N_BOOTSTRAP
ALPHA       = stepm_module.STEPM_ALPHA
TAIL_SIGNIF = 0.05          # a share is "clearly" off alpha when its binomial p is below this
RANDOM_SEED = 42

# =============================================================================
# NO COSTS: set in this process and in every backtest worker (loky initializer)
# =============================================================================
def _zero_costs() -> None:
    import setup.config_backtest as config_backtest
    import pipeline.backtest_runner as runner
    config_backtest.COMISION = 0.0
    runner.COMISION = 0.0


def _worker_commission() -> float:
    import pipeline.backtest_runner as runner
    return float(runner.COMISION)


def check_zero_costs() -> None:
    values = Parallel(n_jobs=backtest_module.BACKTEST_N_JOBS)(delayed(_worker_commission)() for _ in range(64))
    if backtest_module.COMISION != 0.0 or any(v != 0.0 for v in values):
        raise RuntimeError("COMISION is not 0 in the backtest workers: the true Sharpe would not be 0")

# =============================================================================
# RULES: real signals, replayed on the synthetic paths
# =============================================================================
def make_fixed_signal_fn(signal: np.ndarray):

    def signal_fn(arr: dict, live_trading: bool = True, bank=None, **_) -> np.ndarray:
        if len(arr["close"]) != signal.shape[0]:
            raise ValueError(f"synthetic path has {len(arr['close'])} candles, the real signal {signal.shape[0]}")
        return signal

    return signal_fn


def build_rules(rule_templates: list, ohlcv_arr: dict, sym: str, combo_key: str, timeframe: str) -> tuple:
    rules = build_rule_dicts(rule_templates, combo_key, timeframe)
    rules = pipe_signal_cleaning_jaccard(rules=rules, ohlcv_arr=ohlcv_arr, timeframe=timeframe)

    arr, bank = ohlcv_arr[sym], ConditionBank(ohlcv_arr[sym])

    def real_signal(rule: dict) -> np.ndarray:     # as the backtest: live_trading=False (acts on the next candle)
        return np.asarray(rule["signal_fn"](arr, live_trading=False, bank=bank), dtype=np.int8)

    min_trades = backtest_module.BACKTEST_MIN_TRADES
    eligible = [i for i, r in enumerate(rules) if np.count_nonzero(real_signal(r)) >= min_trades]
    rng  = np.random.default_rng(RANDOM_SEED)
    pick = np.sort(rng.choice(len(eligible), size=min(MAX_RULES, len(eligible)), replace=False))
    fixed = [{**rules[eligible[i]], "signal_fn": make_fixed_signal_fn(real_signal(rules[eligible[i]]))} for i in pick]
    return fixed, len(rules), len(eligible)

# =============================================================================
# SYNTHETIC PRICES: multiverse permutation, mean gross return 1 per candle (no drift for long nor short)
# =============================================================================
def build_martingale_bundle(data: dict, timeframe: str) -> dict:
    bundle = _build_path_bundle(data, raw_columns=["volume"], timeframe=timeframe)
    for symbol_bundle in bundle["symbols"].values():
        log_ret = symbol_bundle["data_array"][:, _COL_LOG_RET_CLOSE]
        log_ret -= np.log(np.mean(np.exp(log_ret)))
    return bundle

# =============================================================================
# ONE PATH: backtest of the whole grid, then the StepM bootstrap per test and block
# =============================================================================
def build_tests() -> dict:
    # one test per SELL_AFTER: the combo ids of its whole TP/SL grid (the matrix main_back_fx gives StepM)
    return {sa: [_combo_id({"SELL_AFTER": sa, "TP_PCT": tp, "SL_PCT": sl}) for tp in TP_PCT for sl in SL_PCT]
            for sa in SELL_AFTER}


def _build_weight_matrix_fast(
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    n_obs: int,
    n_replicas: int,
) -> np.ndarray:
    # same weights as stepM_is._build_bootstrap_weight_matrix (bincount instead of np.add.at: much faster with
    # small blocks, where there are thousands of blocks per replica)
    width = n_obs + 1
    base  = (np.arange(n_replicas, dtype=np.int64) * width)[:, None]
    up    = np.concatenate(((base + starts_full).ravel(), base[:, 0] + starts_last))
    down  = np.concatenate(((base + starts_full + block_size).ravel(), base[:, 0] + starts_last + len_last))
    diff  = np.bincount(up, minlength=n_replicas * width) - np.bincount(down, minlength=n_replicas * width)
    return np.cumsum(diff.reshape(n_replicas, width)[:, :n_obs], axis=1).astype(np.float32)


def global_p_value(sub: np.ndarray, names: list, block_size: int, seed: int) -> float:
    # sub only has columns with std > 0, so compute_bootstrap_null never compacts it in place: no copy needed.
    # The global p only needs the max of each replica: top-M of 1 instead of the FDP's
    null = stepm_module.compute_bootstrap_null(
        sub, names, n_bootstrap=N_BOOTSTRAP, block_size=block_size, seed=seed, topm_size=1,
    )
    if null.n_kept < 2:
        return np.nan
    return stepm_module.compute_global_pvalue(null.row_max, null.z_stat)["global_p"]


def path_block_candles(timeframe: str) -> int:
    return PATH_BLOCK_DAYS * CANDLES_PER_DAY[timeframe]


def run_path(rules: list, bundle: dict, path_idx: int, base_seed: int, timeframe: str, tests: dict) -> tuple:
    t0 = time.time()
    ohlcv_arr, _layout = _build_synthetic_ohlcv_arr(bundle, path_idx, path_block_candles(timeframe), base_seed)
    _, _, matrix_arr, col_names = backtest_module.pipe_backtesting(
        rules        = rules,
        ohlcv_arr    = ohlcv_arr,
        param_grid   = {"SELL_AFTER": SELL_AFTER, "TP_PCT": TP_PCT, "SL_PCT": SL_PCT},
        order_amount = ORDER_AMOUNT,
        timeframe    = timeframe,
    )
    combo_of_col = np.array([name.rsplit("__", 1)[1] for name in col_names])
    t1 = time.time()

    p_by_key, cols_by_sa = {}, {}
    for sa, combo_ids in tests.items():
        cols = np.flatnonzero(np.isin(combo_of_col, combo_ids))
        sub  = matrix_arr[:, cols]
        sub  = sub[np.any(sub != 0, axis=1)]       # rows with activity: as a backtest of these columns alone
        keep = sub.std(axis=0, ddof=1, dtype=np.float64) > 0 if sub.shape[0] > 1 else np.zeros(cols.shape[0], bool)
        cols, sub = cols[keep], np.ascontiguousarray(sub[:, keep])
        cols_by_sa[sa] = cols.shape[0]
        if cols.shape[0] < 2:
            continue
        names = [col_names[c] for c in cols]
        for block in BLOCKS:
            p_by_key[(sa, block)] = global_p_value(sub, names, block, RANDOM_SEED + path_idx)
    t2 = time.time()
    return p_by_key, cols_by_sa, (t1 - t0, t2 - t1)

# =============================================================================
# SIZE OF THE TEST
# =============================================================================
def _binom_tail_ge(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(x, n + 1))


def _binom_tail_le(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, x + 1))


def size_row(p_values: np.ndarray) -> dict:
    p_values = p_values[np.isfinite(p_values)]
    n = len(p_values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "le05": np.nan, "le": np.nan, "ge": np.nan, "verdict": "-"}
    k_low = int((p_values <= ALPHA).sum())
    if _binom_tail_ge(k_low, n, ALPHA) < TAIL_SIGNIF:
        verdict = "permissive"
    elif _binom_tail_le(k_low, n, ALPHA) < TAIL_SIGNIF:
        verdict = "conservative"
    else:
        verdict = "ok"
    return {"n": n, "mean": float(p_values.mean()), "le05": float((p_values <= 0.05).mean()),
            "le": k_low / n, "ge": float((p_values >= 1.0 - ALPHA).mean()), "verdict": verdict}

# =============================================================================
# ONE TIMEFRAME
# =============================================================================
def run_timeframe(timeframe: str) -> None:
    folder = DATA_FOLDER_BY_DATASET[DATASET]
    tests  = build_tests()
    p_values = {(sa, block): [] for sa in tests for block in BLOCKS}
    n_cols   = {sa: [] for sa in tests}
    rule_templates = None

    for i, sym in enumerate(SYMBOLS_BY_TIMEFRAME[timeframe]):
        combo_key = f"{timeframe}_{sym}"
        data      = build_universe(folder, {timeframe: [sym]}, dataset=DATASET)[timeframe]
        ohlcv_arr = prepare_ohlcv_arrays(data)
        if rule_templates is None:
            rule_templates = build_rule_templates(ohlcv_arr, timeframe, RULE_MAX_DEPTH)
        rules, n_jaccard, n_eligible = build_rules(rule_templates, ohlcv_arr, sym, combo_key, timeframe)
        logger.info(f"\n{timeframe} {sym}: {len(rules)} rules (sampled from {n_eligible} with >= "
                    f"{backtest_module.BACKTEST_MIN_TRADES} signals, {n_jaccard} after Jaccard)")
        if len(rules) == 0:
            continue

        bundle    = build_martingale_bundle(data, timeframe)
        base_seed = RANDOM_SEED + 1_000_003 * (i + 1)
        n_blocks  = bundle["n_ref_rows"] / path_block_candles(timeframe)
        if n_blocks < 20:
            logger.warning(f"{timeframe} {sym}: only {n_blocks:.0f} permutation blocks: the synthetic paths will look alike")
        with tqdm(range(N_PATHS), desc=f"PATHS {timeframe} {sym}", dynamic_ncols=True) as bar:
            for path_idx in bar:
                p_by_key, cols_by_sa, (t_backtest, t_bootstrap) = run_path(rules, bundle, path_idx, base_seed, timeframe, tests)
                bar.set_postfix_str(f"backtest {t_backtest:.1f}s ── bootstrap {t_bootstrap:.1f}s")
                for key, p in p_by_key.items():
                    p_values[key].append(p)
                for sa, n in cols_by_sa.items():
                    n_cols[sa].append(n)

    rows = [{"sa": sa, "block": block,
             "cols": float(np.mean(n_cols[sa])) if n_cols[sa] else 0.0,
             **size_row(np.array(p_values[(sa, block)], dtype=float))}
            for sa in tests for block in BLOCKS]
    log_table(timeframe, rows)


def recommended_block(rows: list, sa: int):
    # largest block before the first permissive one, scanning from the shortest
    chosen = None
    for block in sorted(BLOCKS):
        row = next((r for r in rows if r["sa"] == sa and r["block"] == block), None)
        if row is None or row["n"] == 0 or row["verdict"] == "permissive":
            break
        chosen = block
    return chosen


def log_table(timeframe: str, rows: list) -> None:
    logger.info(f"\n{'─' * 86}")
    logger.info(f"  {timeframe} ── size of StepM with no-edge rules (valid: ~{ALPHA:.0%} of the paths at p <= {ALPHA:g})")
    logger.info(f"{'─' * 86}")
    logger.info(f"  {'SA':>4}  {'BLOCK':>5}  {'COLS':>7}  {'N':>5}  {'MEAN_P':>7}  {'P<=0.05':>8}  "
                f"{f'P<={ALPHA:g}':>8}  {f'P>={1 - ALPHA:g}':>8}  VERDICT")
    previous = None
    for r in rows:
        if previous is not None and r["sa"] != previous:
            logger.info("")
        previous = r["sa"]
        logger.info(f"  {r['sa']:>4}  {r['block']:>5}  {r['cols']:>7.0f}  {r['n']:>5}  {r['mean']:>7.3f}  "
                    f"{r['le05']:>8.1%}  {r['le']:>8.1%}  {r['ge']:>8.1%}  {r['verdict']}")
    logger.info(f"{'─' * 86}")
    parts = []
    for sa in SELL_AFTER:
        block = recommended_block(rows, sa)
        parts.append(f"SA{sa}: {block if block is not None else '-'}")
    logger.info(f"  RECOMMENDED {timeframe} ── largest block before the first permissive one ── " + " | ".join(parts))
    logger.info(f"{'─' * 86}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    backtest_module.tqdm = partial(tqdm, disable=True)     # one bar per path, not per backtest / bootstrap
    stepm_module.tqdm    = partial(tqdm, disable=True)
    stepm_module._build_bootstrap_weight_matrix = _build_weight_matrix_fast
    _zero_costs()

    logger.info(f"DATASET: {DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}")
    logger.info(f"GRID: SA {SELL_AFTER} x TP {TP_PCT} x SL {SL_PCT} ── one StepM matrix per SA (as main_back_fx)")
    path_blocks = {tf: path_block_candles(tf) for tf in TIMEFRAMES}
    logger.info(f"BLOCKS: {BLOCKS} ── PATHS: {N_PATHS} per symbol (permutation block {PATH_BLOCK_DAYS} days = "
                f"{path_blocks} candles) ── RULES: {MAX_RULES} per symbol ── BOOTSTRAP: {N_BOOTSTRAP} ── ALPHA: {ALPHA:g}")

    with parallel_config(backend="loky", initializer=_zero_costs):
        check_zero_costs()
        for tf in TIMEFRAMES:
            run_timeframe(tf)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL ── {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")