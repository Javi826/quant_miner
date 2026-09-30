#darwinex/develop/calibrations/calibration_BLOCK_mv_size.py
"""
Block size of multiverse.py's MCPT, checked by the size of the test: random-entry rules (no edge by construction)
go through the real WFO and the real multiverse on the OOS data, once per candidate block. With a valid block their
p-values are uniform: about 10% of them at p <= 0.10 and about 10% at p >= 0.90.
  - permissive  : clearly more than 10% at p <= 0.10 (the null is too calm: false positives)
  - conservative: clearly less than 10% in both tails (the null is too close to the real path: no power)
The recommended block is the shortest one that is not permissive.
"""
import os
import sys
import math
import time
import logging
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "core")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.multiverse_block_size_check")
for noisy_logger in ("BOT_batch.pipeline.multiverse", "BOT_batch.pipeline.wfo", "joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from symbols.universe import build_universe
from setup.config_paths import DATA_FOLDER_BY_DATASET
from setup.config_backtest import ORDER_AMOUNT
from utils.ohlcv_utils import prepare_ohlcv_arrays
from pipeline.backtest_runner import _combo_id
from pipeline.wfo import pipe_wfo
from pipeline.multiverse import pipe_multiverse, MULTIVERSE_PVALUE_TH

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET    = "OOS"          # the multiverse runs on the OOS data
TIMEFRAMES = ["4H", "1H"]
BLOCKS_BY_TIMEFRAME = {     # candidate block sizes, in candles
    "4H": [6, 12, 30, 100],
    "1H": [24, 48, 120, 400],
}
SYMBOLS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "AUDCAD", "GBPCAD", "NZDJPY", "GBPCHF",
]
SELL_AFTER = [20, 40, 100]  # each rule gets one, blocked as in production (best_combo_id)
TP_PCT     = 1.0
SL_PCT     = 1.0

N_RULES_PER_SYMBOL = 12     # cycles SELL_AFTER x (long, short)
FIRE_RATE          = 0.05   # share of candles where a random rule fires
N_PATHS            = 200    # multiverse paths per rule (production: 1000)
ALPHA              = MULTIVERSE_PVALUE_TH
TAIL_SIGNIF        = 0.05   # a tail share is "clearly" off 10% when its binomial p is below this
RANDOM_SEED        = 42

# =============================================================================
# RANDOM RULES — the entry depends only on the candle's timestamp: the same candles on the real data, on every WFO
# window and on every multiverse path. It never looks at prices, so it has no edge
# =============================================================================
def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def make_random_signal_fn(seed: int, side: str, fire_rate: float):
    sign = 1 if side == "long" else -1

    def signal_fn(arr: dict, live_trading: bool = True, bank=None, **_) -> np.ndarray:
        ts = np.asarray(arr["ts"]).astype("datetime64[ns]").astype(np.int64).astype(np.uint64)
        u  = (_splitmix64(ts ^ np.uint64(seed)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        signal = np.where(u < fire_rate, sign, 0).astype(np.int32)
        if not live_trading:                        # as signal_builder: the signal acts on the next candle
            signal = np.roll(signal, 1)
            signal[0] = 0
        return signal

    return signal_fn


def build_random_rules(combo_key: str, timeframe: str, symbol_idx: int) -> list:
    rules = []
    for k in range(N_RULES_PER_SYMBOL):
        sa   = SELL_AFTER[k % len(SELL_AFTER)]
        side = ("long", "short")[(k // len(SELL_AFTER)) % 2]
        seed = (RANDOM_SEED * 1_000_003 + symbol_idx * 1_009 + k) & 0xFFFFFFFFFFFFFFFF
        rules.append({
            "rule_id":       f"{k:04d}_{combo_key}_{side}_random_SA{sa}",
            "combo_key":     combo_key,
            "timeframe":     timeframe,
            "side":          side,
            "label":         f"random {FIRE_RATE:.0%} {side} SA{sa}",
            "signal_fn":     make_random_signal_fn(seed, side, FIRE_RATE),
            "best_combo_id": _combo_id({"SELL_AFTER": sa, "TP_PCT": TP_PCT, "SL_PCT": SL_PCT}),
            "sell_after":    sa,
        })
    return rules

# =============================================================================
# SIZE OF THE TEST
# =============================================================================
def _binom_tail_ge(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(x, n + 1))


def _binom_tail_le(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, x + 1))


def size_row(p_values: np.ndarray) -> dict:
    n = len(p_values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "le05": np.nan, "le": np.nan, "ge": np.nan, "verdict": "-"}
    k_low, k_high = int((p_values <= ALPHA).sum()), int((p_values >= 1.0 - ALPHA).sum())
    if _binom_tail_ge(k_low, n, ALPHA) < TAIL_SIGNIF:
        verdict = "permissive"
    elif _binom_tail_le(k_low, n, ALPHA) < TAIL_SIGNIF and _binom_tail_le(k_high, n, ALPHA) < TAIL_SIGNIF:
        verdict = "conservative"
    else:
        verdict = "ok"
    return {"n": n, "mean": float(p_values.mean()), "le05": float((p_values <= 0.05).mean()),
            "le": k_low / n, "ge": k_high / n, "verdict": verdict}

# =============================================================================
# ONE TIMEFRAME
# =============================================================================
def run_timeframe(timeframe: str) -> None:
    param_grid = {"SELL_AFTER": SELL_AFTER, "TP_PCT": [TP_PCT], "SL_PCT": [SL_PCT]}
    folder     = DATA_FOLDER_BY_DATASET[DATASET]

    # real WFO of every random rule, one symbol (combo) at a time, as main_back_fx
    data_by_combo, wfo_rules = {}, []
    for i, sym in enumerate(SYMBOLS):
        combo_key = f"{timeframe}_{sym}"
        data = build_universe(folder, {timeframe: [sym]}, dataset=DATASET)[timeframe]
        data_by_combo[combo_key] = data
        wfo_rules += pipe_wfo(
            rules        = build_random_rules(combo_key, timeframe, i),
            ohlcv_arr    = prepare_ohlcv_arrays(data),
            param_grid   = param_grid,
            order_amount = ORDER_AMOUNT,
            timeframe    = timeframe,
            combo_key    = combo_key,
            log_level    = logging.WARNING,
        )
    evaluable = [r for r in wfo_rules if r["wfo_test_trades"] is not None and not r["wfo_test_trades"].empty]
    sa_by_id  = {r["rule_id"]: r["sell_after"] for r in evaluable}
    logger.info(f"\n{timeframe}: {len(evaluable)}/{len(wfo_rules)} random rules with WFO test trades "
                f"({len(SYMBOLS)} symbols x {N_RULES_PER_SYMBOL}, fire rate {FIRE_RATE:.0%}, "
                f"TP {TP_PCT} SL {SL_PCT}, {N_PATHS} paths)")

    rows = []
    for block in BLOCKS_BY_TIMEFRAME[timeframe]:
        t0 = time.time()
        res = pipe_multiverse(
            rules               = evaluable,
            ohlcv_data_by_combo = data_by_combo,
            param_grid          = {timeframe: param_grid},
            order_amount        = ORDER_AMOUNT,
            p_value_th          = ALPHA,
            n_paths             = N_PATHS,
            block_size          = block,
        )
        p_by_id = {r["rule_id"]: r["multiverse_p_value"] for r in res}
        logger.info(f"{timeframe}: block {block} done in {(time.time() - t0) / 60:.1f} min")
        for sa in ["ALL", *SELL_AFTER]:
            ps = np.array([p for rid, p in p_by_id.items() if sa == "ALL" or sa_by_id[rid] == sa], dtype=float)
            rows.append({"block": block, "sa": sa, **size_row(ps)})

    log_table(timeframe, rows)


def log_table(timeframe: str, rows: list) -> None:
    logger.info(f"\n{'─' * 86}")
    logger.info(f"  {timeframe} ── size of the multiverse test with random rules (valid: ~{ALPHA:.0%} in each tail)")
    logger.info(f"{'─' * 86}")
    logger.info(f"  {'BLOCK':>6}  {'SA':>4}  {'N':>5}  {'MEAN_P':>7}  {'P<=0.05':>8}  "
                f"{f'P<={ALPHA:g}':>8}  {f'P>={1 - ALPHA:g}':>8}  VERDICT")
    for r in rows:
        logger.info(f"  {r['block']:>6}  {r['sa']:>4}  {r['n']:>5}  {r['mean']:>7.3f}  {r['le05']:>8.1%}  "
                    f"{r['le']:>8.1%}  {r['ge']:>8.1%}  {r['verdict']}")
    valid = [r["block"] for r in rows if r["sa"] == "ALL" and r["verdict"] != "permissive"]
    logger.info(f"{'─' * 86}")
    logger.info(f"  RECOMMENDED {timeframe}: " + (f"{min(valid)} candles (shortest block that is not permissive)"
                                                 if valid else "none: every block is permissive"))
    logger.info(f"{'─' * 86}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    logger.info(f"DATASET: {DATASET} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET])}")
    for tf in TIMEFRAMES:
        run_timeframe(tf)
    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")