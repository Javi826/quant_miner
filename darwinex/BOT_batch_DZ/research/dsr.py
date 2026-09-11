#sets/dsr.py

import time
import logging
import numpy as np
from scipy.stats import norm
from shared_batchs.utils.reporting import print_dsr_train_metrics
logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR FORMULA CONFIG
# =============================================================================
EULER_GAMMA         = 0.5772156649015328606
SHARPE_PERIODS_YEAR = 365.0
DSR_MAX_SHARPE_ANN  = 10.0  # DSR's own trust threshold — a combo above this is not run through

#shared_batchs/pipeline/dsr.py (continued)
# =============================================================================
# PRIVATE HELPERS — N_eff estimation (streaming Gram accumulation, eigenvalue method)
# =============================================================================

def _participation_ratio_from_gram(gram: np.ndarray, n_const: int) -> float:

    sum_eig    = float(np.einsum("ii->", gram)) + n_const
    sum_eig_sq = float(np.einsum("ij,ij->", gram, gram)) + n_const
    if sum_eig_sq <= 0:
        return 1.0
    return float((sum_eig ** 2) / sum_eig_sq)

BATCH_SIZE_N_EFF = 2000  # standardized columns accumulated before each BLAS matmul —
                         # bounds RAM to T x BATCH_SIZE_N_EFF while keeping each matmul
def _estimate_n_eff_streaming(matrix_arr: np.ndarray, batch_size: int = BATCH_SIZE_N_EFF) -> tuple:

    t_days      = matrix_arr.shape[0]
    n_cols      = matrix_arr.shape[1]
    gram        = np.zeros((t_days, t_days), dtype=np.float64)
    n_const     = 0
    n_untrusted = 0
    n_valid     = 0
    batch_cols  = []
    sharpe_list = []

    for col_idx in range(n_cols):
        col = matrix_arr[:, col_idx]

        mean = col.mean()
        std  = col.std(ddof=1)
        if std <= 0:
            n_const += 1
            continue

        sharpe_ann = float(mean / std) * np.sqrt(SHARPE_PERIODS_YEAR)
        if abs(sharpe_ann) > DSR_MAX_SHARPE_ANN:
            # DSR doesn't trust this combo enough to let it influence n_eff or
            # var_sr — backtest_runner.py reported it faithfully; this is
            # purely DSR's own call, independent of what any other consumer
            # (e.g. StepM) would decide about the same combo.
            n_untrusted += 1
            continue

        batch_cols.append((col - mean) / std)
        sharpe_list.append(float(mean / std))  # unannualized Sharpe of this combo — same population as N_eff
        n_valid += 1

        if len(batch_cols) >= batch_size:
            x_batch = np.column_stack(batch_cols)
            gram   += x_batch @ x_batch.T
            batch_cols = []

    if batch_cols:
        x_batch = np.column_stack(batch_cols)
        gram   += x_batch @ x_batch.T

    if n_valid == 0:
        n_eff = float(n_const) if n_const > 0 else 1.0
        return n_eff, 0.0

    if n_untrusted > 0:
        logger.debug(f"DSR ── excluded {n_untrusted} combo(s) above DSR_MAX_SHARPE_ANN={DSR_MAX_SHARPE_ANN} from n_eff/var_sr")

    gram  /= (t_days - 1)
    n_eff  = _participation_ratio_from_gram(gram, n_const)

    sharpe_arr = np.asarray(sharpe_list, dtype=np.float64)
    var_sr     = float(np.var(sharpe_arr, ddof=1)) if sharpe_arr.size > 1 else 0.0

    return n_eff, var_sr
def estimate_n_eff_and_var_sr(matrix_arr: np.ndarray) -> tuple | None:

    if matrix_arr is None or matrix_arr.shape[1] < 2:
        return None
    return _estimate_n_eff_streaming(matrix_arr)

# =============================================================================
# PRIVATE HELPERS — DSR formula (paper Eq. 1-2)
# =============================================================================
def _unannualize_sharpe(sharpe_annualized: float, periods_per_year: float = SHARPE_PERIODS_YEAR) -> float:
    if sharpe_annualized is None or not np.isfinite(sharpe_annualized):
        return np.nan
    return float(sharpe_annualized / np.sqrt(periods_per_year))

def _expected_max_sharpe(var_sr: float, n_trials: float) -> float:
    """Eq. 1 — expected maximum Sharpe ratio under N independent trials, assuming null skill."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z_n  = norm.ppf(1.0 - 1.0 / n_trials)
    z_ne = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    term = (1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne
    return float(np.sqrt(var_sr) * term)


def _deflated_sharpe_ratio(sr: float, sr0: float, t_obs: int, skew_r: float, kurt_r: float) -> float:
    """Eq. 2. sr and sr0 must both be UNANNUALIZED. kurt_r is raw kurtosis (fisher=False)."""
    if t_obs <= 1 or not np.isfinite(sr):
        return 0.0
    moment_term = 1.0 - skew_r * sr + ((kurt_r - 1.0) / 4.0) * (sr ** 2)
    if moment_term <= 0:
        return 0.0
    numerator = (sr - sr0) * np.sqrt(t_obs - 1)
    return float(norm.cdf(numerator / np.sqrt(moment_term)))

# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_dsr_approval(dsr_value: float, dsr_th: float) -> bool:
    return dsr_value >= dsr_th

# =============================================================================
# CORE DSR CALCULATION (across a set of candidate trials — typically one timeframe)
# =============================================================================
def _compute_dsr(all_raw_results: list, matrix_arr: np.ndarray, dsr_th: float, n_combos: int) -> dict:

    total_candidates = len(all_raw_results)
    n_bruto           = total_candidates * max(n_combos, 1)

    n_eff_var_sr = estimate_n_eff_and_var_sr(matrix_arr)

    n_bruto_str    = f"{n_bruto:,}".replace(",", ".")
    m_str          = f"{total_candidates:,}".replace(",", ".")
    n_eff_str      = f"{n_eff_var_sr[0]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if n_eff_var_sr is not None else "n/a (insufficient data)"

    logger.info(
        f"DSR ── N_bruto={n_bruto_str} (M={m_str} x n_combos={n_combos})  "
        f"N_eff={n_eff_str}"
    )

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    if n_eff_var_sr is None:
        logger.debug("DSR ── N_eff unavailable — setting DSR=0.0 for all rules (no rules pass).")
        dsr_by_id = {rule_id: 0.0 for rule_id in raw_by_id}
        return {
            "passed_dsr_ids": [],
            "dsr_by_rule_id": dsr_by_id,
            "n_eff":          None,
            "n_bruto":        n_bruto,
            "sr0":            np.nan,
        }

    n_eff, var_sr = n_eff_var_sr

    sr_by_id = {
        rule_id: _unannualize_sharpe(r.get("sharpe_train", np.nan))
        for rule_id, r in raw_by_id.items()
    }

    sr0 = _expected_max_sharpe(var_sr, n_eff)

    logger.debug(
        f"DSR ── SR0 terms ── total_candidates={total_candidates} n_combos={n_combos} "
        f"n_eff={n_eff:.4f} var_sr={var_sr:.6f} -> SR0={sr0:.4f}"
    )

    dsr_by_id = {}
    for rule_id, r in raw_by_id.items():
        t_days     = int(r.get("n_days_train", 0))
        skew_r     = float(r.get("skew_train", np.nan))
        kurt_r     = float(r.get("kurtosis_train", np.nan))
        sharpe_ann = r.get("sharpe_train", np.nan)

        if not (np.isfinite(skew_r) and np.isfinite(kurt_r)):
            dsr_by_id[rule_id] = 0.0
            continue

        if np.isfinite(sharpe_ann) and abs(sharpe_ann) > DSR_MAX_SHARPE_ANN:
            # The winning combo itself is above DSR's trust threshold — DSR
            # refuses to score it, same as it refuses a rule with unusable
            # skew/kurtosis. This is independent of whatever any other
            # consumer of the same raw backtest output decides to do with it.
            dsr_by_id[rule_id] = 0.0
            continue

        dsr_by_id[rule_id] = _deflated_sharpe_ratio(sr_by_id[rule_id], sr0, t_days, skew_r, kurt_r)

    passed_dsr_ids = [rid for rid, dsr_val in dsr_by_id.items() if _evaluate_dsr_approval(dsr_val, dsr_th)]

    if logger.isEnabledFor(logging.DEBUG):
        print_dsr_train_metrics(raw_by_id, dsr_by_id, sr_by_id, set(passed_dsr_ids), set(passed_dsr_ids), sr0)

    logger.debug(
        f"DSR ── M={total_candidates} n_combos={n_combos} N_bruto={n_bruto} N_eff={n_eff:.4f} SR0={sr0:.3f} "
        f"-> {len(passed_dsr_ids)}/{total_candidates} significant at th={dsr_th}"
    )

    return {
        "passed_dsr_ids": passed_dsr_ids,
        "dsr_by_rule_id": dsr_by_id,
        "n_eff":          n_eff,
        "n_bruto":        n_bruto,
        "sr0":            sr0,
    }


# =============================================================================
# PIPE DSR — one timeframe at a time
# =============================================================================
def empty_dsr_fields() -> dict:
    """Placeholder DSR fields for rules that were never evaluated (pipe disabled)."""
    return {
        "passed_dsr":     True,
        "passed_mbias":   True,
        "dsr":            0.0,
        "sharpe_train":   None,
        "skew_train":     None,
        "kurtosis_train": None,
        "n_days_train":   None,
        "net_gain_train": None,
        "max_dd_train":   None,
        "best_combo_id":  None,
    }

def pipe_dsr(
    raw_results: list,
    matrix_arr: np.ndarray,
    dsr_th: float,
    n_combos: int,
    timeframe: str = "",
) -> list:

    start = time.time()

    dsr_result     = _compute_dsr(raw_results, matrix_arr, dsr_th=dsr_th, n_combos=n_combos)
    passed_dsr_ids = set(dsr_result["passed_dsr_ids"])
    dsr_by_id      = dsr_result["dsr_by_rule_id"]

    logger.info(f"DSR ── {timeframe} ── {len(passed_dsr_ids)}/{len(raw_results)} rules pass")

    results = []
    for r in raw_results:
        rid    = r["rule_id"]
        passed = rid in passed_dsr_ids
        results.append({
            **r,
            "passed_dsr":     passed,
            "passed_mbias":   passed,
            "dsr":            dsr_by_id.get(rid, 0.0),
            "sharpe_train":   r["sharpe_train"],
            "skew_train":     r["skew_train"],
            "kurtosis_train": r["kurtosis_train"],
            "n_days_train":   r["n_days_train"],
            "net_gain_train": r["net_gain_train"],
            "max_dd_train":   r["max_dd_train"],
            "best_combo_id":  r["best_combo_id"] if passed else None,
        })

    elapsed = int(time.time() - start)
    logger.info(f"DSR ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results