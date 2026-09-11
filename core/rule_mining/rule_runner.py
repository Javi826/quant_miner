# core/rule_mining/rule_runner.py
import os
import logging
from pipeline.wfo import pipe_wfo
from pipeline.backtest_runner import pipe_backtesting
from pipeline.stepM import pipe_stepm
from pipeline.correlation import pipe_correlation
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from utils.plotting import plot_rule_mining_filter_comparison, plot_rule_mining_portfolio_comparison
from setup.config_backtest import INITIAL_BALANCE
from runs.run_portfolio import find_best_portfolio_combination_wfo
from rule_mining.rule_generator import generate_all_rules, MAX_DEPTH
from rule_mining.rule_writter import run_deploy_rule, save_rule_deploy_batch
from utils.reporting import print_rule_mining_ranking, print_rule_mining_min_by_group
from pipeline.multiverse import pipe_multiverse
logger = logging.getLogger("BOT_batch.rule_mining.runner")

_OP_SLUG = {">": "gt", "<": "lt"}

def _slugify_label(label: str) -> str:
    slug = label
    for op, tag in _OP_SLUG.items():
        slug = slug.replace(op, tag)
    slug = slug.replace(" AND ", "_AND_").replace(" ", "_")
    slug = slug.replace("[", "").replace("]", "").replace("-", "m")
    return slug


def _build_rule_id(i: int, combo_key: str, rule: dict) -> str:
    return f"{i:06d}_{combo_key}_{rule['side']}_{_slugify_label(rule['label'])}"

def _build_rule_dicts(ohlcv_data: dict, combo_key: str, timeframe: str, max_depth: int) -> list:

    arr_sample = next(iter(ohlcv_data.values()))
    all_rules  = generate_all_rules({
        "open":  arr_sample["open"],
        "high":  arr_sample["high"],
        "low":   arr_sample["low"],
        "close": arr_sample["close"],
        "volume": arr_sample["volume"],
    }, max_depth=max_depth)

    return [
        {
            "rule_id":   _build_rule_id(i, combo_key, rule),
            "combo_key": combo_key,
            "timeframe": timeframe,
            "side":      rule["side"],
            "specs":     rule["specs"],
            "signal_fn": rule["signal_fn"],
            "label":     rule["label"],
        }
        for i, rule in enumerate(all_rules)
    ]

def _empty_wfo_fields() -> dict:

    return {
        "approved":        False,
        "net_gain":        0.0,
        "max_dd":          0.0,
        "n_trades":        0,
        "n_windows":       0,
        "win_rate":        0.0,
        "profit_factor":   0.0,
        "calmar":          0.0,
        "r_squared":       0.0,
        "wfr":             0.0,
        "best_params":     None,
        "wfo_test_trades": None,
    }
# =============================================================================
# ORCHESTRATOR
# =============================================================================
def run_rule_mining_pipeline(
    ohlcv_data_mining_by_combo: dict,
    ohlcv_arr_mining_by_combo: dict,
    ohlcv_data_validation_by_combo: dict,
    ohlcv_arr_validation_by_combo: dict,
    combos: list,
    param_grid: dict,
    order_amount: int,
    data_folder: str,
    show_progress: bool = False,
    max_depth: int = MAX_DEPTH,
    log_level: int = logging.INFO,
    save_trades: bool = False,
    brief_trades_folder: str = None,
    show_plots: bool = False,
    deploy_output_path: str = None,
    run_config: dict = None,
    pipeline_wfo: bool = True,
    pipeline_correlation: bool = True,  
    pipeline_multiverse: bool = True,
    run_best_portfolio: bool = True,
    run_deploy: bool = False,
) -> list:
    #-----------------------------------------------------------------
    # BACKTESTING — one combo at a time, ALL combos before moving on.
    # -------------------------------------------------------------------
    all_mbias_results = []
    FF_by_timeframe  = {}
    for combo in combos:
        combo_key, timeframe = combo["combo_key"], combo["timeframe"]
        rules = _build_rule_dicts(
            ohlcv_data_mining_by_combo[combo_key], combo_key, timeframe, max_depth,
        )
        logger.info(f"\n\033[36m{'=' * 70}")
        logger.info(f"======== RULE MINING ── {combo_key} {combo['symbols']} ── rules: {format(len(rules), ',').replace(',', '.')}")
        logger.info(f"{'=' * 70}\033[0m")

        rules = pipe_signal_cleaning_jaccard(
            rules     = rules,
            ohlcv_arr = ohlcv_arr_mining_by_combo[combo_key],
            timeframe = timeframe,
        )

        raw_results, n_combos, matrix_arr, col_names = pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr_mining_by_combo[combo_key],
            param_grid   = param_grid,
            order_amount = order_amount,
            timeframe    = timeframe,
        )

        mbias_results = pipe_stepm(
            raw_results = raw_results,
            matrix_arr  = matrix_arr,
            col_names   = col_names,
            timeframe   = timeframe,
        )

        del raw_results, matrix_arr

        all_mbias_results.extend([{**r, **_empty_wfo_fields()} for r in mbias_results])

    passed_mbias_ids = {r["rule_id"] for r in all_mbias_results if r["passed_mbias"]}

    print_rule_mining_ranking(all_mbias_results, list(passed_mbias_ids), "POST-MBIAS", survivor_ids=list(passed_mbias_ids))
    wfo_by_id = {}
    for combo in combos:
        combo_key, timeframe = combo["combo_key"], combo["timeframe"]
        rules_this_combo = [
            r for r in all_mbias_results
            if r["combo_key"] == combo_key and r["passed_mbias"]
        ]

        wfo_results = pipe_wfo(
            rules               = rules_this_combo,
            ohlcv_arr           = ohlcv_arr_validation_by_combo[combo_key],
            param_grid          = param_grid,
            order_amount        = order_amount,
            timeframe           = timeframe,
            combo_key           = combo_key,
            enabled             = pipeline_wfo,
            show_progress       = show_progress,
            log_level           = log_level,
            save_trades         = save_trades,
            brief_trades_folder = brief_trades_folder,
        )
        wfo_by_id.update({r["rule_id"]: r for r in wfo_results})

    all_raw_results = [
        wfo_by_id[r["rule_id"]] if r["rule_id"] in wfo_by_id else r
        for r in all_mbias_results
    ]

    wfo_candidate_ids = list(wfo_by_id.keys())
    print_rule_mining_ranking(all_raw_results, wfo_candidate_ids, "POST-WFO", survivor_ids=[r["rule_id"] for r in all_raw_results if r["approved"]])

    # -------------------------------------------------------------------
    # Portfolio construction on the flattened, cross-combo pool.
    # -------------------------------------------------------------------
    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    validated_after_wfo = [
        (r["rule_id"], r["wfo_test_trades"])
        for r in all_raw_results
        if r["approved"] and r["wfo_test_trades"] is not None and not r["wfo_test_trades"].empty
    ]

    print_rule_mining_min_by_group(all_raw_results, [rid for rid, _ in validated_after_wfo], "POST-WFO", wfo_candidate_ids)

    # -------------------------------------------------------------------
    # Correlation (portfolio construction: drop redundant rules)
    # -------------------------------------------------------------------
    if validated_after_wfo:
        candidates_before_corr = [rid for rid, _ in validated_after_wfo]
        rules_for_corr = [raw_by_id[rid] for rid in candidates_before_corr]
        survivors_rules = pipe_correlation(
            rules           = rules_for_corr,
            initial_balance = INITIAL_BALANCE,
            enabled         = pipeline_correlation,
        )
        survivors_corr              = [r["rule_id"] for r in survivors_rules]
        validated_after_correlation = [(rid, raw_by_id[rid]["wfo_test_trades"]) for rid in survivors_corr]
        print_rule_mining_ranking(all_raw_results, candidates_before_corr, "POST-CORRELATION", survivor_ids=survivors_corr)
        print_rule_mining_min_by_group(all_raw_results, survivors_corr, "POST-CORRELATION", candidates_before_corr)
    else:
        validated_after_correlation = validated_after_wfo
        print_rule_mining_ranking(all_raw_results, [rid for rid, _ in validated_after_correlation], "POST-CORRELATION")


    # -------------------------------------------------------------------
    # Multiverse / MCPT (log-return permutation — validates EDGE via p-value)
    # -------------------------------------------------------------------
    if validated_after_correlation:
        
        candidates_before_mv = [rid for rid, _ in validated_after_correlation]
        rules_for_mv = [raw_by_id[rid] for rid in candidates_before_mv]
        mv_results = pipe_multiverse(
            rules               = rules_for_mv,
            ohlcv_data_by_combo = ohlcv_data_validation_by_combo,
            param_grid          = param_grid,
            order_amount        = order_amount,
            enabled             = pipeline_multiverse,
        )
        for r in mv_results:
            raw_by_id[r["rule_id"]]["multiverse_p_value"] = r["multiverse_p_value"]

        validated_after_multiverse = [
            (r["rule_id"], r["wfo_test_trades"]) for r in mv_results if r["passed_multiverse"]
        ]
        survivors_mv = [rid for rid, _ in validated_after_multiverse]
        print_rule_mining_ranking(all_raw_results, candidates_before_mv, "POST-MULTIVERSE", survivor_ids=survivors_mv)
        print_rule_mining_min_by_group(all_raw_results, survivors_mv, "POST-MULTIVERSE", candidates_before_mv)
    else:
        validated_after_multiverse = validated_after_correlation
        print_rule_mining_ranking(all_raw_results, [rid for rid, _ in validated_after_multiverse], "POST-MULTIVERSE")

    if show_plots:
        for rule_id, trades in validated_after_multiverse:
            plot_rule_mining_filter_comparison(
                strategy_id        = f"{rule_id}_wfo_test",
                trades_df_baseline = trades,
                trades_df_r01      = None,
                data_folder        = data_folder,
                initial_balance    = INITIAL_BALANCE,
                regime_enabled     = False,
            )

        if validated_after_multiverse:
            plot_rule_mining_portfolio_comparison(
                strategy_trades_baseline = validated_after_multiverse,
                strategy_trades_regime01 = None,
                data_folder              = data_folder,
                initial_balance          = INITIAL_BALANCE,
                title                    = "Portfolio WFO Test — Validated only",
            )

    top_portfolios = []
    if run_best_portfolio and validated_after_multiverse:
        top_portfolios = find_best_portfolio_combination_wfo(
            validated_wfo_trades = validated_after_multiverse,
            initial_balance      = INITIAL_BALANCE,
            show_plots            = show_plots,
        )

    if run_deploy and top_portfolios:
        for top_idx, portfolio in enumerate(top_portfolios, start=1):
            combo_ids = list(portfolio["combo"])
            if not combo_ids:
                continue

            deploy_map  = {}
            label_width = max(len(rid) for rid in combo_ids) + 2

            for rule_id in combo_ids:
                rule_info = raw_by_id[rule_id]
                rule_tf   = rule_info["timeframe"]

                run_deploy_rule(
                    rule_id             = rule_id,
                    specs               = rule_info["specs"],
                    side                = rule_info["side"],
                    timeframe           = rule_tf,
                    ohlcv_is            = ohlcv_data_validation_by_combo[rule_info["combo_key"]],
                    signal_fn           = rule_info["signal_fn"],
                    param_grid          = param_grid,
                    order_amount        = order_amount,
                    approved            = rule_info["approved"],
                    deploy_map          = deploy_map,
                    label_width         = label_width,
                )

            save_rule_deploy_batch(
                output_path = _build_top_output_path(deploy_output_path, top_idx),
                deploy_map  = deploy_map,
                run_config  = run_config,
            )

    return validated_after_multiverse, all_mbias_results

def _build_top_output_path(base_path: str, top_idx: int) -> str:
    """Insert a _topN suffix before the file extension, e.g. rules_batch.py -> rules_batch_top1.py."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_top{top_idx}{ext}"
