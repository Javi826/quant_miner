#BOT_batch/check_wfo_multiverse.py
import os
import sys
import glob
import time
import pickle
import struct
import pstats
import argparse
import cProfile
import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
try:
    import cloudpickle                        # pickles signal_fn closures by value, as joblib does for its workers
except ImportError:
    from joblib.externals import cloudpickle

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import backtesting_fx as fx  # sets sys.path, logging and configuration; its main block does not run on import

import pipeline.wfo as wfo
import pipeline.multiverse as mv
import runs.run_deploy as rd
import rule_mining.rule_runner as rr

DEFAULT_ARGS = "run --cap ~/wfo_mv_cap"   # used when no arguments arrive (Spyder Run)
LINE         = "=" * 110
CAPTURE_FILE = "capture.pkl"


# =============================================================================
# ARGS
# =============================================================================
def _parse_args():
    p   = argparse.ArgumentParser(description="WFO, multiverse and deploy: capture real inputs, then time and verify them bit for bit")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="run the mining pipeline and save the WFO, multiverse and deploy inputs")
    c.add_argument("--out", required=True, help="empty folder for the capture")

    r = sub.add_parser("run", help="replay WFO, multiverse and deploy on the capture (parallel, as production)")
    r.add_argument("--cap", required=True, help="capture folder")
    r.add_argument("--paths", type=int, default=0, help="multiverse paths (0 = as the pipeline, keep it for baselines)")
    r.add_argument("--save-baseline", default=None, metavar="PATH", help="save every output to PATH.pkl")
    r.add_argument("--compare", default=None, metavar="PATH", help="compare every output with PATH.pkl")

    f = sub.add_parser("profile", help="cProfile a few multiverse paths and a few WFO rules in one process")
    f.add_argument("--cap", required=True, help="capture folder")
    f.add_argument("--paths", type=int, default=10, help="multiverse paths")
    f.add_argument("--wfo-rules", type=int, default=50, help="WFO rules of the first combo")
    f.add_argument("--top", type=int, default=30, help="functions listed per table")

    argv = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ARGS.split()
    args = p.parse_args(argv)
    for name in ("out", "cap", "save_baseline", "compare"):
        if getattr(args, name, None):
            setattr(args, name, os.path.expanduser(getattr(args, name)))
    return args


@contextmanager
def _patched_everywhere(name: str, original, replacement):
    # replaces the function in every loaded module that imported it by name, whatever the caller module is
    owners = [m for m in list(sys.modules.values()) if m is not None and getattr(m, name, None) is original]
    for m in owners:
        setattr(m, name, replacement)
    try:
        yield
    finally:
        for m in owners:
            setattr(m, name, original)


@contextmanager
def _patched(owner, **attrs):
    orig = {name: getattr(owner, name) for name in attrs}
    for name, value in attrs.items():
        setattr(owner, name, value)
    try:
        yield
    finally:
        for name, value in orig.items():
            setattr(owner, name, value)


# =============================================================================
# CAPTURE
# =============================================================================
def run_capture(args) -> None:
    if glob.glob(os.path.join(args.out, CAPTURE_FILE)):
        raise SystemExit(f"capture folder is not empty: {args.out} (delete it or use another --out)")
    os.makedirs(args.out, exist_ok=True)
    calls = {"wfo": [], "multiverse": None, "deploy": []}
    orig_wfo, orig_mv, orig_deploy = wfo.pipe_wfo, mv.pipe_multiverse, rd.run_wfo_deploy_ema

    def wfo_capture(*a, **kw):
        calls["wfo"].append((a, kw))
        return orig_wfo(*a, **kw)

    def multiverse_capture(*a, **kw):
        calls["multiverse"] = (a, kw)
        return orig_mv(*a, **kw)

    def deploy_capture(*a, **kw):
        calls["deploy"].append((a, kw))
        return orig_deploy(*a, **kw)

    print(f"\n{LINE}")
    print(f" CAPTURE WFO + MULTIVERSE + DEPLOY | mode {fx.settings.BACKTEST_MODE} | {args.out}")
    print(LINE)
    t0     = time.perf_counter()
    combos = fx.build_combos()
    data_is, arr_is = fx.load_ohlcv_by_combo(combos, fx.DATA_FOLDER_BY_DATASET[fx.DATASET_IS], fx.DATASET_IS)
    if fx.SPLIT_MODE:
        data_oos, arr_oos = fx.load_ohlcv_by_combo(combos, fx.DATA_FOLDER_BY_DATASET[fx.DATASET_OOS], fx.DATASET_OOS)
    else:
        data_oos, arr_oos = data_is, arr_is

    # deploy writes its rule files inside the capture folder: the production rules_batch files are never touched
    deploy_path = os.path.join(args.out, "deploy", "rules_batch.py")
    os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
    with _patched_everywhere("pipe_wfo", orig_wfo, wfo_capture), \
         _patched_everywhere("pipe_multiverse", orig_mv, multiverse_capture), \
         _patched_everywhere("run_wfo_deploy_ema", orig_deploy, deploy_capture):
        rr.run_rule_mining_pipeline(
            ohlcv_data_is_by_combo  = data_is,
            ohlcv_arr_is_by_combo   = arr_is,
            ohlcv_data_oos_by_combo = data_oos,
            ohlcv_arr_oos_by_combo  = arr_oos,
            combos                  = combos,
            param_grid              = fx.PARAM_GRID_BY_TIMEFRAME,
            indicators_by_timeframe = fx.SELECTED_INDICATORS_BY_TIMEFRAME,
            order_amount            = fx.ORDER_AMOUNT,
            data_folder             = fx.DATA_FOLDER_BY_DATASET[fx.DATASET_OOS],
            max_depth               = fx.RULE_MAX_DEPTH,
            log_level               = fx.MODULE_LOG_LEVELS["BOT_batch.pipeline.wfo"],
            save_trades             = False,
            brief_trades_folder     = fx.BRIEF_TRADES_FOLDER,
            show_plots              = False,
            deploy_output_path      = deploy_path,
            run_config              = fx.run_config,
            run_deploy              = True,
        )

    if calls["multiverse"] is None:
        print("\n WARNING: the pipeline never reached multiverse (no rule survived correlation)")
    if not calls["deploy"]:
        print("\n WARNING: the pipeline deployed no rule: deploy is not covered by this capture")
    with open(os.path.join(args.out, CAPTURE_FILE), "wb") as f:
        cloudpickle.dump({
            "created":    datetime.datetime.now().isoformat(timespec="seconds"),
            "mode":       fx.settings.BACKTEST_MODE,
            "wfo":        calls["wfo"],
            "multiverse": calls["multiverse"],
            "deploy":     calls["deploy"],
        }, f)
    n_wfo = sum(len(_rules_of(c)) for c in calls["wfo"])
    n_mv  = len(_rules_of(calls["multiverse"])) if calls["multiverse"] else 0
    print(f"\n captured: {len(calls['wfo'])} WFO calls ({n_wfo:,} rules) ── multiverse {n_mv} rules ── "
          f"deploy {len(calls['deploy'])} calls ── {time.perf_counter() - t0:,.0f} s")
    print(LINE)


def _rules_of(call) -> list:
    a, kw = call
    return kw["rules"] if "rules" in kw else a[0]


def _load_capture(folder: str) -> dict:
    with open(os.path.join(folder, CAPTURE_FILE), "rb") as f:
        cap = pickle.load(f)
    if cap["mode"] != fx.settings.BACKTEST_MODE:
        raise SystemExit(f"capture was made in mode {cap['mode']}, BACKTEST_MODE is now {fx.settings.BACKTEST_MODE}")
    return cap


def _with(call, **overrides) -> tuple:
    a, kw = call
    if a:
        raise SystemExit("captured call used positional arguments: overrides need keyword calls")
    return a, {**kw, **overrides}


# =============================================================================
# RUN (production replay) + BASELINE
# =============================================================================
def _record(value):
    # plain data only: functions (signal_fn) are dropped, everything else is kept for the bit-for-bit comparison
    if isinstance(value, dict):
        return {k: _record(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return type(value)(_record(v) for v in value)
    return value


def _replay(cap: dict, n_paths: int) -> dict:
    out = {"wfo": [], "wfo_s": [], "mv": None, "mv_nulls": [], "mv_s": 0.0, "deploy": [], "deploy_s": 0.0}
    for call in cap["wfo"]:
        a, kw = call
        t0  = time.perf_counter()
        res = wfo.pipe_wfo(*a, **kw)
        out["wfo_s"].append((kw.get("combo_key", "?"), len(_rules_of(call)), time.perf_counter() - t0))
        out["wfo"].append(_record(res))

    if cap["multiverse"] is not None:
        a, kw = cap["multiverse"] if not n_paths else _with(cap["multiverse"], n_paths=n_paths)
        orig_p = mv._compute_p_value

        def p_capture(real_profit, permuted_profits):
            out["mv_nulls"].append((float(real_profit), np.array(permuted_profits, dtype=np.float64, copy=True)))
            return orig_p(real_profit, permuted_profits)

        t0 = time.perf_counter()
        with _patched(mv, _compute_p_value=p_capture):
            res = mv.pipe_multiverse(*a, **kw)
        out["mv_s"] = time.perf_counter() - t0
        out["mv"]   = _record(res)

    t0 = time.perf_counter()
    for a, kw in cap["deploy"]:
        out["deploy"].append(_record(rd.run_wfo_deploy_ema(*a, **kw)))
    out["deploy_s"] = time.perf_counter() - t0
    return out


def _same_float(a, b) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return struct.pack("<d", float(a)) == struct.pack("<d", float(b))


def _diff(a, b, path: str):
    # None when a and b are identical (same types, same values, same float bits), else the first difference
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, pd.DataFrame):
        if list(a.columns) != list(b.columns):
            return f"{path}: columns {list(a.columns)} vs {list(b.columns)}"
        if not a.index.equals(b.index) or a.index.dtype != b.index.dtype:
            return f"{path}: index differs"
        for col in a.columns:
            d = _diff(a[col].to_numpy(), b[col].to_numpy(), f"{path}[{col!r}]")
            if d:
                return d
        return None
    if isinstance(a, pd.Series):
        if not a.index.equals(b.index):
            return f"{path}: index differs"
        return _diff(a.to_numpy(), b.to_numpy(), path)
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"{path}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        if a.dtype == object:
            for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
                d = _diff(x, y, f"{path}[{i}]")
                if d:
                    return d
            return None
        if a.dtype.kind in "fc":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            same = np.array_equal(nan_a, nan_b) and a[~nan_a].tobytes() == b[~nan_b].tobytes()
            return None if same else f"{path}: values differ"
        return None if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes() else f"{path}: values differ"
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return f"{path}: keys {sorted(set(a) ^ set(b))} differ"
        for k in a:
            d = _diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a is b:
        return None
    if isinstance(a, (float, np.floating)):
        return None if _same_float(a, b) else f"{path}: {a!r} vs {b!r}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def _rule_label(out: dict, kind: str, i: int, j: int = None) -> str:
    if kind == "wfo":
        return f"WFO {out['wfo_s'][i][0]} rule {out['wfo'][i][j].get('rule_id', j)}"
    return f"MULTIVERSE rule {out['mv'][i].get('rule_id', i)}"


def _compare(base: dict, now: dict) -> list:
    issues = []
    if len(base["wfo"]) != len(now["wfo"]):
        return [f"WFO calls: {len(base['wfo'])} in baseline vs {len(now['wfo'])} now"]
    for i, (rb, rn) in enumerate(zip(base["wfo"], now["wfo"])):
        if len(rb) != len(rn):
            issues.append(f"WFO call {i}: {len(rb)} rules vs {len(rn)}")
            continue
        for j, (x, y) in enumerate(zip(rb, rn)):
            d = _diff(x, y, "")
            if d:
                issues.append(f"{_rule_label(now, 'wfo', i, j)}{d}")
    if (base["mv"] is None) != (now["mv"] is None):
        issues.append("multiverse ran in only one of the two runs")
    elif base["mv"] is not None:
        if base["mv_paths"] != now["mv_paths"]:
            issues.append(f"multiverse paths: {base['mv_paths']} in baseline vs {now['mv_paths']} now")
        for i, (x, y) in enumerate(zip(base["mv"], now["mv"])):
            d = _diff(x, y, "")
            if d:
                issues.append(f"{_rule_label(now, 'mv', i)}{d}")
        d = _diff(base["mv_nulls"], now["mv_nulls"], "MULTIVERSE per-path profits")
        if d:
            issues.append(d)
    d = _diff(base["deploy"], now["deploy"], "DEPLOY (params, symbols, train window)")
    if d:
        issues.append(d)
    return issues


def run_replay(args) -> None:
    cap = _load_capture(args.cap)
    n_paths = args.paths or (cap["multiverse"][1].get("n_paths", mv.N_PERMUTATIONS) if cap["multiverse"] else 0)
    if args.paths and (args.save_baseline or args.compare):
        print(" note: --paths changes the multiverse workload; baseline and compare must use the same value")

    print(f"\n{LINE}")
    print(f" CHECK WFO + MULTIVERSE + DEPLOY | mode {fx.settings.BACKTEST_MODE} | capture {cap['created']} | "
          f"multiverse paths {n_paths}")
    print(LINE)
    t0  = time.perf_counter()
    out = _replay(cap, args.paths)
    out["mv_paths"] = n_paths

    print(f"\n {'stage':<26}{'rules':>8}{'seconds':>10}")
    for combo_key, n_rules, secs in out["wfo_s"]:
        print(f" {'WFO ' + combo_key:<26}{n_rules:>8,}{secs:>10.1f}")
    if out["mv"] is not None:
        print(f" {'MULTIVERSE (all combos)':<26}{len(out['mv']):>8,}{out['mv_s']:>10.1f}")
    if cap["deploy"]:
        print(f" {'DEPLOY (EMA params)':<26}{len(out['deploy']):>8,}{out['deploy_s']:>10.1f}")
    if out["mv"] is not None:
        p_values = ", ".join(f"{r['multiverse_p_value']:.3f}" for r in out["mv"])
        print(f"\n multiverse p-values: {p_values}")
    print(f"\n total: {time.perf_counter() - t0:,.1f} s")

    failed = []
    if args.save_baseline:
        with open(args.save_baseline + ".pkl", "wb") as f:
            pickle.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                         "capture": cap["created"],
                         **{k: out[k] for k in ("wfo", "mv", "mv_nulls", "mv_paths", "deploy")}}, f)
        print(f"\n Baseline saved: {args.save_baseline}.pkl")
    if args.compare:
        with open(args.compare + ".pkl", "rb") as f:
            base = pickle.load(f)
        if base["capture"] != cap["created"]:
            print(f"\n WARNING: baseline made from capture {base['capture']}, this capture is {cap['created']}")
        failed = _compare(base, out)

    n_wfo = sum(len(r) for r in out["wfo"])
    print(f"\n VERIFICATION")
    if failed:
        for msg in failed[:20]:
            print(f"   DIFFERENT ── {msg}")
        if len(failed) > 20:
            print(f"   ... {len(failed) - 20} more")
    elif args.compare:
        print(f"   identical: WFO {n_wfo:,} rules (every output field, trades bit for bit) ── multiverse "
              f"{len(out['mv'] or []):,} rules x {n_paths:,} paths (profit of every path, p-values) ── "
              f"deploy {len(out['deploy']):,} rules (params, symbols, train window)")
    else:
        print("   nothing to verify: use --compare")
    print(LINE)
    if failed:
        sys.exit(1)


# =============================================================================
# PROFILE
# =============================================================================
def _print_stats(profile: cProfile.Profile, title: str, top: int) -> None:
    stats = pstats.Stats(profile).strip_dirs()
    total = stats.total_tt
    print(f"\n {title} ── {total:,.2f} s in one process")
    for key, label in (("tottime", "own time"), ("cumulative", "time including callees")):
        print(f"\n   top {top} by {label}")
        stats.sort_stats(key).print_stats(top)


def run_profile(args) -> None:
    cap = _load_capture(args.cap)
    print(f"\n{LINE}")
    print(f" PROFILE WFO + MULTIVERSE | mode {fx.settings.BACKTEST_MODE} | capture {cap['created']}")
    print(LINE)
    if cap["multiverse"] is not None:
        a, kw = _with(cap["multiverse"], n_paths=args.paths, n_jobs=1)
        prof = cProfile.Profile()
        prof.runcall(mv.pipe_multiverse, *a, **kw)
        _print_stats(prof, f"MULTIVERSE {len(_rules_of(cap['multiverse']))} rules x {args.paths} paths", args.top)
    if cap["wfo"]:
        rules = _rules_of(cap["wfo"][0])[:args.wfo_rules]
        a, kw = _with(cap["wfo"][0], rules=rules, rules_n_jobs=1)
        prof = cProfile.Profile()
        prof.runcall(wfo.pipe_wfo, *a, **kw)
        _print_stats(prof, f"WFO {kw.get('combo_key', '')} {len(rules)} rules", args.top)
    print(LINE)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = _parse_args()
    if args.cmd == "capture":
        run_capture(args)
    elif args.cmd == "run":
        run_replay(args)
    else:
        run_profile(args)


if __name__ == "__main__":
    main()