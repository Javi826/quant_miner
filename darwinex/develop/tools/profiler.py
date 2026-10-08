#BOT_batch/profiler.py
"""
Perfilado por fases del backtest IS con datos reales: réplica monoproceso + pasada real con joblib.

Modos
  --list          lista los combos de build_combos() con su rejilla
  --all-combos    réplica monoproceso sobre todos los combos, una línea por combo, ordenada por CPU total estimada
  (por defecto)   réplica por fases sobre un combo, pasada real con joblib y verificación bit a bit

Opciones
  --combo KEY            combo a perfilar (por defecto el primero de build_combos())
  --n-rules N            reglas de la muestra, 0 = todas (por defecto 200)
  --contiguous           muestra contigua (bloque central) en vez de repartida uniformemente
  --min-of K             cronometra cada regla K veces y toma el mínimo por fase (robusto con la máquina cargada)
  --cpu-time             cronometra las fases con thread_time_ns (descuenta tiempo desplanificado, no la caché fría)
  --repeat N             N pasadas reales seguidas en el mismo proceso; la última es la limpia (executor e imports calientes)
  --no-full              sin pasada real
  --save-baseline RUTA   guarda RUTA.pkl y RUTA.npy con la salida de la última pasada real
  --compare RUTA         compara bit a bit la última pasada real con RUTA.pkl / RUTA.npy

Flujo típico (desde Spyder, editando DEFAULT_ARGS; desde terminal, pasando los argumentos)
  1. --all-combos --n-rules 100                                   -> elegir el combo que más CPU suma
  2. --combo KEY --n-rules 0 --repeat 2 --save-baseline base_KEY  (máquina libre)
  3. --combo KEY --n-rules 0 --repeat 2 --compare base_KEY        tras cada cambio
"""
import os
import sys
import gc
import math
import time
import pickle
import struct
import hashlib
import argparse
import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import backtesting_fx as fx  # fija sys.path, logging y la configuración; su bloque main no corre al importar

import pipeline.backtest_runner as br
from pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
from pipeline.spec_table import build_spec_table, rule_spec_index
from rule_mining.rule_runner import build_rule_templates, build_rule_dicts
from utils.batch_metrics import sharpes_from_daily_rows

_wall = time.perf_counter_ns   # etapas y pasadas reales: siempre reloj de pared
_ns   = time.perf_counter_ns   # fases por regla: pasa a thread_time_ns con --cpu-time
LINE  = "=" * 110

DEFAULT_ARGS = "--all-combos --n-rules 100"   # se usa cuando no llegan argumentos (Run de Spyder)

METRIC_KEYS = list(br._empty_winner_metrics().keys()) + ["best_combo_id"]

CTX_PHASES = [
    ("prepare_static_arrays",             "ctx.prepare_static_arrays"),
    ("Calendario de trading",             "ctx.calendar"),
    ("day_2d (1a llamada market_arrays)", "ctx.day_2d"),
]
RULE_PHASES = [
    ("Eventos (tabla de specs)",          "rule.events"),
    ("Motor (backtest_grid)",             "rule.engine"),
    ("Post: sharpe",                      "rule.post_sharpe"),
    ("Post: escritura al segmento",       "rule.post_segment"),
    ("Post: resto del bucle",             "rule.post_loop"),
    ("Métricas del ganador",              "rule.winner"),
]
BLOCK_PHASES = [
    ("Índices de specs",                  "block.spec_index"),
    ("Creación de segmentos",             "block.create_segments"),
    ("Máscara de días",                   "block.day_mask"),
    ("Gather final",                      "final.gather"),
]


# =============================================================================
# ARGS / DATA
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="Perfilado por fases de pipe_backtesting con datos reales")
    p.add_argument("--list", action="store_true", help="lista los combos disponibles y sale")
    p.add_argument("--all-combos", action="store_true", help="réplica monoproceso sobre todos los combos (sin pasada real)")
    p.add_argument("--combo", default=None, help="combo_key (por defecto, el primero de build_combos())")
    p.add_argument("--n-rules", type=int, default=200, help="reglas de la muestra (0 = todas)")
    p.add_argument("--contiguous", action="store_true", help="muestra contigua (bloque central) en vez de uniforme")
    p.add_argument("--min-of", type=int, default=1, help="repeticiones por regla; se toma el mínimo por fase")
    p.add_argument("--cpu-time", action="store_true", help="cronometrar fases con thread_time_ns")
    p.add_argument("--repeat", type=int, default=1, help="pasadas reales seguidas (la última es la limpia)")
    p.add_argument("--no-full", action="store_true", help="no ejecutar la pasada real con joblib")
    p.add_argument("--save-baseline", default=None, metavar="RUTA", help="guarda RUTA.pkl y RUTA.npy con la última pasada real")
    p.add_argument("--compare", default=None, metavar="RUTA", help="compara bit a bit la última pasada real con RUTA.pkl / RUTA.npy")
    argv = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ARGS.split()
    return p.parse_args(argv)


def _find_combo(combo_key):
    combos = fx.build_combos()
    if combo_key is None:
        return combos[0]
    for c in combos:
        if c["combo_key"] == combo_key:
            return c
    raise SystemExit(f"combo_key desconocido: {combo_key}. Disponibles: {[c['combo_key'] for c in combos]}")


def _load_ohlcv(combo):
    _, arr_by_combo = fx.load_ohlcv_by_combo([combo], fx.DATA_FOLDER_BY_DATASET[fx.DATASET_IS], fx.DATASET_IS)
    return arr_by_combo[combo["combo_key"]]


def _n_combos(param_grid) -> int:
    return math.prod(len(v) for v in param_grid.values())


def _build_rules(combo, ohlcv_arr, stages) -> tuple:
    timeframe = combo["timeframe"]
    t0 = _wall()
    templates = build_rule_templates(ohlcv_arr, indicators=fx.SELECTED_INDICATORS_BY_TIMEFRAME[timeframe],
                                     max_depth=fx.RULE_MAX_DEPTH)
    rules = build_rule_dicts(templates, combo["combo_key"], timeframe)
    stages["Generación de reglas"] = _wall() - t0
    n_generated = len(rules)

    t0 = _wall()
    spec_table = build_spec_table(rules=rules, ohlcv_arr=ohlcv_arr, timeframe=timeframe)
    stages["Tabla de specs"] = _wall() - t0

    t0 = _wall()
    rules = pipe_signal_cleaning_jaccard(rules=rules, ohlcv_arr=ohlcv_arr, timeframe=timeframe, spec_table=spec_table)
    stages["Limpieza Jaccard"] = _wall() - t0
    return rules, n_generated, spec_table


def _sample_rules(rules, n, contiguous):
    if n <= 0 or n >= len(rules):
        return list(rules)
    if contiguous:
        start = (len(rules) - n) // 2
        return list(rules[start:start + n])
    idx = np.unique(np.linspace(0, len(rules) - 1, n).round().astype(np.int64))
    return [rules[i] for i in idx]


def _ctx_cacheable(ohlcv_arr) -> bool:
    # Misma forma de metadata que arrays_to_shared_memory, sin crear segmentos: si la clave es None,
    # cada tarea reconstruye el contexto del worker.
    meta = {
        sym: {k: ({"name": ""} if isinstance(v, np.ndarray) else {"value": v}) for k, v in d.items()}
        for sym, d in ohlcv_arr.items()
    }
    return br._static_bundle_cache_key(meta) is not None


def _rules_per_task(n_rules, n_workers) -> int:
    return max(1, min(br.BACKTEST_MAX_RULES_PER_TASK, -(-n_rules // (n_workers * br.BACKTEST_TASKS_PER_WORKER))))


def _velas_desc(ohlcv_arr) -> str:
    lens = sorted(len(arr["ts"]) for arr in ohlcv_arr.values())
    if lens[0] == lens[-1]:
        return f"{len(lens)} símbolos x {lens[0]:,} velas"
    return f"{len(lens)} símbolos, {lens[0]:,} .. {lens[-1]:,} velas"


# =============================================================================
# SINGLE-PROCESS REPLICA OF run_full_period_search (timed by phase)
# =============================================================================
def _profile_replica(rules, ohlcv_arr, spec_table, param_grid, order_amount, n_days_range, min_of):
    acc, cnt, info = defaultdict(int), defaultdict(int), {}

    combos      = br._combo_grid(param_grid)
    n_combos    = len(combos)
    combo_ids   = [br._combo_id(p) for p in combos]
    engine_grid = br._engine_grid(combos)
    min_trades  = br.BACKTEST_MIN_TRADES
    balance     = float(br.INITIAL_BALANCE)
    comi_factor = float(br.COMISION) / 100.0
    neg_inf     = -np.inf

    # Contexto del worker (en la pasada real se construye una vez por worker)
    t0 = _ns()
    static_bundle = br.prepare_static_arrays(ohlcv_arr)
    t1 = _ns()
    static_bundle["calendar"] = br._trading_calendar(ohlcv_arr, br._global_day_grid(ohlcv_arr)[0])
    t2 = _ns()
    br.market_arrays(static_bundle)  # la primera llamada construye y cachea day_2d
    t3 = _ns()
    acc["ctx.prepare_static_arrays"] += t1 - t0
    acc["ctx.calendar"]              += t2 - t1
    acc["ctx.day_2d"]                += t3 - t2
    cnt["ticks"] = int(static_bundle["all_timestamps_int"].shape[0])
    br.check_spec_table_layout(spec_table.symbols, spec_table.n_bars, static_bundle)
    info["spec_table"] = (spec_table.n_specs, int(spec_table.words.shape[1]), spec_table.words.nbytes)

    t0 = _ns()
    spec_idx, sides = rule_spec_index(rules, spec_table.index_by_identity)
    is_short = sides != "long"
    acc["block.spec_index"] += _ns() - t0

    def run_rule(rule_idx, seg_rows, seg_cols):
        # Copia literal de _run_full_period_for_rule con un cronómetro por fase. Devuelve (resultado, fases, contadores).
        ph, st = {}, {}

        t0 = _ns()
        signal_events, ev_short, timeline = br.build_rule_events_from_words(
            spec_table.words, spec_table.word_offsets, spec_idx[rule_idx], bool(is_short[rule_idx]),
            static_bundle["ts_int_2d"], static_bundle["sym_len"],
            static_bundle["tick_pos_2d"], static_bundle["all_timestamps_int"], static_bundle["idx_workspace"],
        )
        ph["rule.events"] = _ns() - t0

        n_events     = int(signal_events.shape[0])
        st["events"] = n_events
        if n_events:
            _, counts = np.unique(signal_events[:, 0], return_counts=True)
            st["ts_distinct"]      = int(counts.shape[0])
            st["events_shared_ts"] = int(counts[counts > 1].sum())
            st["max_batch"]        = int(counts.max())

        if n_events < min_trades:
            st["rules_skipped"] = 1
            return (rule_idx, {**br._empty_winner_metrics(), "best_combo_id": combo_ids[0]}), ph, st

        t0 = _ns()
        n_trades, day_start, n_days, n_nonzero, daily, duration = br.backtest_grid(
            br.market_arrays(static_bundle), signal_events, ev_short, timeline,
            *engine_grid,
            balance, comi_factor, order_amount, min_trades,
            *static_bundle["calendar"], n_days_range,
        )
        ph["rule.engine"]  = _ns() - t0
        st["rules_engine"] = 1
        st["trades"]       = int(n_trades.sum())

        t_loop0  = _ns()
        t_seg    = 0
        col_base = rule_idx * n_combos
        sharpes  = sharpes_from_daily_rows(daily, day_start, n_days)
        t_sharpe = _ns() - t_loop0

        best_rank   = None
        best_idx    = 0
        best_valid  = False
        best_sharpe = np.nan

        for combo_idx in range(n_combos):
            combo_trades = int(n_trades[combo_idx])
            if combo_trades == 0 or combo_trades < min_trades:
                rank, valid, sharpe_metric = neg_inf, False, np.nan
            else:
                start        = int(day_start[combo_idx])
                stop         = start + int(n_days[combo_idx])
                daily_values = daily[combo_idx, start:stop]

                sharpe_metric = sharpes[combo_idx]
                rank  = sharpe_metric if math.isfinite(sharpe_metric) else neg_inf
                valid = True
                st["combos_valid"] = st.get("combos_valid", 0) + 1
                st["trades_valid"] = st.get("trades_valid", 0) + combo_trades

                if n_nonzero[combo_idx] > 1 and seg_rows is not None:
                    ts0 = _ns()
                    seg_rows[len(seg_cols), start:stop] = daily_values
                    seg_cols.append(col_base + combo_idx)
                    t_seg += _ns() - ts0

            if best_rank is None or rank > best_rank:
                best_rank, best_idx, best_valid, best_sharpe = rank, combo_idx, valid, sharpe_metric

        t_loop = _ns() - t_loop0
        ph["rule.post_sharpe"]  = t_sharpe
        ph["rule.post_segment"] = t_seg
        ph["rule.post_loop"]    = t_loop - t_sharpe - t_seg

        t0 = _ns()
        if not best_valid:
            winner_metrics = br._empty_winner_metrics()
        else:
            best_trades      = int(n_trades[best_idx])
            best_start       = int(day_start[best_idx])
            best_n_days      = int(n_days[best_idx])
            best_duration_is = float(np.mean(duration[best_idx, :best_trades])) / 1e9 / 86400.0
            winner_metrics = br._winner_metrics_from_daily_values(
                daily[best_idx, best_start:best_start + best_n_days], best_n_days, best_sharpe, best_duration_is,
            )
        ph["rule.winner"] = _ns() - t0
        return (rule_idx, {**winner_metrics, "best_combo_id": combo_ids[best_idx]}), ph, st

    def timed_rule(rule_idx, seg_rows, seg_cols):
        # Con --min-of K la regla corre K veces: mismo resultado, mínimo por fase. El segmento se reescribe con
        # los mismos valores en las mismas filas, así que la salida no cambia.
        n0   = len(seg_cols)
        best = None
        for _ in range(max(1, min_of)):
            del seg_cols[n0:]
            result, ph, st = run_rule(rule_idx, seg_rows, seg_cols)
            best = ph if best is None else {k: min(best[k], v) for k, v in ph.items()}
        for k, v in best.items():
            acc[k] += v
        for k, v in st.items():
            if k == "max_batch":
                cnt[k] = max(cnt[k], v)
            else:
                cnt[k] += v
        return result

    # Calentamiento con la primera regla, descartado
    run_rule(0, None, None)

    # Bloques y segmentos, dimensionados como en run_full_period_search
    n_rules        = len(rules)
    row_size       = n_days_range * np.dtype(np.float32).itemsize
    n_workers      = max(1, br.effective_n_jobs(br.BACKTEST_N_JOBS))
    rules_per_task = _rules_per_task(n_rules, n_workers)
    blocks         = [(i, rules[i:i + rules_per_task]) for i in range(0, n_rules, rules_per_task)]

    pending      = []
    results      = []
    seg_cols_all = []
    day_mask     = np.zeros(n_days_range, dtype=bool)
    try:
        t0 = _ns()
        for _, block_rules in blocks:
            pending.append(br._create_segment(len(block_rules) * n_combos * row_size))
        acc["block.create_segments"] += _ns() - t0

        for (rule_start, block_rules), seg_name in zip(blocks, pending):
            seg      = SharedMemory(name=seg_name, create=False)
            seg_rows = None
            try:
                seg_rows = np.ndarray((len(block_rules) * n_combos, n_days_range), dtype=np.float32, buffer=seg.buf)
                seg_cols = []
                for offset in range(len(block_rules)):
                    results.append(timed_rule(rule_start + offset, seg_rows, seg_cols))

                t0 = _ns()
                n_valid    = len(seg_cols)
                block_mask = np.zeros(n_days_range, dtype=bool)
                for start in range(0, n_valid, br.MATRIX_GATHER_ROWS):
                    block_mask |= np.any(seg_rows[start:min(start + br.MATRIX_GATHER_ROWS, n_valid)] != 0, axis=0)
                acc["block.day_mask"] += _ns() - t0
            finally:
                seg_rows = None
                seg.close()
            seg_cols_all.append(np.asarray(seg_cols, dtype=np.int64))
            day_mask |= block_mask

        t0 = _ns()
        valid_cols = np.concatenate(seg_cols_all) if seg_cols_all else np.empty(0, dtype=np.int64)
        n_valid    = valid_cols.shape[0]
        if n_valid == 0:
            matrix_arr = np.empty((n_days_range, 0), dtype=np.float32)
            for name in pending:
                br._unlink_segment(name)
        else:
            days       = np.flatnonzero(day_mask)
            matrix_arr = np.empty((days.shape[0], n_valid), dtype=np.float32)
            col_starts = np.concatenate(([0], np.cumsum([c.shape[0] for c in seg_cols_all])[:-1]))
            n_threads  = max(1, min(br.MATRIX_GATHER_THREADS, os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                list(pool.map(
                    lambda args: br._gather_segment(args[0], args[1], n_days_range, days, matrix_arr, args[2]),
                    [(name, c.shape[0], int(k0)) for name, c, k0 in zip(pending, seg_cols_all, col_starts)],
                ))
        acc["final.gather"] += _ns() - t0
        pending = []
    finally:
        for name in pending:
            br._unlink_segment(name)

    metrics   = {rules[rule_idx]["rule_id"]: m for rule_idx, m in results}
    col_names = [f"{rules[c // n_combos]['rule_id']}__{combo_ids[c % n_combos]}" for c in valid_cols.tolist()]
    cnt["matrix_cols"] = len(col_names)
    return metrics, matrix_arr, col_names, n_combos, acc, cnt, info


# =============================================================================
# REAL PASSES (joblib)
# =============================================================================
def _real_passes(sample, ohlcv_arr, spec_table, param_grid, timeframe, n_repeat) -> list:
    # Cada elemento: (wall_ns, metrics, matrix, col_names). La primera pasada paga spawn e imports de los workers.
    passes = []
    for _ in range(max(1, n_repeat)):
        t0 = _wall()
        raw_results, _, matrix, cols = br.pipe_backtesting(
            rules        = sample,
            ohlcv_arr    = ohlcv_arr,
            param_grid   = param_grid,
            order_amount = fx.ORDER_AMOUNT,
            timeframe    = timeframe,
            spec_table   = spec_table,
        )
        wall    = _wall() - t0
        metrics = {r["rule_id"]: {k: r[k] for k in METRIC_KEYS} for r in raw_results}
        passes.append((wall, metrics, matrix, cols))
    return passes


# =============================================================================
# COMPARISON / BASELINE
# =============================================================================
def _same_value(a, b) -> bool:
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return struct.pack("<d", fa) == struct.pack("<d", fb)


def _compare(metrics_a, matrix_a, cols_a, metrics_b, matrix_b, cols_b) -> list:
    issues = []
    ids_a, ids_b = list(metrics_a), list(metrics_b)
    if ids_a != ids_b:
        issues.append(f"rule_ids distintos ({len(ids_a)} vs {len(ids_b)})")

    diff = []
    for rid in ids_a:
        if rid not in metrics_b:
            continue
        ma, mb = metrics_a[rid], metrics_b[rid]
        keys = [k for k in METRIC_KEYS if k not in ma or k not in mb or not _same_value(ma[k], mb[k])]
        if keys:
            diff.append((rid, keys))
    if diff:
        rid, keys = diff[0]
        issues.append(f"{len(diff)} reglas con métricas distintas (primera: {rid} en {keys})")

    if list(cols_a) != list(cols_b):
        issues.append(f"col_names distintos ({len(cols_a)} vs {len(cols_b)})")

    if matrix_a.shape != matrix_b.shape:
        issues.append(f"matriz con forma distinta {matrix_a.shape} vs {matrix_b.shape}")
    else:
        bits_a = np.ascontiguousarray(matrix_a).view(np.uint32)
        bits_b = np.ascontiguousarray(matrix_b).view(np.uint32)
        n_diff = int(np.count_nonzero(bits_a != bits_b))
        if n_diff:
            diff_abs = np.abs(matrix_a.astype(np.float64) - matrix_b.astype(np.float64))
            max_abs  = float(np.nanmax(diff_abs)) if np.isfinite(diff_abs).any() else float("nan")
            issues.append(f"matriz: {n_diff:,} celdas distintas bit a bit (máx. |dif| {max_abs:.3e})")
    return issues


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _save_baseline(stem, meta, metrics, matrix, col_names) -> None:
    with open(stem + ".pkl", "wb") as f:
        pickle.dump({
            "meta":          meta,
            "metrics":       metrics,
            "col_names":     list(col_names),
            "matrix_shape":  tuple(matrix.shape),
            "matrix_sha256": _sha256(matrix),
        }, f)
    np.save(stem + ".npy", matrix)


def _load_baseline(stem) -> dict:
    with open(stem + ".pkl", "rb") as f:
        base = pickle.load(f)
    base["matrix"] = np.load(stem + ".npy")
    return base


def _print_issues(label, issues) -> None:
    if issues:
        print(f"   {label:<36}: DIFERENTES")
        for msg in issues:
            print(f"     . {msg}")
    else:
        print(f"   {label:<36}: idénticas bit a bit")


# =============================================================================
# REPORT (single combo)
# =============================================================================
def _print_report(info, stages, acc, cnt, passes) -> None:
    n_sample  = info["n_sample"]
    n_combos  = info["n_combos"]
    n_workers = info["n_workers"]
    n_total   = info["n_total"]

    print(f"\n{LINE}")
    print(f" PERFIL BACKTEST IS | {info['combo_key']} | {info['timeframe']} | modo {info['mode']} | {info['symbols']}")
    print(LINE)
    print(f" Velas                  : {info['velas']}")
    print(f" Timeline global        : {cnt['ticks']:,} velas | n_days_range: {info['n_days_range']:,}")
    print(f" Reglas                 : {info['n_generated']:,} generadas | {n_total:,} tras Jaccard | muestra {n_sample:,} ({info['sample_mode']})")
    print(f" Grid                   : {n_combos} combos {info['param_grid']}")
    print(f" Workers                : {n_workers}")
    print(f" Cronómetro de fases    : {info['clock']}" + (f" | mínimo de {info['min_of']} repeticiones por regla" if info["min_of"] > 1 else ""))
    n_specs, n_words, n_bytes = info["spec_table"]
    print(f" Tabla de specs         : {n_specs:,} specs x {n_words:,} palabras de 64 bits ({n_bytes / 2**20:,.1f} MB)")
    print(f" Contexto del worker    : "
          f"{'cacheable entre tareas' if info['ctx_cacheable'] else 'NO cacheable, se reconstruye en cada tarea'}")

    print(f"\n ETAPAS DEL COMBO (una vez, todas las reglas)")
    for label, v in stages.items():
        print(f"   {label:<36}{v / 1e9:>10.2f} s")

    print(f"\n CONTEXTO DEL WORKER (una vez por worker)")
    for label, key in CTX_PHASES:
        print(f"   {label:<36}{acc[key] / 1e6:>10.1f} ms")

    rule_ns  = sum(acc[k] for _, k in RULE_PHASES)
    block_ns = sum(acc[k] for _, k in BLOCK_PHASES)
    base     = (rule_ns + block_ns) or 1
    print(f"\n POR REGLA (muestra de {n_sample:,} reglas, un solo proceso)")
    print(f"   {'Fase':<36}{'total ms':>10}{'ms/regla':>11}{'µs/combo':>11}{'%':>8}")
    for label, key in RULE_PHASES + BLOCK_PHASES:
        v = acc[key]
        print(f"   {label:<36}{v / 1e6:>10.1f}{v / 1e6 / n_sample:>11.3f}"
              f"{v / 1e3 / (n_sample * n_combos):>11.2f}{100.0 * v / base:>7.1f}%")
    print(f"   {'TOTAL':<36}{base / 1e6:>10.1f}{base / 1e6 / n_sample:>11.3f}"
          f"{base / 1e3 / (n_sample * n_combos):>11.2f}{100.0:>7.1f}%")

    n_engine = cnt["rules_engine"]
    print(f"\n CONTADORES")
    print(f"   Eventos medios por regla          : {cnt['events'] / n_sample:,.0f}")
    if cnt["events"]:
        print(f"   Eventos por timestamp distinto    : {cnt['events'] / max(cnt['ts_distinct'], 1):.2f} | "
              f"con timestamp compartido: {100.0 * cnt['events_shared_ts'] / cnt['events']:.0f}% | "
              f"lote máximo posible: {cnt['max_batch']}")
    print(f"   Reglas que llegan al motor        : {n_engine:,} de {n_sample:,} (resto con < {br.BACKTEST_MIN_TRADES} eventos)")
    if n_engine:
        print(f"   Combos válidos                    : {cnt['combos_valid']:,} de {n_engine * n_combos:,}")
        print(f"   Trades medios por combo (motor)   : {cnt['trades'] / (n_engine * n_combos):,.0f}")
    if cnt["combos_valid"]:
        print(f"   Trades medios por combo válido    : {cnt['trades_valid'] / cnt['combos_valid']:,.0f}")
    if cnt["trades"]:
        print(f"   Motor por trade                   : {acc['rule.engine'] / cnt['trades']:,.0f} ns "
              f"(incluye reservas, memo y costes fijos por regla)")
    print(f"   Columnas de la matriz             : {cnt['matrix_cols']:,}")

    ms_rule    = rule_ns / 1e6 / n_sample
    ctx_ns     = sum(acc[k] for _, k in CTX_PHASES)
    rpt_sample = _rules_per_task(n_sample, n_workers)
    rpt_total  = _rules_per_task(n_total, n_workers)
    print(f"\n PARALELIZACIÓN")
    print(f"   Muestra : {rpt_sample:>4} reglas/tarea | {-(-n_sample // rpt_sample):>6,} tareas | {rpt_sample * ms_rule:>9,.1f} ms de trabajo por tarea")
    print(f"   Total   : {rpt_total:>4} reglas/tarea | {-(-n_total // rpt_total):>6,} tareas | {rpt_total * ms_rule:>9,.1f} ms de trabajo por tarea")
    print(f"   CPU total estimada, {n_total:,} reglas en un core   : {ms_rule * n_total / 1e3:,.1f} s")
    print(f"   Ideal con {n_workers} workers, sin overhead          : {ms_rule * n_total / n_workers / 1e3 + ctx_ns / 1e9:,.2f} s")
    if passes:
        ideal_sample = base / n_workers + ctx_ns
        last         = passes[-1][0]
        print(f"   Pasadas reales sobre la muestra   : " + " | ".join(f"{w / 1e9:,.2f} s" for w, *_ in passes))
        print(f"   Última pasada                     : {last / 1e9:,.2f} s ({last / 1e6 / n_sample:,.3f} ms/regla) | "
              f"eficiencia {100.0 * ideal_sample / last:,.0f}%")
        if len(passes) == 1:
            print(f"   (una sola pasada: incluye spawn de workers e imports; con --repeat 2 o más la última es la limpia)")


# =============================================================================
# ALL COMBOS SUMMARY
# =============================================================================
def _list_combos() -> None:
    print(f"\n {'combo':<14}{'tf':<6}{'símb':>5}{'combos':>8}  símbolos")
    for c in fx.build_combos():
        grid = fx.PARAM_GRID_BY_TIMEFRAME[c["timeframe"]]
        print(f" {c['combo_key']:<14}{c['timeframe']:<6}{len(c['symbols']):>5}{_n_combos(grid):>8}  {c['symbols']}")


def _summarize_combos(args) -> None:
    n_workers = max(1, br.effective_n_jobs(br.BACKTEST_N_JOBS))
    rows = []
    for combo in fx.build_combos():
        stages     = {}
        ohlcv_arr  = _load_ohlcv(combo)
        rules, n_generated, spec_table = _build_rules(combo, ohlcv_arr, stages)
        sample     = _sample_rules(rules, args.n_rules, args.contiguous)
        param_grid = fx.PARAM_GRID_BY_TIMEFRAME[combo["timeframe"]]
        row = {
            "combo":     combo["combo_key"],
            "tf":        combo["timeframe"],
            "syms":      len(combo["symbols"]),
            "velas":     max(len(arr["ts"]) for arr in ohlcv_arr.values()),
            "n_combos":  _n_combos(param_grid),
            "n_gen":     n_generated,
            "n_rules":   len(rules),
            "table_s":   stages["Tabla de specs"] / 1e9,
            "jaccard_s": stages["Limpieza Jaccard"] / 1e9,
            "ms_rule":   0.0, "cpu_s": 0.0, "events": 0.0, "engine": 0.0, "post": 0.0,
            "ev_rule":   0.0, "tr_combo": 0.0,
        }
        if sample:
            _, n_days_range = br._global_day_grid(ohlcv_arr)
            _, _, _, _, acc, cnt, _ = _profile_replica(
                sample, ohlcv_arr, spec_table, param_grid, float(fx.ORDER_AMOUNT), n_days_range, args.min_of,
            )
            n_sample = len(sample)
            rule_ns  = sum(acc[k] for _, k in RULE_PHASES) or 1
            row.update({
                "ms_rule":  rule_ns / 1e6 / n_sample,
                "cpu_s":    rule_ns / 1e9 / n_sample * len(rules),
                "events":   100.0 * acc["rule.events"] / rule_ns,
                "engine":   100.0 * acc["rule.engine"] / rule_ns,
                "post":     100.0 * (rule_ns - acc["rule.events"] - acc["rule.engine"]) / rule_ns,
                "ev_rule":  cnt["events"] / n_sample,
                "tr_combo": cnt["trades"] / max(cnt["rules_engine"] * row["n_combos"], 1),
            })
        rows.append(row)
        del ohlcv_arr, rules, sample, spec_table
        gc.collect()

    rows.sort(key=lambda r: r["cpu_s"], reverse=True)
    total_cpu = sum(r["cpu_s"] for r in rows) or 1.0
    total_jac = sum(r["jaccard_s"] for r in rows)
    total_tab = sum(r["table_s"] for r in rows)

    print(f"\n{LINE}")
    print(f" RESUMEN POR COMBO | modo {fx.settings.BACKTEST_MODE} | muestra {args.n_rules} reglas/combo, un solo proceso | {n_workers} workers")
    print(LINE)
    hdr = (f" {'combo':<12}{'tf':<5}{'símb':>5}{'velas':>8}{'combos':>7}{'reglas gen':>11}{'jaccard':>8}"
           f"{'ev/regla':>9}{'tr/combo':>9}{'ms/regla':>9}{'CPU s':>8}{'% tot':>7}{'event':>7}{'motor':>7}{'post':>7}{'tabla s':>8}{'jacc s':>8}")
    print(hdr)
    for r in rows:
        print(f" {r['combo']:<12}{r['tf']:<5}{r['syms']:>5}{r['velas']:>8,}{r['n_combos']:>7}{r['n_gen']:>11,}{r['n_rules']:>8,}"
              f"{r['ev_rule']:>9,.0f}{r['tr_combo']:>9,.0f}{r['ms_rule']:>9.2f}{r['cpu_s']:>8.1f}{100.0 * r['cpu_s'] / total_cpu:>6.0f}%"
              f"{r['events']:>6.0f}%{r['engine']:>6.0f}%{r['post']:>6.0f}%{r['table_s']:>8.1f}{r['jaccard_s']:>8.1f}")
    print(f"\n CPU total del backtest (un core)    : {total_cpu:,.1f} s | ideal con {n_workers} workers: {total_cpu / n_workers:,.1f} s")
    print(f" Tabla de specs, suma de los combos  : {total_tab:,.1f} s (el primero incluye el spawn de workers)")
    print(f" Jaccard, suma de todos los combos   : {total_jac:,.1f} s")
    if rows and rows[0]["cpu_s"] > 0:
        top = rows[0]
        print(f"\n Combo que más pesa: {top['combo']} ({100.0 * top['cpu_s'] / total_cpu:.0f}% del CPU del backtest)")
        print(f"   profiler.py --combo {top['combo']} --n-rules 0 --repeat 2 --save-baseline base_{top['combo']}")
    print(LINE)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    global _ns
    args = _parse_args()
    if args.cpu_time:
        _ns = time.thread_time_ns

    if args.list:
        _list_combos()
        return
    if br.backtest_grid is None:
        raise SystemExit(f"ZX_compute_BT_{fx.settings.BACKTEST_MODE} no implementa backtest_grid")
    if br.build_rule_events_from_words is None:
        raise SystemExit(f"ZX_compute_BT_{fx.settings.BACKTEST_MODE} no implementa build_rule_events_from_words")
    if args.all_combos:
        _summarize_combos(args)
        return
    if args.no_full and (args.save_baseline or args.compare):
        raise SystemExit("--save-baseline y --compare necesitan la pasada real: quita --no-full")

    stages = {}
    t0 = _wall()
    combo     = _find_combo(args.combo)
    ohlcv_arr = _load_ohlcv(combo)
    stages["Carga de datos"] = _wall() - t0
    timeframe  = combo["timeframe"]
    param_grid = fx.PARAM_GRID_BY_TIMEFRAME[timeframe]

    rules, n_generated, spec_table = _build_rules(combo, ohlcv_arr, stages)
    sample = _sample_rules(rules, args.n_rules, args.contiguous)
    if not sample:
        raise SystemExit("No quedan reglas tras la limpieza Jaccard")
    _, n_days_range = br._global_day_grid(ohlcv_arr)

    rep_metrics, rep_matrix, rep_cols, n_combos, acc, cnt, rep_info = _profile_replica(
        sample, ohlcv_arr, spec_table, param_grid, float(fx.ORDER_AMOUNT), n_days_range, args.min_of,
    )

    passes = [] if args.no_full else _real_passes(sample, ohlcv_arr, spec_table, param_grid, timeframe, args.repeat)

    syms = combo["symbols"]
    info = {
        **rep_info,
        "combo_key":     combo["combo_key"],
        "symbols":       syms if len(syms) <= 8 else syms[:8] + ["..."],
        "timeframe":     timeframe,
        "mode":          fx.settings.BACKTEST_MODE,
        "velas":         _velas_desc(ohlcv_arr),
        "n_days_range":  int(n_days_range),
        "n_generated":   n_generated,
        "n_total":       len(rules),
        "n_sample":      len(sample),
        "sample_mode":   "todas" if len(sample) == len(rules) else ("contigua" if args.contiguous else "uniforme"),
        "n_combos":      n_combos,
        "param_grid":    param_grid,
        "n_workers":     max(1, br.effective_n_jobs(br.BACKTEST_N_JOBS)),
        "clock":         "thread_time_ns (CPU del hilo)" if args.cpu_time else "perf_counter_ns (pared)",
        "min_of":        max(1, args.min_of),
        "ctx_cacheable": _ctx_cacheable(ohlcv_arr),
    }
    _print_report(info, stages, acc, cnt, passes)

    if passes:
        print(f"\n VERIFICACIÓN")
        _, last_metrics, last_matrix, last_cols = passes[-1]
        _print_issues("Réplica vs última pasada real", _compare(rep_metrics, rep_matrix, rep_cols, last_metrics, last_matrix, last_cols))
        for i in range(1, len(passes)):
            _, ma, xa, ca = passes[i - 1]
            _, mb, xb, cb = passes[i]
            _print_issues(f"Pasada {i} vs pasada {i + 1}", _compare(ma, xa, ca, mb, xb, cb))

        meta = {
            "combo_key":     combo["combo_key"],
            "timeframe":     timeframe,
            "symbols":       syms,
            "rule_ids":      [r["rule_id"] for r in sample],
            "param_grid":    param_grid,
            "backtest_mode": fx.settings.BACKTEST_MODE,
            "n_days_range":  int(n_days_range),
            "created":       datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if args.save_baseline:
            _save_baseline(args.save_baseline, meta, last_metrics, last_matrix, last_cols)
            print(f"   {'Baseline guardado':<36}: {args.save_baseline}.pkl / .npy")
        if args.compare:
            base = _load_baseline(args.compare)
            if base["meta"]["combo_key"] != meta["combo_key"] or base["meta"]["rule_ids"] != meta["rule_ids"]:
                print(f"   Aviso: el baseline es de otro combo o de otra muestra de reglas")
            _print_issues(f"Baseline ({base['meta']['created']})",
                          _compare(base["metrics"], base["matrix"], base["col_names"], last_metrics, last_matrix, last_cols))
    print(LINE)


if __name__ == "__main__":
    main()