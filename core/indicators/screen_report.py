# core/indicators/screen_report.py
import logging

import numpy as np

logger = logging.getLogger(__name__)

SEP        = "=" * 124
SIDES      = ("long", "short")
LOW_COVER  = "low_cover"
HIGH_COVER = "high_cover"

MODES      = ("YPY", "NPY")
NPY_KEYS   = {"mean1": ("mean_npy1", "n_npy1"), "mean2": ("mean_npy2", "n_npy2")}

def validate_selection(group_n, n_symbols, phi_th, min_cover, max_cover, min_score=None, mode="YPY"):
    if not (0 < group_n <= n_symbols):          # the whole selection depends on it
        raise ValueError(f"GROUP_N must be in [1, {n_symbols}]: {group_n}")
    if not (0.0 <= phi_th <= 1.0):
        raise ValueError(f"PHI_TH must be in [0, 1]: {phi_th}")
    if not (0.0 <= min_cover <= 1.0):
        raise ValueError(f"MIN_COVER must be in [0, 1]: {min_cover}")
    if not (0.0 <= max_cover <= 1.0):
        raise ValueError(f"MAX_COVER must be in [0, 1]: {max_cover}")
    if min_cover > max_cover:
        raise ValueError(f"MIN_COVER must be <= MAX_COVER: {min_cover} > {max_cover}")
    if min_score is not None and not np.isfinite(min_score):
        raise ValueError(f"MIN_SCORE must be a finite number or None: {min_score}")
    if mode not in MODES:
        raise ValueError(f"MODE must be one of {MODES}: {mode}")

# =============================================================================
# 1. SELECTION
# =============================================================================
def _null_stats(null_sym, null_pct):
    with np.errstate(invalid="ignore"):
        floor_sym = np.percentile(null_sym, null_pct, axis=0)
        med_sym = np.percentile(null_sym, 50, axis=0)
        p84_sym = np.percentile(null_sym, 84, axis=0)
    return floor_sym, med_sym, p84_sym


def _scores(t_sym, med_sym, p84_sym):
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = (t_sym - med_sym) / (p84_sym - med_sym)
    return np.where(np.isfinite(sc), sc, np.nan)


def _score(score_sym, pass_sym):
    v = score_sym[pass_sym]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def _median_ok(v, mask):
    w = np.asarray(v)[mask]
    w = w[np.isfinite(w)]
    return float(np.median(w)) if w.size else np.nan


def evaluate1(r, null_pct):

    floor1, med1, p841 = _null_stats(r["null1"], null_pct)
    pass1 = r["T1"] > floor1
    score1 = _scores(r["T1"], med1, p841)
    return {"n_pass": int(pass1.sum()), "score": _score(score1, pass1), "mean": _median_ok(r["mean1"], pass1),
            "pass": pass1}


def evaluate2(r, null_pct):

    floor2_A, med2_A, p842_A = _null_stats(r["null2_A"], null_pct)
    floor2_B, med2_B, p842_B = _null_stats(r["null2_B"], null_pct)
    pass2 = (r["T2"] > floor2_A) & (r["T2"] > floor2_B)
    score2 = np.minimum(_scores(r["T2"], med2_A, p842_A), _scores(r["T2"], med2_B, p842_B))
    return {"n_pass": int(pass2.sum()), "score": _score(score2, pass2), "mean": _median_ok(r["mean2"], pass2),
            "pass": pass2}


def _nan_low(v):

    return v if np.isfinite(v) else -np.inf


def _rank_key(o):
    """Ranking: higher edge_bp, then higher score."""
    return _nan_low(o["edge"]), _nan_low(o["score"])


def select(names, res1, res2, group_n, min_score, min_cover, max_cover, measure):

    def ok(q):
        return q["n_pass"] >= group_n and (min_score is None or _nan_low(q["score"]) >= min_score)

    def option(q, via, cand):
        cv, ed = measure(cand)
        if not (min_cover <= cv <= max_cover):
            return None
        return {"score": q["score"], "n_pass": q["n_pass"], "mean": q["mean"], "via": via,
                "cover": cv, "edge": ed}

    pairs_of = {}
    for p in res2:
        for nm in p:
            pairs_of.setdefault(nm, []).append(p)
    best = {}
    for nm in names:
        opts = []
        if ok(res1[nm]):
            opts.append(option(res1[nm], "alone", ("alone", nm)))
        for p in pairs_of.get(nm, []):
            if ok(res2[p]):
                opts.append(option(res2[p], p, ("pair", p)))
        opts = [o for o in opts if o is not None]
        if opts:
            best[nm] = max(opts, key=_rank_key)
    return best


# =============================================================================
# 2. REDUNDANCY
# =============================================================================
def candidates(best):
    """{candidate: its entry}: the distinct best ways in ("alone", name) or ("pair", (a, b))."""
    return {cand_of(nm, best): o for nm, o in best.items()}


def cand_of(nm, best):
    """The candidate of an indicator: its best way in."""
    via = best[nm]["via"]
    return ("alone", nm) if via == "alone" else ("pair", via)


def members(cand):
    return [cand[1]] if cand[0] == "alone" else list(cand[1])


def cand_name(cand):
    return cand[1] if cand[0] == "alone" else pair_name(*cand[1])


def _cand_raw(raw, res1, res2, cand):
    """(raw results, passing symbols, mean key) of a candidate."""
    if cand[0] == "alone":
        return raw["p1"][cand[1]], res1[cand[1]]["pass"], "mean1"
    return raw["p2"][cand[1]], res2[cand[1]]["pass"], "mean2"


def rule_rows(r, pass_sym, n):
    """(2, n_sym, n) bool, [0] long and [1] short: candles where the winning rule fires, only in passing symbols."""
    side, mask = r["side"], r["mask"]
    n_sym = side.shape[0]
    row_of = {int(s): k for k, s in enumerate(np.flatnonzero(side >= 0))}
    out = np.zeros((2, n_sym, n), dtype=bool)
    for s in np.flatnonzero(pass_sym):
        if s not in row_of:
            raise RuntimeError("The cache has no rule for a passing symbol: recompute it")
        out[side[s], s] = np.unpackbits(mask[row_of[s]], count=n).astype(bool)
    return out


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

def sym_measures(r, pass_sym, key, n, mode="YPY"):

    side, mask = np.asarray(r["side"]), r["mask"]
    row_of = {int(s): k for k, s in enumerate(np.flatnonzero(side >= 0))}
    fired = np.zeros(side.shape[0], dtype=np.int64)
    for s in np.flatnonzero(pass_sym):
        if s not in row_of:
            raise RuntimeError("The cache has no rule for a passing symbol: recompute it")
        fired[s] = _POPCOUNT[mask[row_of[s]]].sum()
    with np.errstate(invalid="ignore"):
        if mode == "NPY":
            k_mean, k_n = NPY_KEYS[key]
            edge_sym = np.where(pass_sym, np.asarray(r[k_mean], dtype=float) * r[k_n] / n * 100.0, np.nan)
        else:
            edge_sym = np.where(pass_sym, np.asarray(r[key], dtype=float) * fired / n * 100.0, np.nan)
    return fired, edge_sym


def pooled_cover(fired, sel, n):
    """Cover over the selected symbols: fired candles / candles of those that fire (as _cover of their rows)."""
    f = fired[sel]
    on = np.count_nonzero(f)
    return float(f.sum()) / (on * n) if on else 0.0


def side_edges(r, pass_sym, edge_sym):
    """(long, short): median edge_bp of the passing symbols whose winning rule is on that side."""
    side = np.asarray(r["side"])
    return tuple(_median_ok(edge_sym, pass_sym & (side == d)) for d in range(2))


def _phi_on(a, b):

    on = a.any(axis=1)
    a, b = a[on], b[on]
    n = a.size
    pa, pb = np.count_nonzero(a) / n, np.count_nonzero(b) / n
    den = np.sqrt(pa * (1.0 - pa) * pb * (1.0 - pb))
    return (np.count_nonzero(a & b) / n - pa * pb) / den if den > 0.0 else 0.0

def _cover(row):
    on = row.any(axis=1)
    return np.count_nonzero(row) / row[on].size if on.any() else 0.0

def side_stats(cands, rows, edges):
    """{(candidate, side): (cover, edge_bp)} on every side where the candidate fires."""
    out = {}
    for c in cands:
        for d in range(2):
            a = rows[c][d]
            if a.any():
                out[(c, d)] = (_cover(a), edges[c][d])
    return out


def side_order(cands, stats, d):
    """Candidates firing on side d, by edge_bp on that side, then score (stable: first on ties)."""
    cs = [c for c in cands if (c, d) in stats]
    return sorted(cs, key=lambda c: (-_nan_low(stats[(c, d)][1]), -_nan_low(cands[c]["score"])))


def prune(cands, rows, stats, phi_th, min_cover, max_cover):
    status = {}
    for d in range(2):
        order = side_order(cands, stats, d)
        for kind in ("alone", "pair"):
            kept = []
            for c in order:
                if c[0] != kind:
                    continue
                a = rows[c][d]
                cv, ed = stats[(c, d)]
                if cv < min_cover:
                    status[(c, d)] = (LOW_COVER, cv)
                    continue
                if cv > max_cover:
                    status[(c, d)] = (HIGH_COVER, cv)
                    continue
                r_max, by = -np.inf, None
                for k in kept:
                    r = _phi_on(a, rows[k][d])
                    if r > r_max:
                        r_max, by = r, k
                if r_max > phi_th:
                    status[(c, d)] = (by, r_max)
                else:
                    status[(c, d)] = None
                    kept.append(c)
    return status

def survivors(cands, status):
    """Indicators in some candidate kept on some side."""
    names = set()
    for (c, _d), st in status.items():
        if st is None:
            names.update(members(c))
    return names


# =============================================================================
# 3. OUTPUT
# =============================================================================
# Same columns in both tables: indicator | group | edge_bp | n_pass | score | cover | via; redundancy adds side | status
NAME_W, COL2_W, SIDE_W = 40, 18, 7
N_W, SCORE_W, EDGE_W, COVER_W = 8, 8, 9, 8


def pair_name(a, b):
    return f"{a} + {b}"


def _fmt(v, w=6):
    return f"{v:>{w}.2f}" if np.isfinite(v) else f"{'-':>{w}}"


def _fmt_signed(v, w=8, dec=3):
    return f"{v:>+{w}.{dec}f}" if np.isfinite(v) else f"{'-':>{w}}"


def _group(pool, name):
    g = pool.registry[name]["group"]
    return pool.group_names.get(g, g)


def _py_list(var_name, items):

    lines = [f"{var_name} = ["]
    for it in items:
        lines.append(f'    "{it}",')
    lines.append("]")
    return "\n".join(lines)


def _cols_header(name, col2, tail):
    return (f"{name:<{NAME_W}}{col2:<{COL2_W}}{'edge_bp':>{EDGE_W}}{'n_pass':>{N_W}}{'score':>{SCORE_W}}"
            f"{'cover':>{COVER_W}}  {tail}")


def _cols(name, col2, n_pass, score, cover, edge_v):
    return (f"{name:<{NAME_W}}{col2:<{COL2_W}}{_fmt_signed(edge_v, EDGE_W, 2)}{n_pass:>{N_W}}"
            f"{_fmt(score, SCORE_W)}{cover:>{COVER_W}.0%}")


def _via_text(via):
    return via if via == "alone" else pair_name(*via)


def _status_text(st):
    if st is None:
        return "kept"
    if st[0] == LOW_COVER:
        return f"low cover ({st[1]:.0%})"
    if st[0] == HIGH_COVER:
        return f"high cover ({st[1]:.0%})"
    return f"absorbed by {cand_name(st[0])} (phi={st[1]:.2f})"


def _report_selected(pool, names, best, mode):
    items = [nm for nm in names if nm in best]
    items.sort(key=lambda nm: _rank_key(best[nm]), reverse=True)
    logger.info(f"\n{SEP}\nSELECTED ({len(items)}) [{mode}]\n{SEP}")
    logger.info(_cols_header("indicator", "group", "via"))
    for nm in items:
        o = best[nm]
        logger.info(_cols(nm, _group(pool, nm), o["n_pass"], o["score"], o["cover"], o["edge"])
                    + f"  {_via_text(o['via'])}")
    if not items:
        logger.info("(empty)")
    logger.info("\n" + _py_list("SYMBOL_INDICATORS", items))
    return items


def _report_redundancy(pool, items, best, cands, rows, stats, status, mode):
    logger.info(f"\n{SEP}\nREDUNDANCY [{mode}]\n{SEP}")
    logger.info(_cols_header("indicator", "group", f"{'via':<{NAME_W}}{'side':<{SIDE_W}}status"))
    of_cand = {}
    for nm in items:
        of_cand.setdefault(cand_of(nm, best), []).append(nm)
    for d in range(2):
        order = side_order(cands, stats, d)
        for kind in ("alone", "pair"):
            for c in order:
                if c[0] != kind or (c, d) not in status:
                    continue
                cv, ed = stats[(c, d)]
                n_side = int(rows[c][d].any(axis=1).sum())
                for nm in of_cand[c]:
                    logger.info(_cols(nm, _group(pool, nm), n_side, cands[c]["score"], cv, ed)
                                + f"  {cand_name(c):<{NAME_W}}{SIDES[d]:<{SIDE_W}}{_status_text(status[(c, d)])}")
    if not status:
        logger.info("(empty)")


def _report_pruned(items, best, status, kept):
    out = [nm for nm in items if nm in kept]
    pct = len(out) / len(items) if items else 0.0
    logger.info(f"\n{SEP}\nAFTER REDUNDANCY ({len(out)} of {len(items)}, {pct:.0%})\n{SEP}")
    logger.info(_py_list("SYMBOL_INDICATORS_PRUNED", out) + "\n")
    removed = [nm for nm in items if nm not in kept]
    logger.info(f"Removed ({len(removed)}):")
    for nm in removed:
        c = cand_of(nm, best)
        why = ", ".join(f"{SIDES[d]} {_status_text(status[(c, d)])}" for d in range(2) if (c, d) in status)
        logger.info(f"  {nm:<26}{cand_name(c)}: {why}")
    if not removed:
        logger.info("  (none)")
    return out


def report_selection(raw, pool, null_pct, group_n, phi_th, min_cover, max_cover, min_score=None, mode="YPY"):

    validate_selection(group_n, len(raw["symbols"]), phi_th, min_cover, max_cover, min_score, mode)
    if null_pct < raw["null_pct"]:
        raise ValueError(f"NULL_PCT={null_pct} is below the {raw['null_pct']} of these raw results")
    names, n = raw["names"], raw["n"]
    res1 = {nm: evaluate1(raw["p1"][nm], null_pct) for nm in names}
    res2 = {p: evaluate2(raw["p2"][p], null_pct) for p in raw["pairs"]}
    sym = {}

    def sym_of(c):
        """(raw results, passing symbols, fired, edge_sym) of a candidate, computed once."""
        if c not in sym:
            r, pass_sym, key = _cand_raw(raw, res1, res2, c)
            sym[c] = (r, pass_sym) + sym_measures(r, pass_sym, key, n, mode)
        return sym[c]

    def measure(c):
        """(cover, edge_bp) of a candidate, both sides together."""
        _r, pass_sym, fired, edge_sym = sym_of(c)
        return pooled_cover(fired, pass_sym, n), _median_ok(edge_sym, pass_sym)

    best = select(names, res1, res2, group_n, min_score, min_cover, max_cover, measure)
    selected = _report_selected(pool, names, best, mode)

    cands = candidates(best)
    rows, edges = {}, {}
    for c in cands:
        r, pass_sym, _fired, edge_sym = sym_of(c)
        rows[c] = rule_rows(r, pass_sym, n)
        edges[c] = side_edges(r, pass_sym, edge_sym)
    stats = side_stats(cands, rows, edges)
    status = prune(cands, rows, stats, phi_th, min_cover, max_cover)
    _report_redundancy(pool, selected, best, cands, rows, stats, status, mode)
    pruned = _report_pruned(selected, best, status, survivors(cands, status))
    return {"selected": selected, "pruned": pruned}