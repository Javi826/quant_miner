# core/indicators/screen_report.py
"""Selection, redundancy and report of the screening. Never cached and out of the cache key: tune freely.

Reads the raw results of screen_engine.
    Selection    phase 1 passes where T1 > its null floor, phase 2 where T2 beats both floors (A shifted, B shifted).
                 An indicator is selected when it passes, alone or inside a pair, in >= group_n symbols. Its way in
                 (alone or one pair) is the one with the best score.
    Redundancy   the selected candidates (every distinct way in: an indicator alone, or a pair) are compared by the
                 candles where their winning rule fires, only in the symbols where they pass, per side (long,
                 short), alones against alones and pairs against pairs. Ranked by score, a candidate is absorbed when
                 its phi (correlation of the 1/0 signals) with a better one already kept is > phi_th, on the symbols
                 where the candidate has candles (what the kept one does elsewhere does not count against it). Phi is
                 0 for unrelated rules however wide they are (Jaccard is not: two unrelated rules firing on 80% of
                 the candles give J = 0.67). An indicator stays if any candidate it belongs to is kept on any side.
Market agnostic.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

SEP        = "=" * 124
SIDES      = ("long", "short")
LOW_COVER  = "low_cover"
HIGH_COVER = "high_cover"

def validate_selection(group_n, n_symbols, phi_th, min_cover, max_cover):
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


def select(names, res1, res2, group_n):
    """{indicator: (score, n_pass, mean, via)}, via "alone" or the pair (a, b): its best way in (first on ties)."""
    best = {}
    for nm in names:
        cand = []
        r = res1[nm]
        if r["n_pass"] >= group_n:
            cand.append((r["score"], r["n_pass"], r["mean"], "alone"))
        for p, q in res2.items():
            if nm in p and q["n_pass"] >= group_n:
                cand.append((q["score"], q["n_pass"], q["mean"], p))
        if cand:
            best[nm] = max(cand, key=lambda c: _nan_low(c[0]))
    return best


# =============================================================================
# 2. REDUNDANCY
# =============================================================================
def candidates(best):

    cands = {}
    for nm, (score, _n, _m, via) in best.items():
        cands[("alone", nm) if via == "alone" else ("pair", via)] = score
    return cands


def members(cand):
    return [cand[1]] if cand[0] == "alone" else list(cand[1])


def cand_name(cand):
    return cand[1] if cand[0] == "alone" else pair_name(*cand[1])


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

def prune(cands, rows, phi_th, min_cover, max_cover):
    status = {}
    order = sorted(cands, key=lambda c: -_nan_low(cands[c]))
    for d in range(2):
        for kind in ("alone", "pair"):
            kept = []
            for c in order:
                if c[0] != kind:
                    continue
                a = rows[c][d]
                if not a.any():
                    continue
                cv = _cover(a)
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
def pair_name(a, b):
    return f"{a} + {b}"


def _fmt(v, w=6):
    return f"{v:>{w}.2f}" if np.isfinite(v) else f"{'-':>{w}}"


def _fmt_mean(v, w=8):
    return f"{v:>+{w}.3f}" if np.isfinite(v) else f"{'-':>{w}}"


def _group(pool, name):
    g = pool.registry[name]["group"]
    return pool.group_names.get(g, g)


def _py_list(var_name, items):

    lines = [f"{var_name} = ["]
    for it in items:
        lines.append(f'    "{it}",')
    lines.append("]")
    return "\n".join(lines)


def _role_header():
    return (f"{'indicator':<26}{'group':<20}{'n_pass':>7}{'score':>8}{'mean%':>9}"
            f"  {'via':<48}")


def _role_row(pool, nm, n_pass, score, mean, via):
    return (f"{nm:<26}{_group(pool, nm):<20}{n_pass:>7}{_fmt(score, 8)}{_fmt_mean(mean, 9)}"
            f"  {via:<48}")


def _status_text(st):
    if st is None:
        return "kept"
    if st[0] == LOW_COVER:
        return f"low cover ({st[1]:.0%})"
    if st[0] == HIGH_COVER:
        return f"high cover ({st[1]:.0%})"
    return f"absorbed by {cand_name(st[0])} (phi={st[1]:.2f})"

ROLES = (("SIGNALS", "signal", "SYMBOL_SIGNAL"), ("FILTERS", "filter", "SYMBOL_FILTER"))


def _report_roles(pool, names, best, group_n):
    lists = {}
    for title, role, var in ROLES:
        items = [nm for nm in names if pool.registry[nm]["role"] == role and nm in best]
        items.sort(key=lambda nm: -_nan_low(best[nm][0]))
        logger.info(f"\n{SEP}\n{title} ({len(items)}) | passing in >= {group_n} symbols, "
                    f"alone or inside a pair; ranked by score\n{SEP}")
        logger.info(_role_header())
        for nm in items:
            score, n_pass, mean, via = best[nm]
            logger.info(_role_row(pool, nm, n_pass, score, mean, via if via == "alone" else pair_name(*via)))
        if not items:
            logger.info("(empty)")
        logger.info("\n" + _py_list(var, items))
        lists[role] = items
    return lists


def _report_redundancy(cands, rows, status, phi_th, min_cover, max_cover):
    logger.info(f"\n{SEP}\nREDUNDANCY ({len(cands)} candidates) | per side, alones vs alones and pairs vs pairs, "
                f"ranked by score;\nabsorbed if phi > {phi_th:.2f} with a better candidate already kept, on the "
                f"symbols where it has candles;\ncover: % of the candles of its symbols where it fires "
                f"(dropped if < {min_cover:.0%} or > {max_cover:.0%})\n{SEP}")
    order = sorted(cands, key=lambda c: -_nan_low(cands[c]))
    for d in range(2):
        for kind in ("alone", "pair"):
            for c in order:
                if c[0] != kind or (c, d) not in status:
                    continue
                r = rows[c][d]
                logger.info(f"{SIDES[d]:<7}{cand_name(c):<50}{_fmt(cands[c], 7)}{int(r.any(axis=1).sum()):>7}"
                            f"{_cover(r):>8.0%}  {_status_text(status[(c, d)])}")
    if not status:
        logger.info("(empty)")


def _report_pruned(lists, best, status, kept):
    n_all = sum(len(v) for v in lists.values())
    n_kept = sum(nm in kept for v in lists.values() for nm in v)
    logger.info(f"\n{SEP}\nAFTER REDUNDANCY ({n_kept} of {n_all}) | an indicator stays if any candidate it belongs "
                f"to is kept on any side\n{SEP}")
    out = {}
    for _title, role, var in ROLES:
        items = [nm for nm in lists[role] if nm in kept]
        out[role] = items
        logger.info(_py_list(f"{var}_PRUNED", items) + "\n")
    removed = [nm for v in lists.values() for nm in v if nm not in kept]
    logger.info(f"Removed ({len(removed)}):")
    for nm in removed:
        via = best[nm][3]
        c = ("alone", nm) if via == "alone" else ("pair", via)
        why = ", ".join(f"{SIDES[d]} {_status_text(status[(c, d)])}" for d in range(2) if (c, d) in status)
        logger.info(f"  {nm:<26}{cand_name(c)}: {why}")
    if not removed:
        logger.info("  (none)")
    return out


def report_selection(raw, pool, null_pct, group_n, phi_th, min_cover, max_cover):

    validate_selection(group_n, len(raw["symbols"]), phi_th, min_cover, max_cover)
    if null_pct < raw["null_pct"]:
        raise ValueError(f"NULL_PCT={null_pct} is below the {raw['null_pct']} of these raw results")
    names, n = raw["names"], raw["n"]
    res1 = {nm: evaluate1(raw["p1"][nm], null_pct) for nm in names}
    res2 = {p: evaluate2(raw["p2"][p], null_pct) for p in raw["pairs"]}
    best = select(names, res1, res2, group_n)
    lists = _report_roles(pool, names, best, group_n)

    cands = candidates(best)
    rows = {c: rule_rows(raw["p1"][c[1]], res1[c[1]]["pass"], n) if c[0] == "alone"
            else rule_rows(raw["p2"][c[1]], res2[c[1]]["pass"], n) for c in cands}
    status = prune(cands, rows, phi_th, min_cover, max_cover)
    _report_redundancy(cands, rows, status, phi_th, min_cover, max_cover)
    pruned = _report_pruned(lists, best, status, survivors(cands, status))
    return {"signal": lists["signal"], "filter": lists["filter"],
            "signal_pruned": pruned["signal"], "filter_pruned": pruned["filter"]}