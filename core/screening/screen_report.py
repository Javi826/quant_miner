# core/screening/screen_report.py
import logging

import numpy as np

logger = logging.getLogger(__name__)

SEP        = "=" * 124

MODES      = ("YPY", "NPY")
NPY_KEYS   = {"mean1": ("mean_npy1", "n_npy1"), "mean2": ("mean_npy2", "n_npy2")}
BASE_KEYS  = {"mean1": "base1", "mean2": "base2"}      # unconditional mean of the target of every pick

def validate_selection(group_n, n_symbols, j_th, x_th, min_cover, max_cover, mode="YPY", top_i=None):
    if not (0 < group_n <= n_symbols):          # the whole selection depends on it
        raise ValueError(f"GROUP_N must be in [1, {n_symbols}]: {group_n}")
    if not (0.0 < j_th <= 1.0):
        raise ValueError(f"J_TH must be in (0, 1]: {j_th}")
    if not (0.0 < x_th <= 1.0):
        raise ValueError(f"X_TH must be in (0, 1]: {x_th}")
    if not (0.0 <= min_cover <= 1.0):
        raise ValueError(f"MIN_COVER must be in [0, 1]: {min_cover}")
    if not (0.0 <= max_cover <= 1.0):
        raise ValueError(f"MAX_COVER must be in [0, 1]: {max_cover}")
    if min_cover > max_cover:
        raise ValueError(f"MIN_COVER must be <= MAX_COVER: {min_cover} > {max_cover}")
    if mode not in MODES:
        raise ValueError(f"MODE must be one of {MODES}: {mode}")
    if top_i is not None and not (isinstance(top_i, (int, np.integer)) and top_i >= 1):
        raise ValueError(f"TOP_I must be an integer >= 1 or None: {top_i}")

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
    return {"n_pass": int(pass1.sum()), "score": _score(score1, pass1), "pass": pass1, "z": r["z1"], "med": med1}


def evaluate2(r, null_pct):

    floor2_A, med2_A, p842_A = _null_stats(r["null2_A"], null_pct)
    floor2_B, med2_B, p842_B = _null_stats(r["null2_B"], null_pct)
    pass2 = (r["T2"] > floor2_A) & (r["T2"] > floor2_B)
    score2 = np.minimum(_scores(r["T2"], med2_A, p842_A), _scores(r["T2"], med2_B, p842_B))
    return {"n_pass": int(pass2.sum()), "score": _score(score2, pass2), "pass": pass2, "z": r["z2"],
            "med": np.maximum(med2_A, med2_B)}


def _nan_low(v):

    return v if np.isfinite(v) else -np.inf


def luck_share(z_sym, med_sym):
    """Share of the edge that comes by luck, per symbol: the null's median (what the best of the same combinations
    gets on noise) over the z of the combination picked, in [0, 1]. z and the null are in the same units (sigmas);
    edge and z are proportional for the same rule, so the same share of its edge is luck."""
    z = np.asarray(z_sym, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.maximum(np.asarray(med_sym, dtype=float), 0.0) / z
    return np.where(z > 0, np.clip(np.nan_to_num(s, nan=1.0), 0.0, 1.0), 1.0)


def _rank_key(o):
    """Ranking: higher real_bp (the edge in excess over the unconditional mean, without its luck share), then higher
    score."""
    return _nan_low(o["real"]), _nan_low(o["score"])


def select(names, res1, res2, group_n, min_cover, max_cover, measure):

    def ok(q):
        return q["n_pass"] >= group_n

    def option(q, via, cand):
        cv, rl = measure(cand)
        if not (min_cover <= cv <= max_cover):
            return None
        return {"score": q["score"], "n_pass": q["n_pass"], "via": via, "cover": cv, "real": rl}

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
    """Per symbol: candles the winning rule fires on, and its edge_bp in excess over the unconditional mean of its
    target (NaN where the symbol does not pass)."""
    if BASE_KEYS[key] not in r:
        raise RuntimeError("The cache has no baseline (base1 / base2): recompute it")
    side, mask = np.asarray(r["side"]), r["mask"]
    row_of = {int(s): k for k, s in enumerate(np.flatnonzero(side >= 0))}
    fired = np.zeros(side.shape[0], dtype=np.int64)
    for s in np.flatnonzero(pass_sym):
        if s not in row_of:
            raise RuntimeError("The cache has no rule for a passing symbol: recompute it")
        fired[s] = _POPCOUNT[mask[row_of[s]]].sum()
    base = np.asarray(r[BASE_KEYS[key]], dtype=float)
    with np.errstate(invalid="ignore"):
        if mode == "NPY":
            k_mean, k_n = NPY_KEYS[key]
            mean, cnt = np.asarray(r[k_mean], dtype=float), r[k_n]
        else:
            mean, cnt = np.asarray(r[key], dtype=float), fired
        exc_sym = np.where(pass_sym, (mean - base) * cnt / n * 100.0, np.nan)
    return fired, exc_sym


def pooled_cover(fired, sel, n):
    """Cover over the selected symbols: fired candles / candles of those that fire (as _cover of their rows)."""
    f = fired[sel]
    on = np.count_nonzero(f)
    return float(f.sum()) / (on * n) if on else 0.0


# --- Alones: signal Jaccard ---------------------------------------------------
# Signals of an indicator: the screening segments ("value < cut" and "value > cut") of every instance and cut, on the
# candles of every symbol concatenated. Useful: cover (over the candles where the instance has a value) in
# [MIN_COVER, MAX_COVER]. Two signals are twins if their Jaccard >= J_TH.
# An alone A is absorbed by an alone K above it in the ranking and kept if both:
#   R(A|K) >= X_TH: share of the useful signals of A with a twin among the signals of K
#   the winning rule of A (its passing symbols) has a twin among the signals of K (on those symbols)


def _signal_sets(nc):
    """(2*nc, 2*nc + 2) float 0/1: row 2c is "< cut c", row 2c+1 is "> cut c" (as _seg_mask); column k is code k-1."""
    codes = np.arange(-1, 2 * nc + 1)
    m = np.zeros((2 * nc, 2 * nc + 2))
    for c in range(nc):
        m[2 * c] = (codes >= 0) & (codes <= 2 * c)
        m[2 * c + 1] = codes >= 2 * c + 2
    return m


def _joint_hist(a, b, rb, r_total):
    """Counts of (a, code b + 1) over the same candles, as (r_total // rb, rb). a is already a row index >= 0."""
    return np.bincount(a * rb + (b.astype(np.int32) + 1), minlength=r_total).reshape(-1, rb).astype(np.float64)


def _jaccard(ma, h, mb):
    """Jaccard of every signal in ma (rows) with every signal in mb (rows), from their joint histogram h."""
    inter = ma @ h @ mb.T
    union = (ma @ h.sum(axis=1))[:, None] + (mb @ h.sum(axis=0))[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def ind_signals(pool, bins, nm, min_cover, max_cover):
    """Signals of an indicator: its instances' codes (every symbol concatenated), signal sets and useful signals."""
    nc = len(pool.cuts(nm))
    m = _signal_sets(nc)
    codes, useful = [], []
    for i in pool.by_ind[nm]:
        x = bins[i].reshape(-1)
        h = np.bincount(x.astype(np.int32) + 1, minlength=2 * nc + 2).astype(np.float64)
        n_ok = h[1:].sum()
        cov = m @ h / n_ok if n_ok > 0 else np.zeros(len(m))
        codes.append(x)
        useful.append((cov >= min_cover) & (cov <= max_cover))
    return {"nc": nc, "m": m, "codes": codes, "useful": useful, "n_useful": int(sum(u.sum() for u in useful))}


def best_j(sa, sk):
    """Best Jaccard of every useful signal of A with any signal of K (1-D, instances in order)."""
    ra, rk = 2 * sa["nc"] + 2, 2 * sk["nc"] + 2
    out = []
    for xa, u in zip(sa["codes"], sa["useful"]):
        if not u.any():
            continue
        a = xa.astype(np.int32) + 1
        best = np.zeros(int(u.sum()))
        for xk in sk["codes"]:
            best = np.maximum(best, _jaccard(sa["m"][u], _joint_hist(a, xk, rk, ra * rk), sk["m"]).max(axis=1))
        out.append(best)
    return np.concatenate(out) if out else np.zeros(0)


def rule_best_j(win, syms, sk):
    """Jaccard of a winning rule with its twin in K: in every symbol of the rule, the signal of K closest to it (the
    rule's cut and instance can change from symbol to symbol, so can K's); then pooled, sum(inter) / sum(union)."""
    rk = 2 * sk["nc"] + 2
    n_s, n = win.shape
    n_sym = len(sk["codes"][0]) // n
    m = sk["m"]
    a = (np.arange(n_s, dtype=np.int32)[:, None] * 2 + win.astype(np.int32)).reshape(-1)   # row: (symbol, in rule)
    best_j = np.full(n_s, -1.0)
    best_i = np.zeros(n_s)
    best_u = np.zeros(n_s)
    for xk in sk["codes"]:
        xs = xk.reshape(n_sym, n)[syms].reshape(-1)
        h = _joint_hist(a, xs, rk, 2 * n_s * rk).reshape(n_s, 2, rk)
        inter = h[:, 1, :] @ m.T                                # (symbols, signals of K)
        union = h[:, 1, :].sum(axis=1)[:, None] + h.sum(axis=1) @ m.T - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            j = np.where(union > 0, inter / union, 0.0)
        c = j.argmax(axis=1)
        r = np.arange(n_s)
        better = j[r, c] > best_j
        best_j = np.where(better, j[r, c], best_j)
        best_i = np.where(better, inter[r, c], best_i)
        best_u = np.where(better, union[r, c], best_u)
    return float(best_i.sum() / best_u.sum()) if best_u.sum() > 0 else 0.0


def prune_alones(order, sig, wins, j_th, x_th):

    status, kept = {}, []
    for a in order:
        sa = sig[a]
        cand = []                                                  # (r, rule_j, k, bj) against every kept alone
        for k in kept:
            bj = best_j(sa, sig[k])
            if bj.size:
                cand.append((float((bj >= j_th).mean()), rule_best_j(*wins[a], sig[k]), k, bj))
        hits = [c for c in cand if c[0] >= x_th and c[1] >= j_th]
        pick = max(hits or cand, key=lambda c: c[0]) if cand else None
        status[a] = {"by": pick[2] if hits else None, "near": pick[2] if pick else None,
                     "r": pick[0] if pick else np.nan, "rule_j": pick[1] if pick else np.nan,
                     "n_useful": sa["n_useful"]}
        if not hits:
            kept.append(a)
    return status


# --- Pairs: member twins and rule twin ------------------------------------------
# A pair P = (A, B) is absorbed by a pair Q = (C, D) above it in the ranking and kept if both:

def pair_rule_best_j(win, syms, sc, sd):
    """As rule_best_j, against the rules "signal of C and signal of D" (every instance and cut of both)."""
    rc, rd = 2 * sc["nc"] + 2, 2 * sd["nc"] + 2
    n_s, n = win.shape
    n_sym = len(sc["codes"][0]) // n
    mc, md = sc["m"], sd["m"]
    a = (np.arange(n_s, dtype=np.int32)[:, None] * 2 + win.astype(np.int32)).reshape(-1)   # row: (symbol, in rule)
    best_j = np.full(n_s, -1.0)
    best_i = np.zeros(n_s)
    best_u = np.zeros(n_s)
    r = np.arange(n_s)
    for xc in sc["codes"]:
        ac = a * rc + (xc.reshape(n_sym, n)[syms].reshape(-1).astype(np.int32) + 1)
        for xd in sd["codes"]:
            xs = xd.reshape(n_sym, n)[syms].reshape(-1)
            h = _joint_hist(ac, xs, rd, 2 * n_s * rc * rd).reshape(n_s, 2, rc, rd)
            inter = np.einsum("ia,sab,jb->sij", mc, h[:, 1], md).reshape(n_s, -1)
            conj = np.einsum("ia,sab,jb->sij", mc, h.sum(axis=1), md).reshape(n_s, -1)
            union = h[:, 1].sum(axis=(1, 2))[:, None] + conj - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                j = np.where(union > 0, inter / union, 0.0)
            c = j.argmax(axis=1)
            better = j[r, c] > best_j
            best_j = np.where(better, j[r, c], best_j)
            best_i = np.where(better, inter[r, c], best_i)
            best_u = np.where(better, union[r, c], best_u)
    return float(best_i.sum() / best_u.sum()) if best_u.sum() > 0 else 0.0


def prune_pairs(order, sig, wins, j_th, x_th):

    bj = {}

    def r_of(x, y, jt):
        if x == y:
            return 1.0
        if (x, y) not in bj:
            bj[(x, y)] = best_j(sig[x], sig[y])
        v = bj[(x, y)]
        return float((v >= jt).mean()) if v.size else 0.0

    def twins(p, q, jt):
        return [max(r_of(x, y, jt) for y in q) for x in p]

    status, kept = {}, []
    for p in order:
        cand = [(q, twins(p, q, j_th), pair_rule_best_j(*wins[p], sig[q[0]], sig[q[1]])) for q in kept]
        hits = [c for c in cand if min(c[1]) >= x_th and c[2] >= j_th]
        pick = max(hits or cand, key=lambda c: min(c[1])) if cand else None
        status[p] = {"by": pick[0] if hits else None, "near": pick[0] if pick else None,
                     "r": pick[1] if pick else None, "rule_j": pick[2] if pick else np.nan,
                     "n_useful": [sig[x]["n_useful"] for x in p]}
        if not hits:
            kept.append(p)
    return status


def survivors(st_alone, st_pairs):
    """Indicators kept: the alones not absorbed and both members of every pair not absorbed."""
    names = {nm for nm, st in st_alone.items() if st["by"] is None}
    for p, st in st_pairs.items():
        if st["by"] is None:
            names.update(p)
    return names


# --- Rules of the final list ------------------------------------------------------
RULE_DEPTHS = (2, 3)    # MAX_DEPTH of rule_generator for the rule count of the final list


def rule_counts(pool, names, depths=RULE_DEPTHS):
    """{MAX_DEPTH: rules} as rule_generator (both sides): 1..MAX_DEPTH conditions, at most one per indicator, and
    every indicator gives instances x thresholds x ops conditions (as build_flat_specs)."""
    sizes = [len(pool.by_ind[nm]) * len(pool.registry[nm]["thresholds"])
             * len(pool.registry[nm].get("ops", (">", "<"))) for nm in names]
    e = [1] + [0] * max(depths)                              # e[k]: combinations of k indicators, one condition each
    for v in sizes:
        for k in range(max(depths), 0, -1):
            e[k] += e[k - 1] * v
    return {d: 2 * sum(e[1:d + 1]) for d in depths}


# =============================================================================
# 3. OUTPUT
# =============================================================================
# SELECTED and alones: indicator | group | real_bp | n_pass | score | cover, then via or n_sig | status
NAME_W, COL2_W, PAIR_W = 40, 18, 52
R_W, J_W = 11, 6
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


def _log_names(title, items):
    """A list of indicators in DEBUG: its title, then one per line."""
    logger.debug(title)
    for nm in items:
        logger.debug(f"  {nm}")
    if not items:
        logger.debug("  (none)")


def _cols_header(name, col2, tail):
    return (f"{name:<{NAME_W}}{col2:<{COL2_W}}{'real_bp':>{EDGE_W}}{'n_pass':>{N_W}}{'score':>{SCORE_W}}"
            f"{'cover':>{COVER_W}}  {tail}")


def _cols(name, col2, o):
    """indicator | group | real_bp | n_pass | score | cover of an entry of best."""
    return (f"{name:<{NAME_W}}{col2:<{COL2_W}}{_fmt_signed(o['real'], EDGE_W, 3)}{o['n_pass']:>{N_W}}"
            f"{_fmt(o['score'], SCORE_W)}{o['cover']:>{COVER_W}.0%}")


def _via_text(via):
    return via if via == "alone" else pair_name(*via)


def _pct(v):
    return f"{v:.0%}" if np.isfinite(v) else "-"


def _rescuer(nm, st_alone, st_pairs):
    """Why a member of an absorbed candidate stays: a kept alone or a kept pair with it, or None."""
    if nm in st_alone and st_alone[nm]["by"] is None:
        return "alone"
    for p, st in st_pairs.items():
        if st["by"] is None and nm in p:
            return f"pair {pair_name(*p)}"
    return None


def _report_selected(pool, names, best, mode):
    items = [nm for nm in names if nm in best]
    items.sort(key=lambda nm: _rank_key(best[nm]), reverse=True)
    logger.info(f"\n{SEP}\nSELECTED ({len(items)}) [{mode}]\n{SEP}")
    logger.info(_cols_header("indicator", "group", "via"))
    for nm in items:
        o = best[nm]
        logger.info(_cols(nm, _group(pool, nm), o) + f"  {_via_text(o['via'])}")
    if not items:
        logger.debug("(empty)")
    logger.debug("")
    _log_names(f"Selected ({len(items)}):", items)
    return items


def _status(st, q_name, stays):
    """kept / kept (closest: Q) / absorbed by Q, plus the members that stay for another reason."""
    if st["by"] is not None:
        return f"absorbed by {q_name}" + "".join(f", {nm} stays ({w})" for nm, w in stays)
    return f"kept (closest: {q_name})" if st["near"] is not None else "kept"


def _rj_cols(r_txt, st):
    return f"{r_txt:>{R_W}}{_fmt(st['rule_j'], J_W)}" if st["near"] is not None else f"{'-':>{R_W}}{'-':>{J_W}}"


def _report_redundancy_alone(pool, alones, best, st_alone, st_pairs, j_th, x_th, mode):
    logger.debug(f"\n{SEP}\nREDUNDANCY, ALONES [{mode}]\n{SEP}")
    logger.debug(_cols_header("indicator", "group", f"{'n_sig':>5}{'R':>{R_W}}{'J':>{J_W}}  status"))
    for nm in alones:
        o, st = best[nm], st_alone[nm]
        stays = [(nm, w) for w in [_rescuer(nm, {}, st_pairs)] if w] if st["by"] is not None else []
        logger.debug(_cols(nm, _group(pool, nm), o)
                     + f"  {st['n_useful']:>5}{_rj_cols(_pct(st['r']), st)}  {_status(st, st['near'], stays)}")
    if not alones:
        logger.debug("(empty)")
    logger.debug(f"R: share of its useful signals (n_sig) with a twin in the closest alone above. J: Jaccard of its "
                 f"winning rule with its twin there. Absorbed if R >= {x_th:.0%} and J >= {j_th:.2f}")


def _report_redundancy_pairs(pairs, pair_entry, st_alone, st_pairs, j_th, x_th, mode):
    logger.debug(f"\n{SEP}\nREDUNDANCY, PAIRS [{mode}]\n{SEP}")
    logger.debug(f"{'pair':<{PAIR_W}}{'real_bp':>{EDGE_W}}{'n_pass':>{N_W}}{'score':>{SCORE_W}}{'cover':>{COVER_W}}"
                 f"  {'n_sig':>7}{'R':>{R_W}}{'J':>{J_W}}  status")
    for p in pairs:
        o, st = pair_entry[p], st_pairs[p]
        n_sig = "/".join(str(v) for v in st["n_useful"])
        r_txt = "/".join(_pct(v) for v in st["r"]) if st["r"] is not None else "-"
        q_name = pair_name(*st["near"]) if st["near"] is not None else None
        stays = ([(nm, w) for nm in p for w in [_rescuer(nm, st_alone, st_pairs)] if w]
                 if st["by"] is not None else [])
        logger.debug(f"{pair_name(*p):<{PAIR_W}}{_fmt_signed(o['real'], EDGE_W, 3)}{o['n_pass']:>{N_W}}"
                     f"{_fmt(o['score'], SCORE_W)}{o['cover']:>{COVER_W}.0%}  {n_sig:>7}{_rj_cols(r_txt, st)}"
                     f"  {_status(st, q_name, stays)}")
    if not pairs:
        logger.debug("(empty)")
    logger.debug(f"R: share of the useful signals (n_sig) of each member with a twin in the closest pair above. J: "
                 f"Jaccard of its winning rule with its twin there. Absorbed if R >= {x_th:.0%} (both) and "
                 f"J >= {j_th:.2f}")

def _fmt_int(v):
    return format(int(v), ",").replace(",", ".")


def _report_pruned(items, best, st_alone, st_pairs, kept):
    out = [nm for nm in items if nm in kept]
    pct = len(out) / len(items) if items else 0.0
    logger.info(f"\n{SEP}\nAFTER REDUNDANCY ({len(out)} of {len(items)}, {pct:.0%})\n{SEP}")
    _log_names(f"Kept ({len(out)}):", out)
    removed = [nm for nm in items if nm not in kept]
    logger.info(f"Removed ({len(removed)}):")
    for nm in removed:
        c = cand_of(nm, best)
        st = st_alone[nm] if c[0] == "alone" else st_pairs[c[1]]
        r = _pct(st["r"]) if c[0] == "alone" else "/".join(_pct(v) for v in st["r"])
        logger.info(f"  {nm:<26}{'alone' if c[0] == 'alone' else cand_name(c)}: absorbed by "
                    f"{st['by'] if c[0] == 'alone' else pair_name(*st['by'])} (R={r}, J={st['rule_j']:.2f})")
    if not removed:
        logger.info("  (none)")
    return out


def top_list(order, st_alone, st_pairs, top_i):
    """The kept candidates in ranking order, both members of a pair together, until there are top_i or more
    indicators (a pair is never split: it may pass top_i by one). top_i None: all of them."""
    out = []
    for c in order:
        st = st_alone[c[1]] if c[0] == "alone" else st_pairs[c[1]]
        if st["by"] is not None:
            continue
        if top_i is not None and len(out) >= top_i:
            break
        out += [nm for nm in members(c) if nm not in out]
    return out


def _top_candidates(order, st_alone, st_pairs, top_i):
    """The candidates of the TOP: order walked exactly as top_list (keep both in sync), keeping the candidates that
    add at least one new indicator. Every indicator of the TOP entered through one of them (a pair appears once)."""
    out, seen = [], []
    for c in order:
        st = st_alone[c[1]] if c[0] == "alone" else st_pairs[c[1]]
        if st["by"] is not None:
            continue
        if top_i is not None and len(seen) >= top_i:
            break
        new = [nm for nm in members(c) if nm not in seen]
        if new:
            out.append(c)
            seen += new
    return out


def _report_top(pool, pruned, top, top_i):
    """TOP_I (DEBUG): the indicators in the TOP and out of it, and the rules of the final list. Returns the rules,
    {MAX_DEPTH: rules}."""
    if top_i is not None:
        logger.debug(f"\n{SEP}\nTOP_I={top_i} ({len(top)} of {len(pruned)}, in ranking order; a pair is never split)"
                     f"\n{SEP}")
        cut = [nm for nm in pruned if nm not in top]
        _log_names(f"In the TOP ({len(top)}):", top)
        _log_names(f"Out by TOP_I ({len(cut)}):", cut)
    rules = rule_counts(pool, top)
    logger.debug(f"\nRules of the {'TOP' if top_i is not None else 'PRUNED'} (rule_generator, both sides): "
                 + " | ".join(f"MAX_DEPTH={d}: {_fmt_int(v)}" for d, v in rules.items()))
    return rules


def _check_bins(bins, pool, raw):
    want = (len(pool.instances), len(raw["symbols"]), raw["n"])
    if bins is None or tuple(bins.shape) != want:
        raise ValueError(f"bins must be the pool's bins on these symbols' common candles, shape {want}: "
                         f"{None if bins is None else tuple(bins.shape)}")


def report_selection(raw, pool, bins, null_pct, group_n, j_th, x_th, min_cover, max_cover, mode="YPY", top_i=None):
    """Selection, redundancy and TOP_I from the raw results. bins: build_bins of the pool on the same data."""
    validate_selection(group_n, len(raw["symbols"]), j_th, x_th, min_cover, max_cover, mode, top_i)
    raw_mode = raw.get("mode", "YPY")                       # caches from before the modes are YPY
    if mode != raw_mode:
        raise ValueError(f"MODE={mode} but these raw results were computed in {raw_mode}")
    if float(null_pct) != float(raw["null_pct"]):          # it picks the combination of every symbol
        raise ValueError(f"NULL_PCT={null_pct} but these raw results were computed with NULL_PCT={raw['null_pct']}")
    _check_bins(bins, pool, raw)
    names, n = raw["names"], raw["n"]
    res1 = {nm: evaluate1(raw["p1"][nm], null_pct) for nm in names}
    res2 = {p: evaluate2(raw["p2"][p], null_pct) for p in raw["pairs"]}
    sym = {}

    def res_of(c):
        return res1[c[1]] if c[0] == "alone" else res2[c[1]]

    def sym_of(c):
        """(raw results, passing symbols, fired, real_sym) of a candidate, computed once. real_sym: its edge_bp in
        excess over the unconditional mean, without its luck share."""
        if c not in sym:
            r, pass_sym, key = _cand_raw(raw, res1, res2, c)
            q = res_of(c)
            fired, exc_sym = sym_measures(r, pass_sym, key, n, mode)
            sym[c] = (r, pass_sym, fired, exc_sym * (1.0 - luck_share(q["z"], q["med"])))
        return sym[c]

    def measure(c):
        """(cover, real_bp) of a candidate, both sides together."""
        _r, pass_sym, fired, real_sym = sym_of(c)
        return pooled_cover(fired, pass_sym, n), _median_ok(real_sym, pass_sym)

    best = select(names, res1, res2, group_n, min_cover, max_cover, measure)
    selected = _report_selected(pool, names, best, mode)

    # candidates in ranking order: the alones, and the pairs that are the way in of some indicator
    cands = candidates(best)
    order = sorted(cands, key=lambda c: _rank_key(cands[c]), reverse=True)
    alones = [c[1] for c in order if c[0] == "alone"]
    pairs = [c[1] for c in order if c[0] == "pair"]
    sig, wins = {}, {}
    for c in order:
        for nm in members(c):
            if nm not in sig:
                sig[nm] = ind_signals(pool, bins, nm, min_cover, max_cover)
        r, pass_sym = sym_of(c)[:2]
        syms = np.flatnonzero(pass_sym)
        wins[c[1]] = (rule_rows(r, pass_sym, n).any(axis=0)[syms], syms)
    st_alone = prune_alones(alones, sig, wins, j_th, x_th)
    st_pairs = prune_pairs(pairs, sig, wins, j_th, x_th)

    _report_redundancy_alone(pool, alones, best, st_alone, st_pairs, j_th, x_th, mode)
    _report_redundancy_pairs(pairs, {c[1]: o for c, o in cands.items() if c[0] == "pair"}, st_alone, st_pairs,
                             j_th, x_th, mode)
    pruned = _report_pruned(selected, best, st_alone, st_pairs, survivors(st_alone, st_pairs))
    top = top_list(order, st_alone, st_pairs, top_i)
    rules = _report_top(pool, pruned, top, top_i)

    # the TOP's candidates (a pair once) and, for every indicator of the TOP, the one it entered through
    top_cands = _top_candidates(order, st_alone, st_pairs, top_i)
    entry = {}
    for c in top_cands:
        for nm in members(c):
            entry.setdefault(nm, c)
    if list(entry) != top:
        raise RuntimeError("_top_candidates is out of sync with top_list")
    top_symbols = _report_symbols(raw["symbols"], [sym_of(entry[nm])[1] for nm in top])
    return {"selected": selected, "pruned": pruned, "top": top, "symbols": top_symbols, "rules": rules}


def exclude_indicators(raw, exclude):
    """Raw results without the excluded indicators and every pair with them: the same as computing without them."""
    exclude = set(exclude)
    if not exclude:
        return raw
    unknown = sorted(exclude - set(raw["names"]))
    if unknown:
        raise ValueError(f"EXCLUDE has indicators missing from these raw results: {unknown}")
    names = [nm for nm in raw["names"] if nm not in exclude]
    if not names:
        raise ValueError("EXCLUDE leaves no indicators")
    pairs = [p for p in raw["pairs"] if p[0] not in exclude and p[1] not in exclude]
    return {**raw, "names": names, "pairs": pairs,
            "p1": {nm: raw["p1"][nm] for nm in names},
            "p2": {p: raw["p2"][p] for p in pairs}}


def _report_symbols(symbols, masks):
    """Symbols where the indicators of the top pass, each through the candidate it entered the TOP with, ranked by
    how many of them pass on each one (the table in DEBUG)."""
    count = np.zeros(len(symbols), dtype=int)
    for m in masks:
        count += m
    ranked = sorted(np.flatnonzero(count), key=lambda k: count[k], reverse=True)
    out = [str(symbols[k]) for k in ranked]
    logger.debug(f"\n{SEP}\nSYMBOLS OF THE TOP ({len(out)} of {len(symbols)})\n{SEP}")
    for k in ranked:
        logger.debug(f"{str(symbols[k]):<10}{count[k]:>{N_W}}")
    return out