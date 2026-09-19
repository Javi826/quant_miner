# research/screening/calibration_BLOCK_sc.py

import os
import sys
import time
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.screening.calibrate_block")
logger.setLevel(logging.INFO)
for noisy in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("BOT_batch.screening.panel").setLevel(logging.INFO)

from panel import build_panel, DATASET, TIMEFRAME, SYMBOLS, HORIZONS, IS_A_FRACTION, SCOPES

# =============================================================================
# CONFIG
# =============================================================================
PW_C_SIGNIF    = 2.0     # multiplicador de la banda sqrt(log10(n)/n)
PW_QUANTILES   = [50, 75, 90, 95, 99]
PW_RECOMMEND_Q = 99      # percentil sobre columnas del que sale la recomendacion.
                         # La nula del StepM es la de un MAXIMO sobre la familia:
                         # un p90 deja al 10% de las columnas por debajo de su
                         # propio requisito de bloque, y en un estadistico de
                         # maximo la cola es justo lo que decide.

ACF_CHUNK_COLUMNS = 250
MIN_PAIRS_PER_LAG = 30      # lags con menos pares validos se anulan, no se extrapolan

MIN_EFFECTIVE_BLOCKS = 20   # aviso si n_obs/block cae por debajo
MIN_COVERAGE_P10     = 0.60 # aviso si el decil bajo de cobertura cae por debajo

# =============================================================================
# FUNCION DE INFLUENCIA DEL ESTADISTICO mean/std, tolerante a NaN
# =============================================================================
def _influence_function(values: np.ndarray, mask: np.ndarray) -> tuple:
    """psi_t = z_t - 0.5 * S * (z_t^2 - 1), con z estandarizado sobre barras
    validas. Devuelve tambien la mascara EFECTIVA por barra: mask AND psi finita.
    Las barras no validas quedan a 0 y se excluyen del conteo de pares en la
    autocovarianza, asi que ese 0 no entra en ningun denominador."""
    x64 = values.astype(np.float64, copy=True)
    x64[~mask] = np.nan

    n_valid = mask.sum(axis=0)
    mu      = np.nanmean(np.where(mask, x64, np.nan), axis=0)
    sd      = np.nanstd(np.where(mask, x64, np.nan), axis=0, ddof=1)

    valid = (sd > 0) & (n_valid > 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        z         = (x64 - mu[None, :]) / sd[None, :]
        statistic = mu / sd

    psi      = z - 0.5 * statistic[None, :] * (z * z - 1.0)
    bar_mask = mask & np.isfinite(psi)
    psi      = np.where(bar_mask, psi, 0.0)
    return psi, bar_mask, valid, statistic

# =============================================================================
# AUTOCOVARIANZA POR COLUMNA — FFT, denominador por pares validos de cada lag
# =============================================================================
def _autocovariance_by_column(values: np.ndarray, mask: np.ndarray, max_lag: int,
                              chunk_columns: int = ACF_CHUNK_COLUMNS,
                              min_pairs: int = MIN_PAIRS_PER_LAG) -> np.ndarray:
    """R(k) dividida por el numero de pares (t, t+k) en que AMBAS barras son
    validas, no por n_obs. Dividir por n_obs contrae la ACF por el factor de
    cobertura p: los ceros de relleno entran en el denominador pero no aportan
    covarianza, asi que rho_hat ~ p * rho. Eso resuelve m_hat antes de tiempo,
    recorta la ventana M y acorta el bloque de forma sistematica.

    El precio es que el estimador por ratio ya no es semidefinido positivo:
    g_hat(0) puede salir no positivo en columnas muy huecas y |acf| puede pasar
    de 1. Lo primero lo filtra `usable` aguas arriba; lo segundo solo empuja la
    ventana hacia arriba, que es la direccion conservadora."""
    n_obs, n_cols = values.shape
    max_lag = min(max_lag, n_obs - 1)
    fft_len = 1 << int(np.ceil(np.log2(2 * n_obs)))

    acov = np.empty((max_lag + 1, n_cols), dtype=np.float64)
    for start in range(0, n_cols, chunk_columns):
        end   = min(start + chunk_columns, n_cols)
        chunk = np.ascontiguousarray(values[:, start:end].astype(np.float64))
        m     = np.ascontiguousarray(mask[:, start:end].astype(np.float64))

        # centrado sobre barras validas; las invalidas quedan exactamente a 0
        cnt   = m.sum(axis=0)
        mu    = (chunk * m).sum(axis=0) / np.maximum(cnt, 1.0)
        chunk = (chunk - mu[None, :]) * m

        spectrum = np.fft.rfft(chunk, n=fft_len, axis=0)
        cross    = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_len, axis=0)[: max_lag + 1]

        mask_spec = np.fft.rfft(m, n=fft_len, axis=0)
        pairs     = np.fft.irfft(mask_spec * np.conjugate(mask_spec), n=fft_len, axis=0)[: max_lag + 1]
        pairs     = np.maximum(np.rint(pairs), 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            block = cross / pairs
        block = np.where(pairs >= min_pairs, block, 0.0)
        acov[:, start:end] = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    return acov


def _flat_top_lag_window(lags: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    """Ventana flat-top de Politis & White: 1 en [0,1/2], 2(1-s) en [1/2,1], 0 mas alla."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = lags[:, None] / bandwidth[None, :]
    weights = np.where(s <= 0.5, 1.0, np.where(s <= 1.0, 2.0 * (1.0 - s), 0.0))
    return np.where(np.isfinite(weights), weights, 0.0)

# =============================================================================
# M3 — SELECCION AUTOMATICA DE LONGITUD DE BLOQUE
# =============================================================================
def method_politis_white(values: np.ndarray, mask: np.ndarray, label: str) -> dict:
    psi, bar_mask, valid, _ = _influence_function(values, mask)
    psi      = np.ascontiguousarray(psi[:, valid])
    bar_mask = np.ascontiguousarray(bar_mask[:, valid])
    n_obs, n_cols = psi.shape

    if n_cols == 0:
        logger.warning(f"  {label}: ninguna columna con varianza positiva — omitido")
        return None

    n_valid_col = bar_mask.sum(axis=0).astype(np.float64)
    coverage    = n_valid_col / max(n_obs, 1)

    k_n     = int(max(5, np.ceil(np.sqrt(np.log10(n_obs)))))
    m_max   = int(np.ceil(np.sqrt(n_obs)) + k_n)
    b_max   = int(np.ceil(min(3.0 * np.sqrt(n_obs), n_obs / 3.0)))
    lag_max = min(m_max + k_n, n_obs - 1)

    acov = _autocovariance_by_column(psi, bar_mask, max_lag=lag_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = acov / acov[0][None, :]
    acf = np.nan_to_num(acf, nan=0.0, posinf=0.0, neginf=0.0)

    band       = PW_C_SIGNIF * np.sqrt(np.log10(n_obs) / n_obs)
    signif     = (np.abs(acf[1:]) >= band).astype(np.int32)
    cumulative = np.vstack([np.zeros((1, n_cols), dtype=np.int32), np.cumsum(signif, axis=0)])

    n_windows = signif.shape[0] - k_n + 1
    m_hat     = np.full(n_cols, lag_max, dtype=np.int64)
    resolved  = np.zeros(n_cols, dtype=bool)
    for window_start in range(max(n_windows, 0)):
        clean = (cumulative[window_start + k_n] - cumulative[window_start]) == 0
        newly = clean & (~resolved)
        if newly.any():
            m_hat[newly] = window_start + 1
            resolved |= newly
        if resolved.all():
            break

    bandwidth = np.clip(2 * m_hat, 1, m_max).astype(np.float64)
    lags      = np.arange(1, lag_max + 1, dtype=np.float64)
    weights   = _flat_top_lag_window(lags, bandwidth)

    g_hat = acov[0] + 2.0 * (weights * acov[1:]).sum(axis=0)
    g_big = 2.0 * (weights * lags[:, None] * acov[1:]).sum(axis=0)

    # D_CB = (4/3) g_hat(0)^2 — constante del moving/circular block bootstrap.
    #
    # El n del escalado es n_obs, la longitud de la REJILLA, no las barras
    # validas de cada columna. El MSE que PW minimiza tiene sesgo ~ G/b con b en
    # lags de rejilla, y varianza ~ D*b/n donde ese n es el numero de filas que
    # se remuestrean: el bootstrap coge bloques de rejilla y no sabe nada de la
    # cobertura de cada columna. Escalar por n_valid_col mezcla los dos ejes y
    # recorta el bloque en proporcion a la cobertura, que es la direccion
    # anticonservadora. La cobertura se reporta abajo, pero no entra aqui.
    usable = (g_hat > 0) & np.isfinite(g_big) & (n_valid_col > 2)
    if not usable.any():
        logger.warning(f"  {label}: densidad espectral no positiva en toda columna — omitido")
        return None

    d_mbb = (4.0 / 3.0) * g_hat[usable] ** 2
    b_opt = np.cbrt(2.0 * g_big[usable] ** 2 / d_mbb) * np.cbrt(float(n_obs))
    b_opt = np.clip(b_opt, 1.0, b_max)

    if b_opt.size == 0:
        logger.warning(f"  {label}: sin columnas utilizables tras el filtro — omitido")
        return None

    quantiles = {q: float(np.percentile(b_opt, q)) for q in PW_QUANTILES}
    recommended = int(np.ceil(quantiles[PW_RECOMMEND_Q]))

    cov_p10 = float(np.percentile(coverage, 10))

    logger.info(f"\n  {label}")
    logger.info(f"    n_obs={n_obs}  n_cols={n_cols}  usables={int(usable.sum())}  K_N={k_n}  M_max={m_max}  b_max={b_max}")
    logger.info(
        f"    cobertura      : p10={cov_p10:.2f}  "
        f"p50={np.percentile(coverage, 50):.2f}  p90={np.percentile(coverage, 90):.2f}"
    )
    logger.info(f"    bandwidth M    : p50={np.percentile(bandwidth, 50):.0f}  p90={np.percentile(bandwidth, 90):.0f}")
    logger.info(
        "    b_opt          : "
        + "  ".join(f"p{q}={quantiles[q]:.1f}" for q in PW_QUANTILES)
        + f"  max={b_opt.max():.1f}"
    )
    logger.info(f"    --> b (p{PW_RECOMMEND_Q})   : {recommended}")

    if cov_p10 < MIN_COVERAGE_P10:
        logger.warning(
            f"    AVISO: cobertura p10={cov_p10:.2f} < {MIN_COVERAGE_P10:.2f}. El decil bajo "
            f"de columnas apenas tiene barras en la rejilla comun: el problema esta en la "
            f"rejilla (MIN_VALID_FRACTION en panel.py), no en el bloque."
        )

    return {
        "recommended": recommended,
        "quantiles":   quantiles,
        "n_obs":       n_obs,
        "n_cols":      n_cols,
        "n_usable":    int(usable.sum()),
        "coverage_p10": cov_p10,
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'=' * 78}")
    logger.info(f"  CALIBRACION DE LONGITUD DE BLOQUE — CRIBA DE INDICADORES")
    logger.info(f"{'=' * 78}")
    logger.info(f"  DATASET       : {DATASET}")
    logger.info(f"  TIMEFRAME     : {TIMEFRAME}")
    logger.info(f"  SIMBOLOS      : {len(SYMBOLS)}")
    logger.info(f"  HORIZONTES    : {HORIZONS}")
    logger.info(f"  SCOPES        : {SCOPES}")
    logger.info(f"  IS-A / IS-B   : {IS_A_FRACTION:.0%} / {1 - IS_A_FRACTION:.0%}")
    logger.info(f"  PW_RECOMMEND_Q: p{PW_RECOMMEND_Q}")
    logger.info(f"  ACF           : denominador por pares validos (min {MIN_PAIRS_PER_LAG} por lag)")
    logger.info(f"{'=' * 78}\n")

    built = build_panel()
    panel, mask, col_meta, seg_a = built["panel"], built["mask"], built["col_meta"], built["seg_a"]

    if panel.shape[1] == 0:
        logger.error("  panel vacio — nada que calibrar")
        sys.exit(1)

    panel_a = np.ascontiguousarray(panel[seg_a])
    mask_a  = np.ascontiguousarray(mask[seg_a])
    n_obs_a = panel_a.shape[0]

    logger.info(f"\n{'─' * 78}")
    logger.info(f"  POLITIS & WHITE POR HORIZONTE (sobre IS-A, {n_obs_a} barras)")
    logger.info(f"{'─' * 78}")

    horizons_in_panel = sorted({m["horizon"] for m in col_meta})
    results = {}
    for horizon in horizons_in_panel:
        cols = np.array([m["horizon"] == horizon for m in col_meta])
        result = method_politis_white(
            np.ascontiguousarray(panel_a[:, cols]),
            np.ascontiguousarray(mask_a[:, cols]),
            label=f"h={horizon}",
        )
        if result is not None:
            results[horizon] = result

    if not results:
        logger.error("\n  ningun horizonte produjo una recomendacion — revisa el panel")
        sys.exit(1)

    block = max(r["recommended"] for r in results.values())

    logger.info(f"\n{'=' * 78}")
    logger.info(f"  RESULTADO")
    logger.info(f"{'=' * 78}")
    logger.info(
        f"  {'HORIZONTE':<14}{'b (p' + str(PW_RECOMMEND_Q) + ')':<12}{'b / h':<10}"
        f"{'b ref MA(h-1)':<16}{'bloques efectivos':<20}"
    )
    logger.info(f"  {'-' * 72}")
    for horizon in horizons_in_panel:
        if horizon not in results:
            continue
        b   = results[horizon]["recommended"]
        eff = n_obs_a / b
        # Referencia: solapamiento puro de la etiqueta, rho(k) = 1 - k/h, da
        # g(0)=sigma^2*h y G=sigma^2*h^2/3, de donde b = (h^2/6)^(1/3)*n^(1/3).
        # Es la cota que veria una columna cuyo peso w_t fuese constante: los
        # indicadores lentos deberian acercarse, los rapidos quedar por debajo.
        b_ref = (horizon ** 2 / 6.0) ** (1.0 / 3.0) * n_obs_a ** (1.0 / 3.0)
        logger.info(f"  h={horizon:<12}{b:<12}{b / horizon:<10.2f}{b_ref:<16.0f}{eff:<20.0f}")

    effective = n_obs_a / block
    logger.info(f"\n  ==> BLOCK_SIZE = {block}   (maximo sobre horizontes)")
    logger.info(f"      bloques efectivos en IS-A: {effective:.0f}")

    if effective < MIN_EFFECTIVE_BLOCKS:
        logger.warning(
            f"\n  AVISO: solo {effective:.0f} bloques efectivos (< {MIN_EFFECTIVE_BLOCKS}). "
            f"El bootstrap tiene poco que remuestrear: recorta el horizonte mayor "
            f"o alarga la muestra antes de correr la criba."
        )

    logger.info(f"{'=' * 78}\n")

    elapsed = int(time.time() - start)
    logger.info(f"🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
