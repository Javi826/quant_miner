# BOT_crypto/validation/candle_validator.py
"""
Autonomous candle-close validation (parallel check for the MT5 migration).

Hybrid design:
    Method A (clock)  - the live bot's arithmetic trigger
                        (calculate_next_candle_time + fixed buffer).
    Method B (broker) - real broker push over the public WebSocket candle
                        channel, used ONLY to detect the exact instant a
                        candle closes (open_time rollover). The WebSocket
                        never supplies OHLC values: the in-flight candle it
                        streams can still receive late trade updates after
                        the rollover, which makes its own close value
                        unreliable.
    Data source        - REST history-candles is the single source of truth
                        for OHLC, exactly like the live strategies use. It
                        is queried once method B confirms a close.

This module is strictly observational:
  - It never raises into the caller (every public entry point is guarded).
  - It owns its own WebSocket connection, separate from the trading one.
  - Removing the two injection lines from the orchestrator disables it
    completely.

All output is emitted through the bot logger with the [VERIFY] prefix.
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger('BOT_crypto.validation.candle_validator')

# ==========================================================================
# PATH SETUP
# ==========================================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))   # validation/
_bot_dir     = os.path.dirname(_current_dir)                # BOT_crypto/
_bitget_dir  = os.path.dirname(_bot_dir)                    # bitget/

for _path in (
    _bot_dir,
    os.path.join(_bitget_dir, "broker_client", "broker_api"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ==========================================================================
# IMPORTS
# ==========================================================================

from market_data.websocket_manager import BitgetWSManager
from api_client import _call_history_candles

# ==========================================================================
# CONSTANTS
# ==========================================================================

# Buffer added by bot_utils.calculate_next_candle_time after the theoretical
# candle close. Method A is expected to fire at close + this value.
CANDLE_TRIGGER_BUFFER_SECONDS = 20

# Accepted deviation between the expected and the observed trigger instant.
CANDLE_TRIGGER_TOLERANCE_SECONDS = 10

# Accepted deviation between the real broker close and the WebSocket push
# that reveals it (network + broker publication latency).
WS_DETECTION_TOLERANCE_SECONDS = 5

INSTRUMENT_TYPE = "USDT-FUTURES"

MARK_OK   = "✅"
MARK_FAIL = "❌"

# ==========================================================================
# HELPERS
# ==========================================================================


def timeframe_to_seconds(timeframe: str) -> int:
    """Converts a Bitget granularity label into its duration in seconds."""
    if timeframe.endswith('Hutc'):
        return int(timeframe[:-4]) * 3600
    if timeframe.endswith('Dutc'):
        return int(timeframe[:-4]) * 86400
    if timeframe.endswith('H'):
        return int(timeframe[:-1]) * 3600
    if timeframe.endswith('m'):
        return int(timeframe[:-1]) * 60
    raise ValueError(f"Invalid timeframe: {timeframe!r}")


def _epoch_to_utc_str(epoch: float) -> str:
    """Formats an epoch value as a readable UTC timestamp."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def _mark(condition: bool) -> str:
    """Maps a boolean check result to its display mark."""
    return MARK_OK if condition else MARK_FAIL


# ==========================================================================
# CANDLE VALIDATOR
# ==========================================================================
class CandleValidator(BitgetWSManager):
    """
    Independent candle-close detector built on the public WebSocket candle
    channel. Inherits connection, reconnection and keepalive handling from
    BitgetWSManager; only public-message parsing and public-channel
    subscription are specialised.

    The WebSocket is used exclusively to detect the instant a candle closes
    (open_time rollover). It never supplies OHLC values — those are always
    read from REST once a close is detected, avoiding the race condition
    where a late trade update for the closing candle arrives after the
    rollover message.

    Runs on its own socket and its own thread: the trading WebSocket
    instance is never touched.
    """

    def __init__(self, symbol: str, timeframes: List[str]):
        super().__init__()

        self.symbol     = symbol
        self.timeframes = list(timeframes)

        # channel name -> timeframe label (e.g. "candle1H" -> "1H")
        self._channels: Dict[str, str] = {
            f"candle{tf}": tf for tf in self.timeframes
        }

        # Own memory, equivalent to the MT5 last_bar_time pattern.
        # Only the open_time is tracked: it is fixed at bar creation and
        # never mutates, unlike OHLC fields.
        self._open_bar_time_ms:   Dict[str, int]   = {}   # timeframe -> in-progress bar
        self._closed_bar:         Dict[str, dict]  = {}   # timeframe -> {open_time_ms, detected_epoch}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------
    # LIFECYCLE
    # ----------------------------------------------------------------------
    def start(self, connect_timeout: float = 5.0) -> None:
        """Opens the dedicated socket and subscribes to every candle channel."""
        super().start()

        deadline = time.time() + connect_timeout
        while time.time() < deadline and not self._is_public_connected():
            time.sleep(0.1)

        self._resubscribe_public()

        state = "connected" if self._is_public_connected() else "not connected"
        logger.info(
            f"[VERIFY] validator started | symbol={self.symbol} "
            f"| timeframes={', '.join(self.timeframes)} | WS {state}"
        )

    def _is_public_connected(self) -> bool:
        """Reports whether the dedicated public socket is usable."""
        sock = getattr(self.public_ws, 'sock', None)
        return bool(sock and getattr(sock, 'connected', False))

    def _on_public_open(self, ws) -> None:
        """
        Overrides the inherited hook. The base implementation only
        resubscribes when self.subscribed_public is non-empty, a set this
        class never populates (candle channels use their own tracking via
        self._channels). Without this override, every reconnect leaves the
        socket open but silently unsubscribed, freezing detection forever.
        """
        logger.info("[VERIFY] public WS (re)connected")
        self._resubscribe_public()

    # ----------------------------------------------------------------------
    # SUBSCRIPTION (overrides the inherited ticker-only behaviour)
    # ----------------------------------------------------------------------
    def _resubscribe_public(self) -> None:
        """Subscribes to all candle channels. Also runs on every reconnect."""
        if not self._is_public_connected():
            logger.warning("[VERIFY] cannot subscribe | public WS not connected")
            return

        args = [
            {"instType": INSTRUMENT_TYPE, "channel": channel, "instId": self.symbol}
            for channel in self._channels
        ]

        try:
            self.public_ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.debug(f"[VERIFY] subscribed to {len(args)} candle channels")
        except Exception as e:
            logger.error(f"[VERIFY] subscription failed: {e}")

    # ----------------------------------------------------------------------
    # MESSAGE HANDLING (overrides the inherited ticker parser)
    # ----------------------------------------------------------------------
    def _on_public_message(self, ws, message) -> None:
        """Processes candle pushes only; every other payload is ignored."""
        try:
            if not message or message == "pong" or message[0] not in ("{", "["):
                return

            payload = json.loads(message)
            if payload.get('event') in ('pong', 'subscribe', 'error'):
                return
            if payload.get('action') not in ('snapshot', 'update'):
                return

            timeframe = self._channels.get(payload.get('arg', {}).get('channel'))
            if timeframe is None:
                return

            rows = payload.get('data') or []
            if not rows:
                return

            # Only the open_time is needed to detect a rollover; the newest
            # row is the one that matters.
            newest_open_time_ms = max(int(row[0]) for row in rows)
            self._ingest_open_time(timeframe, newest_open_time_ms)

        except Exception as e:
            logger.error(f"[VERIFY] error processing candle message: {e}")

    def _ingest_open_time(self, timeframe: str, open_time_ms: int) -> None:
        """
        Tracks the open_time of the bar currently being built. When it rolls
        over to a newer value, the previous bar is final and this is the
        detection instant for its close — OHLC is deliberately not read
        here; it is always fetched from REST afterwards.
        """
        with self._lock:
            previous_open_time_ms = self._open_bar_time_ms.get(timeframe)
            self._open_bar_time_ms[timeframe] = open_time_ms

            if previous_open_time_ms is None or open_time_ms <= previous_open_time_ms:
                return

            closed = {
                'open_time_ms':    previous_open_time_ms,
                'detected_epoch':  time.time(),
            }
            self._closed_bar[timeframe] = closed

        self._log_ws_detection(timeframe, closed)

    def _log_ws_detection(self, timeframe: str, closed: dict) -> None:
        """Reports the close detected by method B and its publication lag."""
        real_close_epoch = closed['open_time_ms'] / 1000 + timeframe_to_seconds(timeframe)
        push_lag         = closed['detected_epoch'] - real_close_epoch

        logger.info(
            f"[VERIFY] {timeframe} close detected via WS "
            f"| bar_open={_epoch_to_utc_str(closed['open_time_ms'] / 1000)} "
            f"| push_lag={push_lag:+.1f}s {_mark(abs(push_lag) <= WS_DETECTION_TOLERANCE_SECONDS)}"
        )

    # ----------------------------------------------------------------------
    # PUBLIC API — CLOCK TRIGGER COMPARISON
    # ----------------------------------------------------------------------
    def register_clock_trigger(self, timeframe: str, trigger_epoch: Optional[float] = None) -> None:
        """
        Entry point for the orchestrator. Called the moment method A decides
        a candle has closed; emits the full A-vs-B comparison.

        Never raises: any failure is logged and swallowed.
        """
        try:
            trigger_epoch = time.time() if trigger_epoch is None else trigger_epoch

            with self._lock:
                closed = self._closed_bar.get(timeframe)
                closed = dict(closed) if closed else None

            if closed is None:
                logger.info(
                    f"[VERIFY] {timeframe} | clock triggered at "
                    f"{_epoch_to_utc_str(trigger_epoch)} | no WS close recorded yet | skipped"
                )
                return

            self._compare_detection(timeframe, trigger_epoch, closed)
            self._check_identity_and_log_data(timeframe, closed)

        except Exception as e:
            logger.error(f"[VERIFY] {timeframe} | clock trigger validation failed: {e}")

    def _compare_detection(self, timeframe: str, trigger_epoch: float, closed: dict) -> None:
        """
        Check 1 — detection timing.

        Measures how late method A fires relative to the real close observed
        by method B, and whether that delay matches the configured buffer.
        """
        real_close_epoch = closed['open_time_ms'] / 1000 + timeframe_to_seconds(timeframe)
        expected_epoch   = real_close_epoch + CANDLE_TRIGGER_BUFFER_SECONDS

        trigger_delay = trigger_epoch - real_close_epoch
        buffer_drift  = trigger_epoch - expected_epoch
        detection_gap = trigger_epoch - closed['detected_epoch']

        logger.info(
            f"[VERIFY] {timeframe} | detection:"
            f"{_mark(abs(buffer_drift) <= CANDLE_TRIGGER_TOLERANCE_SECONDS)} "
            f"| real_close={_epoch_to_utc_str(real_close_epoch)} "
            f"| clock_delay={trigger_delay:+.1f}s (buffer={CANDLE_TRIGGER_BUFFER_SECONDS}s, "
            f"drift={buffer_drift:+.1f}s) | A_after_B={detection_gap:+.1f}s"
        )

    def _check_identity_and_log_data(self, timeframe: str, closed: dict) -> None:
        """
        Check 2 — identity, plus a reference dump of the REST OHLC.

        Confirms the bar REST reports as the latest candle is the same one
        method B detected as closed. REST is the sole source of OHLC data —
        there is nothing from the WebSocket to compare it against, since
        that would reintroduce the in-flight-candle race condition.
        """
        raw = _call_history_candles(symbol=self.symbol, granularity=timeframe, limit=1)
        if not raw:
            logger.warning(f"[VERIFY] {timeframe} | REST returned no candle | identity skipped")
            return

        row = max(raw, key=lambda r: int(r[0]))
        rest_open_time_ms = int(row[0])
        identity_match     = closed['open_time_ms'] == rest_open_time_ms

        logger.info(
            f"[VERIFY] {timeframe} | identity:{_mark(identity_match)} "
            f"| WS_bar_open={_epoch_to_utc_str(closed['open_time_ms'] / 1000)} "
            f"REST_bar_open={_epoch_to_utc_str(rest_open_time_ms / 1000)} "
            f"| REST_OHLC=(o={row[1]} h={row[2]} l={row[3]} c={row[4]})"
        )

    # ----------------------------------------------------------------------
    # PUBLIC API — READ ACCESS
    # ----------------------------------------------------------------------
    def get_last_closed_bar(self, timeframe: str) -> Optional[dict]:
        """Returns the last bar method B reported as closed, if any."""
        with self._lock:
            closed = self._closed_bar.get(timeframe)
            return dict(closed) if closed else None


# ==========================================================================
# MODULE-LEVEL ACCESS
# ==========================================================================

_validator: Optional[CandleValidator] = None


def init_candle_validator(symbol: str, timeframes: List[str]) -> Optional[CandleValidator]:
    """
    Builds and starts the validator. Returns None on failure so the caller
    can keep running unaffected.
    """
    global _validator

    if _validator is not None:
        return _validator

    try:
        _validator = CandleValidator(symbol=symbol, timeframes=timeframes)
        _validator.start()
        return _validator
    except Exception as e:
        logger.error(f"[VERIFY] validator initialization failed: {e}")
        _validator = None
        return None


def get_candle_validator() -> Optional[CandleValidator]:
    """Returns the running validator instance, or None if not initialized."""
    return _validator


def stop_candle_validator() -> None:
    """Stops the validator and releases the module-level reference."""
    global _validator

    if _validator is None:
        return

    try:
        _validator.stop()
        logger.info("[VERIFY] validator stopped")
    except Exception as e:
        logger.error(f"[VERIFY] validator shutdown failed: {e}")
    finally:
        _validator = None