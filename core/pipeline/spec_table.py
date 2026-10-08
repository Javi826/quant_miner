# core/pipeline/spec_table.py
import sys
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from signals.indicators_bank import ConditionBank
from signals.signal_builder import build_signal_fn

if sys.byteorder != "little":
    raise ImportError("spec_table: the bit layout of the spec table assumes a little-endian machine")

#==============================================================================
# CONFIG
#==============================================================================
SPEC_TABLE_N_JOBS = -1
#------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecTable:
    # words[s, w]: bit b of word w is bar 64 * (w - word_offsets[i]) + b of symbols[i] for spec s,
    # shifted one bar like the backtest-mode signal. Every symbol starts on a word boundary and its padding
    # bits are zero, so popcounts and ANDs over whole rows match the unpadded concatenation.
    words: np.ndarray
    symbols: tuple
    word_offsets: np.ndarray
    n_bars: np.ndarray
    index_by_identity: dict

    @property
    def n_specs(self) -> int:
        return self.words.shape[0]

    def shared_arrays(self) -> dict:
        return {
            "words":        self.words,
            "word_offsets": self.word_offsets,
            "n_bars":       self.n_bars,
            "symbols":      self.symbols,
        }


def spec_identity(spec: dict) -> tuple:
    return (spec["key"], spec["op"], spec["threshold"])


def collect_unique_specs(rules: list) -> tuple:
    unique_specs      = []
    index_by_identity = {}
    for rule in rules:
        for spec in rule["specs"]:
            identity = spec_identity(spec)
            row      = index_by_identity.get(identity)
            if row is None:
                index_by_identity[identity] = len(unique_specs)
                unique_specs.append(spec)
            elif (spec["indicator"], spec["params"]) != (unique_specs[row]["indicator"], unique_specs[row]["params"]):
                raise ValueError(f"spec identity {identity} is shared by different indicators or params")
    return unique_specs, index_by_identity


def rule_spec_index(rules: list, index_by_identity: dict) -> tuple:
    if not rules:
        return np.empty((0, 1), dtype=np.int32), np.array([], dtype=str)
    # Rules share their specs lists (the long and short rule of a template) and their spec dicts (every template),
    # so each list and each spec is resolved once per call; `rules` keeps those objects alive meanwhile.
    row_by_list = {}
    row_by_spec = {}
    flat        = []
    lengths     = []
    rule_lists  = []
    try:
        for rule in rules:
            specs = rule["specs"]
            row   = row_by_list.get(id(specs))
            if row is None:
                if not specs:
                    raise ValueError(f"rule {rule.get('rule_id', '?')} has no specs")
                for spec in specs:
                    idx = row_by_spec.get(id(spec))
                    if idx is None:
                        idx = row_by_spec[id(spec)] = index_by_identity[spec_identity(spec)]
                    flat.append(idx)
                row = row_by_list[id(specs)] = len(lengths)
                lengths.append(len(specs))
            rule_lists.append(row)
    except KeyError as exc:
        raise ValueError(f"spec {exc.args[0]} is not in the spec table: build the table from these rules") from None

    lengths = np.asarray(lengths, dtype=np.int64)
    starts  = np.cumsum(lengths) - lengths
    cols    = np.arange(int(lengths.max()))
    # AND is idempotent: padding with the rule's first spec leaves its signal unchanged.
    offsets  = np.where(cols[None, :] < lengths[:, None], cols[None, :], 0)
    by_list  = np.asarray(flat, dtype=np.int32)[starts[:, None] + offsets]
    spec_idx = np.ascontiguousarray(by_list[np.asarray(rule_lists, dtype=np.int64)])
    sides    = np.array([rule["side"] for rule in rules])
    return spec_idx, sides


def _symbol_words(unique_specs: list, arr: dict) -> np.ndarray:
    bank = ConditionBank(arr)
    rows = np.empty((len(unique_specs), bank.n), dtype=bool)
    for i, spec in enumerate(unique_specs):
        rows[i] = build_signal_fn([spec], "long")(arr, live_trading=False, bank=bank).astype(bool)

    packed       = np.packbits(rows, axis=1, bitorder="little")
    word_padding = (-packed.shape[1]) % np.dtype(np.uint64).itemsize
    if word_padding:
        packed = np.pad(packed, ((0, 0), (0, word_padding)))
    return np.ascontiguousarray(packed).view(np.uint64)


def build_spec_table(
    rules: list,
    ohlcv_arr: dict,
    timeframe: str = "",
    n_jobs: int = SPEC_TABLE_N_JOBS,
    show_progress: bool = True,
) -> SpecTable:
    unique_specs, index_by_identity = collect_unique_specs(rules)
    symbols = tuple(sorted(ohlcv_arr))

    # disable is only passed when hiding the bar: callers that silence this module's tqdm keep control otherwise
    bar_kwargs = {"desc": f"SIGNAL MASK     {timeframe}", "total": len(symbols), "dynamic_ncols": True}
    if not show_progress:
        bar_kwargs["disable"] = True
    words_by_symbol = list(tqdm(
        Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
            delayed(_symbol_words)(unique_specs, ohlcv_arr[sym])
            for sym in symbols
        ),
        **bar_kwargs,
    ))

    n_words      = np.array([w.shape[1] for w in words_by_symbol], dtype=np.int64)
    word_offsets = np.concatenate(([0], np.cumsum(n_words))).astype(np.int64)
    words        = (np.ascontiguousarray(np.concatenate(words_by_symbol, axis=1)) if words_by_symbol
                    else np.empty((len(unique_specs), 0), dtype=np.uint64))
    n_bars       = np.array([len(ohlcv_arr[sym]["close"]) for sym in symbols], dtype=np.int64)

    return SpecTable(
        words             = words,
        symbols           = symbols,
        word_offsets      = word_offsets,
        n_bars            = n_bars,
        index_by_identity = index_by_identity,
    )