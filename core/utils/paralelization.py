# core/utils/paralelization.py
import numpy as np
from multiprocessing.shared_memory import SharedMemory


def arrays_to_shared_memory(base_arrays: dict) -> tuple:
    shm_list = []
    metadata = {}
    for sym, arr_dict in base_arrays.items():
        metadata[sym] = {}
        for key, arr in arr_dict.items():
            if isinstance(arr, np.ndarray):
                shm    = SharedMemory(create=True, size=max(arr.nbytes, 1))
                buf    = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                buf[:] = arr
                shm_list.append(shm)
                metadata[sym][key] = {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
            else:
                metadata[sym][key] = {"value": arr}
    return shm_list, metadata


def arrays_from_shared_memory(metadata: dict) -> tuple:
    base_arrays = {}
    shm_handles = []
    for sym, fields in metadata.items():
        base_arrays[sym] = {}
        for key, info in fields.items():
            if "name" in info:
                shm = SharedMemory(name=info["name"], create=False)
                shm_handles.append(shm)
                base_arrays[sym][key] = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
            else:
                base_arrays[sym][key] = info["value"]
    return base_arrays, shm_handles


def compact_columns_inplace(mask: np.ndarray, *arrays: np.ndarray, chunk_size: int = 5000, axis: int = 1) -> int:

    keep_idx = np.flatnonzero(mask)
    n_keep = keep_idx.size

    for arr in arrays:
        for chunk_start in range(0, n_keep, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_keep)
            src_idx = keep_idx[chunk_start:chunk_end]
            if arr.ndim == 1:
                arr[chunk_start:chunk_end] = arr[src_idx]
            elif axis == 1:
                arr[:, chunk_start:chunk_end] = arr[:, src_idx]
            else:
                arr[chunk_start:chunk_end, :] = arr[src_idx, :]

    return n_keep
