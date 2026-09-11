from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


NUMPY_DTYPES = {
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


def bfloat16_to_float32(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype="<u2")
    return (words.astype(np.uint32) << 16).view(np.float32)


def load_raw_logits(capture_dir: str | Path, row: dict[str, Any]) -> np.ndarray:
    directory = Path(capture_dir)
    path = directory / row["raw_file"]
    offset = int(row["raw_offset_bytes"])
    length = int(row["raw_length_bytes"])
    dtype_name = str(row["raw_dtype"])
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(length)
    if len(payload) != length:
        raise EOFError(f"short raw-logits read at {path}:{offset}: {len(payload)} != {length}")
    if dtype_name == "bfloat16":
        values = bfloat16_to_float32(np.frombuffer(payload, dtype="<u2"))
    else:
        try:
            values = np.frombuffer(payload, dtype=NUMPY_DTYPES[dtype_name]).astype(np.float32)
        except KeyError as exc:
            raise ValueError(f"unsupported raw dtype: {dtype_name}") from exc
    vocab_size = int(row["vocab_size"])
    if values.size != vocab_size:
        raise ValueError(f"vocab size mismatch: decoded {values.size}, metadata {vocab_size}")
    return values


def raw_bytes(capture_dir: str | Path, row: dict[str, Any]) -> bytes:
    path = Path(capture_dir) / row["raw_file"]
    with path.open("rb") as handle:
        handle.seek(int(row["raw_offset_bytes"]))
        return handle.read(int(row["raw_length_bytes"]))

