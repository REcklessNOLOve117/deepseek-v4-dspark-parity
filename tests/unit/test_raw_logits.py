from pathlib import Path

import numpy as np

from dsv4_parity.raw_logits import bfloat16_to_float32, load_raw_logits


def test_bfloat16_decoder():
    source = np.array([0.0, 1.0, -2.5, np.inf], dtype=np.float32)
    words = (source.view(np.uint32) >> 16).astype("<u2")
    decoded = bfloat16_to_float32(words)
    np.testing.assert_array_equal(decoded, source)


def test_offset_raw_load(tmp_path: Path):
    path = tmp_path / "raw_logits.bin"
    path.write_bytes(b"skip" + np.array([1.0, 2.0], dtype="<f4").tobytes())
    row = {
        "raw_file": path.name,
        "raw_offset_bytes": 4,
        "raw_length_bytes": 8,
        "raw_dtype": "float32",
        "vocab_size": 2,
    }
    np.testing.assert_array_equal(load_raw_logits(tmp_path, row), [1.0, 2.0])

