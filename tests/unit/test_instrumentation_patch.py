from pathlib import Path

from dsv4_parity.instrumentation_patch import (
    GRAPH_NEW,
    GRAPH_OLD,
    RETURN_NEW,
    RETURN_OLD,
    REPLAY_DRAFT_NEW,
    REPLAY_DRAFT_OLD,
    REPLAY_PREPARE_NEW,
    REPLAY_PREPARE_OLD,
    SAMPLE_NEW,
    SAMPLE_OLD,
    install,
    install_legacy,
    restore,
)


def test_patch_and_hash_verified_restore(tmp_path: Path):
    target = tmp_path / "runner.py"
    backup = tmp_path / "runner.py.orig"
    original = (
        "prefix\n"
        + GRAPH_OLD
        + SAMPLE_OLD
        + RETURN_OLD
        + REPLAY_PREPARE_OLD
        + REPLAY_DRAFT_OLD
        + "suffix\n"
    )
    target.write_text(original, encoding="utf-8")
    hashes = install(target, backup)
    patched = target.read_text(encoding="utf-8")
    assert SAMPLE_NEW in patched
    assert RETURN_NEW in patched
    assert GRAPH_NEW in patched
    assert REPLAY_PREPARE_NEW in patched
    assert REPLAY_DRAFT_NEW in patched
    assert hashes["original_sha256"] != hashes["patched_sha256"]
    restored = restore(target, backup)
    assert restored["backup_sha256"] == restored["restored_sha256"]
    assert target.read_text(encoding="utf-8") == original


def test_legacy_patch_and_restore(tmp_path: Path):
    from dsv4_parity.instrumentation_patch import (
        LEGACY_ACCEPT_NEW,
        LEGACY_ACCEPT_OLD,
        LEGACY_CAPTURE_NEW,
        LEGACY_CAPTURE_OLD,
    )

    target = tmp_path / "gpu_model_runner.py"
    backup = tmp_path / "gpu_model_runner.py.orig"
    original = (
        "class Runner:\n"
        "    def execute(self):\n"
        + LEGACY_CAPTURE_OLD
        + "            None\n        )\n"
        + LEGACY_ACCEPT_OLD
        + "            None\n        )\n"
    )
    target.write_text(original, encoding="utf-8")
    install_legacy(target, backup)
    patched = target.read_text(encoding="utf-8")
    assert LEGACY_CAPTURE_NEW in patched
    assert LEGACY_ACCEPT_NEW in patched
    compile(patched, str(target), "exec")
    restore(target, backup)
    assert target.read_text(encoding="utf-8") == original
