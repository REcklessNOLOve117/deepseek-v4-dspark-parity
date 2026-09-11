from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


SAMPLE_OLD = """        logits = self.model.compute_logits(sample_hidden_states)\n        if grammar_output is not None:\n"""
SAMPLE_NEW = """        logits = self.model.compute_logits(sample_hidden_states)\n        from dsv4_parity.capture_runtime import maybe_capture_target_logits\n        maybe_capture_target_logits(self, logits, input_batch)\n        if grammar_output is not None:\n"""

RETURN_OLD = """        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected\n"""
RETURN_NEW = """        from dsv4_parity.replay_runtime import maybe_force_replay_sample\n        maybe_force_replay_sample(self, sampler_output, input_batch)\n        from dsv4_parity.capture_runtime import maybe_capture_acceptance\n        maybe_capture_acceptance(self, sampler_output, input_batch)\n        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected\n"""

GRAPH_OLD = """        if batch_desc.num_tokens == 0:\n"""
GRAPH_NEW = """        self._dsv4_parity_graph_mode = batch_desc.cg_mode\n\n        if batch_desc.num_tokens == 0:\n"""

REPLAY_PREPARE_OLD = """            input_batch = self.prepare_inputs(scheduler_output, batch_desc)\n            block_tables, slot_mappings = self.prepare_attn(input_batch)\n            # Mamba \"align\" pre-copy: migrate recurrent state across block\n"""
REPLAY_PREPARE_NEW = """            input_batch = self.prepare_inputs(scheduler_output, batch_desc)\n            block_tables, slot_mappings = self.prepare_attn(input_batch)\n            from dsv4_parity.replay_runtime import maybe_run_fixed_replay\n            maybe_run_fixed_replay(\n                self, scheduler_output, batch_desc, input_batch,\n                block_tables, slot_mappings,\n            )\n            # Mamba \"align\" pre-copy: migrate recurrent state across block\n"""

REPLAY_DRAFT_OLD = """            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"""
REPLAY_DRAFT_NEW = """            from dsv4_parity.replay_runtime import maybe_force_replay_drafts\n            draft_tokens = maybe_force_replay_drafts(self, input_batch, draft_tokens)\n            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"""

LEGACY_CAPTURE_OLD = """        self.execute_model_state = ExecuteModelState(\n"""
LEGACY_CAPTURE_NEW = """        from dsv4_parity.capture_runtime import maybe_capture_target_logits_legacy\n        maybe_capture_target_logits_legacy(\n            self, logits, scheduler_output, logits_indices, positions,\n            cudagraph_mode, num_tokens_unpadded, batch_desc.num_tokens,\n            spec_decode_metadata,\n        )\n\n        self.execute_model_state = ExecuteModelState(\n"""
LEGACY_ACCEPT_OLD = """        self._update_states_after_model_execute(\n"""
LEGACY_ACCEPT_NEW = """        from dsv4_parity.capture_runtime import maybe_capture_acceptance_legacy\n        maybe_capture_acceptance_legacy(self, sampler_output)\n\n        self._update_states_after_model_execute(\n"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(target: Path, backup: Path) -> dict[str, str]:
    original = target.read_text(encoding="utf-8")
    for needle in (
        SAMPLE_OLD,
        RETURN_OLD,
        GRAPH_OLD,
        REPLAY_PREPARE_OLD,
        REPLAY_DRAFT_OLD,
    ):
        if original.count(needle) != 1:
            raise RuntimeError(f"expected exactly one patch anchor, found {original.count(needle)}")
    if backup.exists():
        if digest(backup) != digest(target):
            raise RuntimeError("backup already exists but differs from current target")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    patched = (
        original.replace(SAMPLE_OLD, SAMPLE_NEW)
        .replace(RETURN_OLD, RETURN_NEW)
        .replace(GRAPH_OLD, GRAPH_NEW)
        .replace(REPLAY_PREPARE_OLD, REPLAY_PREPARE_NEW)
        .replace(REPLAY_DRAFT_OLD, REPLAY_DRAFT_NEW)
    )
    target.write_text(patched, encoding="utf-8")
    return {"original_sha256": digest(backup), "patched_sha256": digest(target)}


def restore(target: Path, backup: Path) -> dict[str, str]:
    if not backup.exists():
        raise FileNotFoundError(backup)
    backup_hash = digest(backup)
    shutil.copy2(backup, target)
    restored_hash = digest(target)
    if backup_hash != restored_hash:
        raise RuntimeError("restore hash mismatch")
    return {"backup_sha256": backup_hash, "restored_sha256": restored_hash}


def install_legacy(target: Path, backup: Path) -> dict[str, str]:
    original = target.read_text(encoding="utf-8")
    for needle in (LEGACY_CAPTURE_OLD, LEGACY_ACCEPT_OLD):
        if original.count(needle) != 1:
            raise RuntimeError(f"expected exactly one legacy patch anchor, found {original.count(needle)}")
    if backup.exists():
        if digest(backup) != digest(target):
            raise RuntimeError("legacy backup already exists but differs from current target")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    patched = original.replace(LEGACY_CAPTURE_OLD, LEGACY_CAPTURE_NEW).replace(
        LEGACY_ACCEPT_OLD, LEGACY_ACCEPT_NEW
    )
    target.write_text(patched, encoding="utf-8")
    return {"original_sha256": digest(backup), "patched_sha256": digest(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "restore"))
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--variant", choices=("modular", "legacy"), default="modular")
    args = parser.parse_args()
    if args.action == "restore":
        result = restore(args.target, args.backup)
    elif args.variant == "legacy":
        result = install_legacy(args.target, args.backup)
    else:
        result = install(args.target, args.backup)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
