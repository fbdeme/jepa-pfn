"""Upload one save_at snapshot of a run to a private HF model repo (called in the background by train_jepa).

    HF_TOKEN=... uv run python -m train.hf_upload runs/RUN STEP fbdeme/jepa-pfn-ckpt

Uploads ckpt_STEP.pt, resume_STEP.pt, metrics.jsonl, config.yaml under RUN/ in one commit. The token
comes from the environment only (never a config key: config.yaml is itself uploaded). The repo must
already exist (created once from the laptop; the box token needs write access to it and nothing else).
"""
import os
import sys
import time
from pathlib import Path

# 2026-09-09: the Xet (CAS) path returned 401 on the second snapshot of a long run after the first one succeeded
# with the same token (hf_xet's short-lived CAS token cache); plain LFS HTTP upload has no such state.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi  # noqa: E402


def main():
    run_dir, step, repo = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    files = [f"ckpt_{step}.pt", f"resume_{step}.pt", "metrics.jsonl", "config.yaml"]
    missing = [f for f in files if not (run_dir / f).exists()]
    assert not missing, f"missing {missing} in {run_dir}"
    api = HfApi(token=os.environ["HF_TOKEN"])
    for attempt in range(5):   # ponytail: flat retry; the box's uplink drops now and then
        try:
            api.upload_folder(folder_path=str(run_dir), path_in_repo=run_dir.name, repo_id=repo, repo_type="model",
                              allow_patterns=files, commit_message=f"{run_dir.name} step {step}")
            print(f"{time.strftime('%H:%M:%S')} uploaded {run_dir.name} step {step}: {files}", flush=True)
            return
        except Exception as e:   # noqa: BLE001
            print(f"{time.strftime('%H:%M:%S')} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(30 * (attempt + 1))
    sys.exit(1)


if __name__ == "__main__":
    main()
