"""Convert Kokoro voice style tables (``.pt``) to torch-free ``.npy`` files.

The runtime reads ``voices/<id>.npy`` with plain numpy; this one-off tool reads the
upstream ``voices/<id>.pt`` (torch tensors) and writes ``.npy`` so the service can
ship without torch.

    python tools/convert_voices.py \
        --voices-dir models/kokoro-82m-v1.0/voices \
        --voices af_heart,af_sarah,am_michael,bf_emma
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


def convert_voices(voices_dir: Path, source_repo: str, voice_ids: list[str]) -> list[Path]:
    import torch  # only here, to read .pt; not needed at service runtime

    out = Path(voices_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for v in voice_ids:
        src = hf_hub_download(source_repo, f"voices/{v}.pt")
        table = torch.load(src, map_location="cpu", weights_only=True).numpy().astype(np.float32)
        dst = out / f"{v}.npy"
        np.save(dst, table)
        produced.append(dst)
        print(f"  {dst}  shape={table.shape}")
    return produced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voices-dir", required=True, help="Dir to write voices/<id>.npy into")
    ap.add_argument("--source-repo", default="hexgrad/Kokoro-82M", help="HF repo with voices/*.pt")
    ap.add_argument("--voices", required=True, help="Comma-separated voice ids")
    a = ap.parse_args()
    ids = [v.strip() for v in a.voices.split(",") if v.strip()]
    print(f"Converting {len(ids)} voice(s) from {a.source_repo} -> {a.voices_dir}")
    convert_voices(Path(a.voices_dir), a.source_repo, ids)
    print("done")


if __name__ == "__main__":
    main()
