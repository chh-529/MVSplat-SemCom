"""Assemble a larger RE10K working set from a full pixelSplat-format source.

Selects whole chunks (each ~13 scenes) until the target scene count is reached,
symlinks them into <out>/<split>/, and writes a filtered index.json covering
exactly the selected scenes. For the test split, chunks are prioritized by how
many of their scenes appear in the evaluation index, so the eval set is dense.

Usage:
  python scripts/build_re10k_subset.py \
    --source /path/to/full_re10k \
    --out datasets/re10k_big \
    --train-scenes 3000 --test-scenes 500 \
    --eval-index assets/evaluation_index_re10k.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def select_split(src_dir: Path, out_dir: Path, target: int, eval_keys: set | None, link: bool):
    index = json.loads((src_dir / "index.json").read_text())  # scene_key -> chunk filename
    by_chunk = defaultdict(list)
    for scene, chunk in index.items():
        by_chunk[chunk].append(scene)

    def chunk_score(chunk):
        scenes = by_chunk[chunk]
        if eval_keys is None:
            return len(scenes)
        return sum(1 for s in scenes if s in eval_keys)  # prioritize evaluable scenes

    chunks = sorted(by_chunk, key=chunk_score, reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    picked, n_scenes, n_eval = [], 0, 0
    sub_index = {}
    for chunk in chunks:
        scenes = by_chunk[chunk]
        if eval_keys is not None and chunk_score(chunk) == 0:
            break  # remaining chunks have no evaluable scenes
        picked.append(chunk)
        for s in scenes:
            sub_index[s] = chunk
        n_scenes += len(scenes)
        if eval_keys is not None:
            n_eval += sum(1 for s in scenes if s in eval_keys)
        count = n_eval if eval_keys is not None else n_scenes
        if count >= target:
            break

    for chunk in picked:
        dst = out_dir / chunk
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            dst.symlink_to((src_dir / chunk).resolve())
        else:
            import shutil
            shutil.copy2(src_dir / chunk, dst)
    (out_dir / "index.json").write_text(json.dumps(sub_index))

    msg = f"{out_dir}: {len(picked)} chunks, {n_scenes} scenes"
    if eval_keys is not None:
        msg += f", {n_eval} evaluable"
    print(msg)
    return n_scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="full re10k dir with train/ and test/")
    ap.add_argument("--out", type=Path, default=Path("datasets/re10k_big"))
    ap.add_argument("--train-scenes", type=int, default=3000)
    ap.add_argument("--test-scenes", type=int, default=500)
    ap.add_argument("--eval-index", type=Path, default=Path("assets/evaluation_index_re10k.json"))
    ap.add_argument("--copy", action="store_true", help="copy chunks instead of symlinking")
    args = ap.parse_args()

    eval_keys = set(json.loads(args.eval_index.read_text()).keys()) if args.eval_index.exists() else None

    print("== train ==")
    select_split(args.source / "train", args.out / "train", args.train_scenes, None, not args.copy)
    print("== test ==")
    select_split(args.source / "test", args.out / "test", args.test_scenes, eval_keys, not args.copy)
    print(f"\nDone. Point experiments at: dataset.roots=[{args.out}]")


if __name__ == "__main__":
    main()
