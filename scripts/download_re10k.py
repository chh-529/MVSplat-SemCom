"""Route C: download RealEstate10K clips from YouTube and pack them directly
into the pixelSplat/MVSplat chunk format (matches datasets/re10k exactly).

Prerequisite: the official RealEstate10K pose .txt files (from
https://google.github.io/realestate10k/), laid out as
    <poses-dir>/<stage>/<key>.txt
Each .txt: line 0 = YouTube URL; following lines = "<timestamp_us> <18 cam vals>".

For each video (clips are grouped by video id so each video downloads once) we
grab the 360p stream, single-pass decode to pull the frame nearest each
timestamp, JPEG-encode it, and assemble an example dict identical to the ground
truth: {url, timestamps int64[N], cameras float32[N,18], images list[uint8 jpg],
key}. Chunks flush at ~100 MB. Videos that are gone/blocked are skipped.

Usage:
  python scripts/download_re10k.py --poses-dir /tmp2/cshsieh/re10k_raw/RealEstate10K \
    --stage train --out datasets/re10k_dl --target-scenes 3000
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

TARGET_BYTES_PER_CHUNK = int(1e8)
W, H = 640, 360


def parse_pose(txt_path: Path):
    lines = txt_path.read_text().splitlines()
    url = lines[0].strip()
    ts, cams = [], []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        t, *cam = ln.split()
        ts.append(int(t))
        cams.append(np.asarray(cam, dtype=np.float64))
    return url, ts, np.stack(cams)


def video_id(url: str) -> str:
    for sep in ("v=", "youtu.be/"):
        if sep in url:
            return url.split(sep)[1][:11]
    return url[-11:]


def download_video(url: str, dst: Path) -> bool:
    """Grab a <=360p stream. 134 = 360p video-only mp4 (no audio needed)."""
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "--quiet", "--no-playlist",
        "-f", "134/18/best[height<=360]",
        "-o", str(dst), url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst.exists()
    except Exception:
        return False


def extract_frames(video_path: Path, timestamps_us: list[int]) -> dict[int, bytes] | None:
    """Single-pass decode; for each target timestamp keep the frame whose PTS is
    closest. Returns {timestamp: jpg_bytes} or None if the video can't cover all
    requested timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    targets = sorted(timestamps_us)
    out, ti = {}, 0
    prev_frame, prev_us = None, -1.0
    while ti < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        cur_us = cap.get(cv2.CAP_PROP_POS_MSEC) * 1000.0
        while ti < len(targets) and cur_us >= targets[ti]:
            # pick whichever of prev/cur frame is closer in time
            if prev_frame is not None and abs(prev_us - targets[ti]) < abs(cur_us - targets[ti]):
                chosen = prev_frame
            else:
                chosen = frame
            if chosen.shape[1] != W or chosen.shape[0] != H:
                chosen = cv2.resize(chosen, (W, H), interpolation=cv2.INTER_AREA)
            ok_enc, buf = cv2.imencode(".jpg", chosen, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out[targets[ti]] = buf.tobytes()
            ti += 1
        prev_frame, prev_us = frame, cur_us
    cap.release()
    return out if ti == len(targets) else None


def build_example(url, key, timestamps, cameras, frames: dict[int, bytes]):
    imgs = [torch.frombuffer(bytearray(frames[t]), dtype=torch.uint8) for t in timestamps]
    return {
        "url": url,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(cameras, dtype=torch.float32),
        "images": imgs,
        "key": key,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=["train", "test"], required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/re10k_dl"))
    ap.add_argument("--target-scenes", type=int, default=3000)
    ap.add_argument("--max-frames", type=int, default=200,
                    help="skip clips with more frames than this (keeps chunks small)")
    ap.add_argument("--shuffle", action="store_true", help="random clip order")
    ap.add_argument("--only-keys-from", type=Path, default=None,
                    help="restrict to scene keys present in this eval-index json "
                         "(use for the test split so scenes are evaluable)")
    args = ap.parse_args()

    keep_keys = None
    if args.only_keys_from is not None:
        keep_keys = set(json.loads(args.only_keys_from.read_text()).keys())
        print(f"restricting to {len(keep_keys)} evaluable keys", flush=True)

    out_dir = args.out / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    done_keys = set(index.keys())
    chunk_index = len(set(index.values()))

    txts = sorted((args.poses_dir / args.stage).glob("*.txt"))
    if args.shuffle:
        random.seed(0)
        random.shuffle(txts)

    by_video = defaultdict(list)
    for t in txts:
        if t.stem in done_keys:
            continue
        if keep_keys is not None and t.stem not in keep_keys:
            continue
        by_video[video_id(parse_pose(t)[0])].append(t)

    chunk, chunk_bytes = [], 0
    n_done = len(done_keys)
    print(f"[{args.stage}] resume from {n_done} scenes, target {args.target_scenes}", flush=True)

    def flush():
        nonlocal chunk, chunk_bytes, chunk_index
        if not chunk:
            return
        name = f"{chunk_index:0>6}.torch"
        torch.save(chunk, out_dir / name)
        for ex in chunk:
            index[ex["key"]] = name
        index_path.write_text(json.dumps(index))
        print(f"  saved {name}: {len(chunk)} scenes ({chunk_bytes/1e6:.0f} MB), total {len(index)}", flush=True)
        chunk, chunk_bytes, chunk_index = [], 0, chunk_index + 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for vid, clips in by_video.items():
            if n_done >= args.target_scenes:
                break
            vpath = tmp / f"{vid}.mp4"
            time.sleep(random.uniform(0.5, 2.0))  # be gentle on YouTube
            if not download_video(clips[0].read_text().splitlines()[0].strip(), vpath):
                print(f"  skip video {vid} (unavailable)", flush=True)
                continue
            for clip in clips:
                if n_done >= args.target_scenes:
                    break
                try:
                    url, ts, cams = parse_pose(clip)
                    if len(ts) > args.max_frames:
                        continue
                    frames = extract_frames(vpath, ts)
                    if frames is None:
                        continue
                    ex = build_example(url, clip.stem, ts, cams, frames)
                    nbytes = sum(im.numel() for im in ex["images"])
                    chunk.append(ex)
                    chunk_bytes += nbytes
                    n_done += 1
                    if chunk_bytes >= TARGET_BYTES_PER_CHUNK:
                        flush()
                except Exception as e:
                    print(f"  clip {clip.stem} failed: {e}", flush=True)
            vpath.unlink(missing_ok=True)
        flush()

    print(f"[{args.stage}] DONE: {len(index)} scenes total", flush=True)


if __name__ == "__main__":
    main()
