"""
labeling/write_auto_labels.py — writes auto_labels.json into each episode.hdf5
so training can use auto labels without opening every episode in the GUI.

Episodes without saved labels get the auto labels (source=auto, reviewed=False).
Episodes with saved labels are skipped, unless --refresh-auto is given: then only
frames whose source is still "auto" are replaced and manual frames are kept.

Usage:
    python labeling/write_auto_labels.py <store_root>
    python labeling/write_auto_labels.py <store_root> --refresh-auto
    python labeling/write_auto_labels.py <store_root> --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema.yaml"
AUTO_NAME = "auto_labels.json"
SRC_DEFAULT, SRC_AUTO, SRC_MANUAL = 0, 1, 2


def load_schema():
    with open(SCHEMA_PATH) as f:
        s = yaml.safe_load(f)
    ids = {p["name"]: p["id"] for p in s["phases"]}
    legend = "  ".join(f"{p['id']}={p['name']}" for p in sorted(s["phases"], key=lambda p: p["id"]))
    return s, ids, legend


def _decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def rasterize(arm: dict, ts: np.ndarray, ids: dict, null: str):
    T = len(ts)
    ph = np.zeros(T, np.int8)
    ob = [null] * T
    go = [null] * T
    covered = np.zeros(T, bool)
    for s in arm.get("segments", []):
        a = int(np.clip(np.searchsorted(ts, s["start_ns"]), 0, T - 1))
        b = min(max(int(np.searchsorted(ts, s["end_ns"], side="right")), a + 1), T)
        ph[a:b] = ids.get(s["phase"], 0)
        for t in range(a, b):
            ob[t], go[t] = s["object"], s["goal"]
        covered[a:b] = True
    return ph, ob, go, covered


def read_existing(lbl, side, T, null):
    pk, tk, gk, sk = (f"arm_{side}_phase", f"arm_{side}_target_name", f"arm_{side}_goal_name", f"arm_{side}_source")
    ph = lbl[pk][:].astype(np.int8) if pk in lbl and len(lbl[pk]) == T else np.zeros(T, np.int8)
    ob = [_decode(v).rstrip("\x00") for v in lbl[tk][:]] if tk in lbl and len(lbl[tk]) == T else [null] * T
    go = [_decode(v).rstrip("\x00") for v in lbl[gk][:]] if gk in lbl and len(lbl[gk]) == T else [null] * T
    if sk in lbl and len(lbl[sk]) == T:
        src = lbl[sk][:].astype(np.int8)
    else:
        src = np.full(T, SRC_MANUAL, np.int8)
    return ph, ob, go, src


def write_side(lbl, side, ph, ob, go, src, legend):
    for k in ("phase", "target_name", "goal_name", "source"):
        n = f"arm_{side}_{k}"
        if n in lbl:
            del lbl[n]
    ds = lbl.create_dataset(f"arm_{side}_phase", data=ph)
    ds.attrs["legend"] = legend
    lbl.create_dataset(f"arm_{side}_target_name", data=np.array([n.encode() for n in ob], dtype="S32"))
    lbl.create_dataset(f"arm_{side}_goal_name", data=np.array([n.encode() for n in go], dtype="S32"))
    ds = lbl.create_dataset(f"arm_{side}_source", data=src)
    ds.attrs["legend"] = "0=default 1=auto 2=manual"


def process(folder: Path, refresh: bool, dry: bool, schema, ids, legend) -> str:
    h5p, jp = folder / "episode.hdf5", folder / AUTO_NAME
    if not h5p.exists():
        return "no episode.hdf5"
    if not jp.exists():
        return "no auto_labels.json"
    data = json.loads(jp.read_text())
    null = schema.get("null_target", "null")
    with h5py.File(h5p, "r") as ep:
        has = "labels" in ep and any(k.endswith("_phase") for k in ep["labels"].keys())
        if has and not refresh:
            rv = bool(ep["labels"].attrs.get("reviewed", False))
            return f"skipped (already {'reviewed' if rv else 'labeled'})"
    if dry:
        return "would refresh auto frames" if has else "would write auto labels"

    with h5py.File(h5p, "r+") as ep:
        ts = ep["observations"]["timestamp_ns"][:].astype(np.int64)
        T = len(ts)
        sides = [s for s in ("left", "right") if f"arm_{s}" in ep["observations"]] or ["left"]
        lbl = ep["labels"] if "labels" in ep else ep.create_group("labels")
        n_manual = 0
        events = []
        for side in sides:
            arm = data.get("arms", {}).get(side)
            if arm is None:
                aph, aob, ago, cov = np.zeros(T, np.int8), [null] * T, [null] * T, np.zeros(T, bool)
            else:
                aph, aob, ago, cov = rasterize(arm, ts, ids, null)
                events += [dict(e, side=side) for e in arm.get("events", [])]
            new_src = np.where(cov, SRC_AUTO, SRC_DEFAULT).astype(np.int8)
            if has:
                ph, ob, go, src = read_existing(lbl, side, T, null)
                keep = src == SRC_MANUAL
                n_manual += int(keep.sum())
                for t in np.flatnonzero(~keep):
                    ph[t], ob[t], go[t], src[t] = aph[t], aob[t], ago[t], new_src[t]
            else:
                ph, ob, go, src = aph, aob, ago, new_src
            write_side(lbl, side, ph, ob, go, src, legend)
        if "segments" in lbl:
            del lbl["segments"]
        lbl.attrs["pass2_complete"] = True
        lbl.attrs["schema_version"] = schema["schema_version"]
        lbl.attrs["task"] = data.get("task", "")
        lbl.attrs["events"] = json.dumps(events)
        if not has:
            lbl.attrs["reviewed"] = False
        lbl.attrs["label_source"] = "auto+manual" if n_manual else "auto"
    return f"refreshed auto frames, kept {n_manual} manual" if has else "wrote auto labels"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store_root")
    ap.add_argument("--refresh-auto", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    schema, ids, legend = load_schema()
    root = Path(args.store_root)
    folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "episode.hdf5").exists())
    for f in folders:
        try:
            print(f"  {f.name}: {process(f, args.refresh_auto, args.dry_run, schema, ids, legend)}")
        except Exception as e:
            print(f"  [error] {f.name}: {e}")


if __name__ == "__main__":
    main()
