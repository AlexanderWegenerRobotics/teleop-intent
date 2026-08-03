"""Fits GRUIntentModel from labeled episodes and saves a checkpoint.

Run from the repo root, same shape of invocation as the HMM:

    python -m models.gru.train --store-root /path/to/avatar --out checkpoints/gru/v1.pt

Reads the same splits and the same is_genuinely_labeled filter, so the two
models are trained and held out on identical episodes.

FOUR DECISIONS THAT CARRY OVER FROM THE HMM WORK
------------------------------------------------
1. CLASS-WEIGHTED PHASE LOSS. Plain cross-entropy on this dataset optimises
   accuracy, and accuracy is won by predicting IDLE -- which is exactly the
   failure the HMM's emission bias produced, arrived at by a different route.
   Weighting inversely by class frequency makes the loss track macro F1, the
   metric the model is actually selected and reported on.

2. SELECTION ON VAL MACRO F1, NOT VAL LOSS. Same reason. Loss and macro F1
   disagree precisely on the frames that matter here.

3. TRUNCATED BPTT. Episodes run 2000-3800 frames. Backpropagating through all
   of them is slow, memory-hungry and unnecessary: the hidden state is carried
   forward across chunks (detached), so the model still sees an unbroken
   sequence exactly as it will at deployment, while gradients only flow within
   a chunk. `--tbptt 0` disables chunking if you want the full-sequence
   gradient for comparison.

4. A SHUFFLED-FEATURE FLOOR. The HMM's emission could simply be switched off
   to measure how much the sensing was worth (it was +0.63 macro F1 over
   structure alone). A GRU has no such switch, so the equivalent control is to
   train an identical model on time-shuffled features: same labels, same
   architecture, same schedule, no usable observation. --shuffled-control
   reports that floor. Do not report a GRU number without it -- a recurrent
   net on a stereotyped task can learn the script and look excellent.
"""

from __future__ import annotations

import argparse
import os
import signal
import time

import h5py
import numpy as np

from teleop_orchestrator.sources import ReplaySource
from teleop_orchestrator.contracts import NULL_TARGET, Phase

import labels as label_loader
from metrics import classification_metrics

from .features import EpisodeFeatureBuilder, ARM_FEATURE_NAMES, candidate_feature_names
from .model import GRUIntentModel, SIDES, _require_torch

TRAIN_DEFAULTS = dict(
    epochs=60, lr=2e-3, weight_decay=1e-4, tbptt=256, grad_clip=1.0,
    target_loss_weight=1.0, patience=12, seed=0, phase_class_weight="inverse",
    checkpoint_path=None,   # written on every val improvement; see fit_model
)


class _GracefulStop:
    """Turns Ctrl-C into 'stop at the next safe point', not 'lose the run'.

    A first SIGINT sets a flag the training loop checks between chunks, so the
    optimiser step in flight completes and the model is left in a consistent
    state. A second SIGINT restores Python's default handler and aborts
    immediately -- without that escape hatch, a graceful path that hangs would
    leave no way out but killing the process.

    The handler is installed only for the duration of fit_model and always
    restored, so importing this module never changes the interpreter's
    behaviour for anything else. Signal handlers can only be set from the main
    thread; if that fails we degrade to no handler rather than refusing to
    train.
    """

    def __init__(self):
        self.requested = False
        self._previous = None
        self._installed = False

    def _handle(self, _signum, _frame):
        if self.requested:
            print("\n  second interrupt -- aborting now, last saved checkpoint is on disk")
            if self._previous is not None:
                signal.signal(signal.SIGINT, self._previous)
            raise KeyboardInterrupt
        self.requested = True
        print("\n  interrupt received -- finishing the current chunk, then stopping cleanly."
              "\n  (Ctrl-C again to abort immediately.)")

    def __enter__(self):
        try:
            self._previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        except (ValueError, OSError):
            pass                     # not the main thread; run without the handler
        return self

    def __exit__(self, *exc):
        if self._installed and self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        return False


def read_split(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_episode_arrays(hdf5_path: str, rich: bool = False) -> list[dict] | None:
    """One dict per (episode, side), or None if the episode has no real labels."""
    with h5py.File(hdf5_path, "r") as ep:
        if not label_loader.is_genuinely_labeled(ep):
            return None
        per_side = {s: label_loader.load_arm_labels(ep, s) for s in SIDES}

    src = ReplaySource(hdf5_path, load_images=False)
    frames = [src.frame_at(t) for t in range(len(src))]
    src.close()

    out = []
    for side in SIDES:
        phase_labels, target_labels = per_side[side]
        b = EpisodeFeatureBuilder(side, rich=rich)
        b.reset()
        arm, cand, mask, raw = [], [], [], []
        for fr in frames:
            a, c, m = b.step(fr)
            arm.append(a); cand.append(c); mask.append(m)
            # The mask BEFORE held.target_exclusion_mask is applied. Kept only
            # so an unreachable label can be attributed to the right cause:
            # "the candidate was not visible" and "we decided a held object
            # cannot be a target" are different problems with different fixes.
            raw.append(fr.candidate_mask.copy())
        out.append({
            "arm": np.stack(arm).astype(np.float32),
            "cand": np.stack(cand).astype(np.float32),
            "mask": np.stack(mask),
            "mask_raw": np.stack(raw),
            "phase": np.asarray(phase_labels, dtype=np.int64),
            "target": np.asarray(target_labels, dtype=np.int64),
            "episode_id": os.path.basename(os.path.dirname(hdf5_path)),
            "side": side,
        })
    return out


def gather(store_root: str, episode_ids: list[str], rich: bool = False) -> list[dict]:
    data, n = [], 0
    for eid in sorted(episode_ids):
        path = os.path.join(store_root, eid, "episode.hdf5")
        if not os.path.exists(path):
            continue
        arrays = build_episode_arrays(path, rich=rich)
        if arrays is None:
            continue
        n += 1
        data.extend(arrays)
    print(f"  {n} labeled episode(s) -> {len(data)} arm-sequences")
    report_unreachable_targets(data)
    return data


def report_unreachable_targets(data: list[dict]) -> float:
    """Counts frames whose labelled target the candidate mask excludes.

    Printed rather than silently handled because it is a disagreement between
    the hand labels and held.target_exclusion_mask, and only one of those can
    be right. The HMM ran into the same frames and clamped them away inside
    _score, so this number has never been visible before.
    """
    committed = unreachable = 0
    by_cause = {"held-object exclusion": 0, "candidate not visible": 0, "label out of range": 0}
    by_phase = np.zeros(Phase.N_CLASSES, dtype=int)
    worst = []
    for d in data:
        _idx, bad = target_index_masked(d["target"], d["mask"])
        committed += int((d["target"] != NULL_TARGET).sum())
        unreachable += int(bad.sum())
        if not bad.any():
            continue
        worst.append((int(bad.sum()), d["episode_id"], d["side"]))
        by_phase += np.bincount(d["phase"][bad], minlength=Phase.N_CLASSES)
        raw = d.get("mask_raw")
        rows = np.flatnonzero(bad)
        cols = d["target"][rows]
        for r, c in zip(rows, cols):
            if not (0 <= c < d["mask"].shape[1]):
                by_cause["label out of range"] += 1
            elif raw is not None and raw[r, c]:
                by_cause["held-object exclusion"] += 1     # visible, but we excluded it
            else:
                by_cause["candidate not visible"] += 1
    if not committed:
        return 0.0
    frac = unreachable / committed
    print(f"  target labels unreachable under the candidate mask: {unreachable}/{committed} "
          f"committed frames ({100 * frac:.2f}%)")
    if unreachable:
        worst.sort(reverse=True)
        print("    by cause:  " + "   ".join(f"{k} {v}" for k, v in by_cause.items() if v))
        print("    by phase:  " + "   ".join(
            f"{Phase.NAMES[i]} {by_phase[i]}" for i in range(Phase.N_CLASSES) if by_phase[i]))
        print("    worst sequences: " + ", ".join(f"{e}/{s}:{n}" for n, e, s in worst[:5]))
        if frac > 0.02:
            print("    OVER 2% -- excluded from the target loss. If 'held-object exclusion'"
                  " dominates and the phases are transport/place, the labeller marked the"
                  " CARRIED object as the target while held.target_exclusion_mask says a held"
                  " object cannot be one. Only one of those can be right, and it is a data"
                  " decision, not a modelling one.")
    return frac


# cross_entropy skips frames carrying this label. Used for frames whose target
# label is unreachable given the candidate mask -- see target_index_masked.
IGNORE_INDEX = -100


def _target_index(target_labels: np.ndarray, n_cand: int) -> np.ndarray:
    """NULL_TARGET -> the last column, matching the network's output layout."""
    return np.where(target_labels == NULL_TARGET, n_cand, target_labels).astype(np.int64)


def target_index_masked(target_labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (training index [T], unreachable flag [T]).

    A frame is UNREACHABLE when its labelled target is a candidate the mask
    excludes -- either not visible that frame, or an object-type candidate
    dropped from the pool because the arm is holding something
    (held.target_exclusion_mask). The label and the exclusion rule disagree on
    those frames, and no model can be right on them: the softmax assigns the
    labelled class exactly zero probability by construction.

    Left in, they do real damage. The gradient through masked_fill is zero at
    the masked position, so the model gets no push toward the labelled answer;
    what it does get is a push DOWN on every candidate that is valid, plus
    null. The frame therefore teaches "nothing here is correct", which is pure
    noise. Marking them IGNORE_INDEX drops them from the target loss instead.

    Worth knowing: the HMM hit exactly the same frames and hid them, because
    _score clamped the true-state probability at 1e-12 before taking the log.
    It was silently penalised on them for every checkpoint from v1 to v11. The
    count printed at load time is the first look anyone has had at how many
    there are -- if it is more than a fraction of a percent, the labels and the
    exclusion rule need reconciling, and that is a data fix, not a model one.
    """
    n_cand = mask.shape[1]
    idx = _target_index(target_labels, n_cand)
    committed = target_labels != NULL_TARGET
    reachable = np.ones(len(idx), dtype=bool)
    if committed.any():
        rows = np.flatnonzero(committed)
        cols = target_labels[rows]
        in_range = (cols >= 0) & (cols < n_cand)
        reachable[rows[~in_range]] = False
        ok = rows[in_range]
        reachable[ok] = mask[ok, cols[in_range]]
    unreachable = committed & ~reachable
    return np.where(unreachable, IGNORE_INDEX, idx).astype(np.int64), unreachable


def compute_norm_stats(data: list[dict]):
    """Mean/std per feature over the TRAINING split only.

    Constant features get std 1 rather than 0, so a channel that never varies
    contributes zero rather than infinity. Candidate statistics are pooled over
    valid candidates only -- masked rows are structural zeros and would drag
    the mean toward nothing.
    """
    arm = np.concatenate([d["arm"] for d in data], axis=0)
    a_mean, a_std = arm.mean(0), arm.std(0)
    valid = np.concatenate([d["cand"][d["mask"]] for d in data], axis=0)
    c_mean, c_std = (valid.mean(0), valid.std(0)) if valid.size else (0.0, 1.0)
    tidy = lambda s: np.where(np.asarray(s) < 1e-8, 1.0, s)  # noqa: E731
    return (a_mean.astype(np.float64), tidy(a_std).astype(np.float64),
            np.asarray(c_mean, dtype=np.float64), tidy(c_std).astype(np.float64))


def phase_class_weights(data: list[dict], mode: str):
    counts = np.zeros(Phase.N_CLASSES)
    for d in data:
        counts += np.bincount(d["phase"], minlength=Phase.N_CLASSES)
    counts = np.maximum(counts, 1.0)
    if mode == "none":
        w = np.ones(Phase.N_CLASSES)
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(counts)
    else:                                   # "inverse"
        w = 1.0 / counts
    return w / w.mean(), counts


def fit_model(model: GRUIntentModel, train_data: list[dict], val_data: list[dict],
              config: dict) -> dict:
    _require_torch()
    import torch
    import torch.nn.functional as F

    cfg = dict(TRAIN_DEFAULTS, **config)
    if cfg.get("num_threads"):
        # Worth tuning by hand: this network is small enough that torch's
        # default (one thread per core) can spend more time synchronising
        # across threads than computing. 4-8 is usually faster here than 24.
        torch.set_num_threads(int(cfg["num_threads"]))
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    model.norm_mean, model.norm_std, model.cand_mean, model.cand_std = compute_norm_stats(train_data)
    d_arm = train_data[0]["arm"].shape[1]
    d_cand = train_data[0]["cand"].shape[2]
    model.build(d_arm, d_cand)
    net = model.net

    w, counts = phase_class_weights(train_data, str(cfg["phase_class_weight"]))
    print("  phase frame counts: " + "  ".join(
        f"{Phase.NAMES[i]}={int(counts[i])}" for i in range(Phase.N_CLASSES)))
    print("  phase loss weights: " + "  ".join(
        f"{Phase.NAMES[i]}={w[i]:.2f}" for i in range(Phase.N_CLASSES)))
    weight = torch.as_tensor(w, dtype=torch.float32)

    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  {n_params} parameters over {len(train_data)} arm-sequences")

    # Normalised tensors are built once and reused every epoch. Rebuilding
    # them per epoch re-ran identical numpy arithmetic and identical host
    # copies 116 times a pass for no benefit; the arrays total ~10 MB, so
    # holding them costs nothing worth measuring.
    _cache: dict[int, tuple] = {}

    def tensors(d):
        key = id(d)
        cached = _cache.get(key)
        if cached is not None:
            return cached
        arm = (d["arm"] - model.norm_mean) / model.norm_std
        cand = (d["cand"] - model.cand_mean) / model.cand_std
        tgt_idx, _bad = target_index_masked(d["target"], d["mask"])
        out = (torch.as_tensor(arm[None], dtype=torch.float32),
               torch.as_tensor(cand[None], dtype=torch.float32),
               torch.as_tensor(d["mask"][None], dtype=torch.bool),
               torch.as_tensor(d["phase"][None], dtype=torch.long),
               torch.as_tensor(tgt_idx[None], dtype=torch.long))
        _cache[key] = out
        return out

    def evaluate(dataset):
        net.eval()
        true, pred, tgt_ok, tgt_n = [], [], 0, 0
        with torch.no_grad():
            for d in dataset:
                arm, cand, mask, ph, tg = tensors(d)
                pl, tl, _ = net(arm, cand, mask)
                pred.extend(pl[0].argmax(-1).tolist())
                true.extend(ph[0].tolist())
                # Scored on committed frames the mask can actually reach, so
                # this number is comparable with eval/score.py rather than
                # carrying a fixed unwinnable penalty.
                idx, bad = target_index_masked(d["target"], d["mask"])
                scorable = (d["target"] != NULL_TARGET) & ~bad
                if scorable.any():
                    tgt_ok += int((tl[0].argmax(-1).numpy()[scorable] == idx[scorable]).sum())
                    tgt_n += int(scorable.sum())
        m = classification_metrics(true, pred, Phase.N_CLASSES)
        m["target_accuracy"] = tgt_ok / max(tgt_n, 1)
        return m

    chunk = int(cfg["tbptt"])
    ckpt_path = cfg.get("checkpoint_path")
    history, best = [], {"macro_f1": -1.0, "epoch": -1, "state": None}
    order = list(range(len(train_data)))

    def snapshot(tag: str) -> None:
        """Writes the current weights to the output path.

        Called on every val improvement, so the best model so far is always on
        disk and neither a Ctrl-C nor a crash costs more than the epochs since
        the last improvement. The alternative -- saving only at the end -- makes
        every long run an all-or-nothing bet.
        """
        if not ckpt_path:
            return
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        model.save(ckpt_path)
        print(f"    saved {tag} -> {ckpt_path}")

    stop = _GracefulStop()
    interrupted = False
    with stop:
        for epoch in range(int(cfg["epochs"])):
            net.train()
            np.random.shuffle(order)
            tot_p, tot_t, nb, t0 = 0.0, 0.0, 0, time.time()
            for i in order:
                if stop.requested:
                    break
                d = train_data[i]
                arm, cand, mask, ph, tg = tensors(d)
                T = arm.shape[1]
                h = None
                for s in range(0, T, chunk if chunk > 0 else T):
                    e = min(s + (chunk if chunk > 0 else T), T)
                    pl, tl, h = net(arm[:, s:e], cand[:, s:e], mask[:, s:e], h)
                    # Detach between chunks: the hidden state still flows forward
                    # unbroken (as it must, to match deployment) but the gradient
                    # does not reach back into the previous chunk.
                    h = h.detach()
                    loss_p = F.cross_entropy(pl.reshape(-1, Phase.N_CLASSES),
                                             ph[:, s:e].reshape(-1), weight=weight)
                    # ignore_index drops frames whose labelled target the mask
                    # excludes; if a chunk is entirely such frames the result is
                    # NaN, so fall back to zero rather than poisoning the step.
                    loss_t = F.cross_entropy(tl.reshape(-1, tl.shape[-1]), tg[:, s:e].reshape(-1),
                                             ignore_index=IGNORE_INDEX)
                    if not torch.isfinite(loss_t):
                        loss_t = torch.zeros((), dtype=loss_p.dtype)
                    loss = loss_p + float(cfg["target_loss_weight"]) * loss_t
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["grad_clip"]))
                    opt.step()
                    tot_p += float(loss_p.item()); tot_t += float(loss_t.item()); nb += 1
                    if stop.requested:
                        break

            # Evaluated even on an interrupted epoch: a partial epoch can still be
            # the best one seen, and discarding it would throw away exactly the
            # work the user was watching when they decided to stop.
            m = evaluate(val_data) if val_data else None
            row = {"epoch": epoch, "phase_loss": tot_p / max(nb, 1),
                   "target_loss": tot_t / max(nb, 1), "secs": time.time() - t0,
                   "partial": stop.requested}
            if m:
                row.update({"val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                            "val_target_accuracy": m["target_accuracy"]})
                improved = m["macro_f1"] > best["macro_f1"]
                if improved:
                    best = {"macro_f1": m["macro_f1"], "epoch": epoch,
                            "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
            history.append(row)
            if m:
                print(f"  epoch {epoch:3d}  loss p{row['phase_loss']:.3f}/t{row['target_loss']:.3f}  "
                      f"val macroF1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}  "
                      f"target {m['target_accuracy']:.3f}  ({row['secs']:.0f}s)"
                      + ("  [partial epoch]" if stop.requested else ""))
                if improved:
                    snapshot(f"epoch {epoch} (val macro F1 {m['macro_f1']:.3f})")

            if stop.requested:
                interrupted = True
                print(f"  stopped by request at epoch {epoch}")
                break
            if m and epoch - best["epoch"] >= int(cfg["patience"]):
                print(f"  early stop: no val improvement for {cfg['patience']} epochs")
                break

    if best["state"] is not None:
        # Selection on val macro F1, not on the last epoch and not on val loss.
        # Applies to an interrupted run too: stopping early should give you the
        # best model the run found, not whatever it happened to hold when the
        # key was pressed.
        net.load_state_dict(best["state"])
        print(f"  restored epoch {best['epoch']} (val macro F1 {best['macro_f1']:.3f})")
    net.eval()
    if interrupted:
        # Recorded in the checkpoint via config, so a checkpoint that stopped
        # early is identifiable months later without consulting a shell log.
        model.config["interrupted_at_epoch"] = int(history[-1]["epoch"]) if history else -1
    return {"history": history, "best_epoch": best["epoch"], "best_val_macro_f1": best["macro_f1"],
            "n_params": n_params, "interrupted": interrupted}


def write_history(checkpoint_path: str, hist: dict) -> str:
    """Writes the per-epoch curve next to the checkpoint as CSV.

    Kept as a sibling file rather than inside the .pt so it can be read
    without torch, diffed between runs, and plotted by eval/plot_training.py.
    The curve is the only record of HOW a number was reached -- whether val
    climbed steadily or plateaued at epoch 3 and then wandered for thirty more
    is invisible in the final figure and changes what the figure means.
    """
    path = os.path.splitext(checkpoint_path)[0] + "_history.csv"
    rows = hist.get("history") or []
    if not rows:
        return path
    keys = sorted({k for r in rows for k in r})
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved training history to {path}")
    return path


def shuffle_features_in_time(data: list[dict], seed: int = 0) -> list[dict]:
    """Permutes each sequence's features along time, keeping labels in place.

    The control for "is the recurrent net reading the robot, or has it learned
    the task's script?". Labels, label order, architecture and schedule are all
    unchanged; only the observation-to-label correspondence is destroyed. What
    this model reaches is the floor any GRU gets on this dataset for free, and
    the real model has to beat it by a margin worth reporting.
    """
    rng = np.random.default_rng(seed)
    out = []
    for d in data:
        perm = rng.permutation(d["arm"].shape[0])
        e = dict(d)
        e["arm"] = d["arm"][perm]
        e["cand"] = d["cand"][perm]
        e["mask"] = d["mask"][perm]
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--train-split", default="data/splits/train.txt")
    ap.add_argument("--val-split", default="data/splits/val.txt")
    ap.add_argument("--out", default="checkpoints/gru/v1.pt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    ap.add_argument("--lr", type=float, default=TRAIN_DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=TRAIN_DEFAULTS["weight_decay"])
    ap.add_argument("--tbptt", type=int, default=TRAIN_DEFAULTS["tbptt"],
                    help="truncated-BPTT chunk length in frames; 0 = full sequence")
    ap.add_argument("--patience", type=int, default=TRAIN_DEFAULTS["patience"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="cap torch CPU threads; try 4-8, the default one-per-core often "
                         "costs more in synchronisation than it saves on a net this small")
    ap.add_argument("--phase-class-weight", default="inverse", choices=["inverse", "sqrt", "none"])
    ap.add_argument("--rich", action="store_true",
                    help="add candidate features the HMM never had; off by default so the "
                         "model-class comparison is not confounded by an input change")
    ap.add_argument("--memoryless", action="store_true",
                    help="ABLATION: same weights and parameter count, but the recurrence "
                         "restarts every frame. Isolates whether the GRU's advantage comes "
                         "from memory or merely from being a better per-frame classifier")
    ap.add_argument("--shuffled-control", action="store_true",
                    help="train on time-shuffled features to measure the script-only floor")
    args = ap.parse_args()

    cfg = dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout, rich=args.rich,
               memoryless=args.memoryless,
               epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               tbptt=args.tbptt, patience=args.patience, seed=args.seed,
               phase_class_weight=args.phase_class_weight, num_threads=args.num_threads,
               # Lets fit_model write the best model as it goes, so Ctrl-C or a
               # crash never costs more than the epochs since the last
               # improvement.
               checkpoint_path=args.out)

    print("Gathering train split...")
    train_data = gather(args.store_root, read_split(args.train_split), rich=args.rich)
    print("Gathering val split...")
    val_data = gather(args.store_root, read_split(args.val_split), rich=args.rich)
    if not train_data:
        raise SystemExit("No labeled training episodes found -- check --store-root and the split file")

    if args.shuffled_control:
        print("\nSHUFFLED CONTROL: features permuted in time, labels untouched.")
        train_data = shuffle_features_in_time(train_data, seed=args.seed)
        val_data = shuffle_features_in_time(val_data, seed=args.seed + 1)

    print(f"\narm features ({len(ARM_FEATURE_NAMES)}): {', '.join(ARM_FEATURE_NAMES)}")
    names = candidate_feature_names(args.rich)
    print(f"candidate features ({len(names)}): {', '.join(names)}\n")

    if args.memoryless:
        print("\nMEMORYLESS ABLATION: recurrence restarts every frame; parameter count "
              "unchanged.\n")
    model = GRUIntentModel(cfg)
    print("Training. Ctrl-C stops cleanly and keeps the best model so far; "
          "Ctrl-C twice aborts.\n")
    hist = model.fit(train_data, val_data, cfg)
    print(f"\nbest val macro F1 {hist['best_val_macro_f1']:.3f} at epoch {hist['best_epoch']} "
          f"({hist['n_params']} parameters)")
    if hist.get("interrupted"):
        print("Run was INTERRUPTED -- this checkpoint is the best epoch reached, not a "
              "converged model. Recorded in its config as interrupted_at_epoch.")
    if args.shuffled_control:
        print("This is the FLOOR, not a result: compare the real run's macro F1 against it.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"saved checkpoint to {args.out}")
    write_history(args.out, hist)


if __name__ == "__main__":
    main()
