"""Fits TransformerIntentModel from labeled episodes and saves a checkpoint.

    python -m models.transformer.train --store-root /path/to/avatar \
        --out checkpoints/transformer/matched.pt --preset matched
    python -m models.transformer.train --store-root ... --preset small \
        --out checkpoints/transformer/small.pt

Same splits, same is_genuinely_labeled filter and the same episode arrays as
models.gru.train -- the data pipeline is imported from it rather than copied,
so all three models are provably trained on identical inputs and any score
difference is attributable to the model.

THE CONTROLS ARE THE POINT. Report a transformer number only alongside:
  --shuffled-control   features permuted in time; the chance floor
  --window 1           attention restricted to the current frame, which is the
                       exact analogue of the GRU's --memoryless and isolates
                       what looking backwards is worth
Run both presets. `matched` holds capacity at the GRU's ~28k parameters;
`small` is a conventional size. Reporting both separates "attention does not
help on this task" from "it was starved" or "it overfitted 116 sequences".

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

from ..gru.features import ARM_FEATURE_NAMES, candidate_feature_names
from ..gru.train import (build_episode_arrays, gather, read_split, target_index_masked,
                         compute_norm_stats, phase_class_weights, shuffle_features_in_time,
                         write_history, report_unreachable_targets, _GracefulStop,
                         IGNORE_INDEX, TRAIN_DEFAULTS)
from .model import TransformerIntentModel, SIDES, PRESETS, _require_torch

def fit_model(model: TransformerIntentModel, train_data: list[dict], val_data: list[dict],
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
                # Same overlap scheme as training, for the same reason: scoring
                # a whole 3800-frame episode in one forward would both blow up
                # memory and evaluate a receptive field step() never provides.
                pl_parts, tl_parts = [], []
                span = chunk if chunk > 0 else arm.shape[1]
                for s0 in range(0, arm.shape[1], span):
                    e0 = min(s0 + span, arm.shape[1])
                    c0 = max(0, s0 - window + 1)
                    p_, t_, _ = net(arm[:, c0:e0], cand[:, c0:e0], mask[:, c0:e0])
                    pl_parts.append(p_[:, s0 - c0:e0 - c0])
                    tl_parts.append(t_[:, s0 - c0:e0 - c0])
                pl = torch.cat(pl_parts, 1)
                tl = torch.cat(tl_parts, 1)
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
    # Overlap by the receptive field, not the per-layer window: a stack of L
    # local-attention layers reaches L*(W-1)+1 frames back, and overlapping by
    # less would leave the first scored frame of each chunk with a truncated
    # context that step() never reproduces.
    window = net.receptive_field
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
                # CHUNKING WITH CONTEXT OVERLAP, not the GRU's plain slicing.
                #
                # A transformer carries no state between calls, so a chunk
                # boundary is not a detach point -- it is a hole in the
                # receptive field. Feeding [s, e) alone would let the frame at
                # s attend to nothing before it, while at deployment step()
                # gives that same frame a full W-frame buffer. The model would
                # be trained on one computation and served another, which is
                # the exact class of bug that cost the HMM its target head.
                #
                # So each chunk is EVALUATED on [s, e) but ENCODED over
                # [s - W + 1, e): every scored frame sees its complete window,
                # and the context frames contribute no loss. Cost is
                # (W + chunk)^2 attention per chunk rather than T^2 for the
                # whole episode -- 3800 frames at once would need ~230 MB of
                # attention matrix per layer before the backward pass.
                for s in range(0, T, chunk if chunk > 0 else T):
                    e = min(s + (chunk if chunk > 0 else T), T)
                    ctx = max(0, s - window + 1)
                    keep = slice(s - ctx, e - ctx)      # scored rows inside the encoded span
                    pl, tl, _ = net(arm[:, ctx:e], cand[:, ctx:e], mask[:, ctx:e])
                    pl, tl = pl[:, keep], tl[:, keep]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--train-split", default="data/splits/train.txt")
    ap.add_argument("--val-split", default="data/splits/val.txt")
    ap.add_argument("--out", default="checkpoints/gru/v1.pt")
    ap.add_argument("--preset", default="matched", choices=sorted(PRESETS),
                    help="matched = the GRU's parameter count; small = a conventional transformer")
    ap.add_argument("--window", type=int, default=512,
                    help="causal attention span in frames (~17s at 30Hz). 1 is the memoryless "
                         "control -- the analogue of the GRU's --memoryless")
    ap.add_argument("--dropout", type=float, default=None)
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
    ap.add_argument("--shuffled-control", action="store_true",
                    help="train on time-shuffled features to measure the script-only floor")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset], preset=args.preset, window=args.window, rich=args.rich,
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

    print(f"\npreset {args.preset}: {PRESETS[args.preset]}   window {args.window} frames")
    print(f"arm features ({len(ARM_FEATURE_NAMES)}): {', '.join(ARM_FEATURE_NAMES)}")
    names = candidate_feature_names(args.rich)
    print(f"candidate features ({len(names)}): {', '.join(names)}\n")

    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    if args.window <= 1:
        print("\nMEMORYLESS CONTROL: attention restricted to the current frame.\n")
    model = TransformerIntentModel(cfg)
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
