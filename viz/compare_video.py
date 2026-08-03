"""Renders a side-by-side video: head camera, and what each model believes.

    python viz/compare_video.py --store-root /path/to/episodes --episode 008 \
        --start 300 --frames 600 \
        HMM=models.hmm.model.HMMIntentModel:checkpoints/hmm/v12.npz \
        GRU=models.gru.model.GRUIntentModel:checkpoints/gru/v2.pt \
        Transformer=models.transformer.model.TransformerIntentModel:checkpoints/transformer/matched.pt

WHY RIBBONS AND NOT THREE PROBABILITY PLOTS
-------------------------------------------
The three models agree on roughly five frames in six, so three overlaid belief
curves mostly show three lines on top of each other -- busy, and it buries the
finding. Drawn as segmentation ribbons against ground truth, the eye goes
straight to the thing that is actually true: every model identifies the phases,
and what separates them is where the boundaries land and how much they flicker.
That is the result, so it should be what the picture shows.

The playhead moves over a static timeline rather than the timeline scrolling
under a fixed playhead. Scrolling makes it impossible to see structure the
model has not reached yet; a static strip lets a viewer take in the whole
episode at once and watch the cursor cross it.

STRICTLY REPLAYED, STRICTLY CAUSAL. Every model is stepped frame by frame in
order through the same episode, exactly as it would run online -- no lookahead,
no smoothing, no re-running with hindsight. The ribbons are what the models
said at the time.

Writes MP4 if ffmpeg is available, otherwise a PNG sequence plus the ffmpeg
command to assemble it.

SHIPPING IT IN THE README. GitHub does not play a repo-relative .mp4 inline --
it renders as a download link -- so the README embeds a GIF and links the MP4.
Matplotlib's writer output is far larger than it needs to be for a mostly
static frame, so both are recompressed before committing:

    ffmpeg -i docs/intent_comparison.mp4 -c:v libx264 -crf 26 -preset slow \
        -pix_fmt yuv420p -movflags +faststart docs/_small.mp4

    ffmpeg -i docs/intent_comparison.mp4 \
        -vf "fps=8,scale=820:-1:flags=lanczos,palettegen=max_colors=48:stats_mode=diff" \
        docs/_palette.png
    ffmpeg -i docs/intent_comparison.mp4 -i docs/_palette.png \
        -lavfi "fps=8,scale=820:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" \
        docs/intent_comparison.gif

The two-pass palette matters: a GIF is limited to 256 colours, and the default
per-frame palette makes the camera image crawl with dithering noise between
frames, which both looks bad and inflates the file because almost every pixel
changes every frame. One palette computed over the whole clip keeps the static
background static.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from teleop_orchestrator.contracts import Phase  # noqa: E402
from teleop_orchestrator.sources import ReplaySource  # noqa: E402

import labels as label_loader  # noqa: E402

# One colour per phase, ordered idle -> place. Chosen to stay distinguishable
# in greyscale and for the two phases that matter most (grasp, place) to be the
# most saturated, since those are where the models differ.
PHASE_COLOURS = ["#e8edef", "#9ec6d8", "#1c7293", "#f0a848", "#c0392b"]
INK, MUTED = "#1f2a30", "#5f7079"


def find_camera(ep, requested: str | None) -> str | None:
    """Locates an image stream in the episode, since the dataset path differs
    between conversions. Returns the requested one if present, else the first
    image-shaped dataset found, else None -- and prints what is available so a
    wrong --camera is a one-line fix rather than a hunt through the file.
    """
    found = []

    def visit(name, obj):
        shape = getattr(obj, "shape", None)
        if shape is not None and len(shape) in (3, 4) and shape[0] > 1:
            # (T, H, W) or (T, H, W, C) with plausible image dimensions
            if len(shape) == 4 and shape[-1] in (1, 3, 4) and min(shape[1:3]) >= 32:
                found.append(name)
            elif len(shape) == 3 and min(shape[1:]) >= 32 and obj.dtype == "uint8":
                found.append(name)

    ep.visititems(visit)
    if requested:
        for name in found:
            if name.endswith(requested) or requested in name:
                return name
    if found:
        print(f"available image streams: {', '.join(found)}")
        print(f"using {found[0]} (override with --camera)")
        return found[0]
    print("no image stream found in this episode -- rendering ribbons only")
    return None


def clip_accuracy(preds: dict, truth: np.ndarray, lo: int, hi: int) -> dict:
    return {k: float((v[lo:hi] == truth[lo:hi]).mean()) for k, v in preds.items()}


def scan_clips(preds: dict, truth: np.ndarray, length: int, stride: int) -> None:
    """Prints per-model accuracy for every candidate window in the episode.

    A single 20-second clip is a tiny sample -- per-episode macro F1 varies by
    about 0.16 across this dataset, so an arbitrary window can easily reverse
    the aggregate ordering, as the first clip rendered here did. The point of
    this scan is NOT to find the window where the preferred model wins. It is
    to find a window whose accuracies sit near each model's episode-level
    figure, so the video illustrates the result rather than contradicting it.
    Pick a typical row, not the best one.
    """
    full = clip_accuracy(preds, truth, 0, len(truth))
    names = list(preds)
    print(f"\nwhole episode: " + "  ".join(f"{k} {full[k]:.3f}" for k in names))
    print(f"\n  {'start':>7}{'end':>7}" + "".join(f"{k[:11]:>12}" for k in names)
          + f"{'deviation':>11}")
    for lo in range(0, max(1, len(truth) - length), stride):
        hi = lo + length
        acc = clip_accuracy(preds, truth, lo, hi)
        dev = sum(abs(acc[k] - full[k]) for k in names) / len(names)
        print(f"  {lo:>7}{hi:>7}" + "".join(f"{acc[k]:>12.3f}" for k in names)
              + f"{dev:>11.3f}")
    print("\n  deviation = mean |clip - episode| across models; smallest is most "
          "representative")


def load_model(spec: str, checkpoint: str):
    module_path, class_name = spec.rsplit(".", 1)
    model = getattr(importlib.import_module(module_path), class_name)()
    model.load(checkpoint)
    return model


def predict(model, frames, side: str, warmup_from: int) -> np.ndarray:
    """Steps the model over every frame from the episode start, returning the
    argmax phase for the requested clip only.

    Playback always begins at frame 0 even when the clip starts later: these
    models carry state, and dropping them into the middle of an episode with an
    empty buffer would show worse predictions than the deployed system makes.
    """
    model.reset()
    out = []
    for i, f in enumerate(frames):
        result = model.step(f)
        if i >= warmup_from:
            out.append(result.arm(side).top_phase())
    return np.asarray(out, dtype=int)


def render(images, truth, preds: dict, out_dir: str, fps: int, title: str,
           mp4_path: str | None, subtitle: str = "") -> None:
    """Camera on the left, segmentation ribbons on the right.

    Row labels sit on the RIGHT of the ribbon panel rather than the left. On
    the left they overlapped the camera image, which no amount of padding fixes
    robustly -- an image axis sizes itself to its own aspect ratio, so the gap
    between the two panels changes with every episode's resolution. Putting the
    labels on the outside edge makes the collision impossible rather than
    unlikely.

    Each row also carries a live swatch and phase name for the current frame,
    so a viewer can read "what does each model think right now" without
    decoding colours against a legend. The legend sits above the ribbons, where
    it is read once before the eye moves down to the data.
    """
    names = list(preds)
    labels = ["ground truth"] + names
    clip_acc = clip_accuracy(preds, truth, 0, len(truth))
    ribbons = [truth] + [preds[k] for k in names]
    n_rows, T = len(ribbons), len(truth)
    cmap = ListedColormap(PHASE_COLOURS)

    fig = plt.figure(figsize=(15.0, 6.4))
    top, bottom = 0.80, 0.12
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.04,
                          left=0.01, right=0.775, top=top, bottom=bottom)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_rib = fig.add_subplot(gs[0, 1])

    ax_img.axis("off")
    im = ax_img.imshow(images[0] if images is not None else np.zeros((240, 320, 3), np.uint8))

    ax_rib.imshow(np.stack(ribbons), aspect="auto", cmap=cmap,
                  vmin=0, vmax=Phase.N_CLASSES - 1, interpolation="nearest",
                  extent=[0, T / fps, n_rows, 0])
    ax_rib.set_yticks([])
    ax_rib.set_xlabel("time (s)", fontsize=11)
    ax_rib.tick_params(colors=MUTED, labelsize=10)
    for sp in ax_rib.spines.values():
        sp.set_visible(False)
    for r in range(1, n_rows):
        ax_rib.axhline(r, color="white", lw=2.5)
    playhead = ax_rib.axvline(0, color=INK, lw=2.5)

    # Row label + live swatch + live phase name, all outside the axes on the
    # right where nothing can overlap them.
    # imshow with extent [.., n_rows, 0] puts row 0 at the TOP, so figure-space
    # y must count down from the axes top, not up from its bottom.
    # The live readout is LEFT-aligned even though it sits at the right edge:
    # right-aligning it would make the word jump sideways every time the phase
    # changes, which is exactly the moment a viewer is looking at it. Left
    # alignment means the x positions must instead budget for the longest
    # phase name ("transport"), which is what sets the 0.90 start.
    row_swatch, row_phase = [], []
    for i, name in enumerate(labels):
        y = top - (top - bottom) * (i + 0.5) / n_rows
        fig.text(0.790, y, name, fontsize=11, va="center", ha="left",
                 color=INK if i == 0 else MUTED, weight="bold" if i == 0 else "normal")
        row_swatch.append(fig.text(0.880, y, "\u25a0", fontsize=15, va="center",
                                   ha="left", color=PHASE_COLOURS[0]))
        row_phase.append(fig.text(0.900, y, "", fontsize=10.5, va="center",
                                  ha="left", color=INK))

    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOURS[i]) for i in range(Phase.N_CLASSES)]
    ax_rib.legend(handles, [Phase.NAMES[i] for i in range(Phase.N_CLASSES)],
                  loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=Phase.N_CLASSES,
                  frameon=False, fontsize=10.5, handlelength=1.4, columnspacing=1.4)

    fig.text(0.012, 0.965, title, fontsize=14, color=INK, ha="left", va="top")
    if subtitle:
        # Kept short and on the left half only: the phase legend sits above the
        # ribbon panel on the right, and a full-width subtitle runs into it.
        fig.text(0.012, 0.912, subtitle, fontsize=10.5, color=MUTED, ha="left", va="top")
    caption = fig.text(0.012, 0.03, "", fontsize=11, color=MUTED, ha="left")

    def update(t: int):
        if images is not None:
            im.set_data(images[t])
        playhead.set_xdata([t / fps, t / fps])
        for i, series in enumerate(ribbons):
            phase = int(series[t])
            row_swatch[i].set_color(PHASE_COLOURS[phase])
            row_phase[i].set_text(Phase.NAMES[phase])
            correct = i == 0 or phase == int(truth[t])
            row_phase[i].set_color(INK if correct else "#c0392b")
        agree = sum(int(preds[k][t] == truth[t]) for k in names)
        caption.set_text(f"t = {t / fps:5.1f} s      {agree}/{len(names)} models agree "
                         f"with the label      clip accuracy: "
                         + "   ".join(f"{k} {clip_acc[k]:.0%}" for k in names))
        return [im, playhead, caption] + row_swatch + row_phase

    writer = None
    if mp4_path:
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=4500)
        except Exception:
            writer = None

    if writer is not None:
        os.makedirs(os.path.dirname(mp4_path) or ".", exist_ok=True)
        with writer.saving(fig, mp4_path, dpi=110):
            for t in range(T):
                update(t)
                writer.grab_frame()
        print(f"wrote {mp4_path}")
    else:
        os.makedirs(out_dir, exist_ok=True)
        for t in range(T):
            update(t)
            fig.savefig(os.path.join(out_dir, f"frame_{t:05d}.png"), dpi=110)
        print(f"wrote {T} PNGs to {out_dir}\nassemble with:")
        print(f"  ffmpeg -framerate {fps} -i {out_dir}/frame_%05d.png "
              f"-c:v libx264 -pix_fmt yuv420p {mp4_path or 'out.mp4'}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", metavar="LABEL=module.Class:checkpoint")
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--start", type=int, default=0, help="first frame of the clip")
    ap.add_argument("--frames", type=int, default=600, help="clip length (~20 s at 30 fps)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", default=None,
                    help="image dataset name or suffix; auto-detected if omitted")
    ap.add_argument("--scan", action="store_true",
                    help="print per-model accuracy for every candidate clip and exit, "
                         "so a representative window can be chosen rather than an "
                         "arbitrary one")
    ap.add_argument("--scan-stride", type=int, default=150)
    ap.add_argument("--out", default="docs/intent_comparison.mp4")
    ap.add_argument("--frame-dir", default="docs/_video_frames")
    args = ap.parse_args()

    path = os.path.join(args.store_root, args.episode, "episode.hdf5")
    if not os.path.exists(path):
        raise SystemExit(f"no episode at {path}")

    import h5py
    with h5py.File(path, "r") as ep:
        phase_labels, _ = label_loader.load_arm_labels(ep, args.side)
        cam = find_camera(ep, args.camera)
        images = None
        if cam is not None and not args.scan:
            end = min(args.start + args.frames, ep[cam].shape[0])
            images = np.asarray(ep[cam][args.start:end])
            if images.ndim == 3:                      # greyscale -> RGB for imshow
                images = np.repeat(images[..., None], 3, axis=-1)
            print(f"loaded {len(images)} frames of {cam}")

    src = ReplaySource(path, load_images=False)
    total = len(src) if args.scan else min(args.start + args.frames, len(src))
    frames = [src.frame_at(t) for t in range(total)]
    src.close()

    start = 0 if args.scan else args.start
    truth = np.asarray(phase_labels[start:total], dtype=int)
    preds = {}
    for spec in args.models:
        label, rest = spec.split("=", 1)
        cls, checkpoint = rest.rsplit(":", 1)
        preds[label] = predict(load_model(cls, checkpoint), frames, args.side, start)
        print(f"  {label}: {(preds[label] == truth).mean():.1%} correct")

    if args.scan:
        scan_clips(preds, truth, args.frames, args.scan_stride)
        return

    title = f"Operator intent - episode {args.episode}, {args.side} arm"
    acc = clip_accuracy(preds, truth, 0, len(truth))
    subtitle = "causal - each model sees only the past, one frame at a time"
    render(images, truth, preds, args.frame_dir, args.fps, title, args.out, subtitle)


if __name__ == "__main__":
    main()
