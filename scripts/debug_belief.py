"""Quick diagnostic: for one episode+frame, print gaze position, each slot's
projected position, and the belief assigned to it. Shows exactly whether
gaze-to-slot matching is correct or there's an index shift."""

import sys, os, h5py, numpy as np

def main():
    ep_path = sys.argv[1]          # path to episode.hdf5 or episode folder
    frame   = int(sys.argv[2]) if len(sys.argv) > 2 else None  # timestep index (optional)

    if os.path.isdir(ep_path):
        ep_path = os.path.join(ep_path, "episode.hdf5")

    with h5py.File(ep_path, "r") as f:
        intent = f["observations"]["intent"]

        slot_order = intent.attrs.get("slot_order", "original (not recomputed)")
        recomputed = intent.attrs.get("belief_recomputed", False)
        print(f"slot_order={slot_order}  recomputed={recomputed}")
        print()

        T = len(intent["gaze_px_x"])
        image_scale = float(f.attrs.get("image_scale", 0.25))

        # Pick a frame with valid gaze near the middle if not specified
        gaze_valid = intent["gaze_valid"][:] if "gaze_valid" in intent else np.ones(T, dtype=np.uint8)
        if frame is None:
            valid_frames = np.where(gaze_valid)[0]
            frame = int(valid_frames[len(valid_frames) // 2]) if len(valid_frames) else T // 2

        print(f"Frame t={frame} / T={T}  (gaze_valid={bool(gaze_valid[frame])})")

        gaze_x = float(intent["gaze_px_x"][frame])
        gaze_y = float(intent["gaze_px_y"][frame])
        px_norm = bool(intent.attrs.get("px_normalized", False))
        cam_w = 1280.0
        cam_h = 960.0
        if px_norm:
            gaze_u_native = gaze_x * cam_w
            gaze_v_native = gaze_y * cam_h
            gaze_u_stored = gaze_x * (cam_w * image_scale)
            gaze_v_stored = gaze_y * (cam_h * image_scale)
        else:
            gaze_u_native = gaze_x * 0.5
            gaze_v_native = gaze_y
            gaze_u_stored = gaze_x * 0.5 * image_scale
            gaze_v_stored = gaze_y * image_scale

        gaze_y = float(intent["gaze_px_y"][frame])
        if px_norm:
            gaze_v_native = gaze_y * cam_h
        else:
            gaze_v_native = gaze_y

        print(f"  gaze  u_native={gaze_u_native:.1f}  v_native={gaze_v_native:.1f}  "
              f"u_stored={gaze_u_stored:.1f}  v_stored={gaze_v_native * image_scale:.1f}")
        print()

        # Collect all slot keys
        belief_keys = sorted((k for k in intent.keys() if k.startswith("slot_belief_")), key=lambda k: int(k.rsplit("_", 1)[-1]))
        n_slots = int(intent["n_slots"][frame]) if "n_slots" in intent else 0

        print(f"  {'i':>3}  {'name':<20}  {'u_native':>9}  {'v_native':>9}  {'u_stored':>9}  {'belief':>8}  {'Δu':>7}  {'Δv':>7}  {'dist_2d':>8}")
        print(f"  {'---':>3}  {'----':<20}  {'---------':>9}  {'---------':>9}  {'---------':>9}  {'------':>8}  {'---':>7}  {'---':>7}  {'-------':>8}")

        for i in range(n_slots):
            name_key = f"slot_name_{i}"
            name = ""
            if name_key in intent:
                raw = intent[name_key][frame]
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
            else:
                name = f"slot_{i}"

            u_key = f"slot_px_u_{i}"
            v_key = f"slot_px_v_{i}"
            u_native = float(intent[u_key][frame]) if u_key in intent else -1.0
            v_native = float(intent[v_key][frame]) if v_key in intent else -1.0
            u_stored = u_native * image_scale if u_native >= 0 else -1.0
            v_stored = v_native * image_scale if v_native >= 0 else -1.0

            bk = belief_keys[i] if i < len(belief_keys) else None
            belief = float(intent[bk][frame]) if bk else 0.0

            if u_native >= 0:
                du = gaze_u_native - u_native
                dv = gaze_v_native - v_native
                d2d = (du**2 + dv**2) ** 0.5
                # Back-compute sigma from belief and 2D distance (ignoring temperature for now)
                # belief is post-temp-post-norm so we can't perfectly invert, but print d2d
            else:
                du = dv = d2d = float("nan")

            print(f"  {i:>3}  {name:<20}  {u_native:>9.1f}  {v_native:>9.1f}  {u_stored:>9.1f}  {belief:>8.4f}  {du:>7.1f}  {dv:>7.1f}  {d2d:>8.1f}")

        # Null belief
        null_key = belief_keys[n_slots] if n_slots < len(belief_keys) else None
        null_b = float(intent[null_key][frame]) if null_key else 0.0
        print(f"  {n_slots:>3}  {'[null]':<20}  {'':>9}  {'':>9}  {'':>9}  {null_b:>8.4f}")

        print()
        print("  Expected belief peak at slot with smallest dist_2d.")
        print("  If peak is elsewhere, sigma in recompute doesn't match C++ sigma.")
        print(f"  C++ default: gaze_sigma_px=60.0  |  recompute GAZE_SIGMA_PX check:")

        # Quick sigma sanity: find slot with max belief (excluding null) and compute implied sigma
        max_b_slot = max(range(n_slots), key=lambda i: (
            float(intent[belief_keys[i]][frame]) if i < len(belief_keys) else 0.0
        ), default=None)
        if max_b_slot is not None:
            bk = belief_keys[max_b_slot]
            b_max = float(intent[bk][frame])
            u_s = float(intent[f"slot_px_u_{max_b_slot}"][frame]) if f"slot_px_u_{max_b_slot}" in intent else -1
            v_s = float(intent[f"slot_px_v_{max_b_slot}"][frame]) if f"slot_px_v_{max_b_slot}" in intent else -1
            if u_s >= 0 and b_max > 0 and null_b > 0:
                # After temperature T=3: belief[i] ∝ raw[i]^(1/3), null ∝ null_prior^(1/3)=0.464
                # ratio = belief_max/null_b = raw_max^(1/3) / null_prior^(1/3)
                NULL_PRIOR = 0.1; T = 3.0
                ratio = b_max / null_b
                raw_max = NULL_PRIOR * ratio**T
                d2 = (gaze_u_native - u_s)**2 + (gaze_v_native - v_s)**2
                if raw_max > 0 and raw_max < 1 and d2 > 0:
                    import math
                    implied_sigma = math.sqrt(-d2 / (2 * math.log(raw_max)))
                    print(f"  slot {max_b_slot} ({intent[f'slot_name_{max_b_slot}'][frame] if f'slot_name_{max_b_slot}' in intent else '?'}) "
                          f"has max belief={b_max:.4f}, dist_2d={d2**0.5:.1f}px → implied σ≈{implied_sigma:.1f} native px "
                          f"({implied_sigma/4:.1f} stored px)")


if __name__ == "__main__":
    main()
