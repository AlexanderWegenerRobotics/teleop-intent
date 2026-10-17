"""
labeling/auto_label.py — rule-based intent labels from simulator ground truth.

Anchors (grasp close/open, lift, seated) come from logged commands and object
poses. Intent onsets (approach start, grasp start, insert start) are found by
searching backward from those anchors with the thresholds in the task yaml.
Output is a segment list per arm in auto_labels.json next to the raw logs;
labeling/label_tool.py loads it for review.

Usage:
    python labeling/auto_label.py <episode_folder> [--plot]
    python labeling/auto_label.py <store_root> --all [--plot]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema.yaml"
DEFAULT_TASK = HERE / "tasks" / "fmb_board_1.yaml"
OUT_NAME = "auto_labels.json"


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def load_schema(path=SCHEMA_PATH):
    s = load_yaml(path)
    s["phase_ids"] = {p["name"]: p["id"] for p in s["phases"]}
    s["phase_names"] = {p["id"]: p["name"] for p in s["phases"]}
    return s


def part_type(name: str) -> str:
    n = name.replace("spare_", "")
    parts = n.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else n


def quat_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class Episode:
    folder: Path
    ns: np.ndarray
    t: np.ndarray
    ee: np.ndarray
    gcmd: np.ndarray
    gwidth: np.ndarray
    names: list
    pos: np.ndarray
    quat: np.ndarray
    outcome: str = ""
    arm: str = "left"


def _meta_events(path: Path):
    if not path.exists():
        return {}
    m = pd.read_csv(path, sep=";")
    out = {}
    for _, r in m.iterrows():
        out.setdefault(r["event"], r)
    return out


def load_episode(folder: Path, arm: str = "left") -> Episode:
    folder = Path(folder)
    side = f"arm_{arm}"
    acols = ["wall_clock_ns", "gripper_width", "gripper_cmd", "grasp_cmd",
             "O_T_EE_world_12", "O_T_EE_world_13", "O_T_EE_world_14"]
    hdr = pd.read_csv(folder / f"{side}.csv", sep=";", nrows=0).columns
    a = pd.read_csv(folder / f"{side}.csv", sep=";", usecols=[c for c in acols if c in hdr])
    s = pd.read_csv(folder / "scene.csv", sep=";", low_memory=False)

    n_obj = 0
    while f"obj{n_obj}_x" in s.columns:
        n_obj += 1
    names = [str(s[f"obj{i}_name"].dropna().iloc[0]) if s[f"obj{i}_name"].notna().any() else ""
             for i in range(n_obj)]
    pos = np.stack([s[[f"obj{i}_x", f"obj{i}_y", f"obj{i}_z"]].to_numpy(float) for i in range(n_obj)], 1)
    quat = np.stack([s[[f"obj{i}_qw", f"obj{i}_qx", f"obj{i}_qy", f"obj{i}_qz"]].to_numpy(float)
                     for i in range(n_obj)], 1)
    colors = s[[f"obj{i}_color" for i in range(n_obj)]].fillna("").astype(str).to_numpy()

    ns = s["wall_clock_ns"].to_numpy(np.int64)
    end = len(ns)
    meta = _meta_events(folder / "scene_meta.csv")
    if "episode_start" in meta and "episode_end" in meta:
        dur = float(meta["episode_end"]["time_s"]) - float(meta["episode_start"]["time_s"])
        end = min(end, int(np.searchsorted(ns, ns[0] + int(dur * 1e9))))
    step = np.linalg.norm(np.diff(pos, axis=0), axis=2).max(1)
    jumps = np.flatnonzero(step > 0.05)
    tail = jumps[jumps > end - 200]
    if len(tail):
        end = min(end, int(tail[0]) + 1)
    ns, pos, quat, colors = ns[:end], pos[:end], quat[:end], colors[:end]

    probe = min(len(ns) - 1, 50)
    keep = [i for i in range(n_obj) if names[i] and colors[probe, i] != ""]
    names = [names[i] for i in keep]
    pos, quat = pos[:, keep], quat[:, keep]

    aw = a["wall_clock_ns"].to_numpy(np.int64)
    idx = np.clip(np.searchsorted(aw, ns), 0, len(aw) - 1)
    ee = a[["O_T_EE_world_12", "O_T_EE_world_13", "O_T_EE_world_14"]].to_numpy(float)[idx]
    gsrc = "grasp_cmd" if "grasp_cmd" in a.columns else "gripper_cmd"
    graw = a[gsrc].to_numpy(float)[idx]
    gcmd = (graw > 0.5).astype(np.int8) if gsrc == "grasp_cmd" else (graw < 0.02).astype(np.int8)
    gwidth = a["gripper_width"].to_numpy(float)[idx]

    am = _meta_events(folder / f"{side}_meta.csv")
    outcome = ""
    if "episode_end" in am:
        outcome = str(am["episode_end"].get("color_bin_mapping", "") or "")
    return Episode(folder, ns, (ns - ns[0]) / 1e9, ee, gcmd, gwidth, names, pos, quat, outcome, arm)


@dataclass
class Labels:
    T: int
    phase: np.ndarray
    obj: list
    goal: list
    events: list = field(default_factory=list)
    flags: list = field(default_factory=list)


class AutoLabeler:
    def __init__(self, task: dict, schema: dict):
        self.task = task
        self.schema = schema
        self.th = task["thresholds"]
        self.P = schema["phase_ids"]
        self.goals = {k: (v["type"], np.array(v["pos"], float)) for k, v in task["goals"].items()}
        self.sym = np.radians(task.get("yaw_symmetry_deg", 360))

    def _yaw_err(self, yaw):
        e = np.mod(yaw, self.sym)
        return np.minimum(e, self.sym - e)

    def seated_mask(self, ep: Episode):
        T, N = ep.pos.shape[:2]
        out = np.zeros((T, N), bool)
        yaw = quat_yaw(ep.quat)
        for i, n in enumerate(ep.names):
            typ = part_type(n)
            for g, (gt, gp) in self.goals.items():
                if gt != typ:
                    continue
                ok = (np.linalg.norm(ep.pos[:, i] - gp, axis=1) < self.th["seated_pos_m"]) & \
                     (self._yaw_err(yaw[:, i]) < np.radians(self.th["seated_yaw_deg"]))
                out[:, i] |= ok
        hold = max(1, int(self.th["seated_hold_s"] * self._rate(ep)))
        for i in range(N):
            out[:, i] = _min_run(out[:, i], hold)
        return out

    def nearest_goal(self, name, p, radius=None, xy=False):
        typ = part_type(name)
        best, bd = None, np.inf
        for g, (gt, gp) in self.goals.items():
            if gt != typ:
                continue
            d = np.linalg.norm((p - gp)[:2]) if xy else np.linalg.norm(p - gp)
            if d < bd:
                best, bd = g, d
        if radius is not None and bd > radius:
            return None, bd
        return best, bd

    @staticmethod
    def _rate(ep):
        return 1.0 / np.median(np.diff(ep.t))

    def label(self, ep: Episode) -> Labels:
        th, P = self.th, self.P
        T, N = ep.pos.shape[:2]
        rate = self._rate(ep)
        L = Labels(T, np.full(T, P["IDLE"], np.int8), ["null"] * T, ["null"] * T)
        seated = self.seated_mask(ep)
        speed = np.r_[0, np.linalg.norm(np.diff(ep.ee, axis=0), axis=1) * rate]
        speed = np.convolve(speed, np.ones(9) / 9, mode="same")

        g = ep.gcmd
        edges = np.flatnonzero(np.diff(g) != 0) + 1
        closes = [int(k) for k in edges if g[k] == 1]
        if g[0] == 1:
            closes = [0] + closes
        cycles = []
        for k in closes:
            ko = next((int(x) for x in edges if x > k and g[x] == 0), T - 1)
            cycles.append([k, ko])

        prev_end = 0
        for ci, (k, ko) in enumerate(cycles):
            seg = ep.pos[k:min(T, ko + 1)]
            disp = np.linalg.norm(seg - seg[0], axis=2).max(0) if len(seg) else np.zeros(N)
            moved = disp.max() > th["moved_m"]
            if moved:
                i = int(disp.argmax())
            else:
                i = int(np.argmin(np.linalg.norm(ep.pos[k, :, :2] - ep.ee[k, :2], axis=1)))
            name = ep.names[i]
            if k == 0 and not moved:
                prev_end = ko
                continue

            if moved:
                R = quat_to_mat(ep.quat[k, i])
                r_obj = R.T @ (ep.ee[k] - ep.pos[k, i])
                gp = ep.pos[:, i] + np.einsum("tij,j->ti", quat_to_mat(ep.quat[:, i]), r_obj)
            else:
                gp = np.repeat(ep.ee[k][None], T, 0)
            d = np.linalg.norm(ep.ee - gp, axis=1)

            far = np.flatnonzero(d[prev_end:k] > th["grasp_far_m"])
            jf = prev_end + int(far[-1]) if len(far) else prev_end
            inside = np.flatnonzero(d[jf:k + 1] < th["grasp_radius_m"])
            gs = jf + int(inside[0]) if len(inside) else k
            if (k - gs) / rate > th["long_grasp_s"]:
                L.flags.append({"t": float(ep.t[gs]), "kind": "long_grasp", "object": name})

            a0 = self._approach_start(d, speed, min(gs, jf), prev_end, rate)

            lift = None
            if moved:
                z0 = ep.pos[k, i, 2]
                up = np.flatnonzero(ep.pos[k:ko + 1, i, 2] > z0 + th["lift_m"])
                lift = k + int(up[0]) if len(up) else None

            settle_end = min(T - 1, ko + int(th["settle_s"] * rate))
            rel = ep.pos[settle_end, i]
            goal, _ = self.nearest_goal(name, rel, radius=th["goal_capture_xy_m"], xy=True)

            near_at_close, dclose = self.nearest_goal(name, ep.pos[k, i], radius=th["insert_radius_m"])
            correcting = near_at_close is not None and not seated[k, i]
            if correcting and goal is None:
                goal = near_at_close

            L.events.append({"t": float(ep.t[k]), "type": "grasp_close", "object": name})
            if ko < T - 1:
                L.events.append({"t": float(ep.t[ko]), "type": "grasp_open", "object": name})
            if not moved:
                L.events.append({"t": float(ep.t[k]), "type": "grasp_miss", "object": name})
                L.flags.append({"t": float(ep.t[k]), "kind": "grasp_miss", "object": name})
            if lift is not None:
                L.events.append({"t": float(ep.t[lift]), "type": "lift", "object": name})

            self._fill(L, a0, gs, "APPROACH", name, goal)
            if correcting:
                gpos = self.goals[goal][1]
                inside = np.linalg.norm(ep.pos[gs:ko, i] - gpos, axis=1) < th["insert_radius_m"]
                out_idx = np.flatnonzero(~inside)
                if len(out_idx):
                    x = gs + int(out_idx[0])
                    self._fill(L, gs, x, "CORRECT", name, goal)
                    for a_, b_ in _runs(~inside[x - gs:]):
                        self._fill(L, x + a_, x + b_, "TRANSPORT", name, goal)
                    for a_, b_ in _runs(inside[x - gs:]):
                        self._fill(L, x + a_, x + b_, "INSERT", name, goal)
                    L.flags.append({"t": float(ep.t[x]), "kind": "correct_lifted_out", "object": name})
                else:
                    self._fill(L, gs, ko, "CORRECT", name, goal)
                L.flags.append({"t": float(ep.t[gs]), "kind": "correct", "object": name})
            elif not moved or lift is None:
                self._fill(L, gs, ko, "GRASP", name, goal)
            else:
                self._fill(L, gs, lift, "GRASP", name, goal)
                ins = None
                if goal is not None:
                    gpos = self.goals[goal][1]
                    dg = np.linalg.norm(ep.pos[:, i] - gpos, axis=1)
                    j = ko
                    while j - 1 > lift and dg[j - 1] < th["insert_radius_m"]:
                        j -= 1
                    ins = j if dg[min(ko, T - 1)] < th["insert_radius_m"] else None
                if ins is not None:
                    self._fill(L, lift, ins, "TRANSPORT", name, goal)
                    self._fill(L, ins, ko, "INSERT", name, goal)
                else:
                    self._fill(L, lift, ko, "TRANSPORT", name, goal)
                    if goal is None:
                        L.flags.append({"t": float(ep.t[lift]), "kind": "no_goal", "object": name})

            if ko < T - 1:
                nxt = cycles[ci + 1][0] if ci + 1 < len(cycles) else T
                pe = ko
                lim = min(nxt, ko + int(th["place_max_s"] * rate), T)
                while pe < lim and np.linalg.norm(ep.ee[pe] - ep.ee[ko]) < th["place_retreat_m"]:
                    pe += 1
                self._fill(L, ko, pe, "PLACE", name, goal)
                prev_end = pe
            else:
                L.phase[ko] = L.phase[ko - 1] if ko > 0 else L.phase[ko]
                L.obj[ko], L.goal[ko] = name, goal or "null"
                prev_end = T

        self._pushes(ep, L, seated, cycles, rate)
        self._attempts(ep, L)
        self._seat_events(ep, L, seated)
        return L

    def _approach_start(self, d, speed, gs, lo, rate):
        th = self.th
        hyst = th["approach_hysteresis_m"]
        idle_n = int(th["idle_min_s"] * rate)
        j = gs
        best = gs
        mx = d[gs] if gs < len(d) else 0.0
        still = 0
        while j - 1 >= lo:
            j -= 1
            if speed[j] < th["idle_speed_mps"]:
                still += 1
                if still >= idle_n:
                    return min(gs, j + still)
            else:
                still = 0
            if d[j] > mx:
                mx, best = d[j], j
            elif d[j] < mx - hyst:
                return best
        return max(lo, best if mx - d[lo] >= hyst else lo)

    def _fill(self, L, a, b, phase, name, goal):
        if b <= a:
            return
        L.phase[a:b] = self.P[phase]
        for t in range(a, b):
            L.obj[t] = name
            L.goal[t] = goal or "null"

    def _pushes(self, ep, L, seated, cycles, rate):
        th = self.th
        T, N = ep.pos.shape[:2]
        busy = np.zeros(T, bool)
        settle = int(th["settle_s"] * rate)
        for k, ko in cycles:
            busy[k:min(T, ko + settle)] = True
        win = int(0.2 * rate)
        for i, name in enumerate(ep.names):
            v = np.r_[np.zeros(win), np.linalg.norm(ep.pos[win:, i] - ep.pos[:-win, i], axis=1)]
            near = np.linalg.norm(ep.ee - ep.pos[:, i], axis=1) < th["push_ee_radius_m"]
            m = (v > th["push_moved_m"]) & near & ~busy & ~seated[:, i]
            for a, b in _runs(m):
                a = max(0, a - win)
                goal, _ = self.nearest_goal(name, ep.pos[a, i], radius=th["insert_radius_m"])
                if goal is None:
                    continue
                free = L.phase[a:b] == self.P["IDLE"]
                idx = np.arange(a, b)[free]
                for t in idx:
                    L.phase[t] = self.P["CORRECT"]
                    L.obj[t], L.goal[t] = name, goal
                L.events.append({"t": float(ep.t[a]), "type": "push", "object": name})
                L.flags.append({"t": float(ep.t[a]), "kind": "push", "object": name})

    def _attempts(self, ep, L):
        th = self.th
        rise = th.get("retract_m", 0.01)
        P = self.P
        active = (L.phase == P["INSERT"]) | (L.phase == P["CORRECT"])
        count = {}
        for a, b in _runs(active):
            name, goal = L.obj[a], L.goal[a]
            if name not in ep.names:
                continue
            i = ep.names.index(name)
            z = ep.pos[a:b, i, 2]
            key = (name, goal)
            count[key] = count.get(key, 0) + 1
            L.events.append({"t": float(ep.t[a]), "type": "insert_attempt", "object": name, "goal": goal,
                             "attempt": count[key]})
            zmin, jmin = z[0], 0
            j = 1
            while j < len(z):
                if z[j] < zmin:
                    zmin, jmin = z[j], j
                if z[j] - zmin > rise:
                    k = j
                    zmax, jmax = z[j], j
                    while k + 1 < len(z) and z[k + 1] >= zmax - 0.002:
                        k += 1
                        if z[k] > zmax:
                            zmax, jmax = z[k], k
                    js = j
                    while js - 1 > jmin and z[js - 1] - zmin > 0.002:
                        js -= 1
                    ts, te = ep.t[a + js], ep.t[a + jmax]
                    L.events.append({"t": float(ts), "t_end": float(te), "type": "retract", "object": name,
                                     "goal": goal, "rise_m": round(float(zmax - zmin), 4)})
                    L.flags.append({"t": float(ts), "kind": "retract", "object": name})
                    count[key] += 1
                    L.events.append({"t": float(te), "type": "insert_attempt", "object": name, "goal": goal,
                                     "attempt": count[key]})
                    zmin, jmin = zmax, jmax
                    j = jmax + 1
                    continue
                j += 1

    def _seat_events(self, ep, L, seated):
        for i, name in enumerate(ep.names):
            s = seated[:, i].astype(np.int8)
            for k in np.flatnonzero(np.diff(s) == 1) + 1:
                L.events.append({"t": float(ep.t[k]), "type": "seated", "object": name})
            for k in np.flatnonzero(np.diff(s) == -1) + 1:
                L.events.append({"t": float(ep.t[k]), "type": "unseated", "object": name})


def _runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _min_run(mask, n):
    out = np.zeros_like(mask)
    for a, b in _runs(mask):
        if b - a >= n:
            out[a:b] = True
    return out


def to_segments(ep: Episode, L: Labels, schema):
    segs = []
    T = L.T
    a = 0
    for t in range(1, T + 1):
        if t == T or L.phase[t] != L.phase[a] or L.obj[t] != L.obj[a] or L.goal[t] != L.goal[a]:
            segs.append({"start_ns": int(ep.ns[a]), "end_ns": int(ep.ns[t - 1]),
                         "phase": schema["phase_names"][int(L.phase[a])], "object": L.obj[a], "goal": L.goal[a]})
            a = t
    return segs


def _event_json(e, ns0, arm):
    out = {k: v for k, v in e.items() if k not in ("t", "t_end")}
    out["t_ns"] = ns0 + int(round(e["t"] * 1e9))
    if "t_end" in e:
        out["t_end_ns"] = ns0 + int(round(e["t_end"] * 1e9))
    out["arm"] = arm
    return out


def write_json(ep: Episode, L: Labels, task, schema, out: Path):
    ns0 = int(ep.ns[0])
    data = {
        "schema_version": schema["schema_version"],
        "task": task["task"],
        "source": "auto",
        "episode": ep.folder.name,
        "outcome": ep.outcome,
        "start_ns": ns0,
        "end_ns": int(ep.ns[-1]),
        "objects": ep.names,
        "goals": list(task["goals"].keys()),
        "thresholds": task["thresholds"],
        "arms": {
            ep.arm: {
                "segments": to_segments(ep, L, schema),
                "events": [_event_json(e, ns0, ep.arm) for e in sorted(L.events, key=lambda e: e["t"])],
                "flags": [{**{k: v for k, v in f.items() if k != "t"}, "t_ns": ns0 + int(round(f["t"] * 1e9))}
                          for f in sorted(L.flags, key=lambda f: f["t"])],
            }
        },
    }
    out.write_text(json.dumps(data, indent=1))
    return data


def plot_episode(ep: Episode, L: Labels, schema, task, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {p["id"]: p["color"] for p in schema["phases"]}
    t = ep.t
    fig, ax = plt.subplots(5, 1, figsize=(16, 9), sharex=True,
                           gridspec_kw={"height_ratios": [0.6, 1, 1, 1, 0.8]})
    for a, b in _segments_idx(L.phase):
        ax[0].axvspan(t[a], t[min(b, len(t) - 1)], color=colors[int(L.phase[a])], lw=0)
        if b - a > 0.04 * len(t) and L.obj[a] != "null":
            ax[0].text((t[a] + t[min(b, len(t) - 1)]) / 2, 0.5, L.obj[a].replace("spare_", "s_"),
                       ha="center", va="center", fontsize=7)
    ax[0].set_yticks([])
    ax[0].set_title(f"{ep.folder.name}  {ep.outcome}")
    ax[0].legend(handles=[Patch(color=colors[p["id"]], label=p["name"]) for p in schema["phases"]
                          if p["name"] in task["phases"]], ncol=8, fontsize=7, loc="upper right",
                 bbox_to_anchor=(1, 1.9))

    sp = np.r_[0, np.linalg.norm(np.diff(ep.ee, axis=0), axis=1) / np.diff(t)]
    ax[1].plot(t, np.convolve(sp, np.ones(9) / 9, mode="same"), lw=0.7)
    ax[1].set_ylabel("EE speed")
    ax[2].plot(t, ep.gwidth, lw=0.8, label="width")
    ax[2].plot(t, ep.gcmd * ep.gwidth.max(), lw=0.8, ls="--", label="close cmd")
    ax[2].set_ylabel("gripper")
    for i, n in enumerate(ep.names):
        gps = [gp for gt, gp in AutoLabeler(task, schema).goals.values() if gt == part_type(n)]
        dmin = np.min([np.linalg.norm(ep.pos[:, i] - gp, axis=1) for gp in gps], axis=0) if gps else np.full(len(t), 0.3)
        ax[3].plot(t, np.minimum(dmin, 0.3), lw=0.8, label=n)
    ax[3].axhline(task["thresholds"]["insert_radius_m"], color="k", lw=0.5, ls=":")
    ax[3].set_ylabel("part→goal [m]")
    ax[3].legend(fontsize=6, ncol=4, loc="upper right")
    for i, n in enumerate(ep.names):
        ax[4].plot(t, ep.pos[:, i, 2], lw=0.8)
    ax[4].set_ylabel("part z")
    ax[4].set_xlabel("t [s]")
    for e in L.events:
        if e["type"] in ("seated", "grasp_miss", "push"):
            c = {"seated": "g", "grasp_miss": "r", "push": "m"}[e["type"]]
            for a_ in ax[1:]:
                a_.axvline(e["t"], color=c, lw=0.6, alpha=0.6)
        elif e["type"] == "retract":
            for a_ in ax[1:]:
                a_.axvspan(e["t"], e["t_end"], color="orange", alpha=0.3, lw=0)
        elif e["type"] == "insert_attempt":
            ax[4].axvline(e["t"], color="b", lw=0.6, ls="--", alpha=0.7)
            ax[4].text(e["t"], ax[4].get_ylim()[1], f"#{e['attempt']}", fontsize=6, color="b", va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _segments_idx(phase):
    out, a = [], 0
    for t in range(1, len(phase) + 1):
        if t == len(phase) or phase[t] != phase[a]:
            out.append((a, t))
            a = t
    return out


def run_one(folder: Path, task, schema, plot=False, arm="left"):
    ep = load_episode(folder, arm)
    L = AutoLabeler(task, schema).label(ep)
    data = write_json(ep, L, task, schema, folder / OUT_NAME)
    if plot:
        plot_episode(ep, L, schema, task, folder / "auto_labels.png")
    counts = {}
    for s in data["arms"][arm]["segments"]:
        counts[s["phase"]] = counts.get(s["phase"], 0) + 1
    print(f"  {folder.name}: {ep.t[-1]:.1f}s  {ep.outcome or '-'}  "
          f"{counts}  flags={len(data['arms'][arm]['flags'])}")
    return ep, L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--task", default=str(DEFAULT_TASK))
    ap.add_argument("--arm", default="left")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--min-duration", type=float, default=10.0)
    args = ap.parse_args()
    task, schema = load_yaml(args.task), load_schema()
    root = Path(args.path)
    folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "scene.csv").exists()) if args.all else [root]
    for f in folders:
        try:
            ep = load_episode(f, args.arm)
            if ep.t[-1] < args.min_duration:
                print(f"  {f.name}: skipped ({ep.t[-1]:.1f}s)")
                continue
            run_one(f, task, schema, args.plot, args.arm)
        except Exception as e:
            print(f"  [error] {f.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
