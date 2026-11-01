from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .utils import box_ioa_np, box_iou_np

GREEN, YELLOW, RED = 0, 1, 2
LEVEL_NAMES = ["Green", "Yellow", "Red"]


def level(x, theta_low: float, theta_mid: float):
    x = np.asarray(x)
    return np.where(x < theta_low, GREEN, np.where(x < theta_mid, YELLOW, RED))


@dataclass
class SeatMap:
    views: dict[int, dict[str, np.ndarray]]

    @classmethod
    def load(cls, path: Path) -> "SeatMap":
        d = json.loads(Path(path).read_text())
        return cls({int(v): {s: np.asarray(b, float) for s, b in seats.items()}
                    for v, seats in d["views"].items()})

    def students(self) -> list[str]:
        return sorted({s for seats in self.views.values() for s in seats})

    def seat_costs(self, view: int, box: np.ndarray) -> tuple[list[str], np.ndarray]:
        seats = self.views.get(view, {})
        ids = list(seats)
        if not ids:
            return [], np.zeros(0)
        ioa = box_ioa_np(box[None], np.stack([seats[s] for s in ids]))[0]
        return ids, 1.0 - np.clip(ioa, 0, 1)


def best_and_margin(ids: list[str], costs: np.ndarray):
    if len(ids) == 0:
        return None, 1.0, 0.0
    order = np.argsort(costs)
    best = order[0]
    margin = float(costs[order[1]] - costs[best]) if len(order) > 1 else 1.0
    return ids[best], float(costs[best]), margin


class _Dets:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.asarray(cls, np.float32).reshape(-1)
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        self.xywh = np.concatenate([xy, wh], 1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


@dataclass
class TrackState:
    streak: int = 0
    student: str | None = None


class ViewTracker:
    def __init__(self, view: int, seat_map: SeatMap, cfg: dict, fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        tc, ec = cfg["tracker"], cfg["engine"]
        lost_frames = int(round(ec["lost_track_seconds"] * fps))
        args = SimpleNamespace(track_high_thresh=tc["track_high_thresh"], track_low_thresh=tc["track_low_thresh"],
                               new_track_thresh=tc["new_track_thresh"], track_buffer=lost_frames,
                               match_thresh=tc["match_thresh"], fuse_score=tc["fuse_score"])
        self.bt = BYTETracker(args, frame_rate=30)
        self.view, self.seat_map = view, seat_map
        self.tau = max(1, int(round(ec["persistence_tau"] * fps)))
        self.tau_conf = ec["track_conf"]
        self.tracks: dict[int, TrackState] = {}

    def update(self, xyxy, conf, cls):
        out = self.bt.update(_Dets(xyxy, conf, cls)) if len(conf) else self.bt.update(_Dets(np.zeros((0, 4)), [], []))
        alive = set()
        res = []
        for row in out:
            x1, y1, x2, y2, tid, score = row[:6]
            tid = int(tid)
            alive.add(tid)
            st = self.tracks.setdefault(tid, TrackState())
            st.streak = st.streak + 1 if score >= self.tau_conf else 0
            box = np.array([x1, y1, x2, y2])
            ids, costs = self.seat_map.seat_costs(self.view, box)
            if st.student is None and st.streak >= self.tau:
                sid, c, _ = best_and_margin(ids, costs)
                if sid is not None and c < 1.0:
                    st.student = sid
            if st.student is not None and st.streak >= 1:
                if st.student in ids:
                    j = ids.index(st.student)
                    others = np.delete(costs, j)
                    margin = float(others.min() - costs[j]) if len(others) else 1.0
                else:
                    margin = 1.0
                res.append((st.student, box, float(score), margin, tid))
        lost = {t.track_id for t in self.bt.lost_stracks}
        for tid in list(self.tracks):
            if tid not in alive:
                if tid in lost:
                    self.tracks[tid].streak = 0
                else:
                    del self.tracks[tid]
        return res


def attribute(det_box: np.ndarray, students: list[tuple], seat_map: SeatMap, view: int):
    if not students:
        return None, 0.0
    ids, regions = [], []
    seats = seat_map.views.get(view, {})
    for sid, box, *_ in students:
        r = box.copy()
        if sid in seats:
            s = seats[sid]
            r = np.array([min(r[0], s[0]), min(r[1], s[1]), max(r[2], s[2]), max(r[3], s[3])])
        ids.append(sid)
        regions.append(r)
    costs = 1.0 - np.clip(box_ioa_np(det_box[None], np.stack(regions))[0], 0, 1)
    sid, c, margin = best_and_margin(ids, costs)
    if c >= 1.0:
        return None, 0.0
    return sid, margin


@dataclass
class StepEvidence:
    conf: dict[str, dict[int, float]] = field(default_factory=dict)
    tracked: set = field(default_factory=set)
    ambiguous: list = field(default_factory=list)
    attributions: list = field(default_factory=list)


def fuse_step(view_students: dict[int, list[tuple]], behavior: dict[int, tuple], seat_map: SeatMap,
              delta: float, sharing_idx: int | None, sharing_exempt: bool = True) -> StepEvidence:
    ev = StepEvidence()
    for v, studs in view_students.items():
        for s in studs:
            ev.tracked.add(s[0])
    for v, (xyxy, conf, cls) in behavior.items():
        studs = view_students.get(v, [])
        track_margin = {s[0]: s[3] for s in studs}
        for k in range(len(conf)):
            c = int(cls[k])
            sid, m_det = attribute(np.asarray(xyxy[k], float), studs, seat_map, v)
            if sid is None:
                ev.attributions.append((v, k, None, 0.0, True))
                continue
            margin = min(m_det, track_margin.get(sid, 1.0))
            exempt = sharing_exempt and sharing_idx is not None and c == sharing_idx
            if margin < delta and not exempt:
                ev.ambiguous.append((v, k, sid, c, float(conf[k]), margin))
                ev.attributions.append((v, k, sid, margin, True))
                continue
            ev.attributions.append((v, k, sid, margin, False))
            d = ev.conf.setdefault(sid, {})
            d[c] = max(d.get(c, 0.0), float(conf[k]))
    return ev


def compute_scores(evidence: np.ndarray, weights: np.ndarray, tau: int, phi: float = 1.5,
                   use_persistence: bool = True, t_end: int | None = None) -> dict:
    N, T, C = evidence.shape
    T = T if t_end is None else t_end
    ev = evidence[:, :T]
    det = ev > 0
    streak = np.zeros_like(ev, dtype=np.int32)
    run = np.zeros((N, C), np.int32)
    for t in range(T):
        run = np.where(det[:, t], run + 1, 0)
        streak[:, t] = run
    factor = np.where(use_persistence & (streak >= tau), phi, 1.0)
    e = (ev * weights[None, None, :] * factor).sum(-1)
    E = np.cumsum(e, 1)
    t_idx = np.arange(1, T + 1)[None, :]
    S = E / t_idx
    return {"e": e, "E": E, "S": S, "S_final": S[:, -1], "E_final": E[:, -1]}


class OnlineEngine:
    def __init__(self, students: list[str], weights: np.ndarray, tau: int, phi: float,
                 theta_low: float, theta_mid: float):
        self.students = list(students)
        self.w, self.tau, self.phi = np.asarray(weights, float), tau, phi
        self.th = (theta_low, theta_mid)
        C = len(self.w)
        self.E = {s: 0.0 for s in students}
        self.streak = {s: np.zeros(C, int) for s in students}
        self.tracked_steps = {s: 0 for s in students}
        self.t = 0

    def step(self, ev: StepEvidence) -> dict:
        self.t += 1
        out = {}
        for s in self.students:
            conf = np.zeros(len(self.w))
            for c, v in ev.conf.get(s, {}).items():
                conf[c] = v
            det = conf > 0
            self.streak[s] = np.where(det, self.streak[s] + 1, 0)
            factor = np.where(self.streak[s] >= self.tau, self.phi, 1.0)
            e = float((conf * self.w * factor).sum())
            self.E[s] += e
            self.tracked_steps[s] += s in ev.tracked
            S = self.E[s] / self.t
            out[s] = {"e": e, "E": self.E[s], "S": S, "alert": LEVEL_NAMES[int(level(S, *self.th))],
                      "observed_fraction": self.tracked_steps[s] / self.t}
        return out

    def finalize(self) -> dict:
        return {s: {"S": self.E[s] / self.t, "A": LEVEL_NAMES[int(level(self.E[s] / self.t, *self.th))],
                    "S_obs": self.E[s] / max(self.tracked_steps[s], 1),
                    "observed_fraction": self.tracked_steps[s] / self.t} for s in self.students}
