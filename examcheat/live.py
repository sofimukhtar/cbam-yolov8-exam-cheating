from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .engine import LEVEL_NAMES, OnlineEngine, SeatMap, ViewTracker, fuse_step
from .system import weights_vector
from .utils import log


def run_live(cfg: dict, sources: dict[int, str], seat_map_path: Path, w_status, w_behavior,
             state_path: Path, max_steps: int | None = None) -> dict:
    from ultralytics import YOLO
    import examcheat.models
    fps = cfg["data"]["sample_fps"]
    ec = cfg["engine"]
    ms, mb = YOLO(str(w_status)), YOLO(str(w_behavior))
    seat_map = SeatMap.load(seat_map_path)
    caps = {v: cv2.VideoCapture(s) for v, s in sources.items()}
    src_fps = {v: c.get(cv2.CAP_PROP_FPS) or cfg["data"]["video_fps"] for v, c in caps.items()}
    trackers = {v: ViewTracker(v, seat_map, cfg, fps) for v in caps}
    names = cfg["classes"]["behavior"]
    status_names = cfg["classes"]["status"]
    cand = [status_names.index(n) for n in cfg["classes"]["status_candidate"]]
    sharing = names.index("Sharing")
    tau = max(1, int(round(ec["persistence_tau"] * fps)))
    eng = OnlineEngine(seat_map.students(), weights_vector(cfg), tau, ec["persistence_factor"],
                       ec["theta_low"], ec["theta_mid"])
    kw = dict(iou=cfg["eval"]["nms_iou"], imgsz=cfg["model"]["imgsz"], verbose=False)
    step = 0
    state_path.parent.mkdir(parents=True, exist_ok=True)
    while max_steps is None or step < max_steps:
        frames = {}
        for v, cap in caps.items():
            skip = max(1, int(round(src_fps[v] / fps)))
            ok, frame = cap.read()
            for _ in range(skip - 1):
                if not ok:
                    break
                ok, _f = cap.read()
            if not ok:
                frames = None
                break
            frames[v] = frame
        if frames is None:
            break
        t0 = time.perf_counter()
        view_students, behavior = {}, {}
        for v, frame in frames.items():
            r = ms.predict(frame, conf=cfg["tracker"]["track_low_thresh"], **kw)[0].boxes
            keep = np.isin(r.cls.cpu().numpy().astype(int), cand)
            view_students[v] = trackers[v].update(r.xyxy.cpu().numpy()[keep], r.conf.cpu().numpy()[keep],
                                                  r.cls.cpu().numpy()[keep])
            b = mb.predict(frame, conf=cfg["eval"]["conf"], **kw)[0].boxes
            behavior[v] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())
        ev = fuse_step(view_students, behavior, seat_map, ec["ambiguity_margin"], sharing, ec["sharing_exempt"])
        out = eng.step(ev)
        step += 1
        state = {"t": step, "time_s": step / fps, "latency_ms": (time.perf_counter() - t0) * 1e3,
                 "thresholds": [ec["theta_low"], ec["theta_mid"]],
                 "students": out, "ambiguous": len(ev.ambiguous)}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)
    final = eng.finalize()
    state_path.with_name("final.json").write_text(json.dumps(final, indent=1))
    log(f"live session finished after {step} steps")
    return final
