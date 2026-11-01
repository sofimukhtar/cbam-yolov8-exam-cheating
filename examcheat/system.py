from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import clone, resolve
from .engine import (GREEN, LEVEL_NAMES, RED, YELLOW, SeatMap, ViewTracker, best_and_margin,
                     compute_scores, fuse_step, level)
from .utils import box_iou_np, log, save_json

LABEL_ID = {n.lower(): i for i, n in enumerate(LEVEL_NAMES)}


@dataclass
class SessionGT:
    labels: dict[str, int]
    events: list[tuple[str, str, float, float]]
    identities: list[tuple] | None = None
    behaviors: list[tuple] | None = None


def _rows(path: Path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_session_gt(cfg: dict, session: int) -> SessionGT:
    d = resolve(cfg, "system_gt") / f"session_{session:02d}"
    labels = {r["student"]: LABEL_ID[r["label"].strip().lower()] for r in _rows(d / "students.csv")}
    events = [(r["student"], r["class"], float(r["t_start"]), float(r["t_end"]))
              for r in (_rows(d / "events.csv") or [])]
    ids = _rows(d / "identities.csv")
    if ids is not None:
        ids = [(int(r["t"]), int(r["view"]), r["student"],
                np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")]),
                float(r.get("visibility") or 1.0)) for r in ids]
    beh = _rows(d / "behaviors.csv")
    if beh is not None:
        beh = [(int(r["t"]), int(r["view"]), r["student"], r["class"],
                np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])) for r in beh]
    return SessionGT(labels, events, ids, beh)


def frames_dir(cfg: dict, fps: float) -> Path:
    base = resolve(cfg, "frames")
    return base if fps == cfg["data"]["sample_fps"] else base.parent / f"{base.name}_{fps:g}fps"


def detect_session(cfg: dict, session: int, fps: float, w_status, w_behavior, cache: Path | None = None) -> dict:
    if cache and cache.exists():
        return np.load(cache, allow_pickle=True)["d"].item()
    from ultralytics import YOLO
    import examcheat.models
    ms, mb = YOLO(str(w_status)), YOLO(str(w_behavior))
    sess_dir = frames_dir(cfg, fps) / f"session_{session:02d}"
    views = sorted(int(p.name.split("_")[1]) for p in sess_dir.glob("view_*"))
    frames = {v: sorted((sess_dir / f"view_{v}").glob("*" + cfg["data"]["image_ext"])) for v in views}
    T = min(len(f) for f in frames.values())
    low = cfg["tracker"]["track_low_thresh"]
    out = {"T": T, "fps": fps, "views": views, "status": {}, "behavior": {}}
    kw = dict(iou=cfg["eval"]["nms_iou"], imgsz=cfg["model"]["imgsz"], verbose=False)
    for v in views:
        st, be = [], []
        for i in range(0, T, 32):
            chunk = [str(p) for p in frames[v][i:min(i + 32, T)]]
            for r in ms.predict(chunk, conf=low, **kw):
                b = r.boxes
                st.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                          b.cls.cpu().numpy()[:, None]], 1))
            for r in mb.predict(chunk, conf=cfg["eval"]["conf"], **kw):
                b = r.boxes
                be.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                          b.cls.cpu().numpy()[:, None]], 1))
        out["status"][v], out["behavior"][v] = st, be
        log(f"session {session} view {v}: {T} steps detected")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, d=np.array(out, dtype=object))
    return out


def associate_session(cfg: dict, dets: dict, seat_map: SeatMap, oracle: SessionGT | None = None,
                      fps: float | None = None, mode: str = "proposed") -> dict:
    if mode in ("no_ambiguity", "per_frame"):
        cfg = clone(cfg)
        cfg["engine"]["ambiguity_margin"] = 0.0
    if mode == "oracle" and oracle is None:
        raise ValueError("oracle mode requires ground-truth identities")
    if mode != "oracle":
        oracle = None
    fps = fps or dets.get("fps", 1)
    status_names = cfg["classes"]["status"]
    beh_names = cfg["classes"]["behavior"]
    cand = [status_names.index(n) for n in cfg["classes"]["status_candidate"]]
    sharing = beh_names.index("Sharing") if "Sharing" in beh_names else None
    ec = cfg["engine"]
    students = seat_map.students()
    sidx = {s: i for i, s in enumerate(students)}
    T, views = dets["T"], dets["views"]
    C = len(beh_names)
    evidence = np.zeros((len(students), T, C), np.float32)
    tracked = np.zeros((len(students), T), bool)
    trackers = {v: ViewTracker(v, seat_map, cfg, fps) for v in views}
    track_log, attr_log, n_amb = [], [], 0
    gt_ids = {}
    if oracle is not None:
        for t, v, s, box, _ in oracle.identities:
            gt_ids.setdefault((t, v), []).append((s, box))
    for t in range(1, T + 1):
        view_students, behavior = {}, {}
        for v in views:
            if mode == "per_frame":
                st = dets["status"][v][t - 1]
                st = st[np.isin(st[:, 5].astype(int), cand) & (st[:, 4] >= ec["track_conf"])]
                best = {}
                for row in st:
                    ids, costs = seat_map.seat_costs(v, row[:4])
                    sid, c, m = best_and_margin(ids, costs)
                    if sid is not None and c < 1.0 and (sid not in best or row[4] > best[sid][2]):
                        best[sid] = (sid, row[:4].astype(float), float(row[4]), m, -1)
                studs = list(best.values())
            elif oracle is None:
                st = dets["status"][v][t - 1]
                st = st[np.isin(st[:, 5].astype(int), cand)]
                studs = trackers[v].update(st[:, :4], st[:, 4], st[:, 5])
            else:
                studs = []
                for s, box in gt_ids.get((t, v), []):
                    ids, costs = seat_map.seat_costs(v, box)
                    m = 1.0
                    if s in ids and len(ids) > 1:
                        j = ids.index(s)
                        m = float(np.delete(costs, j).min() - costs[j])
                    studs.append((s, box, 1.0, m, -1))
            view_students[v] = studs
            for s, box, conf, margin, tid in studs:
                track_log.append((t, v, s, box, tid))
            be = dets["behavior"][v][t - 1]
            behavior[v] = (be[:, :4], be[:, 4], be[:, 5])
        ev = fuse_step(view_students, behavior, seat_map, ec["ambiguity_margin"], sharing, ec["sharing_exempt"])
        n_amb += len(ev.ambiguous)
        for v, k, s, margin, excluded in ev.attributions:
            attr_log.append((t, v, k, s, margin, excluded))
        for s in ev.tracked:
            if s in sidx:
                tracked[sidx[s], t - 1] = True
        for s, d in ev.conf.items():
            if s in sidx:
                for c, conf in d.items():
                    evidence[sidx[s], t - 1, c] = conf
    return {"students": students, "evidence": evidence, "tracked": tracked, "fps": fps,
            "track_log": track_log, "attr_log": attr_log, "n_ambiguous": n_amb}


def weights_vector(cfg: dict, override: dict | None = None) -> np.ndarray:
    w = dict(cfg["engine"]["severity_weights"])
    w.update(override or {})
    return np.array([w[n] for n in cfg["classes"]["behavior"]], float)


def score_session(assoc: dict, cfg: dict, weights=None, tau=None, use_persistence=True,
                  t_end=None, normalization="session") -> dict:
    fps = assoc["fps"]
    ec = cfg["engine"]
    tau = tau if tau is not None else max(1, int(round(ec["persistence_tau"] * fps)))
    w = weights if weights is not None else weights_vector(cfg)
    r = compute_scores(assoc["evidence"], w, tau, ec["persistence_factor"], use_persistence, t_end)
    T = r["S"].shape[1]
    Ti = assoc["tracked"][:, :T].sum(1)
    r["S_obs"] = r["E_final"] / np.maximum(Ti, 1)
    r["observed_fraction"] = Ti / T
    r["final"] = r["S_obs"] if normalization == "observed" else r["S_final"]
    r["T"], r["fps"] = T, fps
    return r


def prf_from_confusion(M: np.ndarray) -> dict:
    per = {}
    for k, name in enumerate(LEVEL_NAMES):
        tp = M[k, k]; fp = M[:, k].sum() - tp; fn = M[k, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[name] = {"support": int(M[k].sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn),
                     "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    macro = {k: float(np.mean([v[k] for v in per.values()])) for k in ("precision", "recall", "f1")}
    return {"per_class": per, "macro": macro}


def active_runs(active: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    runs = []
    idx = np.where(active)[0]
    if not len(idx):
        return runs
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            runs.append((start, prev)); start = i
        prev = i
    runs.append((start, prev))
    return runs


def evaluate_sessions(results: list[dict], cfg: dict, theta_low: float, theta_mid: float,
                      hours_per_student: list[float] | None = None) -> dict:
    sc = cfg["system_eval"]
    M = np.zeros((3, 3), int)
    lat, n_events, ev_tp, ev_fp, n_fa, hours = [], 0, 0, 0, 0, 0.0
    per_student = []
    for res in results:
        gt, sco, fps = res["gt"], res["scores"], res["fps"]
        S, final = sco["S"], sco["final"]
        T = S.shape[1]
        H = T / fps / 3600.0
        tol = int(round(sc["event_tolerance_s"] * fps))
        gap = int(round(sc["fa_merge_gap_s"] * fps))
        for i, s in enumerate(res["students"]):
            if s not in gt.labels:
                continue
            y = gt.labels[s]
            a = int(level(final[i], theta_low, theta_mid))
            M[y, a] += 1
            hours += H
            n_fa += (y == GREEN) and (a != GREEN)
            active = S[i] >= theta_low
            evs = [(c, ts, te) for st, c, ts, te in gt.events if st == s]
            windows = []
            for c, ts, te in evs:
                a0 = max(0, int(np.floor(ts * fps)) - tol)
                a1 = min(T - 1, int(np.ceil(te * fps)) + tol)
                if a0 > T - 1:
                    continue
                n_events += 1
                hit = active[a0:a1 + 1].any()
                ev_tp += bool(hit)
                windows.append((a0, a1))
            for r0, r1 in active_runs(active, gap):
                if not any(r0 <= w1 and r1 >= w0 for w0, w1 in windows):
                    ev_fp += 1
            if y != GREEN and evs:
                t0 = int(np.floor(min(e[1] for e in evs) * fps))
                after = np.where(active[t0:])[0] if t0 < T else []
                if len(after):
                    lat.append(after[0] / fps)
            per_student.append({"student": s, "label": LEVEL_NAMES[y], "alert": LEVEL_NAMES[a],
                                "S": float(final[i]), "observed_fraction": float(sco["observed_fraction"][i])})
    if hours_per_student is not None:
        hours = float(sum(hours_per_student))
    out = prf_from_confusion(M)
    TP = int(M[1:, 1:].sum()); FN = int(M[1:, 0].sum()); FP = int(M[0, 1:].sum()); TN = int(M[0, 0])
    bp = TP / (TP + FP) if TP + FP else 0.0
    br = TP / (TP + FN) if TP + FN else 0.0
    ev_fn = n_events - ev_tp
    ep = ev_tp / (ev_tp + ev_fp) if ev_tp + ev_fp else 0.0
    er = ev_tp / n_events if n_events else 0.0
    out.update({
        "confusion": M.tolist(),
        "binary": {"tp": TP, "fp": FP, "fn": FN, "tn": TN, "precision": bp, "recall": br,
                   "f1": 2 * bp * br / (bp + br) if bp + br else 0.0,
                   "specificity": TN / (TN + FP) if TN + FP else 0.0,
                   "accuracy": (TP + TN) / max(M.sum(), 1)},
        "student_hours": hours, "n_false_alarms": int(n_fa),
        "fa_per_student_hour": n_fa / hours if hours else 0.0,
        "latency_s": {"median": float(np.median(lat)) if lat else float("nan"),
                      "iqr": np.quantile(lat, [0.25, 0.75]).tolist() if lat else [float("nan")] * 2,
                      "n": len(lat)},
        "event": {"n_events": n_events, "tp": ev_tp, "fn": ev_fn, "fp": ev_fp, "precision": ep, "recall": er,
                  "f1": 2 * ev_tp / (2 * ev_tp + ev_fp + ev_fn) if n_events + ev_fp else 0.0,
                  "fp_per_student_hour": ev_fp / hours if hours else 0.0},
        "students": per_student,
    })
    return out


def select_thresholds(val_results: list[dict], cfg: dict) -> dict:
    ts = cfg["engine"]["threshold_search"]
    step = ts["step"]
    lows = np.round(np.arange(ts["low_range"][0], ts["low_range"][1] + step / 2, step), 4)
    mids = np.round(np.arange(ts["mid_range"][0], ts["mid_range"][1] + step / 2, step), 4)
    best, fallback = None, None
    for lo, mi in itertools.product(lows, mids):
        if lo >= mi:
            continue
        r = evaluate_sessions(val_results, cfg, lo, mi)
        key = (r["macro"]["f1"], -r["fa_per_student_hour"], -lo, -mi)
        if r["fa_per_student_hour"] <= ts["max_fa_per_student_hour"] + 1e-12:
            if best is None or key > best[0]:
                best = (key, lo, mi, r)
        fkey = (-r["fa_per_student_hour"], r["macro"]["f1"])
        if fallback is None or fkey > fallback[0]:
            fallback = (fkey, lo, mi, r)
    if best is None:
        log("no threshold pair satisfies the false-alarm constraint; using the lowest-FA pair")
        best = fallback
    _, lo, mi, r = best
    return {"theta_low": float(lo), "theta_mid": float(mi), "val_macro_f1": r["macro"]["f1"],
            "val_fa_per_student_hour": r["fa_per_student_hour"]}


def _frame_matches(gt_boxes, pr_boxes, thr=0.5):
    from scipy.optimize import linear_sum_assignment
    if not len(gt_boxes) or not len(pr_boxes):
        return []
    iou = box_iou_np(np.stack(gt_boxes), np.stack(pr_boxes))
    r, c = linear_sum_assignment(-iou)
    return [(i, j) for i, j in zip(r, c) if iou[i, j] >= thr]


def tracking_metrics(track_log, gt: SessionGT, seat_map: SeatMap, cfg: dict, frames: set | None = None) -> dict:
    from scipy.optimize import linear_sum_assignment
    gt_by, pr_by = {}, {}
    for t, v, s, box, _ in gt.identities:
        if frames is None or (t, v) in frames:
            gt_by.setdefault((t, v), []).append((s, box))
    for t, v, s, box, _ in track_log:
        if frames is None or (t, v) in frames:
            pr_by.setdefault((t, v), []).append((s, box))
    gids = sorted({s for l in gt_by.values() for s, _ in l})
    pids = sorted({s for l in pr_by.values() for s, _ in l})
    gi, pi = {s: k for k, s in enumerate(gids)}, {s: k for k, s in enumerate(pids)}
    cnt = np.zeros((len(gids), len(pids)))
    last, idsw = {}, 0
    n_gt = sum(len(l) for l in gt_by.values())
    n_pr = sum(len(l) for l in pr_by.values())
    for key in sorted(set(gt_by) | set(pr_by)):
        g, p = gt_by.get(key, []), pr_by.get(key, [])
        if g and p:
            iou = box_iou_np(np.stack([b for _, b in g]), np.stack([b for _, b in p]))
            for a in range(len(g)):
                for b in range(len(p)):
                    if iou[a, b] >= 0.5:
                        cnt[gi[g[a][0]], pi[p[b][0]]] += 1
        for a, b in _frame_matches([b for _, b in g], [b for _, b in p]):
            gs, ps = g[a][0], p[b][0]
            k2 = (gs, key[1])
            if k2 in last and last[k2] != ps:
                idsw += 1
            last[k2] = ps
    idtp = 0.0
    if cnt.size:
        r, c = linear_sum_assignment(-cnt)
        idtp = float(cnt[r, c].sum())
    idfp, idfn = n_pr - idtp, n_gt - idtp
    return {"idf1": 2 * idtp / max(2 * idtp + idfp + idfn, 1), "idsw": idsw,
            "frames": len(set(gt_by)), "gt_boxes": n_gt, "pred_boxes": n_pr}


def condition_frames(gt: SessionGT, seat_map: SeatMap, cfg: dict) -> dict[str, set]:
    sc = cfg["system_eval"]
    occ, mov, ovl = set(), set(), set()
    by = {}
    for t, v, s, box, vis in gt.identities:
        by.setdefault((t, v), []).append((s, box))
        if vis < sc["occlusion_visibility"]:
            occ.add((t, v))
        seat = seat_map.views.get(v, {}).get(s)
        if seat is not None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (seat[0] <= cx <= seat[2] and seat[1] <= cy <= seat[3]):
                mov.add((t, v))
    for key, l in by.items():
        if len(l) > 1:
            iou = box_iou_np(np.stack([b for _, b in l]), np.stack([b for _, b in l]))
            np.fill_diagonal(iou, 0)
            if iou.max() >= sc["overlap_iou"]:
                ovl.add(key)
    return {"occlusion": occ, "movement": mov, "overlap": ovl}


def identity_accuracy(assoc: dict, dets: dict, gt: SessionGT, cfg: dict, frames: set | None = None) -> dict:
    names = cfg["classes"]["behavior"]
    gt_by = {}
    for t, v, s, c, box in gt.behaviors:
        if c in names:
            gt_by.setdefault((t, v), []).append((s, names.index(c), box))
    correct = attributed = excluded = matched = 0
    for t, v, k, s, margin, exc in assoc["attr_log"]:
        if frames is not None and (t, v) not in frames:
            continue
        g = gt_by.get((t, v), [])
        if not g:
            continue
        d = dets["behavior"][v][t - 1][k]
        cands = [(gs, box) for gs, gc, box in g if gc == int(d[5])]
        if not cands:
            continue
        iou = box_iou_np(d[None, :4], np.stack([b for _, b in cands]))[0]
        j = int(np.argmax(iou))
        if iou[j] < 0.5:
            continue
        matched += 1
        if exc:
            excluded += 1
        else:
            attributed += 1
            correct += s == cands[j][0]
    return {"iaa": correct / max(attributed, 1), "ambiguous_fraction": excluded / max(matched, 1),
            "matched": matched}


def sensitivity(cfg: dict, val: dict, test: dict, ref: dict) -> list[dict]:
    sc = cfg["system_eval"]
    rows = []

    def run(sessions, **kw):
        return [{"students": s["assoc"]["students"], "gt": s["gt"], "fps": s["assoc"]["fps"],
                 "scores": score_session(s["assoc"], cfg, **kw)} for s in sessions]

    def row(factor, setting, res, th):
        r = evaluate_sessions(res, cfg, th["theta_low"], th["theta_mid"])
        rows.append({"factor": factor, "setting": setting, "macro_f1": r["macro"]["f1"],
                     "alert_recall": r["binary"]["recall"], "fa_per_student_hour": r["fa_per_student_hour"],
                     "median_latency_s": r["latency_s"]["median"],
                     "thresholds": [th["theta_low"], th["theta_mid"]]})
        return r

    base_fps = 1
    for fps in sc["sampling_rates"]:
        if fps not in test or fps not in val:
            log(f"sampling rate {fps} fps skipped (no detections at this rate)")
            continue
        th = ref if fps == base_fps else select_thresholds(run(val[fps]), cfg)
        row("Sampling rate", f"{fps} fps (tau={int(round(cfg['engine']['persistence_tau'] * fps))})",
            run(test[fps]), th)
    th = select_thresholds(run(val[base_fps], use_persistence=False), cfg)
    row("Persistence", "No persistence (phi=1.0)", run(test[base_fps], use_persistence=False), th)
    for tau in sc["persistence_taus"]:
        th = ref if tau == cfg["engine"]["persistence_tau"] else \
            select_thresholds(run(val[base_fps], tau=tau), cfg)
        row("Persistence", f"tau={tau}", run(test[base_fps], tau=tau), th)
    row("Severity weights", "Class-tiered (reference)", run(test[base_fps]), ref)
    names = cfg["classes"]["behavior"]
    uni = np.array([0.0 if n == "Normal" else 1.0 for n in names])
    th = select_thresholds(run(val[base_fps], weights=uni), cfg)
    row("Severity weights", "Uniform (w_c=1, c != Normal)", run(test[base_fps], weights=uni), th)
    rng = np.random.default_rng(0)
    w0 = weights_vector(cfg)
    ref_levels = [int(level(x, ref["theta_low"], ref["theta_mid"])) for res in run(test[base_fps])
                  for x in res["scores"]["final"]]
    pert, changed = [], []
    lo, hi = sc["perturbation_range"]
    for _ in range(sc["perturbation_draws"]):
        f = rng.uniform(lo, hi, len(w0))
        w = np.where(np.array(names) == "Normal", 0.0, w0 * f)
        res = run(test[base_fps], weights=w)
        r = evaluate_sessions(res, cfg, ref["theta_low"], ref["theta_mid"])
        lv = [int(level(x, ref["theta_low"], ref["theta_mid"])) for rr in res for x in rr["scores"]["final"]]
        changed.append(np.mean(np.array(lv) != np.array(ref_levels)))
        pert.append([r["macro"]["f1"], r["binary"]["recall"], r["fa_per_student_hour"], r["latency_s"]["median"]])
    pert = np.array(pert)
    rows.append({"factor": "Severity weights", "setting": "Random perturbation (+-20 %)",
                 "macro_f1": [pert[:, 0].mean(), pert[:, 0].std()],
                 "alert_recall": [pert[:, 1].mean(), pert[:, 1].std()],
                 "fa_per_student_hour": [pert[:, 2].mean(), pert[:, 2].std()],
                 "median_latency_s": [np.nanmean(pert[:, 3]), np.nanstd(pert[:, 3])],
                 "alert_level_changed": float(np.mean(changed))})
    for frac in sc["observation_fractions"]:
        res = []
        for s in test[base_fps]:
            T = s["assoc"]["evidence"].shape[1]
            res.append({"students": s["assoc"]["students"], "gt": s["gt"], "fps": s["assoc"]["fps"],
                        "scores": score_session(s["assoc"], cfg, t_end=max(1, int(round(frac * T))))})
        row("Observation duration", f"First {int(frac * 100)} %", res, ref)
    row("Normalization", "Session length (T)", run(test[base_fps]), ref)
    th = select_thresholds(run(val[base_fps], normalization="observed"), cfg)
    row("Normalization", "Tracked observation time (T_i)", run(test[base_fps], normalization="observed"), th)
    return rows
