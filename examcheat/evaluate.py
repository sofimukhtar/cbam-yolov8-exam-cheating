from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import box_iou_np, log, save_json

IOU_THRS = np.linspace(0.5, 0.95, 10)


@dataclass
class ImageRecord:
    name: str
    pred_cls: np.ndarray
    pred_conf: np.ndarray
    tp: np.ndarray
    gt_cls: np.ndarray
    pred_box: np.ndarray = field(default=None, repr=False)
    gt_box: np.ndarray = field(default=None, repr=False)


def match_image(pred_box, pred_conf, pred_cls, gt_box, gt_cls, iou_thrs=IOU_THRS) -> np.ndarray:
    P, T = len(pred_box), len(iou_thrs)
    tp = np.zeros((P, T), dtype=bool)
    if P == 0 or len(gt_box) == 0:
        return tp
    order = np.argsort(-pred_conf, kind="stable")
    iou = box_iou_np(pred_box, gt_box)
    same = pred_cls[:, None] == gt_cls[None, :]
    iou = np.where(same, iou, 0.0)
    for t, thr in enumerate(iou_thrs):
        used = np.zeros(len(gt_box), dtype=bool)
        for p in order:
            cand = np.where(~used & (iou[p] >= thr))[0]
            if len(cand):
                g = cand[np.argmax(iou[p, cand])]
                used[g] = True
                tp[p, t] = True
    return tp


def make_record(name, pred_box, pred_conf, pred_cls, gt_box, gt_cls, keep_boxes=False) -> ImageRecord:
    pred_box = np.asarray(pred_box, dtype=np.float64).reshape(-1, 4)
    gt_box = np.asarray(gt_box, dtype=np.float64).reshape(-1, 4)
    pred_conf = np.asarray(pred_conf, dtype=np.float64).reshape(-1)
    pred_cls = np.asarray(pred_cls).astype(int).reshape(-1)
    gt_cls = np.asarray(gt_cls).astype(int).reshape(-1)
    tp = match_image(pred_box, pred_conf, pred_cls, gt_box, gt_cls)
    return ImageRecord(name, pred_cls, pred_conf, tp, gt_cls,
                       pred_box if keep_boxes else None, gt_box if keep_boxes else None)


def restrict(records: list[ImageRecord], classes: list[int]) -> list[ImageRecord]:
    keep = set(classes)
    out = []
    for r in records:
        mp = np.isin(r.pred_cls, list(keep))
        mg = np.isin(r.gt_cls, list(keep))
        out.append(ImageRecord(r.name, r.pred_cls[mp], r.pred_conf[mp], r.tp[mp], r.gt_cls[mg],
                               None if r.pred_box is None else r.pred_box[mp],
                               None if r.gt_box is None else r.gt_box[mg]))
    return out


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def compute_metrics(records: list[ImageRecord], classes: list[int], conf: float = 0.25,
                    names: list[str] | None = None) -> dict:
    pc = np.concatenate([r.pred_cls for r in records]) if records else np.zeros(0, int)
    pf = np.concatenate([r.pred_conf for r in records]) if records else np.zeros(0)
    tp = np.concatenate([r.tp for r in records]) if records else np.zeros((0, 10), bool)
    gc = np.concatenate([r.gt_cls for r in records]) if records else np.zeros(0, int)
    per = {}
    for c in classes:
        n_gt = int((gc == c).sum())
        m = pc == c
        conf_c, tp_c = pf[m], tp[m]
        order = np.argsort(-conf_c, kind="stable")
        conf_c, tp_c = conf_c[order], tp_c[order]
        aps = np.zeros(10)
        if n_gt and len(conf_c):
            ctp = np.cumsum(tp_c, 0)
            cfp = np.cumsum(~tp_c, 0)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            aps = np.array([compute_ap(rec[:, t], prec[:, t]) for t in range(10)])
        sel = conf_c >= conf
        TP = int(tp_c[sel, 0].sum())
        FP = int(sel.sum() - TP)
        FN = n_gt - TP
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        key = names[c] if names else c
        per[key] = {"ap50": aps[0], "ap50_95": aps.mean(), "precision": P, "recall": R, "f1": F1,
                    "tp": TP, "fp": FP, "fn": FN, "support": n_gt}
    vals = [v for v in per.values() if v["support"] > 0] or list(per.values())
    n_gt_all = np.array([v["support"] for v in vals])
    f1s = np.array([v["f1"] for v in vals])
    mp = float(np.mean([v["precision"] for v in vals]))
    mr = float(np.mean([v["recall"] for v in vals]))
    overall = {
        "map50": float(np.mean([v["ap50"] for v in vals])),
        "map50_95": float(np.mean([v["ap50_95"] for v in vals])),
        "precision_macro": mp, "recall_macro": mr,
        "f1_macro": float(f1s.mean()),
        "f1_of_macro_pr": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "f1_weighted": float((n_gt_all * f1s).sum() / max(n_gt_all.sum(), 1)),
    }
    overall.update(image_level(records, classes, conf))
    return {"per_class": per, "overall": overall}


def image_level(records: list[ImageRecord], classes: list[int], conf: float = 0.25) -> dict:
    perfect = 0
    TP = TN = FP = FN = 0
    for r in records:
        sel = r.pred_conf >= conf
        n_tp = int(r.tp[sel, 0].sum())
        if n_tp == len(r.gt_cls) and n_tp == int(sel.sum()):
            perfect += 1
        det = set(r.pred_cls[sel].tolist())
        gt = set(r.gt_cls.tolist())
        for c in classes:
            g, d = c in gt, c in det
            TP += g and d
            TN += (not g) and (not d)
            FP += (not g) and d
            FN += g and (not d)
    n = max(len(records), 1)
    return {"pdr": perfect / n, "n_images": len(records), "n_perfect": perfect,
            "accuracy": (TP + TN) / max(TP + TN + FP + FN, 1),
            "far": FP / max(FP + TN, 1), "img_tp": TP, "img_tn": TN, "img_fp": FP, "img_fn": FN}


def binary_decision_accuracy(records: list[ImageRecord], cheat_cls: int, conf: float = 0.25) -> dict:
    TP = TN = FP = FN = 0
    for r in records:
        g = cheat_cls in set(r.gt_cls.tolist())
        d = cheat_cls in set(r.pred_cls[r.pred_conf >= conf].tolist())
        TP += g and d; TN += (not g) and (not d); FP += (not g) and d; FN += g and (not d)
    return {"accuracy": (TP + TN) / max(TP + TN + FP + FN, 1), "tp": TP, "tn": TN, "fp": FP, "fn": FN}


def confusion_matrix(records: list[ImageRecord], nc: int, conf: float = 0.25, iou: float = 0.5) -> np.ndarray:
    M = np.zeros((nc + 1, nc + 1), dtype=int)
    for r in records:
        sel = r.pred_conf >= conf
        pb, pc = r.pred_box[sel], r.pred_cls[sel]
        gb, gc = r.gt_box, r.gt_cls
        used_p = np.zeros(len(pb), bool)
        if len(pb) and len(gb):
            ious = box_iou_np(gb, pb)
            pairs = sorted(((ious[g, p], g, p) for g in range(len(gb)) for p in range(len(pb))
                            if ious[g, p] >= iou), reverse=True)
            used_g = np.zeros(len(gb), bool)
            for _, g, p in pairs:
                if used_g[g] or used_p[p]:
                    continue
                used_g[g] = used_p[p] = True
                M[gc[g], pc[p]] += 1
            for g in np.where(~used_g)[0]:
                M[gc[g], nc] += 1
        else:
            for g in range(len(gb)):
                M[gc[g], nc] += 1
        for p in np.where(~used_p)[0]:
            M[nc, pc[p]] += 1
    return M


def load_gt(label_file: Path, w: int, h: int):
    if not label_file.exists():
        return np.zeros((0, 4)), np.zeros(0, int)
    rows = [l.split() for l in label_file.read_text().split("\n") if l.strip()]
    a = np.array(rows, dtype=np.float64).reshape(-1, 5)
    xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                     (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h], 1)
    return xyxy, a[:, 0].astype(int)


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def predict_records(weights: str | Path, images: list[Path], cfg: dict, class_map: dict[int, int] | None = None,
                    keep_boxes: bool = True, batch: int = 16) -> list[ImageRecord]:
    from ultralytics import YOLO
    import examcheat.models
    model = YOLO(str(weights))
    ev = cfg["eval"]
    records = []
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        for res, p in zip(model.predict(chunk, conf=ev["map_conf"], iou=ev["nms_iou"],
                                        imgsz=cfg["model"]["imgsz"], verbose=False, max_det=300), chunk):
            h, w = res.orig_shape
            gb, gc = load_gt(label_path(Path(p)), w, h)
            if class_map is not None:
                keep = np.array([c in class_map for c in gc], bool)
                gb, gc = gb[keep], np.array([class_map[c] for c in gc[keep]], int)
            b = res.boxes
            records.append(make_record(Path(p).stem, b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                       b.cls.cpu().numpy(), gb, gc, keep_boxes))
    return records


def list_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts)


def save_records(records: list[ImageRecord], path: Path) -> None:
    np.savez_compressed(path, data=np.array([r.__dict__ for r in records], dtype=object))


def load_records(path: Path) -> list[ImageRecord]:
    return [ImageRecord(**d) for d in np.load(path, allow_pickle=True)["data"]]


def _summary(records, classes, conf):
    o = compute_metrics(records, classes, conf)["overall"]
    return np.array([o["map50"], o["map50_95"], o["f1_macro"]])


def bootstrap_ci(records, classes, B=1000, conf=0.25, seed=0, alpha=0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(records)
    stats = np.array([_summary([records[i] for i in rng.integers(0, n, n)], classes, conf) for _ in range(B)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2], axis=0)
    keys = ["map50", "map50_95", "f1_macro"]
    return {k: [float(lo[j]), float(hi[j])] for j, k in enumerate(keys)}


def paired_bootstrap(rec_a, rec_b, classes, B=1000, conf=0.25, seed=0) -> dict:
    assert [r.name for r in rec_a] == [r.name for r in rec_b], "records must cover the same images"
    rng = np.random.default_rng(seed)
    n = len(rec_a)
    d = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d.append(_summary([rec_b[i] for i in idx], classes, conf) - _summary([rec_a[i] for i in idx], classes, conf))
    d = np.array(d)
    keys = ["map50", "map50_95", "f1_macro"]
    out = {}
    for j, k in enumerate(keys):
        p = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
        out[k] = {"mean": float(d[:, j].mean()), "ci": np.quantile(d[:, j], [0.025, 0.975]).tolist(),
                  "p": float(max(p, 1 / B))}
    return out


def paired_ttest(a: list[float], b: list[float]) -> dict:
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = len(d)
    t, p = stats.ttest_rel(b, a)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"mean_diff": float(d.mean()), "ci": [float(d.mean() - half), float(d.mean() + half)],
            "t": float(t), "p": float(p), "a_mean": float(a.mean()), "a_std": float(a.std(ddof=1)),
            "b_mean": float(b.mean()), "b_std": float(b.std(ddof=1))}


def model_complexity(cfg: dict, nc: int, device: str = "cpu", runs: int = 200) -> dict:
    import io
    import time

    import torch
    from ultralytics.utils.torch_utils import get_flops

    from .config import clone
    from .models import ExamDetectionModel, count_params

    out = {}
    for tag, cbam in (("vanilla", False), ("proposed", True)):
        c = clone(cfg)
        c["model"]["cbam"]["enabled"] = cbam
        m = ExamDetectionModel(c["model"]["cfg"], nc=nc, verbose=False, cbam=c["model"]["cbam"]).to(device).eval()
        buf = io.BytesIO()
        torch.save({k: v.half() for k, v in m.state_dict().items()}, buf)
        out[tag] = {"params": count_params(m), "gflops": float(get_flops(m, cfg["model"]["imgsz"])),
                    "fp16_mb": buf.tell() / 1e6, "cbam_params": m.cbam_params}
        if cbam:
            lat = []
            x = torch.zeros(1, 3, cfg["model"]["imgsz"], cfg["model"]["imgsz"], device=device)
            for i in cfg["model"]["cbam"]["layers"]:
                blk = m.model[i].cbam
                c_in = blk.ca.mlp[0].in_channels
                s = cfg["model"]["imgsz"] // {4: 8, 6: 16, 8: 32}.get(i, 8)
                xi = torch.randn(1, c_in, s, s, device=device)
                if device != "cpu":
                    xi = xi.half(); blk = blk.half()
                with torch.no_grad():
                    for _ in range(20):
                        blk(xi)
                    if device != "cpu":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(runs):
                        blk(xi)
                    if device != "cpu":
                        torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) / runs * 1e3)
            out[tag]["cbam_latency_ms"] = lat
            out[tag]["cbam_latency_total_ms"] = float(sum(lat))
    v, p = out["vanilla"], out["proposed"]
    out["delta"] = {"params": p["params"] - v["params"],
                    "params_pct": 100 * (p["params"] - v["params"]) / v["params"],
                    "gflops": p["gflops"] - v["gflops"],
                    "gflops_pct": 100 * (p["gflops"] - v["gflops"]) / v["gflops"],
                    "fp16_mb": p["fp16_mb"] - v["fp16_mb"]}
    return out
