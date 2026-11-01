from __future__ import annotations

import json
import math
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import resolve
from .utils import box_iou_np, log

SESSION_RE = re.compile(r"session_(\d+)")
VIEW_RE = re.compile(r"view_(\d+)")


def canon(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip()).lower()


def class_lookup(names: list[str]) -> dict[str, int]:
    return {canon(n): i for i, n in enumerate(names)}


def extract_frames(video: Path, out_dir: Path, sample_fps: float, ext: str = ".jpg") -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir.mkdir(parents=True, exist_ok=True)
    idx, written, next_t = 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if t + 1e-9 >= next_t:
            written += 1
            cv2.imwrite(str(out_dir / f"{written:06d}{ext}"), frame)
            next_t = written / sample_fps
        idx += 1
    cap.release()
    return written


def extract_all(cfg: dict) -> None:
    raw, frames = resolve(cfg, "raw_videos"), resolve(cfg, "frames")
    fps = cfg["data"]["sample_fps"]
    for video in sorted(raw.glob("session_*/view_*.*")):
        out = frames / video.parent.name / video.stem
        n = extract_frames(video, out, fps, cfg["data"]["image_ext"])
        log(f"{video} -> {n} frames")


def parse_voc(xml_path: Path) -> tuple[list[tuple[str, float, float, float, float]], tuple[int, int]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    objs = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        objs.append((o.find("name").text,
                     float(bb.find("xmin").text), float(bb.find("ymin").text),
                     float(bb.find("xmax").text), float(bb.find("ymax").text)))
    return objs, (w, h)


def parse_yolo_txt(txt: Path, names: list[str], wh: tuple[int, int]):
    w, h = wh
    objs = []
    for line in txt.read_text().split("\n"):
        p = line.split()
        if len(p) < 5:
            continue
        c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
        objs.append((names[c], (xc - bw / 2) * w, (yc - bh / 2) * h,
                     (xc + bw / 2) * w, (yc + bh / 2) * h))
    return objs


def read_annotation(ann_dir: Path, stem: str, img_path: Path):
    xml = ann_dir / f"{stem}.xml"
    if xml.exists():
        return parse_voc(xml)[0]
    txt = ann_dir / f"{stem}.txt"
    if txt.exists():
        names = (ann_dir / "classes.txt").read_text().split()
        img = cv2.imread(str(img_path))
        return parse_yolo_txt(txt, names, (img.shape[1], img.shape[0]))
    return None


def to_yolo_lines(objs, lookup: dict[str, int], w: int, h: int) -> list[str]:
    lines = []
    for name, x1, y1, x2, y2 in objs:
        c = lookup.get(canon(name))
        if c is None:
            continue
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def session_split(cfg: dict) -> dict[int, str]:
    out = {}
    for split, sessions in cfg["data"]["split"].items():
        for s in sessions:
            if s in out:
                raise ValueError(f"session {s} assigned to two splits")
            out[s] = split
    return out


def build_datasets(cfg: dict) -> dict[str, Path]:
    frames, anns = resolve(cfg, "frames"), resolve(cfg, "annotations")
    out_root = resolve(cfg, "datasets")
    split_of = session_split(cfg)
    tasks = {"status": cfg["classes"]["status"], "behavior": cfg["classes"]["behavior"]}
    lookups = {t: class_lookup(n) for t, n in tasks.items()}
    for t in tasks:
        shutil.rmtree(out_root / t, ignore_errors=True)
    n_img = Counter()
    manifest = defaultdict(list)
    for img in sorted(frames.glob("session_*/view_*/*" + cfg["data"]["image_ext"])):
        s = int(SESSION_RE.search(str(img)).group(1))
        v = int(VIEW_RE.search(str(img)).group(1))
        split = split_of.get(s)
        if split is None:
            continue
        ann_dir = anns / img.parent.parent.name / img.parent.name
        objs = read_annotation(ann_dir, img.stem, img)
        if objs is None:
            continue
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        name = f"s{s:02d}_v{v}_{img.stem}"
        for t in tasks:
            d = out_root / t
            (d / "images" / split).mkdir(parents=True, exist_ok=True)
            (d / "labels" / split).mkdir(parents=True, exist_ok=True)
            dst = d / "images" / split / f"{name}{img.suffix}"
            if not dst.exists():
                try:
                    dst.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dst)
            lines = to_yolo_lines(objs, lookups[t], w, h)
            (d / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        n_img[split] += 1
        manifest[split].append(name)
    paths = {}
    for t, names in tasks.items():
        paths[t] = write_data_yaml(out_root / t, names)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"images per split: {dict(n_img)}")
    return paths


def write_data_yaml(root: Path, names: list[str], splits=("train", "val", "test")) -> Path:
    data = {"path": str(root.resolve()), "names": {i: n for i, n in enumerate(names)}}
    for s in splits:
        if (root / "images" / s).exists():
            data[s] = f"images/{s}"
    p = root / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def read_labels(label_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for f in sorted(label_dir.glob("*.txt")):
        rows = [l.split() for l in f.read_text().split("\n") if l.strip()]
        out[f.stem] = np.array(rows, dtype=np.float32).reshape(-1, 5)
    return out


def class_counts(label_dir: Path, nc: int) -> np.ndarray:
    counts = np.zeros(nc, dtype=np.int64)
    for arr in read_labels(label_dir).values():
        for c in arr[:, 0].astype(int):
            counts[c] += 1
    return counts


def class_weights(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, float)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    if (counts == 0).any():
        log(f"classes without training annotations: {np.where(counts == 0)[0].tolist()} (weight set to 1.0)")
        w[counts == 0] = 1.0
    return w


def dataset_stats(cfg: dict) -> dict:
    root = resolve(cfg, "datasets")
    report = {}
    for task in ("status", "behavior"):
        names = cfg["classes"][task]
        rep = {}
        for split in ("train", "val", "test"):
            ld = root / task / "labels" / split
            if not ld.exists():
                continue
            labels = read_labels(ld)
            cnt = class_counts(ld, len(names))
            per_img = [len(a) for a in labels.values()]
            rep[split] = {"images": len(labels),
                          "counts": dict(zip(names, cnt.tolist())),
                          "total": int(cnt.sum()),
                          "mean_per_image": float(np.mean(per_img)) if per_img else 0,
                          "max_per_image": int(max(per_img)) if per_img else 0}
        tr = np.array([rep["train"]["counts"][n] for n in names])
        use_w = cfg["imbalance"]["class_weighting"][task]
        w = class_weights(tr) if use_w else np.ones(len(names))
        rep["class_weights"] = dict(zip(names, np.round(w, 4).tolist()))
        report[task] = rep
        (root / task / "class_weights.json").write_text(json.dumps(rep["class_weights"], indent=1))
    (root / "stats.json").write_text(json.dumps(report, indent=1))
    return report


def annotation_heatmap(label_dir: Path, out_png: Path, grid: int = 25) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = np.zeros((grid, grid))
    for arr in read_labels(label_dir).values():
        for _, xc, yc, _, _ in arr:
            H[min(int(yc * grid), grid - 1), min(int(xc * grid), grid - 1)] += 1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(H, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Annotations per cell")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return H


def _rotate_boxes(boxes_xyxy: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    out = []
    for x1, y1, x2, y2 in boxes_xyxy:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]).T
        r = M @ pts
        nx1, ny1 = r[0].min(), r[1].min()
        nx2, ny2 = r[0].max(), r[1].max()
        out.append([np.clip(nx1, 0, w), np.clip(ny1, 0, h), np.clip(nx2, 0, w), np.clip(ny2, 0, h)])
    return np.array(out, dtype=np.float32).reshape(-1, 4)


def augment_minority(cfg: dict, seed: int = 0) -> int:
    acfg = cfg["imbalance"]["targeted_augmentation"]
    if not acfg["enabled"]:
        return 0
    rng = random.Random(seed)
    names = cfg["classes"]["behavior"]
    minority = {names.index(n) for n in cfg["classes"]["minority"]}
    root = resolve(cfg, "datasets") / "behavior"
    img_dir, lab_dir = root / "images" / "train", root / "labels" / "train"
    created = 0
    for lab in sorted(lab_dir.glob("*.txt")):
        if "_aug" in lab.stem:
            continue
        arr = np.array([l.split() for l in lab.read_text().split("\n") if l.strip()],
                       dtype=np.float32).reshape(-1, 5)
        if not (set(arr[:, 0].astype(int)) & minority):
            continue
        img_path = next(img_dir.glob(lab.stem + ".*"))
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        cls = arr[:, 0]
        xyxy = np.stack([(arr[:, 1] - arr[:, 3] / 2) * w, (arr[:, 2] - arr[:, 4] / 2) * h,
                         (arr[:, 1] + arr[:, 3] / 2) * w, (arr[:, 2] + arr[:, 4] / 2) * h], 1)
        for k in range(acfg["copies_per_image"]):
            im, b = img.copy(), xyxy.copy()
            if acfg["hflip"] and rng.random() < 0.5:
                im = im[:, ::-1].copy()
                b[:, [0, 2]] = w - b[:, [2, 0]]
            f = 1 + rng.uniform(-acfg["brightness"], acfg["brightness"])
            im = np.clip(im.astype(np.float32) * f, 0, 255).astype(np.uint8)
            ang = rng.uniform(-acfg["rotation_deg"], acfg["rotation_deg"])
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            im = cv2.warpAffine(im, M, (w, h), borderValue=(114, 114, 114))
            b = _rotate_boxes(b, M, w, h)
            keep = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2)
            stem = f"{lab.stem}_aug{k}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), im)
            lines = [f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                     for c, (x1, y1, x2, y2) in zip(cls[keep], b[keep])]
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines))
            created += 1
    log(f"targeted augmentation: {created} images created")
    return created


def stratified_sample(manifest: list[str], frac: float = 0.10, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for n in manifest:
        groups[n.split("_")[0] + n.split("_")[1]].append(n)
    out = []
    for g in groups.values():
        k = max(1, round(len(g) * frac))
        out += rng.sample(g, min(k, len(g)))
    return sorted(out)


def fleiss_kappa(ratings: np.ndarray) -> float:
    n = ratings.sum(1)[0]
    N = ratings.shape[0]
    p_j = ratings.sum(0) / (N * n)
    P_i = (np.square(ratings).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), np.square(p_j).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def annotator_agreement(ann_dirs: list[Path], classes: list[str], iou_thr: float = 0.5) -> dict:
    R = len(ann_dirs)
    lookup = class_lookup(classes)
    cats = len(classes) + 1
    rows, ious = [], []
    stems = sorted({p.stem for p in ann_dirs[0].glob("*.xml")})
    for stem in stems:
        per = []
        for d in ann_dirs:
            objs, _ = parse_voc(d / f"{stem}.xml")
            per.append([(lookup.get(canon(o[0]), -1), np.array(o[1:])) for o in objs
                        if canon(o[0]) in lookup])
        clusters: list[list] = []
        for r, boxes in enumerate(per):
            for c, b in boxes:
                best, bi = None, None
                for ci, cl in enumerate(clusters):
                    if any(m[0] == r for m in cl):
                        continue
                    iou = max(box_iou_np(b[None], m[2][None])[0, 0] for m in cl)
                    if iou < iou_thr:
                        continue
                    key = (any(m[1] == c for m in cl), iou)
                    if best is None or key > best:
                        best, bi = key, ci
                if bi is None:
                    clusters.append([(r, c, b)])
                else:
                    clusters[bi].append((r, c, b))
        for cl in clusters:
            row = np.zeros(cats)
            for _, c, _ in cl:
                row[c] += 1
            row[-1] += R - len(cl)
            rows.append(row)
            for a in range(len(cl)):
                for b2 in range(a + 1, len(cl)):
                    ious.append(box_iou_np(cl[a][2][None], cl[b2][2][None])[0, 0])
    kappa = fleiss_kappa(np.array(rows)) if rows else float("nan")
    return {"fleiss_kappa": kappa, "mean_box_iou": float(np.mean(ious)) if ious else float("nan"),
            "clusters": len(rows), "images": len(stems)}
