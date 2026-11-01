from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
from ultralytics.data.augment import Mosaic
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from .config import clone, resolve
from .models import ExamDetectionModel
from .utils import log, save_json


def image_sampling_weights(dataset, class_counts: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.maximum(class_counts, 1)
    w = np.full(len(dataset.labels), inv.min())
    for i, lab in enumerate(dataset.labels):
        cls = lab["cls"].reshape(-1).astype(int)
        if len(cls):
            w[i] = inv[cls].max()
    return w / w.sum()


def _find(obj, typ, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    out = [obj] if isinstance(obj, typ) else []
    for attr in ("transforms", "pre_transform"):
        sub = getattr(obj, attr, None)
        if sub is None:
            continue
        for s in (sub if isinstance(sub, (list, tuple)) else [sub]):
            out += _find(s, typ, seen)
    return out


def patch_mosaic(dataset, weights: np.ndarray) -> int:
    mosaics = _find(dataset.transforms, Mosaic)

    def make(m):
        def get_indexes(buffer=True):
            pool = list(m.dataset.buffer) if buffer and len(m.dataset.buffer) else list(range(len(m.dataset)))
            return random.choices(pool, weights=[weights[i] for i in pool], k=m.n - 1)
        return get_indexes

    for m in mosaics:
        m.get_indexes = make(m)
    return len(mosaics)


class ExamTrainer(DetectionTrainer):
    exam: dict = {}

    def get_model(self, cfg=None, weights=None, verbose=True):
        e = self.exam
        model = ExamDetectionModel(cfg or "yolov8m.yaml", nc=self.data["nc"], verbose=verbose and RANK == -1,
                                   cbam=e.get("cbam"), loss_cfg=e.get("loss"),
                                   class_weights=e.get("class_weights"))
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        e = self.exam
        if mode == "train" and e.get("oversample") and e.get("class_counts") is not None:
            w = image_sampling_weights(ds, np.asarray(e["class_counts"]))
            n = patch_mosaic(ds, w)
            log(f"mosaic oversampling enabled on {n} transform(s)")
        return ds


def task_class_info(cfg: dict, task: str):
    from .data import class_counts, class_weights
    root = resolve(cfg, "datasets") / task
    names = cfg["classes"][task]
    counts = class_counts(root / "labels" / "train", len(names))
    w = class_weights(counts) if cfg["imbalance"]["class_weighting"][task] else None
    return w, counts


def train_detector(cfg: dict, task: str, name: str, seed: int = 0, data_yaml: str | Path | None = None,
                   class_weights="auto") -> Path:
    data_yaml = Path(data_yaml or resolve(cfg, "datasets") / task / "data.yaml")
    if class_weights == "auto":
        w, counts = task_class_info(cfg, task) if task in cfg["classes"] else (None, None)
    else:
        w, counts = class_weights, None
        if cfg["imbalance"]["mosaic_oversampling"]:
            try:
                _, counts = task_class_info(cfg, task)
            except Exception:
                counts = None
    t = cfg["train"]
    overrides = dict(model=cfg["model"]["cfg"], pretrained=cfg["model"]["weights"] or False, data=str(data_yaml),
                     epochs=t["epochs"], batch=t["batch"], imgsz=cfg["model"]["imgsz"], lr0=t["lr0"],
                     optimizer=t["optimizer"], momentum=t["momentum"], weight_decay=t["weight_decay"],
                     box=cfg["loss"]["box_gain"], cls=cfg["loss"]["cls_gain"], dfl=cfg["loss"]["dfl_gain"],
                     conf=None, iou=cfg["eval"]["nms_iou"], seed=seed, deterministic=True,
                     amp=t["amp"], workers=t["workers"], device=t["device"],
                     project=str(resolve(cfg, "runs") / "train"), name=name, exist_ok=True, plots=True)
    overrides.update(t.get("extra", {}) or {})
    ExamTrainer.exam = {"cbam": cfg["model"]["cbam"], "loss": cfg["loss"],
                        "class_weights": None if w is None else list(map(float, w)),
                        "class_counts": None if counts is None else counts.tolist(),
                        "oversample": cfg["imbalance"]["mosaic_oversampling"]}
    log(f"training {name}: task={task} seed={seed} cbam={cfg['model']['cbam']['enabled']} "
        f"loss={cfg['loss']['cls_type']} weighted={w is not None}")
    trainer = ExamTrainer(overrides=overrides)
    trainer.train()
    save_json({"overrides": overrides, "exam": ExamTrainer.exam}, Path(trainer.save_dir) / "exam_config.json")
    return Path(trainer.best)


def _cfg(base, cbam, cls_type, weighted, cls_gain, box, dfl, task):
    c = clone(base)
    c["model"]["cbam"]["enabled"] = cbam
    c["loss"].update(cls_type=cls_type, cls_gain=cls_gain, box_gain=box, dfl_gain=dfl)
    c["imbalance"]["class_weighting"][task] = weighted
    return c


def ablation_configs(base: dict, task: str) -> dict[str, dict]:
    if task == "status":
        return {
            "C_vanilla_yolov8m": _cfg(base, False, "bce", False, 0.5, 7.5, 1.5, task),
            "D_cbam": _cfg(base, True, "bce", False, 0.5, 7.5, 1.5, task),
            "E_cbam_vfl": _cfg(base, True, "vfl", False, 0.5, 5.0, 1.0, task),
            "F_full": _cfg(base, True, "vfl", False, 1.0, 7.5, 1.5, task),
        }
    return {
        "N0_vanilla": _cfg(base, False, "bce", False, 0.5, 7.5, 1.5, task),
        "N1_cbam": _cfg(base, True, "bce", False, 0.5, 7.5, 1.5, task),
        "N2_weighted_bce": _cfg(base, False, "bce", True, 0.5, 7.5, 1.5, task),
        "N3_vfl": _cfg(base, False, "vfl", False, 0.5, 7.5, 1.5, task),
        "N4_weighted_vfl": _cfg(base, False, "vfl", True, 0.5, 7.5, 1.5, task),
        "N5_cbam_weighted_vfl": _cfg(base, True, "vfl", True, 0.5, 5.0, 1.0, task),
        "N6_full": _cfg(base, True, "vfl", True, 1.0, 7.5, 1.5, task),
    }


def baseline_config(base: dict, task: str) -> dict:
    return _cfg(base, False, "bce", False, 0.5, 7.5, 1.5, task)


def loss_sweep_configs(base: dict) -> dict[str, dict]:
    out = {}
    for a in (2.5, 5.0, 7.5, 10.0, 12.5):
        c = clone(base); c["loss"]["box_gain"] = a; c["loss"]["dfl_gain"] = 1.5
        out[f"alpha_{a}"] = c
    for b in (0.5, 1.0, 1.5, 2.0, 2.5):
        c = clone(base); c["loss"]["box_gain"] = 7.5; c["loss"]["dfl_gain"] = b
        out[f"beta_{b}"] = c
    return out
