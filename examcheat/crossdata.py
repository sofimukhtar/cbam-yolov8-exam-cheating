from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import yaml

from .config import resolve
from .data import canon, write_data_yaml
from .utils import log


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def _remap_label(src: Path, dst: Path, idmap: dict[int, int]):
    lines = []
    if src.exists():
        for l in src.read_text().split("\n"):
            p = l.split()
            if len(p) >= 5 and int(p[0]) in idmap:
                lines.append(" ".join([str(idmap[int(p[0])])] + p[1:5]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines))


def build_custom_two_class(cfg: dict) -> Path:
    src = resolve(cfg, "datasets") / "status"
    dst = resolve(cfg, "datasets") / "status2"
    names3, names2 = cfg["classes"]["status"], cfg["classes"]["status_candidate"]
    idmap = {names3.index(n): i for i, n in enumerate(names2)}
    for split in ("train", "val", "test"):
        for img in _images(src / "images" / split):
            _link(img, dst / "images" / split / img.name)
            _remap_label(src / "labels" / split / f"{img.stem}.txt", dst / "labels" / split / f"{img.stem}.txt", idmap)
    return write_data_yaml(dst, names2)


def prepare_examcev2(cfg: dict, name_map: dict[str, str] | None = None, val_frac: float = 0.1,
                     seed: int = 0) -> tuple[Path, list[Path]]:
    src = resolve(cfg, "examcev2")
    if (src / "data.yaml").exists():
        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        names = list(names.values()) if isinstance(names, dict) else list(names)
    else:
        names = (src / "classes.txt").read_text().split("\n")
        names = [n for n in names if n.strip()]
    names2 = cfg["classes"]["status_candidate"]
    target = {canon(n): i for i, n in enumerate(names2)}
    nm = {canon(k): v for k, v in (name_map or {}).items()}
    idmap = {}
    for i, n in enumerate(names):
        key = canon(nm.get(canon(n), n))
        if key in target:
            idmap[i] = target[key]
    if len(set(idmap.values())) != 2:
        raise ValueError(f"could not map ExamCeV2 classes {names} to {names2}; pass name_map")
    dst = resolve(cfg, "datasets") / "examcev2"
    imgs = _images(src / "images")
    for img in imgs:
        rel = img.relative_to(src / "images")
        lab = (src / "labels" / rel).with_suffix(".txt")
        _link(img, dst / "images" / "all" / img.name)
        _remap_label(lab, dst / "labels" / "all" / f"{img.stem}.txt", idmap)
    all_imgs = _images(dst / "images" / "all")
    rng = random.Random(seed)
    order = all_imgs[:]
    rng.shuffle(order)
    n_val = max(1, int(round(val_frac * len(order))))
    for split, part in (("val", order[:n_val]), ("train", order[n_val:])):
        for img in part:
            _link(img, dst / "images" / split / img.name)
            _link(dst / "labels" / "all" / f"{img.stem}.txt", dst / "labels" / split / f"{img.stem}.txt")
    write_data_yaml(dst, names2, splits=("train", "val", "all"))
    log(f"ExamCeV2: {len(all_imgs)} images, class map {idmap}")
    return dst / "data.yaml", all_imgs


def kfold_datasets(cfg: dict, k: int = 5, val_frac: float = 0.1, seed: int = 0) -> list[tuple[Path, list[Path]]]:
    root = resolve(cfg, "datasets") / "examcev2"
    imgs = _images(root / "images" / "all")
    rng = random.Random(seed)
    order = imgs[:]
    rng.shuffle(order)
    folds = [order[i::k] for i in range(k)]
    out = []
    for f in range(k):
        d = resolve(cfg, "datasets") / f"examcev2_fold{f}"
        shutil.rmtree(d, ignore_errors=True)
        train = [p for j in range(k) if j != f for p in folds[j]]
        n_val = max(1, int(round(val_frac * len(train))))
        for split, part in (("val", train[:n_val]), ("train", train[n_val:]), ("test", folds[f])):
            for img in part:
                _link(img, d / "images" / split / img.name)
                _link(root / "labels" / "all" / f"{img.stem}.txt", d / "labels" / split / f"{img.stem}.txt")
        out.append((write_data_yaml(d, cfg["classes"]["status_candidate"]), sorted(d.glob("images/test/*"))))
    return out
