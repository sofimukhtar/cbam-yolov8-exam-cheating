from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import clone, load_config, resolve
from .utils import log, save_json, seed_all


def _out(cfg, *parts) -> Path:
    p = resolve(cfg, "runs").joinpath("results", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cmd_extract(cfg, a):
    from .data import extract_all
    if a.fps:
        cfg["data"]["sample_fps"] = a.fps
        from .system import frames_dir
        base = resolve(cfg, "frames")
        cfg["paths"]["frames"] = str(base.parent / f"{base.name}_{a.fps:g}fps") if a.fps != 1 else str(base)
    extract_all(cfg)


def cmd_build(cfg, a):
    from .data import build_datasets
    build_datasets(cfg)


def cmd_stats(cfg, a):
    from .data import annotation_heatmap, dataset_stats
    rep = dataset_stats(cfg)
    print(json.dumps(rep, indent=1))
    root = resolve(cfg, "datasets")
    annotation_heatmap(root / "behavior" / "labels" / "train", _out(cfg, "annotation_heatmap.png"))


def cmd_augment(cfg, a):
    from .data import augment_minority
    augment_minority(cfg, seed=a.seed)


def cmd_agreement(cfg, a):
    from .data import annotator_agreement, stratified_sample
    if a.sample:
        manifest = json.loads((resolve(cfg, "datasets") / "manifest.json").read_text())
        names = sum(manifest.values(), [])
        sel = stratified_sample(names, 0.10, a.seed)
        Path(a.sample).write_text("\n".join(sel))
        log(f"{len(sel)} images written to {a.sample}")
        return
    classes = cfg["classes"]["status"] + cfg["classes"]["behavior"]
    r = annotator_agreement([Path(d) for d in a.dirs], classes)
    print(json.dumps(r, indent=1))
    save_json(r, _out(cfg, "annotator_agreement.json"))


def cmd_train(cfg, a):
    from .train import baseline_config, train_detector
    c = baseline_config(cfg, a.task) if a.baseline else cfg
    best = train_detector(c, a.task, a.name or f"{a.task}_{'vanilla' if a.baseline else 'proposed'}_s{a.seed}",
                          seed=a.seed)
    print(best)


def evaluate_weights(cfg, task, weights, split="test", tag=None, boot=True, plot=False):
    from .evaluate import (binary_decision_accuracy, bootstrap_ci, compute_metrics, confusion_matrix,
                           list_images, predict_records, restrict, save_records)
    names = cfg["classes"][task]
    imgs = list_images(resolve(cfg, "datasets") / task / "images" / split)
    rec = predict_records(weights, imgs, cfg)
    tag = tag or f"{task}_{Path(weights).parent.parent.name}_{split}"
    save_records(rec, _out(cfg, "records", f"{tag}.npz"))
    conf = cfg["eval"]["conf"]
    allc = list(range(len(names)))
    res = {"weights": str(weights), "split": split, "all": compute_metrics(rec, allc, conf, names)}
    if boot:
        res["all"]["bootstrap_ci"] = bootstrap_ci(rec, allc, cfg["eval"]["bootstrap"], conf)
    if task == "status":
        two = [names.index(n) for n in cfg["classes"]["status_candidate"]]
        r2 = restrict(rec, two)
        res["two_class"] = compute_metrics(r2, two, conf, names)
        res["binary_decision"] = binary_decision_accuracy(rec, names.index("Student_cheating"), conf)
        if boot:
            res["two_class"]["bootstrap_ci"] = bootstrap_ci(r2, two, cfg["eval"]["bootstrap"], conf)
    if plot:
        from .plots import confusion_figure
        M = confusion_matrix(rec, len(names), conf, cfg["eval"]["match_iou"])
        res["confusion_matrix"] = M.tolist()
        confusion_figure(M, names + ["background"], _out(cfg, f"confusion_{tag}"), figsize=(8, 7))
    save_json(res, _out(cfg, f"eval_{tag}.json"))
    return res, rec


def cmd_evaluate(cfg, a):
    res, _ = evaluate_weights(cfg, a.task, a.weights, a.split, a.tag, boot=not a.no_bootstrap, plot=True)
    print(json.dumps(res["all"]["overall"], indent=1))


def cmd_repeated(cfg, a):
    from .evaluate import paired_bootstrap, paired_ttest, restrict
    from .train import baseline_config, train_detector
    task = a.task
    names = cfg["classes"][task]
    seeds = cfg["train"]["seeds"]
    runs = {"vanilla": [], "proposed": []}
    recs = {}
    for seed in seeds:
        for tag, c in (("vanilla", baseline_config(cfg, task)), ("proposed", cfg)):
            w = train_detector(c, task, f"{task}_{tag}_s{seed}", seed=seed)
            res, rec = evaluate_weights(c, task, w, "test", f"{task}_{tag}_s{seed}", boot=(seed == seeds[0]))
            runs[tag].append(res)
            if seed == seeds[0]:
                recs[tag] = rec
    out = {}
    subsets = {"all": list(range(len(names)))}
    if task == "status":
        subsets["two_class"] = [names.index(n) for n in cfg["classes"]["status_candidate"]]
    for sub, cls in subsets.items():
        out[sub] = {}
        for key in ("map50", "map50_95", "f1_macro"):
            a_ = [r[sub]["overall"][key] for r in runs["vanilla"]]
            b_ = [r[sub]["overall"][key] for r in runs["proposed"]]
            out[sub][key] = paired_ttest(a_, b_)
        ra, rb = restrict(recs["vanilla"], cls), restrict(recs["proposed"], cls)
        out[sub]["paired_bootstrap_seed0"] = paired_bootstrap(ra, rb, cls, cfg["eval"]["bootstrap"],
                                                              cfg["eval"]["conf"])
    save_json(out, _out(cfg, f"repeated_{task}.json"))
    print(json.dumps(out, indent=1))


def cmd_ablation(cfg, a):
    from .train import ablation_configs, train_detector
    rows = {}
    for name, c in ablation_configs(cfg, a.task).items():
        w = train_detector(c, a.task, f"ablation_{a.task}_{name}", seed=a.seed)
        res, _ = evaluate_weights(c, a.task, w, "test", f"ablation_{a.task}_{name}", boot=False)
        o = res["all"]["overall"]
        row = {k: o[k] for k in ("map50", "map50_95", "f1_macro", "far")}
        if a.task == "behavior":
            names = cfg["classes"]["behavior"]
            row["minority_ap50"] = float(np.mean([res["all"]["per_class"][n]["ap50"]
                                                  for n in cfg["classes"]["minority"] if n in names]))
        rows[name] = row
        log(f"{name}: {row}")
    save_json(rows, _out(cfg, f"ablation_{a.task}.json"))
    print(json.dumps(rows, indent=1))


def cmd_loss_sweep(cfg, a):
    from .train import loss_sweep_configs, train_detector
    rows = {}
    for name, c in loss_sweep_configs(cfg).items():
        rows[name] = {}
        for task in ("status", "behavior"):
            w = train_detector(c, task, f"sweep_{task}_{name}", seed=0)
            res, _ = evaluate_weights(c, task, w, "val", f"sweep_{task}_{name}", boot=False)
            rows[name][task] = {k: res["all"]["overall"][k] for k in ("map50", "map50_95")}
        log(f"{name}: {rows[name]}")
    best = max(rows, key=lambda k: rows[k]["status"]["map50_95"])
    save_json({"rows": rows, "selected": best}, _out(cfg, "loss_sweep.json"))
    print(json.dumps({"selected": best, "rows": rows}, indent=1))


def cmd_cross(cfg, a):
    from .crossdata import build_custom_two_class, kfold_datasets, prepare_examcev2
    from .evaluate import compute_metrics, list_images, predict_records, restrict, save_records
    from .train import baseline_config, train_detector
    name_map = json.loads(a.name_map) if a.name_map else None
    custom2 = build_custom_two_class(cfg)
    exam_yaml, exam_all = prepare_examcev2(cfg, name_map)
    custom_test = list_images(resolve(cfg, "datasets") / "status2" / "images" / "test")
    two = [0, 1]
    names2 = cfg["classes"]["status_candidate"]
    conf = cfg["eval"]["conf"]
    out = {}
    for tag, c in (("proposed", cfg), ("vanilla", baseline_config(cfg, "status"))):
        c = clone(c)
        c["imbalance"]["class_weighting"]["status"] = False
        w1 = train_detector(c, "status", f"cd1_{tag}", seed=a.seed, data_yaml=custom2, class_weights=None)
        r1 = predict_records(w1, exam_all, c)
        rin = predict_records(w1, custom_test, c)
        w2 = train_detector(c, "status", f"cd2_{tag}", seed=a.seed, data_yaml=exam_yaml, class_weights=None)
        r2 = predict_records(w2, custom_test, c)
        oof = []
        for f, (yml, test_imgs) in enumerate(kfold_datasets(cfg, cfg["eval"]["cv_folds"])):
            wf = train_detector(c, "status", f"cv{f}_{tag}", seed=a.seed, data_yaml=yml, class_weights=None)
            oof += predict_records(wf, test_imgs, c)
        for k, rec in (("cd1", r1), ("cd2", r2), ("indomain_custom", rin), ("cv5_examcev2", oof)):
            save_records(rec, _out(cfg, "records", f"cross_{k}_{tag}.npz"))
        res = {"CD-1": compute_metrics(r1, two, conf, names2)["overall"],
               "CD-2": compute_metrics(r2, two, conf, names2)["overall"],
               "in_domain_custom": compute_metrics(rin, two, conf, names2)["overall"],
               "cv5_examcev2": compute_metrics(oof, two, conf, names2)["overall"]}
        res["change_CD-1"] = {k: res["CD-1"][k] - res["cv5_examcev2"][k] for k in ("map50", "map50_95")}
        res["change_CD-2"] = {k: res["CD-2"][k] - res["in_domain_custom"][k] for k in ("map50", "map50_95")}
        out[tag] = res
        log(f"{tag}: {json.dumps({k: v for k, v in res.items() if k.startswith('change')})}")
    save_json(out, _out(cfg, "cross_dataset.json"))
    print(json.dumps(out, indent=1))


def cmd_complexity(cfg, a):
    from .evaluate import model_complexity
    r = model_complexity(cfg, len(cfg["classes"][a.task]), a.device)
    save_json(r, _out(cfg, f"complexity_{a.task}.json"))
    print(json.dumps(r, indent=1))


def cmd_system(cfg, a):
    from .engine import SeatMap
    from .plots import confusion_figure
    from .system import (associate_session, condition_frames, detect_session, evaluate_sessions,
                         identity_accuracy, load_session_gt, score_session, select_thresholds,
                         sensitivity, tracking_metrics)
    split = cfg["data"]["split"]
    cache_root = resolve(cfg, "runs") / "cache"
    seat_dir = resolve(cfg, "seat_maps")
    rates = sorted(set([1] + (cfg["system_eval"]["sampling_rates"] if a.sensitivity else [])))
    data = {"val": {}, "test": {}}
    dets_1fps, seats = {}, {}
    for part in ("val", "test"):
        for fps in rates:
            data[part][fps] = []
            for s in split[part]:
                from .system import frames_dir
                if not (frames_dir(cfg, fps) / f"session_{s:02d}").exists():
                    log(f"no frames for session {s} at {fps} fps; skipped")
                    continue
                seat_map = SeatMap.load(seat_dir / f"session_{s:02d}.json")
                gt = load_session_gt(cfg, s)
                dets = detect_session(cfg, s, fps, a.status_weights, a.behavior_weights,
                                      cache_root / f"dets_s{s:02d}_{fps:g}fps.npz")
                assoc = associate_session(cfg, dets, seat_map, fps=fps)
                data[part][fps].append({"session": s, "assoc": assoc, "gt": gt})
                if fps == 1:
                    dets_1fps[s], seats[s] = dets, seat_map
            if not data[part][fps]:
                del data[part][fps]

    def results(sessions, **kw):
        return [{"students": x["assoc"]["students"], "gt": x["gt"], "fps": x["assoc"]["fps"],
                 "scores": score_session(x["assoc"], cfg, **kw)} for x in sessions]

    th = select_thresholds(results(data["val"][1]), cfg)
    th_obs = select_thresholds(results(data["val"][1], normalization="observed"), cfg)
    log(f"thresholds: {th}  observation-normalised: {th_obs}")
    test_res = evaluate_sessions(results(data["test"][1]), cfg, th["theta_low"], th["theta_mid"])
    report = {"thresholds": th, "thresholds_obs": th_obs, "student_level": test_res}
    confusion_figure(np.array(test_res["confusion"]), ["Green", "Yellow", "Red"],
                     _out(cfg, "student_confusion_matrix"), tick_step=2)
    save_json({str(x["session"]): {"students": x["assoc"]["students"],
                                   "S": score_session(x["assoc"], cfg)["S"]} for x in data["test"][1]},
              _out(cfg, "running_scores_test.json"))

    if all(x["gt"].identities is not None for x in data["test"][1]):
        track = {"conditions": {}, "variants": {}}
        conds = ["all", "occlusion", "movement", "overlap"]
        agg = {c: {"idtp": 0, "rows": []} for c in conds}
        for x in data["test"][1]:
            s, gt = x["session"], x["gt"]
            fr = condition_frames(gt, seats[s], cfg)
            for c in conds:
                frames = None if c == "all" else fr[c]
                m = tracking_metrics(x["assoc"]["track_log"], gt, seats[s], cfg, frames)
                if gt.behaviors is not None:
                    m.update(identity_accuracy(x["assoc"], dets_1fps[s], gt, cfg, frames))
                agg[c]["rows"].append(m)
        for c in conds:
            rows = agg[c]["rows"]
            w = np.array([r["gt_boxes"] for r in rows], float)
            if sum(r["frames"] for r in rows) == 0:
                track["conditions"][c] = {"frames": 0, "idsw": 0, "idf1": None, "iaa": None,
                                          "ambiguous_fraction": None}
                continue
            track["conditions"][c] = {
                "frames": int(sum(r["frames"] for r in rows)), "idsw": int(sum(r["idsw"] for r in rows)),
                "idf1": float(np.average([r["idf1"] for r in rows], weights=np.maximum(w, 1))),
                "iaa": float(np.mean([r.get("iaa", np.nan) for r in rows])),
                "ambiguous_fraction": float(np.mean([r.get("ambiguous_fraction", np.nan) for r in rows]))}
        for mode in ("per_frame", "no_ambiguity", "proposed", "oracle"):
            sess = []
            ms = []
            for x in data["test"][1]:
                s = x["session"]
                assoc = x["assoc"] if mode == "proposed" else \
                    associate_session(cfg, dets_1fps[s], seats[s], oracle=x["gt"], mode=mode)
                sess.append({"assoc": assoc, "gt": x["gt"]})
                m = {} if mode == "oracle" else tracking_metrics(assoc["track_log"], x["gt"], seats[s], cfg)
                if x["gt"].behaviors is not None:
                    m.update(identity_accuracy(assoc, dets_1fps[s], x["gt"], cfg))
                ms.append(m)
            ev = evaluate_sessions(results(sess), cfg, th["theta_low"], th["theta_mid"])
            track["variants"][mode] = {
                "idf1": float(np.mean([m.get("idf1", np.nan) for m in ms])),
                "idsw": int(sum(m.get("idsw", 0) for m in ms)),
                "iaa": float(np.mean([m.get("iaa", np.nan) for m in ms])),
                "student_macro_f1": ev["macro"]["f1"]}
            if mode == "oracle":
                ref = results(data["test"][1])
                orc = results(sess)
                d = np.concatenate([np.abs(r["scores"]["final"] - o["scores"]["final"]) for r, o in zip(ref, orc)])
                from .engine import level
                lr = np.concatenate([level(r["scores"]["final"], th["theta_low"], th["theta_mid"]) for r in ref])
                lo = np.concatenate([level(o["scores"]["final"], th["theta_low"], th["theta_mid"]) for o in orc])
                track["variants"]["oracle"].update({"mean_abs_score_diff": float(d.mean()),
                                                    "alert_level_changes": int((lr != lo).sum())})
        report["tracking"] = track
    else:
        log("identities.csv missing: tracking evaluation skipped")

    if a.sensitivity:
        report["sensitivity"] = sensitivity(cfg, data["val"], data["test"], th)
    save_json(report, _out(cfg, "system_report.json"))
    print(json.dumps({k: report[k] for k in ("thresholds",)} |
                     {"macro": test_res["macro"], "binary": test_res["binary"],
                      "fa_per_student_hour": test_res["fa_per_student_hour"],
                      "latency_s": test_res["latency_s"], "event": test_res["event"]}, indent=1))


def cmd_live(cfg, a):
    from .live import run_live
    sources = {i + 1: s for i, s in enumerate(a.sources)}
    run_live(cfg, sources, Path(a.seat_map), a.status_weights, a.behavior_weights,
             resolve(cfg, "runs") / "live" / "state.json", a.max_steps)


def cmd_dashboard(cfg, a):
    from .dashboard import create_app
    create_app(resolve(cfg, "runs") / "live" / "state.json").run(host=a.host, port=a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="examcheat")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable, e.g. --set train.epochs=30 --set train.device=cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extract-frames"); s.add_argument("--fps", type=float, default=None); s.set_defaults(f=cmd_extract)
    sub.add_parser("build-datasets").set_defaults(f=cmd_build)
    sub.add_parser("stats").set_defaults(f=cmd_stats)
    s = sub.add_parser("augment"); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_augment)
    s = sub.add_parser("agreement"); s.add_argument("--dirs", nargs="*", default=[])
    s.add_argument("--sample", default=None, help="write a stratified 10 %% sample list to this file")
    s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_agreement)

    s = sub.add_parser("train"); s.add_argument("--task", choices=["status", "behavior"], required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--name", default=None)
    s.add_argument("--baseline", action="store_true", help="vanilla YOLOv8m"); s.set_defaults(f=cmd_train)
    s = sub.add_parser("evaluate"); s.add_argument("--task", choices=["status", "behavior"], required=True)
    s.add_argument("--weights", required=True); s.add_argument("--split", default="test")
    s.add_argument("--tag", default=None); s.add_argument("--no-bootstrap", action="store_true")
    s.set_defaults(f=cmd_evaluate)
    s = sub.add_parser("repeated"); s.add_argument("--task", choices=["status", "behavior"], required=True)
    s.set_defaults(f=cmd_repeated)
    s = sub.add_parser("ablation"); s.add_argument("--task", choices=["status", "behavior"], required=True)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_ablation)
    sub.add_parser("loss-sweep").set_defaults(f=cmd_loss_sweep)
    s = sub.add_parser("cross-dataset"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--name-map", default=None, help='JSON, e.g. {"cheating":"Student_cheating"}')
    s.set_defaults(f=cmd_cross)
    s = sub.add_parser("complexity"); s.add_argument("--task", default="status"); s.add_argument("--device", default="cpu")
    s.set_defaults(f=cmd_complexity)

    s = sub.add_parser("system"); s.add_argument("--status-weights", required=True)
    s.add_argument("--behavior-weights", required=True); s.add_argument("--sensitivity", action="store_true")
    s.set_defaults(f=cmd_system)
    s = sub.add_parser("live"); s.add_argument("--sources", nargs="+", required=True)
    s.add_argument("--seat-map", required=True); s.add_argument("--status-weights", required=True)
    s.add_argument("--behavior-weights", required=True); s.add_argument("--max-steps", type=int, default=None)
    s.set_defaults(f=cmd_live)
    s = sub.add_parser("dashboard"); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=5000)
    s.set_defaults(f=cmd_dashboard)

    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    seed_all(getattr(a, "seed", 0) or 0)
    a.f(cfg, a)


if __name__ == "__main__":
    main()
