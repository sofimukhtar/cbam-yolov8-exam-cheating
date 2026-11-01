# Fine-Grained Cheating Behavior Recognition in Offline Examinations

Source code for the manuscript *"An Expert System for Fine-Grained Cheating Behavior
Recognition in Offline Examinations using CBAM-Enhanced YOLO Architecture"*.

The code implements the full pipeline described in the paper:

| Manuscript component | Module |
|---|---|
| Frame extraction at 1 fps, session-level split (1–6 / 7–8 / 9–10), VOC/YOLO conversion | `examcheat/data.py` |
| Annotation statistics, class-frequency weights `w_c = N_total / (C·N_c)` (Eq. 1), 25×25 heatmap | `examcheat/data.py` |
| Targeted augmentation of Cheat_note, Using_cheat_sheets, Signalling (flip, ±25 % brightness, ±10° rotation) | `examcheat/data.py` |
| Fleiss' κ and mean box IoU of three annotators, stratified 10 % sample | `examcheat/data.py` |
| CBAM after the final C2f block at P3, P4, P5 (r = 8, 7×7 kernel) (Eqs. 2–5) | `examcheat/models.py` |
| Class-weighted Varifocal Loss + CIoU + DFL, unchanged TAL (Eqs. 6, 21–24) | `examcheat/models.py` |
| Mosaic oversampling with tile probability ∝ 1/N_c | `examcheat/train.py` |
| Two separate detectors (status: 3 classes, behavior: 9 classes) | `examcheat/train.py` |
| Ablation C–F and N0–N6; α / β sweeps | `examcheat/train.py` |
| mAP@50, mAP@50–95, P, R, F1, macro/weighted F1, PDR, accuracy, FAR (Eqs. 10–19) | `examcheat/evaluate.py` |
| Bootstrap CIs (B = 1,000), paired t-tests over five seeds, paired bootstrap tests | `examcheat/evaluate.py` |
| Parameters, GFLOPs, FP16 size, CBAM latency | `examcheat/evaluate.py` |
| CD-1, CD-2, five-fold ExamCeV2 reference, two-class restriction | `examcheat/crossdata.py` |
| Per-view ByteTrack, track-to-seat mapping, association margin δ, multi-view max fusion (Eq. 12) | `examcheat/engine.py` |
| Running score S_i(t), final score S_i(T), S_i^obs, alert levels (Eqs. 7–9, Algorithm 1) | `examcheat/engine.py` |
| Threshold grid search on validation (macro-F1, FA/h ≤ 0.05) | `examcheat/system.py` |
| Student-level, event-level, latency, IDF1, IDSW, IAA, oracle identities, association variants | `examcheat/system.py` |
| Sensitivity: sampling rate, persistence, severity weights, observation duration, normalisation | `examcheat/system.py` |
| Real-time operation and invigilator dashboard (Flask) | `examcheat/live.py`, `examcheat/dashboard.py` |

Every parameter in the paper is in `configs/default.yaml`; any value can be overridden on
the command line with `--set key=value`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

Tested with Ultralytics 8.3.40, PyTorch 2.x, Python 3.10–3.13. `yolov8m.pt` is downloaded
automatically by Ultralytics on first use.

## Data layout

Commands are run from the project root; all paths in the config are relative to it.

```
data/
  raw_videos/session_01/view_1.mp4 ... session_10/view_4.mp4
  annotations/session_01/view_1/000001.xml        # LabelImg output, file index = time step t
  examcev2/images/*.jpg, labels/*.txt, data.yaml   # ExamCeV2 in YOLO format
  system_gt/session_07 ... session_10/             # system ground truth (validation and test)
configs/
  seat_maps/session_07.json ... session_10.json    # seat regions per view
```

**Annotations.** One Pascal VOC XML per sampled frame (LabelImg default), containing all
twelve class names. YOLO TXT is also accepted if a `classes.txt` is present in the folder.
Class names are matched case-insensitively with spaces or underscores
(`Student Cheating` = `Student_cheating`). Each XML is split into the status and behavior
datasets by class name.

**Seat maps** (`configs/seat_maps/session_XX.json`); the seat identifier is the student identity:
```json
{"session": 9, "views": {"1": {"S01": [x1, y1, x2, y2], "S02": [...]}, "2": {...}}}
```

**System ground truth** (`data/system_gt/session_XX/`), with `t` the 1-based step at 1 fps:

| File | Columns | Used for |
|---|---|---|
| `students.csv` | `student,label` (Green / Yellow / Red) | student-level metrics, thresholds |
| `events.csv` | `student,class,t_start,t_end` (seconds) | event-level metrics, latency |
| `identities.csv` | `t,view,student,x1,y1,x2,y2,visibility` | IDF1, IDSW, condition subsets, oracle |
| `behaviors.csv` | `t,view,student,class,x1,y1,x2,y2` | identity-association accuracy |

`identities.csv` and `behaviors.csv` are only needed for the tracking evaluation.

## Running the experiments

`scripts/run_all.sh` runs every experiment in the paper's order. The individual steps are:

```bash
python -m examcheat extract-frames            # also --fps 2 and --fps 5 for the sensitivity study
python -m examcheat build-datasets
python -m examcheat stats
python -m examcheat augment
python -m examcheat loss-sweep                # alpha / beta on the validation set
python -m examcheat repeated --task status    # 5 seeds, proposed vs. vanilla, paired tests
python -m examcheat repeated --task behavior
python -m examcheat ablation --task status    # C-F
python -m examcheat ablation --task behavior  # N0-N6
python -m examcheat evaluate --task behavior --weights runs/train/behavior_proposed_s0/weights/best.pt
python -m examcheat cross-dataset --name-map '{"cheating": "Student_cheating", "not cheating": "Student_not_cheating"}'
python -m examcheat complexity --device cuda
python -m examcheat system --sensitivity \
    --status-weights runs/train/status_proposed_s0/weights/best.pt \
    --behavior-weights runs/train/behavior_proposed_s0/weights/best.pt
```

Live operation with four synchronized cameras, and the dashboard (http://localhost:5000):
```bash
python -m examcheat live --sources rtsp://cam1 rtsp://cam2 rtsp://cam3 rtsp://cam4 \
    --seat-map configs/seat_maps/hall.json \
    --status-weights runs/train/status_proposed_s0/weights/best.pt \
    --behavior-weights runs/train/behavior_proposed_s0/weights/best.pt
python -m examcheat dashboard
```

All outputs are written to `runs/results/` as JSON (metrics, CIs, tests, tables) and as
PNG/PDF figures (confusion matrices, heatmap).

## Implementation notes

* **Scores.** `S_i(t) = E_i(t)/t` is the running score that drives online alerts;
  `S_i = S_i(T) = E_i(T)/T` is the final score that sets the session alert. Both use the same
  thresholds and coincide at `t = T`. Because `E_i(t)` is non-decreasing and `t` grows, the
  running alert returns to Green at the first `t > E_i(t0)/θ_low` after evidence stops.
  `compute_scores` (offline, vectorised) and `OnlineEngine` (live) implement identical rules.
* **Accuracy and FAR** need true negatives, which object detection does not define. They are
  computed from image-level presence decisions: for each (image, class) pair, the class is
  positive in the ground truth if annotated and positive in the prediction if detected at
  conf ≥ 0.25. The binary cheating decision for the status task uses the Student_cheating class.
* **mAP averaging.** Classes without ground truth in the evaluated set are excluded from the
  averages, as in COCO and Ultralytics. Matching is greedy in descending confidence, as stated
  in the paper. This can differ by a few tenths of a point from Ultralytics' validator, which
  matches by IoU order.
* **ByteTrack buffer.** Ultralytics computes `max_time_lost = frame_rate/30 · track_buffer`;
  the tracker is created with `frame_rate = 30` and `track_buffer = 15·fps`, giving exactly
  τ_lost = 15 s at any sampling rate.
* **Behavior-detection attribution.** The seat-matching cost is `1 − IoA(detection, R_j)`,
  where `R_j` is the union of student j's tracked box and seat region (so that objects on the
  desk are attributed to its owner). The margin used for the δ rule is the smaller of the
  detection margin and the track's seat margin.
* **Parameter counts.** The CBAM blocks add exactly 9,314 / 36,962 / 83,042 parameters
  (129,318 in total). The vanilla YOLOv8m count depends on the number of classes
  (25,858,057 for three classes; 25,902,640 for the 80-class COCO head).
* **Ablation A and B** (generic anchor-based detector, then + C2f/SPPF) cannot be expressed in
  the anchor-free Ultralytics code base. Train them with an anchor-based implementation
  (e.g. the original YOLOv5 repository) on the same split and hyperparameters.
* **Determinism.** Seeds 0–4 set Python, NumPy and PyTorch seeds and Ultralytics'
  `deterministic=True`. Results still vary slightly across GPUs and library versions.

## Ethics

Video of identifiable people must be collected with informed consent and stored securely.
The cheat score and alert level are supporting evidence for invigilators, not proof of
misconduct; any decision remains with the invigilator and the examination authority.
