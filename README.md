# Fine-Grained Cheating Behavior Recognition in Offline Examinations

Source code for the manuscript *"An Expert System for Fine-Grained Cheating Behavior
Recognition in Offline Examinations using CBAM-Enhanced YOLO Architecture"*.



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



## Ethics

Video of identifiable people must be collected with informed consent and stored securely.
The cheat score and alert level are supporting evidence for invigilators, not proof of
misconduct; any decision remains with the invigilator and the examination authority.
