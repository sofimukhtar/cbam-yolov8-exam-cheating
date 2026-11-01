#!/usr/bin/env bash
# Reproduces every experiment of the manuscript, in order.
# Run from the project root on a GPU machine:  bash scripts/run_all.sh
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
EC="python -m examcheat"

# ---------------- 1. Data ----------------
$EC extract-frames                 # 1 fps (training, inference)
$EC extract-frames --fps 2         # sensitivity analysis only
$EC extract-frames --fps 5
$EC build-datasets                 # session-level split 1-6 / 7-8 / 9-10
$EC stats                          # annotation table, class weights, heatmap
$EC augment                        # targeted augmentation (train split only)
$EC agreement --sample runs/results/agreement_sample.txt
# After three annotators label the sampled images (VOC XML, same file names):
# $EC agreement --dirs data/agreement/annotator1 data/agreement/annotator2 data/agreement/annotator3

# ---------------- 2. Loss coefficients (validation set) ----------------
$EC loss-sweep

# ---------------- 3. Detectors: five seeds, proposed vs. vanilla ----------------
$EC repeated --task status
$EC repeated --task behavior

# ---------------- 4. Ablation (single run, seed 0) ----------------
$EC ablation --task status         # C-F (A, B require an anchor-based code base)
$EC ablation --task behavior       # N0-N6

# ---------------- 5. Final per-class results and confusion matrices (seed 0) ----------------
$EC evaluate --task status   --weights runs/train/status_proposed_s0/weights/best.pt
$EC evaluate --task behavior --weights runs/train/behavior_proposed_s0/weights/best.pt

# ---------------- 6. Cross-dataset (CD-1, CD-2, five-fold reference) ----------------
# Adjust the name map to the class names used in your copy of ExamCeV2.
$EC cross-dataset --name-map '{"cheating": "Student_cheating", "not cheating": "Student_not_cheating"}'

# ---------------- 7. Complexity ----------------
$EC complexity --task status --device cuda

# ---------------- 8. System: tracking, cheat score, alerts, sensitivity ----------------
$EC system --sensitivity \
    --status-weights   runs/train/status_proposed_s0/weights/best.pt \
    --behavior-weights runs/train/behavior_proposed_s0/weights/best.pt

echo "All results are in runs/results/"
