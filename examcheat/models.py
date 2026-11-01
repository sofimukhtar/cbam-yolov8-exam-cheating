from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


class ChannelAttentionMLP(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(c // r, c, 1, bias=False))

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg + mx)


class SpatialAttention7(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return torch.sigmoid(self.conv(a))


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8, k: int = 7):
        super().__init__()
        self.ca = ChannelAttentionMLP(c, r)
        self.sa = SpatialAttention7(k)

    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x


class C2fCBAM(C2f):
    def forward(self, x):
        return self.cbam(super().forward(x))

    def forward_split(self, x):
        return self.cbam(super().forward_split(x))


def insert_cbam(model: DetectionModel, layers=(4, 6, 8), r: int = 8, k: int = 7) -> list[int]:
    added = []
    for i in layers:
        m = model.model[i]
        if isinstance(m, C2fCBAM):
            continue
        if not isinstance(m, C2f):
            raise TypeError(f"layer {i} is {type(m).__name__}, expected C2f")
        c = m.cv2.conv.out_channels
        dev = next(m.parameters()).device
        m.__class__ = C2fCBAM
        m.add_module("cbam", CBAM(c, r, k).to(dev))
        m.type = "C2fCBAM"
        m.np = sum(p.numel() for p in m.parameters())
        added.append(sum(p.numel() for p in m.cbam.parameters()))
    return added


class ExamDetectionLoss(v8DetectionLoss):
    def __init__(self, model, cls_type: str = "vfl", class_weights=None,
                 vfl_alpha: float = 0.75, vfl_gamma: float = 2.0,
                 tal_topk: int = 10, tal_alpha: float = 0.5, tal_beta: float = 6.0):
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc,
                                            alpha=tal_alpha, beta=tal_beta)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.cls_type = cls_type
        self.vfl_alpha, self.vfl_gamma = vfl_alpha, vfl_gamma
        w = torch.ones(self.nc) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float)
        if len(w) != self.nc:
            raise ValueError(f"{len(w)} class weights for {self.nc} classes")
        self.cw = w.to(self.device).view(1, 1, -1)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        bs = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)

        logits = pred_scores.float()
        cw = self.cw.to(logits.device)
        if self.cls_type == "vfl":
            q = torch.zeros_like(logits)
            if fg_mask.any():
                iou = bbox_iou(pred_bboxes.detach()[fg_mask] * stride_tensor.expand_as(pred_bboxes)[fg_mask],
                               target_bboxes[fg_mask], xywh=False).squeeze(-1).clamp(0).float()
                q[fg_mask, target_labels[fg_mask].long()] = iou
            pos = (q > 0).float()
            p = logits.sigmoid()
            weight = q * pos + self.vfl_alpha * p.detach().pow(self.vfl_gamma) * (1 - pos)
            ce = F.binary_cross_entropy_with_logits(logits, q, reduction="none")
            loss[1] = (ce * weight * cw).sum() / target_scores_sum
        else:
            ce = F.binary_cross_entropy_with_logits(logits, target_scores.float(), reduction="none")
            loss[1] = (ce * cw).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes,
                                              target_scores, target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * bs, loss.detach()


class ExamDetectionModel(DetectionModel):
    def __init__(self, cfg="yolov8m.yaml", ch=3, nc=None, verbose=True, cbam: dict | None = None,
                 loss_cfg: dict | None = None, class_weights=None):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.exam_loss_cfg = dict(loss_cfg or {"cls_type": "bce"})
        self.exam_class_weights = None if class_weights is None else list(map(float, class_weights))
        self.cbam_params = []
        if cbam and cbam.get("enabled", False):
            self.cbam_params = insert_cbam(self, cbam.get("layers", (4, 6, 8)),
                                           cbam.get("reduction", 8), cbam.get("spatial_kernel", 7))

    def init_criterion(self):
        lc = self.exam_loss_cfg
        return ExamDetectionLoss(self, cls_type=lc.get("cls_type", "vfl"),
                                 class_weights=self.exam_class_weights,
                                 vfl_alpha=lc.get("vfl_alpha", 0.75), vfl_gamma=lc.get("vfl_gamma", 2.0),
                                 tal_topk=lc.get("tal_topk", 10), tal_alpha=lc.get("tal_alpha", 0.5),
                                 tal_beta=lc.get("tal_beta", 6.0))


def build_model(cfg: dict, nc: int, class_weights=None, verbose=False) -> ExamDetectionModel:
    m = ExamDetectionModel(cfg["model"]["cfg"], nc=nc, verbose=verbose, cbam=cfg["model"]["cbam"],
                           loss_cfg=cfg["loss"], class_weights=class_weights)
    return m


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
