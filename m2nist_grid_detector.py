import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.ops import nms

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



@dataclass(frozen=True)
class Config:
    grid_h: int = 8               # grid_h: grid height (number of cells in vertical direction)
    grid_w: int = 11              # grid_w: grid width (number of cells in horizontal direction)
    num_classes: int = 10         # num_classes: number of digit classes (0-9)
    epochs: int = 10              # epochs: number of training epochs
    batch_size: int = 64          # batch_size: number of images per training batch
    lr: float = 1e-3              # lr: learning rate for optimizer
    weight_decay: float = 1e-4    # weight_decay: L2 regularization strength
    obj_pos_weight: float = 8.0   # obj_pos_weight: positive class (object) weight for objectness loss
    lambda_obj: float = 1.0       # lambda_obj: weighting for objectness loss in total loss
    lambda_box: float = 5.0       # lambda_box: weighting for bounding box regression loss in total loss
    lambda_cls: float = 1.0       # lambda_cls: weighting for classification loss in total loss
    nms_iou: float = 0.3          # nms_iou: Non-Maximum Suppression IoU threshold
    conf_thr: float = 0.4         # conf_thr: confidence threshold for detection (objectness × class score)


def extract_gt_boxes(seg_mask):
    """
    extract gt boxes from a single segmentation mask.

    seg_mask: (h, w, 10) boolean
    returns: list of (x, y, w, h, cls)
    """
    boxes = []
    for cls in range(10):
        m = seg_mask[:, :, cls]
        if not m.any():
            continue
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1, cls))
    return boxes


class GridDetector(nn.Module):
    """
    tiny cnn backbone + fixed-grid detection head.

    output: [b, 1+4+num_classes, gh, gw]
    channels: obj_logit, tx,ty,tw,th, class_logits...
    """

    def __init__(self, gh, gw, num_classes):
        super().__init__()
        self.gh = gh
        self.gw = gw
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x84 -> 32x42
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x21
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # force a fixed grid size regardless of exact backbone shape
        self.to_grid = nn.AdaptiveAvgPool2d((gh, gw))

        out_ch = 1 + 4 + num_classes
        self.head = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feats = self.backbone(x)
        grid = self.to_grid(feats)
        return self.head(grid)


def build_targets(gt_boxes_batch, gh, gw, h, w, device):
    """
    build dense targets per cell.

    returns:
      obj_t: [b, 1, gh, gw] float in {0,1}
      box_t: [b, 4, gh, gw] float (tx,ty,tw,th)
      cls_t: [b, gh, gw] long (0..9), -1 for no object
    """
    b = len(gt_boxes_batch)
    obj_t = torch.zeros((b, 1, gh, gw), dtype=torch.float32, device=device)
    box_t = torch.zeros((b, 4, gh, gw), dtype=torch.float32, device=device)
    cls_t = torch.full((b, gh, gw), -1, dtype=torch.long, device=device)

    cell_w = w / float(gw)
    cell_h = h / float(gh)

    for bi, gt_boxes in enumerate(gt_boxes_batch):
        # if multiple boxes map to same cell, keep the largest area
        best_area = {}
        for (x, y, bw, bh, cls) in gt_boxes:
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            gj = int(np.clip(cx / w * gw, 0, gw - 1))
            gi = int(np.clip(cy / h * gh, 0, gh - 1))
            area = int(bw * bh)
            key = (gi, gj)
            if key in best_area and area <= best_area[key]:
                continue
            best_area[key] = area

            # encode offsets within the cell
            tx = (cx / cell_w) - gj
            ty = (cy / cell_h) - gi
            tw = bw / float(w)
            th = bh / float(h)

            obj_t[bi, 0, gi, gj] = 1.0
            box_t[bi, :, gi, gj] = torch.tensor([tx, ty, tw, th], device=device)
            cls_t[bi, gi, gj] = int(cls)

    return obj_t, box_t, cls_t


def detection_loss(pred, obj_t, box_t, cls_t, cfg):
    """
    pred: [b, 15, gh, gw]
    """
    obj_logit = pred[:, 0:1, :, :]
    box_pred = pred[:, 1:5, :, :]
    cls_logit = pred[:, 5:, :, :]

    pos_mask = (obj_t > 0.5).float()
    pos_mask_2d = (obj_t[:, 0, :, :] > 0.5)

    # objectness loss over all cells (weighted for imbalance)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.obj_pos_weight], device=pred.device))
    l_obj = bce(obj_logit, obj_t)

    # box loss only on positives
    if pos_mask.sum() > 0:
        l_box = F.smooth_l1_loss(box_pred * pos_mask, box_t * pos_mask, reduction="sum") / (pos_mask.sum() + 1e-6)
    else:
        l_box = torch.zeros((), device=pred.device)

    # class loss only on positives
    if pos_mask_2d.any():
        cls_logit_pos = cls_logit.permute(0, 2, 3, 1)[pos_mask_2d]  # [n_pos, C]
        cls_t_pos = cls_t[pos_mask_2d]
        l_cls = F.cross_entropy(cls_logit_pos, cls_t_pos)
    else:
        l_cls = torch.zeros((), device=pred.device)

    loss = cfg.lambda_obj * l_obj + cfg.lambda_box * l_box + cfg.lambda_cls * l_cls
    stats = {"l_obj": float(l_obj.detach()), "l_box": float(l_box.detach()), "l_cls": float(l_cls.detach())}
    return loss, stats



@torch.no_grad()
def decode_predictions(pred, h, w, cfg):
    """
    decode batch predictions into per-image detections.
    output boxes are xywh in image coords: (x,y,w,h,cls,score)
    """
    b, ch, gh, gw = pred.shape
    assert gh == cfg.grid_h and gw == cfg.grid_w

    obj = torch.sigmoid(pred[:, 0, :, :])  # [b, gh, gw]
    box = pred[:, 1:5, :, :]  # [b, 4, gh, gw]
    cls_logits = pred[:, 5:, :, :]  # [b, c, gh, gw]
    cls_prob = torch.softmax(cls_logits, dim=1)  # [b, c, gh, gw]

    cell_w = w / float(gw)
    cell_h = h / float(gh)

    out = []
    for bi in range(b):
        dets = []
        for gi in range(gh):
            for gj in range(gw):
                o = float(obj[bi, gi, gj])
                # if score < cfg.conf_thr:
                #     continue
                tx, ty, tw, th = [float(box[bi, k, gi, gj]) for k in range(4)]
                # clamp offsets to keep decode stable
                tx = float(np.clip(tx, 0.0, 1.0))
                ty = float(np.clip(ty, 0.0, 1.0))
                tw = float(np.clip(tw, 0.02, 1.0))
                th = float(np.clip(th, 0.02, 1.0))

                cx = (gj + tx) * cell_w
                cy = (gi + ty) * cell_h
                bw = tw * w
                bh = th * h
                x0 = int(round(cx - bw / 2.0))
                y0 = int(round(cy - bh / 2.0))
                x0 = max(0, min(x0, w - 1))
                y0 = max(0, min(y0, h - 1))
                bw = int(round(min(bw, w - x0)))
                bh = int(round(min(bh, h - y0)))

                # class + score
                cprob = cls_prob[bi, :, gi, gj]
                cls = int(torch.argmax(cprob).item())
                cls_score = float(cprob[cls].item())
                score = o * cls_score
                if score < cfg.conf_thr:
                    continue
                dets.append((x0, y0, max(1, bw), max(1, bh), cls, float(score)))

        # nms per image (class-agnostic for simplicity)
        if not dets:
            out.append([])
            continue
        boxes_xyxy = torch.tensor([[x, y, x + bw, y + bh] for x, y, bw, bh, _, _ in dets], dtype=torch.float32)
        scores = torch.tensor([s for *_, s in dets], dtype=torch.float32)
        keep = nms(boxes_xyxy, scores, cfg.nms_iou)
        kept = [dets[i] for i in keep]
        kept.sort(key=lambda d: d[5], reverse=True)
        out.append(kept)
    return out


############################### viewer #########################################

def interactive_viewer(model, images, seg_masks, cfg, device):
    model.eval()
    n = len(images)

    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]})
    idx = {"i": 0}

    def render():
        i = idx["i"]
        ax_main.clear()
        ax_info.clear()
        ax_main.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax_main.set_title(f"grid detector - image {i+1}/{n}", fontsize=14, pad=10)
        ax_main.axis("off")

        # gt (accept either raw mask or precomputed box list)
        gt = seg_masks[i]
        for (x, y, bw, bh, cls) in gt:
            rect = Rectangle((x, y), bw, bh, linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
            ax_main.add_patch(rect)
            ax_main.text(x, y - 2, f"GT:{cls}", color="white", fontsize=8, fontweight="bold")

        # preds
        x = torch.tensor(images[i], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        pred = model(x)
        dets = decode_predictions(pred, h=images.shape[1], w=images.shape[2], cfg=cfg)[0]
        for (x0, y0, bw, bh, cls, score) in dets[:10]:
            rect = Rectangle((x0, y0), bw, bh, linewidth=2, edgecolor="green", facecolor="none")
            ax_main.add_patch(rect)
            ax_main.text(
                x0,
                y0 + bh + 2,
                f"{cls}:{score:.2f}",
                color="green",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1),
            )

        ax_info.text(0.05, 0.95, "SUMMARY", fontweight="bold", fontsize=13, transform=ax_info.transAxes)
        ax_info.text(0.05, 0.88, f"gt: {len(gt)}", fontsize=11, transform=ax_info.transAxes)
        ax_info.text(0.05, 0.83, f"pred: {len(dets)}", fontsize=11, transform=ax_info.transAxes)
        
        gt_classes = sorted({int(b[4]) for b in gt}) if gt else []
        pred_classes = [int(d[4]) for d in dets[:10]]
        pred_text = ", ".join([f"{c}:{dets[k][5]:.2f}" for k, c in enumerate(pred_classes)]) if dets else ""

        ax_info.text(0.05, 0.74, f"gt classes: {gt_classes}", fontsize=10, transform=ax_info.transAxes)
        ax_info.text(0.05, 0.69, "pred classes (top10):", fontsize=10, transform=ax_info.transAxes)
        ax_info.text(0.05, 0.64, pred_text, fontsize=9, transform=ax_info.transAxes, wrap=True)

        ax_info.text(0.05, 0.10, "arrow keys to navigate", fontsize=10, style="italic", transform=ax_info.transAxes)
        ax_info.axis("off")

        fig.canvas.draw()

    def on_key(event):
        if event.key in ("right", "down"):
            idx["i"] = (idx["i"] + 1) % n
        elif event.key in ("left", "up"):
            idx["i"] = (idx["i"] - 1) % n
        render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("dataset/imagedata.npy").astype(np.float32) / 255.0
    gt = np.load("dataset/groundtruth.npy").astype(bool)

    data_train, data_test, gt_train, gt_test = train_test_split(data, gt, train_size=0.8)
    
    print(f"train images: {len(data_train)}  test images: {len(data_test)}")

    model = GridDetector(cfg.grid_h, cfg.grid_w, cfg.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # build gt boxes once (cheap) to avoid recomputing masks every epoch
    gt_train = [extract_gt_boxes(gt_train[i]) for i in range(len(gt_train))]
    gt_test = [extract_gt_boxes(gt_test[i]) for i in range(len(gt_test))]

    # simple dataloader over full images
    train_idx = np.arange(len(data_train))
    for epoch in range(cfg.epochs):
        np.random.shuffle(train_idx)
        model.train()
        total = 0.0
        stats_sum = {"l_obj": 0.0, "l_box": 0.0, "l_cls": 0.0}

        for start in range(0, len(train_idx), cfg.batch_size):
            idx = train_idx[start : start + cfg.batch_size]
            imgs = torch.tensor(data_train[idx], dtype=torch.float32, device=device).unsqueeze(1)
            gts = [gt_train[i] for i in idx]
            obj_t, box_t, cls_t = build_targets(gts, cfg.grid_h, cfg.grid_w, h=64, w=84, device=device)

            pred = model(imgs)
            loss, st = detection_loss(pred, obj_t, box_t, cls_t, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.detach())
            for k in stats_sum:
                stats_sum[k] += st[k]

        denom = max(1, len(train_idx) // cfg.batch_size)
        print(
            f"epoch {epoch+1}/{cfg.epochs}  loss {total/denom:.3f}  "
            f"obj {stats_sum['l_obj']/denom:.3f}  box {stats_sum['l_box']/denom:.3f}  cls {stats_sum['l_cls']/denom:.3f}"
        )

    print("launching viewer...")
    interactive_viewer(model, data_test, gt_test, cfg, device)


if __name__ == "__main__":
    main()

