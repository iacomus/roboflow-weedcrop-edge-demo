#!/usr/bin/env python
"""Validate rfdetr_small.mlpackage on the real test set (macOS CoreML runtime).

Checks three things:
  1) Detections are sensible on real weed/crop images (vs ground truth).
  2) Which logit channel maps to which class (logits is width-4 for 2 classes).
  3) Parity: CoreML output vs the original PyTorch model on the same image
     (quantifies the bilinear-pos-embed + fp16 approximation).
Renders an annotated comparison image to test_output/.
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import coremltools as ct
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MLP = os.path.join(HERE, "rfdetr_small.mlpackage")
DS = os.path.join(HERE, "dataset", "test")
OUTDIR = os.path.join(HERE, "test_output")
os.makedirs(OUTDIR, exist_ok=True)
CONF = 0.40
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def preprocess(path):
    im = Image.open(path).convert("RGB").resize((512, 512))
    a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
    return ((a - MEAN) / STD)[None].astype(np.float32), im


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(boxes, logits, conf=CONF, topk=50):
    scores = sigmoid(logits)              # (300, C)
    cls = scores.argmax(1)
    sc = scores.max(1)
    out = []
    for i in np.where(sc > conf)[0]:
        cx, cy, w, h = boxes[i]
        out.append((int(cls[i]), float(sc[i]),
                    ((cx - w / 2) * 512, (cy - h / 2) * 512,
                     (cx + w / 2) * 512, (cy + h / 2) * 512)))
    out.sort(key=lambda t: -t[1])
    return out[:topk]


def main():
    ann = json.load(open(os.path.join(DS, "_annotations.coco.json")))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    fname = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = defaultdict(list)
    for a in ann["annotations"]:
        gt[a["image_id"]].append(a["category_id"])
    print("categories:", cats)

    model = ct.models.MLModel(MLP)

    # ---- run on a sample of test images; tally predicted-channel vs GT class ----
    ch_vs_gt = defaultdict(Counter)
    sample_ids = list(fname)[:12]
    detail = []
    for iid in sample_ids:
        path = os.path.join(DS, fname[iid])
        if not os.path.exists(path):
            continue
        x, im = preprocess(path)
        p = model.predict({"image": im})
        boxes, logits = p["boxes"][0], p["logits"][0]
        dets = decode(boxes, logits)
        gt_counts = Counter(cats[c] for c in gt[iid])
        pred_ch = Counter(d[0] for d in dets)
        detail.append((fname[iid], dict(gt_counts), dict(pred_ch), len(dets)))
        # correlate: if the image is dominated by one GT class, attribute channels to it
        if gt_counts:
            dom = gt_counts.most_common(1)[0][0]
            for ch in pred_ch:
                ch_vs_gt[ch][dom] += pred_ch[ch]

    print("\n=== per-image: GT classes  vs  predicted channels (conf>%.2f) ===" % CONF)
    for fn, g, pc, n in detail:
        print(f"  {fn[:34]:34}  GT={g}  pred_ch={pc}  (n={n})")

    print("\n=== channel -> class correlation (channel: {dominant-GT-class: votes}) ===")
    for ch in sorted(ch_vs_gt):
        print(f"  logit channel {ch}: {dict(ch_vs_gt[ch])}")

    # ---- PyTorch parity on one image ----
    print("\n=== PyTorch parity (original bicubic model vs CoreML) ===")
    import torch
    from rfdetr import RFDETRSmall
    ckpt = os.path.join(HERE, "colab", "checkpoint_best_ema.pth")
    m = RFDETRSmall(pretrain_weights=ckpt)
    net = m.model.model.eval().to("cpu")
    iid = sample_ids[0]
    x, im0 = preprocess(os.path.join(DS, fname[iid]))
    with torch.no_grad():
        o = net(torch.from_numpy(x))
    pb, pl = o["pred_boxes"][0].numpy(), o["pred_logits"][0].numpy()
    p = model.predict({"image": im0})
    cb, cl = p["boxes"][0], p["logits"][0]
    print(f"  boxes  max|d| = {np.abs(pb - cb).max():.4e}")
    print(f"  logits max|d| = {np.abs(pl - cl).max():.4e}")
    # do they agree on the detections?
    print("  PyTorch top-5 (ch,score):", [(int(sigmoid(pl[i]).argmax()), round(float(sigmoid(pl[i]).max()),3))
                                            for i in np.argsort(-sigmoid(pl).max(1))[:5]])
    print("  CoreML  top-5 (ch,score):", [(int(sigmoid(cl[i]).argmax()), round(float(sigmoid(cl[i]).max()),3))
                                            for i in np.argsort(-sigmoid(cl).max(1))[:5]])

    # ---- render the busiest sampled image ----
    busiest = max(detail, key=lambda d: d[3])[0] if detail else fname[sample_ids[0]]
    x, im = preprocess(os.path.join(DS, busiest))
    p = model.predict({"image": im})
    dets = decode(p["boxes"][0], p["logits"][0])
    colors = {1: (0, 200, 0), 2: (220, 40, 40)}  # crop=green, weed=red (assuming ch1/2)
    dr = ImageDraw.Draw(im)
    for ch, sc, (x1, y1, x2, y2) in dets:
        col = colors.get(ch, (255, 200, 0))
        dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
        dr.text((x1 + 2, y1 + 2), f"{cats.get(ch, ch)} {sc:.2f}", fill=col)
    outp = os.path.join(OUTDIR, "coreml_test_detections.png")
    im.save(outp)
    print(f"\nrendered {len(dets)} detections on {busiest[:30]} -> {outp}")


if __name__ == "__main__":
    main()
