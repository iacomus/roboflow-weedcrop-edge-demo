# Architecture choices

Notes on the technical decisions made while building this demo, with the reasoning a Solutions Architect would surface to a customer.

## Dataset: `roboflow-100/weed-crop-aerial`

Picked from the [Roboflow 100 benchmark](https://github.com/roboflow-ai/roboflow-100-benchmark) — Roboflow's Intel-sponsored curated suite of 100 datasets used to evaluate model generalisability.

**Why this version specifically**: there are effectively two Roboflow Universe entries derived from the same Sudars et al. 2020 source data:

| Version | Pre-trained mAP@50 | Notes |
|---|---|---|
| [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial) | **79.8%** | RF100 benchmark snapshot, well-characterised, yolov5-tagged |
| [`new-workspace-csmgu/weedcrop-waifl`](https://universe.roboflow.com/new-workspace-csmgu/weedcrop-waifl) | 58.6% | Active workspace, untagged architecture |

The RF100 snapshot has a better-characterised baseline and a stronger narrative anchor — using a dataset from the platform's own benchmark suite signals familiarity with the platform's research output, not just its dataset library.

## Model architecture: RF-DETR

Roboflow's iOS Swift SDK supports a restricted set of architectures for CoreML deployment: **RF-DETR**, **YoloLite**, and **Classification models**. YOLOv5 (which the pre-trained RF100 model uses) is **not** on that list.

The trade-off table I worked through:

| Architecture | iOS CoreML via SDK | Accuracy ceiling | Latency on Neural Engine | Recommendation |
|---|---|---|---|---|
| RF-DETR | ✅ Native | Highest published (54.7% mAP at 4.52ms on T4) | Best | **Use this** |
| YoloLite | ✅ Native | Solid baseline | Good | Fallback |
| YOLOv5 | ❌ Not directly supported | The RF100 pre-trained baseline | n/a via SDK | Skip |

Going with RF-DETR also signals alignment with Roboflow's current roadmap — it's their 2025 launched model, optimised for edge.

## Why not Raspberry Pi for the demo

Initially considered a Raspberry Pi 3 as the edge target (I have one running OctoPi for a 3D printer). Ruled out for three reasons:

1. **Performance ceiling**. Pi 3's 1.2 GHz ARM + 1 GB RAM struggles with modern object-detection models — expect 1-3 FPS even for YOLOv8n / RF-DETR Nano. iPhone Neural Engine runs the same workload at 30-130 FPS depending on generation.
2. **Use-case fit**. A crop-scout walking through a field naturally holds a phone, not a tethered Pi. Phones are the realistic deployment target for the demo's "field worker" persona; Pis fit "fixed-mount monitoring" or "tractor-cabin compute" — different customer conversation.
3. **Risk to existing setup**. Repurposing my Pi 3 would mean either flashing over the OctoPi image (destructive) or buying a new SD card. iPhone has neither problem.

A Pi-based deployment is still the right answer for some real customers (autonomous tractor compute, fixed greenhouse monitoring), and worth raising in customer discovery — but it's not the right *demo* target.

## On-device vs serverless inference

Roboflow exposes three deployment paths. Each fits a different customer pattern:

| Path | Latency | Internet required | Cost per inference | Best fit |
|---|---|---|---|---|
| **Serverless Hosted API** | 150-500ms | Every frame | Per-call | Quick prototypes, low-volume production, batch processing |
| **Roboflow Inference Server** (self-hosted Docker) | 10-100ms | Only first-load | Hardware + ops cost | On-prem, edge gateway, Jetson/x86 deployments |
| **Native iOS SDK** (CoreML on Neural Engine) | 5-30ms | Only first-load | Zero per-inference | Mobile field deployment, offline-capable |

For this demo, the native SDK is the right pick because the demo persona (field scout with phone) maps to that path. A customer evaluating Roboflow for industrial weed-detection robotics would more likely land on the Inference Server with a Jetson Orin. The fact that the same trained model can target all three paths is itself a sales point.

## UI patterns borrowed from Roboflow's Cash Counter

The official [Cash Counter app](https://apps.apple.com/us/app/roboflow-cash-counter/id1633812788) is a useful reference for the patterns Roboflow themselves think work:

- **De-emphasized bounding boxes** — subtle highlights, not in-your-face overlays
- **Big readout of the business metric** — for Cash Counter it's total cash; for this demo it's `crops / weeds / weed-pressure %`
- **"Incorrect Detection?" feedback button** — wired to `rf.uploadImage()` so user corrections flow back to the dataset
- **FPS counter** as a corner overlay — for technical credibility, not customer-facing

The Cash Counter source isn't public, but these patterns are easy to replicate on top of [`roboflow-ios-starter`](https://github.com/roboflow/roboflow-ios-starter).

## What I'd want to extend with more time

If this were a real customer engagement rather than a portfolio piece, the next-tier work would be:

- **Field-test the model** with a small custom dataset of LA-yard weeds to see how the RF100-trained baseline generalises out of distribution
- **Add temporal smoothing** — averaging detections across frames to reduce flicker
- **Capture geo-tagged inference** for per-row yield estimates rather than per-frame counts
- **Add an offline-capable model cache** so the app works in poor-signal field conditions
- **Benchmark RF-DETR vs YoloLite** on the same dataset to give the customer real comparison numbers, not just architecture claims
