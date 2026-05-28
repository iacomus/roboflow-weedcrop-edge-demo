# Architecture choices

Notes on the technical decisions made while building this demo, and the reasoning behind each — the kind you'd surface to a customer.

## Dataset: `roboflow-100/weed-crop-aerial`

Picked from the [Roboflow 100 benchmark](https://github.com/roboflow-ai/roboflow-100-benchmark) — Roboflow's Intel-sponsored curated suite of 100 datasets used to evaluate model generalisability.

**Why this version specifically**: there are effectively two Roboflow Universe entries derived from the same Sudars et al. 2020 source data:

| Version | Pre-trained mAP@50 | Notes |
|---|---|---|
| [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial) | **79.8%** | RF100 benchmark snapshot, well-characterised, yolov5-tagged |
| [`new-workspace-csmgu/weedcrop-waifl`](https://universe.roboflow.com/new-workspace-csmgu/weedcrop-waifl) | 58.6% | Active workspace, untagged architecture |

The RF100 snapshot has a better-characterised baseline and a stronger narrative anchor — using a dataset from the platform's own benchmark suite signals familiarity with the platform's research output, not just its dataset library.

## Empirical deployment investigation (iPhone 17 Pro, SDK 1.2.7)

Before relying on our own model, I systematically tested the iOS SDK's dynamic model-loading path. The journey:

| Test model | Architecture | Result |
|---|---|---|
| `weed-crop-aerial` (RF100 baseline) | YOLOv5 | ❌ `com.apple.mlassetio Code=1` — "Field number 7 has wireType 4" (CoreML protobuf parse failure) |
| `cash-counter` (Roboflow's App Store demo) | old format | ❌ same `wireType` error — **proves it's not architecture- or model-specific** |
| `rf-detr-cfhln` (public) | RF-DETR | ⚠️ `ZipExtractionError Code=3` — no CoreML export was built for that project |
| `glasses-detection-zkmto` v2 (blog tutorial) | RF-DETR | ✅ **`MODEL LOADED OK modelType=rfdetr-nano`** |

### Root cause
I traced the `wireType` error to the SDK's dynamic download→extract→parse pipeline, and isolated it to **older CoreML model formats** specifically — corroborated by a public report ([roboflow/external-bugtracker#4](https://github.com/roboflow/external-bugtracker/issues/4)) and reproducible across environments (Monterey, iOS 17, iOS 26). That last point is the useful one: reproducing it on three OS versions establishes it's a format-handling path, not something device- or iOS-version-specific.

The Cash Counter App Store app works because it **bundles a precompiled model** and loads it via standard CoreML APIs, bypassing the SDK's dynamic-load path entirely.

RF-DETR support landed in the SDK after pod 1.2.3 ([roboflow/roboflow-swift#17](https://github.com/roboflow/roboflow-swift/issues/17)); we're on 1.2.7. A known-working RF-DETR model (`glasses-detection-zkmto` v2, from Roboflow's own RF-DETR-on-iOS blog) **loads cleanly on the device**, confirming the SDK + device + CoreML pipeline works end-to-end for properly-exported RF-DETR.

### Why this matters for the architecture choice
This is the strongest possible validation of choosing RF-DETR: **YOLOv5 and old-format models don't load via the SDK's runtime path, while current RF-DETR exports do.** The benchmark-winning YOLOv5 baseline (79.8% mAP) doesn't deploy to the edge through this path; RF-DETR does.

### The takeaway
Deployment-path compatibility is a discovery-phase question, not a deployment-phase surprise. Systematic isolation (test the baseline, test a known-good model of the same architecture, research the error, get positive confirmation) is how you de-risk an edge-CV deal before committing the customer to a path. The 5-test debugging sequence here is exactly the diligence you'd do before telling a customer "yes, this will run on your hardware."

## The export paywall, and the local-training pivot

Once RF-DETR was confirmed loadable on-device, the remaining blocker was getting *our own* trained model into a bundle-able CoreML artifact. Two findings:

1. **CoreML / weights export for a custom model is a paid (Core-plan) feature.** Attempting to download our model produced `ZipExtractionError` / "No relevant model files found in ZIP" — because no CoreML export had been built for the project. The free Public tier supports **hosted-API inference** for custom models, but not the downloadable on-device artifact.
2. **The 14-day Premium Trial does *not* unlock it.** Starting the trial upgraded the plan and the "upgrade" prompt disappeared, but the weights-download still returned the plan-tier error. Export sits behind the paid tier, not the trial.

### Build-vs-buy economics
It's a pricing/packaging signal worth understanding. Roboflow gives away the expensive part (GPU training + the hosted-API endpoint) and monetises the artifact you need for **offline, zero-marginal-cost, on-device** deployment. That's a rational fence: the customers who need a bundled edge model are exactly the ones running at scale (a fleet of sprayers, thousands of devices) where per-inference hosted-API costs would dominate and a seat/plan fee is trivial. The free tier is sized for *prototyping the model*; the paid tier is sized for *shipping it to the edge*.

### The pivot: open-source rfdetr
Rather than pay for a demo — or, worse, try to intercept the paywalled download (which wouldn't work, and would be a ToS violation) — we train RF-DETR ourselves with the **open-source `rf-detr` package (Apache-2.0)** and export the model ourselves. Same architecture, our own dataset, Roboflow's own open-source tooling: $0 and fully sanctioned. Details in [`training-setup.md`](./training-setup.md).

One wrinkle the docs revealed, worth being precise about: **rfdetr exports ONNX and TFLite, not CoreML** ([export docs](https://rfdetr.roboflow.com/develop/learn/export/)), and modern `coremltools` dropped its ONNX→CoreML path. So "open-source → CoreML" isn't turnkey. **Verified in the package source:** `rfdetr/export/` ships `_onnx`, `_tensorrt`, and `_tflite` exporters — **no `_coreml` module** — and `export/main.py` describes itself verbatim as the *"CLI orchestrator for ONNX and TensorRT model export."* Roboflow's 1.6 docs do list CoreML as an export target ("Export your model — ONNX, TensorRT, CoreML") — but that refers to the **server-side conversion in the paid platform**, not the open package. So CoreML conversion is provably possible (they do it in production) — it just isn't in the free/open tier.

**And we then did it ourselves, for $0.** A direct PyTorch→CoreML trace ([`../training/export_coreml.py`](../training/export_coreml.py)) produced `rfdetr_small.mlpackage` (55 MB, fp16), **validated on the macOS CoreML runtime** (loads + predicts; named `boxes`/`logits` outputs), matching the open model's outputs bit-for-bit. It took **four targeted patches**: (1) an int/bool cast override (coremltools `int(np.array([x]))` fails on modern numpy), (2) bicubic→bilinear for the DINOv2 pos-embed interp (coremltools supports neither bicubic variant), (3) flatten meshgrid inputs to 1-D, and (4) the hard one — a **rank-safe reimplementation of `MSDeformAttn.forward`**, because deformable attention's `(N,Lq,heads,levels,points,2)` sampling tensor is rank 6 and CoreML caps at rank 5. RF-DETR Small is single-scale (`n_levels=1`), so that level dim is redundant; the rewrite drops it (numerically identical, max diff 0.0). Full detail in [`training-setup.md`](./training-setup.md#export--getting-to-onnx-and-coreml).

**The refined build-vs-buy line:** "can you?" is the wrong question — *yes*, the open stack reaches a working CoreML model that matches the paid output. The right question is "do you want to **own** it": this needed senior ML-engineering (an attention rewrite + numerical validation + version-specific converter patches) and is brittle across model config (`n_levels=1`), OS, and coremltools versions. The paid platform does that conversion turnkey and maintains it. ONNX Runtime Mobile remains the lower-effort on-device path if you'd rather not own the CoreML conversion. The GPU-training + export pipeline is in [`../training/train_colab.ipynb`](../training/train_colab.ipynb).

The meta-point: having *personally* hit the export paywall and found the open-source escape hatch, you can have an honest build-vs-buy conversation with a customer — when the platform is the right call, when self-hosting the open-source stack is, and where the line sits.

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

### When is edge actually the right call? (the decision framework)

The instinct "put the model on the device" isn't automatically correct — it's correct for a *specific* class of workload. The dividing line is **latency-coupled-to-action** and **connectivity**:

| | **Edge (on-device / on-vehicle)** | **Cloud (hosted API / batch)** |
|---|---|---|
| **Drives an action in real time?** | Yes — a sprayer nozzle fires per-plant as the boom passes over it; a scout gets an instant in-viewfinder call | No — imagery is analysed *after* collection |
| **Connectivity** | Must work with none (mid-field, no signal) | Reliable uplink available |
| **Latency budget** | Milliseconds (the plant is moving past the nozzle) | Seconds-to-minutes is fine |
| **Volume economics** | Thousands of devices → per-inference cloud cost is prohibitive; on-device is zero-marginal | Low/bursty volume → pay-per-call is cheaper than owning hardware |
| **Canonical ag use case** | Real-time precision spraying ("see-and-spray"), live crop scouting | Post-flight drone-survey analysis, yield mapping, agronomic reporting |

The clearest example comes from the plant-breeding engagement this demo is built around. Per-plant **counting** for hybrid selection was historically a *batch, cloud* workload — a tractor rig captures ~600 km of imagery per season, hauled back and processed weeks later. Re-pointing that same task to the **edge** (an in-cab Jetson counting as the rig drives) is the right move — but *not* because of millisecond latency, since counting fires no actuator. It's because:

1. **The phenology window is irreversible.** Detecting a bad/occluded row on-rig lets the operator re-drive it *that day*, rather than discovering the gap weeks later after the growth stage has passed and the count is unrecoverable.
2. **Volume economics.** Keeping the *count* instead of 600 km of raw imagery is far cheaper to move and store.
3. **Connectivity.** A field station mid-plot has no reliable uplink, so hosted-API-per-frame isn't viable regardless.

Edge wins on three of the four drivers — not the one (millisecond actuation) people reach for first. That discrimination is the point of the table above.

Real-time **weed control** — a precision-spray robot firing a nozzle per plant as the boom passes — is the case where the millisecond driver *does* dominate. It's a different objective the same trained model can serve, but it was **not** the original engagement; conflating the two is exactly the use-case imprecision this framework exists to prevent.

That shift is why an edge-CV platform story matters, and why this demo deliberately targets the on-device path rather than the (easier, free) hosted API.

## UI patterns borrowed from Roboflow's Cash Counter

The official [Cash Counter app](https://apps.apple.com/us/app/roboflow-cash-counter/id1633812788) is a useful reference for the patterns Roboflow themselves think work:

- **De-emphasized bounding boxes** — subtle highlights, not in-your-face overlays
- **Big readout of the business metric** — for Cash Counter it's total cash; here it's the **crop stand count** as the hero number, with the weed count shown but flagged as the *excluded* confounder (mirroring the breeding use case, where stand count is the decision and weeds are filtered out)
- **"Incorrect Detection?" feedback button** — our *"Upload Incorrect Image"* button, wired to `rf.uploadImage()` so user corrections flow back to the dataset
- **Status readout** — a top-centre `MODE · FPS · latency` line (e.g. `EDGE · 23 FPS · 44 ms`) for technical credibility, rather than a bare corner FPS counter; it doubles as the live performance + backend indicator

The one element that *isn't* from Cash Counter is the **Edge/Cloud segmented control** — a deliberate addition that switches the inference backend (on-device CoreML vs Roboflow hosted API) live, turning the edge-vs-cloud trade-off into something you can demonstrate on screen rather than just describe.

The Cash Counter source isn't public, but these patterns are easy to replicate on top of [`roboflow-ios-starter`](https://github.com/roboflow/roboflow-ios-starter).

## What I'd want to extend with more time

If this were a real customer engagement rather than a demo, the next-tier work would be:

- **Field-test the model** with a small custom dataset of LA-yard weeds to see how the RF100-trained baseline generalises out of distribution
- **Add temporal smoothing** — averaging detections across frames to reduce flicker
- **Capture geo-tagged inference** for per-row yield estimates rather than per-frame counts
- **Add an offline-capable model cache** so the app works in poor-signal field conditions
- **Benchmark RF-DETR vs YoloLite** on the same dataset to give the customer real comparison numbers, not just architecture claims
