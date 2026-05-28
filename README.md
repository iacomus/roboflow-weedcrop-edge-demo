# Weed-Crop Detection on iPhone via Roboflow

> Real-time weed/crop detection on iPhone via Roboflow — and a Solutions Architect's view of what happens to the customer conversation once dataset collection, annotation, model training, and edge deployment all become platform features instead of bespoke engineering programs.

- **Dataset**: [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial)
- **Model**: RF-DETR via Roboflow hosted training
- **Deployment**: iPhone via [`roboflow-swift`](https://github.com/roboflow/roboflow-swift)

---

## The customer story

I previously led delivery of an annotation platform for a **top-3 global crop-science company**'s plant-breeding program. Their tractor-mounted field imaging system captured **~600 km of maize-field imagery per season**, and they needed per-row plant counts to drive breeding decisions where a **3% yield improvement gates whether a hybrid advances to the next trial stage**.

They already had a working object-detection algorithm for sunflowers, but it didn't transfer to maize — overlapping plants after the 4-leaf stage, weed confusion, halogen lighting at night, varietal color variation, and edge artifacts all confounded it. They'd already evaluated and rejected Mechanical Turk; the labeling task was too complex and the turnaround too slow.

**What we built**: a 3-class human-in-the-loop annotation web app — *plants*, *weeds*, *other* — with **5-user consensus per image** and a difficulty-rating feedback loop. Java backend on OpenShift with async messaging, Angular frontend. The output was ground-truth training data to feed a custom detection algorithm.

**Total delivery**: ~12 months, multiple teams, ongoing maintenance burden. Three growth time points: 2-leaf (germination check), 4-leaf (plant stand established), 6-8 leaf (yield prediction). Target dataset: 3,000 images, 1,000 per time point.

---

## What this collapses to on Roboflow today

The same use case now lands in a few hours of work — because the two most expensive things in the original project (**collecting and labeling the dataset** and **training a custom algorithm from scratch**) are off-the-shelf:

1. **The dataset is open source.** Forked [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial) — 1,176 pre-labeled aerial weed/crop images derived from peer-reviewed agritech research ([Sudars et al. 2020](https://www.sciencedirect.com/science/article/pii/S2352340920307277)), shipped as part of the Intel-sponsored [Roboflow 100 benchmark](https://github.com/roboflow-ai/roboflow-100-benchmark). **No annotation project needed.**
2. **The model architectures are pre-built and edge-ready.** Trained an **RF-DETR** variant via Roboflow hosted training — chosen over YOLOv5 for native CoreML compatibility and better on-device latency. The customer's original problem statement was *"build us a maize-counting algorithm from scratch."* The platform-era equivalent is *"pick a variant from the library."*
3. **Deployment is an SDK call.** iPhone 13 Pro via the [`roboflow-swift`](https://github.com/roboflow/roboflow-swift) SDK, built on top of [`roboflow-ios-starter`](https://github.com/roboflow/roboflow-ios-starter).
4. **The labeling loop inverts.** Added a count-card overlay (crops/weeds/weed-pressure %) and an *"Incorrect Detection?"* button wired to `rf.uploadImage()` — so the field scout *using* the app becomes the labeler *in the flow of work*, replacing the original project's centralized 5-user consensus labeling.

## Production deployment target

The demo runs on iPhone because it's the most accessible edge target for a portfolio piece. In a real customer engagement, the production deployment would land somewhere along this spectrum:

- **Jetson Orin Nano / NX on a multirotor agricultural drone** — area surveys at flight pace, on-board inference for in-flight decisions
- **Jetson on tractor-cabin compute** — the modern equivalent of the customer's original tractor-mounted imaging rig, with inference moving from offline batch processing to real-time on-board
- **On-board inference module on a precision-spray robot** — per-row decisions about whether to fire the herbicide nozzle (the John Deere See & Spray family architecture)

The same trained model targets all of the above via Roboflow's deployment paths (CoreML for iOS, ONNX/TensorRT for Jetson, Inference Server for x86). Picking the demo target is itself an SA conversation: *"show me the deployment that matches the customer's actual hardware budget and operational pattern."*

Worth flagging the hardware evolution underneath: agricultural aerial imagery was historically dominated by **fixed-wing UAVs** (longer flight time, larger area coverage); robust **multirotor agricultural drones** with multispectral payloads and the Jetson-class on-board compute they need are a much more recent capability. The platform shift isn't just "better software" — it's better software running on a new generation of edge hardware that genuinely didn't exist for this use case until the past few years.

**Total build wall-clock**: roughly _TODO: actual hours_ across 2 days.

| Metric | Value |
|---|---|
| Dataset size | 1,176 images, 2 classes |
| Architecture | RF-DETR _(TODO: variant)_ |
| Training time | _TODO: minutes_ (Roboflow hosted) |
| Reported mAP@50 | _TODO: %_ |
| iPhone 13 Pro inference | _TODO: FPS_ |
| Model size (CoreML) | _TODO: MB_ |

---

## Demo

_TODO: insert demo video link or embedded clip_

![Live detection on iPhone](assets/screenshots/detection-live.png)
![Count card overlay](assets/screenshots/count-card.png)
![Incorrect-detection feedback](assets/screenshots/incorrect-detection-button.png)

---

## What this changes about the Solutions Architect conversation

Three reframings that come out of building this:

1. **The biggest cost line in the original engagement was data, not modeling.** Collecting raw imagery, building the annotation tool, running the 5-user-consensus labeling project — that was the bulk of the 12 months. The model was downstream. Today the open-source dataset library + the in-app *"Incorrect Detection?"* feedback button collapse both ends of the data lifecycle. That's a different customer business case to discover against, not just a different tech stack.
2. **Edge isn't a deployment afterthought; it's an architecture decision upstream.** RF-DETR vs. YOLOv5 isn't an academic comparison — it determines whether the model can ship to iOS via the Swift SDK, to a Jetson via TensorRT, or only via the hosted API. As an SA, surfacing this trade-off in early discovery (not at deployment time) is what de-risks the deal.
3. **The platform replaces the *workflow that wraps the model*, not the model itself.** The model is commodity; the labeling tooling, hosted training, edge runtime, and data-feedback loop are what the customer is actually buying. Selling Roboflow well means leading with workflow, not weights.

---

## Run it yourself

```bash
# 1. Clone
git clone https://github.com/iacomus/roboflow-weedcrop-edge-demo.git
cd roboflow-weedcrop-edge-demo/ios

# 2. Install pods
pod install

# 3. Add your Roboflow API key
# Edit ViewController.swift and set API_KEY = "YOUR_KEY"

# 4. Open the workspace
open WeedCropDetector.xcworkspace

# 5. In Xcode: select your team under Signing & Capabilities,
#    plug in an iPhone (iOS 15.4+), Build & Run
```

You'll need: Xcode, CocoaPods, a free Apple ID for signing, and a Roboflow account.

---

## Architecture decisions

See [`notes/architecture-choices.md`](notes/architecture-choices.md) for the rationale on:

- RF-DETR vs YOLOv5 vs YoloLite (iOS deployment compatibility)
- Why use a forked RF100 dataset rather than the active workspace version
- Trade-offs between on-device inference and the serverless hosted API
- Why iPhone rather than Raspberry Pi or Jetson for this demo

The customer-story background is in [`notes/customer-story-background.md`](notes/customer-story-background.md).

---

---

James Al-Khatib · [LinkedIn](https://www.linkedin.com/in/jameskhatib) · [GitHub](https://github.com/iacomus)
