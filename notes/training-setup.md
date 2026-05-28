# Model training setup

Exact record of how the model was trained, for reproducibility and for the writeup.

## Dataset

- **Source**: forked from [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial) — part of the Intel-sponsored Roboflow 100 benchmark, originally derived from Sudars et al. 2020 (peer-reviewed agricultural weed/crop dataset)
- **Fork destination**: `iacomus/weed-crop-aerial-mbyst` (the `-mbyst` suffix is Roboflow's automatic collision-avoidance — the bare `weed-crop-aerial` slug was globally taken)
- **Images**: 1,176
- **Classes**: 2 (`weed`, `crop`)
- **Task**: object detection

## Version 1 generation

- **Preprocessing**:
  - Auto-Orient: applied (EXIF-based rotation normalization)
  - Resize: **512×512, "Stretch to"** mode
- **Augmentation**: none (RF100 source is already processed; augmentation would slow training without clear benefit at this dataset size)
- **Train/Test split**: 823 train (70%) / 235 valid (20%) / 118 test (10%)
- **Version notes**: "v1: initial fork from RF100 weed-crop-aerial. Preprocessing: Auto-Orient + Resize 512×512 Stretch. No augmentation. Baseline."

## Model training

- **Architecture**: Roboflow **RF-DETR** (marked "Recommended" by the platform for this dataset)
- **Size variant**: **Small**
  - Chosen over Nano (marginal speed gain not needed) and Medium (Medium would exceed the 15-credit monthly free allocation at 21.63 credits, and is optimized for 576×576 — a mismatch with our 512×512 version)
  - Small is optimized for 512×512, matching our version-1 resize
- **Training base**: Objects365 pretrained weights (transfer learning — all RF-DETR models use these automatically)
- **Hyperparameters**: defaults, 100 epochs
- **Estimated cost**: 10.96 Roboflow credits (within the 15-credit/month Public-tier allocation — $0 out of pocket)
- **Estimated training time**: ~5.5 hours (RF-DETR is transformer-based; trains substantially longer than YOLO-family baselines)

## Why RF-DETR over the alternatives

| Architecture | iOS Swift SDK / CoreML compatible? | Notes |
|---|---|---|
| **RF-DETR** | ✅ Yes (native) | Chosen. Transformer-based, Roboflow-recommended, optimized for edge/CoreML. |
| YoloLite | ✅ Yes | Fallback if RF-DETR had failed to load on device. |
| YOLOv5 (the RF100 published baseline) | ❌ Not via the iOS SDK | The RF100 page's pre-trained model is YOLOv5; it does NOT deploy through the Swift SDK, which is why we trained our own RF-DETR rather than reusing the published checkpoint. |
| YOLO26 / YOLOv11 / Roboflow 3.0 | ⚠️ Not confirmed for iOS SDK | The SDK docs list RF-DETR, YoloLite, and Classification as CoreML-deployable. |

## Reproduce

1. Fork `roboflow-100/weed-crop-aerial` into your workspace
2. Generate a version: Auto-Orient + Resize 512×512 Stretch, no augmentation
3. Train a Model → Custom Training → RF-DETR → Small → Objects365 pretrained weights → default hyperparameters
4. Wait ~5.5 hours
5. Note the resulting mAP@50, model ID, and version for the iOS SDK `rf.load(model:modelVersion:)` call

## Results (filled after training completes — Task 11)

- mAP@50: _TODO_
- Precision: _TODO_
- Recall: _TODO_
- Actual training time: _TODO_
- Model size (CoreML export): _TODO_
- iPhone 17 Pro inference FPS: _TODO_ (measured after Task 13)
