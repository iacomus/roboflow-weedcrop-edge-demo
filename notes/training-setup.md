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
- **Hyperparameters**: defaults, 100-epoch cap (early-stopped at 35 — see Results)
- **Cost**: 10.96 Roboflow credits (within the 15-credit/month Public-tier allocation — $0 out of pocket)
- **Training time**: Roboflow *estimated* 5.5 hours, but it actually finished in **~27 minutes** on their cloud GPU. The 5.5h figure is a conservative queue estimate, not actual compute.

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
4. Wait ~30 minutes (the platform quotes a conservative ~5.5h queue estimate, but actual compute was ~27 min)
5. Note the resulting mAP@50, model ID, and version for the iOS SDK `rf.load(model:modelVersion:)` call

## Results — Roboflow hosted RF-DETR Small

| Metric | Value |
|---|---|
| **Best mAP@50** | **0.7747** (epoch 18) |
| mAP@50 at final epoch (35) | 0.7160 |
| Precision / Recall / F1 (best) | 80.7% / 67.1% / 73.2% |
| Epoch cap | 100 |
| Early-stopped at | epoch 35 |
| Deployed checkpoint | **best (epoch 18)**, not the final epoch |
| Wall-clock training time | ~27 min |

### The key training-dynamics learning: peak-then-overfit

The model **peaked at epoch 18 (0.7747 mAP@50) and then got *worse***, drifting down to 0.7160 by epoch 35 before early stopping kicked in. Classic overfitting after convergence.

Two practical takeaways:

1. **Training longer ≠ better.** Bumping the epoch budget would have produced a *worse* deployed model, not a better one, if you naively took the last checkpoint. The curve is concave: it climbs, peaks, then degrades.
2. **Best-checkpoint tracking is what saves you.** Roboflow (and rfdetr) save the best-by-validation-mAP checkpoint, not the final one — so the *deployed* model is the epoch-18 peak (77.4%), even though training ran 17 more epochs past it. Early stopping (patience-based) then halts the wasted compute. This pairing — best-checkpoint save + early stopping — is why you can set a generous epoch cap and trust the system to land on the peak.

This directly informs the local-training config below: a 50-epoch cap with patience-10 early stopping comfortably brackets an epoch-18-style peak.

## Local training — the on-device path

Roboflow's **hosted training is free**, but **exporting a custom model's weights to CoreML for on-device iOS is a paid (Core-plan) feature** — and the 14-day Premium Trial does *not* unlock it (weights/CoreML export stays gated even on the trial). The free Public tier gives you hosted-API inference for custom models, but not the artifact you need to bundle in an app.

To get genuine on-device deployment at $0 and ToS-clean, we train RF-DETR ourselves with the **open-source `rf-detr` package (Apache-2.0)** and export the model ourselves for on-device use (see the *export caveat* below — rfdetr exports ONNX/TFLite natively; CoreML is a direct PyTorch→CoreML trace with ONNX Runtime as fallback). Open-source architecture + our own dataset + Roboflow's own open-source export tooling — fully sanctioned.

### Environment (Apple Silicon, M-series)

- **Python 3.12 via `uv`** — the system Python (3.14) is too new for the PyTorch / rfdetr wheels, so we pin 3.12 in an isolated venv.
- `uv pip install "rfdetr[train,loggers]" roboflow` — the `[train,loggers]` extras pull in `pytorch_lightning` + TensorBoard (a first run without them failed with `ModuleNotFoundError: No module named 'pytorch_lightning'`).
- **CPU backend** (`device="cpu"`). The first attempt used the Apple-Silicon GPU (MPS) with `PYTORCH_ENABLE_MPS_FALLBACK=1`, but it stalled hard on a single Metal op (see *Why CPU, not MPS* below). CPU is slower per step but actually makes progress.

### Why CPU, not MPS — the Metal `scatter_add` stall

The first run used `device="mps"` (Apple-Silicon GPU). It reached the epoch-0 sanity validation and then **stalled for 40+ minutes with no progress** entering the first training step. A stack sample of the wedged process showed ~100% of its time inside a *single* op:

```
scatter_add_mps_out → scatter_mps_general → MPSStream::executeMPSGraph
  → dispatch_sync_with_rethrow → MPSGraph encodeToCommandBuffer → GPU
```

Not a crash, not a deadlock, not CPU fallback — one `scatter_add` running on the GPU via a *synchronous* dispatch that took minutes. Three compounding causes:

1. **RF-DETR is Deformable-DETR-based**, so `scatter_add`/gather is on the hot path — the loss / Hungarian matching and the deformable-attention sampling call it many times per training step.
2. **Metal's weak scatter support.** Metal historically lacked native strided/non-contiguous memory access, so PyTorch's MPS backend copies tensors to contiguous temporaries around every kernel; `scatter_add` additionally needs atomic accumulation for duplicate indices, which Metal implements inefficiently. With no kernel fusion and synchronous per-op dispatch (the `dispatch_sync_with_rethrow` above), nothing amortises the cost.
3. **`PYTORCH_ENABLE_MPS_FALLBACK=1` does *not* help here.** That flag only reroutes ops with *no* MPS implementation onto the CPU. `scatter_add` *has* an MPS implementation (`scatter_add_mps_out`, visible in the sample) — so it stays on the slow GPU path instead of falling back. Being "supported" is exactly what trapped it; an *unimplemented* op would have fallen back and run fast on CPU.

(A related but separate MPS bug — silent training freezes from non-contiguous *output* tensors in Adam's `addcmul_`/`addcdiv_`, pre-PyTorch-2.4 — is a *correctness* freeze, fixed in 2.4+ / macOS 15+ with native strided support. This machine is newer, so what we hit is the residual *performance* pathology of scatter_add atomics, not that bug.)

**Fix:** in principle `device="cpu"` works (CPU's `scatter_add` is fast and well-supported), but in practice it was **multi-day slow** — a single epoch hadn't finished in 76 minutes (~6 cores pinned). The viable fix was a **cloud CUDA GPU**: [`train_colab.ipynb`](../training/train_colab.ipynb) runs the same open-source `rfdetr` on a **free Colab T4** and completed the full run in **~85 min** (see Results below).

**Export note:** rfdetr exports **ONNX** and **TFLite**, but **not CoreML** ([export docs](https://rfdetr.roboflow.com/develop/learn/export/)), and modern `coremltools` dropped its ONNX→CoreML path — so there's no one-call open-source route to CoreML. It *is* achievable, though, with a direct PyTorch→CoreML trace plus a handful of targeted patches; we did exactly that and validated it on the macOS CoreML runtime. See **Export — getting to ONNX and CoreML** below.

References: [Elana Simon — "the bug that taught me PyTorch"](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/); [Apple — Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/).

### Dataset

- Downloaded `iacomus/weed-crop-aerial-mbyst` v1 in **COCO** format via the `roboflow` package — dataset download is free; only weights/CoreML export is paywalled.
- **Roboflow's v1 preprocessing is baked into the exported images**: they arrive already auto-oriented and resized to 512×512, so the local pipeline inherits the same preprocessing the hosted model trained on. rfdetr also applies its own internal resize during training.
- Note on classes: the COCO export carries **3 categories** (a Roboflow supercategory placeholder + `crop` + `weed`), so the log shows "model is configured for 3" and re-initialises the detection head from the 90-class Objects365 pretrain to 3. This is expected for Roboflow COCO exports, not a bug.

### Config (`training/train.py`)

- **RFDETRSmall** — matches the hosted model and the narrative (not Nano).
- Transfer-learns from the same Objects365 pretrained weights (`rf-detr-small.pth`, 368 MB, auto-downloaded), detection head re-initialised to our classes.
- **50-epoch cap, early stopping (patience 10, min-delta 0.001)** — calibrated from the hosted run's epoch-18 peak so the cap comfortably brackets the peak while early stopping trims the overfit tail.
- `batch_size=4`, `grad_accum_steps=4` → effective batch 16 (safe for 16 GB unified memory).
- `checkpoint_interval=5` so there's always a recent checkpoint; best-by-val-mAP checkpoint tracked via EMA.
- 16-bit Automatic Mixed Precision (AMP) on, ~31.8 M trainable params.

### Augmentation: offline vs online

A useful distinction the demo surfaces:

- **Offline augmentation** (what Roboflow's "Augmentation" step does at *dataset-version* generation time): pre-generates augmented *copies* of images that become permanent rows in the dataset. We left this OFF for v1.
- **Online augmentation** (what rfdetr does at *training* time): applies transforms on-the-fly each epoch — multi-scale resize (`multi_scale`, `expanded_scales`), flips, etc. This is the standard modern approach and is on by default.

So the locally-trained model **is** augmented — just via the training pipeline rather than pre-baked dataset copies. The two approaches are complementary; for a small dataset, online augmentation alone is usually the right call (no dataset bloat, fresh transforms every epoch).

### Results — RF-DETR Small (free Colab T4)

The CPU run was abandoned as non-viable (one epoch unfinished after 76 min → multi-day). The shipped model was trained on a **free Google Colab T4 GPU** via [`train_colab.ipynb`](../training/train_colab.ipynb) — same open-source `rfdetr`, `batch_size=8` / `grad_accum_steps=2` (effective 16):

| Metric | Value |
|---|---|
| Wall-clock | **85.5 min** |
| Stopped | early stopping at **epoch 17** (patience 10; best regular mAP@50:95 plateaued ~epoch 9) |
| **Best mAP@50:95** | **0.497** (regular) / 0.484 (EMA) |
| **mAP@50** | ~**0.76** (0.7606 at final epoch) |
| Per-class mAP@50:95 | crop 0.471 / weed 0.436 |
| Per-epoch | ~3.7 min |

Deployed checkpoint = `checkpoint_best_ema.pth` (best-by-val-mAP, not the final epoch).

### Hosted (paid) vs free-Colab — the build-vs-buy data point

Same architecture, same dataset, trained both ways:

| | Roboflow hosted (paid) | Free Colab T4 |
|---|---|---|
| mAP@50 | **0.7747** (epoch 18) | ~0.76 |
| mAP@50:95 | ~0.46–0.47 | **0.497** |
| Wall-clock | ~27 min (~46 s/epoch) | ~85 min (~3.7 min/epoch) |
| Cost | paid tier for the CoreML/weights export | **$0** |

**Equivalent accuracy** — hosted a hair ahead on mAP@50, Colab a hair ahead on mAP@50:95, both within run-to-run / augmentation noise. The hosted run's only real edges are **speed** (~5× faster per epoch on a better GPU) and the **turnkey CoreML export**. You are *not* paying for model quality.

> **Metric note:** the platform's headline "mAP" is **mAP@50**; rfdetr's `Best EMA mAP` / `effective_map` logs are **mAP@50:95** (the stricter COCO metric). Don't compare 0.77 (@50) against 0.49 (@50:95) — different metrics.

### Export — getting to ONNX and CoreML

All on-device artifacts are produced locally from `checkpoint_best_ema.pth` (open-source `rfdetr` + Apple `coremltools`, $0):

- **ONNX** — rfdetr's native `model.export()` → `rfdetr-small.onnx` (117 MB). Validated with onnxruntime: input `(1,3,512,512)` → `dets (1,300,4)` + `labels (1,300,4)`. Runs on iOS via ONNX Runtime Mobile.
- **CoreML** — *not* a native rfdetr export, but we cracked it with a direct PyTorch→CoreML trace ([`export_coreml.py`](../training/export_coreml.py)) → `rfdetr_small.mlpackage` (55 MB, fp16). **Validated on the macOS CoreML runtime** (loads + predicts; outputs `boxes (1,300,4)` + `logits (1,300,4)`).

**Four patches made RF-DETR CoreML-convertible** — each a real gap between the open model and a CoreML graph:

1. **int/bool cast override** — coremltools' `_cast` does `int(np.array([x]))`, which numpy ≥1.25 rejects for 1-D arrays. Take element 0.
2. **bicubic → bilinear** — coremltools supports *neither* bicubic variant; the DINOv2 pos-embed interpolation re-routes to bilinear (negligible numeric impact).
3. **meshgrid flatten** — coremltools rejects rank>1 meshgrid inputs; flatten to 1-D at trace time.
4. **rank-safe deformable attention** — the hard one. CoreML caps tensors at **rank 5**, but deformable attention's `(N, Lq, heads, levels, points, 2)` sampling tensor is **rank 6**. RF-DETR Small is **single-scale (n_levels=1)**, so that level dim is redundant — we reimplemented `MSDeformAttn.forward` at rank ≤4 (reusing the exact `_bilinear_grid_sample` + conventions) and verified it's **numerically identical (max diff 0.0)** before converting.

**Refined build-vs-buy finding:** the open path *can* produce a CoreML model that matches the paid platform's output bit-for-bit — but it took senior-level ML engineering (a from-scratch attention rewrite + numerical validation + version-specific converter patches), and it's brittle across model config (relies on n_levels=1), OS, and coremltools versions. The paid platform does this conversion turnkey and maintains it. The real line isn't *"can you"* — it's *"do you want to own this conversion and its upkeep."*

### Validated on the test set (118 images)

[`test_coreml.py`](../training/test_coreml.py) runs the `.mlpackage` on the real test split via the macOS CoreML runtime:

- **Correct detections** on real weed/crop imagery — counts track ground truth (e.g., a 19-weed image → 21 detections at conf 0.40; 4-weed → 4).
- **Class-channel mapping resolved** (logits is width-4 for 2 real classes): **channel 1 = crop, channel 2 = weed**. Channels 0 (the COCO `weed-crop-aerial` supercategory placeholder) and 3 (padding) never fire. Boxes are **cxcywh, normalized to [0,1]**. This is exactly what the Swift-side decode needs (sigmoid logits → argmax over channels 1/2 → cxcywh→xyxy × image size).
- **PyTorch parity:** the meaningful detections match the original bicubic PyTorch model to ~0.01 in score (top-5 near-identical). The larger raw max|d| (boxes ~1.0, logits ~4.0) is confined to low-confidence *background* queries, so the bilinear-pos-embed + fp16 approximation is immaterial to output.

_Still TODO: iPhone 17 Pro on-device inference FPS (measure after Swift integration)._
