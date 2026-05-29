# iOS app

The iOS application source lives in this directory. It's built on top of [`roboflow-ios-starter`](https://github.com/roboflow/roboflow-ios-starter) with the following modifications:

- Swapped the mask-detection model for an RF-DETR weed/crop model trained on [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial)
- Replaced the SDK's dynamic model-load path with a **bundled CoreML model run via Vision** (the SDK is kept only for the `rf.uploadImage()` feedback call)
- Added an **Edge↔Cloud toggle** (on-device CoreML vs the Roboflow hosted API) and a `CountCardView` showing the **crop stand count** (weeds detected but excluded from the count)
- Added an *"Upload Incorrect Image"* button wired to `rf.uploadImage()` for active-learning feedback

## Build instructions

See the [main README](../README.md#run-it-yourself) for build steps.

## How this was originally scaffolded

For reference only — the customised source is already in this repo, so you do **not** need to do this to run it (see the main README's "Run it yourself"). This is just how the project was first bootstrapped from the starter:

```bash
# From the repo root, the starter was cloned as a working base, then customised:
#   git clone https://github.com/roboflow/roboflow-ios-starter.git ios
```

Don't commit your API key. Copy `Secrets.example.swift` to `Secrets.swift` (gitignored at the repo root) and put your key there.
