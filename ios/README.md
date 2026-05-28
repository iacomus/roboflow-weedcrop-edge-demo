# iOS app

The iOS application source lives in this directory. It's built on top of [`roboflow-ios-starter`](https://github.com/roboflow/roboflow-ios-starter) with the following modifications:

- Swapped the mask-detection model for an RF-DETR weed/crop model trained on [`roboflow-100/weed-crop-aerial`](https://universe.roboflow.com/roboflow-100/weed-crop-aerial)
- Added a `CountCardView` to display per-class counts (crops, weeds, weed-pressure %)
- Added an *"Incorrect Detection?"* button wired to `rf.uploadImage()` for active-learning feedback

## Build instructions

See the [main README](../README.md#run-it-yourself) for build steps.

## Initial setup notes (for future-me)

```bash
# From the repo root
cd ios
# Clone the starter as a working base
git clone https://github.com/roboflow/roboflow-ios-starter.git .
# Then customise per the changes listed above
pod install
open *.xcworkspace
```

Don't commit your API key. The `.gitignore` at the repo root excludes `APIKey.swift` and `Secrets.swift` patterns — keep the API key in one of those.
