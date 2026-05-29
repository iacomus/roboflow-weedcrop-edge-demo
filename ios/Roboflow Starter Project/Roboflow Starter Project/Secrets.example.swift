//  Secrets.example.swift
//
//  Copy this file to `Secrets.swift` and fill in your Roboflow API key:
//      cp "Secrets.example.swift" "Secrets.swift"
//
//  `Secrets.swift` is gitignored (it holds your private API key) — never commit it.
//  This template is NOT part of the build target (it would clash with Secrets.swift);
//  it's just a reference you copy from.
//
//  First time only: add the new `Secrets.swift` to the app target in Xcode —
//  right-click the "Roboflow Starter Project" group → Add Files… → select
//  Secrets.swift → ensure the app target is checked.

import Foundation

enum Secrets {
    /// Roboflow Private API Key — Settings → API Keys. Used for the hosted
    /// inference API (Cloud mode) and the "Upload Incorrect Image" call.
    static let apiKey = "YOUR_ROBOFLOW_API_KEY"

    /// Model/project slug and version. These are the demo's public values —
    /// change them if you point the app at your own Roboflow model.
    static let model = "weed-crop-aerial-mbyst"
    static let modelVersion = 1
}
