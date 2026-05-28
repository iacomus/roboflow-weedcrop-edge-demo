import CoreVideo

/// Result of one inference call: decoded detections plus measured latency.
struct InferenceOutput {
    let detections: [Detection]
    let latencyMs: Double
}

/// A swappable inference backend. Both edge and cloud return the same
/// `[Detection]` so the draw + count path is identical regardless of source.
protocol InferenceProvider {
    /// "EDGE" / "CLOUD" — shown in the status readout.
    var displayName: String { get }
    /// Runs detection on one frame. Calls `completion` (on an arbitrary queue)
    /// with detections + latency, or an error.
    func infer(_ pixelBuffer: CVPixelBuffer,
               completion: @escaping (Result<InferenceOutput, Error>) -> Void)
}
