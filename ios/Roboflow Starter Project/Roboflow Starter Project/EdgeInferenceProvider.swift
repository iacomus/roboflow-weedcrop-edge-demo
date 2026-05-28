import Foundation
import Vision
import CoreML

final class EdgeInferenceProvider: InferenceProvider {
    let displayName = "EDGE"

    enum EdgeError: Error { case modelMissing, badResults }

    private let confidenceThreshold: Float = 0.4
    private let classNames: [Int: String] = [1: "crop", 2: "weed"]
    private var visionModel: VNCoreMLModel?

    init() {
        if let url = Bundle.main.url(forResource: "rfdetr_small", withExtension: "mlmodelc"),
           let mlModel = try? MLModel(contentsOf: url),
           let vnModel = try? VNCoreMLModel(for: mlModel) {
            visionModel = vnModel
        }
    }

    func infer(_ pixelBuffer: CVPixelBuffer,
               completion: @escaping (Result<InferenceOutput, Error>) -> Void) {
        guard let visionModel = visionModel else {
            completion(.failure(EdgeError.modelMissing)); return
        }
        let start = DispatchTime.now()
        let threshold = confidenceThreshold
        let names = classNames
        let request = VNCoreMLRequest(model: visionModel) { req, err in
            if let err = err { completion(.failure(err)); return }
            guard let results = req.results as? [VNCoreMLFeatureValueObservation] else {
                completion(.failure(EdgeError.badResults)); return
            }
            var boxes: MLMultiArray?
            var logits: MLMultiArray?
            for o in results {
                if o.featureName == "boxes"  { boxes  = o.featureValue.multiArrayValue }
                if o.featureName == "logits" { logits = o.featureValue.multiArrayValue }
            }
            guard let b = boxes, let l = logits else {
                completion(.failure(EdgeError.badResults)); return
            }
            let dets = EdgeInferenceProvider.decode(boxes: b, logits: l,
                                                    threshold: threshold, classNames: names)
            let ms = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
            completion(.success(InferenceOutput(detections: dets, latencyMs: ms)))
        }
        // Match training preprocessing: stretch the full frame to 512x512.
        request.imageCropAndScaleOption = .scaleFill
        // .right maps the back-camera landscape buffer to an upright portrait frame.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        do { try handler.perform([request]) }
        catch { completion(.failure(error)) }
    }

    static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    /// boxes: (1,N,4) cxcywh normalized; logits: (1,N,4) pre-sigmoid.
    static func decode(boxes: MLMultiArray, logits: MLMultiArray,
                       threshold: Float, classNames: [Int: String]) -> [Detection] {
        let numQueries = boxes.shape[1].intValue
        var dets: [Detection] = []
        for q in 0..<numQueries {
            var bestId = -1
            var bestScore: Float = 0
            for c in classNames.keys {
                let s = sigmoid(logits[[0, q, c] as [NSNumber]].floatValue)
                if s > bestScore { bestScore = s; bestId = c }
            }
            guard bestId >= 0, bestScore >= threshold else { continue }
            let cx = boxes[[0, q, 0] as [NSNumber]].floatValue
            let cy = boxes[[0, q, 1] as [NSNumber]].floatValue
            let w  = boxes[[0, q, 2] as [NSNumber]].floatValue
            let h  = boxes[[0, q, 3] as [NSNumber]].floatValue
            let rect = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2),
                              width: CGFloat(w), height: CGFloat(h))
            dets.append(Detection(classId: bestId, label: classNames[bestId] ?? "?",
                                  confidence: bestScore, rect: rect))
        }
        return dets
    }
}
