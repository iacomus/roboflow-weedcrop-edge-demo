import UIKit
import CoreImage
import CoreVideo

final class CloudInferenceProvider: InferenceProvider {
    let displayName = "CLOUD"

    enum CloudError: Error { case encodeFailed, badResponse, http(Int) }

    private let apiKey: String
    private let model: String
    private let version: Int
    private let session: URLSession
    private let ciContext = CIContext()

    init(apiKey: String, model: String, version: Int) {
        self.apiKey = apiKey
        self.model = model
        self.version = version
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 8
        self.session = URLSession(configuration: cfg)
    }

    func infer(_ pixelBuffer: CVPixelBuffer,
               completion: @escaping (Result<InferenceOutput, Error>) -> Void) {
        guard let base64 = CloudInferenceProvider.jpegBase64(from: pixelBuffer, context: ciContext) else {
            completion(.failure(CloudError.encodeFailed)); return
        }
        var comps = URLComponents(string: "https://serverless.roboflow.com/\(model)/\(version)")!
        comps.queryItems = [
            URLQueryItem(name: "api_key", value: apiKey),
            URLQueryItem(name: "confidence", value: "0.4"),
            URLQueryItem(name: "format", value: "json"),
        ]
        var req = URLRequest(url: comps.url!)
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        req.httpBody = base64.data(using: .utf8)

        let start = DispatchTime.now()
        session.dataTask(with: req) { data, resp, err in
            if let err = err { completion(.failure(err)); return }
            guard let http = resp as? HTTPURLResponse else {
                completion(.failure(CloudError.badResponse)); return
            }
            guard (200..<300).contains(http.statusCode) else {
                completion(.failure(CloudError.http(http.statusCode))); return
            }
            guard let data = data else { completion(.failure(CloudError.badResponse)); return }
            let dets = CloudInferenceProvider.parse(data, sentSize: CGSize(width: 512, height: 512))
            let ms = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
            completion(.success(InferenceOutput(detections: dets, latencyMs: ms)))
        }.resume()
    }

    /// Parses the serverless predictions JSON into normalized Detections.
    /// Predictions are pixel-space, center-origin, in the `sentSize` image.
    static func parse(_ data: Data, sentSize: CGSize) -> [Detection] {
        struct Resp: Decodable { let predictions: [Pred] }
        struct Pred: Decodable {
            let x: Double; let y: Double
            let width: Double; let height: Double
            let confidence: Double
            let `class`: String
        }
        guard let resp = try? JSONDecoder().decode(Resp.self, from: data) else { return [] }
        let w = sentSize.width, h = sentSize.height
        let map: [String: Int] = ["crop": 1, "weed": 2]
        return resp.predictions.compactMap { p in
            guard let id = map[p.class.lowercased()] else { return nil }
            let rect = CGRect(x: (p.x - p.width / 2) / w,
                              y: (p.y - p.height / 2) / h,
                              width: p.width / w,
                              height: p.height / h)
            return Detection(classId: id, label: p.class.lowercased(),
                             confidence: Float(p.confidence), rect: rect)
        }
    }

    /// Orient (.right) → stretch to 512x512 (matches v1 dataset preprocessing) → JPEG q0.6 → base64.
    static func jpegBase64(from pixelBuffer: CVPixelBuffer, context: CIContext) -> String? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        let extent = ci.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: 512.0 / extent.width,
                                                          y: 512.0 / extent.height))
        guard let cg = context.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: 512, height: 512)) else { return nil }
        guard let jpeg = UIImage(cgImage: cg).jpegData(compressionQuality: 0.6) else { return nil }
        return jpeg.base64EncodedString()
    }
}
