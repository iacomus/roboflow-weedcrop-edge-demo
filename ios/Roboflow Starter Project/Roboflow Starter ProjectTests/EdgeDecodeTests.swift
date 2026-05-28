import XCTest
import CoreML
@testable import Roboflow_Starter_Project

final class EdgeDecodeTests: XCTestCase {
    private func arr(_ values: [Float]) -> MLMultiArray {
        let a = try! MLMultiArray(shape: [1, 1, 4], dataType: .float32)
        for (i, v) in values.enumerated() { a[[0, 0, NSNumber(value: i)]] = NSNumber(value: v) }
        return a
    }

    func test_decode_picksHighestRealClass_andNormalizesBox() {
        let boxes = arr([0.5, 0.5, 0.2, 0.2])         // cxcywh
        // logits: ch0 low, ch1(crop) low, ch2(weed) high, ch3 low
        let logits = arr([-5, -2, 3, -5])
        let dets = EdgeInferenceProvider.decode(boxes: boxes, logits: logits,
                                                threshold: 0.4, classNames: [1: "crop", 2: "weed"])
        XCTAssertEqual(dets.count, 1)
        XCTAssertEqual(dets.first?.classId, 2)
        XCTAssertEqual(dets.first?.label, "weed")
        XCTAssertEqual(dets.first!.rect.minX, 0.4, accuracy: 0.0001)
        XCTAssertEqual(dets.first!.rect.width, 0.2, accuracy: 0.0001)
    }

    func test_decode_dropsBelowThreshold() {
        let boxes = arr([0.5, 0.5, 0.2, 0.2])
        let logits = arr([-5, -5, -5, -5])             // all sigmoids ~0
        let dets = EdgeInferenceProvider.decode(boxes: boxes, logits: logits,
                                                threshold: 0.4, classNames: [1: "crop", 2: "weed"])
        XCTAssertTrue(dets.isEmpty)
    }
}
