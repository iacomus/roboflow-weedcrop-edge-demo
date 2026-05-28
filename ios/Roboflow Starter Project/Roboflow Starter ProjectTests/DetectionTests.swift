import XCTest
@testable import Roboflow_Starter_Project

final class DetectionTests: XCTestCase {
    func test_counts_separatesCropsAndWeeds() {
        let dets = [
            Detection(classId: 1, label: "crop", confidence: 0.9, rect: .zero),
            Detection(classId: 1, label: "crop", confidence: 0.8, rect: .zero),
            Detection(classId: 2, label: "weed", confidence: 0.7, rect: .zero),
        ]
        let c = Detection.counts(from: dets)
        XCTAssertEqual(c.crops, 2)
        XCTAssertEqual(c.weeds, 1)
    }
}
