import XCTest
@testable import Roboflow_Starter_Project

final class CloudParseTests: XCTestCase {
    func test_parse_normalizesCenterPixelBoxes_andMapsClasses() {
        let json = """
        {"predictions":[
          {"x":256,"y":256,"width":128,"height":64,"confidence":0.9,"class":"crop"},
          {"x":128,"y":384,"width":64,"height":64,"confidence":0.7,"class":"weed"},
          {"x":10,"y":10,"width":4,"height":4,"confidence":0.5,"class":"background"}
        ]}
        """.data(using: .utf8)!

        let dets = CloudInferenceProvider.parse(json, sentSize: CGSize(width: 512, height: 512))

        // "background" is not crop/weed → dropped.
        XCTAssertEqual(dets.count, 2)

        let crop = dets[0]
        XCTAssertEqual(crop.classId, 1)
        XCTAssertEqual(crop.rect.minX, 0.375, accuracy: 0.0001)   // (256-64)/512
        XCTAssertEqual(crop.rect.minY, 0.4375, accuracy: 0.0001)  // (256-32)/512
        XCTAssertEqual(crop.rect.width, 0.25, accuracy: 0.0001)   // 128/512
        XCTAssertEqual(crop.rect.height, 0.125, accuracy: 0.0001) // 64/512

        XCTAssertEqual(dets[1].classId, 2)
    }

    func test_parse_returnsEmptyOnGarbage() {
        XCTAssertTrue(CloudInferenceProvider.parse(Data("nope".utf8), sentSize: CGSize(width: 512, height: 512)).isEmpty)
    }
}
