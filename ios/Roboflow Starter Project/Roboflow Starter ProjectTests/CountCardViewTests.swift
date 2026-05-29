import XCTest
@testable import Roboflow_Starter_Project

final class CountCardViewTests: XCTestCase {
    func test_update_setsHeroCropAndExcludedWeeds() {
        let card = CountCardView(frame: .zero)
        card.update(crops: 12, weeds: 5)
        XCTAssertEqual(card.cropValueLabel.text, "12")
        XCTAssertEqual(card.cropCaptionLabel.text, "🌱 Crops detected")
        XCTAssertEqual(card.weedLabel.text, "🌾 Weeds 5 · excluded")
    }

    func test_showEmpty_blanksValues() {
        let card = CountCardView(frame: .zero)
        card.update(crops: 9, weeds: 1)
        card.showEmpty()
        XCTAssertEqual(card.cropValueLabel.text, "—")
    }
}
