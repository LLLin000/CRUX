import XCTest
@testable import CRUXCore

final class AnalysisTypesTests: XCTestCase {
    func testSegmentationResultPreservesKindAndStableIDs() {
        let geometry = HoldGeometry(
            bboxX: 0.1, bboxY: 0.2, bboxWidth: 0.1, bboxHeight: 0.1,
            maskWidth: 2, maskHeight: 2, maskRLE: Data([0, 4])
        )
        let detections = [
            DetectedHold(id: 7, kind: .hold, confidence: 0.91, geometry: geometry),
            DetectedHold(id: 9, kind: .volume, confidence: 0.42, geometry: geometry)
        ]

        let result = SegmentationResult(
            modelVersion: "v1.0.1",
            inputSize: 648,
            detections: detections
        )

        XCTAssertEqual(result.detections.map(\.id), [7, 9])
        XCTAssertEqual(result.detections.map(\.kind), [.hold, .volume])
        XCTAssertEqual(result.detections.map(\.confidence), [0.91, 0.42])
        XCTAssertEqual(result.modelVersion, "v1.0.1")
        XCTAssertEqual(result.inputSize, 648)
    }

    func testRouteAnalysisCarriesModelAndSelectorProvenance() {
        let geometry = HoldGeometry(
            bboxX: 0, bboxY: 0, bboxWidth: 1, bboxHeight: 1,
            maskWidth: 1, maskHeight: 1, maskRLE: Data([0, 1])
        )
        let hold = DetectedHold(id: 0, kind: .hold, confidence: 0.9, geometry: geometry)
        let analysis = RouteAnalysis(
            modelVersion: "v1.0.1",
            inputSize: 648,
            routeSelectorVersion: "seeded-ds-v1",
            holds: [AnalyzedHold(detection: hold, referenceLab: LabColor(l: 50, a: 0, b: 0))]
        )

        XCTAssertEqual(analysis.modelVersion, "v1.0.1")
        XCTAssertEqual(analysis.inputSize, 648)
        XCTAssertEqual(analysis.routeSelectorVersion, "seeded-ds-v1")
        XCTAssertEqual(analysis.holds.map(\.id), [0])
    }
}
