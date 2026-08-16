import XCTest
@testable import CRUXCore

final class AnalysisTypesTests: XCTestCase {
    func testSegmentationResultPreservesKindAndStableIDs() {
        let geometry = HoldGeometry(
            bboxX: 0.1, bboxY: 0.2, bboxWidth: 0.1, bboxHeight: 0.1,
            maskWidth: 2, maskHeight: 2,
            maskRLE: Data([Int32(0), Int32(4)].withUnsafeBytes { Data($0) })
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
            maskWidth: 1, maskHeight: 1,
            maskRLE: Data([Int32(0), Int32(1)].withUnsafeBytes { Data($0) })
        )
        let hold = DetectedHold(id: 0, kind: .hold, confidence: 0.9, geometry: geometry)
        let analysis = RouteAnalysis(
            modelVersion: "v1.0.1",
            inputSize: 648,
            colorAnalyzerVersion: "lab-mask-v1",
            routeSelectorVersion: SeededRouteSelector.version,
            holds: [AnalyzedHold(detection: hold, referenceLab: LabColor(l: 50, a: 0, b: 0))]
        )

        XCTAssertEqual(analysis.modelVersion, "v1.0.1")
        XCTAssertEqual(analysis.inputSize, 648)
        XCTAssertEqual(analysis.colorAnalyzerVersion, "lab-mask-v1")
        XCTAssertEqual(analysis.routeSelectorVersion, SeededRouteSelector.version)
        XCTAssertEqual(analysis.holds.map(\.id), [0])
    }
    func testHoldColorAnalyzerBuildsAnalysisFromCanonicalPixels() throws {
        let geometry = HoldGeometry(
            bboxX: 0, bboxY: 0, bboxWidth: 1, bboxHeight: 1,
            maskWidth: 1, maskHeight: 1,
            maskRLE: Data([Int32(0), Int32(1)].withUnsafeBytes { Data($0) })
        )
        let detection = DetectedHold(
            id: 3, kind: .hold, confidence: 0.88, geometry: geometry
        )
        let segmentation = SegmentationResult(
            modelVersion: "v1.0.1", inputSize: 648, detections: [detection]
        )

        let analysis = try HoldColorAnalyzer().analyze(
            segmentation: segmentation,
            imageWidth: 1,
            imageHeight: 1,
            imagePixels: { _, _ in SRGBColor(red: 40, green: 110, blue: 235) }
        )
        let expected = ColorMath.srgbToLab(40, 110, 235)

        XCTAssertEqual(analysis.colorAnalyzerVersion, "lab-mask-v1")
        XCTAssertEqual(analysis.holds.map(\.id), [3])
        XCTAssertEqual(analysis.routeSelectorVersion, SeededRouteSelector.version)
        XCTAssertEqual(analysis.holds[0].referenceLab.l, expected.l, accuracy: 0.5)
        XCTAssertEqual(analysis.holds[0].referenceLab.a, expected.a, accuracy: 0.5)
        XCTAssertEqual(analysis.holds[0].referenceLab.b, expected.b, accuracy: 0.5)
    }
    func testHoldColorAnalyzerRejectsInvalidImageDimensions() {
        let segmentation = SegmentationResult(modelVersion: "v1.0.1", inputSize: 648, detections: [])

        XCTAssertThrowsError(
            try HoldColorAnalyzer().analyze(
                segmentation: segmentation,
                imageWidth: 0,
                imageHeight: 1,
                imagePixels: { _, _ in nil }
            )
        ) { error in
            XCTAssertEqual(error as? HoldSegmenterError, .invalidImage)
        }
    }
    func testRouteDraftTracksCorrectionsAndManualHold() {
        let geometry = HoldGeometry(
            bboxX: 0, bboxY: 0, bboxWidth: 0.1, bboxHeight: 0.1,
            maskWidth: 1, maskHeight: 1,
            maskRLE: Data([Int32(0), Int32(1)].withUnsafeBytes { Data($0) })
        )
        let detection = DetectedHold(id: 2, kind: .hold, confidence: 0.9, geometry: geometry)
        let otherDetection = DetectedHold(id: 3, kind: .hold, confidence: 0.8, geometry: geometry)
        let analysis = RouteAnalysis(
            modelVersion: "v1.0.1",
            inputSize: 648,
            colorAnalyzerVersion: "lab-mask-v1",
            routeSelectorVersion: SeededRouteSelector.version,
            holds: [
                AnalyzedHold(detection: detection, referenceLab: LabColor(l: 50, a: 0, b: 0)),
                AnalyzedHold(detection: otherDetection, referenceLab: LabColor(l: 50, a: 0, b: 0))
            ]
        )
        let selected = RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: [2],
            selectedHoldIDs: [2]
        )
        XCTAssertEqual(selected.currentRouteHoldIDs, [2])
        XCTAssertEqual(selected.correctionCount, 0)
        XCTAssertNil(selected.markingStart(id: 3).startHoldID)

        let removed = selected.togglingDetectedHold(id: 2)
        XCTAssertEqual(removed.manualRemovedDetectedHoldIDs, [2])
        XCTAssertEqual(removed.correctionCount, 1)

        let manualID = RouteHoldID.manual(0)
        let manual = ManualHold(
            id: manualID,
            geometry: geometry,
            referenceLab: LabColor(l: 50, a: 0, b: 0)
        )
        let corrected = removed
            .togglingDetectedHold(id: 2)
            .addingManualHold(manual)
            .markingStart(id: manualID)
            .markingFinish(id: 2)
        XCTAssertEqual(corrected.correctionCount, 1)
        XCTAssertEqual(corrected.startHoldID, manualID)
        XCTAssertEqual(corrected.finishHoldID, 2)

        let removedManual = corrected.removingManualHold(id: manualID)
        XCTAssertNil(removedManual.startHoldID)
        let removedFinish = removedManual.togglingDetectedHold(id: 2)
        XCTAssertNil(removedFinish.finishHoldID)
    }
}
