import Foundation

/// The one image coordinate system shared by inference, color sampling, and rendering.
public struct CanonicalImage: Sendable, Equatable {
    public let data: Data
    public let width: Int
    public let height: Int

    public init(data: Data, width: Int, height: Int) {
        self.data = data
        self.width = width
        self.height = height
    }
}

public enum HoldKind: String, Codable, Sendable {
    case hold
    case volume
}

/// A bbox-local mask mapped to canonical normalized image coordinates.
public struct HoldGeometry: Sendable, Equatable {
    public let bboxX: Double
    public let bboxY: Double
    public let bboxWidth: Double
    public let bboxHeight: Double
    public let maskWidth: Int
    public let maskHeight: Int
    public let maskRLE: Data

    public init(
        bboxX: Double,
        bboxY: Double,
        bboxWidth: Double,
        bboxHeight: Double,
        maskWidth: Int,
        maskHeight: Int,
        maskRLE: Data
    ) {
        self.bboxX = bboxX
        self.bboxY = bboxY
        self.bboxWidth = bboxWidth
        self.bboxHeight = bboxHeight
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
        self.maskRLE = maskRLE
    }

    public var centroid: (x: Double, y: Double) {
        (bboxX + bboxWidth / 2, bboxY + bboxHeight / 2)
    }
}

public struct LabColor: Sendable, Equatable {
    public let l: Double
    public let a: Double
    public let b: Double

    public init(l: Double, a: Double, b: Double) {
        self.l = l
        self.a = a
        self.b = b
    }
}

public struct SRGBColor: Sendable, Equatable {
    public let red: Double
    public let green: Double
    public let blue: Double

    public init(red: Double, green: Double, blue: Double) {
        self.red = red
        self.green = green
        self.blue = blue
    }
}

/// Computes the canonical-image color reference for each detected hold.
public struct HoldColorAnalyzer: Sendable {
    public let version: String

    public init(version: String = "lab-mask-v1") {
        self.version = version
    }

    public func analyze(
        segmentation: SegmentationResult,
        imageWidth: Int,
        imageHeight: Int,
        imagePixels: (Int, Int) -> SRGBColor?
    ) throws -> RouteAnalysis {
        guard imageWidth > 0, imageHeight > 0, segmentation.inputSize > 0 else {
            throw HoldSegmenterError.invalidImage
        }

        let labs = SeededRouteSelector.medianLabPerHold(
            holds: segmentation.detections.map(\.geometry),
            imagePixels: { x, y in
                guard let pixel = imagePixels(x, y) else { return nil }
                return (pixel.red, pixel.green, pixel.blue)
            },
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
        guard labs.count == segmentation.detections.count else {
            throw HoldSegmenterError.invalidOutput
        }

        return RouteAnalysis(
            modelVersion: segmentation.modelVersion,
            inputSize: segmentation.inputSize,
            colorAnalyzerVersion: version,
            routeSelectorVersion: SeededRouteSelector.version,
            holds: zip(segmentation.detections, labs).map {
                AnalyzedHold(detection: $0.0, referenceLab: $0.1)
            }
        )
    }
}

public struct DetectedHold: Sendable, Equatable, Identifiable {
    public let id: Int
    public let kind: HoldKind
    public let confidence: Float
    public let geometry: HoldGeometry

    public init(id: Int, kind: HoldKind, confidence: Float, geometry: HoldGeometry) {
        self.id = id
        self.kind = kind
        self.confidence = confidence
        self.geometry = geometry
    }
}

public struct SegmentationResult: Sendable, Equatable {
    public let modelVersion: String
    public let inputSize: Int
    public let detections: [DetectedHold]

    public init(modelVersion: String, inputSize: Int, detections: [DetectedHold]) {
        self.modelVersion = modelVersion
        self.inputSize = inputSize
        self.detections = detections
    }
}

public struct AnalyzedHold: Sendable, Equatable, Identifiable {
    public let detection: DetectedHold
    public let referenceLab: LabColor

    public var id: Int { detection.id }

    public init(detection: DetectedHold, referenceLab: LabColor) {
        self.detection = detection
        self.referenceLab = referenceLab
    }
}

public struct RouteAnalysis: Sendable, Equatable {
    public let modelVersion: String
    public let inputSize: Int
    public let colorAnalyzerVersion: String
    public let routeSelectorVersion: String
    public let holds: [AnalyzedHold]

    public init(
        modelVersion: String,
        inputSize: Int,
        colorAnalyzerVersion: String,
        routeSelectorVersion: String,
        holds: [AnalyzedHold]
    ) {
        self.modelVersion = modelVersion
        self.inputSize = inputSize
        self.colorAnalyzerVersion = colorAnalyzerVersion
        self.routeSelectorVersion = routeSelectorVersion
        self.holds = holds
    }
}

public enum RouteHoldID {
    public static func manual(_ ordinal: Int) -> Int {
        precondition(ordinal >= 0)
        return -(ordinal + 1)
    }

    public static func isManual(_ id: Int) -> Bool {
        id < 0
    }
}

public struct ManualHold: Sendable, Equatable, Identifiable {
    public let id: Int
    public let geometry: HoldGeometry
    public let referenceLab: LabColor

    public init(id: Int, geometry: HoldGeometry, referenceLab: LabColor) {
        precondition(RouteHoldID.isManual(id))
        self.id = id
        self.geometry = geometry
        self.referenceLab = referenceLab
    }
}

/// Mutable-by-replacement route selection state used by the correction UI.
public struct RouteDraft: Sendable, Equatable {
    public let analysis: RouteAnalysis
    public let initialSelectedHoldIDs: Set<Int>
    public let selectedHoldIDs: Set<Int>
    public let manualAdditions: [ManualHold]
    public let startHoldID: Int?
    public let finishHoldID: Int?

    public init(
        analysis: RouteAnalysis,
        initialSelectedHoldIDs: Set<Int> = [],
        selectedHoldIDs: Set<Int> = [],
        manualAdditions: [ManualHold] = [],
        startHoldID: Int? = nil,
        finishHoldID: Int? = nil
    ) {
        let detectedIDs = Set(analysis.holds.map(\.id))
        let initialIDs = initialSelectedHoldIDs.intersection(detectedIDs)
        let selectedIDs = selectedHoldIDs.intersection(detectedIDs)
        let currentIDs = selectedIDs.union(manualAdditions.map(\.id))
        self.analysis = analysis
        self.initialSelectedHoldIDs = initialIDs
        self.selectedHoldIDs = selectedIDs
        self.manualAdditions = manualAdditions
        self.startHoldID = startHoldID.flatMap { currentIDs.contains($0) ? $0 : nil }
        self.finishHoldID = finishHoldID.flatMap { currentIDs.contains($0) ? $0 : nil }
    }

    public var manualAddedDetectedHoldIDs: Set<Int> {
        selectedHoldIDs.subtracting(initialSelectedHoldIDs)
    }

    public var manualRemovedDetectedHoldIDs: Set<Int> {
        initialSelectedHoldIDs.subtracting(selectedHoldIDs)
    }

    public var correctionCount: Int {
        manualAddedDetectedHoldIDs.count
            + manualRemovedDetectedHoldIDs.count
            + manualAdditions.count
    }

    public var currentRouteHoldIDs: Set<Int> {
        selectedHoldIDs.union(manualAdditions.map(\.id))
    }

    public func selecting(seedIndex: Int, selector: SeededRouteSelector = .init()) -> RouteDraft {
        let holds = analysis.holds.map(\.detection.geometry)
        let labs = analysis.holds.map(\.referenceLab)
        let selectedIndices = selector.select(seedIndex: seedIndex, holds: holds, labs: labs)
        let selected = Set(selectedIndices.compactMap { index in
            analysis.holds.indices.contains(index) ? analysis.holds[index].id : nil
        })
        return RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: selected,
            selectedHoldIDs: selected,
            manualAdditions: manualAdditions,
            startHoldID: startHoldID,
            finishHoldID: finishHoldID
        )
    }

    public func togglingDetectedHold(id: Int) -> RouteDraft {
        guard analysis.holds.contains(where: { $0.id == id }) else { return self }
        var selected = selectedHoldIDs
        if !selected.insert(id).inserted {
            selected.remove(id)
        }
        return RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: initialSelectedHoldIDs,
            selectedHoldIDs: selected,
            manualAdditions: manualAdditions,
            startHoldID: startHoldID,
            finishHoldID: finishHoldID
        )
    }

    public func addingManualHold(_ hold: ManualHold) -> RouteDraft {
        guard !manualAdditions.contains(where: { $0.id == hold.id }) else { return self }
        return RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: initialSelectedHoldIDs,
            selectedHoldIDs: selectedHoldIDs,
            manualAdditions: manualAdditions + [hold],
            startHoldID: startHoldID,
            finishHoldID: finishHoldID
        )
    }

    public func removingManualHold(id: Int) -> RouteDraft {
        let additions = manualAdditions.filter { $0.id != id }
        guard additions.count != manualAdditions.count else { return self }
        return RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: initialSelectedHoldIDs,
            selectedHoldIDs: selectedHoldIDs,
            manualAdditions: additions,
            startHoldID: startHoldID,
            finishHoldID: finishHoldID
        )
    }

    public func markingStart(id: Int?) -> RouteDraft {
        if let id, !containsHold(id) { return self }
        return RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: initialSelectedHoldIDs,
            selectedHoldIDs: selectedHoldIDs,
            manualAdditions: manualAdditions,
            startHoldID: id,
            finishHoldID: finishHoldID
        )
    }

    public func markingFinish(id: Int?) -> RouteDraft {
        if let id, !containsHold(id) { return self }
        return RouteDraft(
            analysis: analysis,
            initialSelectedHoldIDs: initialSelectedHoldIDs,
            selectedHoldIDs: selectedHoldIDs,
            manualAdditions: manualAdditions,
            startHoldID: startHoldID,
            finishHoldID: id
        )
    }

    private func containsHold(_ id: Int) -> Bool {
        currentRouteHoldIDs.contains(id)
    }
}

public enum HoldSegmenterError: Error, Equatable, Sendable {
    case modelUnavailable
    case invalidImage
    case invalidOutput
    case inferenceFailed
}

/// Deep seam for on-device model adapters. The caller never sees tensors or providers.
public protocol HoldSegmenter: Sendable {
    var modelVersion: String { get }
    var inputSize: Int { get }
    func segment(_ image: CanonicalImage) async throws -> SegmentationResult
}
