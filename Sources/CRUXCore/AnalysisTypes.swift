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
    public let routeSelectorVersion: String
    public let holds: [AnalyzedHold]

    public init(
        modelVersion: String,
        inputSize: Int,
        routeSelectorVersion: String,
        holds: [AnalyzedHold]
    ) {
        self.modelVersion = modelVersion
        self.inputSize = inputSize
        self.routeSelectorVersion = routeSelectorVersion
        self.holds = holds
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
