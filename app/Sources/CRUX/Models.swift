// CRUX — SwiftData schema v1 (PLAN v1.2.1 §5)
// Coordinate spaces (PLAN §4): bbox/centroid = canonical normalized [0,1];
// maskRLE = bbox-local raster, maskWidth/maskHeight explicit.
// License chain: Swift code, Apache-2.0-friendly (no AGPL deps).

import Foundation
import SwiftData

enum GradeSystem: String, Codable {
    case vScale   // V0–V17
    case font     // Font (French) scale — future
}

enum RouteColor: String, Codable, CaseIterable {
    case blue, red, yellow, green, purple, black, gray, orange
}

enum RouteResult: String, Codable {
    case flash    // first try
    case top      // completed, multiple tries
    case project  // not yet sent
}

@Model
final class ClimbRoute {
    // gym = map-POI snapshot (MKLocalSearch), no Gym entity (PLAN v1.2.1)
    var name: String
    var gymNameSnapshot: String
    var gymLatitude: Double?
    var gymLongitude: Double?
    var gymMapItemID: String?

    var gradeSystem: GradeSystem
    var gradeValue: String          // "V4"

    // color: algorithm fact + UI label, kept separate (PLAN §5)
    var referenceL: Double
    var referenceA: Double
    var referenceB: Double
    var paletteColor: RouteColor

    var result: RouteResult         // user-chosen, never derived
    var attempts: Int
    var durationSeconds: Int
    var note: String
    var date: Date
    var photoFilename: String?      // Documents/Photos/<UUID>.jpg (canonical sRGB)

    // model provenance (PLAN v1.2.1)
    var segmenterModelVersion: String?
    var routeSelectorVersion: String?
    var inferenceInputSize: Int?
    var manualAdditions: Int
    var manualRemovals: Int

    @Relationship(deleteRule: .cascade, inverse: \Hold.route)
    var holds: [Hold]

    @Relationship(deleteRule: .cascade, inverse: \RouteUnionMask.route)
    var unionMasks: [RouteUnionMask]

    init(
        name: String,
        gymNameSnapshot: String,
        gymLatitude: Double? = nil,
        gymLongitude: Double? = nil,
        gymMapItemID: String? = nil,
        gradeSystem: GradeSystem = .vScale,
        gradeValue: String = "V4",
        referenceL: Double = 0, referenceA: Double = 0, referenceB: Double = 0,
        paletteColor: RouteColor = .blue,
        result: RouteResult = .top,
        attempts: Int = 1,
        durationSeconds: Int = 0,
        note: String = "",
        date: Date = .now,
        photoFilename: String? = nil,
        segmenterModelVersion: String? = nil,
        routeSelectorVersion: String? = nil,
        inferenceInputSize: Int? = nil,
        manualAdditions: Int = 0,
        manualRemovals: Int = 0
    ) {
        self.name = name
        self.gymNameSnapshot = gymNameSnapshot
        self.gymLatitude = gymLatitude
        self.gymLongitude = gymLongitude
        self.gymMapItemID = gymMapItemID
        self.gradeSystem = gradeSystem
        self.gradeValue = gradeValue
        self.referenceL = referenceL
        self.referenceA = referenceA
        self.referenceB = referenceB
        self.paletteColor = paletteColor
        self.result = result
        self.attempts = attempts
        self.durationSeconds = durationSeconds
        self.note = note
        self.date = date
        self.photoFilename = photoFilename
        self.segmenterModelVersion = segmenterModelVersion
        self.routeSelectorVersion = routeSelectorVersion
        self.inferenceInputSize = inferenceInputSize
        self.manualAdditions = manualAdditions
        self.manualRemovals = manualRemovals
        self.holds = []
        self.unionMasks = []
    }
}

@Model
final class Hold {
    // canonical normalized (relative to canonical sRGB original image)
    var centroidX: Double
    var centroidY: Double
    var bboxX: Double
    var bboxY: Double
    var bboxWidth: Double
    var bboxHeight: Double

    // bbox-local raster mask (COCO-style RLE encoded, PLAN v1.2.1)
    var maskWidth: Int
    var maskHeight: Int
    var maskRLE: Data

    var isStart: Bool     // user-marked, multiple starts allowed
    var isFinish: Bool

    var route: ClimbRoute?

    init(
        centroidX: Double, centroidY: Double,
        bboxX: Double, bboxY: Double, bboxWidth: Double, bboxHeight: Double,
        maskWidth: Int, maskHeight: Int, maskRLE: Data,
        isStart: Bool = false, isFinish: Bool = false
    ) {
        self.centroidX = centroidX
        self.centroidY = centroidY
        self.bboxX = bboxX
        self.bboxY = bboxY
        self.bboxWidth = bboxWidth
        self.bboxHeight = bboxHeight
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
        self.maskRLE = maskRLE
        self.isStart = isStart
        self.isFinish = isFinish
    }
}

@Model
final class RouteUnionMask {
    var rasterWidth: Int    // full-image raster, canonical aspect (e.g. 512 wide)
    var rasterHeight: Int
    var maskRLE: Data
    var route: ClimbRoute?

    init(rasterWidth: Int, rasterHeight: Int, maskRLE: Data) {
        self.rasterWidth = rasterWidth
        self.rasterHeight = rasterHeight
        self.maskRLE = maskRLE
    }
}
