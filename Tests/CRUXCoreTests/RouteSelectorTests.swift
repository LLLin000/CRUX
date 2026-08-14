import XCTest
@testable import CRUXCore

final class RouteSelectorTests: XCTestCase {
    /// Synthetic wall equivalent to the verified Python prototype selfcheck:
    /// 3 blue holds in a vertical route, 1 stray blue far away, 2 reds.
    /// Seed on route-blue -> must return exactly the 3 route blues.
    func testSeedSelectsSpatialGroupExcludingStraySameColor() throws {
        var imagePixels: [Int: (r: Double, g: Double, b: Double)] = [:]
        let w = 300, h = 300
        func set(_ x: Int, _ y: Int, _ c: (Double, Double, Double)) {
            for dy in -12...12 where y + dy >= 0 && y + dy < h {
                for dx in -12...12 where x + dx >= 0 && x + dx < w {
                    imagePixels[(x + dx) * h + (y + dy)] = c
                }
            }
        }
        let blue: (Double, Double, Double) = (40, 110, 235)
        let red: (Double, Double, Double) = (220, 60, 50)
        let route: [(x: Int, y: Int)] = [(150, 250), (150, 200), (150, 150)]
        let stray: (x: Int, y: Int) = (30, 30)
        let reds: [(x: Int, y: Int)] = [(240, 250), (240, 200)]
        for p in route { set(p.x, p.y, blue) }
        set(stray.x, stray.y, blue)
        for p in reds { set(p.x, p.y, red) }

        // bbox-local raster mask: 24x24 square covering each hold (full = RLE all-ones)
        // RLE counts: alternating 0/1 runs starting with 0s; all-ones = [0, N]
        func holdMaskRLE(count: Int) -> Data {
            var counts = [Int32](repeating: 0, count: 2)
            counts[0] = 0
            counts[1] = Int32(count)  // 24*24 ones
            return counts.withUnsafeBytes { Data($0) }
        }
        func geometry(_ x: Int, _ y: Int) -> HoldGeometry {
            let bboxW = 24.0 / 300.0, bboxH = 24.0 / 300.0
            return HoldGeometry(
                bboxX: Double(x) / 300.0 - bboxW / 2,
                bboxY: Double(y) / 300.0 - bboxH / 2,
                bboxWidth: bboxW, bboxHeight: bboxH,
                maskWidth: 24, maskHeight: 24,
                maskRLE: holdMaskRLE(count: 24 * 24)
            )
        }
        let holds = route.map(geometry) + [geometry(stray.x, stray.y)] + reds.map(geometry)

        let labs = SeededRouteSelector.medianLabPerHold(
            holds: holds,
            imagePixels: { x, y in imagePixels[x * h + y] },
            imageWidth: w, imageHeight: h
        )

        let selector = SeededRouteSelector()
        let picked = selector.select(seedIndex: 0, holds: holds, labs: labs)
        XCTAssertEqual(picked, [0, 1, 2], "route blues only; stray blue + reds excluded, got \(picked)")
    }

    func testMedianLabOfKnownColor() {
        // A uniform blue block must produce its exact Lab median.
        let holds = [HoldGeometry(bboxX: 0, bboxY: 0, bboxWidth: 1, bboxHeight: 1,
                                  maskWidth: 1, maskHeight: 1, maskRLE: Data([Int32(0), Int32(1)].withUnsafeBytes { Data($0) }))]
        let labs = SeededRouteSelector.medianLabPerHold(
            holds: holds,
            imagePixels: { _, _ in (40, 110, 235) },
            imageWidth: 1, imageHeight: 1
        )
        let expected = ColorMath.srgbToLab(40, 110, 235)
        XCTAssertEqual(labs[0].l, expected.l, accuracy: 0.5)
        XCTAssertEqual(labs[0].a, expected.a, accuracy: 0.5)
        XCTAssertEqual(labs[0].b, expected.b, accuracy: 0.5)
    }
}
