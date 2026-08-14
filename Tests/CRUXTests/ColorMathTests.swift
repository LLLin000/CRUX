import XCTest
@testable import CRUX

final class ColorMathTests: XCTestCase {
    func testSRGBToLabKnownValue() {
        // sRGB(255,255,255) -> L=100, a≈0, b≈0 (D65)
        let white = ColorMath.srgbToLab(255, 255, 255)
        XCTAssertEqual(white.l, 100.0, accuracy: 0.1)
        XCTAssertEqual(white.a, 0.0, accuracy: 0.1)
        XCTAssertEqual(white.b, 0.0, accuracy: 0.1)

        // sRGB(0,0,0) -> L=0
        let black = ColorMath.srgbToLab(0, 0, 0)
        XCTAssertEqual(black.l, 0.0, accuracy: 0.1)

        // sRGB(128,128,128) mid gray -> L≈53.6 (D65)
        let gray = ColorMath.srgbToLab(128, 128, 128)
        XCTAssertEqual(gray.l, 53.6, accuracy: 0.5)
    }

    func testDeltaE2000ReferencePairs() {
        // Sharma et al. 2005 test pairs (standard dataset, kL=kC=kH=1)
        let pairs: [(Double, Double, Double, Double, Double, Double, Double)] = [
            // (L1,a1,b1, L2,a2,b2, expected ΔE00)
            (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
            (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
            (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
            (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
            (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
            (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
        ]
        for (l1, a1, b1, l2, a2, b2, expected) in pairs {
            let d = ColorMath.deltaE2000(l1, a1, b1, l2, a2, b2)
            XCTAssertEqual(d, expected, accuracy: 0.0001,
                           "pair (\(l1),\(a1),\(b1)) vs (\(l2),\(a2),\(b2))")
        }
    }

    func testDeltaE2000IdenticalColorsIsZero() {
        let d = ColorMath.deltaE2000(46.2, 17.8, -51.4, 46.2, 17.8, -51.4)
        XCTAssertEqual(d, 0.0, accuracy: 1e-9)
    }
}
