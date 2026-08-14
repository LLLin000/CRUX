// CRUX color math — sRGB -> Lab (D65) and CIEDE2000 (ΔE00).
// Plan D5: all Lab computed from the canonical sRGB image; ΔE00 for color matching.
// Pure Foundation, unit-testable. Standard formulas (CIE 15, Sharma et al. 2005).

import Foundation

enum ColorMath {
    // MARK: sRGB -> Lab (D65 white point)

    static func srgbToLab(_ r: Double, _ g: Double, _ b: Double) -> (l: Double, a: Double, b: Double) {
        func lin(_ c: Double) -> Double {
            let c01 = c / 255.0
            return c01 > 0.04045 ? pow((c01 + 0.055) / 1.055, 2.4) : c01 / 12.92
        }
        let rl = lin(r), gl = lin(g), bl = lin(b)

        // sRGB D65 matrix (IEC 61966-2-1)
        let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
        let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
        let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

        let xn = 0.95047, yn = 1.0, zn = 1.08883
        func f(_ t: Double) -> Double {
            let delta = 6.0 / 29.0
            return t > delta * delta * delta ? cbrt(t) : t / (3 * delta * delta) + 4.0 / 29.0
        }
        let fx = f(x / xn), fy = f(y / yn), fz = f(z / zn)
        return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }

    // MARK: CIEDE2000 (Sharma, Wu & Dalal 2005)

    /// Perceptual delta between two Lab colors (std observer, kL=kC=kH=1).
    static func deltaE2000(_ l1: Double, _ a1: Double, _ b1: Double,
                           _ l2: Double, _ a2: Double, _ b2: Double) -> Double {
        let kL = 1.0, kC = 1.0, kH = 1.0
        let c1 = sqrt(a1 * a1 + b1 * b1), c2 = sqrt(a2 * a2 + b2 * b2)
        let cBar = (c1 + c2) / 2
        let g = 0.5 * (1 - sqrt(pow(cBar, 7) / (pow(cBar, 7) + pow(25.0, 7))))
        let a1p = (1 + g) * a1, a2p = (1 + g) * a2
        let c1p = sqrt(a1p * a1p + b1 * b1), c2p = sqrt(a2p * a2p + b2 * b2)

        func h(_ ap: Double, _ b: Double) -> Double {
            let hDeg = atan2(b, ap) * 180 / .pi
            return hDeg >= 0 ? hDeg : hDeg + 360
        }
        let h1p = h(a1p, b1), h2p = h(a2p, b2)
        let dLp = l2 - l1
        let dCp = c2p - c1p
        var dhp: Double
        if c1p * c2p == 0 {
            dhp = 0
        } else if abs(h2p - h1p) <= 180 {
            dhp = h2p - h1p
        } else if h2p - h1p > 180 {
            dhp = h2p - h1p - 360
        } else {
            dhp = h2p - h1p + 360
        }
        let dHp = 2 * sqrt(c1p * c2p) * sin(dhp / 2 * .pi / 180)

        let lBarP = (l1 + l2) / 2
        let cBarP = (c1p + c2p) / 2
        var hBarP: Double
        if c1p * c2p == 0 {
            hBarP = h1p + h2p
        } else if abs(h1p - h2p) <= 180 {
            hBarP = (h1p + h2p) / 2
        } else if h1p + h2p < 360 {
            hBarP = (h1p + h2p + 360) / 2
        } else {
            hBarP = (h1p + h2p - 360) / 2
        }
        let t = 1 - 0.17 * cos((hBarP - 30) * .pi / 180)
            + 0.24 * cos((2 * hBarP) * .pi / 180)
            + 0.32 * cos((3 * hBarP + 6) * .pi / 180)
            - 0.20 * cos((4 * hBarP - 63) * .pi / 180)
        let dTheta = 30 * exp(-pow((hBarP - 275) / 25, 2))
        let rc = 2 * sqrt(pow(cBarP, 7) / (pow(cBarP, 7) + pow(25.0, 7)))
        let sl = 1 + (0.015 * pow(lBarP - 50, 2)) / sqrt(20 + pow(lBarP - 50, 2))
        let sc = 1 + 0.045 * cBarP
        let sh = 1 + 0.015 * cBarP * t
        let rt = -sin(2 * dTheta * .pi / 180) * rc

        return sqrt(pow(dLp / (kL * sl), 2) + pow(dCp / (kC * sc), 2)
                    + pow(dHp / (kH * sh), 2)
                    + rt * (dCp / (kC * sc)) * (dHp / (kH * sh)))
    }
}
