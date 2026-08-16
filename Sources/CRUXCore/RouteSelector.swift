// CRUX route selection — PLAN D5/D6 core interaction (Swift port of verified
// tools/route_selector.py prototype): mask Lab median -> seed ΔE00 match ->
// DBSCAN spatial grouping -> default group containing the seed.
// Heuristics (deltaE threshold, eps factor) are INITIAL, calibrated vs RouteBenchmark in D1/D2.

import Foundation

struct HoldGeometry {
    /// bbox-local raster mask; coordinates are canonical normalized (PLAN §4)
    let bboxX: Double, bboxY: Double, bboxWidth: Double, bboxHeight: Double
    let maskWidth: Int, maskHeight: Int
    let maskRLE: Data

    var centroid: (x: Double, y: Double) {
        (bboxX + bboxWidth / 2, bboxY + bboxHeight / 2)
    }
}

struct SeededRouteSelector {
    let deltaEThreshold: Double   // initial heuristic ~5–10, calibrate in D1
    let dbscanEpsFactor: Double   // × median hold diameter (2.5 verified in prototype)
    let minSamples: Int           // 2 = noise rejection; 1 ≡ connected components

    init(deltaEThreshold: Double = 8.0,
         dbscanEpsFactor: Double = 2.5,
         minSamples: Int = 2) {
        self.deltaEThreshold = deltaEThreshold
        self.dbscanEpsFactor = dbscanEpsFactor
        self.minSamples = minSamples
    }

    /// Median Lab per hold from the canonical sRGB image pixels inside each mask.
    static func medianLabPerHold(holds: [HoldGeometry],
                                 imagePixels: (Int, Int) -> (r: Double, g: Double, b: Double)?,
                                 imageWidth: Int, imageHeight: Int) -> [(l: Double, a: Double, b: Double)] {
        holds.map { h in
            let labs = sampleMask(h, pixels: imagePixels, w: imageWidth, hgt: imageHeight)
                .map { ColorMath.srgbToLab($0.r, $0.g, $0.b) }
            guard !labs.isEmpty else { return (50, 0, 0) }
            // median of each channel (mask pixels are spatially independent)
            let sorted = { (keyPath: KeyPath<(l: Double, a: Double, b: Double), Double>) -> Double in
                let v = labs.map { $0[keyPath: keyPath] }.sorted()
                let mid = v.count / 2
                return v.count % 2 == 1 ? v[mid] : (v[mid - 1] + v[mid]) / 2
            }
            return (sorted(\.l), sorted(\.a), sorted(\.b))
        }
    }

    /// Select the group containing the seed (by index into `holds`).
    func select(seedIndex: Int, holds: [HoldGeometry],
                labs: [(l: Double, a: Double, b: Double)]) -> Set<Int> {
        guard holds.indices.contains(seedIndex) else { return [] }
        let seedLab = labs[seedIndex]
        let candidates = holds.indices.filter { i in
            ColorMath.deltaE2000(seedLab.l, seedLab.a, seedLab.b,
                                 labs[i].l, labs[i].a, labs[i].b) < deltaEThreshold
        }
        guard candidates.count > 1 else { return candidates.isEmpty ? [] : [seedIndex] }
        let diams = holds.map { h in max(h.bboxWidth, h.bboxHeight) }
        let medianDiam = diams.sorted()[diams.count / 2]
        let eps = dbscanEpsFactor * medianDiam

        // DBSCAN indexes the compact candidate list, not the original holds.
        let clusters = dbscan(points: candidates.map { holds[$0].centroid },
                              eps: eps, minSamples: minSamples)
        guard let seedLocalIndex = candidates.firstIndex(of: seedIndex),
              let seedCluster = clusters[seedLocalIndex] else { return [seedIndex] }
        return Set(candidates.enumerated().compactMap { localIndex, globalIndex in
            clusters[localIndex] == seedCluster ? globalIndex : nil
        })
    }

    // MARK: mask sampling (bbox-local raster -> canonical image pixel coords)

    private static func sampleMask(_ h: HoldGeometry,
                                   pixels: (Int, Int) -> (r: Double, g: Double, b: Double)?,
                                   w: Int, hgt: Int) -> [(r: Double, g: Double, b: Double)] {
        // RLE decode: COCO-style counts array (run-lengths of 0s/1s alternating)
        let counts = decodeRLE(h.maskRLE)
        var out: [(r: Double, g: Double, b: Double)] = []
        var idx = 0
        var value = 0
        for count in counts {
            for _ in 0..<count {
                let localX = idx % h.maskWidth
                let localY = idx / h.maskWidth
                if value == 1 {
                    // bbox-local raster -> canonical normalized -> image px
                    let nx = h.bboxX + (Double(localX) + 0.5) / Double(h.maskWidth) * h.bboxWidth
                    let ny = h.bboxY + (Double(localY) + 0.5) / Double(h.maskHeight) * h.bboxHeight
                    let px = Int(nx * Double(w)), py = Int(ny * Double(hgt))
                    if let c = pixels(px, py) { out.append(c) }
                }
                idx += 1
            }
            value = 1 - value  // flip once per RLE run boundary, not per pixel
        }
        return out
    }

    private static func decodeRLE(_ data: Data) -> [Int] {
        // COCO RLE counts serialized as int32 little-endian array (our storage contract)
        let ints = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        return ints.map { Int($0) }
    }

    // MARK: DBSCAN (minimal, on 2D points; noise = nil)

    private func dbscan(points: [(x: Double, y: Double)],
                        eps: Double, minSamples: Int) -> [Int: Int] {
        var labels: [Int: Int] = [:]
        var cluster = 0
        for i in points.indices where labels[i] == nil {
            let neighbors = points.indices.filter { j in i != j && dist(points[i], points[j]) <= eps }
            if neighbors.count < minSamples - 1 {
                labels[i] = nil  // noise
                continue
            }
            labels[i] = cluster
            var queue = neighbors
            while let j = queue.popLast() {
                if labels[j] == nil { labels[j] = cluster }
                if labels[j] != cluster { continue }
                let jNeighbors = points.indices.filter { k in j != k && dist(points[j], points[k]) <= eps }
                for n in jNeighbors where labels[n] == nil {
                    labels[n] = cluster
                    queue.append(n)
                }
            }
            cluster += 1
        }
        return labels
    }

    private func dist(_ a: (x: Double, y: Double), _ b: (x: Double, y: Double)) -> Double {
        hypot(a.x - b.x, a.y - b.y)
    }
}
