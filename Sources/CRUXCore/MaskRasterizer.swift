import Foundation

public struct RasterizedMask: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let rle: Data

    public init(width: Int, height: Int, rle: Data) {
        self.width = width
        self.height = height
        self.rle = rle
    }
}

/// Reconstructs a bbox-local binary mask using the Python reference's two bilinear resizes.
public enum MaskRasterizer {
    public static func rasterize(
        logits: [Float],
        offset: Int,
        maskSize: Int,
        inputSize: Int,
        imageWidth: Int,
        imageHeight: Int,
        bboxX: Double,
        bboxY: Double,
        bboxWidth: Double,
        bboxHeight: Double
    ) -> RasterizedMask? {
        guard offset >= 0, maskSize > 0, inputSize > 0,
              imageWidth > 0, imageHeight > 0,
              bboxWidth > 0, bboxHeight > 0,
              offset + maskSize * maskSize <= logits.count else {
            return nil
        }

        let x1 = max(0, min(imageWidth, Int(floor(bboxX * Double(imageWidth)))))
        let y1 = max(0, min(imageHeight, Int(floor(bboxY * Double(imageHeight)))))
        let x2 = max(0, min(imageWidth, Int(ceil((bboxX + bboxWidth) * Double(imageWidth)))))
        let y2 = max(0, min(imageHeight, Int(ceil((bboxY + bboxHeight) * Double(imageHeight)))))
        guard x2 > x1, y2 > y1 else { return nil }

        var resized = [Float](repeating: 0, count: inputSize * inputSize)
        for y in 0..<inputSize {
            let sourceY = (Double(y) + 0.5) * Double(maskSize) / Double(inputSize) - 0.5
            for x in 0..<inputSize {
                let sourceX = (Double(x) + 0.5) * Double(maskSize) / Double(inputSize) - 0.5
                resized[y * inputSize + x] = sampleBilinear(
                    logits: logits,
                    offset: offset,
                    size: maskSize,
                    x: sourceX,
                    y: sourceY,
                    applySigmoid: true
                )
            }
        }

        var counts: [Int32] = []
        var currentValue = false
        var runLength: Int32 = 0
        for y in y1..<y2 {
            let sourceY = (Double(y) + 0.5) * Double(inputSize) / Double(imageHeight) - 0.5
            for x in x1..<x2 {
                let sourceX = (Double(x) + 0.5) * Double(inputSize) / Double(imageWidth) - 0.5
                let value = sampleBilinear(
                    pixels: resized,
                    size: inputSize,
                    x: sourceX,
                    y: sourceY
                ) >= 0.5
                if value == currentValue {
                    runLength += 1
                } else {
                    counts.append(runLength)
                    currentValue = value
                    runLength = 1
                }
            }
        }
        counts.append(runLength)

        return RasterizedMask(
            width: x2 - x1,
            height: y2 - y1,
            rle: counts.withUnsafeBytes { Data($0) }
        )
    }

    private static func sampleBilinear(
        logits: [Float],
        offset: Int,
        size: Int,
        x: Double,
        y: Double,
        applySigmoid: Bool
    ) -> Float {
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = x0 + 1
        let y1 = y0 + 1
        let wx = Float(x - Double(x0))
        let wy = Float(y - Double(y0))
        let p00 = logits[offset + clamped(y0, size) * size + clamped(x0, size)]
        let p01 = logits[offset + clamped(y0, size) * size + clamped(x1, size)]
        let p10 = logits[offset + clamped(y1, size) * size + clamped(x0, size)]
        let p11 = logits[offset + clamped(y1, size) * size + clamped(x1, size)]
        let top = p00 + (p01 - p00) * wx
        let bottom = p10 + (p11 - p10) * wx
        let value = top + (bottom - top) * wy
        return applySigmoid ? sigmoid(value) : value
    }

    private static func sampleBilinear(
        pixels: [Float],
        size: Int,
        x: Double,
        y: Double
    ) -> Float {
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = x0 + 1
        let y1 = y0 + 1
        let wx = Float(x - Double(x0))
        let wy = Float(y - Double(y0))
        let p00 = pixels[clamped(y0, size) * size + clamped(x0, size)]
        let p01 = pixels[clamped(y0, size) * size + clamped(x1, size)]
        let p10 = pixels[clamped(y1, size) * size + clamped(x0, size)]
        let p11 = pixels[clamped(y1, size) * size + clamped(x1, size)]
        let top = p00 + (p01 - p00) * wx
        let bottom = p10 + (p11 - p10) * wx
        return top + (bottom - top) * wy
    }

    private static func clamped(_ value: Int, _ size: Int) -> Int {
        max(0, min(size - 1, value))
    }

    private static func sigmoid(_ value: Float) -> Float {
        let clamped = max(-60, min(60, Double(value)))
        return Float(1 / (1 + exp(-clamped)))
    }
}
