import Foundation

/// Python-reference image preprocessing for the RF-DETR ONNX input.
/// Matches `rfdetr.export.benchmark.infer_transforms` semantics exactly:
/// uint8 RGBA -> float RGB/255 -> bilinear stretch to `size`x`size`
/// (torchvision Resize, antialias=false: src = (dst+0.5)*in/out - 0.5,
/// clamped to the edge) -> ImageNet normalize -> CHW float32.
/// No CoreGraphics: decoupled so the math is testable on any platform.
public enum SegmentationPreprocessor {
    public static let mean: [Float] = [0.485, 0.456, 0.406]
    public static let standardDeviation: [Float] = [0.229, 0.224, 0.225]

    /// - Parameters:
    ///   - rgba: RGBA8 pixels, row-major, `width`*`height`*4 bytes.
    ///   - width/height: source image dimensions.
    ///   - size: square model input resolution.
    /// - Returns: CHW float32 tensor, or nil on invalid input.
    public static func preprocess(rgba: [UInt8], width: Int, height: Int, size: Int) -> [Float]? {
        guard width > 0, height > 0, size > 0,
              rgba.count == width * height * 4 else {
            return nil
        }
        let count = size * size
        var resized = [Float](repeating: 0, count: count * 3)

        for y in 0..<size {
            let sourceY = (Double(y) + 0.5) * Double(height) / Double(size) - 0.5
            for x in 0..<size {
                let sourceX = (Double(x) + 0.5) * Double(width) / Double(size) - 0.5
                let dst = y * size + x
                for channel in 0..<3 {
                    resized[channel * count + dst] = sampleChannel(
                        rgba: rgba, width: width, height: height,
                        channel: channel, x: sourceX, y: sourceY
                    )
                }
            }
        }

        var output = [Float](repeating: 0, count: count * 3)
        for channel in 0..<3 {
            let offset = channel * count
            for i in 0..<count {
                output[offset + i] = (resized[offset + i] - mean[channel]) / standardDeviation[channel]
            }
        }
        return output
    }

    private static func sampleChannel(
        rgba: [UInt8], width: Int, height: Int,
        channel: Int, x: Double, y: Double
    ) -> Float {
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let wx = Float(x - Double(x0))
        let wy = Float(y - Double(y0))
        let x1 = x0 + 1
        let y1 = y0 + 1
        let p00 = Float(rgba[(clamp(y0, height) * width + clamp(x0, width)) * 4 + channel]) / 255
        let p01 = Float(rgba[(clamp(y0, height) * width + clamp(x1, width)) * 4 + channel]) / 255
        let p10 = Float(rgba[(clamp(y1, height) * width + clamp(x0, width)) * 4 + channel]) / 255
        let p11 = Float(rgba[(clamp(y1, height) * width + clamp(x1, width)) * 4 + channel]) / 255
        let top = p00 + (p01 - p00) * wx
        let bottom = p10 + (p11 - p10) * wx
        return top + (bottom - top) * wy
    }

    private static func clamp(_ value: Int, _ limit: Int) -> Int {
        max(0, min(limit - 1, value))
    }
}
