#if os(iOS)
import CoreGraphics
import Foundation
import ImageIO
import CRUXCore

/// Bridges canonical JPEG bytes to Core's model-independent color analysis.
public struct RouteAnalysisCoordinator: Sendable {
    private let segmenter: any HoldSegmenter
    private let colorAnalyzer: HoldColorAnalyzer

    public init(
        segmenter: any HoldSegmenter,
        colorAnalyzer: HoldColorAnalyzer = .init()
    ) {
        self.segmenter = segmenter
        self.colorAnalyzer = colorAnalyzer
    }

    public func analyze(_ image: CanonicalImage) async throws -> RouteAnalysis {
        let segmentation = try await segmenter.segment(image)
        let pixels = try CanonicalPixelBuffer(image: image)
        return try colorAnalyzer.analyze(
            segmentation: segmentation,
            imageWidth: pixels.width,
            imageHeight: pixels.height,
            imagePixels: pixels.colorAt
        )
    }
}

private struct CanonicalPixelBuffer: Sendable {
    let width: Int
    let height: Int
    private let rgba: [UInt8]

    init(image: CanonicalImage) throws {
        guard image.width > 0, image.height > 0,
              let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil),
              cgImage.width == image.width,
              cgImage.height == image.height else {
            throw HoldSegmenterError.invalidImage
        }

        var rgba = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: &rgba,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: image.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        ) else {
            throw HoldSegmenterError.invalidImage
        }
        context.interpolationQuality = .none
        context.draw(
            cgImage,
            in: CGRect(x: 0, y: 0, width: image.width, height: image.height)
        )
        self.width = image.width
        self.height = image.height
        self.rgba = rgba
    }

    func colorAt(x: Int, y: Int) -> SRGBColor? {
        guard (0..<width).contains(x), (0..<height).contains(y) else { return nil }
        let offset = (y * width + x) * 4
        return SRGBColor(
            red: Double(rgba[offset]),
            green: Double(rgba[offset + 1]),
            blue: Double(rgba[offset + 2])
        )
    }
}
#endif
