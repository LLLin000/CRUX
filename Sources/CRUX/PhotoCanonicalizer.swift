// Photo canonicalization (PLAN §4, M1): every saved route photo is
//  1. orientation-fixed (EXIF rotation applied to pixels),
//  2. converted to sRGB,
//  3. downscaled to <= maxEdge on the long side,
//  then stored as JPEG. Downstream (Lab sampling, mask rendering, ONNX) can
//  assume upright sRGB pixels with a bounded size.

#if os(iOS)
import CoreImage
import UIKit

enum PhotoCanonicalizer {
    static let maxEdge: CGFloat = 2048

    /// Returns canonical JPEG data, or nil on any failure.
    static func canonicalJPEG(from image: UIImage,
                              maxEdge: CGFloat = maxEdge,
                              compressionQuality: CGFloat = 0.9) -> Data? {
        guard let cgImage = image.cgImage else { return nil }
        let ci = CIImage(cgImage: cgImage)
            .oriented(forExifOrientation: exifOrientation(image.imageOrientation))

        let scale = min(1.0, maxEdge / max(ci.extent.width, ci.extent.height))
        var oriented = ci
        if scale < 1.0 {
            oriented = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let extent = oriented.extent.integral

        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [.workingColorSpace: sRGB])
        guard let out = context.createCGImage(oriented, from: extent,
                                              format: .RGBA8, colorSpace: sRGB) else {
            return nil
        }
        return UIImage(cgImage: out).jpegData(compressionQuality: compressionQuality)
    }

    /// UIImage.Orientation -> EXIF orientation code (1-8), the value
    /// CIImage.oriented(forExifOrientation:) expects.
    static func exifOrientation(_ o: UIImage.Orientation) -> Int32 {
        switch o {
        case .up: return 1
        case .upMirrored: return 2
        case .down: return 3
        case .downMirrored: return 4
        case .leftMirrored: return 5
        case .right: return 6
        case .rightMirrored: return 7
        case .left: return 8
        @unknown default: return 1
        }
    }
}
#endif  // os(iOS)
