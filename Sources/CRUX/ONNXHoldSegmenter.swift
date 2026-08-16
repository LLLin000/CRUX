#if os(iOS)
import CoreGraphics
import Foundation
import ImageIO
import CRUXCore
import OnnxRuntimeBindings

/// ONNX Runtime adapter for the exported RF-DETR-Seg model.
/// Tensor names and model-space decoding stay inside the client boundary.
public actor ONNXHoldSegmenter: HoldSegmenter {
    public nonisolated let modelVersion: String
    public nonisolated let inputSize: Int

    private let session: ORTSession
    private let inputName: String
    private let outputNames: Set<String>
    private let runOptions: ORTRunOptions

    public init(
        modelURL: URL,
        modelVersion: String = "v1.0.1",
        inputSize: Int = 648
    ) throws {
        guard inputSize > 0, FileManager.default.fileExists(atPath: modelURL.path) else {
            throw HoldSegmenterError.modelUnavailable
        }

        do {
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            let runOptions = try ORTRunOptions()
            self.session = try ORTSession(
                env: env,
                modelPath: modelURL.path,
                sessionOptions: options
            )
            self.runOptions = runOptions
            self.modelVersion = modelVersion
            self.inputSize = inputSize
            self.inputName = "input"
            self.outputNames = ["dets", "labels", "masks"]
        } catch {
            throw HoldSegmenterError.modelUnavailable
        }
    }

    public func segment(_ image: CanonicalImage) async throws -> SegmentationResult {
        do {
            let rgba = try Self.decodeRGBA(image)
            guard let input = SegmentationPreprocessor.preprocess(
                rgba: rgba,
                width: image.width,
                height: image.height,
                size: inputSize
            ) else {
                throw HoldSegmenterError.invalidImage
            }
            let tensorData = input.withUnsafeBytes { rawBuffer in
                NSMutableData(bytes: rawBuffer.baseAddress!, length: rawBuffer.count)
            }
            let inputValue = try ORTValue(
                tensorData: tensorData,
                elementType: .float,
                shape: [
                    NSNumber(value: 1),
                    NSNumber(value: 3),
                    NSNumber(value: inputSize),
                    NSNumber(value: inputSize)
                ]
            )
            let outputs = try session.run(
                withInputs: [inputName: inputValue],
                outputNames: outputNames,
                runOptions: runOptions
            )
            guard let detsValue = outputs["dets"],
                  let labelsValue = outputs["labels"],
                  let masksValue = outputs["masks"] else {
                throw HoldSegmenterError.invalidOutput
            }
            let dets = try Self.floats(from: detsValue)
            let labels = try Self.floats(from: labelsValue)
            let masks = try Self.floats(from: masksValue)
            guard let result = SegmentationDecoder.decode(
                dets: dets,
                labels: labels,
                masks: masks,
                imageWidth: image.width,
                imageHeight: image.height,
                inputSize: inputSize,
                modelVersion: modelVersion
            ) else {
                throw HoldSegmenterError.invalidOutput
            }
            return result
        } catch let error as HoldSegmenterError {
            throw error
        } catch {
            throw HoldSegmenterError.inferenceFailed
        }
    }

    /// JPEG -> full-size RGBA8 pixels (no scaling; interpolation happens in
    /// `SegmentationPreprocessor`, matching the Python reference).
    private static func decodeRGBA(_ image: CanonicalImage) throws -> [UInt8] {
        guard image.width > 0, image.height > 0 else {
            throw HoldSegmenterError.invalidImage
        }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil),
              sourceImage.width == image.width,
              sourceImage.height == image.height else {
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
            sourceImage,
            in: CGRect(x: 0, y: 0, width: image.width, height: image.height)
        )
        return rgba
    }

    private static func floats(from value: ORTValue) throws -> [Float] {
        let data = try value.tensorData()
        let rawData = Data(bytes: data.bytes, count: data.length)
        return rawData.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }
}
#endif
