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
            let input = try Self.preprocess(image, size: inputSize)
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
            return try Self.decode(
                outputs: outputs,
                imageWidth: image.width,
                imageHeight: image.height,
                inputSize: inputSize,
                modelVersion: modelVersion
            )
        } catch let error as HoldSegmenterError {
            throw error
        } catch {
            throw HoldSegmenterError.inferenceFailed
        }
    }

    private static func preprocess(_ image: CanonicalImage, size: Int) throws -> [Float] {
        guard image.width > 0, image.height > 0 else {
            throw HoldSegmenterError.invalidImage
        }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil),
              sourceImage.width == image.width,
              sourceImage.height == image.height else {
            throw HoldSegmenterError.invalidImage
        }

        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: &rgba,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw HoldSegmenterError.invalidImage
        }
        context.interpolationQuality = .medium
        context.draw(sourceImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        let means: [Float] = [0.485, 0.456, 0.406]
        let standardDeviations: [Float] = [0.229, 0.224, 0.225]
        var output = [Float](repeating: 0, count: 3 * size * size)
        for y in 0..<size {
            for x in 0..<size {
                let sourceIndex = (y * size + x) * 4
                let destinationIndex = y * size + x
                output[destinationIndex] = (Float(rgba[sourceIndex]) / 255 - means[0]) / standardDeviations[0]
                output[size * size + destinationIndex] = (Float(rgba[sourceIndex + 1]) / 255 - means[1]) / standardDeviations[1]
                output[2 * size * size + destinationIndex] = (Float(rgba[sourceIndex + 2]) / 255 - means[2]) / standardDeviations[2]
            }
        }
        return output
    }

    private static func decode(
        outputs: [String: ORTValue],
        imageWidth: Int,
        imageHeight: Int,
        inputSize: Int,
        modelVersion: String
    ) throws -> SegmentationResult {
        guard imageWidth > 0, imageHeight > 0,
              let detsValue = outputs["dets"],
              let labelsValue = outputs["labels"],
              let masksValue = outputs["masks"] else {
            throw HoldSegmenterError.invalidOutput
        }

        let dets = try floats(from: detsValue)
        let labels = try floats(from: labelsValue)
        let masks = try floats(from: masksValue)
        let queryCount = 100
        let maskSize = inputSize / 4
        guard dets.count >= queryCount * 4,
              labels.count >= queryCount * 3,
              maskSize > 0,
              masks.count >= queryCount * maskSize * maskSize else {
            throw HoldSegmenterError.invalidOutput
        }

        var detections: [DetectedHold] = []
        for query in 0..<queryCount {
            let labelOffset = query * 3
            let confidence = sigmoid(labels[labelOffset])
            let boxOffset = query * 4
            let rawWidth = Double(dets[boxOffset + 2])
            let rawHeight = Double(dets[boxOffset + 3])
            let threshold = rawWidth * rawHeight >= 0.015 ? 0.10 : 0.30
            guard confidence >= threshold else { continue }

            let centerX = Double(dets[boxOffset])
            let centerY = Double(dets[boxOffset + 1])
            let x1 = max(0, min(1, centerX - rawWidth / 2))
            let y1 = max(0, min(1, centerY - rawHeight / 2))
            let x2 = max(0, min(1, centerX + rawWidth / 2))
            let y2 = max(0, min(1, centerY + rawHeight / 2))
            guard x2 > x1, y2 > y1 else { continue }

            let maskOffset = query * maskSize * maskSize
            guard let rasterized = MaskRasterizer.rasterize(
                logits: masks,
                offset: maskOffset,
                maskSize: maskSize,
                inputSize: inputSize,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                bboxX: x1,
                bboxY: y1,
                bboxWidth: x2 - x1,
                bboxHeight: y2 - y1
            ) else {
                continue
            }
            detections.append(
                DetectedHold(
                    id: query,
                    kind: labels[labelOffset + 2] > labels[labelOffset + 1] ? .volume : .hold,
                    confidence: Float(confidence),
                    geometry: HoldGeometry(
                        bboxX: x1,
                        bboxY: y1,
                        bboxWidth: x2 - x1,
                        bboxHeight: y2 - y1,
                        maskWidth: rasterized.width,
                        maskHeight: rasterized.height,
                        maskRLE: rasterized.rle
                    )
                )
            )
        }

        return SegmentationResult(
            modelVersion: modelVersion,
            inputSize: inputSize,
            detections: detections
        )
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
