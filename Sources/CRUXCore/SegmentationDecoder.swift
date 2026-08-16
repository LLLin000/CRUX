import Foundation

/// Decodes RF-DETR ONNX outputs (`dets` cxcywh, `labels` logits, `masks` logits)
/// into domain detections. Same algorithm as `tools/infer_real.py:decode`:
/// sigmoid(obj) confidence with per-size threshold (>=1.5% area -> 0.10,
/// else 0.30), class = argmax of the two class logits, bbox normalized
/// cxcywh clipped to [0,1], bbox-local mask via `MaskRasterizer`
/// (two-stage bilinear + clip + threshold 0.5).
/// Tensor plumbing stays in the client adapter; this is pure math.
public enum SegmentationDecoder {
    public static let queryCount = 100

    /// - Returns: nil when outputs are malformed for the declared input size.
    public static func decode(
        dets: [Float],
        labels: [Float],
        masks: [Float],
        imageWidth: Int,
        imageHeight: Int,
        inputSize: Int,
        modelVersion: String = "",
        queryCount: Int = queryCount
    ) -> SegmentationResult? {
        let maskSize = inputSize / 4
        guard inputSize > 0, inputSize % 4 == 0,
              imageWidth > 0, imageHeight > 0,
              dets.count >= queryCount * 4,
              labels.count >= queryCount * 3,
              masks.count >= queryCount * maskSize * maskSize else {
            return nil
        }

        var detections: [DetectedHold] = []
        detections.reserveCapacity(queryCount)
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

    private static func sigmoid(_ value: Float) -> Double {
        let clamped = max(-60, min(60, Double(value)))
        return 1 / (1 + exp(-clamped))
    }
}
