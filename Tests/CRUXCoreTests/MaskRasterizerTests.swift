import XCTest
@testable import CRUXCore

final class MaskRasterizerTests: XCTestCase {
    func testTwoStageBilinearRasterizationMatchesPythonReference() {
        // Python reference: sigmoid -> cv2.resize(..., INTER_LINEAR) twice
        // -> threshold >= 0.5 -> row-major RLE.
        let logits: [Float] = [-10, 10, 10, -10]
        let rasterized = MaskRasterizer.rasterize(
            logits: logits,
            offset: 0,
            maskSize: 2,
            inputSize: 4,
            imageWidth: 4,
            imageHeight: 4,
            bboxX: 0,
            bboxY: 0,
            bboxWidth: 1,
            bboxHeight: 1
        )

        XCTAssertEqual(rasterized?.width, 4)
        XCTAssertEqual(rasterized.map { decodeRLE($0) }, [Int32(2), 2, 2, 4, 2, 2, 2])
    }

    private func decodeRLE(_ mask: RasterizedMask) -> [Int32] {
        mask.rle.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Int32.self))
        }
    }
}
