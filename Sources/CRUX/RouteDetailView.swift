// Route detail — photo + pseudo-segmentation rendering (PLAN M1, prototype
// glowOnly/glowLine alignment). Holds' masks are decoded from bbox-local
// COCO RLE and composited onto the canonical photo with a glow + optional
// route spine. Real ONNX segmentation lands in M2; this file also carries
// the DEBUG-only demo seed so the full flow is walkable today.
//
// Entire file is iOS-only (UIKit/CoreGraphics): guarded for the macOS
// `swift test` full-package build.

#if os(iOS)
import SwiftUI
import SwiftData
import UIKit
import CoreGraphics

extension UIColor {
    convenience init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  alpha: 1)
    }
}

// MARK: - COCO RLE (bbox-local raster, counts int32 LE, runs start at 0)

enum MaskRLE {
    /// bitmap (0/1, row-major) -> COCO counts (starts with 0-run).
    static func encode(_ bitmap: [UInt8], width: Int, height: Int) -> Data {
        var counts: [Int32] = [0]  // COCO RLE always starts with a 0-run
        var value = 0
        for i in 0..<(width * height) {
            let bit = bitmap[i] > 0 ? 1 : 0
            if bit == value {
                counts[counts.count - 1] += 1
            } else {
                counts.append(1)
                value = bit
            }
        }
        var data = Data()
        counts.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        return data
    }

    /// counts -> bitmap (0/1, row-major). Empty on malformed input.
    static func decode(_ data: Data, width: Int, height: Int) -> [UInt8] {
        let total = width * height
        guard total > 0, total < 50_000_000 else { return [] }
        var out = [UInt8](repeating: 0, count: total)
        let counts: [Int32] = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        var idx = 0
        var value = 0
        for count in counts {
            let c = Int(count)
            if c < 0 || idx + c > total { return [] }
            if value == 1 {
                for j in idx..<(idx + c) { out[j] = 1 }
            }
            idx += c
            value = 1 - value
        }
        return out
    }

    /// Procedural hold: filled circle mask on a WxH bitmap (demo data).
    static func circleBitmap(width: Int, height: Int,
                             cx: Double, cy: Double, radius: Double) -> [UInt8] {
        var bmp = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let dx = (Double(x) + 0.5 - cx) / radius
                let dy = (Double(y) + 0.5 - cy) / radius
                if dx * dx + dy * dy <= 1.0 { bmp[y * width + x] = 1 }
            }
        }
        return bmp
    }
}

// MARK: - Rendering engine

enum RouteRenderMode: String, CaseIterable, Identifiable {
    case glowOnly = "Glow"
    case glowLine = "Glow+线"
    var id: String { rawValue }
}

enum RouteRenderEngine {
    /// Composite canonical photo + hold masks (+ optional spine) into one image.
    static func render(photo: UIImage, holds: [Hold],
                       mode: RouteRenderMode) -> UIImage {
        let size = photo.size
        let scale = photo.scale
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        guard let ctx = UIGraphicsGetCurrentContext() else { return photo }

        photo.draw(in: CGRect(origin: .zero, size: size))
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        let holds = holds.sorted { $0.centroidY < $1.centroidY }  // bottom-up spine

        // spine under the masks
        if mode == .glowLine, holds.count > 1 {
            let spineColor = UIColor(Theme.routeColor(holds[0].route?.paletteColor ?? .blue)).cgColor
            ctx.setStrokeColor(spineColor)
            ctx.setAlpha(0.85)
            ctx.setLineWidth(max(3, size.width * 0.008))
            let pts = holds.map { CGPoint(x: $0.centroidX * size.width,
                                          y: $0.centroidY * size.height) }
            ctx.beginPath()
            ctx.move(to: pts[0])
            for p in pts.dropFirst() { ctx.addLine(to: p) }
            ctx.strokePath()
        }

        for hold in holds {
            drawHold(ctx, hold, size: size)
        }
        return UIGraphicsGetImageFromCurrentImageContext() ?? photo
    }

    private static func drawHold(_ ctx: CGContext, _ hold: Hold, size: CGSize) {
        let uiColor = UIColor(Theme.routeColor(hold.route?.paletteColor ?? .blue))
        let color = uiColor.cgColor
        let bbox = CGRect(x: hold.bboxX * size.width,
                          y: hold.bboxY * size.height,
                          width: hold.bboxWidth * size.width,
                          height: hold.bboxHeight * size.height)
        guard bbox.width > 1, bbox.height > 1 else { return }
        let bitmap = MaskRLE.decode(hold.maskRLE, width: hold.maskWidth,
                                    height: hold.maskHeight)
        guard !bitmap.isEmpty else { return }

        // raster mask -> upscaled to bbox rect, then boundary path
        let (raster, rw, rh) = renderBitmap(bitmap, width: hold.maskWidth,
                                            height: hold.maskHeight, size: bbox.size)
        let path = contourPath(of: raster, width: rw, height: rh)

        // glow: layered translucent strokes
        ctx.setAlpha(0.10); ctx.setLineWidth(max(18, bbox.width * 0.30))
        ctx.setStrokeColor(color); ctx.addPath(path); ctx.strokePath()
        ctx.setAlpha(0.18); ctx.setLineWidth(max(10, bbox.width * 0.16))
        ctx.addPath(path); ctx.strokePath()

        // fill
        ctx.setAlpha(0.38)
        ctx.setFillColor(color)
        ctx.addPath(path); ctx.fillPath()

        // crisp edge
        ctx.setAlpha(0.9)
        ctx.setLineWidth(max(2, bbox.width * 0.035))
        ctx.addPath(path); ctx.strokePath()
    }

    /// Scale the bbox-local mask to the on-image bbox rect (nearest).
    private static func renderBitmap(_ bmp: [UInt8], width: Int, height: Int,
                                     size: CGSize) -> ([UInt8], Int, Int) {
        guard width > 0, height > 0, size.width > 1, size.height > 1 else {
            return (bmp, width, height)
        }
        let sw = Int(max(1, size.width.rounded())), sh = Int(max(1, size.height.rounded()))
        var out = [UInt8](repeating: 0, count: sw * sh)
        for y in 0..<sh {
            let sy = min(height - 1, y * height / sh)
            for x in 0..<sw {
                let sx = min(width - 1, x * width / sw)
                if bmp[sy * width + sx] > 0 { out[y * sw + x] = 1 }
            }
        }
        return (out, sw, sh)
    }

    /// Boundary contour of a binary raster as a closed CGPath.
    private static func contourPath(of bmp: [UInt8], width: Int, height: Int) -> CGPath {
        let path = CGMutablePath()
        var pts: [CGPoint] = []
        var sumX = 0.0, sumY = 0.0
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0..<height {
            for x in 0..<width where bmp[y * width + x] > 0 {
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                let isBoundary = x == 0 || y == 0 || x == width - 1 || y == height - 1 ||
                    bmp[y * width + x - 1] == 0 || bmp[y * width + x + 1] == 0 ||
                    bmp[(y - 1) * width + x] == 0 || bmp[(y + 1) * width + x] == 0
                if isBoundary {
                    pts.append(CGPoint(x: x, y: y))
                    sumX += Double(x); sumY += Double(y)
                }
            }
        }
        guard pts.count >= 3, minX <= maxX, minY <= maxY else {
            let rect = CGRect(x: minX, y: minY,
                              width: max(1, maxX - minX), height: max(1, maxY - minY))
            return CGPath(rect: rect, transform: nil)
        }
        let cx = sumX / Double(pts.count), cy = sumY / Double(pts.count)
        pts.sort { a, b in
            atan2(a.y - cy, a.x - cx) < atan2(b.y - cy, b.x - cx)
        }
        path.move(to: pts[0])
        for p in pts.dropFirst() { path.addLine(to: p) }
        path.closeSubpath()
        return path
    }
}

// MARK: - Detail view

struct RouteDetailView: View {
    let route: ClimbRoute
    @State private var mode: RouteRenderMode = .glowLine
    @State private var photo: UIImage?

    var body: some View {
        Group {
            if let photo {
                let rendered = RouteRenderEngine.render(photo: photo,
                                                        holds: route.holds,
                                                        mode: mode)
                Image(uiImage: rendered)
                    .resizable()
                    .scaledToFit()
                    .padding(.horizontal, 10)
            } else {
                ContentUnavailableView("无法加载照片",
                                       systemImage: "photo.badge.exclamationmark")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
        .navigationTitle(route.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("模式", selection: $mode) {
                    ForEach(RouteRenderMode.allCases) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
            }
        }
        .onAppear(perform: loadPhoto)
        .preferredColorScheme(.dark)
    }

    private func loadPhoto() {
        guard let filename = route.photoFilename, photo == nil else { return }
        let url = URL.documentsDirectory.appending(path: "Photos").appending(path: filename)
        photo = UIImage(contentsOfFile: url.path)
    }
}

// MARK: - DEBUG demo seed (hand-placed holds, no model)

#if DEBUG
enum RouteDemoData {
    /// Procedural "climbing wall" photo (gradient + a few blobs), saved canonical.
    static func demoPhoto() -> String? {
        let side: CGFloat = 1024
        UIGraphicsBeginImageContextWithOptions(CGSize(width: side, height: side), false, 1)
        defer { UIGraphicsEndImageContext() }
        guard let ctx = UIGraphicsGetCurrentContext() else { return nil }
        let colors = [UIColor(hex: 0x2A2F38).cgColor, UIColor(hex: 0x14161A).cgColor]
        let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                              colors: colors as CFArray, locations: [0, 1])!
        ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: 0),
                               end: CGPoint(x: side, y: side), options: [])
        // wall texture noise
        for _ in 0..<120 {
            ctx.setFillColor(UIColor.white.withAlphaComponent(0.04).cgColor)
            ctx.fillEllipse(in: CGRect(x: CGFloat.random(in: 0..<side),
                                       y: CGFloat.random(in: 0..<side),
                                       width: 6, height: 6))
        }
        let img = UIGraphicsGetImageFromCurrentImageContext()!
        guard let data = PhotoCanonicalizer.canonicalJPEG(from: img, maxEdge: 2048) else { return nil }
        let dir = URL.documentsDirectory.appending(path: "Photos")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appending(path: "demo-wall-\(UUID().uuidString.prefix(8)).jpg")
        do { try data.write(to: url); return url.lastPathComponent }
        catch { return nil }
    }

    /// 6 hand-placed holds forming a route: circles at canonical coords.
    static func demoHolds() -> [Hold] {
        let placements: [(Double, Double)] = [
            (0.30, 0.78), (0.42, 0.66), (0.34, 0.54),
            (0.50, 0.44), (0.42, 0.32), (0.58, 0.22),
        ]
        return placements.enumerated().map { i, p in
            let mw = 48, mh = 48
            let r = 0.42
            let bmp = MaskRLE.circleBitmap(width: mw, height: mh,
                                           cx: Double(mw) / 2, cy: Double(mh) / 2,
                                           radius: Double(mw) / 2 * r)
            let bw = 0.10, bh = 0.10
            return Hold(
                centroidX: p.0, centroidY: p.1,
                bboxX: p.0 - bw / 2, bboxY: p.1 - bh / 2,
                bboxWidth: bw, bboxHeight: bh,
                maskWidth: mw, maskHeight: mh,
                maskRLE: MaskRLE.encode(bmp, width: mw, height: mh),
                isStart: i == 0, isFinish: i == placements.count - 1
            )
        }
    }

    /// Insert one demo route (idempotent per session via name marker).
    static func insertDemoRoute(context: ModelContext) {
        let existing = (try? context.fetch(FetchDescriptor<ClimbRoute>(
            predicate: #Predicate { $0.name == "演示蓝色线" })))?.isEmpty ?? true
        guard existing else { return }
        guard let filename = demoPhoto() else { return }
        let route = ClimbRoute(
            name: "演示蓝色线",
            gymNameSnapshot: "Demo Gym",
            gradeValue: "V3",
            referenceL: 30, referenceA: 5, referenceB: -28,
            paletteColor: .blue,
            result: .top, attempts: 2, durationSeconds: 180,
            date: .now,
            photoFilename: filename,
            segmenterModelVersion: nil,
            routeSelectorVersion: "demo-seed",
            inferenceInputSize: nil
        )
        let holds = demoHolds()
        holds.forEach { $0.route = route }
        route.holds = holds
        context.insert(route)
        try? context.save()
    }
}
#endif  // DEBUG

#endif  // os(iOS)
