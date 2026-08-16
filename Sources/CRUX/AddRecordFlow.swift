// Add-record flow — camera/gallery -> form -> save (PLAN M1; gym via MKLocalSearch)
// Camera = wrapped UIImagePickerController; gallery = PhotosPicker (no permission).
// Entire file is iOS-only (UIKit): guarded so the CRUXCore test chain can
// build the CRUX target on macOS for `swift test` (full-package build).

#if os(iOS)
import SwiftUI
import SwiftData
import PhotosUI
import MapKit
import CRUXCore
import CRUXClient

@Observable
final class AddRecordModel {
    var photo: UIImage?
    var gymName = ""
    var gymCoordinate: CLLocationCoordinate2D?
    var gymMapItemID: String?
    var grade: String = "V4"
    var result: RouteResult = .top
    var attempts: Int = 1
    var elapsed: TimeInterval = 0
    var timerRunning = false
    var note = ""
    var palette: RouteColor = .blue
    var referenceLab: (Double, Double, Double) = (0, 0, 0)

    // M2: on-device segmentation + seed selection state
    var analysis: RouteAnalysis?
    var draft: RouteDraft?
    var isAnalyzing = false
    var analyzeError: String?
    private var segmenter: ONNXHoldSegmenter?

    /// Canonicalize, segment, and build an empty draft (first tap = seed).
    func analyze(_ image: UIImage) async {
        isAnalyzing = true
        analyzeError = nil
        analysis = nil
        draft = nil
        defer { isAnalyzing = false }
        guard let jpeg = PhotoCanonicalizer.canonicalJPEG(from: image),
              let canonical = UIImage(data: jpeg),
              let cgImage = canonical.cgImage else {
            analyzeError = "无法处理照片"
            return
        }
        do {
            if segmenter == nil {
                guard let url = Bundle.main.url(
                    forResource: "crux-hold-seg-v1.0.1-648-int8",
                    withExtension: "onnx"
                ) else {
                    analyzeError = "模型缺失"
                    return
                }
                segmenter = try ONNXHoldSegmenter(
                    modelURL: url, modelVersion: "v1.0.1-int8", inputSize: 648
                )
            }
            guard let segmenter else { return }
            let coordinator = RouteAnalysisCoordinator(segmenter: segmenter)
            let result = try await coordinator.analyze(
                CanonicalImage(data: jpeg, width: cgImage.width, height: cgImage.height)
            )
            analysis = result
            draft = RouteDraft(analysis: result)
            if result.holds.isEmpty { analyzeError = "未检测到握点" }
        } catch {
            analyzeError = "分析失败"
        }
    }

    /// First tap on an empty draft = seed; later taps add/remove holds.
    func tapHold(id: Int) {
        guard let analysis, var current = draft else { return }
        if current.selectedHoldIDs.isEmpty, current.manualAdditions.isEmpty {
            guard let index = analysis.holds.firstIndex(where: { $0.id == id }) else { return }
            current = current.selecting(seedIndex: index)
        } else {
            current = current.togglingDetectedHold(id: id)
        }
        draft = current
        // route reference color = seed hold's Lab (schema field; palette stays manual)
        if let seedID = current.startHoldID ?? current.selectedHoldIDs.first,
           let hold = analysis.holds.first(where: { $0.id == seedID }) {
            referenceLab = (hold.referenceLab.l, hold.referenceLab.a, hold.referenceLab.b)
        }
    }

    func resetSelection() {
        guard let analysis else { return }
        draft = RouteDraft(analysis: analysis)
    }

    /// Persisted holds: selected detections + manual additions, in display order.
    func routeHolds(from draft: RouteDraft?) -> [Hold] {
        guard let analysis, let draft else { return [] }
        var result: [Hold] = []
        for hold in analysis.holds where draft.selectedHoldIDs.contains(hold.id) {
            result.append(makeHold(hold.detection.geometry,
                                   isStart: hold.id == draft.startHoldID,
                                   isFinish: hold.id == draft.finishHoldID))
        }
        for manual in draft.manualAdditions {
            result.append(makeHold(manual.geometry,
                                   isStart: manual.id == draft.startHoldID,
                                   isFinish: manual.id == draft.finishHoldID))
        }
        return result
    }

    private func makeHold(_ geometry: HoldGeometry, isStart: Bool, isFinish: Bool) -> Hold {
        Hold(
            centroidX: geometry.centroid.x,
            centroidY: geometry.centroid.y,
            bboxX: geometry.bboxX,
            bboxY: geometry.bboxY,
            bboxWidth: geometry.bboxWidth,
            bboxHeight: geometry.bboxHeight,
            maskWidth: geometry.maskWidth,
            maskHeight: geometry.maskHeight,
            maskRLE: geometry.maskRLE,
            isStart: isStart,
            isFinish: isFinish
        )
    }
}

struct AddRecordFlow: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @State private var model = AddRecordModel()
    @State private var showCamera = false
    @State private var showGymSearch = false
    @State private var photoItem: PhotosPickerItem?
    @State private var toastText: String?
    @State private var toastTask: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            Form {
                Section("照片") {
                    if let photo = model.photo {
                        HoldOverlayPhoto(photo: photo, draft: model.draft) { id in
                            model.tapHold(id: id)
                        }
                        .frame(maxHeight: 320)
                        analysisStatus
                    } else {
                        ContentUnavailableView("未添加照片", systemImage: "photo")
                            .frame(height: 140)
                    }
                    HStack {
                        PhotosPicker(selection: $photoItem, matching: .images) {
                            Label("从相册", systemImage: "photo.on.rectangle")
                        }
                        Button { showCamera = true } label: {
                            Label("拍照", systemImage: "camera")
                        }
                    }
                }

                Section("路线") {
                    // gym: map-POI selection (PLAN v1.2.1, no gym list maintenance)
                    Button {
                        showGymSearch = true
                    } label: {
                        LabeledContent("岩馆") {
                            Text(model.gymName.isEmpty ? "选择" : model.gymName)
                                .foregroundStyle(model.gymName.isEmpty ? Theme.muted : Theme.text)
                        }
                    }
                    Picker("等级", selection: $model.grade) {
                        ForEach(["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"], id: \.self) {
                            Text($0)
                        }
                    }
                    Picker("结果", selection: $model.result) {
                        Text("Flash").tag(RouteResult.flash)
                        Text("完攀 / Top").tag(RouteResult.top)
                        Text("Project").tag(RouteResult.project)
                    }
                    Stepper("尝试 \(model.attempts) 次", value: $model.attempts, in: 1...99)
                    TextField("备注（可选）", text: $model.note, axis: .vertical)
                }

                Section("路线颜色") {
                    HStack(spacing: 12) {
                        ForEach(RouteColor.allCases, id: \.self) { c in
                            Circle()
                                .fill(Theme.routeColor(c))
                                .frame(width: 28, height: 28)
                                .overlay {
                                    if c == model.palette {
                                        Circle().strokeBorder(Theme.accent, lineWidth: 3)
                                    }
                                }
                                .onTapGesture { model.palette = c }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("添加路线")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("保存") { save() }
                        .disabled(model.photo == nil)
                }
            }
            .fullScreenCover(isPresented: $showCamera) { CameraPicker { image in
                model.photo = image
                Task { await model.analyze(image) }
            } }
            .sheet(isPresented: $showGymSearch) { GymSearchView { name, coord, id in
                model.gymName = name
                model.gymCoordinate = coord
                model.gymMapItemID = id
            } }
            .onChange(of: photoItem) { _, item in
                Task {
                    if let data = try? await item?.loadTransferable(type: Data.self),
                       let img = UIImage(data: data) {
                        model.photo = img
                        await model.analyze(img)
                    }
                }
            }
            .overlay(alignment: .bottom) {
                if let toastText {
                    Text(toastText)
                        .font(.subheadline.bold())
                        .foregroundStyle(Theme.accentText)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Theme.accent, in: Capsule())
                        .padding(.bottom, 24)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .preferredColorScheme(.dark)
        }
    }

    @ViewBuilder
    private var analysisStatus: some View {
        if model.isAnalyzing {
            HStack(spacing: 8) {
                ProgressView()
                Text("分析握点…")
            }
            .font(.footnote)
            .foregroundStyle(Theme.muted)
        } else if let error = model.analyzeError {
            Text(error)
                .font(.footnote)
                .foregroundStyle(.red)
        } else if let draft = model.draft {
            HStack {
                Text(selectionStatus(draft))
                    .font(.footnote)
                    .foregroundStyle(Theme.muted)
                Spacer()
                if !draft.selectedHoldIDs.isEmpty || !draft.manualAdditions.isEmpty {
                    Button("重选") { model.resetSelection() }
                        .font(.footnote)
                }
            }
        }
    }

    private func selectionStatus(_ draft: RouteDraft) -> String {
        let count = draft.selectedHoldIDs.count + draft.manualAdditions.count
        if count == 0 { return "点击一个握点作为种子" }
        return "已选 \(count) 个握点 · 点击握点增减"
    }

    private func save() {
        guard let photo = model.photo else { return }
        let filename = saveCanonicalImage(photo)
        let route = ClimbRoute(
            name: "\(model.palette.rawValue.capitalized)线",
            gymNameSnapshot: model.gymName.isEmpty ? "未记录" : model.gymName,
            gymLatitude: model.gymCoordinate?.latitude,
            gymLongitude: model.gymCoordinate?.longitude,
            gymMapItemID: model.gymMapItemID,
            gradeValue: model.grade,
            referenceL: model.referenceLab.0,
            referenceA: model.referenceLab.1,
            referenceB: model.referenceLab.2,
            paletteColor: model.palette,
            result: model.result,
            attempts: model.attempts,
            durationSeconds: Int(model.elapsed),
            note: model.note,
            date: .now,
            photoFilename: filename,
            segmenterModelVersion: model.analysis?.modelVersion,
            routeSelectorVersion: model.analysis?.routeSelectorVersion,
            inferenceInputSize: model.analysis?.inputSize,
            manualAdditions: model.draft?.manualAddedDetectedHoldIDs.count ?? 0,
            manualRemovals: model.draft?.manualRemovedDetectedHoldIDs.count ?? 0
        )
        route.holds = model.routeHolds(from: model.draft)
        modelContext.insert(route)
        try? modelContext.save()
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        showToast("已保存 ✓")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { dismiss() }
    }

    private func showToast(_ text: String) {
        toastTask?.cancel()
        withAnimation(.spring(duration: 0.35)) { toastText = text }
        toastTask = Task {
            try? await Task.sleep(for: .seconds(1.4))
            withAnimation(.easeOut(duration: 0.3)) { toastText = nil }
        }
    }

    /// fixOrientation + sRGB + downscale via PhotoCanonicalizer, then save
    /// canonical JPEG to Documents/Photos (PLAN §4, M1).
    private func saveCanonicalImage(_ image: UIImage) -> String? {
        guard let data = PhotoCanonicalizer.canonicalJPEG(from: image) else { return nil }
        let dir = URL.documentsDirectory.appending(path: "Photos")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appending(path: "\(UUID().uuidString).jpg")
        do { try data.write(to: url); return url.lastPathComponent }
        catch { return nil }
    }
}

/// M2: canonical photo + tap-to-select hold overlay. First tap seeds the
/// route selector; later taps toggle individual holds (correction).
struct HoldOverlayPhoto: View {
    let photo: UIImage
    let draft: RouteDraft?
    let onTap: (Int) -> Void

    var body: some View {
        GeometryReader { geo in
            let fitted = Self.fittedRect(photoSize: photo.size, in: geo.size)
            ZStack {
                Image(uiImage: photo).resizable().scaledToFit()
                if let draft, let analysis = draft.analysis {
                    ForEach(analysis.holds) { hold in
                        let centroid = hold.detection.geometry.centroid
                        let selected = draft.selectedHoldIDs.contains(hold.id)
                        Circle()
                            .fill(selected ? Theme.accent : Color.white.opacity(0.9))
                            .frame(width: 16, height: 16)
                            .overlay(Circle().stroke(Color.black.opacity(0.4), lineWidth: 1))
                            .position(
                                x: fitted.minX + centroid.x * fitted.width,
                                y: fitted.minY + centroid.y * fitted.height
                            )
                            .contentShape(Circle().inset(by: -8))
                            .onTapGesture { onTap(hold.id) }
                    }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    /// aspect-fit rect of the photo inside the container (matches scaledToFit).
    static func fittedRect(photoSize: CGSize, in container: CGSize) -> CGRect {
        guard photoSize.width > 0, photoSize.height > 0 else { return .zero }
        let scale = min(container.width / photoSize.width,
                        container.height / photoSize.height)
        let size = CGSize(width: photoSize.width * scale, height: photoSize.height * scale)
        return CGRect(x: (container.width - size.width) / 2,
                      y: (container.height - size.height) / 2,
                      width: size.width, height: size.height)
    }
}

/// Wrapped UIImagePickerController for camera (PLAN D9).
/// Standard status (verified 2026-08): UIImagePickerController's photo-library
/// half is superseded by PHPickerViewController, but its camera half is NOT
/// deprecated (Apple Dev Forums thread 702859). For simple one-off captures
/// the system picker is the recommended option; AVFoundation
/// (AVCaptureSession + preview layer) is the upgrade path only if we need
/// custom controls/overlays or continuous scanning. NSCameraUsageDescription
/// is declared in Info.plist.
struct CameraPicker: UIViewControllerRepresentable {
    let onCapture: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }
        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.onCapture(image)
            }
            parent.dismiss()
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

/// MKLocalSearch POI selection — user picks any gym in the map (PLAN v1.2.1).
struct GymSearchView: View {
    let onPick: (String, CLLocationCoordinate2D, String?) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var results: [MKMapItem] = []

    var body: some View {
        NavigationStack {
            List(Array(results.enumerated()), id: \.offset) { _, item in
                Button {
                    let mapItemID: String?
                    if #available(iOS 18.0, *) {
                        mapItemID = item.identifier?.rawValue
                    } else {
                        mapItemID = nil  // MKMapItem.identifier is iOS 18+; name+coords still work
                    }
                    onPick(item.name ?? "未知岩馆",
                           item.placemark.coordinate,
                           mapItemID)
                    dismiss()
                } label: {
                    VStack(alignment: .leading) {
                        Text(item.name ?? "—").foregroundStyle(Theme.text)
                        if let city = item.placemark.locality {
                            Text(city).font(.caption).foregroundStyle(Theme.muted)
                        }
                    }
                }
            }
            .searchable(text: $query, prompt: "搜索岩馆（如 HANGDOG）")
            .onSubmit(of: .search) { search() }
            .navigationTitle("选择岩馆")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("取消") { dismiss() } }
            }
            .preferredColorScheme(.dark)
        }
    }

    private func search() {
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = query
        request.resultTypes = .pointOfInterest
        let search = MKLocalSearch(request: request)
        search.start { response, _ in
            results = response?.mapItems ?? []
        }
    }
}
#endif  // os(iOS)
