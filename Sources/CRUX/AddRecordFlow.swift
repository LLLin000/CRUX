// Add-record flow — camera/gallery -> form -> save (PLAN M1; gym via MKLocalSearch)
// Camera = wrapped UIImagePickerController; gallery = PhotosPicker (no permission).
// Entire file is iOS-only (UIKit): guarded so the CRUXCore test chain can
// build the CRUX target on macOS for `swift test` (full-package build).

#if os(iOS)
import SwiftUI
import SwiftData
import PhotosUI
import MapKit

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

    // route spine / hold data filled by segmentation in M2; schema v1 fields exist now
    var holds: [Hold] = []
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
                        Image(uiImage: photo)
                            .resizable().scaledToFit()
                            .frame(maxHeight: 240)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
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
            .fullScreenCover(isPresented: $showCamera) { CameraPicker { model.photo = $0 } }
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
            segmenterModelVersion: nil,
            routeSelectorVersion: "seeded-deltaE-dbscan-v1",
            inferenceInputSize: nil
        )
        route.holds = model.holds
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

/// Wrapped UIImagePickerController for camera (PLAN D9).
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
