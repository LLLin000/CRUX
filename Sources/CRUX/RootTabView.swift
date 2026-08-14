// CRUX root navigation — PLAN D2: 2 main tabs (今日|图鉴) + independent ＋.
// All sheets/pushes are state-driven values on view models (syncups pattern),
// no NavigationLink littered in bodies. Add-flow opens as fullScreenCover (camera).

import SwiftUI
import SwiftData

@Observable
final class AppRouter {
    enum Sheet: Identifiable {
        case addRecord
        var id: String { "addRecord" }
    }

    var sheet: Sheet?
    var showAdd: Bool {
        get { sheet != nil }
        set { sheet = newValue ? .addRecord : nil }
    }
}

struct RootTabView: View {
    @State private var router = AppRouter()
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        TabView {
            TodayView()
                .tabItem { Label("今日", systemImage: "calendar") }
            RoutesView()
                .tabItem { Label("图鉴", systemImage: "square.grid.3x3.fill") }
        }
        .tint(Theme.accent)
        .preferredColorScheme(.dark)
        .overlay(alignment: .bottom) { AddButton { router.showAdd = true } }
        .fullScreenCover(item: $router.sheet) { _ in
            AddRecordFlow()
        }
    }
}

/// Floating ＋ above the tab bar (prototype bottom-nav add button).
private struct AddButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "plus")
                .font(.system(size: 24, weight: .bold))
                .foregroundStyle(Theme.accentText)
                .frame(width: 58, height: 58)
                .background(Theme.accent, in: Circle())
                .shadow(color: Theme.accent.opacity(0.35), radius: 12, y: 4)
        }
        .padding(.bottom, 64)
    }
}

#Preview {
    RootTabView()
        .modelContainer(for: [ClimbRoute.self, Hold.self, RouteUnionMask.self], inMemory: true)
}
