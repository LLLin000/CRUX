// Routes — 图鉴：过滤 + stamp 网格 (prototype stampGrid, PLAN M1)

import SwiftUI
import SwiftData

enum RouteFilter: String, CaseIterable, Identifiable {
    case all = "全部"
    case top = "已完攀"
    case project = "Project"
    case v5 = "V5+"
    var id: String { rawValue }
}

struct RoutesView: View {
    @Query(sort: \ClimbRoute.date, order: .reverse) private var routes: [ClimbRoute]
    @State private var filter: RouteFilter = .all

    private var filtered: [ClimbRoute] {
        switch filter {
        case .all: return routes
        case .top: return routes.filter { $0.result != .project }
        case .project: return routes.filter { $0.result == .project }
        case .v5: return routes.filter { vGrade($0) >= 5 }
        }
    }

    private func vGrade(_ r: ClimbRoute) -> Int {
        Int(r.gradeValue.replacingOccurrences(of: "V", with: "")) ?? 0
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 3),
                          spacing: 12) {
                    ForEach(filtered) { route in
                        StampCell(route: route)
                    }
                }
                .padding(14)
            }
            .background(Theme.bg)
            .navigationTitle("路线图鉴")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        ForEach(RouteFilter.allCases) { f in
                            Button(f.rawValue) { filter = f }
                        }
                    } label: {
                        Image(systemName: "line.3.horizontal.decrease.circle")
                            .foregroundStyle(Theme.text)
                    }
                }
            }
        }
    }
}

/// Colored-ring stamp card (prototype .stamp). Route identity = paletteColor.
struct StampCell: View {
    let route: ClimbRoute

    var body: some View {
        VStack(spacing: 6) {
            ZStack {
                Circle().stroke(Theme.routeColor(route.paletteColor).opacity(0.55), lineWidth: 3)
                    .frame(width: 64, height: 64)
                Circle().fill(Theme.card2).frame(width: 52, height: 52)
                Text(route.gradeValue)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.routeColor(route.paletteColor))
            }
            Text(RouteStatusBadge.label(for: route.result))
                .font(.caption2.bold()).foregroundStyle(Theme.muted)
            Text(route.name)
                .font(.caption).foregroundStyle(Theme.text).lineLimit(1)
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 18))
    }
}
