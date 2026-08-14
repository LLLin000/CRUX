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
    @Environment(\.modelContext) private var modelContext
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
            Group {
                if filtered.isEmpty {
                    ContentUnavailableView(
                        routes.isEmpty ? "还没有路线" : "没有匹配的路线",
                        systemImage: routes.isEmpty ? "figure.climbing" : "line.3.horizontal.decrease.circle",
                        description: Text(routes.isEmpty ? "去拍一张墙，开始你的第一本攀岩日志" : "换个筛选条件试试")
                    )
                } else {
                    ScrollView {
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 3),
                                  spacing: 12) {
                            ForEach(filtered) { route in
                                NavigationLink(value: route) {
                                    StampCell(route: route)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(14)
                    }
                }
            }
            .background(Theme.bg)
            .navigationTitle("路线图鉴")
            .navigationDestination(for: ClimbRoute.self) { route in
                RouteDetailView(route: route)
            }
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {  // topBarTrailing on iOS, primaryAction on macOS
                    #if DEBUG && os(iOS)
                    Button {
                        RouteDemoData.insertDemoRoute(context: modelContext)
                    } label: {
                        Image(systemName: "wand.and.stars")
                            .foregroundStyle(Theme.accent)
                    }
                    .accessibilityLabel("插入演示路线")
                    #endif
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
