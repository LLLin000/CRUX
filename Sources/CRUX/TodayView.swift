// Today — 日历月视图 + 月摘要 (climbfolio SessionMonthCalendar pattern, PLAN M1)

import SwiftUI
import SwiftData

struct TodayView: View {
    @Query(sort: \ClimbRoute.date) private var routes: [ClimbRoute]
    @State private var month: Date = .now

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    CalendarMonthView(month: month, routes: routes) { day in
                        // push day detail (state-driven, PLAN D2)
                    }
                    .padding(.top, 8)

                    MonthSummaryCard(routes: routes)
                }
                .padding(.horizontal, 18)
            }
            .background(Theme.bg)
            .navigationTitle("今日")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { /* settings — secondary entry (PLAN D2) */ } label: {
                        Image(systemName: "gearshape").foregroundStyle(Theme.text)
                    }
                }
            }
        }
    }
}

struct CalendarMonthView: View {
    let month: Date
    let routes: [ClimbRoute]
    let onSelect: (Date) -> Void

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 6), count: 7)
    private let weekdays = ["一", "二", "三", "四", "五", "六", "日"]

    var body: some View {
        VStack(spacing: 8) {
            Text(month.formatted(.dateTime.year().month()))
                .font(.headline).foregroundStyle(Theme.text)
            HStack {
                ForEach(weekdays, id: \.self) {
                    Text($0).font(.caption2).foregroundStyle(Theme.muted)
                        .frame(maxWidth: .infinity)
                }
            }
            LazyVGrid(columns: columns, spacing: 6) {
                ForEach(days, id: \.self) { day in
                    DayCell(day: day, colors: colors(on: day))
                        .onTapGesture { onSelect(day) }
                }
            }
        }
        .padding(14)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: Theme.cornerCard))
    }

    private var days: [Date] {
        let cal = Calendar.current
        guard let range = cal.range(of: .day, in: .month, for: month),
              let first = cal.date(from: cal.dateComponents([.year, .month], from: month))
        else { return [] }
        let firstWeekday = (cal.component(.weekday, from: first) + 5) % 7  // 周一开头
        return (0..<(firstWeekday + range.count)).compactMap {
            cal.date(byAdding: .day, value: $0 - firstWeekday, to: first)
        }
    }

    private func colors(on day: Date) -> [Color] {
        let cal = Calendar.current
        return routes
            .filter { cal.isDate($0.date, inSameDayAs: day) }
            .map { Theme.routeColor($0.paletteColor) }
    }
}

private struct DayCell: View {
    let day: Date
    let colors: [Color]

    var body: some View {
        ZStack {
            Circle().fill(Theme.card2)
            Text(day.formatted(.dateTime.day()))
                .font(.caption).foregroundStyle(Theme.text)
            VStack {
                Spacer()
                HStack(spacing: 2) {
                    ForEach(colors.prefix(3), id: \.self) { c in
                        Circle().fill(c).frame(width: 5, height: 5)
                    }
                }
            }
            .padding(.bottom, 3)
        }
        .frame(height: 34)
    }
}

struct MonthSummaryCard: View {
    let routes: [ClimbRoute]

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("本月").font(.caption).foregroundStyle(Theme.muted)
                Text("\(routes.count)")
                    .font(.system(size: 30, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.text)
                Text("条路线").font(.caption).foregroundStyle(Theme.muted)
            }
            Spacer()
            ForEach(Set(routes.map(\.paletteColor)).sorted { $0.rawValue < $1.rawValue }.prefix(5), id: \.self) { c in
                Circle().fill(Theme.routeColor(c)).frame(width: 12, height: 12)
            }
        }
        .padding(18)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: Theme.cornerCard))
    }
}
