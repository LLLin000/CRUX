// CRUX design tokens — ported 1:1 from climb_crux_minimal_v4.html (PLAN §0 visual baseline)
// Dark theme: bg #0B0C0E, accent #C9FF45, route color system.

import SwiftUI

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: 1)
    }
}

enum Theme {
    static let bg        = Color(hex: 0x0B0C0E)   // app background
    static let card      = Color(hex: 0x15171A)   // cards
    static let card2     = Color(hex: 0x1B1E22)   // nested surfaces
    static let line      = Color(hex: 0x272B30)   // hairline separators
    static let text      = Color(hex: 0xF5F6F7)
    static let muted     = Color(hex: 0x858B93)
    static let accent    = Color(hex: 0xC9FF45)   // primary action, selected day
    static let accentText = Color(hex: 0x11150A)

    static let cornerCard: CGFloat = 28
    static let cornerPill: CGFloat = 999

    static func routeColor(_ c: RouteColor) -> Color {
        switch c {
        case .blue:   return Color(hex: 0x4AA8FF)
        case .red:    return Color(hex: 0xFF6675)
        case .yellow: return Color(hex: 0xFFD34D)
        case .green:  return Color(hex: 0x79DB72)
        case .purple: return Color(hex: 0xAF79F7)
        case .black:  return Color(hex: 0x3A3D42)
        case .gray:   return Color(hex: 0x9AA0A6)
        case .orange: return Color(hex: 0xFF9F43)
        }
    }
}

// Route status derivation is NOT done here — result is user-chosen and stored
// (PLAN §5). These are display helpers only.
enum RouteStatusBadge {
    static func label(for result: RouteResult) -> String {
        switch result {
        case .flash: return "FLASH"
        case .top: return "TOP"
        case .project: return "PROJECT"
        }
    }
}
