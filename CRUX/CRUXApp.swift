// CRUX iOS app entry — @main (separate from SwiftPM library target, no conflict).
// SwiftData container wired at launch; screens under app/Sources/CRUX.

import SwiftUI
import SwiftData

@main
struct CRUXApp: App {
    let container: ModelContainer

    init() {
        do {
            container = try ModelContainer(
                for: ClimbRoute.self, Hold.self, RouteUnionMask.self
            )
        } catch {
            fatalError("ModelContainer init failed: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            RootTabView()
        }
        .modelContainer(container)
    }
}
