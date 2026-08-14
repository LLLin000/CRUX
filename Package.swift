// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "CRUX",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "CRUX", targets: ["CRUX"]),
        .library(name: "CRUXCore", targets: ["CRUXCore"]),
    ],
    targets: [
        // Pure logic (color math, route selection) — compiles on Windows/macOS (no Apple frameworks).
        .target(name: "CRUXCore", path: "Sources/CRUXCore"),
        // UI layer: SwiftUI/UIKit, iOS-only. Not compiled by the CRUXCore test
        // chain (see codemagic.yaml: build --target CRUXCoreTests, then
        // swift test --skip-build); the Xcode project builds it for iOS.
        .target(name: "CRUX", dependencies: ["CRUXCore"], path: "Sources/CRUX"),
        .testTarget(name: "CRUXCoreTests", dependencies: ["CRUXCore"], path: "Tests/CRUXCoreTests"),
    ]
)
