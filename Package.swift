// swift-tools-version: 5.10
import PackageDescription

var products: [Product] = [
    .library(name: "CRUXCore", targets: ["CRUXCore"]),
]
var targets: [Target] = [
    // Pure logic (color math, route selection) — compiles on Windows/macOS (no Apple frameworks).
    .target(name: "CRUXCore", path: "Sources/CRUXCore"),
    .testTarget(name: "CRUXCoreTests", dependencies: ["CRUXCore"], path: "Tests/CRUXCoreTests"),
]
var dependencies: [Package.Dependency] = []

#if !os(Windows)
dependencies.append(
    .package(
        url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
        from: "1.19.2"
    )
)
products.insert(.library(name: "CRUXClient", targets: ["CRUXClient"]), at: 0)
targets.insert(
    // iOS client UI: SwiftUI/UIKit/SwiftData. The Xcode app target is
    // still named CRUX; this package target makes the Client/Core split explicit.
    .target(
        name: "CRUXClient",
        dependencies: [
            "CRUXCore",
            .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
        ],
        path: "Sources/CRUX",
        sources: ["ONNXHoldSegmenter.swift"]
    ),
    at: 0
)
#endif

let package = Package(
    name: "CRUX",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: products,
    dependencies: dependencies,
    targets: targets
)
