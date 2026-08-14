// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "CRUX",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "CRUX", targets: ["CRUX"]),
    ],
    targets: [
        .target(name: "CRUX", path: "Sources/CRUX"),
        .testTarget(name: "CRUXTests", dependencies: ["CRUX"], path: "Tests/CRUXTests"),
    ]
)
