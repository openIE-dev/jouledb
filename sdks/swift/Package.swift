// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "JouleDBKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "JouleDBKit", targets: ["JouleDBKit"]),
    ],
    targets: [
        .target(name: "JouleDBKit"),
    ]
)
