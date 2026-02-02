// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HTDemucsKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v18)
    ],
    products: [
        .library(
            name: "HTDemucsKit",
            targets: ["HTDemucsKit"]
        ),
        .executable(
            name: "htdemucs-cli",
            targets: ["htdemucs-cli"]
        )
    ],
    targets: [
        .target(
            name: "HTDemucsKit",
            dependencies: []
        ),
        .executableTarget(
            name: "htdemucs-cli",
            dependencies: ["HTDemucsKit"]
        ),
        .testTarget(
            name: "HTDemucsKitTests",
            dependencies: ["HTDemucsKit"]
        )
    ]
)
