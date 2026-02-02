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
    dependencies: [
        .package(url: "https://github.com/sunlubo/SwiftFFmpeg.git", branch: "master"),
        .package(url: "https://github.com/apple/swift-testing.git", from: "0.10.0")
    ],
    targets: [
        .target(
            name: "HTDemucsKit",
            dependencies: [
                .product(name: "SwiftFFmpeg", package: "SwiftFFmpeg")
            ]
        ),
        .executableTarget(
            name: "htdemucs-cli",
            dependencies: ["HTDemucsKit"]
        ),
        .testTarget(
            name: "HTDemucsKitTests",
            dependencies: [
                "HTDemucsKit",
                .product(name: "Testing", package: "swift-testing")
            ]
        )
    ]
)
