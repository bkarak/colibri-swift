// swift-tools-version:6.0
//
// colibri-swift: a Swift + Metal port of colibri (https://github.com/JustVugg/colibri,
// Apache-2.0) for Apple Silicon. See docs/port/architecture.md for the module map.

import PackageDescription

// The engine targets use the Swift 5 language mode. The compute code passes raw
// pointers into `DispatchQueue.concurrentPerform` closures and keeps mutable caches in
// classes shared with I/O threads; Swift 6 strict concurrency would demand
// `@unchecked Sendable` wrappers on nearly every hot-path type without adding safety
// (the synchronisation is explicit: locks, conditions, Metal command completion).
//
// Release builds of the engine use -Ounchecked: the kernels index raw buffers in tight
// loops and the bounds/overflow traps of -O measurably cost there. -Ounchecked also
// removes `precondition`, so nothing that guards against wrong output may rely on one:
// backend limits are enforced when a model loads (`Backend.checkSupport`, which throws),
// and the Metal backend's per-op checks use `fatalError`, which stays. The debug-build
// test suite keeps every check.
let engineSettings: [SwiftSetting] = [
    .swiftLanguageMode(.v5),
    .unsafeFlags(["-Ounchecked"], .when(configuration: .release)),
]

let package = Package(
    name: "colibri-swift",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "ColibriCore", targets: ["ColibriCore"]),
        .library(name: "ColibriMetal", targets: ["ColibriMetal"]),
        .executable(name: "coli-swift", targets: ["coli-swift"]),
    ],
    targets: [
        // NEON integer kernels Swift does not vectorise (static inline, no dependencies).
        .target(name: "CColibriKernels"),
        .target(
            name: "ColibriCore",
            dependencies: ["CColibriKernels"],
            swiftSettings: engineSettings
        ),
        .target(
            name: "ColibriMetal",
            dependencies: ["ColibriCore"],
            swiftSettings: engineSettings,
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .executableTarget(
            name: "coli-swift",
            dependencies: ["ColibriCore", "ColibriMetal"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ColibriCoreTests",
            dependencies: ["ColibriCore"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ColibriMetalTests",
            dependencies: ["ColibriCore", "ColibriMetal"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
