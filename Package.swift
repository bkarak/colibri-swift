// swift-tools-version:6.0
//
// colibri-swift: a Swift + Metal port of colibri (https://github.com/JustVugg/colibri,
// Apache-2.0) for Apple Silicon. See docs/port/architecture.md for the module map.

import PackageDescription

// The engine targets use the Swift 5 language mode. The compute code passes raw
// pointers into parallel-loop closures (`Parallel`) and keeps mutable caches in
// classes shared with I/O threads; Swift 6 strict concurrency would demand
// `@unchecked Sendable` wrappers on nearly every hot-path type without adding safety
// (the synchronisation is explicit: locks, conditions, Metal command completion).
//
// No unsafe build flags: SwiftPM refuses them in a package that is a version-pinned
// dependency, so they would keep the package from being used as a library. Release builds
// are SwiftPM's -O (Swift) and -Os (C). The C kernels get what they need from the source
// (forced inlining and explicitly unrolled loops in rows_i8.c; with those, -Os matches
// -O3), and -O instead of -Ounchecked costs nothing measurable on Metal and up to ~7% of
// CPU decode on Qwen3-0.6B (none measurable on the 4B and 30B). Preconditions stay on in
// release builds; backend limits are still enforced when a model loads.
let engineSettings: [SwiftSetting] = [
    .swiftLanguageMode(.v5),
]

let package = Package(
    name: "colibri-swift",
    platforms: [.macOS(.v15)],
    products: [
        // The facade: load a model, stream text (Sources/Colibri). Re-exports ColibriCore.
        .library(name: "Colibri", targets: ["Colibri"]),
        .library(name: "ColibriCore", targets: ["ColibriCore"]),
        .library(name: "ColibriMetal", targets: ["ColibriMetal"]),
        .executable(name: "coli-swift", targets: ["coli-swift"]),
    ],
    targets: [
        // NEON kernels and the CPU worker pool (C, no dependencies).
        .target(name: "CColibriKernels"),
        .target(
            name: "ColibriCore",
            dependencies: ["CColibriKernels"],
            swiftSettings: engineSettings
        ),
        // Swift 6 language mode: it is what apps call, so it is checked for data races.
        .target(name: "Colibri", dependencies: ["ColibriCore", "ColibriMetal"]),
        .testTarget(name: "ColibriTests", dependencies: ["Colibri"]),
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
