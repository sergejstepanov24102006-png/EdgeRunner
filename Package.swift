// swift-tools-version: 6.2

import PackageDescription
#error("FRESH_EDGERUNNER_REVISION")

let package = Package(
    name: "EdgeRunner",

    platforms: [
        .iOS(.v26),
        .macOS(.v26)
    ],

    products: [
        .library(
            name: "EdgeRunner",
            targets: ["EdgeRunner"]
        )
    ],

    targets: [

        // MARK: - Shared types
        // C-target убран для совместимости со Swift Playgrounds.
        .target(
            name: "EdgeRunnerSharedTypes",
            path: "Sources/EdgeRunnerSharedTypes",
            exclude: [
                "ShaderTypes.c",
                "include"
            ],
            sources: [
                "ShaderTypes.swift"
            ]
        ),

        // MARK: - Metal runtime
        // Папки Shaders физически уже нет.
        // Metal-код находится в EmbeddedMetalShaders.swift
        // и компилируется на iPad во время запуска.
        .target(
            name: "EdgeRunnerMetal",
            dependencies: [
                "EdgeRunnerSharedTypes"
            ],
            path: "Sources/EdgeRunnerMetal",
            swiftSettings: [
                .unsafeFlags([
                    "-package-name",
                    "EdgeRunner"
                ])
            ]
        ),

        // MARK: - IO
        .target(
            name: "EdgeRunnerIO",
            dependencies: [
                "EdgeRunnerMetal"
            ],
            path: "Sources/EdgeRunnerIO",
            swiftSettings: [
                .unsafeFlags([
                    "-package-name",
                    "EdgeRunner"
                ])
            ]
        ),

        // MARK: - Core
        .target(
            name: "EdgeRunnerCore",
            dependencies: [
                "EdgeRunnerMetal",
                "EdgeRunnerSharedTypes",
                "EdgeRunnerIO"
            ],
            path: "Sources/EdgeRunnerCore",
            swiftSettings: [
                .unsafeFlags([
                    "-package-name",
                    "EdgeRunner"
                ])
            ]
        ),

        // MARK: - Public EdgeRunner API
        .target(
            name: "EdgeRunner",
            dependencies: [
                "EdgeRunnerCore",
                "EdgeRunnerIO",
                "EdgeRunnerSharedTypes"
            ],
            path: "Sources/EdgeRunner",
            swiftSettings: [
                .unsafeFlags([
                    "-package-name",
                    "EdgeRunner"
                ])
            ]
        )
    ]
)
