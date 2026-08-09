// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "EdgeRunner",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "EdgeRunner", targets: ["EdgeRunner"]),
    ],
    targets: [
        // Swift Playgrounds on iPad cannot build C-family targets.
        // Keep the same module name, but provide the shared ABI parameter
        // structs in pure Swift and exclude the original C headers/shim.
        .target(
            name: "EdgeRunnerSharedTypes",
            path: "Sources/EdgeRunnerSharedTypes",
            exclude: ["ShaderTypes.c", "include"],
            sources: ["ShaderTypes.swift"]
        ),
        .target(
            name: "EdgeRunnerMetal",
            dependencies: ["EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunnerMetal",
            resources: [.process("Shaders")]
        ),
        .target(
            name: "EdgeRunnerIO",
            dependencies: ["EdgeRunnerMetal"],
            path: "Sources/EdgeRunnerIO"
        ),
        .target(
            name: "EdgeRunnerCore",
            dependencies: ["EdgeRunnerMetal", "EdgeRunnerSharedTypes", "EdgeRunnerIO"],
            path: "Sources/EdgeRunnerCore"
        ),
        .target(
            name: "EdgeRunner",
            dependencies: ["EdgeRunnerCore", "EdgeRunnerIO", "EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunner"
        ),
    ]
)
