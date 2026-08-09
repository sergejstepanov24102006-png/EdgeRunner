import Foundation
import Metal

public let ER_KERNEL_PROBE: Int = 12345

public final class KernelRegistry {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init(device: MTLDevice) throws {
        self.device = device

        let options = MTLCompileOptions()

        self.library = try device.makeLibrary(
            source: EmbeddedMetalShaders.source,
            options: options
        )
    }

    public func pipeline(
        for name: String
    ) throws -> MTLComputePipelineState {
        if let cached = pipelines[name] {
            return cached
        }

        guard let function = library.makeFunction(name: name) else {
            throw KernelRegistryError.functionNotFound(name)
        }

        let pipeline = try device.makeComputePipelineState(
            function: function
        )

        pipelines[name] = pipeline
        return pipeline
    }

    public var metalLibrary: MTLLibrary {
        library
    }
}

enum KernelRegistryError: Error {
    case functionNotFound(String)
}
