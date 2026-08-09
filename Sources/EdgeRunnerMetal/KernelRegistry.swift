import Foundation
import Metal

package final class KernelRegistry {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    package init(device: MTLDevice) throws {
        self.device = device

        let options = MTLCompileOptions()

        self.library = try device.makeLibrary(
            source: EmbeddedMetalShaders.source,
            options: options
        )
    }

    package func pipeline(
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

    package var metalLibrary: MTLLibrary {
        library
    }
}

package enum KernelRegistryError: Error {
    case functionNotFound(String)
}
