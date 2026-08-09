// Pure-Swift replacement for the original EdgeRunnerSharedTypes C target.
//
// IMPORTANT: These structs are sent to Metal with MTLComputeCommandEncoder.setBytes.
// Every stored field intentionally remains a 32-bit scalar and in the exact same
// declaration order as the upstream C headers.

@frozen
public struct ERDType: RawRepresentable, Equatable, Hashable, Sendable {
    public let rawValue: UInt32

    @inlinable
    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let float32 = ERDType(rawValue: 0)
    public static let float16 = ERDType(rawValue: 1)
    public static let int8 = ERDType(rawValue: 2)
    public static let uInt8 = ERDType(rawValue: 3)
}

@frozen
public struct ERKVPrecision: RawRepresentable, Equatable, Hashable, Sendable {
    public let rawValue: UInt32

    @inlinable
    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let float32 = ERKVPrecision(rawValue: 0)
    public static let float16 = ERKVPrecision(rawValue: 1)
    public static let float8 = ERKVPrecision(rawValue: 2)
}

@frozen
public struct ERElementwiseParams: Sendable {
    public var elementCount: UInt32

    @inlinable
    public init(elementCount: UInt32 = 0) {
        self.elementCount = elementCount
    }
}

@frozen
public struct ERReductionParams: Sendable {
    public var elementCount: UInt32
    public var reductionSize: UInt32
    public var outerSize: UInt32

    @inlinable
    public init(
        elementCount: UInt32 = 0,
        reductionSize: UInt32 = 0,
        outerSize: UInt32 = 0
    ) {
        self.elementCount = elementCount
        self.reductionSize = reductionSize
        self.outerSize = outerSize
    }
}

@frozen
public struct ERTransposeParams: Sendable {
    public var rows: UInt32
    public var cols: UInt32

    @inlinable
    public init(rows: UInt32 = 0, cols: UInt32 = 0) {
        self.rows = rows
        self.cols = cols
    }
}

@frozen
public struct ERGEMMParams: Sendable {
    public var M: UInt32
    public var N: UInt32
    public var K: UInt32
    public var lda: UInt32
    public var ldb: UInt32
    public var ldc: UInt32

    @inlinable
    public init(
        M: UInt32 = 0,
        N: UInt32 = 0,
        K: UInt32 = 0,
        lda: UInt32 = 0,
        ldb: UInt32 = 0,
        ldc: UInt32 = 0
    ) {
        self.M = M
        self.N = N
        self.K = K
        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc
    }
}

@frozen
public struct ERGEMVParams: Sendable {
    public var M: UInt32
    public var K: UInt32
    public var lda: UInt32

    @inlinable
    public init(M: UInt32 = 0, K: UInt32 = 0, lda: UInt32 = 0) {
        self.M = M
        self.K = K
        self.lda = lda
    }
}

@frozen
public struct ERSoftmaxParams: Sendable {
    public var rows: UInt32
    public var cols: UInt32

    @inlinable
    public init(rows: UInt32 = 0, cols: UInt32 = 0) {
        self.rows = rows
        self.cols = cols
    }
}

@frozen
public struct ERFlashAttentionParams: Sendable {
    public var seqLen: UInt32
    public var headDim: UInt32
    public var scale: Float
    public var causal: UInt32
    public var kvBlockSize: UInt32
    public var qBlockSize: UInt32

    @inlinable
    public init(
        seqLen: UInt32 = 0,
        headDim: UInt32 = 0,
        scale: Float = 0,
        causal: UInt32 = 0,
        kvBlockSize: UInt32 = 0,
        qBlockSize: UInt32 = 0
    ) {
        self.seqLen = seqLen
        self.headDim = headDim
        self.scale = scale
        self.causal = causal
        self.kvBlockSize = kvBlockSize
        self.qBlockSize = qBlockSize
    }
}

@frozen
public struct ERGQAParams: Sendable {
    public var seqLen: UInt32
    public var headDim: UInt32
    public var numHeads: UInt32
    public var numKVHeads: UInt32
    public var groupSize: UInt32
    public var scale: Float
    public var causal: UInt32
    public var kvBlockSize: UInt32
    public var qBlockSize: UInt32
    public var kvSeqLen: UInt32
    public var qOffset: UInt32

    @inlinable
    public init(
        seqLen: UInt32 = 0,
        headDim: UInt32 = 0,
        numHeads: UInt32 = 0,
        numKVHeads: UInt32 = 0,
        groupSize: UInt32 = 0,
        scale: Float = 0,
        causal: UInt32 = 0,
        kvBlockSize: UInt32 = 0,
        qBlockSize: UInt32 = 0,
        kvSeqLen: UInt32 = 0,
        qOffset: UInt32 = 0
    ) {
        self.seqLen = seqLen
        self.headDim = headDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.groupSize = groupSize
        self.scale = scale
        self.causal = causal
        self.kvBlockSize = kvBlockSize
        self.qBlockSize = qBlockSize
        self.kvSeqLen = kvSeqLen
        self.qOffset = qOffset
    }
}

@frozen
public struct ERKVCacheParams: Sendable {
    public var maxSeqLen: UInt32
    public var currentLen: UInt32
    public var writePos: UInt32
    public var numKVHeads: UInt32
    public var headDim: UInt32
    public var precision: UInt32

    @inlinable
    public init(
        maxSeqLen: UInt32 = 0,
        currentLen: UInt32 = 0,
        writePos: UInt32 = 0,
        numKVHeads: UInt32 = 0,
        headDim: UInt32 = 0,
        precision: UInt32 = 0
    ) {
        self.maxSeqLen = maxSeqLen
        self.currentLen = currentLen
        self.writePos = writePos
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.precision = precision
    }
}

@frozen
public struct ERRoPEParams: Sendable {
    public var seqLen: UInt32
    public var numHeads: UInt32
    public var headDim: UInt32
    public var startPos: UInt32
    public var theta: Float
    public var scalingFactor: Float
    public var partialRotaryFactor: Float

    @inlinable
    public init(
        seqLen: UInt32 = 0,
        numHeads: UInt32 = 0,
        headDim: UInt32 = 0,
        startPos: UInt32 = 0,
        theta: Float = 0,
        scalingFactor: Float = 0,
        partialRotaryFactor: Float = 0
    ) {
        self.seqLen = seqLen
        self.numHeads = numHeads
        self.headDim = headDim
        self.startPos = startPos
        self.theta = theta
        self.scalingFactor = scalingFactor
        self.partialRotaryFactor = partialRotaryFactor
    }
}

@frozen
public struct ERRMSNormParams: Sendable {
    public var rows: UInt32
    public var cols: UInt32
    public var eps: Float

    @inlinable
    public init(rows: UInt32 = 0, cols: UInt32 = 0, eps: Float = 0) {
        self.rows = rows
        self.cols = cols
        self.eps = eps
    }
}

@frozen
public struct ERLayerNormParams: Sendable {
    public var rows: UInt32
    public var cols: UInt32
    public var eps: Float

    @inlinable
    public init(rows: UInt32 = 0, cols: UInt32 = 0, eps: Float = 0) {
        self.rows = rows
        self.cols = cols
        self.eps = eps
    }
}

@frozen
public struct ERActivationParams: Sendable {
    public var count: UInt32

    @inlinable
    public init(count: UInt32 = 0) {
        self.count = count
    }
}

@frozen
public struct ERDequantParams: Sendable {
    public var blockCount: UInt32
    public var outputOffset: UInt32

    @inlinable
    public init(blockCount: UInt32 = 0, outputOffset: UInt32 = 0) {
        self.blockCount = blockCount
        self.outputOffset = outputOffset
    }
}

@frozen
public struct ERDequantGEMVParams: Sendable {
    public var rows: UInt32
    public var cols: UInt32
    public var blocksPerRow: UInt32

    @inlinable
    public init(rows: UInt32 = 0, cols: UInt32 = 0, blocksPerRow: UInt32 = 0) {
        self.rows = rows
        self.cols = cols
        self.blocksPerRow = blocksPerRow
    }
}

@frozen
public struct ERDequantQ4KParams: Sendable {
    public var superBlockCount: UInt32
    public var outputOffset: UInt32

    @inlinable
    public init(superBlockCount: UInt32 = 0, outputOffset: UInt32 = 0) {
        self.superBlockCount = superBlockCount
        self.outputOffset = outputOffset
    }
}
