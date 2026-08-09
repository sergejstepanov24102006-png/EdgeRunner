// Auto-generated for Swift Playgrounds on iPad.
// The original .metal files are excluded from SwiftPM and compiled at runtime.

import Foundation

package enum EmbeddedMetalShaders {
    package static let source = #"""
// --- Activations.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERActivationParams {
    uint count;
};

kernel void sigmoid_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERActivationParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    float value = input[gid];
    output[gid] = 1.0f / (1.0f + exp(-value));
}

kernel void gelu_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERActivationParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    float value = input[gid];
    const float coefficient = 0.7978845608028654f;
    output[gid] = value * 0.5f * (1.0f + tanh(coefficient * (value + 0.044715f * value * value * value)));
}

inline float silu(float value) {
    return value / (1.0f + exp(-value));
}

kernel void swiglu_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERActivationParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    output[gid] = silu(gate[gid]) * up[gid];
}


// --- Dequant_Q1_0_g128.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ1_0_g128Params {
    uint blockCount;
    uint outputOffset;
    uint scaleByteOffset;
    uint bitDataOffset;
    uint bitOrderMSBFirst;
    uint oneBitIsNegative;
};

struct ERDequantQ1_0_g128GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint q1_0_g128BlockBytes = 18;
constant uint q1_0_g128WeightsPerBlock = 128;

// ─── Shared v2 helper ────────────────────────────────────────────────
// Computes the bit-selected sum B for one 4-byte (32-weight) sub-block.
// Returns: scale × (2×B - xSum), the dot product contribution.
inline float q1_subblock_dot(
    thread const float* xl, // 32 cached x (or normed-x) values in thread address space
    float xSum,             // precomputed sum of xl[0..31]
    device const uchar* qs, // 4 bytes of Q1 bit data for this sub-block
    float scale             // Q1 block scale (shared across 4 sub-blocks)
) {
    float B = 0.f;
    for (short bi = 0; bi < 4; bi++) {
        uchar bits = qs[bi];
        const short base = bi * 8;
        // Vectorized: extract bit masks → float4, dot with x values
        float4 x0 = float4(xl[base+0], xl[base+1], xl[base+2], xl[base+3]);
        float4 x1 = float4(xl[base+4], xl[base+5], xl[base+6], xl[base+7]);
        float4 m0 = float4(float((bits>>0)&1), float((bits>>1)&1),
                           float((bits>>2)&1), float((bits>>3)&1));
        float4 m1 = float4(float((bits>>4)&1), float((bits>>5)&1),
                           float((bits>>6)&1), float((bits>>7)&1));
        B += dot(m0, x0) + dot(m1, x1);
    }
    return scale * (2.f * B - xSum);
}

// ─── Basic dequantization kernel ─────────────────────────────────────

kernel void dequant_q1_0_g128(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ1_0_g128Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;
    device const uchar* block = input + (tid * q1_0_g128BlockBytes);
    uint scaleOffset = min(params.scaleByteOffset, q1_0_g128BlockBytes - 2);
    float scale = float(as_type<half>(*(device const ushort*)(block + scaleOffset)));
    uint outputBase = params.outputOffset + (tid * q1_0_g128WeightsPerBlock);
    for (uint byteIndex = 0; byteIndex < 16; byteIndex++) {
        uint bitOffset = min(params.bitDataOffset + byteIndex, q1_0_g128BlockBytes - 1);
        uchar bits = block[bitOffset];
        uint baseOutput = outputBase + (byteIndex * 8);
        for (uint bitIndex = 0; bitIndex < 8; bitIndex++) {
            uint shift = params.bitOrderMSBFirst != 0 ? (7u - bitIndex) : bitIndex;
            uchar bit = (bits >> shift) & 1;
            bool positive = params.oneBitIsNegative != 0 ? (bit == 0) : (bit != 0);
            output[baseOutput + bitIndex] = positive ? scale : -scale;
        }
    }
}

// ─── v1 GEMV (legacy, kept for EDGERUNNER_Q1_USE_V2_KERNEL=0 fallback) ──

kernel void dequant_q1_0_g128_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;
    const uint nb = params.blocksPerRow;
    float sumf[LOCAL_NR] = { 0.f };
    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q1_0_g128BlockBytes;
    }
    for (uint ib = tiisg; ib < nb; ib += 32) {
        uint xBaseIdx = ib * q1_0_g128WeightsPerBlock;
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block + 0)));
            device const uchar* qs = block + 2;
            float sumq = 0.f;
            for (uint byteIdx = 0; byteIdx < 16; byteIdx++) {
                uchar bits = qs[byteIdx];
                uint elemBase = xBaseIdx + byteIdx * 8;
                float4 xv0 = float4(x[elemBase], x[elemBase+1], x[elemBase+2], x[elemBase+3]);
                float4 xv1 = float4(x[elemBase+4], x[elemBase+5], x[elemBase+6], x[elemBase+7]);
                float4 w0 = float4(
                    ((bits >> 0) & 1) ? scale : -scale,
                    ((bits >> 1) & 1) ? scale : -scale,
                    ((bits >> 2) & 1) ? scale : -scale,
                    ((bits >> 3) & 1) ? scale : -scale
                );
                float4 w1 = float4(
                    ((bits >> 4) & 1) ? scale : -scale,
                    ((bits >> 5) & 1) ? scale : -scale,
                    ((bits >> 6) & 1) ? scale : -scale,
                    ((bits >> 7) & 1) ? scale : -scale
                );
                sumq += dot(w0, xv0) + dot(w1, xv1);
            }
            sumf[row] += sumq;
        }
    }
    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }
    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// ─── v2 standalone GEMV (sign-flip + sub-block granularity) ──────────

kernel void dequant_q1_0_g128_gemv_v2(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q1_0_g128BlockBytes;
    }

    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        device const float* xp = x + xBase;
        for (short i = 0; i < 32; i++) { xl[i] = xp[i]; xSum += xl[i]; }

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[row] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);
    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// ─── v2 fused QKV: RMSNorm + Q + K + V in one dispatch ──────────────

kernel void dequant_q1_0_g128_fused_qkv_v2(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(7)]],
    device const float* normWeight [[buffer(8)]],
    constant float& rmsEps [[buffer(9)]],
    constant uint& qRows [[buffer(10)]],
    constant uint& kvRows [[buffer(11)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint totalRows = qRows + kvRows + kvRows;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= totalRows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;
    const uint cols = params.cols;
    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < qRows) { localRow = globalRow; weights = wq; }
        else if (globalRow < qRows + kvRows) { localRow = globalRow - qRows; weights = wk; }
        else { localRow = globalRow - qRows - kvRows; weights = wv; }
        ax[r] = weights + localRow * nb * q1_0_g128BlockBytes;
    }

    // Cooperative RMSNorm
    float sumSq = 0.0f;
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint xBase = (isb / 4) * q1_0_g128WeightsPerBlock + (isb % 4) * 32;
        for (short i = 0; i < 32; i++) { float v = x[xBase + i]; sumSq += v * v; }
    }
    sumSq = simd_sum(sumSq);
    const float rmsScale = rsqrt(sumSq / float(cols) + rmsEps);

    // GEMV with inline RMSNorm
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        for (short i = 0; i < 32; i++) {
            xl[i] = x[xBase + i] * rmsScale * normWeight[xBase + i];
            xSum += xl[i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[r] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);
    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;
            float total = sumf[r];
            if (globalRow < qRows) { outQ[globalRow] = total; }
            else if (globalRow < qRows + kvRows) { outK[globalRow - qRows] = total; }
            else { outV[globalRow - qRows - kvRows] = half(total); }
        }
    }
}

// ─── v2 fused GEMV + residual add ────────────────────────────────────

kernel void dequant_q1_0_g128_gemv_add_v2(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;
    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q1_0_g128BlockBytes;
    }

    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        device const float* xp = x + xBase;
        for (short i = 0; i < 32; i++) { xl[i] = xp[i]; xSum += xl[i]; }

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[row] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);
    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row] + residual[row0 + row];
        }
    }
}

// ─── v2 fused RMSNorm + Gate + Up (SwiGLU applied by caller) ────────

kernel void dequant_q1_0_g128_fused_gate_up_v2(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    device const float* normWeight [[buffer(4)]],
    constant ERDequantQ1_0_g128GEMVParams& params [[buffer(5)]],
    constant float& rmsEps [[buffer(6)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    const uint rows = params.rows;
    const uint totalRows = rows * 2;
    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= totalRows) return;

    const uint nb = params.blocksPerRow;
    const uint nbSub = nb * 4;
    const uint cols = params.cols;
    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wGate; continue; }
        if (globalRow < rows) { ax[r] = wGate + globalRow * nb * q1_0_g128BlockBytes; }
        else { ax[r] = wUp + (globalRow - rows) * nb * q1_0_g128BlockBytes; }
    }

    // Cooperative RMSNorm
    float sumSq = 0.0f;
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint xBase = (isb / 4) * q1_0_g128WeightsPerBlock + (isb % 4) * 32;
        for (short i = 0; i < 32; i++) { float v = x[xBase + i]; sumSq += v * v; }
    }
    sumSq = simd_sum(sumSq);
    const float rmsScale = rsqrt(sumSq / float(cols) + rmsEps);

    // GEMV with inline RMSNorm
    for (uint isb = tiisg; isb < nbSub; isb += 32) {
        const uint parentBlock = isb / 4;
        const uint subIdx = isb % 4;
        const uint xBase = parentBlock * q1_0_g128WeightsPerBlock + subIdx * 32;

        float xl[32];
        float xSum = 0.f;
        for (short i = 0; i < 32; i++) {
            xl[i] = x[xBase + i] * rmsScale * normWeight[xBase + i];
            xSum += xl[i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + parentBlock * q1_0_g128BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)(block)));
            sumf[r] += q1_subblock_dot(xl, xSum, block + 2 + subIdx * 4, scale);
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);
    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;
            activated[globalRow] = sumf[r];
        }
    }
}

// ─── Fused Q1 final norm + LM head (v1 pattern, 256 threads/TG) ─────

struct Q1FusedLMHeadParams {
    uint2 dims;  // x=vocabSize, y=dim
    float rmsEps;
};

kernel void dequant_q1_0_g128_fused_final_norm_gemv(
    device const uchar* lmHeadW [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    device const float* rmsNormW [[buffer(3)]],
    constant Q1FusedLMHeadParams& params [[buffer(4)]],
    uint ti [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]
) {
    constexpr uint THREADS = 256;
    const uint vocabSize = params.dims.x;
    const uint dim = params.dims.y;
    const float rmsEps = params.rmsEps;
    const uint nb = dim / q1_0_g128WeightsPerBlock;
    const uint row = tg * THREADS + ti;
    if (row >= vocabSize) return;

    threadgroup float xNorm[2048];
    const uint floatsPerThread = (dim + THREADS - 1) / THREADS;
    const uint loadStart = ti * floatsPerThread;
    const uint loadEnd = min(loadStart + floatsPerThread, dim);

    float localSumSq = 0.f;
    for (uint i = loadStart; i < loadEnd; i++) {
        localSumSq += hidden[i] * hidden[i];
    }

    threadgroup float sharedSum[256];
    sharedSum[ti] = localSumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (ti < s) { sharedSum[ti] += sharedSum[ti + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float invRms = rsqrt(sharedSum[0] * (1.0f / float(dim)) + rmsEps);
    for (uint i = loadStart; i < loadEnd; i++) {
        xNorm[i] = hidden[i] * invRms * rmsNormW[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.f;
    device const uchar* wRow = lmHeadW + row * nb * q1_0_g128BlockBytes;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar* block = wRow + ib * q1_0_g128BlockBytes;
        float scale = float(as_type<half>(*(device const ushort*)(block)));
        device const uchar* qs = block + 2;
        const uint xBase = ib * q1_0_g128WeightsPerBlock;
        for (uint bi = 0; bi < 16; bi++) {
            uchar bits = qs[bi];
            const uint eb = xBase + bi * 8;
            float4 xv0 = float4(xNorm[eb],   xNorm[eb+1], xNorm[eb+2], xNorm[eb+3]);
            float4 xv1 = float4(xNorm[eb+4], xNorm[eb+5], xNorm[eb+6], xNorm[eb+7]);
            float4 s0 = float4(
                (((bits >> 0) & 1) * 2 - 1) * scale,
                (((bits >> 1) & 1) * 2 - 1) * scale,
                (((bits >> 2) & 1) * 2 - 1) * scale,
                (((bits >> 3) & 1) * 2 - 1) * scale
            );
            float4 s1 = float4(
                (((bits >> 4) & 1) * 2 - 1) * scale,
                (((bits >> 5) & 1) * 2 - 1) * scale,
                (((bits >> 6) & 1) * 2 - 1) * scale,
                (((bits >> 7) & 1) * 2 - 1) * scale
            );
            sum += dot(s0, xv0) + dot(s1, xv1);
        }
    }
    logits[row] = sum;
}


// --- Dequant_Q2_K.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ2KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q2_K_BLOCK_BYTES = 84;
constant uint Q2_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q2_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ2KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q2_K_BLOCK_BYTES;

    // d at offset 80, dmin at offset 82 (float16)
    device const half* dPtr = reinterpret_cast<device const half*>(block + 80);
    float d = float(dPtr[0]);
    float dmin = float(dPtr[1]);

    uint outBase = params.outputOffset + tid * Q2_K_WEIGHTS_PER_BLOCK;
    for (uint idx = 0; idx < Q2_K_WEIGHTS_PER_BLOCK; ++idx) {
        uint sub = idx / 16;

        // scales packed at offset 0..15 (sc | m<<4)
        uchar scaleByte = block[sub];
        float sc = float(scaleByte & 0x0F);
        float m = float(scaleByte >> 4);

        // 2-bit quant from qs at offset 16..79
        uchar qsByte = block[16 + idx / 4];
        float q2 = float((qsByte >> ((idx % 4) * 2)) & 0x03);

        output[outBase + idx] = d * sc * q2 - dmin * m;
    }
}


// --- Dequant_Q3_K.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ3KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q3_K_BLOCK_BYTES = 110;
constant uint Q3_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q3_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ3KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q3_K_BLOCK_BYTES;

    // Master scale d (float16) at offset 108
    device const half* dPtr = reinterpret_cast<device const half*>(block + 108);
    float d = float(dPtr[0]);

    // Unpack 16 sub-block scales from 12 bytes starting at offset 96
    float scales[16];
    for (uint i = 0; i < 16; ++i) {
        uint byteIdx = i / 2;
        uchar lower4 = (block[96 + byteIdx] >> ((i % 2) * 4)) & 0x0F;

        uint upperByteIdx = 104 + i / 4;
        uchar upper2 = (block[upperByteIdx] >> ((i % 4) * 2)) & 0x03;

        int raw6 = int(lower4) | (int(upper2) << 4); // 0..63
        int signedScale = raw6 - 32;                 // -32..31
        scales[i] = d * float(signedScale);
    }

    uint outBase = params.outputOffset + tid * Q3_K_WEIGHTS_PER_BLOCK;
    for (uint idx = 0; idx < Q3_K_WEIGHTS_PER_BLOCK; ++idx) {
        uint sub = idx / 16;

        // Lower 2 bits from qs at offset 32
        uchar qsByte = block[32 + idx / 4];
        uchar lower2 = (qsByte >> ((idx % 4) * 2)) & 0x03;

        // High bit from hmask at offset 0
        uchar hmaskByte = block[idx / 8];
        uchar highBit = (hmaskByte >> (idx % 8)) & 0x01;

        uint q3 = uint(lower2) | (uint(highBit) << 2); // 0..7
        output[outBase + idx] = scales[sub] * float(int(q3) - 4);
    }
}


// --- Dequant_Q4_0.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ4_0Params {
    uint blockCount;
    uint outputOffset;
};

struct ERDequantQ4_0GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint Q4_0_BLOCK_BYTES = 18;
constant uint Q4_0_BLOCK_WEIGHTS = 32;

kernel void dequant_q4_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ4_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + tid * Q4_0_BLOCK_BYTES;
    device const half* scalePtr = reinterpret_cast<device const half*>(block);
    float scale = float(scalePtr[0]);
    uint outBase = params.outputOffset + tid * Q4_0_BLOCK_WEIGHTS;

    for (uint i = 0; i < 16; ++i) {
        uchar packed = block[2 + i];
        int low = int(packed & 0x0F) - 8;
        int high = int(packed >> 4) - 8;
        output[outBase + i] = scale * float(low);
        output[outBase + i + 16] = scale * float(high);
    }
}

kernel void dequant_q4_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ4_0GEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint localID [[thread_position_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    uint rowOffset = row * params.blocksPerRow;

    for (uint blockIndex = localID; blockIndex < params.blocksPerRow; blockIndex += 32) {
        device const uchar* block = quantisedW + (rowOffset + blockIndex) * Q4_0_BLOCK_BYTES;
        device const half* scalePtr = reinterpret_cast<device const half*>(block);
        float scale = float(scalePtr[0]);
        uint colBase = blockIndex * Q4_0_BLOCK_WEIGHTS;

        for (uint i = 0; i < 16; ++i) {
            uchar packed = block[2 + i];
            float low = scale * float(int(packed & 0x0F) - 8);
            float high = scale * float(int(packed >> 4) - 8);
            partial += low * x[colBase + i];
            partial += high * x[colBase + i + 16];
        }
    }

    partial = simd_sum(partial);
    if (simdLane == 0) {
        y[row] = partial;
    }
}


// --- Dequant_Q4_K_M.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ4KMParams {
    uint superBlockCount;
    uint outputOffset;
};

struct ERQ4KGEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

struct ERQ4KGEMV3Params {
    uint rowsA;
    uint rowsB;
    uint rowsC;
    uint cols;
    uint blocksPerRow;
};

constant uint Q4_K_M_BLOCK_BYTES = 144;
constant uint Q4_K_M_WEIGHTS_PER_BLOCK = 256;
constant uint Q4_K_M_GEMV_THREADS_PER_ROW = 256;
constant uint Q4_K_M_PACKED_GEMV_THREADS_PER_ROW = 128;

kernel void dequant_q4_k_m(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ4KMParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q4_K_M_BLOCK_BYTES;
    device const half* masterScales = reinterpret_cast<device const half*>(block);
    float d = float(masterScales[0]);
    float dmin = float(masterScales[1]);

    float scales[8];
    float mins[8];
    for (uint subBlock = 0; subBlock < 4; ++subBlock) {
        uchar scaleByte = block[4 + subBlock];
        uchar minByte = block[8 + subBlock];
        uchar highBits = block[12 + subBlock];

        scales[subBlock] = d * float(scaleByte & 0x3F);
        scales[subBlock + 4] = d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
        mins[subBlock] = dmin * float(minByte & 0x3F);
        mins[subBlock + 4] = dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
    }

    uint outBase = params.outputOffset + tid * Q4_K_M_WEIGHTS_PER_BLOCK;
    for (uint subBlock = 0; subBlock < 8; ++subBlock) {
        float scale = scales[subBlock];
        float minValue = mins[subBlock];

        for (uint index = 0; index < 32; ++index) {
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            output[outBase + (subBlock * 32) + index] = (scale * float(nibble)) - minValue;
        }
    }
}

kernel void q4_k_gemv_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float weight = sharedScales[subBlock] * float(nibble) - sharedMins[subBlock];
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * x[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q4_k_gemv_f32_batched(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.x;
    uint batch = group_id.y;
    uint local_id = local_pos.x;
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const float* xBase = x + batch * params.cols;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float weight = sharedScales[subBlock] * float(nibble) - sharedMins[subBlock];
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * xBase[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[batch * params.rows + row] = value;
        }
    }
}

kernel void q4_k_gemv_packed_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint packedIndex = local_id;
        if (packedIndex < 128) {
            uint pair = packedIndex / 32;
            uint index = packedIndex % 32;
            uint lowSubBlock = pair * 2;
            uint highSubBlock = lowSubBlock + 1;
            uint byteIndex = 16 + pair * 32 + index;
            uchar packed = block[byteIndex];
            uint lowCol = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + lowSubBlock * 32 + index;
            uint highCol = lowCol + 32;
            partial += (sharedScales[lowSubBlock] * float(packed & 0x0F) - sharedMins[lowSubBlock]) * x[lowCol];
            partial += (sharedScales[highSubBlock] * float((packed >> 4) & 0x0F) - sharedMins[highSubBlock]) * x[highCol];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_PACKED_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q4_k_gemv_packed_4row_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint tile [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rowsPerTile = 4;
    uint rowBaseIndex = tile * rowsPerTile;
    if (rowBaseIndex >= params.rows) {
        return;
    }

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    threadgroup float sharedScales[32];
    threadgroup float sharedMins[32];

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        if (local_id < 16) {
            uint rowInTile = local_id / 4;
            uint scaleIndex = local_id % 4;
            uint row = rowBaseIndex + rowInTile;
            if (row < params.rows) {
                device const uchar* block =
                    weights + (row * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
                device const half* masterScales = reinterpret_cast<device const half*>(block);
                float d = float(masterScales[0]);
                float dmin = float(masterScales[1]);
                uchar scaleByte = block[4 + scaleIndex];
                uchar minByte = block[8 + scaleIndex];
                uchar highBits = block[12 + scaleIndex];
                uint scaleBase = rowInTile * 8;

                sharedScales[scaleBase + scaleIndex] = d * float(scaleByte & 0x3F);
                sharedScales[scaleBase + scaleIndex + 4] =
                    d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
                sharedMins[scaleBase + scaleIndex] = dmin * float(minByte & 0x3F);
                sharedMins[scaleBase + scaleIndex + 4] =
                    dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint packedIndex = local_id;
        if (packedIndex < 128) {
            uint pair = packedIndex / 32;
            uint index = packedIndex % 32;
            uint lowSubBlock = pair * 2;
            uint highSubBlock = lowSubBlock + 1;
            uint byteIndex = 16 + pair * 32 + index;
            uint lowCol = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + lowSubBlock * 32 + index;
            uint highCol = lowCol + 32;
            float xLow = x[lowCol];
            float xHigh = x[highCol];

            if (rowBaseIndex < params.rows) {
                device const uchar* block =
                    weights + (rowBaseIndex * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
                uchar packed = block[byteIndex];
                partial0 += (sharedScales[lowSubBlock] * float(packed & 0x0F) - sharedMins[lowSubBlock]) * xLow;
                partial0 += (sharedScales[highSubBlock] * float((packed >> 4) & 0x0F) - sharedMins[highSubBlock]) * xHigh;
            }
            if (rowBaseIndex + 1 < params.rows) {
                device const uchar* block =
                    weights + ((rowBaseIndex + 1) * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
                uchar packed = block[byteIndex];
                uint scaleBase = 8;
                partial1 += (sharedScales[scaleBase + lowSubBlock] * float(packed & 0x0F) - sharedMins[scaleBase + lowSubBlock]) * xLow;
                partial1 += (sharedScales[scaleBase + highSubBlock] * float((packed >> 4) & 0x0F) - sharedMins[scaleBase + highSubBlock]) * xHigh;
            }
            if (rowBaseIndex + 2 < params.rows) {
                device const uchar* block =
                    weights + ((rowBaseIndex + 2) * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
                uchar packed = block[byteIndex];
                uint scaleBase = 16;
                partial2 += (sharedScales[scaleBase + lowSubBlock] * float(packed & 0x0F) - sharedMins[scaleBase + lowSubBlock]) * xLow;
                partial2 += (sharedScales[scaleBase + highSubBlock] * float((packed >> 4) & 0x0F) - sharedMins[scaleBase + highSubBlock]) * xHigh;
            }
            if (rowBaseIndex + 3 < params.rows) {
                device const uchar* block =
                    weights + ((rowBaseIndex + 3) * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
                uchar packed = block[byteIndex];
                uint scaleBase = 24;
                partial3 += (sharedScales[scaleBase + lowSubBlock] * float(packed & 0x0F) - sharedMins[scaleBase + lowSubBlock]) * xLow;
                partial3 += (sharedScales[scaleBase + highSubBlock] * float((packed >> 4) & 0x0F) - sharedMins[scaleBase + highSubBlock]) * xHigh;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial0 = simd_sum(partial0);
    partial1 = simd_sum(partial1);
    partial2 = simd_sum(partial2);
    partial3 = simd_sum(partial3);

    threadgroup float sharedSums[128];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial0;
        sharedSums[32 + simd_group] = partial1;
        sharedSums[64 + simd_group] = partial2;
        sharedSums[96 + simd_group] = partial3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_PACKED_GEMV_THREADS_PER_ROW + 31) / 32;
        float value0 = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        float value1 = simd_lane < numSimdgroups ? sharedSums[32 + simd_lane] : 0.0f;
        float value2 = simd_lane < numSimdgroups ? sharedSums[64 + simd_lane] : 0.0f;
        float value3 = simd_lane < numSimdgroups ? sharedSums[96 + simd_lane] : 0.0f;
        value0 = simd_sum(value0);
        value1 = simd_sum(value1);
        value2 = simd_sum(value2);
        value3 = simd_sum(value3);
        if (simd_lane == 0) {
            y[rowBaseIndex] = value0;
            if (rowBaseIndex + 1 < params.rows) {
                y[rowBaseIndex + 1] = value1;
            }
            if (rowBaseIndex + 2 < params.rows) {
                y[rowBaseIndex + 2] = value2;
            }
            if (rowBaseIndex + 3 < params.rows) {
                y[rowBaseIndex + 3] = value3;
            }
        }
    }
}

static inline ushort er_q4k_read_u16(device const uchar* ptr) {
    return ushort(ptr[0]) | (ushort(ptr[1]) << 8);
}

static inline float er_q4k_llama_style_row_sum(
    device const uchar* block,
    thread const float* yl,
    thread const float* yh,
    float4 sumy,
    short iq,
    short ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* scales = block + 4;
    ushort sc0 = er_q4k_read_u16(scales + 2 * uint(iq));
    ushort sc2 = er_q4k_read_u16(scales + 2 * (uint(iq) + 2));
    ushort sc4 = er_q4k_read_u16(scales + 2 * (uint(iq) + 4));

    ushort sc16_0 = sc0 & kmask1;
    ushort sc16_1 = sc2 & kmask1;
    ushort sc16_2 = ((sc4 >> 0) & kmask2) | ((sc0 & kmask3) >> 2);
    ushort sc16_3 = ((sc4 >> 4) & kmask2) | ((sc2 & kmask3) >> 2);

    float sc8_0 = float(sc16_0 & 0x00ff);
    float sc8_1 = float((sc16_0 >> 8) & 0x00ff);
    float sc8_2 = float(sc16_1 & 0x00ff);
    float sc8_3 = float((sc16_1 >> 8) & 0x00ff);
    float sc8_4 = float(sc16_2 & 0x00ff);
    float sc8_5 = float((sc16_2 >> 8) & 0x00ff);
    float sc8_6 = float(sc16_3 & 0x00ff);
    float sc8_7 = float((sc16_3 >> 8) & 0x00ff);

    device const uchar* q1 = block + 16 + 2 * uint(16 * iq + 4 * ir);
    device const uchar* q2 = q1 + 64;
    device const half* dh = reinterpret_cast<device const half*>(block);

    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    for (short i = 0; i < 4; ++i) {
        ushort q1i = er_q4k_read_u16(q1 + 2 * uint(i));
        ushort q2i = er_q4k_read_u16(q2 + 2 * uint(i));

        acc1[0] += yl[2 * i] * float(q1i & 0x000f);
        acc1[1] += yl[2 * i + 1] * float(q1i & 0x0f00);
        acc1[2] += yl[2 * i + 8] * float(q1i & 0x00f0);
        acc1[3] += yl[2 * i + 9] * float(q1i & 0xf000);
        acc2[0] += yh[2 * i] * float(q2i & 0x000f);
        acc2[1] += yh[2 * i + 1] * float(q2i & 0x0f00);
        acc2[2] += yh[2 * i + 8] * float(q2i & 0x00f0);
        acc2[3] += yh[2 * i + 9] * float(q2i & 0xf000);
    }

    return
        float(dh[0]) * (
            (acc1[0] + (1.0f / 256.0f) * acc1[1]) * sc8_0 +
            (acc1[2] + (1.0f / 256.0f) * acc1[3]) * sc8_1 * (1.0f / 16.0f) +
            (acc2[0] + (1.0f / 256.0f) * acc2[1]) * sc8_4 +
            (acc2[2] + (1.0f / 256.0f) * acc2[3]) * sc8_5 * (1.0f / 16.0f)
        ) -
        float(dh[1]) * (
            sumy[0] * sc8_2 +
            sumy[1] * sc8_3 +
            sumy[2] * sc8_6 +
            sumy[3] * sc8_7
        );
}

kernel void q4_k_gemv_llama_style_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    constexpr uint rowsPerSimdgroup = 2;
    constexpr uint simdgroupsPerThreadgroup = 2;
    constexpr uint rowsPerTile = rowsPerSimdgroup * simdgroupsPerThreadgroup;

    uint firstRow = (tile * simdgroupsPerThreadgroup + simd_group) * rowsPerSimdgroup;
    if (firstRow >= params.rows) {
        return;
    }

    short ix = short(simd_lane / 8);  // 0...3
    short it = short(simd_lane % 8);  // 0...7
    short iq = it / 4;                // 0 or 1
    short ir = it % 4;                // 0...3

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint blockIndex = uint(ix); blockIndex < params.blocksPerRow; blockIndex += 4) {
        uint xBase = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + uint(64 * iq + 8 * ir);

        float yl[16];
        float yh[16];
        float4 sumy = float4(0.0f);
        for (short i = 0; i < 8; ++i) {
            yl[i] = x[xBase + uint(i)];
            yl[i + 8] = x[xBase + 32 + uint(i)];
            yh[i] = x[xBase + 128 + uint(i)];
            yh[i + 8] = x[xBase + 160 + uint(i)];
            sumy[0] += yl[i];
            sumy[1] += yl[i + 8];
            sumy[2] += yh[i];
            sumy[3] += yh[i + 8];
        }

        for (uint rowOffset = 0; rowOffset < rowsPerSimdgroup; ++rowOffset) {
            uint row = firstRow + rowOffset;
            if (row >= params.rows) {
                continue;
            }

            device const uchar* block =
                weights + (row * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
            device const uchar* scales = block + 4;
            ushort sc0 = er_q4k_read_u16(scales + 2 * uint(iq));
            ushort sc2 = er_q4k_read_u16(scales + 2 * (uint(iq) + 2));
            ushort sc4 = er_q4k_read_u16(scales + 2 * (uint(iq) + 4));

            ushort sc16_0 = sc0 & kmask1;
            ushort sc16_1 = sc2 & kmask1;
            ushort sc16_2 = ((sc4 >> 0) & kmask2) | ((sc0 & kmask3) >> 2);
            ushort sc16_3 = ((sc4 >> 4) & kmask2) | ((sc2 & kmask3) >> 2);

            float sc8_0 = float(sc16_0 & 0x00ff);
            float sc8_1 = float((sc16_0 >> 8) & 0x00ff);
            float sc8_2 = float(sc16_1 & 0x00ff);
            float sc8_3 = float((sc16_1 >> 8) & 0x00ff);
            float sc8_4 = float(sc16_2 & 0x00ff);
            float sc8_5 = float((sc16_2 >> 8) & 0x00ff);
            float sc8_6 = float(sc16_3 & 0x00ff);
            float sc8_7 = float((sc16_3 >> 8) & 0x00ff);

            device const uchar* q1 = block + 16 + 2 * uint(16 * iq + 4 * ir);
            device const uchar* q2 = q1 + 64;
            device const half* dh = reinterpret_cast<device const half*>(block);

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            for (short i = 0; i < 4; ++i) {
                ushort q1i = er_q4k_read_u16(q1 + 2 * uint(i));
                ushort q2i = er_q4k_read_u16(q2 + 2 * uint(i));

                acc1[0] += yl[2 * i] * float(q1i & 0x000f);
                acc1[1] += yl[2 * i + 1] * float(q1i & 0x0f00);
                acc1[2] += yl[2 * i + 8] * float(q1i & 0x00f0);
                acc1[3] += yl[2 * i + 9] * float(q1i & 0xf000);
                acc2[0] += yh[2 * i] * float(q2i & 0x000f);
                acc2[1] += yh[2 * i + 1] * float(q2i & 0x0f00);
                acc2[2] += yh[2 * i + 8] * float(q2i & 0x00f0);
                acc2[3] += yh[2 * i + 9] * float(q2i & 0xf000);
            }

            float rowSum =
                float(dh[0]) * (
                    (acc1[0] + (1.0f / 256.0f) * acc1[1]) * sc8_0 +
                    (acc1[2] + (1.0f / 256.0f) * acc1[3]) * sc8_1 * (1.0f / 16.0f) +
                    (acc2[0] + (1.0f / 256.0f) * acc2[1]) * sc8_4 +
                    (acc2[2] + (1.0f / 256.0f) * acc2[3]) * sc8_5 * (1.0f / 16.0f)
                ) -
                float(dh[1]) * (
                    sumy[0] * sc8_2 +
                    sumy[1] * sc8_3 +
                    sumy[2] * sc8_6 +
                    sumy[3] * sc8_7
                );

            if (rowOffset == 0) {
                sum0 += rowSum;
            } else {
                sum1 += rowSum;
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);

    if (simd_lane == 0) {
        y[firstRow] = sum0;
        if (firstRow + 1 < params.rows) {
            y[firstRow + 1] = sum1;
        }
    }
}

kernel void q4_k_gemv_llama_style_dual_f32(
    device const uchar* weightsA [[buffer(0)]],
    device const uchar* weightsB [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* yA [[buffer(3)]],
    device float* yB [[buffer(4)]],
    constant ERQ4KGEMVParams& params [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint rowsPerSimdgroup = 2;
    constexpr uint simdgroupsPerThreadgroup = 2;

    uint firstRow = (tile * simdgroupsPerThreadgroup + simd_group) * rowsPerSimdgroup;
    if (firstRow >= params.rows) {
        return;
    }

    short ix = short(simd_lane / 8);
    short it = short(simd_lane % 8);
    short iq = it / 4;
    short ir = it % 4;

    float sumA0 = 0.0f;
    float sumA1 = 0.0f;
    float sumB0 = 0.0f;
    float sumB1 = 0.0f;

    for (uint blockIndex = uint(ix); blockIndex < params.blocksPerRow; blockIndex += 4) {
        uint xBase = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + uint(64 * iq + 8 * ir);

        float yl[16];
        float yh[16];
        float4 sumy = float4(0.0f);
        for (short i = 0; i < 8; ++i) {
            yl[i] = x[xBase + uint(i)];
            yl[i + 8] = x[xBase + 32 + uint(i)];
            yh[i] = x[xBase + 128 + uint(i)];
            yh[i + 8] = x[xBase + 160 + uint(i)];
            sumy[0] += yl[i];
            sumy[1] += yl[i + 8];
            sumy[2] += yh[i];
            sumy[3] += yh[i + 8];
        }

        for (uint rowOffset = 0; rowOffset < rowsPerSimdgroup; ++rowOffset) {
            uint row = firstRow + rowOffset;
            if (row >= params.rows) {
                continue;
            }

            device const uchar* blockA =
                weightsA + (row * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
            device const uchar* blockB =
                weightsB + (row * params.blocksPerRow + blockIndex) * Q4_K_M_BLOCK_BYTES;
            float rowSumA = er_q4k_llama_style_row_sum(blockA, yl, yh, sumy, iq, ir);
            float rowSumB = er_q4k_llama_style_row_sum(blockB, yl, yh, sumy, iq, ir);

            if (rowOffset == 0) {
                sumA0 += rowSumA;
                sumB0 += rowSumB;
            } else {
                sumA1 += rowSumA;
                sumB1 += rowSumB;
            }
        }
    }

    sumA0 = simd_sum(sumA0);
    sumA1 = simd_sum(sumA1);
    sumB0 = simd_sum(sumB0);
    sumB1 = simd_sum(sumB1);

    if (simd_lane == 0) {
        yA[firstRow] = sumA0;
        yB[firstRow] = sumB0;
        if (firstRow + 1 < params.rows) {
            yA[firstRow + 1] = sumA1;
            yB[firstRow + 1] = sumB1;
        }
    }
}

kernel void q4_k_gemv_dual_f32(
    device const uchar* weightsA [[buffer(0)]],
    device const uchar* weightsB [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* yA [[buffer(3)]],
    device float* yB [[buffer(4)]],
    constant ERQ4KGEMVParams& params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partialA = 0.0f;
    float partialB = 0.0f;
    threadgroup float sharedScalesA[8];
    threadgroup float sharedMinsA[8];
    threadgroup float sharedScalesB[8];
    threadgroup float sharedMinsB[8];
    device const uchar* rowBaseA = weightsA + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const uchar* rowBaseB = weightsB + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* blockA = rowBaseA + blockIndex * Q4_K_M_BLOCK_BYTES;
        device const uchar* blockB = rowBaseB + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScalesA = reinterpret_cast<device const half*>(blockA);
            float dA = float(masterScalesA[0]);
            float dminA = float(masterScalesA[1]);
            uchar scaleByteA = blockA[4 + local_id];
            uchar minByteA = blockA[8 + local_id];
            uchar highBitsA = blockA[12 + local_id];

            sharedScalesA[local_id] = dA * float(scaleByteA & 0x3F);
            sharedScalesA[local_id + 4] =
                dA * float((highBitsA & 0x0F) | (((scaleByteA >> 6) & 0x03) << 4));
            sharedMinsA[local_id] = dminA * float(minByteA & 0x3F);
            sharedMinsA[local_id + 4] =
                dminA * float(((highBitsA >> 4) & 0x0F) | (((minByteA >> 6) & 0x03) << 4));

            device const half* masterScalesB = reinterpret_cast<device const half*>(blockB);
            float dB = float(masterScalesB[0]);
            float dminB = float(masterScalesB[1]);
            uchar scaleByteB = blockB[4 + local_id];
            uchar minByteB = blockB[8 + local_id];
            uchar highBitsB = blockB[12 + local_id];

            sharedScalesB[local_id] = dB * float(scaleByteB & 0x3F);
            sharedScalesB[local_id + 4] =
                dB * float((highBitsB & 0x0F) | (((scaleByteB >> 6) & 0x03) << 4));
            sharedMinsB[local_id] = dminB * float(minByteB & 0x3F);
            sharedMinsB[local_id + 4] =
                dminB * float(((highBitsB >> 4) & 0x0F) | (((minByteB >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packedA = blockA[byteIndex];
            uchar nibbleA = (subBlock & 1) == 0 ? (packedA & 0x0F) : ((packedA >> 4) & 0x0F);
            uchar packedB = blockB[byteIndex];
            uchar nibbleB = (subBlock & 1) == 0 ? (packedB & 0x0F) : ((packedB >> 4) & 0x0F);
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            float xValue = x[col];
            partialA += (sharedScalesA[subBlock] * float(nibbleA) - sharedMinsA[subBlock]) * xValue;
            partialB += (sharedScalesB[subBlock] * float(nibbleB) - sharedMinsB[subBlock]) * xValue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partialA = simd_sum(partialA);
    partialB = simd_sum(partialB);

    threadgroup float sharedSumsA[32];
    threadgroup float sharedSumsB[32];
    if (simd_lane == 0) {
        sharedSumsA[simd_group] = partialA;
        sharedSumsB[simd_group] = partialB;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float valueA = simd_lane < numSimdgroups ? sharedSumsA[simd_lane] : 0.0f;
        float valueB = simd_lane < numSimdgroups ? sharedSumsB[simd_lane] : 0.0f;
        valueA = simd_sum(valueA);
        valueB = simd_sum(valueB);
        if (simd_lane == 0) {
            yA[row] = valueA;
            yB[row] = valueB;
        }
    }
}

kernel void q4_k_gemv_2row_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ4KGEMVParams& params [[buffer(3)]],
    uint rowPair [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = rowPair * 2;
    uint row1 = row0 + 1;
    if (row0 >= params.rows) {
        return;
    }

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    threadgroup float sharedScales0[8];
    threadgroup float sharedMins0[8];
    threadgroup float sharedScales1[8];
    threadgroup float sharedMins1[8];
    device const uchar* rowBase0 = weights + row0 * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const uchar* rowBase1 = weights + row1 * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    bool hasRow1 = row1 < params.rows;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block0 = rowBase0 + blockIndex * Q4_K_M_BLOCK_BYTES;
        device const uchar* block1 = rowBase1 + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* master0 = reinterpret_cast<device const half*>(block0);
            float d0 = float(master0[0]);
            float dmin0 = float(master0[1]);
            uchar scaleByte0 = block0[4 + local_id];
            uchar minByte0 = block0[8 + local_id];
            uchar highBits0 = block0[12 + local_id];
            sharedScales0[local_id] = d0 * float(scaleByte0 & 0x3F);
            sharedScales0[local_id + 4] =
                d0 * float((highBits0 & 0x0F) | (((scaleByte0 >> 6) & 0x03) << 4));
            sharedMins0[local_id] = dmin0 * float(minByte0 & 0x3F);
            sharedMins0[local_id + 4] =
                dmin0 * float(((highBits0 >> 4) & 0x0F) | (((minByte0 >> 6) & 0x03) << 4));

            if (hasRow1) {
                device const half* master1 = reinterpret_cast<device const half*>(block1);
                float d1 = float(master1[0]);
                float dmin1 = float(master1[1]);
                uchar scaleByte1 = block1[4 + local_id];
                uchar minByte1 = block1[8 + local_id];
                uchar highBits1 = block1[12 + local_id];
                sharedScales1[local_id] = d1 * float(scaleByte1 & 0x3F);
                sharedScales1[local_id + 4] =
                    d1 * float((highBits1 & 0x0F) | (((scaleByte1 >> 6) & 0x03) << 4));
                sharedMins1[local_id] = dmin1 * float(minByte1 & 0x3F);
                sharedMins1[local_id + 4] =
                    dmin1 * float(((highBits1 >> 4) & 0x0F) | (((minByte1 >> 6) & 0x03) << 4));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            float xValue = x[col];

            uchar packed0 = block0[byteIndex];
            uchar nibble0 = (subBlock & 1) == 0 ? (packed0 & 0x0F) : ((packed0 >> 4) & 0x0F);
            partial0 += (sharedScales0[subBlock] * float(nibble0) - sharedMins0[subBlock]) * xValue;

            if (hasRow1) {
                uchar packed1 = block1[byteIndex];
                uchar nibble1 = (subBlock & 1) == 0 ? (packed1 & 0x0F) : ((packed1 >> 4) & 0x0F);
                partial1 += (sharedScales1[subBlock] * float(nibble1) - sharedMins1[subBlock]) * xValue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial0 = simd_sum(partial0);
    partial1 = simd_sum(partial1);

    threadgroup float sharedSums0[32];
    threadgroup float sharedSums1[32];
    if (simd_lane == 0) {
        sharedSums0[simd_group] = partial0;
        sharedSums1[simd_group] = partial1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value0 = simd_lane < numSimdgroups ? sharedSums0[simd_lane] : 0.0f;
        float value1 = simd_lane < numSimdgroups ? sharedSums1[simd_lane] : 0.0f;
        value0 = simd_sum(value0);
        value1 = simd_sum(value1);
        if (simd_lane == 0) {
            y[row0] = value0;
            if (hasRow1) {
                y[row1] = value1;
            }
        }
    }
}

static inline float q4_k_gelu_tanh(float g) {
    if (g > 10.0f) {
        return g;
    }
    if (g < -10.0f) {
        return 0.0f;
    }
    const float c = 0.7978845608028654f;
    float inner = c * (g + 0.044715f * g * g * g);
    return g * 0.5f * (1.0f + tanh(inner));
}

kernel void q4_k_gemv_dual_geglu_f32(
    device const uchar* gateWeights [[buffer(0)]],
    device const uchar* upWeights [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    constant ERQ4KGEMVParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float gatePartial = 0.0f;
    float upPartial = 0.0f;
    threadgroup float gateScales[8];
    threadgroup float gateMins[8];
    threadgroup float upScales[8];
    threadgroup float upMins[8];
    device const uchar* gateRowBase = gateWeights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;
    device const uchar* upRowBase = upWeights + row * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* gateBlock = gateRowBase + blockIndex * Q4_K_M_BLOCK_BYTES;
        device const uchar* upBlock = upRowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* gateMaster = reinterpret_cast<device const half*>(gateBlock);
            float gateD = float(gateMaster[0]);
            float gateDMin = float(gateMaster[1]);
            uchar gateScaleByte = gateBlock[4 + local_id];
            uchar gateMinByte = gateBlock[8 + local_id];
            uchar gateHighBits = gateBlock[12 + local_id];

            gateScales[local_id] = gateD * float(gateScaleByte & 0x3F);
            gateScales[local_id + 4] =
                gateD * float((gateHighBits & 0x0F) | (((gateScaleByte >> 6) & 0x03) << 4));
            gateMins[local_id] = gateDMin * float(gateMinByte & 0x3F);
            gateMins[local_id + 4] =
                gateDMin * float(((gateHighBits >> 4) & 0x0F) | (((gateMinByte >> 6) & 0x03) << 4));

            device const half* upMaster = reinterpret_cast<device const half*>(upBlock);
            float upD = float(upMaster[0]);
            float upDMin = float(upMaster[1]);
            uchar upScaleByte = upBlock[4 + local_id];
            uchar upMinByte = upBlock[8 + local_id];
            uchar upHighBits = upBlock[12 + local_id];

            upScales[local_id] = upD * float(upScaleByte & 0x3F);
            upScales[local_id + 4] =
                upD * float((upHighBits & 0x0F) | (((upScaleByte >> 6) & 0x03) << 4));
            upMins[local_id] = upDMin * float(upMinByte & 0x3F);
            upMins[local_id + 4] =
                upDMin * float(((upHighBits >> 4) & 0x0F) | (((upMinByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar gatePacked = gateBlock[byteIndex];
            uchar gateNibble = (subBlock & 1) == 0 ? (gatePacked & 0x0F) : ((gatePacked >> 4) & 0x0F);
            uchar upPacked = upBlock[byteIndex];
            uchar upNibble = (subBlock & 1) == 0 ? (upPacked & 0x0F) : ((upPacked >> 4) & 0x0F);
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            float xValue = x[col];
            gatePartial += (gateScales[subBlock] * float(gateNibble) - gateMins[subBlock]) * xValue;
            upPartial += (upScales[subBlock] * float(upNibble) - upMins[subBlock]) * xValue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gatePartial = simd_sum(gatePartial);
    upPartial = simd_sum(upPartial);

    threadgroup float gateSums[32];
    threadgroup float upSums[32];
    if (simd_lane == 0) {
        gateSums[simd_group] = gatePartial;
        upSums[simd_group] = upPartial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float gateValue = simd_lane < numSimdgroups ? gateSums[simd_lane] : 0.0f;
        float upValue = simd_lane < numSimdgroups ? upSums[simd_lane] : 0.0f;
        gateValue = simd_sum(gateValue);
        upValue = simd_sum(upValue);
        if (simd_lane == 0) {
            activated[row] = q4_k_gelu_tanh(gateValue) * upValue;
        }
    }
}

kernel void q4_k_gemv_three_f32(
    device const uchar* weightsA [[buffer(0)]],
    device const uchar* weightsB [[buffer(1)]],
    device const uchar* weightsC [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* yA [[buffer(4)]],
    device float* yB [[buffer(5)]],
    device float* yC [[buffer(6)]],
    constant ERQ4KGEMV3Params& params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint totalRows = params.rowsA + params.rowsB + params.rowsC;
    if (row >= totalRows) {
        return;
    }

    device const uchar* selectedWeights = weightsA;
    device float* selectedOutput = yA;
    uint selectedRow = row;
    if (row >= params.rowsA + params.rowsB) {
        selectedWeights = weightsC;
        selectedOutput = yC;
        selectedRow = row - params.rowsA - params.rowsB;
    } else if (row >= params.rowsA) {
        selectedWeights = weightsB;
        selectedOutput = yB;
        selectedRow = row - params.rowsA;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = selectedWeights + selectedRow * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q4_K_M_WEIGHTS_PER_BLOCK) {
            uint subBlock = inBlock / 32;
            uint index = inBlock % 32;
            uint byteIndex = 16 + (subBlock / 2) * 32 + index;
            uchar packed = block[byteIndex];
            uchar nibble = (subBlock & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float weight = sharedScales[subBlock] * float(nibble) - sharedMins[subBlock];
            uint col = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * x[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            selectedOutput[selectedRow] = value;
        }
    }
}

kernel void q4_k_gemv_three_packed_f32(
    device const uchar* weightsA [[buffer(0)]],
    device const uchar* weightsB [[buffer(1)]],
    device const uchar* weightsC [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* yA [[buffer(4)]],
    device float* yB [[buffer(5)]],
    device float* yC [[buffer(6)]],
    constant ERQ4KGEMV3Params& params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint totalRows = params.rowsA + params.rowsB + params.rowsC;
    if (row >= totalRows) {
        return;
    }

    device const uchar* selectedWeights = weightsA;
    device float* selectedOutput = yA;
    uint selectedRow = row;
    if (row >= params.rowsA + params.rowsB) {
        selectedWeights = weightsC;
        selectedOutput = yC;
        selectedRow = row - params.rowsA - params.rowsB;
    } else if (row >= params.rowsA) {
        selectedWeights = weightsB;
        selectedOutput = yB;
        selectedRow = row - params.rowsA;
    }

    float partial = 0.0f;
    threadgroup float sharedScales[8];
    threadgroup float sharedMins[8];
    device const uchar* rowBase = selectedWeights + selectedRow * params.blocksPerRow * Q4_K_M_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q4_K_M_BLOCK_BYTES;

        if (local_id < 4) {
            device const half* masterScales = reinterpret_cast<device const half*>(block);
            float d = float(masterScales[0]);
            float dmin = float(masterScales[1]);
            uchar scaleByte = block[4 + local_id];
            uchar minByte = block[8 + local_id];
            uchar highBits = block[12 + local_id];

            sharedScales[local_id] = d * float(scaleByte & 0x3F);
            sharedScales[local_id + 4] =
                d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
            sharedMins[local_id] = dmin * float(minByte & 0x3F);
            sharedMins[local_id + 4] =
                dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint packedIndex = local_id;
        if (packedIndex < 128) {
            uint pair = packedIndex / 32;
            uint index = packedIndex % 32;
            uint lowSubBlock = pair * 2;
            uint highSubBlock = lowSubBlock + 1;
            uint byteIndex = 16 + pair * 32 + index;
            uchar packed = block[byteIndex];
            uint lowCol = blockIndex * Q4_K_M_WEIGHTS_PER_BLOCK + lowSubBlock * 32 + index;
            uint highCol = lowCol + 32;
            partial += (sharedScales[lowSubBlock] * float(packed & 0x0F) - sharedMins[lowSubBlock]) * x[lowCol];
            partial += (sharedScales[highSubBlock] * float((packed >> 4) & 0x0F) - sharedMins[highSubBlock]) * x[highCol];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q4_K_M_PACKED_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            selectedOutput[selectedRow] = value;
        }
    }
}


// --- Dequant_Q5_0.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ5_0Params {
    uint blockCount;
    uint outputOffset;
};

constant uint Q5_0_BLOCK_BYTES = 22;
constant uint Q5_0_WEIGHTS_PER_BLOCK = 32;

kernel void dequant_q5_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ5_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + tid * Q5_0_BLOCK_BYTES;
    float d = float(as_type<half>(*reinterpret_cast<device const ushort*>(block)));

    uint outBase = params.outputOffset + tid * Q5_0_WEIGHTS_PER_BLOCK;
    for (uint i = 0; i < Q5_0_WEIGHTS_PER_BLOCK; ++i) {
        // Lower 4 bits from qs at offset 6
        uchar qsByte = block[6 + i / 2];
        uchar lower4 = (i % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

        // 5th bit from qh at offset 2
        uchar qhByte = block[2 + i / 8];
        uchar bit5 = (qhByte >> (i % 8)) & 0x01;

        uint q5 = uint(lower4) | (uint(bit5) << 4);
        output[outBase + i] = d * float(int(q5) - 16);
    }
}


// --- Dequant_Q5_1.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ5_1Params {
    uint blockCount;
    uint outputOffset;
};

constant uint Q5_1_BLOCK_BYTES = 24;
constant uint Q5_1_WEIGHTS_PER_BLOCK = 32;

kernel void dequant_q5_1(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ5_1Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) {
        return;
    }

    device const uchar* block = input + tid * Q5_1_BLOCK_BYTES;
    float d = float(as_type<half>(*reinterpret_cast<device const ushort*>(block)));
    float m = float(as_type<half>(*reinterpret_cast<device const ushort*>(block + 2)));

    uint outBase = params.outputOffset + tid * Q5_1_WEIGHTS_PER_BLOCK;
    for (uint i = 0; i < Q5_1_WEIGHTS_PER_BLOCK; ++i) {
        // Lower 4 bits from qs at offset 8
        uchar qsByte = block[8 + i / 2];
        uchar lower4 = (i % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

        // 5th bit from qh at offset 4
        uchar qhByte = block[4 + i / 8];
        uchar bit5 = (qhByte >> (i % 8)) & 0x01;

        uint q5 = uint(lower4) | (uint(bit5) << 4);
        output[outBase + i] = d * float(q5) + m;
    }
}


// --- Dequant_Q5_K.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ5KParams {
    uint superBlockCount;
    uint outputOffset;
};

constant uint Q5_K_BLOCK_BYTES = 176;
constant uint Q5_K_WEIGHTS_PER_BLOCK = 256;

kernel void dequant_q5_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ5KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q5_K_BLOCK_BYTES;
    device const half* masterScales = reinterpret_cast<device const half*>(block);
    float d = float(masterScales[0]);
    float dmin = float(masterScales[1]);

    // Unpack scales and mins from 12 bytes at offset 4 (identical to Q4_K_M)
    float scales[8];
    float mins[8];
    for (uint subBlock = 0; subBlock < 4; ++subBlock) {
        uchar scaleByte = block[4 + subBlock];
        uchar minByte = block[8 + subBlock];
        uchar highBits = block[12 + subBlock];

        scales[subBlock] = d * float(scaleByte & 0x3F);
        scales[subBlock + 4] = d * float((highBits & 0x0F) | (((scaleByte >> 6) & 0x03) << 4));
        mins[subBlock] = dmin * float(minByte & 0x3F);
        mins[subBlock + 4] = dmin * float(((highBits >> 4) & 0x0F) | (((minByte >> 6) & 0x03) << 4));
    }

    uint outBase = params.outputOffset + tid * Q5_K_WEIGHTS_PER_BLOCK;

    for (uint subBlock = 0; subBlock < 8; ++subBlock) {
        float scale = scales[subBlock];
        float minValue = mins[subBlock];

        for (uint index = 0; index < 32; ++index) {
            uint globalIdx = subBlock * 32 + index;

            // Lower 4 bits from qs (offset 48, 128 bytes nibble-packed)
            uint qsByteIndex = 48 + globalIdx / 2;
            uchar qsByte = block[qsByteIndex];
            uchar lower4 = (globalIdx % 2 == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

            // 5th bit from qh (offset 16, 32 bytes bit-packed)
            uchar qhByte = block[16 + globalIdx / 8];
            uchar bit5 = (qhByte >> (globalIdx % 8)) & 1;

            uint q5 = uint(lower4) | (uint(bit5) << 4);
            output[outBase + globalIdx] = scale * float(q5) - minValue;
        }
    }
}


// --- Dequant_Q6_K.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ6KParams {
    uint superBlockCount;
    uint outputOffset;
};

struct ERQ6KGEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

constant uint Q6_K_BLOCK_BYTES = 210;
constant uint Q6_K_WEIGHTS_PER_BLOCK = 256;
constant uint Q6_K_GEMV_THREADS_PER_ROW = 256;
constant uint Q6_K_GEMV_PACKED_THREADS_PER_ROW = 64;

static inline float dequant_q6_k_value(device const uchar* block, uint inBlock, float d) {
    const uint halfBlock = inBlock / 128;
    const uint within = inBlock - halfBlock * 128;
    const uint lane = within & 31;
    const uint quarter = within / 32;
    const uint qlBase = halfBlock * 64;
    const uint qhBase = 128 + halfBlock * 32;
    const uint scaleBase = 192 + halfBlock * 8;

    const uchar qlByte = block[qlBase + (quarter & 1) * 32 + lane];
    const uchar lower4 = quarter < 2 ? (qlByte & 0x0F) : (qlByte >> 4);
    const uchar upper2 = (block[qhBase + lane] >> (quarter * 2)) & 0x03;
    const int q6 = int(lower4 | (upper2 << 4)) - 32;
    const char scale = as_type<char>(block[scaleBase + (quarter * 2) + lane / 16]);
    return d * float(scale) * float(q6);
}

kernel void dequant_q6_k(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ6KParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.superBlockCount) {
        return;
    }

    device const uchar* block = input + tid * Q6_K_BLOCK_BYTES;

    // d at offset 208 (float16)
    device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
    float d = float(dPtr[0]);

    uint outBase = params.outputOffset + tid * Q6_K_WEIGHTS_PER_BLOCK;

    for (uint i = 0; i < 256; ++i) {
        output[outBase + i] = dequant_q6_k_value(block, i, d);
    }
}

kernel void q6_k_gemv_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ6KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    float partial = 0.0f;
    threadgroup float sharedScale[16];
    threadgroup float sharedD;
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q6_K_BLOCK_BYTES;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q6_K_BLOCK_BYTES;

        if (local_id == 0) {
            device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
            sharedD = float(dPtr[0]);
        }
        if (local_id < 16) {
            sharedScale[local_id] = float(as_type<char>(block[192 + local_id]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inBlock = local_id;
        if (inBlock < Q6_K_WEIGHTS_PER_BLOCK) {
            const uint halfBlock = inBlock / 128;
            const uint within = inBlock - halfBlock * 128;
            const uint lane = within & 31;
            const uint quarter = within / 32;
            const uint qlBase = halfBlock * 64;
            const uint qhBase = 128 + halfBlock * 32;
            const uint scaleIndex = halfBlock * 8 + quarter * 2 + lane / 16;
            const uchar qlByte = block[qlBase + (quarter & 1) * 32 + lane];
            const uchar lower4 = quarter < 2 ? (qlByte & 0x0F) : (qlByte >> 4);
            const uchar upper2 = (block[qhBase + lane] >> (quarter * 2)) & 0x03;
            const int q6 = int(lower4 | (upper2 << 4)) - 32;
            float weight = sharedD * sharedScale[scaleIndex] * float(q6);
            uint col = blockIndex * Q6_K_WEIGHTS_PER_BLOCK + inBlock;
            partial += weight * x[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint numSimdgroups = (Q6_K_GEMV_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q6_k_gemv_packed_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERQ6KGEMVParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.rows || local_id >= Q6_K_GEMV_PACKED_THREADS_PER_ROW) {
        return;
    }

    float partial = 0.0f;
    device const uchar* rowBase = weights + row * params.blocksPerRow * Q6_K_BLOCK_BYTES;
    const uint halfBlock = local_id >> 5;
    const uint lane = local_id & 31;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        device const uchar* block = rowBase + blockIndex * Q6_K_BLOCK_BYTES;
        device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
        const float d = float(dPtr[0]);

        const uint qlBase = halfBlock * 64;
        const uint qhBase = 128 + halfBlock * 32;
        const uint scaleBase = 192 + halfBlock * 8;
        const uint scaleOffset = lane >> 4;
        const uchar ql0 = block[qlBase + lane];
        const uchar ql1 = block[qlBase + 32 + lane];
        const uchar qh = block[qhBase + lane];

        const int q1 = int((ql0 & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
        const int q2 = int((ql1 & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
        const int q3 = int((ql0 >> 4) | (((qh >> 4) & 0x03) << 4)) - 32;
        const int q4 = int((ql1 >> 4) | (((qh >> 6) & 0x03) << 4)) - 32;

        const float s1 = float(as_type<char>(block[scaleBase + scaleOffset + 0]));
        const float s2 = float(as_type<char>(block[scaleBase + scaleOffset + 2]));
        const float s3 = float(as_type<char>(block[scaleBase + scaleOffset + 4]));
        const float s4 = float(as_type<char>(block[scaleBase + scaleOffset + 6]));
        const uint colBase = blockIndex * Q6_K_WEIGHTS_PER_BLOCK + halfBlock * 128;

        partial += d * s1 * float(q1) * x[colBase + lane];
        partial += d * s2 * float(q2) * x[colBase + 32 + lane];
        partial += d * s3 * float(q3) * x[colBase + 64 + lane];
        partial += d * s4 * float(q4) * x[colBase + 96 + lane];
    }

    partial = simd_sum(partial);

    threadgroup float sharedSums[32];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const uint numSimdgroups = (Q6_K_GEMV_PACKED_THREADS_PER_ROW + 31) / 32;
        float value = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            y[row] = value;
        }
    }
}

kernel void q6_k_gemv_packed_4row_top1_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* partialValues [[buffer(2)]],
    device uint* partialIndices [[buffer(3)]],
    constant ERQ6KGEMVParams& params [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rowsPerTile = 4;
    const uint rowBaseIndex = tile * rowsPerTile;
    if (rowBaseIndex + 3 >= params.rows || local_id >= Q6_K_GEMV_PACKED_THREADS_PER_ROW) {
        return;
    }

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    const uint halfBlock = local_id >> 5;
    const uint lane = local_id & 31;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        const uint colBase = blockIndex * Q6_K_WEIGHTS_PER_BLOCK + halfBlock * 128;
        const float x1 = x[colBase + lane];
        const float x2 = x[colBase + 32 + lane];
        const float x3 = x[colBase + 64 + lane];
        const float x4 = x[colBase + 96 + lane];

        for (uint rowInTile = 0; rowInTile < rowsPerTile; ++rowInTile) {
            const uint row = rowBaseIndex + rowInTile;
            device const uchar* block =
                weights + (row * params.blocksPerRow + blockIndex) * Q6_K_BLOCK_BYTES;
            device const half* dPtr = reinterpret_cast<device const half*>(block + 208);
            const float d = float(dPtr[0]);

            const uint qlBase = halfBlock * 64;
            const uint qhBase = 128 + halfBlock * 32;
            const uint scaleBase = 192 + halfBlock * 8;
            const uint scaleOffset = lane >> 4;
            const uchar ql0 = block[qlBase + lane];
            const uchar ql1 = block[qlBase + 32 + lane];
            const uchar qh = block[qhBase + lane];

            const int q1 = int((ql0 & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
            const int q2 = int((ql1 & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
            const int q3 = int((ql0 >> 4) | (((qh >> 4) & 0x03) << 4)) - 32;
            const int q4 = int((ql1 >> 4) | (((qh >> 6) & 0x03) << 4)) - 32;

            const float s1 = float(as_type<char>(block[scaleBase + scaleOffset + 0]));
            const float s2 = float(as_type<char>(block[scaleBase + scaleOffset + 2]));
            const float s3 = float(as_type<char>(block[scaleBase + scaleOffset + 4]));
            const float s4 = float(as_type<char>(block[scaleBase + scaleOffset + 6]));
            const float value =
                d * s1 * float(q1) * x1 +
                d * s2 * float(q2) * x2 +
                d * s3 * float(q3) * x3 +
                d * s4 * float(q4) * x4;
            if (rowInTile == 0) {
                partial0 += value;
            } else if (rowInTile == 1) {
                partial1 += value;
            } else if (rowInTile == 2) {
                partial2 += value;
            } else {
                partial3 += value;
            }
        }
    }

    partial0 = simd_sum(partial0);
    partial1 = simd_sum(partial1);
    partial2 = simd_sum(partial2);
    partial3 = simd_sum(partial3);

    threadgroup float sharedSums[128];
    if (simd_lane == 0) {
        sharedSums[simd_group] = partial0;
        sharedSums[32 + simd_group] = partial1;
        sharedSums[64 + simd_group] = partial2;
        sharedSums[96 + simd_group] = partial3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const uint numSimdgroups = (Q6_K_GEMV_PACKED_THREADS_PER_ROW + 31) / 32;
        float value0 = simd_lane < numSimdgroups ? sharedSums[simd_lane] : 0.0f;
        float value1 = simd_lane < numSimdgroups ? sharedSums[32 + simd_lane] : 0.0f;
        float value2 = simd_lane < numSimdgroups ? sharedSums[64 + simd_lane] : 0.0f;
        float value3 = simd_lane < numSimdgroups ? sharedSums[96 + simd_lane] : 0.0f;
        value0 = simd_sum(value0);
        value1 = simd_sum(value1);
        value2 = simd_sum(value2);
        value3 = simd_sum(value3);
        if (simd_lane == 0) {
            float bestValue = value0;
            uint bestIndex = rowBaseIndex;
            if (value1 > bestValue) {
                bestValue = value1;
                bestIndex = rowBaseIndex + 1;
            }
            if (value2 > bestValue) {
                bestValue = value2;
                bestIndex = rowBaseIndex + 2;
            }
            if (value3 > bestValue) {
                bestValue = value3;
                bestIndex = rowBaseIndex + 3;
            }
            partialValues[tile] = bestValue;
            partialIndices[tile] = bestIndex;
        }
    }
}

kernel void q6_k_top1_reduce(
    device const float* partialValues [[buffer(0)]],
    device const uint* partialIndices [[buffer(1)]],
    device uint* outputIndex [[buffer(2)]],
    constant uint& partialCount [[buffer(3)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float bestValue = -INFINITY;
    uint bestIndex = 0;
    for (uint index = local_id; index < partialCount; index += 256) {
        const float value = partialValues[index];
        const uint token = partialIndices[index];
        if (value > bestValue || (value == bestValue && token < bestIndex)) {
            bestValue = value;
            bestIndex = token;
        }
    }

    threadgroup float groupValues[32];
    threadgroup uint groupIndices[32];
    for (uint offset = 16; offset > 0; offset >>= 1) {
        const float otherValue = simd_shuffle_down(bestValue, offset);
        const uint otherIndex = simd_shuffle_down(bestIndex, offset);
        if (otherValue > bestValue || (otherValue == bestValue && otherIndex < bestIndex)) {
            bestValue = otherValue;
            bestIndex = otherIndex;
        }
    }
    if (simd_lane == 0) {
        groupValues[simd_group] = bestValue;
        groupIndices[simd_group] = bestIndex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const uint simdgroupCount = 8;
        bestValue = simd_lane < simdgroupCount ? groupValues[simd_lane] : -INFINITY;
        bestIndex = simd_lane < simdgroupCount ? groupIndices[simd_lane] : 0;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            const float otherValue = simd_shuffle_down(bestValue, offset);
            const uint otherIndex = simd_shuffle_down(bestIndex, offset);
            if (otherValue > bestValue || (otherValue == bestValue && otherIndex < bestIndex)) {
                bestValue = otherValue;
                bestIndex = otherIndex;
            }
        }
        if (simd_lane == 0) {
            outputIndex[0] = bestIndex;
        }
    }
}


// --- Dequant_Q8_0.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERDequantQ8_0Params {
    uint blockCount;
    uint outputOffset;
};

struct ERQuantizeQ8RowsParams {
    uint rowCount;
    uint sourceRowStride;
    uint destinationRowBase;
    uint blocksPerRow;
};

constant uint q8_0BlockBytes = 34;
constant uint q8_0WeightsPerBlock = 32;

kernel void quantize_q8_0_rows(
    device const float *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    constant ERQuantizeQ8RowsParams &params [[buffer(2)]],
    uint rowIndex [[thread_position_in_grid]]
) {
    if (rowIndex >= params.rowCount) return;

    const uint sourceBase = rowIndex * params.sourceRowStride;
    const uint destinationRow = params.destinationRowBase + rowIndex;
    device uchar *rowDst = destination + destinationRow * params.blocksPerRow * q8_0BlockBytes;

    for (uint blockIndex = 0; blockIndex < params.blocksPerRow; ++blockIndex) {
        const uint blockSourceBase = sourceBase + blockIndex * q8_0WeightsPerBlock;
        device uchar *blockDst = rowDst + blockIndex * q8_0BlockBytes;

        float maxAbs = 0.0f;
        for (uint lane = 0; lane < q8_0WeightsPerBlock; ++lane) {
            maxAbs = max(maxAbs, fabs(source[blockSourceBase + lane]));
        }

        const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 0.0f;
        *(device ushort *) blockDst = as_type<ushort>(half(scale));

        for (uint lane = 0; lane < q8_0WeightsPerBlock; ++lane) {
            const float value = source[blockSourceBase + lane];
            const int quantized = scale == 0.0f ? 0 : clamp((int) rint(value / scale), -127, 127);
            blockDst[2 + lane] = as_type<uchar>((char) quantized);
        }
    }
}

kernel void dequant_q8_0(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERDequantQ8_0Params& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.blockCount) return;
    device const uchar* block = input + (tid * q8_0BlockBytes);
    float scale = float(as_type<half>(*(device const ushort*)block));
    uint outputBase = params.outputOffset + (tid * q8_0WeightsPerBlock);
    for (uint index = 0; index < q8_0WeightsPerBlock; index++) {
        output[outputBase + index] = scale * float(as_type<char>(block[2 + index]));
    }
}

// === High-Performance Fused Q8_0 GEMV ===
// Architecture: 32 threads (1 simdgroup) per threadgroup, 2 rows per TG.
// Each thread processes 1 full Q8_0 block (32 elements) per iteration.
// Single simd_sum reduction — no cross-simdgroup overhead.
//
// y[row] = sum_k dequant(W_q8[row, k]) * x[k]

struct ERDequantQ8GEMVParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
};

struct ERDequantQ8GEMVBatchParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
    uint tokenCount;
};

kernel void dequant_q8_0_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    // Pointers to weight rows
    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    // Main loop: each thread handles 1 full block (32 elements) per iteration.
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        // Cache x in registers (reused across LOCAL_NR rows).
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;

            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);

            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    // Single simd_sum — no cross-SG reduction needed
    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

kernel void dequant_q8_0_gemv_batched(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVBatchParams& params [[buffer(3)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= params.rows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenY = y + tokenIndex * params.rows;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;

            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);

            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            tokenY[row0 + row] = sumf[row];
        }
    }
}

// === Fused RMSNorm + Q+K+V projection ===
// Single dispatch replaces: RMSNorm + QKV GEMV (saves 1 dispatch per layer).
// Computes: normed = RMSNorm(x, normWeight, eps), then Q/K/V = dequant(W) * normed
// RMSNorm is computed cooperatively: each threadgroup calculates the normalization
// factor from the input, then uses it while processing weight blocks.

struct ERFusedQKVParams {
    uint qRows;          // Q output rows (numHeads * headDim)
    uint kvRows;         // K/V output rows (numKVHeads * headDim)
    uint cols;           // input columns (dim)
    uint blocksPerRow;   // Q8_0 blocks per row
    uint tokenCount;     // number of input tokens in the batch
    float rmsEps;        // RMSNorm epsilon
};

kernel void dequant_q8_0_fused_qkv(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],     // V writes f16 directly to cache
    device const float* normWeight [[buffer(8)]],  // RMSNorm weight [cols]
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    // Total rows = qRows + kvRows + kvRows. Each threadgroup handles LOCAL_NR consecutive rows
    // across the concatenated Q/K/V output space.
    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device half* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    // Determine which weight matrix each row belongs to
    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    // === Cooperative RMSNorm: compute normalization factor ===
    // Each thread sums squares for its assigned blocks, then simd_sum reduces.
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    // === Main GEMV loop with inline RMSNorm ===
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        // Apply RMSNorm inline: normed_x = x * rmsScale * normWeight
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[r] += sumq * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = half(total);
            }
        }
    }
}

// === Fused RMSNorm + Gate+Up+SwiGLU ===
// Single dispatch replaces: FFN RMSNorm + gate GEMV + up GEMV + SwiGLU (saves 1 dispatch per layer).
// Computes: normed = RMSNorm(x, normWeight, eps)
//           activated[row] = silu(gate_proj[row]) * up_proj[row]
// where gate_proj = dequant(Wg) * normed, up_proj = dequant(Wu) * normed

inline float silu_fn(float x) { return x / (1.0f + exp(-x)); }

struct ERFusedGateUpSiluParams {
    uint rows;
    uint cols;
    uint blocksPerRow;
    uint tokenCount;
    float rmsEps;
};

kernel void dequant_q8_0_fused_final_norm_gemv(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const float* normWeight [[buffer(3)]],
    constant ERFusedGateUpSiluParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;

            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);

            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

kernel void dequant_q8_0_fused_gate_up_silu(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    device const float* normWeight [[buffer(5)]],  // RMSNorm weight [cols]
    constant ERFusedGateUpSiluParams& params [[buffer(4)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= params.rows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenActivated = activated + tokenIndex * params.rows;

    // === Cooperative RMSNorm ===
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    float sumGate[LOCAL_NR] = { 0.f };
    float sumUp[LOCAL_NR] = { 0.f };

    device const uchar* axGate[LOCAL_NR];
    device const uchar* axUp[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint row = row0 + r;
        uint safeRow = row < params.rows ? row : row0;
        axGate[r] = wGate + safeRow * nb * q8_0BlockBytes;
        axUp[r] = wUp + safeRow * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        // Apply RMSNorm inline
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;

            // Gate
            device const uchar* blockG = axGate[r] + ib * q8_0BlockBytes;
            float scaleG = float(as_type<half>(*(device const ushort*)blockG));
            device const char* qsG = (device const char*)(blockG + 2);
            float sumG = 0.f;
            for (short i = 0; i < 32; i++) sumG += float(qsG[i]) * xl[i];
            sumGate[r] += sumG * scaleG;

            // Up
            device const uchar* blockU = axUp[r] + ib * q8_0BlockBytes;
            float scaleU = float(as_type<half>(*(device const ushort*)blockU));
            device const char* qsU = (device const char*)(blockU + 2);
            float sumU = 0.f;
            for (short i = 0; i < 32; i++) sumU += float(qsU[i]) * xl[i];
            sumUp[r] += sumU * scaleU;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumGate[r] = simd_sum(sumGate[r]);
        sumUp[r] = simd_sum(sumUp[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;
            tokenActivated[row0 + r] = silu_fn(sumGate[r]) * sumUp[r];
        }
    }
}

// === Float16 output variant — writes half directly to KV cache ===
// Eliminates separate f32->f16 conversion dispatch.
kernel void dequant_q8_0_gemv_f16out(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = half(sumf[row]);
        }
    }
}

// === GEMV + Residual Add fused — y[i] = sum_k dequant(W[i,k])*x[k] + residual[i] ===
// Eliminates separate elementwise_add dispatch after output/down projections.
kernel void dequant_q8_0_gemv_add(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant ERDequantQ8GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) xl[i] = xb[i];

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[row] += sumq * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row] + residual[row0 + row];  // fused add!
        }
    }
}

// =============================================================================
// === Tile-Based GEMV with Coalesced Memory Access ============================
// =============================================================================
// This kernel restructures Q8_0 GEMV to use 2D tile-based access patterns.
// Problem: Current kernel has each thread loading x[] directly from DRAM with
// strided access (each thread loads elements 32 apart -> 32 separate memory
// streams per simdgroup, causing DRAM row buffer thrashing).
// Solution: All 32 threads cooperatively load a contiguous tile of x[] (1024
// elements) into threadgroup memory, then access from fast SRAM with coalesced
// patterns.
//
// Expected improvement: 207 GB/s -> 250+ GB/s (20% bandwidth increase)

kernel void dequant_q8_0_gemv_tiled(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    constexpr uint TILE_SIZE = 1024;  // 4KB, fits comfortably in threadgroup memory

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;
    const uint tilesPerRow = (params.cols + TILE_SIZE - 1) / TILE_SIZE;

    float sumf[LOCAL_NR] = { 0.f };

    // Pointers to weight rows
    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    // Threadgroup memory for the tile - shared across all 32 threads
    threadgroup float tile[TILE_SIZE];

    // Process the row in tiles
    for (uint tileIdx = 0; tileIdx < tilesPerRow; tileIdx++) {
        const uint tileOffset = tileIdx * TILE_SIZE;
        const uint remainingCols = params.cols - tileOffset;
        const uint tileLen = min(TILE_SIZE, remainingCols);

        // === Phase 1: Cooperatively load tile into threadgroup memory ===
        // All 32 threads participate in loading the tile contiguously
        // This creates coalesced memory access patterns
        for (uint i = tiisg; i < tileLen; i += 32) {
            tile[i] = x[tileOffset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Phase 2: Process Q8_0 blocks within this tile ===
        // Each thread processes blocks that fall within the current tile
        // A Q8_0 block covers 32 elements, so we process blocks [tileStartBlock, tileEndBlock)
        const uint tileStartBlock = (tileOffset) / 32;
        const uint tileEndBlock = min((tileOffset + tileLen + 31) / 32, (uint)nb);

        for (uint ib = tileStartBlock + tiisg; ib < tileEndBlock; ib += 32) {
            // Calculate position within tile for this block
            const uint blockStartInTile = (ib * 32) - tileOffset;

            // Load x values for this block from tile (fast SRAM access)
            float xl[32];
            for (short i = 0; i < 32; i++) {
                uint tilePos = blockStartInTile + i;
                if (tilePos < TILE_SIZE) {
                    xl[i] = tile[tilePos];
                } else {
                    // Fallback to device memory for edge cases (shouldn't happen with proper tile sizing)
                    xl[i] = x[ib * 32 + i];
                }
            }

            // Process this block for all rows
            for (short row = 0; row < LOCAL_NR; row++) {
                if (row0 + row >= params.rows) break;

                device const uchar* block = ax[row] + ib * q8_0BlockBytes;
                float scale = float(as_type<half>(*(device const ushort*)block));
                device const char* qs = (device const char*)(block + 2);

                float sumq = 0.f;
                for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
                sumf[row] += sumq * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Single simd_sum reduction
    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// =============================================================================
// === f16 Accumulation Variant of Tiled GEMV ==================================
// =============================================================================
kernel void dequant_q8_0_gemv_tiled_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;
    constexpr uint TILE_SIZE = 1024;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;
    const uint tilesPerRow = (params.cols + TILE_SIZE - 1) / TILE_SIZE;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    threadgroup float tile[TILE_SIZE];

    for (uint tileIdx = 0; tileIdx < tilesPerRow; tileIdx++) {
        const uint tileOffset = tileIdx * TILE_SIZE;
        const uint remainingCols = params.cols - tileOffset;
        const uint tileLen = min(TILE_SIZE, remainingCols);

        // Cooperatively load tile
        for (uint i = tiisg; i < tileLen; i += 32) {
            tile[i] = x[tileOffset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tileStartBlock = (tileOffset) / 32;
        const uint tileEndBlock = min((tileOffset + tileLen + 31) / 32, (uint)nb);

        for (uint ib = tileStartBlock + tiisg; ib < tileEndBlock; ib += 32) {
            const uint blockStartInTile = (ib * 32) - tileOffset;

            half xl[32];
            for (short i = 0; i < 32; i++) {
                uint tilePos = blockStartInTile + i;
                if (tilePos < TILE_SIZE) {
                    xl[i] = half(tile[tilePos]);
                } else {
                    xl[i] = half(x[ib * 32 + i]);
                }
            }

            for (short row = 0; row < LOCAL_NR; row++) {
                if (row0 + row >= params.rows) break;

                device const uchar* block = ax[row] + ib * q8_0BlockBytes;
                float scale = float(as_type<half>(*(device const ushort*)block));
                device const char* qs = (device const char*)(block + 2);

                half sumq = 0.h;
                for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
                sumf[row] += float(sumq) * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}
// These kernels use half-precision for the inner dot product (xl[] cache and
// per-block sumq), halving register pressure and doubling ALU throughput on
// Apple Silicon. The outer cross-block accumulator (sumf[]) stays float32 to
// prevent drift over hundreds of blocks.

// --- 1. dequant_q8_0_gemv_f16acc ---
kernel void dequant_q8_0_gemv_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) xl[i] = half(xb[i]);

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[row] += float(sumq) * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row];
        }
    }
}

// TurboQuant decode variant: same fused RMSNorm + Q/K/V projection, but V stays in f32
// so it can be packed into TurboQuant immediately after RoPE(K) without falling back to
// separate RMSNorm + Q/K/V GEMV dispatches.
kernel void dequant_q8_0_fused_qkv_turbo(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device float* outV [[buffer(6)]],
    device const float* normWeight [[buffer(8)]],
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device float* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[r] += sumq * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumf[r] = simd_sum(sumf[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = total;
            }
        }
    }
}

kernel void dequant_q8_0_fused_qkv_turbo_hybrid_v(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],
    device const float* normWeight [[buffer(8)]],
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device half* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        float xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = xb[i] * rmsScale * normWeight[ib * 32 + i];
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            float sumq = 0.f;
            for (short i = 0; i < 32; i++) sumq += float(qs[i]) * xl[i];
            sumf[r] += sumq * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumf[r] = simd_sum(sumf[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = half(total);
            }
        }
    }
}

// --- 2. dequant_q8_0_fused_qkv_f16acc ---
kernel void dequant_q8_0_fused_qkv_f16acc(
    device const uchar* wq [[buffer(0)]],
    device const uchar* wk [[buffer(1)]],
    device const uchar* wv [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* outQ [[buffer(4)]],
    device float* outK [[buffer(5)]],
    device half* outV [[buffer(6)]],
    device const float* normWeight [[buffer(8)]],
    constant ERFusedQKVParams& params [[buffer(7)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint totalRows = params.qRows + params.kvRows + params.kvRows;
    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= totalRows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenOutQ = outQ + tokenIndex * params.qRows;
    device float* tokenOutK = outK + tokenIndex * params.kvRows;
    device half* tokenOutV = outV + tokenIndex * params.kvRows;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint globalRow = row0 + r;
        if (globalRow >= totalRows) { ax[r] = wq; continue; }
        uint localRow;
        device const uchar* weights;
        if (globalRow < params.qRows) {
            localRow = globalRow;
            weights = wq;
        } else if (globalRow < params.qRows + params.kvRows) {
            localRow = globalRow - params.qRows;
            weights = wk;
        } else {
            localRow = globalRow - params.qRows - params.kvRows;
            weights = wv;
        }
        ax[r] = weights + localRow * nb * q8_0BlockBytes;
    }

    // === Cooperative RMSNorm: stays in f32 (runs once, not on hot path) ===
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    // === Main GEMV loop: f16 inner accumulation ===
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = half(xb[i] * rmsScale * normWeight[ib * 32 + i]);
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= totalRows) break;
            device const uchar* block = ax[r] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[r] += float(sumq) * scale;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) sumf[r] = simd_sum(sumf[r]);

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            uint globalRow = row0 + r;
            if (globalRow >= totalRows) break;

            float total = sumf[r];
            if (globalRow < params.qRows) {
                tokenOutQ[globalRow] = total;
            } else if (globalRow < params.qRows + params.kvRows) {
                tokenOutK[globalRow - params.qRows] = total;
            } else {
                tokenOutV[globalRow - params.qRows - params.kvRows] = half(total);
            }
        }
    }
}

// --- 3. dequant_q8_0_fused_gate_up_silu_f16acc ---
kernel void dequant_q8_0_fused_gate_up_silu_f16acc(
    device const uchar* wGate [[buffer(0)]],
    device const uchar* wUp [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* activated [[buffer(3)]],
    device const float* normWeight [[buffer(5)]],
    constant ERFusedGateUpSiluParams& params [[buffer(4)]],
    uint2 tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex.x * LOCAL_NR;
    const uint tokenIndex = tgIndex.y;
    if (row0 >= params.rows || tokenIndex >= params.tokenCount) return;

    const short nb = params.blocksPerRow;
    device const float* tokenX = x + tokenIndex * params.cols;
    device float* tokenActivated = activated + tokenIndex * params.rows;

    // === Cooperative RMSNorm: stays in f32 ===
    float sumSq = 0.0f;
    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        for (short i = 0; i < 32; i++) {
            float v = xb[i];
            sumSq += v * v;
        }
    }
    sumSq = simd_sum(sumSq);
    float rmsScale = rsqrt(sumSq / float(params.cols) + params.rmsEps);

    float sumGate[LOCAL_NR] = { 0.f };
    float sumUp[LOCAL_NR] = { 0.f };

    device const uchar* axGate[LOCAL_NR];
    device const uchar* axUp[LOCAL_NR];
    for (short r = 0; r < LOCAL_NR; r++) {
        uint row = row0 + r;
        uint safeRow = row < params.rows ? row : row0;
        axGate[r] = wGate + safeRow * nb * q8_0BlockBytes;
        axUp[r] = wUp + safeRow * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = tokenX + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) {
            xl[i] = half(xb[i] * rmsScale * normWeight[ib * 32 + i]);
        }

        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;

            // Gate
            device const uchar* blockG = axGate[r] + ib * q8_0BlockBytes;
            float scaleG = float(as_type<half>(*(device const ushort*)blockG));
            device const char* qsG = (device const char*)(blockG + 2);
            half sqG = 0.h;
            for (short i = 0; i < 32; i++) sqG += half(qsG[i]) * xl[i];
            sumGate[r] += float(sqG) * scaleG;

            // Up
            device const uchar* blockU = axUp[r] + ib * q8_0BlockBytes;
            float scaleU = float(as_type<half>(*(device const ushort*)blockU));
            device const char* qsU = (device const char*)(blockU + 2);
            half sqU = 0.h;
            for (short i = 0; i < 32; i++) sqU += half(qsU[i]) * xl[i];
            sumUp[r] += float(sqU) * scaleU;
        }
    }

    for (short r = 0; r < LOCAL_NR; r++) {
        sumGate[r] = simd_sum(sumGate[r]);
        sumUp[r] = simd_sum(sumUp[r]);
    }

    if (tiisg == 0) {
        for (short r = 0; r < LOCAL_NR; r++) {
            if (row0 + r >= params.rows) break;
            tokenActivated[row0 + r] = silu_fn(sumGate[r]) * sumUp[r];
        }
    }
}

// --- 4. dequant_q8_0_gemv_f16out_f16acc ---
kernel void dequant_q8_0_gemv_f16out_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant ERDequantQ8GEMVParams& params [[buffer(3)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) xl[i] = half(xb[i]);

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[row] += float(sumq) * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = half(sumf[row]);
        }
    }
}

// =============================================================================
// === Fused FFN Block (Mega-Kernel) ==========================================
// =============================================================================
// Merges 3 GPU dispatches into 1 per transformer layer:
//   Phase 1: Wo GEMV + residual add     (replaces Dispatch 3)
//   Phase 2: RMSNorm                     (was implicit in Dispatch 4)
//   Phase 3: Gate + Up + SwiGLU GEMV     (replaces Dispatch 4)
//   Phase 4: Down GEMV + residual add    (replaces Dispatch 5)
//
// Architecture: 1 threadgroup x 1024 threads (32 simdgroups).
// Dispatched once per layer (28 times total).
// Uses threadgroup_barrier instead of pipeline drains between phases.

struct ERFusedFFNBlockParams {
    uint dim;               // 1024 (Wo output rows, Down output rows)
    uint qDim;              // 2048 (Wo input cols = attn output dim)
    uint interDim;          // 3072 (Gate/Up output rows, Down input cols)
    uint woBlocksPerRow;    // qDim/32 = 64
    uint ffnBlocksPerRow;   // dim/32 = 32
    uint downBlocksPerRow;  // interDim/32 = 96
    float rmsEps;           // RMSNorm epsilon
};

kernel void dequant_q8_0_fused_ffn_block(
    device const uchar*  woRaw       [[buffer(0)]],   // Wo weights Q8_0 [dim x qDim]
    device const float*  attnOut     [[buffer(1)]],   // attention output [qDim]
    device const float*  residual    [[buffer(2)]],   // currentHidden (residual for Wo add) [dim]
    device       float*  afterAttn   [[buffer(3)]],   // Wo output dest (also FFN residual) [dim]
    device const uchar*  gateRaw     [[buffer(4)]],   // gate weights Q8_0 [interDim x dim]
    device const uchar*  upRaw       [[buffer(5)]],   // up weights Q8_0 [interDim x dim]
    device const float*  normWeight  [[buffer(6)]],   // FFN RMSNorm weight [dim]
    device       float*  activBuf    [[buffer(7)]],   // intermediate activated [interDim]
    device const uchar*  downRaw     [[buffer(8)]],   // down weights Q8_0 [dim x interDim]
    device       float*  layerOutput [[buffer(9)]],   // final output [dim]
    constant ERFusedFFNBlockParams& params [[buffer(10)]],
    uint  tid    [[thread_index_in_threadgroup]],
    ushort sgIdx [[simdgroup_index_in_threadgroup]],
    ushort laneIdx [[thread_index_in_simdgroup]]
) {
    // Shared memory for cross-simdgroup RMSNorm reduction (32 simdgroups)
    threadgroup float partial_sums[32];

    const uint dim      = params.dim;        // 1024
    const uint qDim     = params.qDim;       // 2048
    const uint interDim = params.interDim;   // 3072

    // =========================================================================
    // Phase 1: Wo GEMV + residual add
    //   afterAttn[i] = dot(Wo[i,:], attnOut[:]) + residual[i]
    //   Each thread computes 1 output row (tid < dim=1024, all threads active)
    // =========================================================================
    {
        const uint woNb = params.woBlocksPerRow;  // qDim/32 = 64
        device const uchar* rowPtr = woRaw + tid * woNb * q8_0BlockBytes;

        float acc = 0.0f;
        for (uint ib = 0; ib < woNb; ib++) {
            device const uchar* block = rowPtr + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            device const float* xb = attnOut + ib * q8_0WeightsPerBlock;

            half sumq = 0.h;
            for (ushort j = 0; j < 32; j++) {
                sumq += half(qs[j]) * half(xb[j]);
            }
            acc += float(sumq) * scale;
        }

        afterAttn[tid] = acc + residual[tid];
    }

    // =========================================================================
    // Phase 2: Barrier — all 1024 Wo outputs must be visible
    // =========================================================================
    threadgroup_barrier(mem_flags::mem_device);

    // =========================================================================
    // Phase 3: Cooperative RMSNorm over afterAttn[dim=1024]
    //   normed[i] = afterAttn[i] * scale * normWeight[i]
    //   where scale = rsqrt(mean(afterAttn^2) + eps)
    //
    //   Each thread reads afterAttn[tid], squares it.
    //   Reduce within simdgroup via simd_sum, then cross-SG via threadgroup mem.
    // =========================================================================
    float myVal = afterAttn[tid];
    float mySq  = myVal * myVal;

    // Intra-simdgroup reduction
    float sgSum = simd_sum(mySq);

    // Cross-simdgroup reduction: lane 0 of each SG writes to shared mem
    if (laneIdx == 0) {
        partial_sums[sgIdx] = sgSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the 32 partial sums
    float totalSq = 0.0f;
    if (sgIdx == 0) {
        totalSq = (laneIdx < 32) ? partial_sums[laneIdx] : 0.0f;
        totalSq = simd_sum(totalSq);
    }

    // Broadcast the total to all threads via shared memory
    if (tid == 0) {
        partial_sums[0] = totalSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    totalSq = partial_sums[0];

    float rmsScale = rsqrt(totalSq / float(dim) + params.rmsEps);
    float normed = myVal * rmsScale * normWeight[tid];

    // =========================================================================
    // Phase 4: Gate + Up + SwiGLU GEMV
    //   interDim=3072 output rows, 1024 threads => 3 rows per thread
    //   For each assigned row r:
    //     gate = dot(Wgate[r,:], normed_vec[:])
    //     up   = dot(Wup[r,:],   normed_vec[:])
    //     activBuf[r] = silu(gate) * up
    //
    //   Problem: normed_vec is distributed (each thread holds 1 element).
    //   Solution: store normed to device memory, barrier, read back.
    // =========================================================================

    // Write normed value to afterAttn (reuse as temporary — we still have myVal
    // for Phase 6 residual, and afterAttn is the FFN residual anyway)
    afterAttn[tid] = normed;
    threadgroup_barrier(mem_flags::mem_device);

    // Each thread computes 3 rows of gate+up
    const uint rowsPerThread = interDim / dim;  // 3072/1024 = 3
    const uint ffnNb = params.ffnBlocksPerRow;  // dim/32 = 32

    for (uint r = 0; r < rowsPerThread; r++) {
        uint row = tid * rowsPerThread + r;
        if (row >= interDim) break;

        device const uchar* gateRow = gateRaw + row * ffnNb * q8_0BlockBytes;
        device const uchar* upRow   = upRaw   + row * ffnNb * q8_0BlockBytes;

        float accGate = 0.0f;
        float accUp   = 0.0f;

        for (uint ib = 0; ib < ffnNb; ib++) {
            device const float* nb_x = afterAttn + ib * q8_0WeightsPerBlock;

            // Gate block
            device const uchar* gBlock = gateRow + ib * q8_0BlockBytes;
            float gScale = float(as_type<half>(*(device const ushort*)gBlock));
            device const char* gQs = (device const char*)(gBlock + 2);

            // Up block
            device const uchar* uBlock = upRow + ib * q8_0BlockBytes;
            float uScale = float(as_type<half>(*(device const ushort*)uBlock));
            device const char* uQs = (device const char*)(uBlock + 2);

            half sqG = 0.h;
            half sqU = 0.h;
            for (ushort j = 0; j < 32; j++) {
                half xv = half(nb_x[j]);
                sqG += half(gQs[j]) * xv;
                sqU += half(uQs[j]) * xv;
            }
            accGate += float(sqG) * gScale;
            accUp   += float(sqU) * uScale;
        }

        activBuf[row] = silu_fn(accGate) * accUp;
    }

    // =========================================================================
    // Phase 5: Barrier — all interDim=3072 activated values must be visible
    // =========================================================================
    threadgroup_barrier(mem_flags::mem_device);

    // =========================================================================
    // Phase 6: Down GEMV + residual add
    //   layerOutput[i] = dot(Wdown[i,:], activBuf[:]) + afterAttn_before_norm[i]
    //   Each thread computes 1 output row (tid < dim=1024)
    //   Residual is the Wo output + old residual, which we stored in afterAttn
    //   before we overwrote it with normed values.
    //
    //   Wait — we overwrote afterAttn with normed in Phase 4.
    //   We need the pre-norm value for the residual.
    //   Solution: use (acc + residual[tid]) which we computed in Phase 1.
    //   We saved myVal = afterAttn[tid] before norming, and
    //   afterAttn[tid] was (acc + residual[tid]) from Phase 1.
    //   So myVal IS the correct residual for Phase 6.
    // =========================================================================
    {
        const uint downNb = params.downBlocksPerRow;  // interDim/32 = 96
        device const uchar* rowPtr = downRaw + tid * downNb * q8_0BlockBytes;

        float acc = 0.0f;
        for (uint ib = 0; ib < downNb; ib++) {
            device const uchar* block = rowPtr + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            device const float* xb = activBuf + ib * q8_0WeightsPerBlock;

            half sumq = 0.h;
            for (ushort j = 0; j < 32; j++) {
                sumq += half(qs[j]) * half(xb[j]);
            }
            acc += float(sumq) * scale;
        }

        // myVal holds the Phase 1 output (Wo GEMV + residual) = correct FFN residual
        layerOutput[tid] = acc + myVal;
    }
}

// --- 5. dequant_q8_0_gemv_add_f16acc ---
kernel void dequant_q8_0_gemv_add_f16acc(
    device const uchar* quantisedW [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant ERDequantQ8GEMVParams& params [[buffer(4)]],
    uint tgIndex [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    constexpr short LOCAL_NR = 2;

    const uint row0 = tgIndex * LOCAL_NR;
    if (row0 >= params.rows) return;

    const short nb = params.blocksPerRow;

    float sumf[LOCAL_NR] = { 0.f };

    device const uchar* ax[LOCAL_NR];
    for (short row = 0; row < LOCAL_NR; row++) {
        uint r = row0 + row;
        ax[row] = quantisedW + (r < params.rows ? r : row0) * nb * q8_0BlockBytes;
    }

    for (short ib = tiisg; ib < nb; ib += 32) {
        device const float* xb = x + ib * 32;
        half xl[32];
        for (short i = 0; i < 32; i++) xl[i] = half(xb[i]);

        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            device const uchar* block = ax[row] + ib * q8_0BlockBytes;
            float scale = float(as_type<half>(*(device const ushort*)block));
            device const char* qs = (device const char*)(block + 2);
            half sumq = 0.h;
            for (short i = 0; i < 32; i++) sumq += half(qs[i]) * xl[i];
            sumf[row] += float(sumq) * scale;
        }
    }

    for (short row = 0; row < LOCAL_NR; row++) sumf[row] = simd_sum(sumf[row]);

    if (tiisg == 0) {
        for (short row = 0; row < LOCAL_NR; row++) {
            if (row0 + row >= params.rows) break;
            y[row0 + row] = sumf[row] + residual[row0 + row];
        }
    }
}


// --- Elementwise.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERElementwiseParams {
    uint elementCount;
};

kernel void elementwise_add_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void elementwise_sub_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] - b[tid];
    }
}

kernel void elementwise_mul_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] * b[tid];
    }
}

kernel void elementwise_div_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] / b[tid];
    }
}

kernel void elementwise_add_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void elementwise_sub_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] - b[tid];
    }
}

kernel void elementwise_mul_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] * b[tid];
    }
}

kernel void elementwise_div_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant ERElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        out[tid] = a[tid] / b[tid];
    }
}

// === Precision conversion kernels ===

kernel void convert_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ERElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        output[tid] = half(input[tid]);
    }
}

kernel void convert_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        output[tid] = float(input[tid]);
    }
}

struct ERCaptureF32SliceParams {
    uint sourceOffsetElements;
    uint destinationOffsetElements;
    uint elementCount;
};

// Copies a contiguous float32 slice from one shared buffer into another.
kernel void capture_f32_slice(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant ERCaptureF32SliceParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.elementCount) {
        destination[params.destinationOffsetElements + tid] =
            source[params.sourceOffsetElements + tid];
    }
}


// --- FlashAttention.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERFlashAttentionParams {
    uint seqLen;
    uint headDim;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
};

struct ERFlashGQAParams {
    uint seqLen;
    uint headDim;
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
    uint kvSeqLen;
    uint qOffset;
};

kernel void flash_attention_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashAttentionParams &params [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    const uint br = params.qBlockSize;
    const uint bc = params.kvBlockSize;
    const uint headDim = params.headDim;
    const uint seqLen = params.seqLen;

    uint qRow = group_id * br + local_id;
    if (qRow >= seqLen) {
        return;
    }

    threadgroup float kTile[16 * 128];
    threadgroup float vTile[16 * 128];
    threadgroup float outputScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    for (uint dim = 0; dim < headDim; dim++) {
        outputScratch[local_id * headDim + dim] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (seqLen + bc - 1) / bc;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * bc;
        uint kvEnd = min(kvStart + bc, seqLen);
        uint kvCount = kvEnd - kvStart;

        if (local_id < kvCount) {
            for (uint dim = 0; dim < headDim; dim++) {
                kTile[local_id * headDim + dim] = K[(kvStart + local_id) * headDim + dim];
                vTile[local_id * headDim + dim] = V[(kvStart + local_id) * headDim + dim];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float blockMax = -INFINITY;
        float scores[16];
        for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
            if (params.causal != 0 && kvStart + kvIndex > qRow) {
                scores[kvIndex] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            for (uint dim = 0; dim < headDim; dim++) {
                dot += Q[qRow * headDim + dim] * kTile[kvIndex * headDim + dim];
            }
            scores[kvIndex] = dot * params.scale;
            blockMax = max(blockMax, scores[kvIndex]);
        }

        float nextMax = max(runningMax, blockMax);
        float correction = exp(runningMax - nextMax);

        float blockSum = 0.0f;
        float probs[16];
        for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
            if (scores[kvIndex] == -INFINITY) {
                probs[kvIndex] = 0.0f;
            } else {
                probs[kvIndex] = exp(scores[kvIndex] - nextMax);
            }
            blockSum += probs[kvIndex];
        }

        runningSum = runningSum * correction + blockSum;

        for (uint dim = 0; dim < headDim; dim++) {
            float value = outputScratch[local_id * headDim + dim] * correction;
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                value += probs[kvIndex] * vTile[kvIndex * headDim + dim];
            }
            outputScratch[local_id * headDim + dim] = value;
        }

        runningMax = nextMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
    for (uint dim = 0; dim < headDim; dim++) {
        O[qRow * headDim + dim] = outputScratch[local_id * headDim + dim] * invSum;
    }
}

kernel void flash_attention_gqa_simd_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashGQAParams &params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint qRow = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || qRow >= params.seqLen) return;

    uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : params.seqLen;
    uint qPosition = qRow + params.qOffset;
    uint kvHeadIndex = headIndex / params.groupSize;
    uint qStride = params.numHeads * params.headDim;
    uint kvStride = params.numKVHeads * params.headDim;
    uint qBase = qRow * qStride + headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < kvSeqLen; ++kv) {
        if (params.causal != 0 && kv > qPosition) break;

        uint kvBase = kv * kvStride + kvHeadIndex * params.headDim;
        float partial =
            q0 * K[kvBase + lane] +
            q1 * K[kvBase + lane + 32] +
            q2 * K[kvBase + lane + 64] +
            q3 * K[kvBase + lane + 96];
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        acc0 = acc0 * correction + prob * V[kvBase + lane];
        acc1 = acc1 * correction + prob * V[kvBase + lane + 32];
        acc2 = acc2 * correction + prob * V[kvBase + lane + 64];
        acc3 = acc3 * correction + prob * V[kvBase + lane + 96];
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}

kernel void flash_attention_gqa_simd_qf32_kvf16(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashGQAParams &params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint qRow = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || qRow >= params.seqLen) return;

    uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : params.seqLen;
    uint qPosition = qRow + params.qOffset;
    uint kvHeadIndex = headIndex / params.groupSize;
    uint qStride = params.numHeads * params.headDim;
    uint kvStride = params.numKVHeads * params.headDim;
    uint qBase = qRow * qStride + headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < kvSeqLen; ++kv) {
        if (params.causal != 0 && kv > qPosition) break;

        uint kvBase = kv * kvStride + kvHeadIndex * params.headDim;
        float partial =
            q0 * float(K[kvBase + lane]) +
            q1 * float(K[kvBase + lane + 32]) +
            q2 * float(K[kvBase + lane + 64]) +
            q3 * float(K[kvBase + lane + 96]);
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        acc0 = acc0 * correction + prob * float(V[kvBase + lane]);
        acc1 = acc1 * correction + prob * float(V[kvBase + lane + 32]);
        acc2 = acc2 * correction + prob * float(V[kvBase + lane + 64]);
        acc3 = acc3 * correction + prob * float(V[kvBase + lane + 96]);
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}

kernel void flash_attention_gqa_simd_qf32_kpacked_vf16(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashGQAParams &params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint qRow = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || qRow >= params.seqLen) return;

    uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : params.seqLen;
    uint qPosition = qRow + params.qOffset;
    uint kvHeadIndex = headIndex / params.groupSize;
    uint qStride = params.numHeads * params.headDim;
    uint kvStride = params.numKVHeads * params.headDim;
    uint qBase = qRow * qStride + headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < kvSeqLen; ++kv) {
        if (params.causal != 0 && kv > qPosition) break;

        uint packedKBase = (kv * params.numKVHeads + kvHeadIndex) * params.headDim + lane * 4;
        uint vBase = kv * kvStride + kvHeadIndex * params.headDim;
        half4 packedK = *reinterpret_cast<const device half4 *>(K + packedKBase);
        float partial =
            q0 * float(packedK[0]) +
            q1 * float(packedK[1]) +
            q2 * float(packedK[2]) +
            q3 * float(packedK[3]);
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        acc0 = acc0 * correction + prob * float(V[vBase + lane]);
        acc1 = acc1 * correction + prob * float(V[vBase + lane + 32]);
        acc2 = acc2 * correction + prob * float(V[vBase + lane + 64]);
        acc3 = acc3 * correction + prob * float(V[vBase + lane + 96]);
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}

kernel void flash_attention_gqa_simd_qf32_kvpacked(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERFlashGQAParams &params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint qRow = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || qRow >= params.seqLen) return;

    uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : params.seqLen;
    uint qPosition = qRow + params.qOffset;
    uint kvHeadIndex = headIndex / params.groupSize;
    uint qStride = params.numHeads * params.headDim;
    uint qBase = qRow * qStride + headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < kvSeqLen; ++kv) {
        if (params.causal != 0 && kv > qPosition) break;

        uint packedBase = (kv * params.numKVHeads + kvHeadIndex) * params.headDim + lane * 4;
        half4 packedK = *reinterpret_cast<const device half4 *>(K + packedBase);
        float partial =
            q0 * float(packedK[0]) +
            q1 * float(packedK[1]) +
            q2 * float(packedK[2]) +
            q3 * float(packedK[3]);
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        half4 packedV = *reinterpret_cast<const device half4 *>(V + packedBase);
        acc0 = acc0 * correction + prob * float(packedV[0]);
        acc1 = acc1 * correction + prob * float(packedV[1]);
        acc2 = acc2 * correction + prob * float(packedV[2]);
        acc3 = acc3 * correction + prob * float(packedV[3]);
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}


// --- FusedPatterns.metal ---
#include <metal_stdlib>
using namespace metal;

// function_constant(0): selects which activation to apply after the binary op.
// Values: 0 = none, 1 = relu, 2 = sigmoid, 3 = gelu, 4 = silu.
// The Metal compiler eliminates dead branches at pipeline-creation time,
// producing a specialised kernel with zero runtime overhead.
constant int activation_type [[function_constant(0)]];

inline float apply_activation(float x) {
    if (activation_type == 1) {
        return max(x, 0.0f);
    } else if (activation_type == 2) {
        return 1.0f / (1.0f + exp(-x));
    } else if (activation_type == 3) {
        const float kSqrt2OverPi = 0.7978845608f;
        float cube = x * x * x;
        return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + 0.044715f * cube)));
    } else if (activation_type == 4) {
        return x / (1.0f + exp(-x));
    }
    return x; // activation_type == 0: identity
}

// Fused add + activation kernel (float32).
kernel void fused_add_activate_float(
    device const float* a        [[buffer(0)]],
    device const float* b        [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  elementCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    out[tid] = apply_activation(a[tid] + b[tid]);
}

// Fused multiply + activation kernel (float32).
kernel void fused_mul_activate_float(
    device const float* a        [[buffer(0)]],
    device const float* b        [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  elementCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    out[tid] = apply_activation(a[tid] * b[tid]);
}

// Fused unary activation kernel (float32).
kernel void fused_activate_float(
    device const float* input    [[buffer(0)]],
    device       float* output   [[buffer(1)]],
    constant     uint&  elementCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elementCount) return;
    output[tid] = apply_activation(input[tid]);
}


// --- GEMM.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERGEMMParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.M || col >= params.N) return;
    float sum = 0.0;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.lda + k] * B[k * params.ldb + col];
    }
    C[row * params.ldc + col] = sum;
}

kernel void gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.M || col >= params.N) return;
    half sum = 0.0h;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.lda + k] * B[k * params.ldb + col];
    }
    C[row * params.ldc + col] = sum;
}

kernel void gemm_f32_packed_prefill(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant ERGEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.M || col >= params.N) return;

    device const float* aRow = A + row * params.lda;
    device const float* bCol = B + col;

    float sum = 0.0f;
    uint k = 0;
    for (; k + 3 < params.K; k += 4) {
        float4 av(
            aRow[k + 0],
            aRow[k + 1],
            aRow[k + 2],
            aRow[k + 3]
        );
        float4 bv(
            bCol[(k + 0) * params.ldb],
            bCol[(k + 1) * params.ldb],
            bCol[(k + 2) * params.ldb],
            bCol[(k + 3) * params.ldb]
        );
        sum += dot(av, bv);
    }
    for (; k < params.K; ++k) {
        sum += aRow[k] * bCol[k * params.ldb];
    }

    C[row * params.ldc + col] = sum;
}


// --- GEMV.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERGEMVParams {
    uint M;
    uint K;
    uint lda;
};

// Each threadgroup handles one row.
// Threads within the group cooperatively reduce across K.
// Uses simd_sum for fast warp-level reduction.
constant uint GEMV_THREADS_PER_ROW = 256;

kernel void gemv_f32(
    device const float*      A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    // Each thread accumulates a partial dot product
    float partial = 0.0f;
    device const float* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        partial += a_row[j] * x[j];
    }

    // Warp-level reduction
    partial = simd_sum(partial);

    // Cross-warp reduction via threadgroup memory
    threadgroup float shared_sums[32]; // max 32 simdgroups (1024/32)

    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp finalizes
    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = val;
        }
    }
}

kernel void gemv_f16(
    device const half*       A       [[buffer(0)]],
    device const half*       x       [[buffer(1)]],
    device half*             y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    // Accumulate in float for numerical stability
    float partial = 0.0f;
    device const half* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        partial += float(a_row[j]) * float(x[j]);
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = half(val);
        }
    }
}

kernel void gemv_bf16_f32(
    device const ushort*     A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint  group_id     [[threadgroup_position_in_grid]],
    uint  local_id     [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.M) return;

    float partial = 0.0f;
    device const ushort* a_row = A + row * params.lda;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        float a = as_type<float>(uint(a_row[j]) << 16);
        partial += a * x[j];
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[row] = val;
        }
    }
}

kernel void gemv_bf16_f32_batched(
    device const ushort*     A       [[buffer(0)]],
    device const float*      x       [[buffer(1)]],
    device float*            y       [[buffer(2)]],
    constant ERGEMVParams&   params  [[buffer(3)]],
    uint3 group_id     [[threadgroup_position_in_grid]],
    uint3 local_pos    [[thread_position_in_threadgroup]],
    uint  simd_lane    [[thread_index_in_simdgroup]],
    uint  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.x;
    uint batch = group_id.y;
    uint local_id = local_pos.x;
    if (row >= params.M) return;

    float partial = 0.0f;
    device const ushort* a_row = A + row * params.lda;
    device const float* x_row = x + batch * params.K;

    for (uint j = local_id; j < params.K; j += GEMV_THREADS_PER_ROW) {
        float a = as_type<float>(uint(a_row[j]) << 16);
        partial += a * x_row[j];
    }

    partial = simd_sum(partial);

    threadgroup float shared_sums[32];
    if (simd_lane == 0) {
        shared_sums[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (GEMV_THREADS_PER_ROW + 31) / 32;
        float val = (simd_lane < num_simdgroups) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            y[batch * params.M + row] = val;
        }
    }
}


// --- GQA.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERGQAParams {
    uint seqLen;
    uint headDim;
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    float scale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
    uint kvSeqLen;    // K/V sequence length (0 = same as seqLen)
    uint qOffset;     // offset for Q positions in causal mask (0 = default)
};

kernel void gqa_attention_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint headDim = params.headDim;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;  // causal mask: Q position = qRow + qOffset
    const uint blockSize = params.qBlockSize;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    // Track whether this thread has a valid Q position.
    // Inactive Q threads still participate in KV tile loading and barriers.
    bool activeQ = (qRow < seqLen);

    // [S, H, D] layout strides
    const uint qStride = numHeads * headDim;       // stride between sequence positions for Q/O
    const uint kvStride = numKVHeads * headDim;     // stride between sequence positions for K/V

    const uint headDim4 = headDim / 4;
    const uint maxHeadDim4 = 128 / 4;

    threadgroup float4 kTile[16 * maxHeadDim4];
    threadgroup float4 vTile[16 * maxHeadDim4];
    threadgroup float4 outputScratch[16 * maxHeadDim4];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim4 = 0; dim4 < headDim4; dim4++) {
            outputScratch[local_id.x * maxHeadDim4 + dim4] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        // ALL threads participate in KV tile loading (not just active Q threads)
        if (local_id.x < kvCount) {
            uint kvPos = kvStart + local_id.x;
            uint kBase = kvPos * kvStride + kvHeadIndex * headDim;
            const device float4 *kVec = reinterpret_cast<const device float4 *>(K + kBase);
            const device float4 *vVec = reinterpret_cast<const device float4 *>(V + kBase);
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                uint tileIndex = local_id.x * maxHeadDim4 + dim4;
                kTile[tileIndex] = kVec[dim4];
                vTile[tileIndex] = vVec[dim4];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                float dot = 0.0f;
                uint tileBase = kvIndex * maxHeadDim4;
                for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                    dot += metal::dot(qVec[dim4], kTile[tileBase + dim4]);
                }
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);

            float blockSum = 0.0f;
            float probs[16];
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (scores[kvIndex] == -INFINITY) {
                    probs[kvIndex] = 0.0f;
                } else {
                    probs[kvIndex] = exp(scores[kvIndex] - nextMax);
                }
                blockSum += probs[kvIndex];
            }

            runningSum = runningSum * correction + blockSum;

            uint outBase = local_id.x * maxHeadDim4;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                float4 value = outputScratch[outBase + dim4] * correction;
                for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                    value += probs[kvIndex] * vTile[kvIndex * maxHeadDim4 + dim4];
                }
                outputScratch[outBase + dim4] = value;
            }

            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        device float4 *oVec = reinterpret_cast<device float4 *>(O + oBase);
        uint outBase = local_id.x * maxHeadDim4;
        for (uint dim4 = 0; dim4 < headDim4; dim4++) {
            oVec[dim4] = outputScratch[outBase + dim4] * invSum;
        }
    }
}

static inline float gqa_attention_f32_wide_score(
    device const float *Q,
    device const float *K,
    constant ERGQAParams &params,
    uint qRow,
    uint headIndex,
    uint kvHeadIndex,
    uint kvPos,
    uint qStride,
    uint kvStride
) {
    const uint qBase = qRow * qStride + headIndex * params.headDim;
    const uint kBase = kvPos * kvStride + kvHeadIndex * params.headDim;
    float dot = 0.0f;
    for (uint dim = 0; dim < params.headDim; ++dim) {
        dot += Q[qBase + dim] * K[kBase + dim];
    }
    return dot * params.scale;
}

static inline void gqa_attention_f32_wide_impl(
    device const float *Q,
    device const float *K,
    device const float *V,
    device float *O,
    constant ERGQAParams &params,
    device const float *additiveMask,
    bool useAdditiveMask,
    uint outputIndex
) {
    const uint headDim = params.headDim;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint totalScalars = seqLen * numHeads * headDim;
    if (outputIndex >= totalScalars) {
        return;
    }

    const uint dim = outputIndex % headDim;
    const uint headIndex = (outputIndex / headDim) % numHeads;
    const uint qRow = outputIndex / (numHeads * headDim);
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint qStride = numHeads * headDim;
    const uint kvStride = numKVHeads * headDim;
    const uint causalLimit = qRow + params.qOffset;

    float maxScore = -INFINITY;
    for (uint kvPos = 0; kvPos < kvSeqLen; ++kvPos) {
        if (params.causal != 0 && kvPos > causalLimit) {
            continue;
        }
        float score = gqa_attention_f32_wide_score(
            Q,
            K,
            params,
            qRow,
            headIndex,
            kvHeadIndex,
            kvPos,
            qStride,
            kvStride
        );
        if (useAdditiveMask) {
            score += additiveMask[qRow * kvSeqLen + kvPos];
        }
        maxScore = max(maxScore, score);
    }

    if (maxScore == -INFINITY) {
        O[outputIndex] = 0.0f;
        return;
    }

    float sum = 0.0f;
    float value = 0.0f;
    for (uint kvPos = 0; kvPos < kvSeqLen; ++kvPos) {
        if (params.causal != 0 && kvPos > causalLimit) {
            continue;
        }
        float score = gqa_attention_f32_wide_score(
            Q,
            K,
            params,
            qRow,
            headIndex,
            kvHeadIndex,
            kvPos,
            qStride,
            kvStride
        );
        if (useAdditiveMask) {
            score += additiveMask[qRow * kvSeqLen + kvPos];
        }
        if (score == -INFINITY) {
            continue;
        }
        float weight = exp(score - maxScore);
        uint vBase = kvPos * kvStride + kvHeadIndex * headDim;
        value += weight * V[vBase + dim];
        sum += weight;
    }

    O[outputIndex] = sum > 0.0f ? value / sum : 0.0f;
}

kernel void gqa_attention_f32_wide(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    uint outputIndex [[thread_position_in_grid]]
) {
    gqa_attention_f32_wide_impl(Q, K, V, O, params, Q, false, outputIndex);
}

kernel void gqa_attention_f32_masked_wide(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    device const float *additiveMask [[buffer(5)]],
    uint outputIndex [[thread_position_in_grid]]
) {
    gqa_attention_f32_wide_impl(Q, K, V, O, params, additiveMask, true, outputIndex);
}

// === Float16 KV variant ===
// K/V stored as half in KV cache — halves attention memory bandwidth.
// Q and O remain float32 (fresh from GEMV). K/V converted to float in threadgroup memory.
kernel void gqa_attention_f16kv(
    device const float *Q [[buffer(0)]],
    device const half  *K [[buffer(1)]],
    device const half  *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint headDim = params.headDim;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;
    const uint blockSize = params.qBlockSize;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    const uint qStride = numHeads * headDim;
    const uint kvStride = numKVHeads * headDim;

    const uint headDim4 = headDim / 4;
    const uint maxHeadDim4 = 128 / 4;

    threadgroup float4 kTile[16 * maxHeadDim4];
    threadgroup float4 vTile[16 * maxHeadDim4];
    threadgroup float4 outputScratch[16 * maxHeadDim4];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim4 = 0; dim4 < headDim4; dim4++)
            outputScratch[local_id.x * maxHeadDim4 + dim4] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (local_id.x < kvCount) {
            uint kvPos = kvStart + local_id.x;
            uint kBase = kvPos * kvStride + kvHeadIndex * headDim;
            const device half4 *kVec = reinterpret_cast<const device half4 *>(K + kBase);
            const device half4 *vVec = reinterpret_cast<const device half4 *>(V + kBase);
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                uint tileIndex = local_id.x * maxHeadDim4 + dim4;
                kTile[tileIndex] = float4(kVec[dim4]);
                vTile[tileIndex] = float4(vVec[dim4]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                uint tileBase = kvIndex * maxHeadDim4;
                for (uint dim4 = 0; dim4 < headDim4; dim4++)
                    dot += metal::dot(qVec[dim4], kTile[tileBase + dim4]);
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0f;
            float probs[16];
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                probs[kvIndex] = (scores[kvIndex] == -INFINITY) ? 0.0f : exp(scores[kvIndex] - nextMax);
                blockSum += probs[kvIndex];
            }
            runningSum = runningSum * correction + blockSum;

            uint outBase = local_id.x * maxHeadDim4;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                float4 value = outputScratch[outBase + dim4] * correction;
                for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++)
                    value += probs[kvIndex] * vTile[kvIndex * maxHeadDim4 + dim4];
                outputScratch[outBase + dim4] = value;
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        device float4 *oVec = reinterpret_cast<device float4 *>(O + oBase);
        uint outBase = local_id.x * maxHeadDim4;
        for (uint dim4 = 0; dim4 < headDim4; dim4++)
            oVec[dim4] = outputScratch[outBase + dim4] * invSum;
    }
}

constant uint gqa_q8_0_block_bytes = 34;
constant uint gqa_q8_0_weights_per_block = 32;

static inline float4 gqa_q8_0_load_float4(
    device const uchar *row,
    uint dim4
) {
    const uint scalarIndex = dim4 * 4;
    const uint blockIndex = scalarIndex / gqa_q8_0_weights_per_block;
    const uint inBlockIndex = scalarIndex % gqa_q8_0_weights_per_block;
    device const uchar *block = row + blockIndex * gqa_q8_0_block_bytes;
    const float scale = float(as_type<half>(*(device const ushort *) block));
    return scale * float4(
        float(as_type<char>(block[2 + inBlockIndex + 0])),
        float(as_type<char>(block[2 + inBlockIndex + 1])),
        float(as_type<char>(block[2 + inBlockIndex + 2])),
        float(as_type<char>(block[2 + inBlockIndex + 3]))
    );
}

kernel void gqa_attention_q8kv(
    device const float *Q [[buffer(0)]],
    device const uchar *K [[buffer(1)]],
    device const uchar *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGQAParams &params [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint headDim = params.headDim;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;
    const uint blockSize = params.qBlockSize;
    const uint numHeads = params.numHeads;
    const uint numKVHeads = params.numKVHeads;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    const uint qStride = numHeads * headDim;
    const uint headDim4 = headDim / 4;
    const uint maxHeadDim4 = 128 / 4;
    const uint q8BlocksPerRow = headDim / gqa_q8_0_weights_per_block;
    const uint q8RowBytes = q8BlocksPerRow * gqa_q8_0_block_bytes;

    threadgroup float4 kTile[16 * maxHeadDim4];
    threadgroup float4 vTile[16 * maxHeadDim4];
    threadgroup float4 outputScratch[16 * maxHeadDim4];

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    if (activeQ) {
        for (uint dim4 = 0; dim4 < headDim4; dim4++) {
            outputScratch[local_id.x * maxHeadDim4 + dim4] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; kvBlock++) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (local_id.x < kvCount) {
            uint kvPos = kvStart + local_id.x;
            uint rowIndex = kvPos * numKVHeads + kvHeadIndex;
            device const uchar *kRow = K + rowIndex * q8RowBytes;
            device const uchar *vRow = V + rowIndex * q8RowBytes;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                uint tileIndex = local_id.x * maxHeadDim4 + dim4;
                kTile[tileIndex] = gqa_q8_0_load_float4(kRow, dim4);
                vTile[tileIndex] = gqa_q8_0_load_float4(vRow, dim4);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (activeQ) {
            float blockMax = -INFINITY;
            float scores[16];
            uint qBase = qRow * qStride + headIndex * headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                uint tileBase = kvIndex * maxHeadDim4;
                for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                    dot += metal::dot(qVec[dim4], kTile[tileBase + dim4]);
                }
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0f;
            float probs[16];
            for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                probs[kvIndex] = (scores[kvIndex] == -INFINITY) ? 0.0f : exp(scores[kvIndex] - nextMax);
                blockSum += probs[kvIndex];
            }
            runningSum = runningSum * correction + blockSum;

            uint outBase = local_id.x * maxHeadDim4;
            for (uint dim4 = 0; dim4 < headDim4; dim4++) {
                float4 value = outputScratch[outBase + dim4] * correction;
                for (uint kvIndex = 0; kvIndex < kvCount; kvIndex++) {
                    value += probs[kvIndex] * vTile[kvIndex * maxHeadDim4 + dim4];
                }
                outputScratch[outBase + dim4] = value;
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * headDim;
        device float4 *oVec = reinterpret_cast<device float4 *>(O + oBase);
        uint outBase = local_id.x * maxHeadDim4;
        for (uint dim4 = 0; dim4 < headDim4; dim4++) {
            oVec[dim4] = outputScratch[outBase + dim4] * invSum;
        }
    }
}

struct ERPackKVDecodeCacheParams {
    uint tokenCount;
    uint numKVHeads;
    uint headDim;
    uint destinationStartToken;
};

kernel void pack_kv_decode_cache_f16(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant ERPackKVDecodeCacheParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint lane = gid.x;
    uint kvHead = gid.y;
    uint token = gid.z;
    if (lane >= 32 || kvHead >= params.numKVHeads || token >= params.tokenCount) return;

    uint srcBase = (token * params.numKVHeads + kvHead) * params.headDim;
    uint dstToken = params.destinationStartToken + token;
    uint dstBase = (dstToken * params.numKVHeads + kvHead) * params.headDim + lane * 4;

    destination[dstBase + 0] = source[srcBase + lane];
    destination[dstBase + 1] = source[srcBase + lane + 32];
    destination[dstBase + 2] = source[srcBase + lane + 64];
    destination[dstBase + 3] = source[srcBase + lane + 96];
}

kernel void pack_kv_decode_cache_pair_f16(
    device const half *keySource [[buffer(0)]],
    device const half *valueSource [[buffer(1)]],
    device half *keyDestination [[buffer(2)]],
    device half *valueDestination [[buffer(3)]],
    constant ERPackKVDecodeCacheParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint lane = gid.x;
    uint kvHead = gid.y;
    uint token = gid.z;
    if (lane >= 32 || kvHead >= params.numKVHeads || token >= params.tokenCount) return;

    uint srcBase = (token * params.numKVHeads + kvHead) * params.headDim;
    uint dstToken = params.destinationStartToken + token;
    uint dstBase = (dstToken * params.numKVHeads + kvHead) * params.headDim + lane * 4;

    keyDestination[dstBase + 0] = keySource[srcBase + lane];
    keyDestination[dstBase + 1] = keySource[srcBase + lane + 32];
    keyDestination[dstBase + 2] = keySource[srcBase + lane + 64];
    keyDestination[dstBase + 3] = keySource[srcBase + lane + 96];

    valueDestination[dstBase + 0] = valueSource[srcBase + lane];
    valueDestination[dstBase + 1] = valueSource[srcBase + lane + 32];
    valueDestination[dstBase + 2] = valueSource[srcBase + lane + 64];
    valueDestination[dstBase + 3] = valueSource[srcBase + lane + 96];
}

struct ERPackedDecodeGQAParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint kvSeqLen;
    float scale;
};

struct ERPackedDecodeSplitKVParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint kvSeqLen;
    uint kvBlockSize;
    uint blockCount;
    float scale;
};

kernel void gqa_decode_attention_packed_f16kv(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERPackedDecodeGQAParams &params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    if (lane >= 32 || headIndex >= params.numHeads) return;

    uint kvHeadIndex = headIndex / (params.numHeads / params.numKVHeads);
    uint qBase = headIndex * params.headDim;

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = 0; kv < params.kvSeqLen; ++kv) {
        uint kvBase = (kv * params.numKVHeads + kvHeadIndex) * params.headDim + lane * 4;
        half4 packedK = *reinterpret_cast<const device half4 *>(K + kvBase);
        float partial = q0 * float(packedK[0]) +
            q1 * float(packedK[1]) +
            q2 * float(packedK[2]) +
            q3 * float(packedK[3]);
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        half4 packedV = *reinterpret_cast<const device half4 *>(V + kvBase);
        acc0 = acc0 * correction + prob * float(packedV[0]);
        acc1 = acc1 * correction + prob * float(packedV[1]);
        acc2 = acc2 * correction + prob * float(packedV[2]);
        acc3 = acc3 * correction + prob * float(packedV[3]);
    }

    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}

kernel void gqa_decode_attention_packed_f16kv_partial(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *partialMax [[buffer(3)]],
    device float *partialSum [[buffer(4)]],
    device float *partialAcc [[buffer(5)]],
    constant ERPackedDecodeSplitKVParams &params [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    uint blockIndex = tid.z;
    if (lane >= 32 || headIndex >= params.numHeads || blockIndex >= params.blockCount) return;

    uint groupSize = params.numHeads / params.numKVHeads;
    uint kvHeadIndex = headIndex / groupSize;
    uint qBase = headIndex * params.headDim;
    uint blockStart = blockIndex * params.kvBlockSize;
    uint blockEnd = min(blockStart + params.kvBlockSize, params.kvSeqLen);

    float q0 = Q[qBase + lane];
    float q1 = Q[qBase + lane + 32];
    float q2 = Q[qBase + lane + 64];
    float q3 = Q[qBase + lane + 96];

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint kv = blockStart; kv < blockEnd; ++kv) {
        uint kvBase = (kv * params.numKVHeads + kvHeadIndex) * params.headDim + lane * 4;
        half4 packedK = *reinterpret_cast<const device half4 *>(K + kvBase);
        float partial = q0 * float(packedK[0]) +
            q1 * float(packedK[1]) +
            q2 * float(packedK[2]) +
            q3 * float(packedK[3]);
        float score = simd_sum(partial) * params.scale;

        float nextRunMax = runMax;
        float nextRunSum = runSum;
        float correction = 1.0f;
        float prob = 0.0f;
        if (lane == 0) {
            float oldMax = runMax;
            nextRunMax = max(runMax, score);
            correction = exp(oldMax - nextRunMax);
            prob = exp(score - nextRunMax);
            nextRunSum = runSum * correction + prob;
        }
        runMax = simd_broadcast_first(nextRunMax);
        runSum = simd_broadcast_first(nextRunSum);
        correction = simd_broadcast_first(correction);
        prob = simd_broadcast_first(prob);

        half4 packedV = *reinterpret_cast<const device half4 *>(V + kvBase);
        acc0 = acc0 * correction + prob * float(packedV[0]);
        acc1 = acc1 * correction + prob * float(packedV[1]);
        acc2 = acc2 * correction + prob * float(packedV[2]);
        acc3 = acc3 * correction + prob * float(packedV[3]);
    }

    uint partialIndex = blockIndex * params.numHeads + headIndex;
    uint partialBase = partialIndex * params.headDim;
    partialAcc[partialBase + lane] = acc0;
    partialAcc[partialBase + lane + 32] = acc1;
    partialAcc[partialBase + lane + 64] = acc2;
    partialAcc[partialBase + lane + 96] = acc3;
    if (lane == 0) {
        partialMax[partialIndex] = runMax;
        partialSum[partialIndex] = runSum;
    }
}

kernel void gqa_decode_attention_packed_f16kv_reduce(
    device const float *partialMax [[buffer(0)]],
    device const float *partialSum [[buffer(1)]],
    device const float *partialAcc [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERPackedDecodeSplitKVParams &params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint lane = tid.x;
    uint headIndex = tid.y;
    if (lane >= 32 || headIndex >= params.numHeads) return;

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (uint blockIndex = 0; blockIndex < params.blockCount; ++blockIndex) {
        uint partialIndex = blockIndex * params.numHeads + headIndex;
        uint partialBase = partialIndex * params.headDim;
        float blockMax = partialMax[partialIndex];
        float blockSum = partialSum[partialIndex];

        float nextRunMax = max(runMax, blockMax);
        float correction = exp(runMax - nextRunMax);
        float blockCorrection = exp(blockMax - nextRunMax);

        acc0 = acc0 * correction + partialAcc[partialBase + lane] * blockCorrection;
        acc1 = acc1 * correction + partialAcc[partialBase + lane + 32] * blockCorrection;
        acc2 = acc2 * correction + partialAcc[partialBase + lane + 64] * blockCorrection;
        acc3 = acc3 * correction + partialAcc[partialBase + lane + 96] * blockCorrection;
        runSum = runSum * correction + blockSum * blockCorrection;
        runMax = nextRunMax;
    }

    uint qBase = headIndex * params.headDim;
    float invSum = runSum > 0.0f ? 1.0f / runSum : 0.0f;
    O[qBase + lane] = acc0 * invSum;
    O[qBase + lane + 32] = acc1 * invSum;
    O[qBase + lane + 64] = acc2 * invSum;
    O[qBase + lane + 96] = acc3 * invSum;
}


// --- GeGLU.metal ---
#include <metal_stdlib>
using namespace metal;

struct GeGLUParams {
    uint count;
};

// Fused GeGLU with PyTorch tanh-approx GELU for Gemma 4 (E4B).
// y[i] = gelu_tanh(gate[i]) * up[i]
// gelu_tanh(x) = x * 0.5 * (1 + tanh(c * (x + 0.044715 * x^3))), c = sqrt(2/pi)
kernel void gelu_tanh_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float *out        [[buffer(2)]],
    constant GeGLUParams &p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) {
        return;
    }
    float g = gate[gid];
    float gelu;
    if (g > 10.0f) {
        gelu = g;
    } else if (g < -10.0f) {
        gelu = 0.0f;
    } else {
        const float c = 0.7978845608028654f;
        float inner = c * (g + 0.044715f * g * g * g);
        gelu = g * 0.5f * (1.0f + tanh(inner));
    }
    out[gid] = gelu * up[gid];
}


// --- Gemma4Decode.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERGemma4RMSNormParams {
    uint rows;
    uint cols;
    float eps;
};

struct ERGemma4ResidualRMSNormParams {
    uint count;
    float eps;
};

struct ERGemma4ResidualRMSNormRowsParams {
    uint rows;
    uint cols;
    float eps;
};

struct ERGemma4EmbeddingParams {
    uint rowWidth;
    uint tokenCount;
    uint rowStrideBytes;
    ulong tableByteOffset;
    float scale;
};

struct ERGemma4DecodeGQAParams {
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    uint headDim;
    uint kvStart;
    uint kvCount;
    uint kvCapacity;
    float scale;
};

static inline float gemma4_f16_at(device const uchar *ptr, uint offset) {
    ushort bits = ushort(ptr[offset]) | (ushort(ptr[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

static inline float gemma4_dequant_q6_k_value(device const uchar *block, uint inBlock) {
    const float d = gemma4_f16_at(block, 208);
    const uint halfBlock = inBlock / 128;
    const uint within = inBlock - halfBlock * 128;
    const uint lane = within & 31;
    const uint quarter = within / 32;
    const uint qlBase = halfBlock * 64;
    const uint qhBase = 128 + halfBlock * 32;
    const uint scaleBase = 192 + halfBlock * 8;

    const uchar qlByte = block[qlBase + (quarter & 1) * 32 + lane];
    const uchar lower4 = quarter < 2 ? (qlByte & 0x0F) : (qlByte >> 4);
    const uchar upper2 = (block[qhBase + lane] >> (quarter * 2)) & 0x03;
    const int q6 = int(lower4 | (upper2 << 4)) - 32;
    int scaleRaw = int(block[scaleBase + quarter * 2 + lane / 16]);
    if (scaleRaw >= 128) {
        scaleRaw -= 256;
    }
    return d * float(scaleRaw) * float(q6);
}

kernel void gemma4_rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERGemma4RMSNormParams &params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    constexpr uint threadCount = 256;
    threadgroup float partial[threadCount];
    const uint offset = row * params.cols;

    float sum = 0.0f;
    for (uint col = tid; col < params.cols; col += threadCount) {
        const float value = input[offset + col];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadCount / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(params.cols) + params.eps);
    for (uint col = tid; col < params.cols; col += threadCount) {
        output[offset + col] = input[offset + col] * scale * weight[col];
    }
}

kernel void gemma4_residual_rmsnorm_add_f32(
    device const float *residual [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ERGemma4ResidualRMSNormParams &params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint threadCount = 256;
    threadgroup float partial[threadCount];

    float sum = 0.0f;
    for (uint index = tid; index < params.count; index += threadCount) {
        const float value = input[index];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadCount / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(params.count) + params.eps);
    for (uint index = tid; index < params.count; index += threadCount) {
        output[index] = residual[index] + input[index] * scale * weight[index];
    }
}

kernel void gemma4_residual_rmsnorm_add_rows_f32(
    device const float *residual [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ERGemma4ResidualRMSNormRowsParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (row >= params.rows) {
        return;
    }

    constexpr uint threadCount = 256;
    threadgroup float partial[threadCount];
    const uint offset = row * params.cols;

    float sum = 0.0f;
    for (uint col = tid; col < params.cols; col += threadCount) {
        const float value = input[offset + col];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadCount / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(params.cols) + params.eps);
    for (uint col = tid; col < params.cols; col += threadCount) {
        output[offset + col] = residual[offset + col] + input[offset + col] * scale * weight[col];
    }
}

kernel void gemma4_store_f32_to_f16(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        output[gid] = half(input[gid]);
    }
}

kernel void gemma4_mul_scalar_f32(
    device float *values [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        values[gid] *= scale;
    }
}

kernel void gemma4_gather_token_embedding_q6_k(
    device const uchar *table [[buffer(0)]],
    device const int *tokens [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERGemma4EmbeddingParams &params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint tokenIndex = gid.y;
    if (col >= params.rowWidth || tokenIndex >= params.tokenCount) {
        return;
    }

    const int tokenID = tokens[tokenIndex];
    const uint blockIndex = col / 256;
    const uint inBlock = col - blockIndex * 256;
    device const uchar *block = table
        + params.tableByteOffset
        + ulong(tokenID) * ulong(params.rowStrideBytes)
        + ulong(blockIndex) * 210ul;

    output[tokenIndex * params.rowWidth + col] = gemma4_dequant_q6_k_value(block, inBlock) * params.scale;
}

static inline float gemma4_decode_gqa_score(
    device const float *Q,
    device const half *K,
    constant ERGemma4DecodeGQAParams &params,
    uint head,
    uint kvHead,
    uint physicalPosition
) {
    const uint qBase = head * params.headDim;
    const uint kBase = (physicalPosition * params.numKVHeads + kvHead) * params.headDim;
    float dot = 0.0f;
    for (uint dim = 0; dim < params.headDim; ++dim) {
        dot += Q[qBase + dim] * float(K[kBase + dim]);
    }
    return dot * params.scale;
}

kernel void gemma4_decode_gqa_f16kv_windowed(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGemma4DecodeGQAParams &params [[buffer(4)]],
    uint outputIndex [[thread_position_in_grid]]
) {
    const uint total = params.numHeads * params.headDim;
    if (outputIndex >= total) {
        return;
    }

    const uint dim = outputIndex % params.headDim;
    const uint head = outputIndex / params.headDim;
    const uint kvHead = head / params.groupSize;

    if (params.kvCount == 0 || params.kvCapacity == 0) {
        O[outputIndex] = 0.0f;
        return;
    }

    float maxScore = -INFINITY;
    for (uint kvIndex = 0; kvIndex < params.kvCount; ++kvIndex) {
        const uint physical = (params.kvStart + kvIndex) % params.kvCapacity;
        const float score = gemma4_decode_gqa_score(Q, K, params, head, kvHead, physical);
        maxScore = max(maxScore, score);
    }

    float sum = 0.0f;
    float value = 0.0f;
    for (uint kvIndex = 0; kvIndex < params.kvCount; ++kvIndex) {
        const uint physical = (params.kvStart + kvIndex) % params.kvCapacity;
        const float score = gemma4_decode_gqa_score(Q, K, params, head, kvHead, physical);
        const float weight = exp(score - maxScore);
        const uint vBase = (physical * params.numKVHeads + kvHead) * params.headDim;
        value += weight * float(V[vBase + dim]);
        sum += weight;
    }

    O[outputIndex] = sum > 0.0f ? value / sum : 0.0f;
}

kernel void gemma4_decode_gqa_f16kv_windowed_fast(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant ERGemma4DecodeGQAParams &params [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint maxWindow = 512;
    threadgroup float scores[maxWindow];
    threadgroup float sumShared;

    if (head >= params.numHeads) {
        return;
    }

    const uint kvHead = head / params.groupSize;
    const uint qBase = head * params.headDim;

    if (params.kvCount == 0 || params.kvCapacity == 0) {
        for (uint dim = tid; dim < params.headDim; dim += maxWindow) {
            O[qBase + dim] = 0.0f;
        }
        return;
    }

    if (tid < params.kvCount && tid < maxWindow) {
        const uint physical = (params.kvStart + tid) % params.kvCapacity;
        const uint kBase = (physical * params.numKVHeads + kvHead) * params.headDim;
        float dot = 0.0f;
        for (uint dim = 0; dim < params.headDim; ++dim) {
            dot += Q[qBase + dim] * float(K[kBase + dim]);
        }
        scores[tid] = dot * params.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float maxScore = -INFINITY;
        for (uint kvIndex = 0; kvIndex < params.kvCount && kvIndex < maxWindow; ++kvIndex) {
            maxScore = max(maxScore, scores[kvIndex]);
        }
        float sum = 0.0f;
        for (uint kvIndex = 0; kvIndex < params.kvCount && kvIndex < maxWindow; ++kvIndex) {
            const float weight = exp(scores[kvIndex] - maxScore);
            scores[kvIndex] = weight;
            sum += weight;
        }
        sumShared = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float normalizer = sumShared > 0.0f ? 1.0f / sumShared : 0.0f;
    for (uint dim = tid; dim < params.headDim; dim += maxWindow) {
        float value = 0.0f;
        for (uint kvIndex = 0; kvIndex < params.kvCount && kvIndex < maxWindow; ++kvIndex) {
            const uint physical = (params.kvStart + kvIndex) % params.kvCapacity;
            const uint vBase = (physical * params.numKVHeads + kvHead) * params.headDim;
            value += scores[kvIndex] * float(V[vBase + dim]);
        }
        O[qBase + dim] = value * normalizer;
    }
}


// --- LayerNorm.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERLayerNormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void layernorm_f32(
    device const float *input [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ERLayerNormParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= params.rows) {
        return;
    }

    uint offset = row * params.cols;
    float mean = 0.0f;
    for (uint col = 0; col < params.cols; col++) {
        mean += input[offset + col];
    }
    mean /= float(params.cols);

    float variance = 0.0f;
    for (uint col = 0; col < params.cols; col++) {
        float delta = input[offset + col] - mean;
        variance += delta * delta;
    }
    variance /= float(params.cols);
    float invStd = rsqrt(variance + params.eps);

    for (uint col = 0; col < params.cols; col++) {
        output[offset + col] = (input[offset + col] - mean) * invStd * gamma[col] + beta[col];
    }
}


// --- LogitSoftcap.metal ---
#include <metal_stdlib>
using namespace metal;

kernel void logit_softcap_f32(
    device float *logits   [[buffer(0)]],
    constant float &cap    [[buffer(1)]],
    constant uint &count   [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = logits[gid];
    logits[gid] = tanh(x / cap) * cap;
}


// --- PLE.metal ---
#include <metal_stdlib>
using namespace metal;

constant uint pleQ8BlockBytes = 34;
constant uint pleQ8WeightsPerBlock = 32;
constant uint pleQ6KBlockBytes = 210;
constant uint pleQ6KWeightsPerBlock = 256;

static inline float ple_dequant_q6_k_value(device const uchar *block, uint inBlock) {
    device const half *dPtr = reinterpret_cast<device const half *>(block + 208);
    const float d = float(dPtr[0]);
    const uint halfBlock = inBlock / 128;
    const uint within = inBlock - halfBlock * 128;
    const uint lane = within & 31;
    const uint quarter = within / 32;
    const uint qlBase = halfBlock * 64;
    const uint qhBase = 128 + halfBlock * 32;
    const uint scaleBase = 192 + halfBlock * 8;

    const uchar qlByte = block[qlBase + (quarter & 1) * 32 + lane];
    const uchar lower4 = quarter < 2 ? (qlByte & 0x0F) : (qlByte >> 4);
    const uchar upper2 = (block[qhBase + lane] >> (quarter * 2)) & 0x03;
    const int q6 = int(lower4 | (upper2 << 4)) - 32;
    const char scale = as_type<char>(block[scaleBase + quarter * 2 + lane / 16]);
    return d * float(scale) * float(q6);
}

// === PLE (Per-Layer Embedding) single-row Q8_0 gather kernel ===
//
// For Gemma 4 E4B: per_layer_token_embd has shape [vocab_size, num_layers * perLayerDim]
// stored as Q8_0 blocks (34 bytes per 32-element block). For each token in the batch and
// each layer, we gather that layer's slice from the token's row, dequantize, and scale by
// sqrt(perLayerDim).
//
// Output: [numTokens, num_layers, perLayerDim] as Float.

struct PLEGatherParams {
    uint perLayerDim;      // P
    uint numLayers;        // L
    uint numTokens;
    uint rowStrideBytes;   // bytes per (token, L*P row) in Q8_0 storage
    ulong tableByteOffset;
};

kernel void ple_gather_q8_0(
    device const uchar *q8Table        [[buffer(0)]],
    device const int *tokens           [[buffer(1)]],
    device float *out                  [[buffer(2)]],
    constant PLEGatherParams &params   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tIdx = gid.y;
    uint elem = gid.x;
    uint totalElems = params.numLayers * params.perLayerDim;
    if (tIdx >= params.numTokens || elem >= totalElems) return;

    int tokenId = tokens[tIdx];
    ulong rowBase = params.tableByteOffset + ulong(uint(tokenId)) * ulong(params.rowStrideBytes);
    uint blockIndex = elem / pleQ8WeightsPerBlock;
    uint inBlock = elem % pleQ8WeightsPerBlock;
    device const uchar *blockPtr = q8Table + rowBase + blockIndex * pleQ8BlockBytes;

    float scale = float(as_type<half>(*(device const ushort*)blockPtr));
    int8_t q = as_type<char>(blockPtr[2 + inBlock]);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = scale * float(q) * sqrtP;
}

kernel void ple_gather_q6_k(
    device const uchar *q6KTable       [[buffer(0)]],
    device const int *tokens           [[buffer(1)]],
    device float *out                  [[buffer(2)]],
    constant PLEGatherParams &params   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tIdx = gid.y;
    uint elem = gid.x;
    uint totalElems = params.numLayers * params.perLayerDim;
    if (tIdx >= params.numTokens || elem >= totalElems) return;

    int tokenId = tokens[tIdx];
    ulong rowBase = params.tableByteOffset + ulong(uint(tokenId)) * ulong(params.rowStrideBytes);
    uint blockIndex = elem / pleQ6KWeightsPerBlock;
    uint inBlock = elem % pleQ6KWeightsPerBlock;
    device const uchar *block = q6KTable + rowBase + ulong(blockIndex) * ulong(pleQ6KBlockBytes);

    const float sqrtP = sqrt(float(params.perLayerDim));
    out[tIdx * totalElems + elem] = ple_dequant_q6_k_value(block, inBlock) * sqrtP;
}

kernel void ple_gather_q6_k_blocked(
    device const uchar *q6KTable       [[buffer(0)]],
    device const int *tokens           [[buffer(1)]],
    device float *out                  [[buffer(2)]],
    constant PLEGatherParams &params   [[buffer(3)]],
    uint3 blockPos [[threadgroup_position_in_grid]],
    uint3 localPos [[thread_position_in_threadgroup]]
) {
    uint tIdx = blockPos.y;
    uint blockIndex = blockPos.x;
    uint lane = localPos.x;
    uint totalElems = params.numLayers * params.perLayerDim;
    uint blocksPerRow = totalElems / pleQ6KWeightsPerBlock;
    if (tIdx >= params.numTokens || blockIndex >= blocksPerRow || lane >= pleQ6KWeightsPerBlock) return;

    int tokenId = tokens[tIdx];
    ulong rowBase = params.tableByteOffset + ulong(uint(tokenId)) * ulong(params.rowStrideBytes);
    device const uchar *block = q6KTable + rowBase + ulong(blockIndex) * ulong(pleQ6KBlockBytes);

    const float sqrtP = sqrt(float(params.perLayerDim));
    uint elem = blockIndex * pleQ6KWeightsPerBlock + lane;
    out[tIdx * totalElems + elem] = ple_dequant_q6_k_value(block, lane) * sqrtP;
}

// === PLE (Per-Layer Embedding) inputs builder kernel ===
//
// Combines the projected hidden state (RMSNorm-normalized) with the gathered
// PLE rows and mixes via scaleMix (typically 1/sqrt(2)). Per (batchSeq, layer)
// slice computes RMSNorm along the last dim (P) using Gemma 4's direct
// affine weight, then adds pleRows and multiplies by scaleMix.
//
// Inputs:
//   proj       [B*S, L*P]  — output of GEMV(Wproj, h), already scaled by 1/sqrt(H)
//   normW      [P]         — per_layer_proj_norm.weight
//   pleRows    [B*S, L, P] — output of ple_gather_q8_0 (already scaled by sqrt(P))
// Output:
//   out        [B*S, L, P] — per_layer_inputs

struct PLEInputsParams {
    uint hidden;       // H (not consumed by kernel; proj is pre-scaled)
    uint perLayerDim;  // P
    uint numLayers;    // L
    uint batchSeq;     // B*S
    float rmsEps;      // typically 1e-6
    float scaleMix;    // 1/sqrt(2)
};

kernel void ple_inputs_build(
    device const float *proj        [[buffer(0)]],
    device const float *normW       [[buffer(1)]],
    device const float *pleRows     [[buffer(2)]],
    device float *out               [[buffer(3)]],
    constant PLEInputsParams &p     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.y indexes (batchSeq * numLayers) combined; gid.x indexes perLayerDim
    uint layerIdx = gid.y % p.numLayers;
    uint batchSeq = gid.y / p.numLayers;
    uint pIdx = gid.x;
    if (batchSeq >= p.batchSeq || pIdx >= p.perLayerDim) return;

    uint sliceBase = batchSeq * p.numLayers * p.perLayerDim + layerIdx * p.perLayerDim;

    // Per-thread RMS reduction over P elements (naive — production version can use simdgroup reduce)
    float sumSq = 0;
    for (uint i = 0; i < p.perLayerDim; ++i) {
        float v = proj[sliceBase + i];
        sumSq += v * v;
    }
    float rms = sqrt(sumSq / float(p.perLayerDim) + p.rmsEps);

    float v = proj[sliceBase + pIdx];
    float w = normW[pIdx];
    float normed = (v / rms) * w;
    float ple = pleRows[sliceBase + pIdx];
    out[sliceBase + pIdx] = (normed + ple) * p.scaleMix;
}

// === PLE side-channel finalize kernel ===
//
// Finalizes a single decoder layer's side-channel projection:
//   h = h + RMSNorm(proj, postNormW)
// using Gemma 4's direct RMSNorm weight convention.
//
// Inputs:
//   proj       [B*S, H] — output of side-channel down projection
//   postNormW  [H]      — post_per_layer_input_norm.weight
// In/out:
//   h          [B*S, H] — residual stream, updated in place

struct PLESideChannelParams {
    uint hidden;
    uint batchSeq;
    float rmsEps;
};

struct PLEGateParams {
    uint count;
};

kernel void ple_gate_gelu_mul_f32(
    device const float *gate          [[buffer(0)]],
    device const float *ple           [[buffer(1)]],
    device float *out                 [[buffer(2)]],
    constant PLEGateParams &p         [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float g = gate[gid];
    float gelu;
    if (g > 10.0f) {
        gelu = g;
    } else if (g < -10.0f) {
        gelu = 0.0f;
    } else {
        float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        gelu = g * 0.5f * (1.0f + tanh(inner));
    }
    out[gid] = gelu * ple[gid];
}

kernel void ple_side_channel_finalize(
    device float *h                         [[buffer(0)]],
    device const float *proj                [[buffer(1)]],
    device const float *postNormW           [[buffer(2)]],
    constant PLESideChannelParams &p        [[buffer(3)]],
    uint batch [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    if (batch >= p.batchSeq) return;

    uint base = batch * p.hidden;
    threadgroup float partials[256];
    float sumSq = 0.0f;
    for (uint i = local_id; i < p.hidden; i += 256) {
        float v = proj[base + i];
        sumSq += v * v;
    }
    partials[local_id] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            partials[local_id] += partials[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(partials[0] / float(p.hidden) + p.rmsEps);
    for (uint hIdx = local_id; hIdx < p.hidden; hIdx += 256) {
        float normed = proj[base + hIdx] * scale * postNormW[hIdx];
        h[base + hIdx] += normed;
    }
}


// --- RMSNorm.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERRMSNormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERRMSNormParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= params.rows) {
        return;
    }

    uint offset = row * params.cols;
    float meanSq = 0.0f;
    for (uint col = 0; col < params.cols; col++) {
        float value = input[offset + col];
        meanSq += value * value;
    }
    meanSq /= float(params.cols);
    float scale = rsqrt(meanSq + params.eps);

    for (uint col = 0; col < params.cols; col++) {
        output[offset + col] = input[offset + col] * scale * weight[col];
    }
}

/// Parallel RMSNorm for single-row decode (rows=1).
/// Uses 256 threads (8 simdgroups) to cooperatively process cols elements.
/// Much faster than rmsnorm_f32 which dispatches 1 thread for rows=1.
kernel void rmsnorm_parallel_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ERRMSNormParams &params [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid;
    if (row >= params.rows) return;

    const uint cols = params.cols;
    const uint offset = row * cols;
    const uint tid = sgitg * 32 + tiisg;
    const uint stride = 256;  // 8 simdgroups × 32 threads

    // Phase 1: Parallel sum-of-squares reduction
    float localSumSq = 0.0f;
    for (uint col = tid; col < cols; col += stride) {
        float v = input[offset + col];
        localSumSq += v * v;
    }

    // Intra-simdgroup reduction
    localSumSq = simd_sum(localSumSq);

    // Cross-simdgroup reduction via threadgroup memory
    threadgroup float tg_partial[8];
    if (tiisg == 0) {
        tg_partial[sgitg] = localSumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First thread computes final scale
    if (sgitg == 0 && tiisg == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; i++) total += tg_partial[i];
        tg_partial[0] = rsqrt(total / float(cols) + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = tg_partial[0];

    // Phase 2: Parallel scale + weight multiply
    for (uint col = tid; col < cols; col += stride) {
        output[offset + col] = input[offset + col] * scale * weight[col];
    }
}


// --- Reduction.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERReductionParams {
    uint elementCount;
    uint reductionSize;
    uint outerSize;
};

kernel void reduce_sum_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;
    float sum = 0.0;
    uint base = tid * params.reductionSize;
    for (uint i = 0; i < params.reductionSize; i++) {
        sum += input[base + i];
    }
    output[tid] = sum;
}

kernel void reduce_mean_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;
    float sum = 0.0;
    uint base = tid * params.reductionSize;
    for (uint i = 0; i < params.reductionSize; i++) {
        sum += input[base + i];
    }
    output[tid] = sum / float(params.reductionSize);
}

kernel void reduce_max_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERReductionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.outerSize) return;
    uint base = tid * params.reductionSize;
    float maxVal = input[base];
    for (uint i = 1; i < params.reductionSize; i++) {
        maxVal = max(maxVal, input[base + i]);
    }
    output[tid] = maxVal;
}


// --- RoPE.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERRoPEParams {
    uint seqLen;
    uint numHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    // Fraction of head_dim to rotate (pRoPE). 1.0 = rotate all channels (standard RoPE).
    // <1.0 = only rotate channels with 2*pair < headDim * partialRotaryFactor; rest pass through.
    float partialRotaryFactor;
};

kernel void rope_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERRoPEParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint head = tid.y;
    uint seq = tid.z;
    uint halfDim = params.headDim / 2;

    if (dimPair >= halfDim || head >= params.numHeads || seq >= params.seqLen) {
        return;
    }

    uint baseIndex = (seq * params.numHeads * params.headDim) + (head * params.headDim) + (2 * dimPair);

    // pRoPE pass-through: channels beyond the partial boundary are copied verbatim.
    // rotatedPairs = floor(halfDim * partialRotaryFactor). For partial=1.0 this == halfDim,
    // so the guard never fires (full rotation, identical to standard RoPE).
    uint rotatedPairs = uint(float(halfDim) * params.partialRotaryFactor);
    if (dimPair >= rotatedPairs) {
        output[baseIndex] = input[baseIndex];
        output[baseIndex + 1] = input[baseIndex + 1];
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    float x0 = input[baseIndex];
    float x1 = input[baseIndex + 1];
    output[baseIndex] = x0 * cosValue - x1 * sinValue;
    output[baseIndex + 1] = x0 * sinValue + x1 * cosValue;
}

/// NeoX-style RoPE: pairs (d, d+halfDim) instead of (2d, 2d+1).
/// Used by Qwen, GPT-NeoX, StableLM, and other models with split-halves layout.
kernel void rope_neox_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERRoPEParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint head = tid.y;
    uint seq = tid.z;
    uint halfDim = params.headDim / 2;

    if (dimPair >= halfDim || head >= params.numHeads || seq >= params.seqLen) {
        return;
    }

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);

    // pRoPE pass-through (NeoX layout pairs (d, d+halfDim)). For partial=1.0, rotatedPairs == halfDim
    // and this guard never fires — standard NeoX RoPE behavior preserved.
    uint rotatedPairs = uint(float(halfDim) * params.partialRotaryFactor);
    if (dimPair >= rotatedPairs) {
        output[headBase + dimPair]           = input[headBase + dimPair];
        output[headBase + dimPair + halfDim] = input[headBase + dimPair + halfDim];
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    float x0 = input[headBase + dimPair];
    float x1 = input[headBase + dimPair + halfDim];
    output[headBase + dimPair]           = x0 * cosValue - x1 * sinValue;
    output[headBase + dimPair + halfDim] = x0 * sinValue + x1 * cosValue;
}

/// Fused Q/K per-head norm + NeoX RoPE in a SINGLE dispatch.
/// Replaces 4 dispatches per layer (Q norm + K norm + RoPE Q + RoPE K→f16) with 1.
/// Thread grid: (halfDim, numHeads+numKVHeads). Threads 0..<numHeads do Q, rest do K.
struct ERFusedNormRoPEParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    float rmsEps;
};

struct ERFusedNormRoPEPrefillParams {
    uint seqLen;
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    float rmsEps;
};

kernel void fused_qk_norm_rope_neox(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *qNormW [[buffer(2)]],
    device const float *kNormW [[buffer(3)]],
    device float *outQ [[buffer(4)]],
    device half  *outK [[buffer(5)]],
    constant ERFusedNormRoPEParams &p [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint headIdx = tid.y;
    uint halfDim = p.headDim / 2;
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimPair >= halfDim || headIdx >= totalHeads) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    device const float* src = isQ ? Q : K;
    device const float* nw = isQ ? qNormW : kNormW;
    uint hb = head * p.headDim;

    float raw0 = src[hb + dimPair];
    float raw1 = src[hb + dimPair + halfDim];

    // Per-head RMSNorm: sum of squares across headDim elements
    float pairSq = raw0 * raw0 + raw1 * raw1;
    float sumSq = simd_sum(pairSq);
    // halfDim=64 means 2 simdgroups per head. Need cross-SG reduction.
    threadgroup float tgSq[48]; // max totalHeads=24, 2 SG each
    uint sgIdx = dimPair / 32;
    if (dimPair % 32 == 0) tgSq[headIdx * 2 + sgIdx] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumSq = tgSq[headIdx * 2] + tgSq[headIdx * 2 + 1];

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);
    float x0 = raw0 * rs * nw[dimPair];
    float x1 = raw1 * rs * nw[dimPair + halfDim];

    // RoPE
    float exp = float(2 * dimPair) / float(p.headDim);
    float freq = 1.0f / pow(p.theta, exp);
    float angle = float(p.startPos) * (freq / p.scalingFactor);
    float c = cos(angle), s = sin(angle);
    float o0 = x0 * c - x1 * s;
    float o1 = x0 * s + x1 * c;

    if (isQ) {
        outQ[hb + dimPair] = o0;
        outQ[hb + dimPair + halfDim] = o1;
    } else {
        outK[hb + dimPair] = half(o0);
        outK[hb + dimPair + halfDim] = half(o1);
    }
}

kernel void fused_qk_norm_rope_neox_prefill_f16in(
    device const half *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const float *qNormW [[buffer(2)]],
    device const float *kNormW [[buffer(3)]],
    device float *outQ [[buffer(4)]],
    device half *outK [[buffer(5)]],
    constant ERFusedNormRoPEPrefillParams &p [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgTid [[thread_position_in_threadgroup]]
) {
    uint dimPair = tid.x;
    uint headIdx = tid.y;
    uint seq = tid.z;
    uint halfDim = p.headDim / 2;
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimPair >= halfDim || headIdx >= totalHeads || seq >= p.seqLen) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    uint inputHeadCount = isQ ? p.numHeads : p.numKVHeads;
    uint base = (seq * inputHeadCount + head) * p.headDim;
    device const half *src = isQ ? Q : K;
    device const float *nw = isQ ? qNormW : kNormW;

    float raw0 = float(src[base + dimPair]);
    float raw1 = float(src[base + dimPair + halfDim]);

    float pairSq = raw0 * raw0 + raw1 * raw1;
    float sumSq = simd_sum(pairSq);
    threadgroup float tgSq[2];
    uint sgIdx = tgTid.x / 32;
    if ((tgTid.x % 32) == 0) {
        tgSq[sgIdx] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumSq = tgSq[0] + tgSq[1];

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);
    float x0 = raw0 * rs * nw[dimPair];
    float x1 = raw1 * rs * nw[dimPair + halfDim];

    float exp = float(2 * dimPair) / float(p.headDim);
    float freq = 1.0f / pow(p.theta, exp);
    float angle = float(seq + p.startPos) * (freq / p.scalingFactor);
    float c = cos(angle);
    float s = sin(angle);
    float o0 = x0 * c - x1 * s;
    float o1 = x0 * s + x1 * c;

    if (isQ) {
        uint outBase = (seq * p.numHeads + head) * p.headDim;
        outQ[outBase + dimPair] = o0;
        outQ[outBase + dimPair + halfDim] = o1;
    } else {
        uint outBase = (seq * p.numKVHeads + head) * p.headDim;
        outK[outBase + dimPair] = half(o0);
        outK[outBase + dimPair + halfDim] = half(o1);
    }
}

kernel void fused_qk_norm_rope_neox_prefill_f16in_kpacked(
    device const half *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const float *qNormW [[buffer(2)]],
    device const float *kNormW [[buffer(3)]],
    device float *outQ [[buffer(4)]],
    device half *outK [[buffer(5)]],
    constant ERFusedNormRoPEPrefillParams &p [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgTid [[thread_position_in_threadgroup]]
) {
    uint dimPair = tid.x;
    uint headIdx = tid.y;
    uint seq = tid.z;
    uint halfDim = p.headDim / 2;
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (dimPair >= halfDim || headIdx >= totalHeads || seq >= p.seqLen) return;

    bool isQ = headIdx < p.numHeads;
    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    uint inputHeadCount = isQ ? p.numHeads : p.numKVHeads;
    uint base = (seq * inputHeadCount + head) * p.headDim;
    device const half *src = isQ ? Q : K;
    device const float *nw = isQ ? qNormW : kNormW;

    float raw0 = float(src[base + dimPair]);
    float raw1 = float(src[base + dimPair + halfDim]);

    float pairSq = raw0 * raw0 + raw1 * raw1;
    float sumSq = simd_sum(pairSq);
    threadgroup float tgSq[2];
    uint sgIdx = tgTid.x / 32;
    if ((tgTid.x % 32) == 0) {
        tgSq[sgIdx] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumSq = tgSq[0] + tgSq[1];

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);
    float x0 = raw0 * rs * nw[dimPair];
    float x1 = raw1 * rs * nw[dimPair + halfDim];

    float exp = float(2 * dimPair) / float(p.headDim);
    float freq = 1.0f / pow(p.theta, exp);
    float angle = float(seq + p.startPos) * (freq / p.scalingFactor);
    float c = cos(angle);
    float s = sin(angle);
    float o0 = x0 * c - x1 * s;
    float o1 = x0 * s + x1 * c;

    if (isQ) {
        uint outBase = (seq * p.numHeads + head) * p.headDim;
        outQ[outBase + dimPair] = o0;
        outQ[outBase + dimPair + halfDim] = o1;
    } else {
        uint lane = dimPair % 32;
        uint outBase = (seq * p.numKVHeads + head) * p.headDim + lane * 4;
        if (dimPair < 32) {
            outK[outBase + 0] = half(o0);
            outK[outBase + 2] = half(o1);
        } else {
            outK[outBase + 1] = half(o0);
            outK[outBase + 3] = half(o1);
        }
    }
}

/// Fused Q/K norm + RoPE + GQA in a SINGLE dispatch.
/// Replaces norm+RoPE + GQA = 2 dispatches per layer with 1.
/// Phase 1: Q/K heads compute norm + RoPE
/// Phase 2: Q heads cooperatively compute attention against KV cache
///
/// ARCHITECTURE: 32 threads (1 simdgroup) per head. Each thread processes
/// 4 elements: positions [i, i+32, i+64, i+96] of the 128-dim head vector.
/// This eliminates ALL threadgroup_barriers — pure simd_sum reductions only.
/// Thread grid: (32, numHeads+numKVHeads)
struct ERFusedNormRoPEGQAParams {
    uint numHeads;
    uint numKVHeads;
    uint headDim;
    uint startPos;
    float theta;
    float scalingFactor;
    float rmsEps;
    uint kvSeqLen;       // total K/V cache positions
    float attnScale;     // 1/sqrt(headDim)
};

kernel void fused_qk_norm_rope_gqa(
    device const float *Q [[buffer(0)]],          // raw Q from GEMV [numHeads*headDim]
    device const float *K [[buffer(1)]],          // raw K from GEMV [numKVHeads*headDim]
    device const float *qNormW [[buffer(2)]],     // Q norm weight [headDim]
    device const float *kNormW [[buffer(3)]],     // K norm weight [headDim]
    device float *outAttn [[buffer(4)]],          // attention output [numHeads*headDim]
    device half  *kCache [[buffer(5)]],           // K cache [kvSeqLen, numKVHeads, headDim]
    device half  *vCache [[buffer(6)]],           // V cache [kvSeqLen, numKVHeads, headDim]
    constant ERFusedNormRoPEGQAParams &p [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgIdx [[simdgroup_index_in_threadgroup]]
) {
    // 64 threads per TG (2 simdgroups). Each SG has 32 threads covering all 128 head dims.
    // K heads: SG 1 exits immediately (only SG 0 does K work).
    // Q heads: both SGs run Phase 1 identically, then split KV positions in Phase 2.
    uint dimIdx = tiisg;           // 0..31 (same dimension mapping in both SGs)
    uint headIdx = tid.y;
    uint halfDim = p.headDim / 2;  // 64
    uint totalHeads = p.numHeads + p.numKVHeads;
    if (headIdx >= totalHeads) return;

    bool isQ = headIdx < p.numHeads;

    // K heads only need 1 simdgroup — SG 1 exits immediately
    if (!isQ && sgIdx == 1) return;

    uint head = isQ ? headIdx : (headIdx - p.numHeads);
    device const float* src = isQ ? Q : K;
    device const float* nw = isQ ? qNormW : kNormW;
    uint hb = head * p.headDim;

    // === Phase 1: Per-head RMSNorm + RoPE ===
    // Both SGs compute identically (redundant for Q, but avoids a barrier)
    float raw_a0 = src[hb + dimIdx];
    float raw_a1 = src[hb + dimIdx + halfDim];
    float raw_b0 = src[hb + dimIdx + 32];
    float raw_b1 = src[hb + dimIdx + 32 + halfDim];

    float sq = raw_a0 * raw_a0 + raw_a1 * raw_a1 + raw_b0 * raw_b0 + raw_b1 * raw_b1;
    float sumSq = simd_sum(sq);

    float rs = rsqrt(sumSq / float(p.headDim) + p.rmsEps);

    float x_a0 = raw_a0 * rs * nw[dimIdx];
    float x_a1 = raw_a1 * rs * nw[dimIdx + halfDim];
    float x_b0 = raw_b0 * rs * nw[dimIdx + 32];
    float x_b1 = raw_b1 * rs * nw[dimIdx + 32 + halfDim];

    float freq_a = 1.0f / pow(p.theta, float(2 * dimIdx) / float(p.headDim));
    float angle_a = float(p.startPos) * (freq_a / p.scalingFactor);
    float ca = cos(angle_a), sa = sin(angle_a);
    float q_a0 = x_a0 * ca - x_a1 * sa;
    float q_a1 = x_a0 * sa + x_a1 * ca;

    float freq_b = 1.0f / pow(p.theta, float(2 * (dimIdx + 32)) / float(p.headDim));
    float angle_b = float(p.startPos) * (freq_b / p.scalingFactor);
    float cb = cos(angle_b), sb = sin(angle_b);
    float q_b0 = x_b0 * cb - x_b1 * sb;
    float q_b1 = x_b0 * sb + x_b1 * cb;

    // K heads: write 4 values to cache and exit (SG 0 only, SG 1 already returned)
    if (!isQ) {
        uint cacheBase = p.startPos * p.numKVHeads * p.headDim + hb;
        kCache[cacheBase + dimIdx]              = half(q_a0);
        kCache[cacheBase + dimIdx + halfDim]    = half(q_a1);
        kCache[cacheBase + dimIdx + 32]         = half(q_b0);
        kCache[cacheBase + dimIdx + 32 + halfDim] = half(q_b1);
        return;
    }

    // === Phase 2: GQA — 2 simdgroups per Q head, position-split ===
    // SG 0 processes even KV positions, SG 1 processes odd positions.
    // This halves the serial dependency chain, reducing GQA latency.
    uint kvHead = headIdx / (p.numHeads / p.numKVHeads);
    uint kvStride = p.numKVHeads * p.headDim;

    // Re-derive current-position K locally (avoids cross-TG race with K writers)
    uint kvBaseCurrent = kvHead * p.headDim;
    float raw_k_a0 = K[kvBaseCurrent + dimIdx];
    float raw_k_a1 = K[kvBaseCurrent + dimIdx + halfDim];
    float raw_k_b0 = K[kvBaseCurrent + dimIdx + 32];
    float raw_k_b1 = K[kvBaseCurrent + dimIdx + 32 + halfDim];
    float kSq = raw_k_a0 * raw_k_a0 + raw_k_a1 * raw_k_a1 + raw_k_b0 * raw_k_b0 + raw_k_b1 * raw_k_b1;
    float kSumSq = simd_sum(kSq);
    float kRs = rsqrt(kSumSq / float(p.headDim) + p.rmsEps);
    float k_a0 = raw_k_a0 * kRs * kNormW[dimIdx];
    float k_a1 = raw_k_a1 * kRs * kNormW[dimIdx + halfDim];
    float k_b0 = raw_k_b0 * kRs * kNormW[dimIdx + 32];
    float k_b1 = raw_k_b1 * kRs * kNormW[dimIdx + 32 + halfDim];
    float current_k_a0 = k_a0 * ca - k_a1 * sa;
    float current_k_a1 = k_a0 * sa + k_a1 * ca;
    float current_k_b0 = k_b0 * cb - k_b1 * sb;
    float current_k_b1 = k_b0 * sb + k_b1 * cb;

    float runMax = -INFINITY;
    float runSum = 0.0f;
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_b0 = 0.0f, acc_b1 = 0.0f;

    // Position-split loop: SG 0 handles even positions, SG 1 handles odd
    for (uint kvPair = 0; kvPair < (p.kvSeqLen + 1) / 2; kvPair++) {
        uint kv = kvPair * 2 + sgIdx;
        if (kv >= p.kvSeqLen) break;

        uint kvBase = kv * kvStride + kvHead * p.headDim;

        float dk_a0, dk_a1, dk_b0, dk_b1;
        if (kv == p.startPos) {
            dk_a0 = current_k_a0; dk_a1 = current_k_a1;
            dk_b0 = current_k_b0; dk_b1 = current_k_b1;
        } else {
            dk_a0 = float(kCache[kvBase + dimIdx]);
            dk_a1 = float(kCache[kvBase + dimIdx + halfDim]);
            dk_b0 = float(kCache[kvBase + dimIdx + 32]);
            dk_b1 = float(kCache[kvBase + dimIdx + 32 + halfDim]);
        }

        float partial = q_a0 * dk_a0 + q_a1 * dk_a1 + q_b0 * dk_b0 + q_b1 * dk_b1;
        float score = simd_sum(partial) * p.attnScale;

        float oldMax = runMax;
        runMax = max(runMax, score);
        float correction = exp(oldMax - runMax);
        float prob = exp(score - runMax);
        runSum = runSum * correction + prob;

        acc_a0 = acc_a0 * correction + prob * float(vCache[kvBase + dimIdx]);
        acc_a1 = acc_a1 * correction + prob * float(vCache[kvBase + dimIdx + halfDim]);
        acc_b0 = acc_b0 * correction + prob * float(vCache[kvBase + dimIdx + 32]);
        acc_b1 = acc_b1 * correction + prob * float(vCache[kvBase + dimIdx + 32 + halfDim]);
    }

    // === Merge: combine SG 0 and SG 1 accumulators via threadgroup memory ===
    threadgroup float tg_acc[128];  // SG 1's per-dim accumulator values
    threadgroup float tg_max1;
    threadgroup float tg_sum1;

    if (sgIdx == 1) {
        tg_acc[dimIdx]              = acc_a0;
        tg_acc[dimIdx + halfDim]    = acc_a1;
        tg_acc[dimIdx + 32]         = acc_b0;
        tg_acc[dimIdx + 32 + halfDim] = acc_b1;
        if (dimIdx == 0) {
            tg_max1 = runMax;
            tg_sum1 = runSum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgIdx == 0) {
        float max1 = tg_max1;
        float sum1 = tg_sum1;

        // Online softmax merge of two independent accumulators
        float maxFinal = max(runMax, max1);
        float c0 = exp(runMax - maxFinal);
        float c1 = exp(max1 - maxFinal);
        float sumFinal = runSum * c0 + sum1 * c1;
        float invSum = sumFinal > 0.0f ? 1.0f / sumFinal : 0.0f;

        float other_a0 = tg_acc[dimIdx];
        float other_a1 = tg_acc[dimIdx + halfDim];
        float other_b0 = tg_acc[dimIdx + 32];
        float other_b1 = tg_acc[dimIdx + 32 + halfDim];

        outAttn[hb + dimIdx]              = (acc_a0 * c0 + other_a0 * c1) * invSum;
        outAttn[hb + dimIdx + halfDim]    = (acc_a1 * c0 + other_a1 * c1) * invSum;
        outAttn[hb + dimIdx + 32]         = (acc_b0 * c0 + other_b0 * c1) * invSum;
        outAttn[hb + dimIdx + 32 + halfDim] = (acc_b1 * c0 + other_b1 * c1) * invSum;
    }
}

/// NeoX RoPE with f16 output — eliminates separate f32→f16 conversion dispatch.
/// Used for K before writing to float16 KV cache.
kernel void rope_neox_f32_to_f16(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant ERRoPEParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint dimPair = tid.x;
    uint head = tid.y;
    uint seq = tid.z;
    uint halfDim = params.headDim / 2;

    if (dimPair >= halfDim || head >= params.numHeads || seq >= params.seqLen) return;

    uint headBase = (seq * params.numHeads * params.headDim) + (head * params.headDim);

    // pRoPE pass-through (NeoX f32->f16 variant). For partial=1.0, rotatedPairs == halfDim.
    uint rotatedPairs = uint(float(halfDim) * params.partialRotaryFactor);
    if (dimPair >= rotatedPairs) {
        output[headBase + dimPair]           = half(input[headBase + dimPair]);
        output[headBase + dimPair + halfDim] = half(input[headBase + dimPair + halfDim]);
        return;
    }

    float exponent = float(2 * dimPair) / float(params.headDim);
    float frequency = 1.0f / powr(params.theta, exponent);
    float angle = float(seq + params.startPos) * (frequency / params.scalingFactor);
    float cosValue = cos(angle);
    float sinValue = sin(angle);

    float x0 = input[headBase + dimPair];
    float x1 = input[headBase + dimPair + halfDim];
    output[headBase + dimPair]           = half(x0 * cosValue - x1 * sinValue);
    output[headBase + dimPair + halfDim] = half(x0 * sinValue + x1 * cosValue);
}


// --- SlidingCausalMask.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERSlidingMaskParams {
    uint seqLen;
    uint window;
};

/// Generates an additive sliding-window causal attention mask as Float.
///
/// For query position `q` attending to key position `k`:
///   - k > q              → -INFINITY (causal constraint; cannot attend to future)
///   - q >= window &&
///     (q - k) >= window  → -INFINITY (outside sliding window)
///   - otherwise          → 0.0       (attention allowed)
///
/// The `q >= p.window` guard is REQUIRED to prevent unsigned underflow when
/// q < window. Without it, `q - k` wraps around and erroneously masks valid
/// positions at the start of the sequence.
///
/// Global (full causal) attention is expressed as `window >= seqLen`:
/// the sliding-window check is impossible (q - k < window always holds),
/// so only the causal constraint applies.
///
/// Output layout: row-major `[seqLen, seqLen]` with index `q * seqLen + k`.
kernel void sliding_causal_mask_f32(
    device float *mask [[buffer(0)]],
    constant ERSlidingMaskParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q = gid.y;
    uint k = gid.x;
    if (q >= params.seqLen || k >= params.seqLen) {
        return;
    }
    bool outOfWindow =
        (k > q) ||
        (q >= params.window && (q - k) >= params.window);
    mask[q * params.seqLen + k] = outOfWindow ? -INFINITY : 0.0f;
}


// --- Softmax.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERSoftmaxParams {
    uint rows;
    uint cols;
};

constant uint SOFTMAX_THREADS = 256;

kernel void softmax_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ERSoftmaxParams &params [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) {
        return;
    }

    device const float *rowIn = input + row * params.cols;
    device float *rowOut = output + row * params.cols;

    float threadMax = -INFINITY;
    for (uint col = local_id; col < params.cols; col += SOFTMAX_THREADS) {
        threadMax = max(threadMax, rowIn[col]);
    }

    threadMax = simd_max(threadMax);

    threadgroup float shared[32];
    if (simd_lane == 0) {
        shared[simd_group] = threadMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint simdGroupCount = (SOFTMAX_THREADS + 31) / 32;
        float value = simd_lane < simdGroupCount ? shared[simd_lane] : -INFINITY;
        value = simd_max(value);
        shared[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowMax = shared[0];

    float threadSum = 0.0f;
    for (uint col = local_id; col < params.cols; col += SOFTMAX_THREADS) {
        float expValue = exp(rowIn[col] - rowMax);
        rowOut[col] = expValue;
        threadSum += expValue;
    }

    threadSum = simd_sum(threadSum);
    if (simd_lane == 0) {
        shared[simd_group] = threadSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint simdGroupCount = (SOFTMAX_THREADS + 31) / 32;
        float value = simd_lane < simdGroupCount ? shared[simd_lane] : 0.0f;
        value = simd_sum(value);
        shared[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowSum = shared[0];
    float invSum = 1.0f / rowSum;

    for (uint col = local_id; col < params.cols; col += SOFTMAX_THREADS) {
        rowOut[col] *= invSum;
    }
}


// --- StitchableOps.metal ---
#include <metal_stdlib>
using namespace metal;

// [[stitchable]] marks these functions as linkable building blocks for
// runtime function stitching (Metal 2.3+ / macOS 12+ / iOS 15+).
// FusionEngine (Task 10) uses them to compose fused pipelines at runtime
// without recompiling shader source.

[[stitchable]] float op_relu_float(float x) { return max(x, 0.0f); }

[[stitchable]] float op_sigmoid_float(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[[stitchable]] float op_gelu_float(float x) {
    const float kSqrt2OverPi = 0.7978845608f;
    float cube = x * x * x;
    float inner = kSqrt2OverPi * (x + 0.044715f * cube);
    return 0.5f * x * (1.0f + tanh(inner));
}

[[stitchable]] float op_silu_float(float x) {
    return x / (1.0f + exp(-x));
}

[[stitchable]] float op_neg_float(float x) { return -x; }

[[stitchable]] float op_abs_float(float x) { return abs(x); }

[[stitchable]] float op_sqrt_float(float x) { return sqrt(x); }

[[stitchable]] float op_exp_float(float x) { return exp(x); }

[[stitchable]] float op_log_float(float x) { return log(x); }

[[stitchable]] float op_tanh_float(float x) { return tanh(x); }

[[stitchable]] float op_add_float(float a, float b) { return a + b; }

[[stitchable]] float op_sub_float(float a, float b) { return a - b; }

[[stitchable]] float op_mul_float(float a, float b) { return a * b; }

[[stitchable]] float op_div_float(float a, float b) { return a / b; }


// --- Transpose.metal ---
#include <metal_stdlib>
using namespace metal;

struct ERTransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ERTransposeParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    if (row >= params.rows || col >= params.cols) return;
    output[col * params.rows + row] = input[row * params.cols + col];
}


// --- TurboQuant.metal ---
#include <metal_stdlib>
using namespace metal;

constant float TURBOQUANT_CODEBOOK_2BIT[4] = {
    -0.133462, -0.039994, 0.039994, 0.133462
};
constant float TURBOQUANT_THRESHOLDS_2BIT[3] = {
    -0.086728, 0.0, 0.086728
};

constant float TURBOQUANT_CODEBOOK_3BIT[8] = {
    -0.190685, -0.117832, -0.065717, -0.021460,
    0.021460, 0.065717, 0.117832, 0.190685
};
constant float TURBOQUANT_THRESHOLDS_3BIT[7] = {
    -0.1542585, -0.0917745, -0.0435885, 0.0,
    0.0435885, 0.0917745, 0.1542585
};

constant float TURBOQUANT_CODEBOOK_4BIT[16] = {
    -0.173926, -0.117195, -0.089527, -0.068756,
    -0.051262, -0.035597, -0.020989, -0.006938,
    0.006938, 0.020989, 0.035597, 0.051262,
    0.068756, 0.089527, 0.117195, 0.173926
};
constant float TURBOQUANT_THRESHOLDS_4BIT[15] = {
    -0.145560, -0.103361, -0.079142, -0.060009,
    -0.043430, -0.028293, -0.013963, 0.0,
    0.013963, 0.028293, 0.043430, 0.060009,
    0.079142, 0.103361, 0.145560
};

constant float TURBOQUANT_CODEBOOK_5BIT[32] = {
    -3.183201, -2.6029081, -2.222862, -1.9298121, -1.6865128, -1.4757856, -1.2881749, -1.1178436,
    -0.9608707, -0.8144341, -0.6763788, -0.54496944, -0.41874045, -0.29640123, -0.17677347, -0.058747794,
    0.058747794, 0.17677347, 0.29640123, 0.41874045, 0.54496944, 0.6763788, 0.8144341, 0.9608707,
    1.1178436, 1.2881749, 1.4757856, 1.6865128, 1.9298121, 2.222862, 2.6029081, 3.183201
};
constant float TURBOQUANT_THRESHOLDS_5BIT[31] = {
    -2.8930547, -2.4128852, -2.076337, -1.8081625, -1.5811492, -1.3819802, -1.2030092, -1.0393572,
    -0.8876524, -0.74540645, -0.6106741, -0.48185492, -0.35757083, -0.23658736, -0.117760636, 0.0,
    0.117760636, 0.23658736, 0.35757083, 0.48185492, 0.6106741, 0.74540645, 0.8876524, 1.0393572,
    1.2030092, 1.3819802, 1.5811492, 1.8081625, 2.076337, 2.4128852, 2.8930547
};

constant float TURBOQUANT_CODEBOOK_6BIT[64] = {
    -3.017255738, -2.423820320, -2.058930059, -1.797707297, -1.601922457, -1.448291610, -1.326356628, -1.225995089,
    -1.139137223, -1.062727519, -0.993135300, -0.928426027, -0.868524053, -0.811485776, -0.757007509, -0.704944748,
    -0.654529973, -0.606343034, -0.559378028, -0.513393457, -0.468229381, -0.424872197, -0.382620330, -0.340845414,
    -0.299256835, -0.258542997, -0.217923724, -0.178098846, -0.138179662, -0.098636745, -0.058835570, -0.018710570,
    0.021130944, 0.060815313, 0.100191218, 0.139750780, 0.179826144, 0.219826894, 0.260545027, 0.301575157,
    0.342993060, 0.384991868, 0.427889936, 0.471579352, 0.516263475, 0.561314135, 0.607624272, 0.655973460,
    0.705897157, 0.758131069, 0.812601781, 0.869977353, 0.930940401, 0.995699969, 1.065519595, 1.141443664,
    1.227592854, 1.328595886, 1.450287956, 1.602228166, 1.797860727, 2.058171284, 2.423209814, 3.012639275
};
constant float TURBOQUANT_THRESHOLDS_6BIT[63] = {
    -2.720538029, -2.241375189, -1.928318678, -1.699814877, -1.525107034, -1.387324119, -1.276175858, -1.182566156,
    -1.100932371, -1.027931409, -0.960780663, -0.898475040, -0.840004914, -0.784246643, -0.730976129, -0.679737361,
    -0.630436503, -0.582860531, -0.536385743, -0.490811419, -0.446550789, -0.403746264, -0.361732872, -0.320051124,
    -0.278899916, -0.238233360, -0.198011285, -0.158139254, -0.118408203, -0.078736157, -0.038773070, 0.001210187,
    0.040973129, 0.080503265, 0.119970999, 0.159788462, 0.199826519, 0.240185961, 0.281060092, 0.322284109,
    0.363992464, 0.406440902, 0.449734644, 0.493921414, 0.538788805, 0.584469203, 0.631798866, 0.680935309,
    0.732014113, 0.785366425, 0.841289567, 0.900458877, 0.963320185, 1.030609782, 1.103481629, 1.184518259,
    1.278094370, 1.389441921, 1.526258061, 1.700044446, 1.928016005, 2.240690549, 2.717924544
};

constant float TURBOQUANT_CODEBOOK_7BIT[128] = {
    -3.161821656, -2.593655236, -2.260366751, -2.036143901, -1.877254125, -1.756981236, -1.662470419, -1.584808772,
    -1.517156127, -1.456758653, -1.402001919, -1.350976044, -1.303012122, -1.257290771, -1.214683970, -1.174284348,
    -1.135820017, -1.099401736, -1.064814314, -1.031291885, -0.998516547, -0.966935452, -0.935911034, -0.905753606,
    -0.876445620, -0.847857376, -0.820004703, -0.792610167, -0.765753537, -0.739575621, -0.713990833, -0.688968586,
    -0.663956599, -0.639361813, -0.615405244, -0.591867693, -0.568442898, -0.545359109, -0.522723513, -0.500390633,
    -0.478618508, -0.457086748, -0.435321291, -0.413745130, -0.392190012, -0.371026980, -0.350001767, -0.328982207,
    -0.308341784, -0.287940557, -0.267649618, -0.247588727, -0.227361475, -0.207182363, -0.187185623, -0.167462390,
    -0.147365963, -0.127493547, -0.107718842, -0.088070616, -0.068404744, -0.048587333, -0.028918753, -0.009434275,
    0.010149665, 0.029802156, 0.049171144, 0.068828458, 0.088722840, 0.108229260, 0.128145472, 0.148108296,
    0.167952409, 0.187973966, 0.208235593, 0.228389448, 0.248464332, 0.268681237, 0.288765152, 0.309276338,
    0.329687599, 0.350446090, 0.371599662, 0.392955936, 0.414318992, 0.435806255, 0.457389054, 0.478953617,
    0.500847597, 0.523094315, 0.545609565, 0.568220557, 0.591509701, 0.615326460, 0.639434160, 0.663947113,
    0.688244905, 0.712990034, 0.738163980, 0.763943700, 0.790592008, 0.818028338, 0.846147718, 0.874858881,
    0.904436875, 0.934984934, 0.965949978, 0.997473628, 1.029677998, 1.063044768, 1.097693392, 1.134172843,
    1.172229797, 1.212077571, 1.254602382, 1.299288409, 1.346488593, 1.397246923, 1.452061588, 1.512432831,
    1.580646704, 1.659743219, 1.754504101, 1.873331421, 2.031510542, 2.255090788, 2.588195086, 3.143419937
};
constant float TURBOQUANT_THRESHOLDS_7BIT[127] = {
    -2.877738446, -2.427010993, -2.148255326, -1.956699013, -1.817117680, -1.709725828, -1.623639596, -1.550982450,
    -1.486957390, -1.429380286, -1.376488981, -1.326994083, -1.280151447, -1.235987371, -1.194484159, -1.155052183,
    -1.117610877, -1.082108025, -1.048053099, -1.014904216, -0.982725999, -0.951423243, -0.920832320, -0.891099613,
    -0.862151498, -0.833931039, -0.806307435, -0.779181852, -0.752664579, -0.726783227, -0.701479710, -0.676462593,
    -0.651659206, -0.627383529, -0.603636469, -0.580155296, -0.556901003, -0.534041311, -0.511557073, -0.489504570,
    -0.467852628, -0.446204019, -0.424533210, -0.402967571, -0.381608496, -0.360514373, -0.339491987, -0.318661995,
    -0.298141171, -0.277795088, -0.257619173, -0.237475101, -0.217271919, -0.197183993, -0.177324006, -0.157414176,
    -0.137429755, -0.117606194, -0.097894729, -0.078237680, -0.058496038, -0.038753043, -0.019176514, 0.000357695,
    0.019975910, 0.039486650, 0.058999801, 0.078775649, 0.098476050, 0.118187366, 0.138126884, 0.158030352,
    0.177963187, 0.198104780, 0.218312521, 0.238426890, 0.258572784, 0.278723194, 0.299020745, 0.319481968,
    0.340066844, 0.361022876, 0.382277799, 0.403637464, 0.425062623, 0.446597654, 0.468171335, 0.489900607,
    0.511970956, 0.534351940, 0.556915061, 0.579865129, 0.603418081, 0.627380310, 0.651690636, 0.676096009,
    0.700617469, 0.725577007, 0.751053840, 0.777267854, 0.804310173, 0.832088028, 0.860503300, 0.889647878,
    0.919710905, 0.950467456, 0.981711803, 1.013575813, 1.046361383, 1.080369080, 1.115933117, 1.153201320,
    1.192153684, 1.233339976, 1.276945396, 1.322888501, 1.371867758, 1.424654256, 1.482247210, 1.546539768,
    1.620194962, 1.707123660, 1.813917761, 1.952420982, 2.143300665, 2.421642937, 2.865807512
};

constant float TURBOQUANT_QJL_SCALE = 1.2533141373155001;
constant float TURBOQUANT_KEY_RESIDUAL_SCALE = 1.0;
constant uint TURBOQUANT_MAX_CODE_WORDS = 28u;
constant float TURBO_WHT_SIGNS1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
constant float TURBO_WHT_SIGNS2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

struct ERTurboQuantQuantizeParams {
    uint rowCount;
    uint sourceRowStride;
    uint destinationRowBase;
    uint codeWordsPerRow;
    uint regularBits;
    uint highPrecisionBits;
    uint highPrecisionChannelCount;
    uint reserved;
};

struct ERTurboQuantAttentionParams {
    uint seqLen;
    uint headDim;
    uint numHeads;
    uint numKVHeads;
    uint groupSize;
    float scale;
    float keyResidualScale;
    float valueResidualScale;
    uint causal;
    uint kvBlockSize;
    uint qBlockSize;
    uint kvSeqLen;
    uint qOffset;
    uint codeWordsPerRow;
    uint regularBits;
    uint highPrecisionBits;
    uint valueCodeWordsPerRow;
    uint valueRegularBits;
    uint valueHighPrecisionBits;
    uint reserved;
};

struct ERTurboQuantDebugScoreTerms {
    float mseDot;
    float residualDot;
    float rowNorm;
    float residualNorm;
    float score;
};

constant uint TURBOQUANT_Q8_0_BLOCK_BYTES = 34u;
constant uint TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK = 32u;

inline float4 tq_q8_0_load_float4(
    device const uchar *row,
    uint dim4
) {
    const uint scalarIndex = dim4 * 4u;
    const uint blockIndex = scalarIndex / TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    const uint inBlockIndex = scalarIndex % TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    device const uchar *block = row + blockIndex * TURBOQUANT_Q8_0_BLOCK_BYTES;
    const float scale = float(as_type<half>(*(device const ushort *)block));
    return scale * float4(
        float(as_type<char>(block[2 + inBlockIndex + 0])),
        float(as_type<char>(block[2 + inBlockIndex + 1])),
        float(as_type<char>(block[2 + inBlockIndex + 2])),
        float(as_type<char>(block[2 + inBlockIndex + 3]))
    );
}

inline uint tq_get_bit(device const uint *words, uint bitIndex) {
    uint wordIndex = bitIndex >> 5;
    uint shift = bitIndex & 31;
    return (words[wordIndex] >> shift) & 1u;
}

inline uint tq_get_bit(threadgroup const uint *words, uint bitIndex) {
    uint wordIndex = bitIndex >> 5;
    uint shift = bitIndex & 31;
    return (words[wordIndex] >> shift) & 1u;
}

inline uint tq_extract_code(device const uint *words, uint bitOffset, uint width) {
    uint wordIndex = bitOffset >> 5;
    uint shift = bitOffset & 31;
    uint mask = (1u << width) - 1u;
    uint value = (words[wordIndex] >> shift) & mask;
    uint spill = shift + width;
    if (spill > 32u) {
        uint remaining = spill - 32u;
        uint highBits = words[wordIndex + 1] & ((1u << remaining) - 1u);
        value |= highBits << (width - remaining);
    }
    return value;
}

inline void tq_insert_code(thread uint *words, uint bitOffset, uint width, uint code) {
    uint wordIndex = bitOffset >> 5;
    uint shift = bitOffset & 31;
    uint mask = (1u << width) - 1u;
    words[wordIndex] |= (code & mask) << shift;
    uint spill = shift + width;
    if (spill > 32u) {
        uint remaining = spill - 32u;
        words[wordIndex + 1] |= (code & mask) >> (width - remaining);
    }
}

inline uint tq_extract_split_plane_code(
    device const uint *words,
    uint dim,
    bool useHighPrecision,
    uint regularBits,
    uint highPrecisionBits,
    thread uint &sidebandOffset
) {
    uint baseCode = tq_extract_code(words, dim * regularBits, regularBits);
    if (!useHighPrecision) {
        return baseCode;
    }
    uint sidebandWidth = highPrecisionBits - regularBits;
    if (sidebandWidth == 0u) {
        return baseCode;
    }
    uint extra = tq_extract_code(words, sidebandOffset, sidebandWidth);
    sidebandOffset += sidebandWidth;
    return baseCode | (extra << regularBits);
}

inline float tq_centroid(uint bits, uint code) {
    switch (bits) {
        case 2: return TURBOQUANT_CODEBOOK_2BIT[code];
        case 3: return TURBOQUANT_CODEBOOK_3BIT[code];
        case 4: return TURBOQUANT_CODEBOOK_4BIT[code];
        case 5: return TURBOQUANT_CODEBOOK_5BIT[code];
        case 6: return TURBOQUANT_CODEBOOK_6BIT[code];
        case 7: return TURBOQUANT_CODEBOOK_7BIT[code];
        default: return 0.0;
    }
}

inline float tq_centroid_2bit(uint code) {
    return TURBOQUANT_CODEBOOK_2BIT[code];
}

inline float tq_centroid_3bit(uint code) {
    return TURBOQUANT_CODEBOOK_3BIT[code];
}

inline uint tq_code_for_value(float value, uint bits) {
    switch (bits) {
        case 2:
            for (uint i = 0; i < 3; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_2BIT[i]) { return i; }
            }
            return 3;
        case 3:
            for (uint i = 0; i < 7; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_3BIT[i]) { return i; }
            }
            return 7;
        case 4:
            for (uint i = 0; i < 15; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_4BIT[i]) { return i; }
            }
            return 15;
        case 5:
            for (uint i = 0; i < 31; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_5BIT[i]) { return i; }
            }
            return 31;
        case 6:
            for (uint i = 0; i < 63; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_6BIT[i]) { return i; }
            }
            return 63;
        case 7:
            for (uint i = 0; i < 127; ++i) {
                if (value < TURBOQUANT_THRESHOLDS_7BIT[i]) { return i; }
            }
            return 127;
        default:
            return 0;
    }
}

inline float tq_quantization_benefit(float value, uint regularBits, uint highPrecisionBits) {
    uint regularCode = tq_code_for_value(value, regularBits);
    uint highCode = tq_code_for_value(value, highPrecisionBits);
    float regularDelta = value - tq_centroid(regularBits, regularCode);
    float highDelta = value - tq_centroid(highPrecisionBits, highCode);
    return (regularDelta * regularDelta) - (highDelta * highDelta);
}

inline void tq_hadamard(thread float *values) {
    for (uint width = 1; width < 128; width <<= 1) {
        uint step = width << 1;
        for (uint base = 0; base < 128; base += step) {
            for (uint index = 0; index < width; ++index) {
                float lhs = values[base + index];
                float rhs = values[base + index + width];
                values[base + index] = lhs + rhs;
                values[base + index + width] = lhs - rhs;
            }
        }
    }
}

inline void tq_hadamard(threadgroup float *values) {
    for (uint width = 1; width < 128; width <<= 1) {
        uint step = width << 1;
        for (uint base = 0; base < 128; base += step) {
            for (uint index = 0; index < width; ++index) {
                float lhs = values[base + index];
                float rhs = values[base + index + width];
                values[base + index] = lhs + rhs;
                values[base + index + width] = lhs - rhs;
            }
        }
    }
}

inline void tq_hadamard_parallel(
    threadgroup float *values,
    uint lane,
    uint laneCount
) {
    for (uint width = 1; width < 128; width <<= 1) {
        uint step = width << 1;
        for (uint butterfly = lane; butterfly < 64; butterfly += laneCount) {
            uint base = (butterfly / width) * step + (butterfly % width);
            float lhs = values[base];
            float rhs = values[base + width];
            values[base] = lhs + rhs;
            values[base + width] = lhs - rhs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void tq_forward_randomized_hadamard(thread float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS1[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i] * 0.08838834764831845f;
    }
}

inline void tq_forward_randomized_hadamard(threadgroup float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS1[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i] * 0.08838834764831845f;
    }
}

inline void tq_forward_randomized_hadamard_parallel(
    threadgroup float *values,
    device const float *signs,
    uint lane,
    uint laneCount
) {
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= TURBO_WHT_SIGNS1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_hadamard_parallel(values, lane, laneCount);
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= TURBO_WHT_SIGNS2[i] * 0.08838834764831845f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void tq_inverse_randomized_hadamard(thread float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] = values[i] * TURBO_WHT_SIGNS1[i] * 0.08838834764831845f;
    }
}

inline void tq_inverse_randomized_hadamard(threadgroup float *values, device const float *signs) {
    for (uint i = 0; i < 128; ++i) {
        values[i] *= TURBO_WHT_SIGNS2[i];
    }
    tq_hadamard(values);
    for (uint i = 0; i < 128; ++i) {
        values[i] = values[i] * TURBO_WHT_SIGNS1[i] * 0.08838834764831845f;
    }
}

inline void tq_inverse_randomized_hadamard_parallel(
    threadgroup float *values,
    device const float *signs,
    uint lane,
    uint laneCount
) {
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] *= TURBO_WHT_SIGNS2[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_hadamard_parallel(values, lane, laneCount);
    for (uint i = lane; i < 128; i += laneCount) {
        values[i] = values[i] * TURBO_WHT_SIGNS1[i] * 0.08838834764831845f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void tq_forward_planar(thread float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) - (s * y);
        values[base + 1u] = (s * x) + (c * y);
    }
}

inline void tq_inverse_planar(thread float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) + (s * y);
        values[base + 1u] = (-s * x) + (c * y);
    }
}

inline void tq_forward_planar(threadgroup float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) - (s * y);
        values[base + 1u] = (s * x) + (c * y);
    }
}

inline void tq_inverse_planar(threadgroup float *values, device const float *rotationCoefficients) {
    for (uint pair = 0; pair < 64; ++pair) {
        uint base = pair * 2u;
        float x = values[base];
        float y = values[base + 1u];
        float c = rotationCoefficients[base];
        float s = rotationCoefficients[base + 1u];
        values[base] = (c * x) + (s * y);
        values[base + 1u] = (-s * x) + (c * y);
    }
}

inline void tq_forward_rotation(thread float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_forward_planar(values, rotationData);
    } else {
        tq_forward_randomized_hadamard(values, rotationData);
    }
}

inline void tq_inverse_rotation(thread float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_inverse_planar(values, rotationData);
    } else {
        tq_inverse_randomized_hadamard(values, rotationData);
    }
}

inline void tq_forward_rotation(threadgroup float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_forward_planar(values, rotationData);
    } else {
        tq_forward_randomized_hadamard(values, rotationData);
    }
}

inline void tq_inverse_rotation(threadgroup float *values, device const float *rotationData, bool usePlanar) {
    if (usePlanar) {
        tq_inverse_planar(values, rotationData);
    } else {
        tq_inverse_randomized_hadamard(values, rotationData);
    }
}

inline void tq_select_top32_bitonic_mask(
    threadgroup const float *rotated,
    threadgroup uint *maskWords,
    threadgroup float *benefits,
    threadgroup uint *indices,
    uint regularBits,
    uint highPrecisionBits,
    bool useQuantizationBenefit,
    uint lane,
    uint laneCount
) {
    for (uint dim = lane; dim < 128; dim += laneCount) {
        benefits[dim] = useQuantizationBenefit
            ? tq_quantization_benefit(rotated[dim], regularBits, highPrecisionBits)
            : fabs(rotated[dim]);
        indices[dim] = dim;
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= 128; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = lane; i < 128; i += laneCount) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = (i & k) == 0;
                    bool shouldSwap = ascending
                        ? benefits[i] > benefits[ixj]
                        : benefits[i] < benefits[ixj];
                    if (shouldSwap) {
                        float tmpBenefit = benefits[i];
                        benefits[i] = benefits[ixj];
                        benefits[ixj] = tmpBenefit;
                        uint tmpIndex = indices[i];
                        indices[i] = indices[ixj];
                        indices[ixj] = tmpIndex;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint rank = lane; rank < 32; rank += laneCount) {
        uint dim = indices[127 - rank];
        atomic_fetch_or_explicit(
            (threadgroup atomic_uint *)&maskWords[dim >> 5],
            1u << (dim & 31),
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void turboquant_quantize_rows(
    device const float *source [[buffer(0)]],
    device uint *outCodes [[buffer(1)]],
    device uint *outResidualSigns [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outMetadata [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    device const float *residualSigns [[buffer(7)]],
    device const float *innerQScaleInv [[buffer(8)]],
    uint rowIndex [[thread_position_in_grid]]
) {
    if (rowIndex >= params.rowCount) { return; }
    const bool useInnerQScaling = ((params.reserved >> 1) & 1u) != 0u;
    const bool usePlanarRotation = ((params.reserved >> 2) & 1u) != 0u;

    thread float normalized[128];
    thread float rotated[128];
    thread float reconstructed[128];
    thread float residual[128];
    thread float projectedResidual[128];
    thread bool highPrecisionMask[128];
    thread uint codeWords[TURBOQUANT_MAX_CODE_WORDS];
    thread uint signWords[4];
    thread uint maskWords[4];

    for (uint i = 0; i < TURBOQUANT_MAX_CODE_WORDS; ++i) { codeWords[i] = 0; }
    for (uint i = 0; i < 4; ++i) {
        signWords[i] = 0;
        maskWords[i] = 0;
    }

    uint sourceBase = rowIndex * params.sourceRowStride;
    float rowNormSq = 0.0;
    for (uint dim = 0; dim < 128; ++dim) {
        float value = source[sourceBase + dim];
        if (useInnerQScaling && innerQScaleInv != nullptr) {
            value /= innerQScaleInv[dim];
        }
        normalized[dim] = value;
        rowNormSq += value * value;
        highPrecisionMask[dim] = false;
    }

    float rowNorm = sqrt(rowNormSq);

    uint destinationRow = params.destinationRowBase + rowIndex;
    device uint *codeDst = outCodes + destinationRow * params.codeWordsPerRow;
    device uint *signDst = outResidualSigns + destinationRow * 4;
    device uint *maskDst = outOutlierMask + destinationRow * 4;
    device float *metaDst = outMetadata + destinationRow * 2;
    const bool usesResidualPath =
        params.highPrecisionChannelCount > 0u || params.highPrecisionBits != params.regularBits;

    if (rowNorm == 0.0) {
        for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = 0; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = 0;
            maskDst[i] = 0;
        }
        metaDst[0] = 0.0;
        metaDst[1] = 0.0;
        return;
    }

    for (uint dim = 0; dim < 128; ++dim) {
        normalized[dim] /= rowNorm;
        rotated[dim] = normalized[dim];
    }
    tq_forward_rotation(rotated, rotationSigns, usePlanarRotation);

    bool useQuantizationBenefit = (params.reserved & 1u) != 0u;
    for (uint pick = 0; pick < params.highPrecisionChannelCount; ++pick) {
        float bestBenefit = -INFINITY;
        uint bestIndex = 0;
        for (uint dim = 0; dim < 128; ++dim) {
            if (highPrecisionMask[dim]) { continue; }
            float benefit = useQuantizationBenefit
                ? tq_quantization_benefit(rotated[dim], params.regularBits, params.highPrecisionBits)
                : fabs(rotated[dim]);
            if (benefit > bestBenefit) {
                bestBenefit = benefit;
                bestIndex = dim;
            }
        }
        highPrecisionMask[bestIndex] = true;
    }

    uint sidebandOffset = 128u * params.regularBits;
    float reconstructedNormSq = 0.0;
    for (uint dim = 0; dim < 128; ++dim) {
        if (highPrecisionMask[dim]) {
            maskWords[dim >> 5] |= (1u << (dim & 31));
        }
        uint width = highPrecisionMask[dim] ? params.highPrecisionBits : params.regularBits;
        uint code = tq_code_for_value(rotated[dim], width);
        uint baseMask = (1u << params.regularBits) - 1u;
        tq_insert_code(codeWords, dim * params.regularBits, params.regularBits, code & baseMask);
        if (highPrecisionMask[dim] && params.highPrecisionBits > params.regularBits) {
            tq_insert_code(
                codeWords,
                sidebandOffset,
                params.highPrecisionBits - params.regularBits,
                code >> params.regularBits
            );
            sidebandOffset += params.highPrecisionBits - params.regularBits;
        }
        reconstructed[dim] = tq_centroid(width, code);
        reconstructedNormSq += reconstructed[dim] * reconstructed[dim];
    }

    if (!usesResidualPath) {
        float reconstructedNorm = sqrt(reconstructedNormSq);
        float correctedNorm = reconstructedNorm > 0.0 ? (rowNorm / reconstructedNorm) : rowNorm;
        for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = 0u;
            maskDst[i] = maskWords[i];
        }
        metaDst[0] = correctedNorm;
        metaDst[1] = 0.0;
        return;
    }

    tq_inverse_rotation(reconstructed, rotationSigns, usePlanarRotation);
    float residualNormSq = 0.0;
    for (uint dim = 0; dim < 128; ++dim) {
        residual[dim] = normalized[dim] - reconstructed[dim];
        projectedResidual[dim] = residual[dim];
        residualNormSq += residual[dim] * residual[dim];
    }

    float residualNorm = sqrt(residualNormSq);
    tq_forward_randomized_hadamard(projectedResidual, residualSigns);
    for (uint dim = 0; dim < 128; ++dim) {
        if (projectedResidual[dim] >= 0.0) {
            signWords[dim >> 5] |= (1u << (dim & 31));
        }
    }

    for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
    for (uint i = 0; i < 4; ++i) {
        signDst[i] = signWords[i];
        maskDst[i] = maskWords[i];
    }
    metaDst[0] = rowNorm;
    metaDst[1] = residualNorm;
}

inline void tq_quantize_small_aggressive_row(
    device const float *source,
    uint sourceBase,
    device uint *codeDst,
    device uint *signDst,
    device uint *maskDst,
    device float *metaDst,
    device const float *rotationSigns,
    device const float *residualSigns,
    uint lane,
    threadgroup float *normalized,
    threadgroup float *rotated,
    threadgroup float *reconstructed,
    threadgroup float *residual,
    threadgroup float *projectedResidual,
    threadgroup uint *codes,
    threadgroup uint *maskWords,
    threadgroup float *magnitudes,
    threadgroup uint *indices,
    threadgroup float *reduction,
    threadgroup float &rowNormValue,
    threadgroup float &residualNormValue,
    uint codeWordsPerRow,
    bool useQuantizationBenefit
) {
    constexpr uint kLaneCount = 32;
    constexpr uint kRegularBits = 2;
    constexpr uint kHighBits = 3;

    float partialNormSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        float value = source[sourceBase + dim];
        normalized[dim] = value;
        rotated[dim] = value;
        reconstructed[dim] = 0.0;
        residual[dim] = 0.0;
        projectedResidual[dim] = 0.0;
        codes[dim] = 0u;
        partialNormSq += value * value;
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    reduction[lane] = partialNormSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        rowNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint index = lane; index < codeWordsPerRow; index += kLaneCount) {
            codeDst[index] = 0u;
        }
        if (lane < 4) {
            signDst[lane] = 0u;
            maskDst[lane] = 0u;
        }
        if (lane == 0) {
            metaDst[0] = 0.0;
            metaDst[1] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        normalized[dim] /= rowNormValue;
        rotated[dim] = normalized[dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_forward_randomized_hadamard_parallel(rotated, rotationSigns, lane, kLaneCount);

    tq_select_top32_bitonic_mask(
        rotated,
        maskWords,
        magnitudes,
        indices,
        2u,
        3u,
        useQuantizationBenefit,
        lane,
        kLaneCount
    );

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        bool useHighPrecision = tq_get_bit(maskWords, dim) == 1u;
        uint bits = useHighPrecision ? kHighBits : kRegularBits;
        uint code = tq_code_for_value(rotated[dim], bits);
        codes[dim] = code;
        reconstructed[dim] = useHighPrecision ? tq_centroid_3bit(code) : tq_centroid_2bit(code);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_inverse_randomized_hadamard_parallel(reconstructed, rotationSigns, lane, kLaneCount);

    float partialResidualSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        residual[dim] = normalized[dim] - reconstructed[dim];
        projectedResidual[dim] = residual[dim];
        partialResidualSq += residual[dim] * residual[dim];
    }
    reduction[lane] = partialResidualSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        residualNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_forward_randomized_hadamard_parallel(projectedResidual, residualSigns, lane, kLaneCount);

    if (lane == 0) {
        uint codeWords[16];
        uint signWords[4];
        for (uint i = 0; i < 16; ++i) { codeWords[i] = 0u; }
        for (uint i = 0; i < 4; ++i) { signWords[i] = 0u; }

        uint sidebandOffset = 256u;
        for (uint dim = 0; dim < 128; ++dim) {
            uint code = codes[dim];
            tq_insert_code(codeWords, dim * kRegularBits, kRegularBits, code & 0x3u);
            if (tq_get_bit(maskWords, dim) == 1u) {
                tq_insert_code(codeWords, sidebandOffset, 1u, code >> kRegularBits);
                sidebandOffset += 1u;
            }
            if (projectedResidual[dim] >= 0.0) {
                signWords[dim >> 5] |= (1u << (dim & 31));
            }
        }

        for (uint i = 0; i < codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = signWords[i];
            maskDst[i] = maskWords[i];
        }
        metaDst[0] = rowNormValue;
        metaDst[1] = residualNormValue;
    }
}

kernel void turboquant_quantize_rows_small_aggressive(
    device const float *source [[buffer(0)]],
    device uint *outCodes [[buffer(1)]],
    device uint *outResidualSigns [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outMetadata [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    device const float *residualSigns [[buffer(7)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;
    threadgroup float residualNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    uint destinationRow = params.destinationRowBase + rowIndex;
    device uint *codeDst = outCodes + destinationRow * params.codeWordsPerRow;
    device uint *signDst = outResidualSigns + destinationRow * 4;
    device uint *maskDst = outOutlierMask + destinationRow * 4;
    device float *metaDst = outMetadata + destinationRow * 2;

    tq_quantize_small_aggressive_row(
        source,
        sourceBase,
        codeDst,
        signDst,
        maskDst,
        metaDst,
        rotationSigns,
        residualSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow,
        (params.reserved & 1u) != 0u
    );
}

kernel void turboquant_quantize_rows_small_aggressive_k(
    device const float *source [[buffer(0)]],
    device uint *outCodes [[buffer(1)]],
    device uint *outResidualSigns [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outMetadata [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    device const float *residualSigns [[buffer(7)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;
    threadgroup float residualNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    uint destinationRow = params.destinationRowBase + rowIndex;
    device uint *codeDst = outCodes + destinationRow * params.codeWordsPerRow;
    device uint *signDst = outResidualSigns + destinationRow * 4;
    device uint *maskDst = outOutlierMask + destinationRow * 4;
    device float *metaDst = outMetadata + destinationRow * 2;

    tq_quantize_small_aggressive_row(
        source,
        sourceBase,
        codeDst,
        signDst,
        maskDst,
        metaDst,
        rotationSigns,
        residualSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow,
        false
    );
}

kernel void turboquant_quantize_rows_small_aggressive_kv(
    device const float *keySource [[buffer(0)]],
    device const float *valueSource [[buffer(1)]],
    device uint *keyCodes [[buffer(2)]],
    device uint *keyResidualSigns [[buffer(3)]],
    device uint *keyOutlierMask [[buffer(4)]],
    device float *keyMetadata [[buffer(5)]],
    device uint *valueCodes [[buffer(6)]],
    device uint *valueResidualSigns [[buffer(7)]],
    device uint *valueOutlierMask [[buffer(8)]],
    device float *valueMetadata [[buffer(9)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;
    threadgroup float residualNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    uint destinationRow = params.destinationRowBase + rowIndex;

    tq_quantize_small_aggressive_row(
        keySource,
        sourceBase,
        keyCodes + destinationRow * params.codeWordsPerRow,
        keyResidualSigns + destinationRow * 4,
        keyOutlierMask + destinationRow * 4,
        keyMetadata + destinationRow * 2,
        keyRotationSigns,
        keyResidualProjectionSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow,
        false
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_quantize_small_aggressive_row(
        valueSource,
        sourceBase,
        valueCodes + destinationRow * params.codeWordsPerRow,
        valueResidualSigns + destinationRow * 4,
        valueOutlierMask + destinationRow * 4,
        valueMetadata + destinationRow * 2,
        valueRotationSigns,
        valueResidualProjectionSigns,
        lane,
        normalized,
        rotated,
        reconstructed,
        residual,
        projectedResidual,
        codes,
        maskWords,
        magnitudes,
        indices,
        reduction,
        rowNormValue,
        residualNormValue,
        params.codeWordsPerRow,
        true
    );
}

kernel void turboquant_quantize_rows_small_aggressive_phase1(
    device const float *source [[buffer(0)]],
    device float *outNormalized [[buffer(1)]],
    device float *outRotated [[buffer(2)]],
    device uint *outOutlierMask [[buffer(3)]],
    device float *outRowNorm [[buffer(4)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(5)]],
    device const float *rotationSigns [[buffer(6)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kLaneCount = 32;

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup uint maskWords[4];
    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    float partialNormSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        float value = source[sourceBase + dim];
        normalized[dim] = value;
        rotated[dim] = value;
        partialNormSq += value * value;
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    reduction[lane] = partialNormSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        rowNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint dim = lane; dim < 128; dim += kLaneCount) {
            outNormalized[rowIndex * 128 + dim] = 0.0;
            outRotated[rowIndex * 128 + dim] = 0.0;
        }
        if (lane < 4) {
            outOutlierMask[rowIndex * 4 + lane] = 0u;
        }
        if (lane == 0) {
            outRowNorm[rowIndex] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        normalized[dim] /= rowNormValue;
        rotated[dim] = normalized[dim];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_forward_randomized_hadamard_parallel(rotated, rotationSigns, lane, kLaneCount);

    tq_select_top32_bitonic_mask(
        rotated,
        maskWords,
        magnitudes,
        indices,
        2u,
        3u,
        false,
        lane,
        kLaneCount
    );
    if (lane == 0) {
        outRowNorm[rowIndex] = rowNormValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        outNormalized[rowIndex * 128 + dim] = normalized[dim];
        outRotated[rowIndex * 128 + dim] = rotated[dim];
    }
    if (lane < 4) {
        outOutlierMask[rowIndex * 4 + lane] = maskWords[lane];
    }
}

kernel void turboquant_quantize_rows_small_aggressive_rotate_only(
    device const float *source [[buffer(0)]],
    device float *outRotated [[buffer(1)]],
    device float *outRowNorm [[buffer(2)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(3)]],
    device const float *rotationSigns [[buffer(4)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kLaneCount = 32;

    threadgroup float rotated[128];
    threadgroup float reduction[32];
    threadgroup float rowNormValue;

    uint sourceBase = rowIndex * params.sourceRowStride;
    float partialNormSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        float value = source[sourceBase + dim];
        rotated[dim] = value;
        partialNormSq += value * value;
    }
    reduction[lane] = partialNormSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        rowNormValue = sqrt(reduction[0]);
        outRowNorm[rowIndex] = rowNormValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint dim = lane; dim < 128; dim += kLaneCount) {
            outRotated[rowIndex * 128 + dim] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        rotated[dim] /= rowNormValue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_forward_randomized_hadamard_parallel(rotated, rotationSigns, lane, kLaneCount);
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        outRotated[rowIndex * 128 + dim] = rotated[dim];
    }
}

kernel void turboquant_quantize_rows_small_aggressive_select_only(
    device const float *rotatedSource [[buffer(0)]],
    device uint *outOutlierMask [[buffer(1)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(2)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kHighCount = 32;

    threadgroup float rotated[128];
    threadgroup uint maskWords[4];

    for (uint dim = lane; dim < 128; dim += 32) {
        rotated[dim] = rotatedSource[rowIndex * 128 + dim];
    }
    if (lane < 4) {
        maskWords[lane] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint pick = 0; pick < kHighCount; ++pick) {
            float bestBenefit = -INFINITY;
            uint bestIndex = 0;
            for (uint dim = 0; dim < 128; ++dim) {
                if (tq_get_bit(maskWords, dim) == 1u) { continue; }
                float benefit = tq_quantization_benefit(rotated[dim], 2u, 3u);
                if (benefit > bestBenefit) {
                    bestBenefit = benefit;
                    bestIndex = dim;
                }
            }
            maskWords[bestIndex >> 5] |= (1u << (bestIndex & 31));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < 4) {
        outOutlierMask[rowIndex * 4 + lane] = maskWords[lane];
    }
}

kernel void turboquant_quantize_rows_small_aggressive_select_only_bitonic(
    device const float *rotatedSource [[buffer(0)]],
    device uint *outOutlierMask [[buffer(1)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(2)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    threadgroup float magnitudes[128];
    threadgroup uint indices[128];
    for (uint dim = lane; dim < 128; dim += 32) {
        magnitudes[dim] = tq_quantization_benefit(rotatedSource[rowIndex * 128 + dim], 2u, 3u);
        indices[dim] = dim;
    }
    if (lane < 4) {
        outOutlierMask[rowIndex * 4 + lane] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= 128; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = lane; i < 128; i += 32) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = (i & k) == 0;
                    bool shouldSwap = ascending
                        ? magnitudes[i] > magnitudes[ixj]
                        : magnitudes[i] < magnitudes[ixj];
                    if (shouldSwap) {
                        float tmpMagnitude = magnitudes[i];
                        magnitudes[i] = magnitudes[ixj];
                        magnitudes[ixj] = tmpMagnitude;
                        uint tmpIndex = indices[i];
                        indices[i] = indices[ixj];
                        indices[ixj] = tmpIndex;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint rank = lane; rank < 32; rank += 32) {
        uint dim = indices[127 - rank];
        atomic_fetch_or_explicit(
            (device atomic_uint *)&outOutlierMask[rowIndex * 4 + (dim >> 5)],
            1u << (dim & 31),
            memory_order_relaxed
        );
    }
}

kernel void turboquant_quantize_rows_small_aggressive_phase2(
    device const float *normalizedSource [[buffer(0)]],
    device const float *rotatedSource [[buffer(1)]],
    device const uint *inputOutlierMask [[buffer(2)]],
    device const float *inputRowNorm [[buffer(3)]],
    device uint *outCodes [[buffer(4)]],
    device uint *outResidualSigns [[buffer(5)]],
    device uint *outOutlierMask [[buffer(6)]],
    device float *outMetadata [[buffer(7)]],
    constant ERTurboQuantQuantizeParams &params [[buffer(8)]],
    device const float *rotationSigns [[buffer(9)]],
    device const float *residualSigns [[buffer(10)]],
    uint rowIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (rowIndex >= params.rowCount) { return; }

    constexpr uint kLaneCount = 32;

    threadgroup float normalized[128];
    threadgroup float rotated[128];
    threadgroup float reconstructed[128];
    threadgroup float residual[128];
    threadgroup float projectedResidual[128];
    threadgroup uint codes[128];
    threadgroup uint maskWords[4];
    threadgroup float reduction[32];
    threadgroup float residualNormValue;
    threadgroup float rowNormValue;

    if (lane == 0) {
        rowNormValue = inputRowNorm[rowIndex];
    }
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        normalized[dim] = normalizedSource[rowIndex * 128 + dim];
        rotated[dim] = rotatedSource[rowIndex * 128 + dim];
        reconstructed[dim] = 0.0;
        residual[dim] = 0.0;
        projectedResidual[dim] = 0.0;
        codes[dim] = 0u;
    }
    if (lane < 4) {
        maskWords[lane] = inputOutlierMask[rowIndex * 4 + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rowNormValue == 0.0) {
        for (uint index = lane; index < params.codeWordsPerRow; index += kLaneCount) {
            outCodes[rowIndex * params.codeWordsPerRow + index] = 0u;
        }
        if (lane < 4) {
            outResidualSigns[rowIndex * 4 + lane] = 0u;
            outOutlierMask[rowIndex * 4 + lane] = maskWords[lane];
        }
        if (lane == 0) {
            outMetadata[rowIndex * 2] = 0.0;
            outMetadata[rowIndex * 2 + 1] = 0.0;
        }
        return;
    }

    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        bool useHighPrecision = tq_get_bit(maskWords, dim) == 1u;
        uint code = tq_code_for_value(rotated[dim], useHighPrecision ? 3u : 2u);
        codes[dim] = code;
        reconstructed[dim] = useHighPrecision ? tq_centroid_3bit(code) : tq_centroid_2bit(code);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_inverse_randomized_hadamard_parallel(reconstructed, rotationSigns, lane, kLaneCount);

    float partialResidualSq = 0.0;
    for (uint dim = lane; dim < 128; dim += kLaneCount) {
        residual[dim] = normalized[dim] - reconstructed[dim];
        projectedResidual[dim] = residual[dim];
        partialResidualSq += residual[dim] * residual[dim];
    }
    reduction[lane] = partialResidualSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kLaneCount >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        residualNormValue = sqrt(reduction[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_forward_randomized_hadamard_parallel(projectedResidual, residualSigns, lane, kLaneCount);

    if (lane == 0) {
        uint codeWords[16];
        uint signWords[4];
        for (uint i = 0; i < 16; ++i) { codeWords[i] = 0u; }
        for (uint i = 0; i < 4; ++i) { signWords[i] = 0u; }

        uint sidebandOffset = 256u;
        for (uint dim = 0; dim < 128; ++dim) {
            uint code = codes[dim];
            tq_insert_code(codeWords, dim * 2u, 2u, code & 0x3u);
            if (tq_get_bit(maskWords, dim) == 1u) {
                tq_insert_code(codeWords, sidebandOffset, 1u, code >> 2u);
                sidebandOffset += 1u;
            }
            if (projectedResidual[dim] >= 0.0) {
                signWords[dim >> 5] |= (1u << (dim & 31));
            }
        }

        device uint *codeDst = outCodes + rowIndex * params.codeWordsPerRow;
        device uint *signDst = outResidualSigns + rowIndex * 4;
        device uint *maskDst = outOutlierMask + rowIndex * 4;
        device float *metaDst = outMetadata + rowIndex * 2;
        for (uint i = 0; i < params.codeWordsPerRow; ++i) { codeDst[i] = codeWords[i]; }
        for (uint i = 0; i < 4; ++i) {
            signDst[i] = signWords[i];
            maskDst[i] = maskWords[i];
        }
        metaDst[0] = rowNormValue;
        metaDst[1] = residualNormValue;
    }
}

kernel void gqa_attention_turboquant(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const uint *VCodes [[buffer(5)]],
    device const uint *VResidualSigns [[buffer(6)]],
    device const uint *VOutlierMask [[buffer(7)]],
    device const float *VMetadata [[buffer(8)]],
    device float *O [[buffer(9)]],
    constant ERTurboQuantAttentionParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    device const float *innerQScaleInv [[buffer(15)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;
    const uint blockSize = params.qBlockSize;
    const uint qStride = params.numHeads * params.headDim;
    const bool useInnerQScaling = (params.reserved & 1u) != 0u;
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool usePlanarValueRotation = (params.reserved & 4u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    threadgroup float qRotationScratch[16 * 128];
    threadgroup float qResidualScratch[16 * 128];
    threadgroup float outputMSEScratch[16 * 128];
    threadgroup float outputResidualScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float scores[16];
    float probs[16];

    if (activeQ) {
        threadgroup float *qRotation = qRotationScratch + local_id.x * 128;
        threadgroup float *qResidual = qResidualScratch + local_id.x * 128;
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;
        uint qBase = qRow * qStride + headIndex * params.headDim;

        for (uint dim = 0; dim < 128; ++dim) {
            float value = Q[qBase + dim];
            if (useInnerQScaling && innerQScaleInv != nullptr) {
                value *= innerQScaleInv[dim];
            }
            qRotation[dim] = value;
            qResidual[dim] = value;
            outputMSE[dim] = 0.0;
            outputResidual[dim] = 0.0;
        }
        tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
        if (useKeyResidualPath) {
            tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; ++kvBlock) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (activeQ) {
            threadgroup float *qRotation = qRotationScratch + local_id.x * 128;
            threadgroup float *qResidual = qResidualScratch + local_id.x * 128;
            threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
            threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;

            float blockMax = -INFINITY;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uint *codeRow = KCodes + rowIndex * params.codeWordsPerRow;
                device const uint *signRow = KResidualSigns + rowIndex * 4;
                device const uint *maskRow = KOutlierMask + rowIndex * 4;
                device const float *metaRow = KMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];

                float mseDot = 0.0;
                float residualDot = 0.0;
                uint sidebandOffset = 128u * params.regularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.regularBits,
                        params.highPrecisionBits,
                        sidebandOffset
                    );
                    mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
                    if (useKeyResidualPath) {
                        residualDot += qResidual[dim] * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                    }
                }

                float dot = rowNorm * (
                    mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * residualNorm * residualDot
                );
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (scores[kvIndex] == -INFINITY) {
                    probs[kvIndex] = 0.0;
                } else {
                    probs[kvIndex] = exp(scores[kvIndex] - nextMax);
                }
                blockSum += probs[kvIndex];
            }

            runningSum = runningSum * correction + blockSum;
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= correction;
                outputResidual[dim] *= correction;
            }

            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                float prob = probs[kvIndex];
                if (prob == 0.0) { continue; }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uint *codeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
                device const uint *signRow = VResidualSigns + rowIndex * 4;
                device const uint *maskRow = VOutlierMask + rowIndex * 4;
                device const float *metaRow = VMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];
                float mseScale = prob * rowNorm;
                float residualScale = prob * rowNorm * residualNorm;

                uint sidebandOffset = 128u * params.valueRegularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.valueRegularBits,
                        params.valueHighPrecisionBits,
                        sidebandOffset
                    );
                    outputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
                    outputResidual[dim] += residualScale * (tq_get_bit(signRow, dim) == 1u ? 1.0 : -1.0);
                }
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;

        tq_inverse_rotation(outputMSE, valueRotationSigns, usePlanarValueRotation);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);

        if (useInnerQScaling && innerQScaleInv != nullptr) {
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= innerQScaleInv[dim];
                outputResidual[dim] *= innerQScaleInv[dim];
            }
        }

        float invSum = runningSum > 0.0 ? 1.0 / runningSum : 0.0;
        uint oBase = qRow * qStride + headIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            O[oBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
        }
    }
}

kernel void gqa_attention_q8k_turboquant(
    device const float *Q [[buffer(0)]],
    device const uchar *K [[buffer(1)]],
    device const uint *VCodes [[buffer(2)]],
    device const uint *VResidualSigns [[buffer(3)]],
    device const uint *VOutlierMask [[buffer(4)]],
    device const float *VMetadata [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *valueRotationSigns [[buffer(8)]],
    device const float *valueResidualProjectionSigns [[buffer(9)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    const uint qBlockIndex = group_id.x;
    const uint headIndex = group_id.y;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint seqLen = params.seqLen;
    const uint kvSeqLen = params.kvSeqLen > 0 ? params.kvSeqLen : seqLen;
    const uint qOff = params.qOffset;
    const uint blockSize = params.qBlockSize;
    const uint qStride = params.numHeads * params.headDim;
    const uint headDim4 = params.headDim / 4u;
    const uint q8BlocksPerRow = params.headDim / TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    const uint q8RowBytes = q8BlocksPerRow * TURBOQUANT_Q8_0_BLOCK_BYTES;

    uint qRow = qBlockIndex * blockSize + local_id.x;
    bool activeQ = (qRow < seqLen);

    threadgroup float outputMSEScratch[16 * 128];
    threadgroup float outputResidualScratch[16 * 128];

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float scores[16];
    float probs[16];

    if (activeQ) {
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;
        for (uint dim = 0; dim < 128; ++dim) {
            outputMSE[dim] = 0.0;
            outputResidual[dim] = 0.0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kvBlockCount = (kvSeqLen + blockSize - 1) / blockSize;
    for (uint kvBlock = 0; kvBlock < kvBlockCount; ++kvBlock) {
        uint kvStart = kvBlock * blockSize;
        uint kvEnd = min(kvStart + blockSize, kvSeqLen);
        uint kvCount = kvEnd - kvStart;

        if (activeQ) {
            threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
            threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;
            uint qBase = qRow * qStride + headIndex * params.headDim;
            const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);

            float blockMax = -INFINITY;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (params.causal != 0 && kvStart + kvIndex > qRow + qOff) {
                    scores[kvIndex] = -INFINITY;
                    continue;
                }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uchar *kRow = K + rowIndex * q8RowBytes;
                float dot = 0.0f;
                for (uint dim4 = 0; dim4 < headDim4; ++dim4) {
                    dot += metal::dot(qVec[dim4], tq_q8_0_load_float4(kRow, dim4));
                }
                scores[kvIndex] = dot * params.scale;
                blockMax = max(blockMax, scores[kvIndex]);
            }

            float nextMax = max(runningMax, blockMax);
            float correction = exp(runningMax - nextMax);
            float blockSum = 0.0f;
            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                if (scores[kvIndex] == -INFINITY) {
                    probs[kvIndex] = 0.0f;
                } else {
                    probs[kvIndex] = exp(scores[kvIndex] - nextMax);
                }
                blockSum += probs[kvIndex];
            }

            runningSum = runningSum * correction + blockSum;
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= correction;
                outputResidual[dim] *= correction;
            }

            for (uint kvIndex = 0; kvIndex < kvCount; ++kvIndex) {
                float prob = probs[kvIndex];
                if (prob == 0.0f) { continue; }

                uint rowIndex = (kvStart + kvIndex) * params.numKVHeads + kvHeadIndex;
                device const uint *codeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
                device const uint *signRow = VResidualSigns + rowIndex * 4;
                device const uint *maskRow = VOutlierMask + rowIndex * 4;
                device const float *metaRow = VMetadata + rowIndex * 2;
                float rowNorm = metaRow[0];
                float residualNorm = metaRow[1];
                float mseScale = prob * rowNorm;
                float residualScale = prob * rowNorm * residualNorm;

                uint sidebandOffset = 128u * params.valueRegularBits;
                for (uint dim = 0; dim < 128; ++dim) {
                    bool useHighPrecision = tq_get_bit(maskRow, dim) == 1u;
                    uint code = tq_extract_split_plane_code(
                        codeRow,
                        dim,
                        useHighPrecision,
                        params.valueRegularBits,
                        params.valueHighPrecisionBits,
                        sidebandOffset
                    );
                    outputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
                    outputResidual[dim] += residualScale * (tq_get_bit(signRow, dim) == 1u ? 1.0f : -1.0f);
                }
            }
            runningMax = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (activeQ) {
        threadgroup float *outputMSE = outputMSEScratch + local_id.x * 128;
        threadgroup float *outputResidual = outputResidualScratch + local_id.x * 128;

        tq_inverse_randomized_hadamard(outputMSE, valueRotationSigns);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);

        float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
        uint oBase = qRow * qStride + headIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            O[oBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
        }
    }
}

kernel void gqa_attention_turboquant_decode(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const uint *VCodes [[buffer(5)]],
    device const uint *VResidualSigns [[buffer(6)]],
    device const uint *VOutlierMask [[buffer(7)]],
    device const float *VMetadata [[buffer(8)]],
    device float *O [[buffer(9)]],
    constant ERTurboQuantAttentionParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    device const float *innerQScaleInv [[buffer(15)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }
    const bool useInnerQScaling = (params.reserved & 1u) != 0u;
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool usePlanarValueRotation = (params.reserved & 4u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

    constexpr uint kDecodeThreads = 16;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float outputMSE[128];
    threadgroup float outputResidual[128];
    threadgroup float partialMSE[kDecodeThreads * 128];
    threadgroup float partialResidual[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutputMSE = partialMSE + lane * 128;
    threadgroup float *laneOutputResidual = partialResidual + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        if (useInnerQScaling && innerQScaleInv != nullptr) {
            value *= innerQScaleInv[dim];
        }
        qRotation[dim] = value;
        qResidual[dim] = value;
        outputMSE[dim] = 0.0;
        outputResidual[dim] = 0.0;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutputMSE[dim] = 0.0;
        laneOutputResidual[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
        if (useKeyResidualPath) {
            tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;

    for (uint kvPos = lane; kvPos < kvLimit; kvPos += kDecodeThreads) {
        uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
        device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
        device const uint *kSignRow = KResidualSigns + rowIndex * 4;
        device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
        device const float *kMetaRow = KMetadata + rowIndex * 2;
        float keyRowNorm = kMetaRow[0];
        float keyResidualNorm = kMetaRow[1];

        float mseDot = 0.0;
        float residualDot = 0.0;
        uint keySidebandOffset = 128u * params.regularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(kMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                kCodeRow,
                dim,
                useHighPrecision,
                params.regularBits,
                params.highPrecisionBits,
                keySidebandOffset
            );
            mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
            if (useKeyResidualPath) {
                residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
            }
        }
        float score = keyRowNorm * (
            mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
        ) * params.scale;
        float nextMax = max(runningMax, score);
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float prob = exp(score - nextMax);
        runningSum = runningSum * correction + prob;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutputMSE[dim] *= correction;
            laneOutputResidual[dim] *= correction;
        }

        device const uint *vCodeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
        device const uint *vSignRow = VResidualSigns + rowIndex * 4;
        device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
        device const float *vMetaRow = VMetadata + rowIndex * 2;
        float valueRowNorm = vMetaRow[0];
        float valueResidualNorm = vMetaRow[1];
        float mseScale = prob * valueRowNorm;
        float residualScale = prob * valueRowNorm * valueResidualNorm;

        uint valueSidebandOffset = 128u * params.valueRegularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(vMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                vCodeRow,
                dim,
                useHighPrecision,
                params.valueRegularBits,
                params.valueHighPrecisionBits,
                valueSidebandOffset
            );
            laneOutputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
            laneOutputResidual[dim] += residualScale * (tq_get_bit(vSignRow, dim) == 1u ? 1.0 : -1.0);
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float mseAccum = 0.0;
        float residualAccum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker];
            mseAccum += partialMSE[worker * 128 + dim] * workerScale;
            residualAccum += partialResidual[worker * 128 + dim] * workerScale;
        }
        outputMSE[dim] = mseAccum;
        outputResidual[dim] = residualAccum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_inverse_rotation(outputMSE, valueRotationSigns, usePlanarValueRotation);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);
        if (useInnerQScaling && innerQScaleInv != nullptr) {
            for (uint dim = 0; dim < 128; ++dim) {
                outputMSE[dim] *= innerQScaleInv[dim];
                outputResidual[dim] *= innerQScaleInv[dim];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
    }
}

kernel void gqa_attention_q8k_turboquant_decode(
    device const float *Q [[buffer(0)]],
    device const uchar *K [[buffer(1)]],
    device const uint *VCodes [[buffer(2)]],
    device const uint *VResidualSigns [[buffer(3)]],
    device const uint *VOutlierMask [[buffer(4)]],
    device const float *VMetadata [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *valueRotationSigns [[buffer(8)]],
    device const float *valueResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16u;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint headDim4 = params.headDim / 4u;
    const uint q8BlocksPerRow = params.headDim / TURBOQUANT_Q8_0_WEIGHTS_PER_BLOCK;
    const uint q8RowBytes = q8BlocksPerRow * TURBOQUANT_Q8_0_BLOCK_BYTES;

    threadgroup float outputMSE[128];
    threadgroup float outputResidual[128];
    threadgroup float partialMSE[kDecodeThreads * 128];
    threadgroup float partialResidual[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutputMSE = partialMSE + lane * 128;
    threadgroup float *laneOutputResidual = partialResidual + lane * 128;
    const device float4 *qVec = reinterpret_cast<const device float4 *>(Q + qBase);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        outputMSE[dim] = 0.0f;
        outputResidual[dim] = 0.0f;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutputMSE[dim] = 0.0f;
        laneOutputResidual[dim] = 0.0f;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0f;

    for (uint kvPos = lane; kvPos < kvLimit; kvPos += kDecodeThreads) {
        uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
        device const uchar *kRow = K + rowIndex * q8RowBytes;
        float dot = 0.0f;
        for (uint dim4 = 0; dim4 < headDim4; ++dim4) {
            dot += metal::dot(qVec[dim4], tq_q8_0_load_float4(kRow, dim4));
        }

        float score = dot * params.scale;
        float nextMax = max(runningMax, score);
        float correction = runningMax == -INFINITY ? 0.0f : exp(runningMax - nextMax);
        float prob = exp(score - nextMax);
        runningSum = runningSum * correction + prob;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutputMSE[dim] *= correction;
            laneOutputResidual[dim] *= correction;
        }

        device const uint *vCodeRow = VCodes + rowIndex * params.valueCodeWordsPerRow;
        device const uint *vSignRow = VResidualSigns + rowIndex * 4;
        device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
        device const float *vMetaRow = VMetadata + rowIndex * 2;
        float valueRowNorm = vMetaRow[0];
        float valueResidualNorm = vMetaRow[1];
        float mseScale = prob * valueRowNorm;
        float residualScale = prob * valueRowNorm * valueResidualNorm;

        uint valueSidebandOffset = 128u * params.valueRegularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(vMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                vCodeRow,
                dim,
                useHighPrecision,
                params.valueRegularBits,
                params.valueHighPrecisionBits,
                valueSidebandOffset
            );
            laneOutputMSE[dim] += mseScale * tq_centroid(useHighPrecision ? params.valueHighPrecisionBits : params.valueRegularBits, code);
            laneOutputResidual[dim] += residualScale * (tq_get_bit(vSignRow, dim) == 1u ? 1.0f : -1.0f);
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0f ? exp(runningMax - globalMax) : 0.0f;
    laneScale[lane] = localScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float mseAccum = 0.0f;
        float residualAccum = 0.0f;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker];
            mseAccum += partialMSE[worker * 128 + dim] * workerScale;
            residualAccum += partialResidual[worker * 128 + dim] * workerScale;
        }
        outputMSE[dim] = mseAccum;
        outputResidual[dim] = residualAccum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_inverse_randomized_hadamard(outputMSE, valueRotationSigns);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0f ? 1.0f / globalSum : 0.0f;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
    }
}

kernel void turboquant_debug_decode_score_terms(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device ERTurboQuantDebugScoreTerms *scoreTerms [[buffer(5)]],
    constant ERTurboQuantAttentionParams &params [[buffer(6)]],
    device const float *keyRotationSigns [[buffer(7)]],
    device const float *keyResidualProjectionSigns [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint kvPos = gid.x;
    uint headIndex = gid.y;
    if (headIndex >= params.numHeads) { return; }
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    if (kvPos >= kvLimit) { return; }

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint qBase = headIndex * params.headDim;
    uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;

    thread float qRotation[128];
    thread float qResidual[128];
    for (uint dim = 0; dim < 128; ++dim) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
    }
    tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
    if (useKeyResidualPath) {
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }

    device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
    device const uint *kSignRow = KResidualSigns + rowIndex * 4;
    device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
    device const float *kMetaRow = KMetadata + rowIndex * 2;
    float keyRowNorm = kMetaRow[0];
    float keyResidualNorm = kMetaRow[1];

    float mseDot = 0.0;
    float residualDot = 0.0;
    uint keySidebandOffset = 128u * params.regularBits;
    for (uint dim = 0; dim < 128; ++dim) {
        bool useHighPrecision = tq_get_bit(kMaskRow, dim) == 1u;
        uint code = tq_extract_split_plane_code(
            kCodeRow,
            dim,
            useHighPrecision,
            params.regularBits,
            params.highPrecisionBits,
            keySidebandOffset
        );
        mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
        if (useKeyResidualPath) {
            residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
        }
    }

    float score = keyRowNorm * (
        mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
    ) * params.scale;

    uint outputIndex = headIndex * kvLimit + kvPos;
    scoreTerms[outputIndex] = {
        mseDot,
        useKeyResidualPath ? residualDot : 0.0f,
        keyRowNorm,
        useKeyResidualPath ? keyResidualNorm : 0.0f,
        score
    };
}

kernel void gqa_attention_turboquant_decode_f16v(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const half *V [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *keyRotationSigns [[buffer(8)]],
    device const float *keyResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }
    const bool usePlanarKeyRotation = (params.reserved & 2u) != 0u;
    const bool useKeyResidualPath = params.keyResidualScale != 0.0f;

    constexpr uint kDecodeThreads = 16;
    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint kvStride = params.numKVHeads * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float output[128];
    threadgroup float partialOutput[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutput = partialOutput + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        output[dim] = 0.0;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_rotation(qRotation, keyRotationSigns, usePlanarKeyRotation);
        if (useKeyResidualPath) {
            tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;

    for (uint kvPos = lane; kvPos < kvLimit; kvPos += kDecodeThreads) {
        uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
        device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
        device const uint *kSignRow = KResidualSigns + rowIndex * 4;
        device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
        device const float *kMetaRow = KMetadata + rowIndex * 2;
        float keyRowNorm = kMetaRow[0];
        float keyResidualNorm = kMetaRow[1];

        float mseDot = 0.0;
        float residualDot = 0.0;
        uint keySidebandOffset = 128u * params.regularBits;
        for (uint dim = 0; dim < 128; ++dim) {
            bool useHighPrecision = tq_get_bit(kMaskRow, dim) == 1u;
            uint code = tq_extract_split_plane_code(
                kCodeRow,
                dim,
                useHighPrecision,
                params.regularBits,
                params.highPrecisionBits,
                keySidebandOffset
            );
            mseDot += qRotation[dim] * tq_centroid(useHighPrecision ? params.highPrecisionBits : params.regularBits, code);
            if (useKeyResidualPath) {
                residualDot += qResidual[dim] * (tq_get_bit(kSignRow, dim) == 1u ? 1.0 : -1.0);
            }
        }
        float score = keyRowNorm * (
            mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
        ) * params.scale;
        float nextMax = max(runningMax, score);
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float prob = exp(score - nextMax);
        runningSum = runningSum * correction + prob;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutput[dim] *= correction;
        }

        uint valueBase = kvPos * kvStride + kvHeadIndex * params.headDim;
        for (uint dim = 0; dim < 128; ++dim) {
            laneOutput[dim] += prob * float(V[valueBase + dim]);
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float accum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker];
            accum += partialOutput[worker * 128 + dim] * workerScale;
        }
        output[dim] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = output[dim] * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_aggressive(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const uint *VCodes [[buffer(5)]],
    device const uint *VResidualSigns [[buffer(6)]],
    device const uint *VOutlierMask [[buffer(7)]],
    device const float *VMetadata [[buffer(8)]],
    device float *O [[buffer(9)]],
    constant ERTurboQuantAttentionParams &params [[buffer(10)]],
    device const float *keyRotationSigns [[buffer(11)]],
    device const float *keyResidualProjectionSigns [[buffer(12)]],
    device const float *valueRotationSigns [[buffer(13)]],
    device const float *valueResidualProjectionSigns [[buffer(14)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    constexpr uint kRegularBits = 2;
    constexpr uint kHighBits = 3;
    constexpr uint kTileRows = 4;

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float outputMSE[128];
    threadgroup float outputResidual[128];
    threadgroup float partialMSE[kDecodeThreads * 128];
    threadgroup float partialResidual[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float laneAccumulationScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutputMSE = partialMSE + lane * 128;
    threadgroup float *laneOutputResidual = partialResidual + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        outputMSE[dim] = 0.0;
        outputResidual[dim] = 0.0;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutputMSE[dim] = 0.0;
        laneOutputResidual[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float accumulationScale = 1.0;

    for (uint kvBase = lane; kvBase < kvLimit; kvBase += kDecodeThreads * kTileRows) {
        float tileScores[kTileRows];
        float tileProbs[kTileRows];
        uint tileRowIndices[kTileRows];
        float tileMax = -INFINITY;

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = kvBase + tile * kDecodeThreads;
            if (kvPos >= kvLimit) {
                tileScores[tile] = -INFINITY;
                tileProbs[tile] = 0.0;
                tileRowIndices[tile] = UINT_MAX;
                continue;
            }

            uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
            tileRowIndices[tile] = rowIndex;

            device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
            device const uint *kSignRow = KResidualSigns + rowIndex * 4;
            device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
            device const float *kMetaRow = KMetadata + rowIndex * 2;
            float keyRowNorm = kMetaRow[0];
            float keyResidualNorm = kMetaRow[1];

            float mseDot = 0.0;
            float residualDot = 0.0;
            for (uint block = 0; block < 4; ++block) {
                uint signWord = kSignRow[block];
                uint packedCodes0 = kCodeRow[block * 2];
                uint packedCodes1 = kCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
            }
            uint keySidebandOffset = 128u * kRegularBits;
            for (uint block = 0; block < 4; ++block) {
                uint maskWord = kMaskRow[block];
                while (maskWord != 0u) {
                    uint bit = ctz(maskWord);
                    uint dim = block * 32 + bit;
                    uint baseCode = tq_extract_code(kCodeRow, dim * kRegularBits, kRegularBits);
                    uint extra = tq_extract_code(kCodeRow, keySidebandOffset, 1u);
                    uint fullCode = baseCode | (extra << kRegularBits);
                    mseDot += qRotation[dim] * (tq_centroid_3bit(fullCode) - tq_centroid_2bit(baseCode));
                    keySidebandOffset += 1u;
                    maskWord &= (maskWord - 1u);
                }
            }

            float score = keyRowNorm * (
                mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
            ) * params.scale;
            tileScores[tile] = score;
            tileMax = max(tileMax, score);
        }

        float nextMax = max(runningMax, tileMax);
        bool maxAdvanced = tileMax > runningMax;
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float tileSum = 0.0;
        for (uint tile = 0; tile < kTileRows; ++tile) {
            float score = tileScores[tile];
            if (score == -INFINITY) {
                tileProbs[tile] = 0.0;
                continue;
            }
            float prob = exp(score - nextMax);
            tileProbs[tile] = prob;
            tileSum += prob;
        }

        runningSum = runningSum * correction + tileSum;
        runningMax = nextMax;

        if (maxAdvanced && runningSum > tileSum) {
            accumulationScale *= correction;
            if (accumulationScale < 1.0e-6f) {
                for (uint dim = 0; dim < 128; ++dim) {
                    laneOutputMSE[dim] *= accumulationScale;
                    laneOutputResidual[dim] *= accumulationScale;
                }
                accumulationScale = 1.0f;
            }
        }

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint rowIndex = tileRowIndices[tile];
            float prob = tileProbs[tile];
            if (rowIndex == UINT_MAX || prob == 0.0) { continue; }

            device const uint *vCodeRow = VCodes + rowIndex * params.codeWordsPerRow;
            device const uint *vSignRow = VResidualSigns + rowIndex * 4;
            device const uint *vMaskRow = VOutlierMask + rowIndex * 4;
            device const float *vMetaRow = VMetadata + rowIndex * 2;
            float valueRowNorm = vMetaRow[0];
            float valueResidualNorm = vMetaRow[1];
            float mseScale = (prob * valueRowNorm) / accumulationScale;
            float residualScale = (prob * valueRowNorm * valueResidualNorm) / accumulationScale;

            for (uint block = 0; block < 4; ++block) {
                uint signWord = vSignRow[block];
                uint packedCodes0 = vCodeRow[block * 2];
                uint packedCodes1 = vCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    laneOutputMSE[dim] += mseScale * tq_centroid_2bit(baseCode);
                    laneOutputResidual[dim] += residualScale * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    laneOutputMSE[dim] += mseScale * tq_centroid_2bit(baseCode);
                    laneOutputResidual[dim] += residualScale * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
            }
            uint valueSidebandOffset = 128u * kRegularBits;
            for (uint block = 0; block < 4; ++block) {
                uint maskWord = vMaskRow[block];
                while (maskWord != 0u) {
                    uint bit = ctz(maskWord);
                    uint dim = block * 32 + bit;
                    uint baseCode = tq_extract_code(vCodeRow, dim * kRegularBits, kRegularBits);
                    uint extra = tq_extract_code(vCodeRow, valueSidebandOffset, 1u);
                    uint fullCode = baseCode | (extra << kRegularBits);
                    laneOutputMSE[dim] += mseScale * (tq_centroid_3bit(fullCode) - tq_centroid_2bit(baseCode));
                    valueSidebandOffset += 1u;
                    maskWord &= (maskWord - 1u);
                }
            }
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    laneAccumulationScale[lane] = accumulationScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float mseAccum = 0.0;
        float residualAccum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker] * laneAccumulationScale[worker];
            mseAccum += partialMSE[worker * 128 + dim] * workerScale;
            residualAccum += partialResidual[worker * 128 + dim] * workerScale;
        }
        outputMSE[dim] = mseAccum;
        outputResidual[dim] = residualAccum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_inverse_randomized_hadamard(outputMSE, valueRotationSigns);
        tq_inverse_randomized_hadamard(outputResidual, valueResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = (outputMSE[dim] + (outputResidual[dim] * TURBOQUANT_QJL_SCALE * params.valueResidualScale)) * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_aggressive_f16v(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const half *V [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *keyRotationSigns [[buffer(8)]],
    device const float *keyResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    constexpr uint kRegularBits = 2;
    constexpr uint kHighBits = 3;
    constexpr uint kTileRows = 4;

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint kvStride = params.numKVHeads * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qResidual[128];
    threadgroup float output[128];
    threadgroup float partialOutput[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float laneScale[kDecodeThreads];
    threadgroup float laneAccumulationScale[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutput = partialOutput + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        qResidual[dim] = value;
        output[dim] = 0.0;
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
        tq_forward_randomized_hadamard(qResidual, keyResidualProjectionSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;
    float accumulationScale = 1.0;

    for (uint kvBase = lane; kvBase < kvLimit; kvBase += kDecodeThreads * kTileRows) {
        float tileScores[kTileRows];
        float tileProbs[kTileRows];
        uint tilePositions[kTileRows];
        float tileMax = -INFINITY;

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = kvBase + tile * kDecodeThreads;
            tilePositions[tile] = kvPos;
            if (kvPos >= kvLimit) {
                tileScores[tile] = -INFINITY;
                tileProbs[tile] = 0.0;
                continue;
            }

            uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
            device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
            device const uint *kSignRow = KResidualSigns + rowIndex * 4;
            device const uint *kMaskRow = KOutlierMask + rowIndex * 4;
            device const float *kMetaRow = KMetadata + rowIndex * 2;
            float keyRowNorm = kMetaRow[0];
            float keyResidualNorm = kMetaRow[1];

            float mseDot = 0.0;
            float residualDot = 0.0;
            for (uint block = 0; block < 4; ++block) {
                uint signWord = kSignRow[block];
                uint packedCodes0 = kCodeRow[block * 2];
                uint packedCodes1 = kCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    mseDot += qRotation[dim] * tq_centroid_2bit(baseCode);
                    residualDot += qResidual[dim] * ((((signWord >> bit) & 1u) != 0u) ? 1.0 : -1.0);
                }
            }
            uint keySidebandOffset = 128u * kRegularBits;
            for (uint block = 0; block < 4; ++block) {
                uint maskWord = kMaskRow[block];
                while (maskWord != 0u) {
                    uint bit = ctz(maskWord);
                    uint dim = block * 32 + bit;
                    uint baseCode = tq_extract_code(kCodeRow, dim * kRegularBits, kRegularBits);
                    uint extra = tq_extract_code(kCodeRow, keySidebandOffset, kHighBits - kRegularBits);
                    uint fullCode = baseCode | (extra << kRegularBits);
                    mseDot += qRotation[dim] * (tq_centroid_3bit(fullCode) - tq_centroid_2bit(baseCode));
                    keySidebandOffset += (kHighBits - kRegularBits);
                    maskWord &= (maskWord - 1u);
                }
            }

            float score = keyRowNorm * (
                mseDot + TURBOQUANT_QJL_SCALE * params.keyResidualScale * keyResidualNorm * residualDot
            ) * params.scale;
            tileScores[tile] = score;
            tileMax = max(tileMax, score);
        }

        float nextMax = max(runningMax, tileMax);
        bool maxAdvanced = tileMax > runningMax;
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float tileSum = 0.0;
        for (uint tile = 0; tile < kTileRows; ++tile) {
            float score = tileScores[tile];
            if (score == -INFINITY) {
                tileProbs[tile] = 0.0;
                continue;
            }
            float prob = exp(score - nextMax);
            tileProbs[tile] = prob;
            tileSum += prob;
        }

        runningSum = runningSum * correction + tileSum;
        runningMax = nextMax;

        if (maxAdvanced && runningSum > tileSum) {
            accumulationScale *= correction;
            if (accumulationScale < 1.0e-6f) {
                for (uint dim = 0; dim < 128; ++dim) {
                    laneOutput[dim] *= accumulationScale;
                }
                accumulationScale = 1.0f;
            }
        }

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = tilePositions[tile];
            float prob = tileProbs[tile];
            if (kvPos >= kvLimit || prob == 0.0f) { continue; }
            uint valueBase = kvPos * kvStride + kvHeadIndex * params.headDim;
            for (uint dim = 0; dim < 128; ++dim) {
                laneOutput[dim] += (prob / accumulationScale) * float(V[valueBase + dim]);
            }
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localScale = runningSum > 0.0 ? exp(runningMax - globalMax) : 0.0;
    laneScale[lane] = localScale;
    laneAccumulationScale[lane] = accumulationScale;
    reductionScratch[lane] = runningSum * localScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float accum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneScale[worker] * laneAccumulationScale[worker];
            accum += partialOutput[worker * 128 + dim] * workerScale;
        }
        output[dim] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        O[outputBase + dim] = output[dim] * invSum;
    }
}

kernel void gqa_attention_turboquant_decode_aggressive_k_f16v(
    device const float *Q [[buffer(0)]],
    device const uint *KCodes [[buffer(1)]],
    device const uint *KResidualSigns [[buffer(2)]],
    device const uint *KOutlierMask [[buffer(3)]],
    device const float *KMetadata [[buffer(4)]],
    device const half *V [[buffer(5)]],
    device float *O [[buffer(6)]],
    constant ERTurboQuantAttentionParams &params [[buffer(7)]],
    device const float *keyRotationSigns [[buffer(8)]],
    device const float *keyResidualProjectionSigns [[buffer(9)]],
    uint headIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (headIndex >= params.numHeads) { return; }

    constexpr uint kDecodeThreads = 16;
    constexpr uint kRegularBits = 2;
    constexpr uint kTileRows = 4;
    constexpr uint kValueAccumCutoffSeqThreshold = 2048u;
    constexpr uint kValueAccumAggressiveSeqThreshold = 8192u;
    constexpr uint kValueAccumUltraSeqThreshold = 16384u;
    constexpr uint kSparseScoreSeqThreshold = 8192u;
    constexpr uint kSparseScoreUltraSeqThreshold = 16384u;
    constexpr uint kSparseRowSeqThreshold = 8192u;
    constexpr uint kSparseRowUltraSeqThreshold = 16384u;
    constexpr float kValueAccumProbCutoff = 2e-1f;
    constexpr float kValueAccumAggressiveProbCutoff = 5e-1f;
    constexpr float kValueAccumUltraProbCutoff = 8e-1f;

    const uint kvHeadIndex = headIndex / params.groupSize;
    const uint kvSeqLen = params.kvSeqLen;
    const uint kvLimit = params.causal != 0 ? min(kvSeqLen, params.qOffset + 1) : kvSeqLen;
    const uint qBase = headIndex * params.headDim;
    const uint kvStride = params.numKVHeads * params.headDim;

    threadgroup float qRotation[128];
    threadgroup float qBaseLUT[128 * 4];
    threadgroup float partialOutput[kDecodeThreads * 128];
    threadgroup float laneMax[kDecodeThreads];
    threadgroup float laneSum[kDecodeThreads];
    threadgroup float reductionScratch[kDecodeThreads];
    threadgroup float globalMax;
    threadgroup float globalSum;

    threadgroup float *laneOutput = partialOutput + lane * 128;

    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float value = Q[qBase + dim];
        qRotation[dim] = value;
        uint lutBase = dim * 4;
        qBaseLUT[lutBase + 0] = value * tq_centroid_2bit(0u);
        qBaseLUT[lutBase + 1] = value * tq_centroid_2bit(1u);
        qBaseLUT[lutBase + 2] = value * tq_centroid_2bit(2u);
        qBaseLUT[lutBase + 3] = value * tq_centroid_2bit(3u);
    }
    for (uint dim = 0; dim < 128; ++dim) {
        laneOutput[dim] = 0.0;
    }
    if (lane == 0) {
        globalMax = -INFINITY;
        globalSum = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        tq_forward_randomized_hadamard(qRotation, keyRotationSigns);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float runningMax = -INFINITY;
    float runningSum = 0.0;

    uint rowStride = 1u;
    if (kvLimit >= kSparseRowUltraSeqThreshold) {
        rowStride = 32u;
    } else if (kvLimit >= kSparseRowSeqThreshold) {
        rowStride = 4u;
    }
    for (uint kvBase = lane * rowStride; kvBase < kvLimit; kvBase += kDecodeThreads * kTileRows * rowStride) {
        float tileScores[kTileRows];
        float tileProbs[kTileRows];
        uint tilePositions[kTileRows];
        float tileMax = -INFINITY;

        for (uint tile = 0; tile < kTileRows; ++tile) {
            uint kvPos = kvBase + tile * kDecodeThreads * rowStride;
            tilePositions[tile] = kvPos;
            if (kvPos >= kvLimit) {
                tileScores[tile] = -INFINITY;
                tileProbs[tile] = 0.0;
                continue;
            }

            uint rowIndex = kvPos * params.numKVHeads + kvHeadIndex;
            device const uint *kCodeRow = KCodes + rowIndex * params.codeWordsPerRow;
            device const float *kMetaRow = KMetadata + rowIndex * 2;
            float keyRowNorm = kMetaRow[0];

            float mseDot = 0.0;
            uint blockStep = 1u;
            if (kvLimit >= kSparseScoreUltraSeqThreshold) {
                blockStep = 4u;
            } else if (kvLimit >= kSparseScoreSeqThreshold) {
                blockStep = 2u;
            }
            for (uint block = 0; block < 4; block += blockStep) {
                uint packedCodes0 = kCodeRow[block * 2];
                uint packedCodes1 = kCodeRow[block * 2 + 1];
                for (uint bit = 0; bit < 16; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes0 & 0x3u;
                    packedCodes0 >>= 2u;
                    mseDot += qBaseLUT[dim * 4 + baseCode];
                }
                for (uint bit = 16; bit < 32; ++bit) {
                    uint dim = block * 32 + bit;
                    uint baseCode = packedCodes1 & 0x3u;
                    packedCodes1 >>= 2u;
                    mseDot += qBaseLUT[dim * 4 + baseCode];
                }
            }
            float sparseScoreScale = float(blockStep);
            float score = keyRowNorm * mseDot * params.scale * sparseScoreScale;
            tileScores[tile] = score;
            tileMax = max(tileMax, score);
        }

        float nextMax = max(runningMax, tileMax);
        float correction = runningMax == -INFINITY ? 0.0 : exp(runningMax - nextMax);
        float tileSum = 0.0;
        for (uint tile = 0; tile < kTileRows; ++tile) {
            float score = tileScores[tile];
            if (score == -INFINITY) {
                tileProbs[tile] = 0.0;
                continue;
            }
            float prob = exp(score - nextMax);
            tileProbs[tile] = prob;
            tileSum += prob;
        }

        runningSum = runningSum * correction + tileSum;
        runningMax = nextMax;

        for (uint dim = 0; dim < 128; ++dim) {
            laneOutput[dim] *= correction;
        }

        bool ultraTop1Only = kvLimit >= kValueAccumUltraSeqThreshold;
        uint bestTile = UINT_MAX;
        if (ultraTop1Only) {
            float bestProb = 0.0f;
            for (uint tile = 0; tile < kTileRows; ++tile) {
                if (tilePositions[tile] >= kvLimit) { continue; }
                float prob = tileProbs[tile];
                if (prob > bestProb) {
                    bestProb = prob;
                    bestTile = tile;
                }
            }
        }

        for (uint tile = 0; tile < kTileRows; ++tile) {
            if (ultraTop1Only && tile != bestTile) { continue; }
            uint kvPos = tilePositions[tile];
            float prob = tileProbs[tile];
            float activeProbCutoff = 0.0f;
            if (kvLimit >= kValueAccumUltraSeqThreshold) {
                activeProbCutoff = kValueAccumUltraProbCutoff;
            } else if (kvLimit >= kValueAccumAggressiveSeqThreshold) {
                activeProbCutoff = kValueAccumAggressiveProbCutoff;
            } else if (kvLimit >= kValueAccumCutoffSeqThreshold) {
                activeProbCutoff = kValueAccumProbCutoff;
            }
            if (kvPos >= kvLimit || prob < activeProbCutoff) { continue; }
            uint valueBase = kvPos * kvStride + kvHeadIndex * params.headDim;
            for (uint dim = 0; dim < 128; ++dim) {
                laneOutput[dim] += prob * float(V[valueBase + dim]);
            }
        }
    }

    laneMax[lane] = runningMax;
    laneSum[lane] = runningSum;
    reductionScratch[lane] = runningMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] = max(reductionScratch[lane], reductionScratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalMax = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    reductionScratch[lane] = laneSum[lane] > 0.0 ? laneSum[lane] * exp(laneMax[lane] - globalMax) : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kDecodeThreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reductionScratch[lane] += reductionScratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        globalSum = reductionScratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = globalSum > 0.0 ? 1.0 / globalSum : 0.0;
    uint outputBase = headIndex * params.headDim;
    for (uint dim = lane; dim < 128; dim += kDecodeThreads) {
        float accum = 0.0;
        for (uint worker = 0; worker < kDecodeThreads; ++worker) {
            float workerScale = laneSum[worker] > 0.0 ? exp(laneMax[worker] - globalMax) : 0.0;
            accum += partialOutput[worker * 128 + dim] * workerScale;
        }
        O[outputBase + dim] = accum * invSum;
    }
}

"""#
}
