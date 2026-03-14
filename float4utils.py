#!/usr/bin/env python3
# nvfp4_block16_bench.py
#
# Fully-dynamic NVFP4 (E2M1 FP4, packed x2) matmul with blockwise 1x16 scaling.
# Includes optional fused RMSNorm along the last dimension before LHS quantization.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType

from gn_kernels import (
    FP4_DTYPE,
    cutlass_fp8_mm,
    cutlass_int4_mm,
    cutlass_mxfp4_mm,
    cutlass_nvfp4_mm,
    cutlass_nvfp4_mm_relu,
    cutlass_nvfp4_mm_relu_2,
    cutlass_row_scaled_fp8_mm,
    cutlass_row_scaled_int4_mm,
    triton_block2d_scaled_mm,
    triton_mm,
)

import cupy as cp

try:
    import triton
    import triton.testing
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

# -----------------------------
# Helpers & Checks
# -----------------------------
def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

def require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

# -----------------------------
# DType checks
# -----------------------------
require(hasattr(torch, "float4_e2m1fn_x2"), "Your torch build lacks torch.float4_e2m1fn_x2.")
require(hasattr(torch, "float8_e4m3fn"), "Your torch build lacks torch.float8_e4m3fn.")
FP4_DTYPE = torch.float4_e2m1fn_x2
FP8_SCALE_DTYPE = torch.float8_e4m3fn

# ============================================================
# 1) NVRTC Kernels (Standard & RMSNorm-Fused)
# ============================================================

_KERNEL_NVFP4_FUSED = r"""
extern "C" {
__device__ __forceinline__ unsigned char fp8_e4m3_from_f32(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}
__device__ __forceinline__ float f32_from_fp8_e4m3(unsigned char b) {
    unsigned short e4m3x2 = (unsigned short)b | ((unsigned short)b << 8);
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(e4m3x2));
    unsigned short h0 = (unsigned short)(f16x2 & 0xFFFF);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(h0));
    return out;
}

__global__ void quant_nvfp4_fused_opt(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_u8,
    unsigned char* __restrict__ s_u8,
    int num_blocks16, int cols, int pad_cols
){
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks16) return;

    int elem_base = blk << 4; 

    const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
    uint4 v0 = in4[0], v1 = in4[1];

    unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float f[16];
    float amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned short lo = (unsigned short)(w[i] & 0xffff);
        unsigned short hi = (unsigned short)(w[i] >> 16);
        float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
        float a1 = __int_as_float((int)(((unsigned int)hi) << 16));
        f[2*i+0] = a0;
        f[2*i+1] = a1;
        amax = fmaxf(amax, fabsf(a0));
        amax = fmaxf(amax, fabsf(a1));
    }

    unsigned char scale_b = 0;
    float inv_scale = 1.0f;
    if (amax > 0.0f) {
        float scale_f = amax * (1.0f / 6.0f);
        scale_b = fp8_e4m3_from_f32(scale_f);
        float scale_fq = f32_from_fp8_e4m3(scale_b);
        inv_scale = (scale_fq > 0.0f) ? (1.0f / scale_fq) : 1.0f;
    }

    int r = blk / cols, c = blk % cols;
    int r_blk = r >> 7, c_blk = c >> 2, r_in = r & 127, c_in = c & 3;
    int tile_idx = r_blk * (pad_cols >> 2) + c_blk;
    int new_row = r_in & 31, new_col = ((r_in >> 5) << 2) + c_in;
    int swizzled_blk = (tile_idx << 9) + (new_row << 4) + new_col;
    s_u8[swizzled_blk] = scale_b;

    unsigned int out[2] = {0, 0};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float ax0 = fabsf(f[2*i] * inv_scale);
        float ax1 = fabsf(f[2*i+1] * inv_scale);
        if (ax0 > 6.0f) ax0 = 6.0f;
        if (ax1 > 6.0f) ax1 = 6.0f;

        int sign0 = f[2*i] < 0.0f;
        int sign1 = f[2*i+1] < 0.0f;

        unsigned int code0 = (!(ax0 < 0.25f)) + (!(ax0 < 0.75f)) + (!(ax0 < 1.25f)) + (!(ax0 < 1.75f)) + (!(ax0 < 2.5f)) + (!(ax0 < 3.5f)) + (!(ax0 < 5.0f));
        unsigned int code1 = (!(ax1 < 0.25f)) + (!(ax1 < 0.75f)) + (!(ax1 < 1.25f)) + (!(ax1 < 1.75f)) + (!(ax1 < 2.5f)) + (!(ax1 < 3.5f)) + (!(ax1 < 5.0f));

        unsigned char n0 = (sign0 << 3) | code0;
        unsigned char n1 = (sign1 << 3) | code1;
        unsigned char packed = (n0 & 0xF) | ((n1 & 0xF) << 4);

        if (i < 4) out[0] |= ((unsigned int)packed) << (i * 8);
        else       out[1] |= ((unsigned int)packed) << ((i - 4) * 8);
    }

    ((uint2*)(q_u8 + (blk << 3)))[0] = make_uint2(out[0], out[1]);
}
}
"""

_KERNEL_RMSNORM_NVFP4_SMEM = r"""
extern "C" {
__device__ __forceinline__ unsigned char fp8_e4m3_from_f32(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}
__device__ __forceinline__ float f32_from_fp8_e4m3(unsigned char b) {
    unsigned short e4m3x2 = (unsigned short)b | ((unsigned short)b << 8);
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(e4m3x2));
    unsigned short h0 = (unsigned short)(f16x2 & 0xFFFF);
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(h0));
    return out;
}

__global__ void rmsnorm_quant_nvfp4_smem_opt(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_u8,
    unsigned char* __restrict__ s_u8,
    float* __restrict__ inv_rms_out, // NEW: Output pointer for 1/RMS
    int R, int K, int pad_cols, float epsilon
){
    extern __shared__ unsigned short smem_row[];
    int r = blockIdx.x;
    if (r >= R) return;

    int tid = threadIdx.x;
    int cols = K / 16; 

    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        int elem_base = r * K + (c * 16);
        const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
        uint4 v0 = in4[0], v1 = in4[1];

        uint4* smem_out = (uint4*)(smem_row + (c * 16));
        smem_out[0] = v0;
        smem_out[1] = v1;

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);
            float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16));
            sum_sq += a0 * a0 + a1 * a1;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    static __shared__ float shared_sum[32]; 
    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) shared_sum[wid] = sum_sq;
    __syncthreads(); 

    float block_sum = (tid < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (tid == 0) {
            float irms = rsqrtf((block_sum / (float)K) + epsilon);
            shared_sum[0] = irms;
            inv_rms_out[r] = irms; // NEW: Write to Global Memory
        }
    }
    __syncthreads(); 

    float inv_rms = shared_sum[0];

    // ... (Phase 2 remains exactly the same as before) ...
    for (int c = tid; c < cols; c += blockDim.x) {
        int blk = r * cols + c; 
        const uint4* in4 = (const uint4*)(smem_row + (c * 16));
        uint4 v0 = in4[0], v1 = in4[1];

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float f[16];
        float amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);
            
            float a0 = __int_as_float((int)(((unsigned int)lo) << 16)) * inv_rms;
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16)) * inv_rms;
            
            f[2*i+0] = a0;
            f[2*i+1] = a1;
            amax = fmaxf(amax, fabsf(a0));
            amax = fmaxf(amax, fabsf(a1));
        }

        unsigned char scale_b = 0;
        float inv_scale = 1.0f;
        if (amax > 0.0f) {
            float scale_f = amax * (1.0f / 6.0f);
            scale_b = fp8_e4m3_from_f32(scale_f);
            float scale_fq = f32_from_fp8_e4m3(scale_b);
            inv_scale = (scale_fq > 0.0f) ? (1.0f / scale_fq) : 1.0f;
        }

        int r_blk = r >> 7, c_blk = c >> 2, r_in = r & 127, c_in = c & 3;
        int tile_idx = r_blk * (pad_cols >> 2) + c_blk;
        int new_row = r_in & 31, new_col = ((r_in >> 5) << 2) + c_in;
        int swizzled_blk = (tile_idx << 9) + (new_row << 4) + new_col;
        s_u8[swizzled_blk] = scale_b;

        unsigned int out[2] = {0, 0};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float ax0 = fabsf(f[2*i] * inv_scale);
            float ax1 = fabsf(f[2*i+1] * inv_scale);
            if (ax0 > 6.0f) ax0 = 6.0f;
            if (ax1 > 6.0f) ax1 = 6.0f;

            int sign0 = f[2*i] < 0.0f;
            int sign1 = f[2*i+1] < 0.0f;

            unsigned int code0 = (!(ax0 < 0.25f)) + (!(ax0 < 0.75f)) + (!(ax0 < 1.25f)) + (!(ax0 < 1.75f)) + (!(ax0 < 2.5f)) + (!(ax0 < 3.5f)) + (!(ax0 < 5.0f));
            unsigned int code1 = (!(ax1 < 0.25f)) + (!(ax1 < 0.75f)) + (!(ax1 < 1.25f)) + (!(ax1 < 1.75f)) + (!(ax1 < 2.5f)) + (!(ax1 < 3.5f)) + (!(ax1 < 5.0f));

            unsigned char n0 = (sign0 << 3) | code0;
            unsigned char n1 = (sign1 << 3) | code1;
            unsigned char packed = (n0 & 0xF) | ((n1 & 0xF) << 4);

            if (i < 4) out[0] |= ((unsigned int)packed) << (i * 8);
            else       out[1] |= ((unsigned int)packed) << ((i - 4) * 8);
        }

        ((uint2*)(q_u8 + (blk << 3)))[0] = make_uint2(out[0], out[1]);
    }
}
}
"""

@dataclass
class _NVFP4QuantKernels:
    standard: cp.RawKernel
    rmsnorm_smem: cp.RawKernel

    @staticmethod
    def build() -> "_NVFP4QuantKernels":
        mod_std = cp.RawModule(code=_KERNEL_NVFP4_FUSED, options=("--std=c++17",))
        fn_std = mod_std.get_function("quant_nvfp4_fused_opt")
        
        mod_rms = cp.RawModule(code=_KERNEL_RMSNORM_NVFP4_SMEM, options=("--std=c++17",))
        fn_rms = mod_rms.get_function("rmsnorm_quant_nvfp4_smem_opt")
        fn_rms.max_dynamic_shared_size_bytes = 98304 
        
        return _NVFP4QuantKernels(standard=fn_std, rmsnorm_smem=fn_rms)

_KERNELS: Optional[_NVFP4QuantKernels] = None

def _get_kernels() -> _NVFP4QuantKernels:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _NVFP4QuantKernels.build()
    return _KERNELS

# ============================================================
# 2) Python wrappers: quant + pad + swizzle
# ============================================================

def quant_nvfp4_fused(x_bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    require(x_bf16.is_cuda and x_bf16.dtype == torch.bfloat16,
            "quant_nvfp4_fused expects a CUDA BF16 tensor.")

    x = x_bf16.contiguous()
    shape = x.shape
    K = shape[-1]
    batch_dims = shape[:-1]
    
    # Calculate logical R (rows) dynamically for multidimensional tensors
    R = math.prod(batch_dims) if batch_dims else 1
    require(K % 16 == 0, "K must be divisible by 16 for NVFP4 1x16 scaling.")

    num_elements = x.numel()
    cols = K // 16
    
    pad_R = round_up(R, 128)
    pad_cols = round_up(cols, 4)

    # Allocate directly to target dtype and ND shapes, eliminating .view()
    q = torch.empty((*batch_dims, K // 2), device=x.device, dtype=FP4_DTYPE)
    s = torch.zeros((pad_R, pad_cols), device=x.device, dtype=FP8_SCALE_DTYPE)

    threads = 256
    blocks16 = num_elements // 16
    grid = ((blocks16 + threads - 1) // threads,)

    kern = _get_kernels().standard
    kern(grid, (threads,), (x.data_ptr(), q.data_ptr(), s.data_ptr(), blocks16, cols, pad_cols))

    return q, s

def rmsnorm_quant_nvfp4_fused(x_bf16: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(x_bf16.is_cuda and x_bf16.dtype == torch.bfloat16,
            "rmsnorm_quant_nvfp4_fused expects a CUDA BF16 tensor.")

    x = x_bf16.contiguous()
    shape = x.shape
    K = shape[-1]
    batch_dims = shape[:-1]
    
    R = math.prod(batch_dims) if batch_dims else 1
    require(K % 16 == 0, "K must be divisible by 16 for NVFP4 1x16 scaling.")

    num_elements = x.numel()
    cols = K // 16
    
    pad_R = round_up(R, 128)
    pad_cols = round_up(cols, 4)

    # Allocate directly to target dtype and ND shapes, eliminating .view()
    q = torch.empty((*batch_dims, K // 2), device=x.device, dtype=FP4_DTYPE)
    s = torch.zeros((pad_R, pad_cols), device=x.device, dtype=FP8_SCALE_DTYPE)
    
    # ND support for 1/RMS out
    inv_rms_out = torch.empty(batch_dims if batch_dims else (1,), device=x.device, dtype=torch.float32)

    threads = 256
    grid = (R,)
    smem_bytes = K * 2 

    kern = _get_kernels().rmsnorm_smem
    kern(
        grid, (threads,), 
        (
            x.data_ptr(), 
            q.data_ptr(), 
            s.data_ptr(), 
            inv_rms_out.data_ptr(), 
            R, K, pad_cols, cp.float32(epsilon)
        ),
        shared_mem=smem_bytes
    )

    return q, s, inv_rms_out

# ============================================================
# 3) NVFP4 matmul via torch.scaled_mm & cutlass
# ============================================================

@torch.no_grad()
def nvfp4mm_torch(
    A_bf16: torch.Tensor, 
    B_bf16: torch.Tensor, 
    *, 
    out_dtype: torch.dtype = torch.bfloat16,
    apply_rmsnorm_lhs: bool = False,
    rms_eps: float = 1e-6
) -> torch.Tensor:
    """ Fully dynamic NVFP4 GEMM: C = A @ B^T """
    require(A_bf16.is_cuda and B_bf16.is_cuda, "nvfp4mm requires CUDA tensors.")
    
    if apply_rmsnorm_lhs:
        a_fp4, s_a_swz, _ = rmsnorm_quant_nvfp4_fused(A_bf16, epsilon=rms_eps)
    else:
        a_fp4, s_a_swz = quant_nvfp4_fused(A_bf16) 
        
    b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 

    # Swapped .t() for .mT to robustly handle >2D batches
    C = F.scaled_mm(
        a_fp4,
        b_fp4.mT,
        s_a_swz, ScalingType.BlockWise1x16,
        s_b_swz, ScalingType.BlockWise1x16,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=out_dtype,
        contraction_dim=(1, 0),
    )
    return C

@torch.no_grad()
@torch._dynamo.disable
def nvfp4mm(
    A_bf16: torch.Tensor, 
    B_bf16: torch.Tensor, 
    *, 
    out_dtype: torch.dtype = torch.bfloat16,
    apply_rmsnorm_lhs: bool = False,
    rms_eps: float = 1e-6
) -> torch.Tensor:
    """ Fully dynamic NVFP4 GEMM: C = A @ B^T """
    require(A_bf16.is_cuda and B_bf16.is_cuda, "nvfp4mm requires CUDA tensors.")

    if apply_rmsnorm_lhs:
        a_fp4, s_a_swz, inv_rms = rmsnorm_quant_nvfp4_fused(A_bf16, epsilon=rms_eps)
        b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 
    
        C = cutlass_nvfp4_mm(a_fp4, b_fp4.mT, s_a_swz, s_b_swz, torch.tensor(1.0, device='cuda'))
        return C, inv_rms
    else:
        a_fp4, s_a_swz = quant_nvfp4_fused(A_bf16) 
        b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 
    
        C = cutlass_nvfp4_mm(a_fp4, b_fp4.mT, s_a_swz, s_b_swz, torch.tensor(1.0, device='cuda'))
        return C

@torch.no_grad()
@torch._dynamo.disable
def nvfp4mm_relu(
    A_bf16: torch.Tensor, 
    B_bf16: torch.Tensor, 
    *, 
    out_dtype: torch.dtype = torch.bfloat16,
    apply_rmsnorm_lhs: bool = False,
    rms_eps: float = 1e-6
) -> torch.Tensor:
    """ Fully dynamic NVFP4 GEMM: C = Relu(A @ B^T) """
    require(A_bf16.is_cuda and B_bf16.is_cuda, "nvfp4mm requires CUDA tensors.")

    if apply_rmsnorm_lhs:
        a_fp4, s_a_swz, inv_rms = rmsnorm_quant_nvfp4_fused(A_bf16, epsilon=rms_eps)
        b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 
    
        C = cutlass_nvfp4_mm_relu(a_fp4, b_fp4.mT, s_a_swz, s_b_swz, torch.tensor(1.0, device='cuda'))
        return C, inv_rms
    else:
        a_fp4, s_a_swz = quant_nvfp4_fused(A_bf16) 
        b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 
    
        C = cutlass_nvfp4_mm_relu(a_fp4, b_fp4.mT, s_a_swz, s_b_swz, torch.tensor(1.0, device='cuda'))
        return C

@torch.no_grad()
@torch._dynamo.disable
def nvfp4mm_relu_2(
    A_bf16: torch.Tensor, 
    B_bf16: torch.Tensor, 
    *, 
    out_dtype: torch.dtype = torch.bfloat16,
    apply_rmsnorm_lhs: bool = False,
    rms_eps: float = 1e-6
) -> torch.Tensor:
    """ Fully dynamic NVFP4 GEMM: C = Relu^2(A @ B^T) """
    require(A_bf16.is_cuda and B_bf16.is_cuda, "nvfp4mm requires CUDA tensors.")

    if apply_rmsnorm_lhs:
        a_fp4, s_a_swz, inv_rms = rmsnorm_quant_nvfp4_fused(A_bf16, epsilon=rms_eps)
        b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 
    
        C = cutlass_nvfp4_mm_relu_2(a_fp4, b_fp4.mT, s_a_swz, s_b_swz, torch.tensor(1.0, device='cuda'))
        return C, inv_rms
    else:
        a_fp4, s_a_swz = quant_nvfp4_fused(A_bf16) 
        b_fp4, s_b_swz = quant_nvfp4_fused(B_bf16) 
    
        C = cutlass_nvfp4_mm_relu_2(a_fp4, b_fp4.mT, s_a_swz, s_b_swz, torch.tensor(1.0, device='cuda'))
        return C

# ============================================================
# 4) Accuracy + Benchmarks
# ============================================================

@torch.no_grad()
def test_accuracy(M=1024, N=2048, K=4096):
    print(f"\n--- NVFP4 Accuracy (M={M}, N={N}, K={K}) ---")
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C_ref = A @ B.t()
    C = nvfp4mm(A, B, out_dtype=torch.bfloat16, apply_rmsnorm_lhs=False)

    diff = (C_ref - C).float().abs()
    print(f"MSE:    {(diff ** 2).mean().item():.6e}")
    print(f"Max Abs:  {diff.max().item():.6e}")
    print(f"Mean Rel: {(diff.mean() / C_ref.float().abs().mean()).item():.6%}")

def _bytes_nvfp4_quant_fused(R: int, K: int) -> float:
    pad_R = round_up(R, 128)
    pad_cols = round_up(K // 16, 4)
    # Traffic: Read bf16(2), Write fp4(0.5), Write scales(1)
    return (R * K * 2.0) + (R * K * 0.5) + (pad_R * pad_cols * 1.0)

if _HAS_TRITON:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[1024 * i for i in range(4, 33, 4)],
            line_arg="provider",
            line_vals=["quant_A", "quant_B", "rmsnorm_quant_A"],
            line_names=["Quant A (GB/s)", "Quant B (GB/s)", "RMSNorm+Quant A (GB/s)"],
            ylabel="GB/s",
            plot_name="nvfp4-quant-1x16-gbps",
            args={"M": 4096, "N": 8192},
        )
    )
    def bench_quant_gbps(M, N, K, provider):
        quantiles = [0.5, 0.2, 0.8]
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).contiguous()

        if provider == "quant_A":
            fn = lambda: quant_nvfp4_fused(A)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
            gbps = lambda t: _bytes_nvfp4_quant_fused(M, K) / (t * 1e-3) / 1e9
            return gbps(ms), gbps(max_ms), gbps(min_ms)
        elif provider == "quant_B":
            fn = lambda: quant_nvfp4_fused(B)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
            gbps = lambda t: _bytes_nvfp4_quant_fused(N, K) / (t * 1e-3) / 1e9
            return gbps(ms), gbps(max_ms), gbps(min_ms)
        else: # rmsnorm_quant_A
            fn = lambda: rmsnorm_quant_nvfp4_fused(A)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
            gbps = lambda t: _bytes_nvfp4_quant_fused(M, K) / (t * 1e-3) / 1e9
            return gbps(ms), gbps(max_ms), gbps(min_ms)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[128, 256] + [512 * i for i in range(1, 33, 1)],
            line_arg="provider",
            line_vals=["static", "dynamic", "dynamic_rms"],
            line_names=["Static (quant once)", "Dynamic (end-to-end)", "Dynamic w/ RMSNorm LHS"],
            ylabel="TFLOPs",
            plot_name="nvfp4-static-vs-dynamic-tflops",
            args={"M": 4096, "N": 8192},
        )
    )
    def bench_gemm_tflops(M, N, K, provider):
        quantiles = [0.5, 0.2, 0.8]
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).contiguous()
        ones = torch.tensor(1.0, device='cuda')

        if provider == "static":
            a_fp4, s_a = quant_nvfp4_fused(A)
            b_fp4, s_b = quant_nvfp4_fused(B)
            fn = lambda: F.scaled_mm(
                a_fp4, b_fp4.t(),
                s_a, ScalingType.BlockWise1x16,
                s_b, ScalingType.BlockWise1x16,
                swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.bfloat16,
                contraction_dim=(1, 0),
            )
        elif provider == "dynamic": 
            def fn():
                a, sa = quant_nvfp4_fused(A)
                b, sb = quant_nvfp4_fused(B)
                return cutlass_nvfp4_mm(a, b.t(), sa, sb, ones)
        else: # dynamic_rms
            def fn():
                a, sa = rmsnorm_quant_nvfp4_fused(A)
                b, sb = quant_nvfp4_fused(B)
                return cutlass_nvfp4_mm(a, b.t(), sa, sb, ones)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        tflops = lambda t_ms: (2.0 * M * N * K) / (t_ms * 1e-3) / 1e12
        return tflops(ms), tflops(max_ms), tflops(min_ms)

import torch.nn as nn

class NVFP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Forward pass: Computes X @ W^T using fully dynamic NVFP4 quantization.
        """
        # Save tensors needed for the backward pass (stored in their native dtype, e.g., BF16)
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        
        # nvfp4mm expects 2D tensors. Flatten input if it's > 2D (e.g., [B, S, K] -> [B*S, K])
        input_shape = input.shape
        if input.dim() > 2:
            x_2d = input.view(-1, input_shape[-1])
        else:
            x_2d = input
            
        # Execute the NVFP4 GEMM (C = A @ B^T)
        # x_2d is [M, K], weight is [N, K] -> out_2d is [M, N]
        out_2d = nvfp4mm(x_2d, weight, out_dtype=input.dtype)
        
        # Reshape output back to the original batch/sequence dimensions
        if input.dim() > 2:
            out = out_2d.view(*input_shape[:-1], weight.shape[0])
        else:
            out = out_2d
            
        if bias is not None:
            out += bias
            
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: Computes gradients in standard precision (BF16/FP32).
        """
        input, weight = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None
        
        # Flatten tensors to 2D for straightforward matmuls
        if input.dim() > 2:
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            input_2d = input.reshape(-1, input.shape[-1])
        else:
            grad_output_2d = grad_output
            input_2d = input

        # 1. Gradient with respect to input (dX = dY @ W)
        if ctx.needs_input_grad[0]:
            grad_input_2d = grad_output_2d.matmul(weight)
            grad_input = grad_input_2d.view_as(input)
            
        # 2. Gradient with respect to weights (dW = dY^T @ X)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_2d.t().matmul(input_2d)
            
        # 3. Gradient with respect to bias (db = sum(dY))
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)
            
        return grad_input, grad_weight, grad_bias


class NVFP4Linear(nn.Module):
    """
    Drop-in replacement for torch.nn.Linear using NVFP4 for the forward pass.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights are stored in normal precision (BF16) and dynamically quantized in the forward pass
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Standard PyTorch Linear initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return NVFP4LinearFunction.apply(input, self.weight, self.bias)

if __name__ == "__main__":
    # Sanity check
    test_accuracy(M=1024, N=2048, K=4096)

    if not _HAS_TRITON:
        print("\n(triton not installed; skipping benchmarks)")
    else:
        print("\n--- Quantization bandwidth (GB/s) ---")
        bench_quant_gbps.run(print_data=True, save_path=".")

        print("\n--- GEMM performance (TFLOPs) ---")
        bench_gemm_tflops.run(print_data=True, save_path=".")