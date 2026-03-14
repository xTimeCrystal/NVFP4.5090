Install [gn-kernels](https://github.com/gau-nernst/gn-kernels), and replace `gn_kernels/csrc/sm120a/cutlass_mm_fp4.cu` with the version in this repo, then `cd ~/gn-kernels` and `pip install -e . --no-build-isolation`. Example NVFP4 MLP impl:

- ~10% e2e speed up for transformer training loop
- mild loss degradation, memory already optimized far below standard PyTorch ReLU^2 MLP by using in-place scratchpad tensors with built-in checkpointing --> **don't** wrap this implementation using `torch.utils.checkpoint.checkpoint` or it will run MUCH SLOWER with higher memory usage

```python
import torch
import triton
import triton.language as tl
import torch.nn as nn

from float4utils import nvfp4mm, nvfp4mm_relu, nvfp4mm_relu_2

# =====================================================================
# TRITON KERNEL: RMSNorm Backward
# =====================================================================
@triton.jit
def _rmsnorm_bwd_inplace_kernel(
    dX_norm_ptr, X_ptr, inv_rms_ptr,
    stride_dxn_m, stride_dxn_k,
    stride_x_m, stride_x_k,
    K,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Offset pointers to the start of the current row
    dX_norm_row_ptr = dX_norm_ptr + pid * stride_dxn_m
    X_row_ptr = X_ptr + pid * stride_x_m
    
    inv_rms = tl.load(inv_rms_ptr + pid)
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < K
    
    # Calculate pointers for dX_norm so we can load and store to the exact same spots
    dxn_ptrs = dX_norm_row_ptr + cols * stride_dxn_k
    
    dxn = tl.load(dxn_ptrs, mask=mask, other=0.0)
    x = tl.load(X_row_ptr + cols * stride_x_k, mask=mask, other=0.0)
    
    dxn_fp32 = dxn.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    inv_rms_fp32 = inv_rms.to(tl.float32)
    
    dot_prod = tl.sum(dxn_fp32 * x_fp32, axis=0)
    term1 = dxn_fp32 * inv_rms_fp32
    scale_x = (inv_rms_fp32 * inv_rms_fp32 * inv_rms_fp32 * dot_prod) / K
    term2 = x_fp32 * scale_x
    
    dx = term1 - term2
    
    # Store back directly into dX_norm's memory space
    tl.store(dxn_ptrs, dx.to(dxn.dtype), mask=mask)

def triton_rmsnorm_bwd_inplace(dX_norm: torch.Tensor, X: torch.Tensor, inv_rms: torch.Tensor) -> torch.Tensor:
    M, K = X.shape
    
    BLOCK_SIZE = triton.next_power_of_2(K)
    num_warps = 4
    if BLOCK_SIZE >= 4096: num_warps = 8
    if BLOCK_SIZE >= 8192: num_warps = 16
    
    grid = (M,)
    _rmsnorm_bwd_inplace_kernel[grid](
        dX_norm, X, inv_rms,
        dX_norm.stride(0), dX_norm.stride(1),
        X.stride(0), X.stride(1),
        K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    # Return the mutated tensor
    return dX_norm

@triton.jit
def _fused_dz_inplace_kernel(
    dY_ptr, W2_ptr, S_ptr,
    M, N, P,
    stride_dy_m, stride_dy_p,
    stride_w2_p, stride_w2_n,
    stride_s_m, stride_s_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, BLOCK_P)
    
    dY_ptrs = dY_ptr + (offs_m[:, None] * stride_dy_m + offs_p[None, :] * stride_dy_p)
    W2_ptrs = W2_ptr + (offs_p[:, None] * stride_w2_p + offs_n[None, :] * stride_w2_n)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for p in range(0, tl.cdiv(P, BLOCK_P)):
        dy_mask = (offs_m[:, None] < M) & ((offs_p[None, :] + p * BLOCK_P) < P)
        w2_mask = ((offs_p[:, None] + p * BLOCK_P) < P) & (offs_n[None, :] < N)
        
        dy = tl.load(dY_ptrs, mask=dy_mask, other=0.0)
        w2 = tl.load(W2_ptrs, mask=w2_mask, other=0.0)
        acc = tl.dot(dy, w2, acc)
        
        dY_ptrs += BLOCK_P * stride_dy_p
        W2_ptrs += BLOCK_P * stride_w2_p
        
    # Single pointer logic for S
    S_ptrs = S_ptr + (offs_m[:, None] * stride_s_m + offs_n[None, :] * stride_s_n)
    s_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    s_val = tl.load(S_ptrs, mask=s_mask, other=0.0)
    dz = acc.to(s_val.dtype) * 2.0 * s_val
    
    # Store directly back to the same pointer
    tl.store(S_ptrs, dz, mask=s_mask)

def triton_fused_dz_inplace(dY: torch.Tensor, W2: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    M, P = dY.shape
    _, N = S.shape
    
    if M >= 2048 and N >= 2048:
        BLOCK_M, BLOCK_N, BLOCK_P, num_warps, num_stages = 128, 128, 32, 4, 4
    else:
        BLOCK_M, BLOCK_N, BLOCK_P, num_warps, num_stages = 64, 64, 32, 4, 3

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    _fused_dz_inplace_kernel[grid](
        dY, W2, S,
        M, N, P,
        dY.stride(0), dY.stride(1),
        W2.stride(0), W2.stride(1),
        S.stride(0), S.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_P=BLOCK_P,
        GROUP_M=8,
        num_warps=num_warps, num_stages=num_stages
    )
    return S

@triton.jit
def _fused_dw1_kernel(
    dZ_ptr, X_ptr, inv_rms_ptr, dW1_ptr,
    M, N, K,
    stride_dz_m, stride_dz_n,
    stride_x_m, stride_x_k,
    stride_dw1_n, stride_dw1_k,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    GROUP_N: tl.constexpr,
    DTYPE: tl.constexpr  # <--- NEW: Pass dtype explicitly
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    
    num_pid_in_group = GROUP_N * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_k = (pid % num_pid_in_group) // group_size_n
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    
    dz_ptrs = dZ_ptr + (offs_n[:, None] * stride_dz_n + offs_m[None, :] * stride_dz_m)
    x_ptrs = X_ptr + (offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    rms_ptrs = inv_rms_ptr + offs_m
    
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_M)):
        m_mask = (offs_m + m * BLOCK_M) < M
        n_mask = offs_n < N
        k_mask = offs_k < K
        
        dz_mask = n_mask[:, None] & m_mask[None, :]
        x_mask = m_mask[:, None] & k_mask[None, :]
        
        dz = tl.load(dz_ptrs, mask=dz_mask, other=0.0)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        rms = tl.load(rms_ptrs, mask=m_mask, other=0.0)
        
        # USE DTYPE HERE
        x_scaled = x * rms[:, None].to(DTYPE)
        
        acc = tl.dot(dz, x_scaled, acc)
        
        dz_ptrs += BLOCK_M * stride_dz_m
        x_ptrs += BLOCK_M * stride_x_m
        rms_ptrs += BLOCK_M
        
    dw1_ptrs = dW1_ptr + (offs_n[:, None] * stride_dw1_n + offs_k[None, :] * stride_dw1_k)
    dw1_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    
    # USE DTYPE HERE
    tl.store(dw1_ptrs, acc.to(DTYPE), mask=dw1_mask)


def triton_fused_dw1(dZ: torch.Tensor, X: torch.Tensor, inv_rms: torch.Tensor) -> torch.Tensor:
    M, N = dZ.shape
    _, K = X.shape
    dW1 = torch.empty((N, K), device=dZ.device, dtype=dZ.dtype)
    
    # <--- NEW: Map torch.dtype to tl.dtype
    tl_dtype = tl.float16 if dZ.dtype == torch.float16 else tl.bfloat16
    
    if N >= 2048 and K >= 2048:
        BLOCK_N, BLOCK_K, BLOCK_M, num_warps, num_stages = 128, 128, 32, 4, 4
    else:
        BLOCK_N, BLOCK_K, BLOCK_M, num_warps, num_stages = 64, 64, 32, 4, 3

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']) * triton.cdiv(K, meta['BLOCK_K']),)
    
    _fused_dw1_kernel[grid](
        dZ, X, inv_rms, dW1,
        M, N, K,
        dZ.stride(0), dZ.stride(1),
        X.stride(0), X.stride(1),
        dW1.stride(0), dW1.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
        GROUP_N=8,
        DTYPE=tl_dtype, # <--- NEW: Pass it into the kernel
        num_warps=num_warps, num_stages=num_stages
    )
    return dW1
    
# ══════════════════════════════════════════════════════════════════════════════
# AUTOGRAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class NVFP4MLPFunction(torch.autograd.Function):
    """
    Five Triton kernels (2 fwd + 3 bwd).

    X, W1, W2 are pre-cast to FP8 once at the top of forward() and those
    copies are saved for backward — every kernel receives FP8 inputs directly.
    dA is never materialised: k_bwd_dZ fuses dY@W2ᵀ with the ⊙2S epilogue.
    """

    @staticmethod
    @torch.compile
    def forward(ctx, X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor):
        assert X.dtype == W1.dtype == W2.dtype and \
               X.dtype in (torch.float16, torch.bfloat16), \
               "inputs must all be float16 or all bfloat16"
        assert X.is_contiguous() and W1.is_contiguous() and W2.is_contiguous()

        dtype = X.dtype          # float16 or bfloat16 — preserved for all outputs
        tl_dtype = tl.float16 if dtype == torch.float16 else tl.bfloat16

        orig_shape = X.shape

        X = X.reshape(-1, orig_shape[-1])

        M, K = X.shape
        N, _ = W1.shape
        P, _ = W2.shape
        
        S = torch.empty((M, N), device='cuda', dtype=torch.bfloat16)
        Y = torch.empty((M, P), device='cuda', dtype=dtype)

        S, _ = nvfp4mm_relu_2(X, W1, apply_rmsnorm_lhs=True)

        Y = nvfp4mm(S, W2)

        ctx.save_for_backward(X, W1, W2)
        ctx.orig_shape = orig_shape
        ctx.dtype = dtype
        return Y.reshape(*orig_shape[:-1], -1)

    @staticmethod
    @torch.compile
    def backward(ctx, dY: torch.Tensor):
        X, W1, W2  = ctx.saved_tensors
        dtype      = ctx.dtype
        orig_shape = ctx.orig_shape

        P = dY.shape[-1]

        dY = dY.reshape(-1, P)

        S, inv_rms = nvfp4mm_relu(X, W1, apply_rmsnorm_lhs=True)

        dW2 = dY.t() @ S.square()
        
        dZ = triton_fused_dz_inplace(dY, W2, S)
        
        dW1 = triton_fused_dw1(dZ, X, inv_rms)

        dX_norm = dZ @ W1
        
        dX = triton_rmsnorm_bwd_inplace(dX_norm, X, inv_rms)

        dX = dX.reshape(orig_shape)

        return dX, dW1, dW2

def nvfp4_mlp(X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """Y = relu(X @ W1)² @ W2  with full FP8 forward + backward."""
    return NVFP4MLPFunction.apply(X, W1, W2)
```
