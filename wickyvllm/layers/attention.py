import torch
from torch import nn
import triton
import triton.language as tl

from wickyvllm.utils.context import get_context

# Conditional import
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
    # Check GPU capability at import time
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            print(f"Warning: FlashAttention requires Ampere GPU or newer (compute capability >= 8.0), "
                  f"but found compute capability {capability[0]}.{capability[1]}. "
                  f"Falling back to standard PyTorch attention.")
            FLASH_ATTN_AVAILABLE = False
except (ImportError, RuntimeError):
    FLASH_ATTN_AVAILABLE = False


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = Q + qvk_offset
    K_block_ptr = K + qvk_offset
    V_block_ptr = V + qvk_offset
    O_block_ptr = Out + qvk_offset
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    q_ptrs = Q_block_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K_block_ptr + (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        # Causal mask
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_ptrs = V_block_ptr + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i = m_ij
    
    acc = acc / l_i[:, None]
    o_ptrs = O_block_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def triton_attention(q, k, v, scale, causal=True):
    # q, k, v: [seq, num_heads, head_dim]
    # Reduce block size to fit in shared memory
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    
    # Adjust block sizes based on head_dim
    if BLOCK_DMODEL > 128:
        BLOCK_M = 32
        BLOCK_N = 32
    
    total_seq, num_heads, head_dim = q.shape
    
    o = torch.empty_like(q)
    
    grid = (triton.cdiv(total_seq, BLOCK_M), num_heads)
    
    _fwd_kernel[grid](
        q, k, v, scale, o,
        q.stride(0), q.stride(1), q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(0), k.stride(2),
        v.stride(0), v.stride(1), v.stride(0), v.stride(2),
        o.stride(0), o.stride(1), o.stride(0), o.stride(2),
        1, num_heads, total_seq,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
    )
    
    return o


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if FLASH_ATTN_AVAILABLE:
            # Use FlashAttention
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        else:
            # Fallback to Triton attention
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                # Process each sequence in the batch
                batch_size = len(context.cu_seqlens_q) - 1
                o_list = []
                for i in range(batch_size):
                    start_q, end_q = context.cu_seqlens_q[i], context.cu_seqlens_q[i + 1]
                    start_k, end_k = context.cu_seqlens_k[i], context.cu_seqlens_k[i + 1]
                    qi = q[start_q:end_q]
                    ki = k[start_k:end_k]
                    vi = v[start_k:end_k]
                    oi = triton_attention(qi, ki, vi, self.scale, causal=True)
                    o_list.append(oi)
                o = torch.cat(o_list, dim=0)
            else:  # decode
                batch_size = q.size(0)
                o_list = []
                for i in range(batch_size):
                    context_len = context.context_lens[i].item()
                    block_table = context.block_tables[i]
                    qi = q[i:i+1]
                    cache_indices = block_table[:context_len]
                    ki = k_cache[cache_indices]
                    vi = v_cache[cache_indices]
                    oi = triton_attention(qi, ki, vi, self.scale, causal=False)
                    o_list.append(oi)
                o = torch.cat(o_list, dim=0)
        
        return o
