import torch
import time
from wickyvllm.layers.attention import triton_attention

def benchmark_attention(seq_len, num_heads, head_dim, batch_size=1, num_iters=100, warmup=10):
    """Benchmark attention implementations"""
    
    # Create random inputs
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
    scale = 1.0 / (head_dim ** 0.5)
    
    # Test Triton implementation
    print(f"\nBenchmarking seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print("-" * 60)
    
    # Warmup
    for _ in range(warmup):
        o_triton = triton_attention(q, k, v, scale, causal=True)
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.time()
    for _ in range(num_iters):
        o_triton = triton_attention(q, k, v, scale, causal=True)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark PyTorch SDPA
    q_pt = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_dim]
    k_pt = k.unsqueeze(0).transpose(1, 2)
    v_pt = v.unsqueeze(0).transpose(1, 2)
    
    for _ in range(warmup):
        o_pt = torch.nn.functional.scaled_dot_product_attention(
            q_pt, k_pt, v_pt, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        o_pt = torch.nn.functional.scaled_dot_product_attention(
            q_pt, k_pt, v_pt, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Check correctness
    o_pt_reshaped = o_pt.transpose(1, 2).squeeze(0)
    max_diff = (o_triton - o_pt_reshaped).abs().max().item()
    
    print(f"Triton:  {triton_time:.3f} ms")
    print(f"PyTorch: {pytorch_time:.3f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    
    # Memory usage
    triton_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU Memory: {triton_memory:.2f} MB")
    
    return triton_time, pytorch_time, max_diff


if __name__ == "__main__":
    print("=" * 60)
    print("Attention Kernel Performance Benchmark")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        (512, 12, 64),    # Small
        (1024, 12, 64),   # Medium
        (2048, 12, 64),   # Large
        (512, 32, 128),   # Wide heads
    ]
    
    results = []
    for seq_len, num_heads, head_dim in configs:
        try:
            torch.cuda.reset_peak_memory_stats()
            triton_t, pytorch_t, diff = benchmark_attention(seq_len, num_heads, head_dim)
            results.append((seq_len, num_heads, head_dim, triton_t, pytorch_t, diff))
        except Exception as e:
            print(f"Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Config':<20} {'Triton(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10} {'MaxDiff':<10}")
    print("-" * 60)
    for seq_len, num_heads, head_dim, triton_t, pytorch_t, diff in results:
        config_str = f"{seq_len}x{num_heads}x{head_dim}"
        speedup = pytorch_t / triton_t
        print(f"{config_str:<20} {triton_t:<12.3f} {pytorch_t:<12.3f} {speedup:<10.2f}x {diff:<10.6f}")
