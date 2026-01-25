import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    use_flash_attn: bool = True  # Add this line
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # Model path is already resolved in LLM.__init__, so it's always a valid directory here
        # Load HuggingFace config if not provided
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)

        # Check and adjust dtype based on GPU capability
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # T4 (compute capability 7.5) doesn't support bfloat16
            if capability[0] < 8 and hasattr(self.hf_config, 'torch_dtype'):
                if self.hf_config.torch_dtype == torch.bfloat16:
                    print(f"Warning: GPU compute capability {capability[0]}.{capability[1]} does not support bfloat16. "
                          f"Converting model to float16.")
                    self.hf_config.torch_dtype = torch.float16

        # Load EOS token if not set
        if self.eos == -1:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            self.eos = tokenizer.eos_token_id

        # Check if FlashAttention is supported
        if self.use_flash_attn:
            import torch
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                # FlashAttention requires compute capability >= 8.0 (Ampere)
                if capability[0] < 8:
                    print(f"Warning: FlashAttention requires Ampere GPU or newer (compute capability >= 8.0), "
                          f"but found compute capability {capability[0]}.{capability[1]}. "
                          f"Falling back to standard PyTorch attention.")
                    self.use_flash_attn = False

        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size must be divisible by 256"
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size must be between 1 and 8"
