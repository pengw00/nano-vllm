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
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # Model path is already resolved in LLM.__init__, so it's always a valid directory here
        # Load HuggingFace config if not provided
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)

        # Load EOS token if not set
        if self.eos == -1:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            self.eos = tokenizer.eos_token_id

        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size must be divisible by 256"
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size must be between 1 and 8"
