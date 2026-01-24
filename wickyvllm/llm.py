import os
from wickyvllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __init__(self, model, **kwargs):
        # Support both local paths and Hugging Face model IDs
        if not os.path.isdir(model):
            # Try to download from Hugging Face if it's a model ID (e.g., "Qwen/Qwen3-0.6B")
            if "/" in model:
                try:
                    from huggingface_hub import snapshot_download
                    print(f"Downloading model from Hugging Face: {model}")
                    model = snapshot_download(repo_id=model, resume_download=True)
                    print(f"Model downloaded to: {model}")
                except Exception as e:
                    raise ValueError(f"Failed to download model '{model}' from Hugging Face: {e}")
            else:
                raise ValueError(f"Model path does not exist or is not a directory: {model}")
        
        super().__init__(model, **kwargs)
