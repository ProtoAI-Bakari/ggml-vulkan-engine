#!/usr/bin/env python3
"""Fix tokenizer discovery for all model families (T08)"""

with open('ggml_vllm_backend.py', 'r') as f:
    content = f.read()

# Find and replace the _find_tokenizer method
old_method = '''    def _find_tokenizer(self):
        """Find HF tokenizer for the model."""
        # Check if there's a matching HF model in cache
        base = os.path.basename(self.model_path).lower()
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        if "llama-3.1-8b" in base:
            candidates = [
                os.path.join(cache, d, "snapshots")
                for d in os.listdir(cache)
                if "llama-3.1-8b-instruct" in d.lower()
            ] if os.path.exists(cache) else []
            for c in candidates:
                if os.path.exists(c):
                    snaps = os.listdir(c)
                    if snaps:
                        return os.path.join(c, snaps[0])
        return None'''

new_method = '''    def _find_tokenizer(self):
        """Find HF tokenizer for the model (T08: support all model families)."""
        base = os.path.basename(self.model_path).lower()
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        
        if not os.path.exists(cache):
            return None
        
        # Model family mappings: GGUF name pattern -> HF model name patterns
        family_patterns = {
            "llama": ["llama-3", "llama-3.1", "llama-3.2", "llama-3-instruct", "meta-llama"],
            "qwen": ["qwen2.5", "qwen-2.5", "qwen2", "qwen-2", "qwen"],
            "gpt-oss": ["gpt-oss", "openai-gpt"],
            "mistral": ["mistral", "mixtral", "codestral"],
            "gemma": ["gemma", "gemma2", "gemma-2"],
            "phi": ["phi-3", "phi-3.5", "phi3", "microsoft-phi"],
            "deepseek": ["deepseek", "deepseek-coder"],
            "yi": ["yi-34b", "yi-6b", "01-ai-yi"],
        }
        
        # Detect model family from filename
        detected_family = None
        for family, patterns in family_patterns.items():
            for pattern in patterns:
                if pattern in base:
                    detected_family = family
                    break
            if detected_family:
                break
        
        # If no family detected, try generic search
        if not detected_family:
            # Look for any HF model snapshot
            for d in os.listdir(cache):
                if d.startswith("models--"):
                    snapshot_dir = os.path.join(cache, d, "snapshots")
                    if os.path.exists(snapshot_dir):
                        snaps = os.listdir(snapshot_dir)
                        if snaps and os.path.exists(os.path.join(snapshot_dir, snaps[0], "tokenizer.json")):
                            return os.path.join(snapshot_dir, snaps[0])
            return None
        
        # Search for matching model family in cache
        patterns = family_patterns[detected_family]
        for d in os.listdir(cache):
            if d.startswith("models--"):
                # Decode HF model name (e.g., models--meta-llama--Llama-3.1-8B-Instruct)
                parts = d.replace("models--", "").split("--")
                if len(parts) >= 2:
                    model_name = "--".join(parts[1:]).lower()
                    for pattern in patterns:
                        if pattern in model_name:
                            snapshot_dir = os.path.join(cache, d, "snapshots")
                            if os.path.exists(snapshot_dir):
                                snaps = os.listdir(snapshot_dir)
                                if snaps:
                                    snap_path = os.path.join(snapshot_dir, snaps[0])
                                    # Verify it has tokenizer files
                                    if (os.path.exists(os.path.join(snap_path, "tokenizer.json")) or
                                        os.path.exists(os.path.join(snap_path, "tokenizer.model"))):
                                        print(f"[tokenizer] Found {detected_family} tokenizer at {snap_path}")
                                        return snap_path
        
        # Fallback: try to download from HF
        print(f"[tokenizer] No cached tokenizer found for {detected_family}, attempting download...")
        try:
            from transformers import AutoTokenizer
            # Map detected family to common HF model IDs
            hf_model_map = {
                "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "qwen": "Qwen/Qwen2.5-32B-Instruct",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
                "gemma": "google/gemma-2-27b-it",
                "phi": "microsoft/Phi-3.5-mini-instruct",
                "deepseek": "deepseek-ai/deepseek-coder-33b-instruct",
                "gpt-oss": "openai/gpt-oss-120b",
            }
            model_id = hf_model_map.get(detected_family, "meta-llama/Meta-Llama-3.1-8B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # Save tokenizer to a temp location
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="ggml_tokenizer_")
            tokenizer.save_pretrained(temp_dir)
            print(f"[tokenizer] Downloaded tokenizer to {temp_dir}")
            return temp_dir
        except Exception as e:
            print(f"[tokenizer] Failed to download tokenizer: {e}")
            return None'''

content = content.replace(old_method, new_method)

with open('ggml_vllm_backend.py', 'w') as f:
    f.write(content)

print("Tokenizer discovery fixed for all model families")
