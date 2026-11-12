"""Download the Qwen3 language models used by the investigator and the chat.

    python scripts/download_models.py                 # Qwen3-1.7B (4 GB), Qwen3-4B (8 GB), Qwen3-8B (16 GB)
    python scripts/download_models.py Qwen/Qwen3-4B   # just one

The weights go to plain folders under LAD_MODELS_DIR (default ~/.cache/lad_models/<name>), not into the Hugging Face
cache: Windows needs special privileges for the cache's symlinks, and this keeps 24 GB out of a OneDrive-synced
project folder. Qwen3-4B and Qwen3-8B are loaded in 4-bit (bitsandbytes) so they fit an 8 GB GPU, which means the
download is larger than the model in memory. All three are Apache-2.0 licensed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import snapshot_download  # noqa: E402

from log_agents import config  # noqa: E402
from log_agents.slm import model_available  # noqa: E402

SIZES = {"Qwen/Qwen3-1.7B": "4.1 GB", "Qwen/Qwen3-4B": "8.0 GB", "Qwen/Qwen3-8B": "16.4 GB"}


def main():
    repos = sys.argv[1:] or config.LLM_CHOICES
    for repo in repos:
        if model_available(repo):
            print(f"{repo}: already on this machine")
            continue
        print(f"{repo}: downloading {SIZES.get(repo, '?')} to {config.MODELS_DIR / repo.split('/')[-1]}", flush=True)
        snapshot_download(repo, local_dir=str(config.MODELS_DIR / repo.split("/")[-1]),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print(f"{repo}: done")


if __name__ == "__main__":
    main()
