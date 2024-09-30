import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download model using Hugging Face Hub")
parser.add_argument('--local_dir', type=str, required=True, help='Local directory to download the model to')

args = parser.parse_args()

snapshot_download(
    "regisss/llama2-70b-fused-qkv-mlperf",
    local_dir=args.local_dir,
    local_dir_use_symlinks=False,
)
print(f"Model downloaded to {args.local_dir}")
