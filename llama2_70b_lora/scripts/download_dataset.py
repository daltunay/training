import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download dataset using Hugging Face Hub")
parser.add_argument('--local_dir', type=str, required=True, help='Local directory to download the dataset to')

args = parser.parse_args()

snapshot_download(
    "regisss/scrolls_gov_report_preprocessed_mlperf_2",
    local_dir=args.local_dir,
    local_dir_use_symlinks=False,
    repo_type="dataset",
)
