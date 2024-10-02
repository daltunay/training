# LLaMa2 70B LoRa: Reference Implementation Guide

## Table of Contents
1. [Setup](#setup)
    - [Clone Repository and Create Directories](#clone-repository-and-create-directories)
    - [Add Scripts](#add-scripts)
    - [Optional: Custom Configuration](#optional-custom-configuration)
2. [Docker Setup](#docker-setup)
    - [Option 1: Using Dockerfile](#option-1-using-dockerfile)
    - [Option 2: Manual Docker Setup](#option-2-manual-docker-setup)
3. [Download Resources](#download-resources)
    - [Dataset](#dataset)
    - [Model](#model)
4. [Launch Training](#launch-training)

## Setup

### Clone Repository and Create Directories

```bash
BASE_DIR=/persistent_storage-daniel/daniel/mlperf_benchmarks/  # Change to your base directory
cd $BASE_DIR

RESOURCES_DIR=$BASE_DIR/resources/
mkdir -p $RESOURCES_DIR/dataset $RESOURCES_DIR/model

git clone https://github.com/mlcommons/training.git

LLAMA_DIR=$BASE_DIR/training/llama2_70b_lora/
cd $LLAMA_DIR
```

### Add Scripts

The following scripts were adapted from [MLPerf Training v4.0 Results Oracle submission](https://github.com/mlcommons/training_results_v4.0/tree/main/Oracle/benchmarks/llama2_70b_lora/implementations/BM.GPU.H100.8/scripts), as the instructions from [_Download Data and Model_](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/README.md#download-data-and-model) section was not possible (permission denied).

#### Dataset: `scripts/download_dataset.py`

```python
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
```

#### Model: `scripts/download_model.py`

```python
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
```

### Optional: Custom Configuration

Copy and modify the configuration file to change parameters like the number of GPUs:

```bash
cp configs/default_config.yaml configs/flex_config.yaml
```

Example: Change the number of GPUs:

```diff
@@ -14,7 +14,7 @@ machine_rank: 0
 main_training_function: main
 mixed_precision: bf16
 num_machines: 1
-num_processes: 8
+num_processes: 2
 rdzv_backend: static
 same_network: true
 tpu_env: []
```

## Docker Setup

NOTE: Some versions (`pytorch` and `flash-attn`) are inconsistent between the [setup instructions](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/README.md#setup) and the provided original [`Dockerfile`](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/Dockerfile). We choose to use the latest versions.

### Option 1: Using Dockerfile

Create a new `Dockerfile` in `llama2_70b_lora/` with the following content:

```Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /root/workspace

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install flash-attn==2.4.1 --no-build-isolation

ENTRYPOINT ["/bin/bash"]
```

Build the Docker image:

```bash
docker build -t mlperf-llama-image .
```

Run the Docker container:

```bash
docker run \
  -it \
  --rm \
  --gpus all \
  --name mlperf-llama-container \
  --volume $LLAMA_DIR:/root/workspace \
  --volume $RESOURCES_DIR/dataset:/root/workspace/dataset \
  --volume $RESOURCES_DIR/model:/root/workspace/model \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  mlperf-llama-image
```

### Option 2: Manual Docker Setup

Pull the Docker image:

```bash
DOCKER_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
docker pull $DOCKER_IMAGE
```

Run the Docker container:

```bash
docker run \
  -it \
  --rm \
  --gpus all \
  --name mlperf-llama-container \
  --volume $LLAMA_DIR:/root/workspace \
  --volume $RESOURCES_DIR/dataset:/root/workspace/dataset \
  --volume $RESOURCES_DIR/model:/root/workspace/model \
  --workdir /root/workspace \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  $DOCKER_IMAGE
```
Estimated time: ~5 minutes

Install the required Python packages:

```bash
pip install -r requirements.txt
pip install flash-attn==2.4.1 --no-build-isolation
```
Estimated time: ~1.5 minutes

## Download Resources

### Dataset

Size: 107 MB

```bash
python3 ./scripts/download_dataset.py --local_dir ./dataset/
```
Estimated time: ~10 seconds

### Model

Size: 129 GB

```bash
python3 ./scripts/download_model.py --local_dir ./model/
```
Estimated time: ~30 minutes

## Launch Training

Use the `flex_config.yaml` file for custom configurations. All other parameters are the original ones.

```bash
accelerate launch --config_file ./configs/default_config.yaml ./scripts/train.py \
  --dataset_path "./dataset/data/" \
  --model_path "./model/" \
  --max_seq_len 8192 \
  --bf16 True \
  --logging_steps 24 \
  --eval_steps 48 \
  --output_dir "./results/" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type "cosine" \
  --learning_rate 4e-4 \
  --weight_decay 0.0001 \
  --warmup_ratio 0 \
  --max_grad_norm 0.3 \
  --use_gradient_checkpointing True \
  --target_eval_loss 0.925 \
  --use_peft_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --max_steps 1024 \
  --use_flash_attn \
  --seed 1234 \
  --lora_target_modules "qkv_proj,o_proj"
```
Estimated time: ~1 hour
