# LLaMa2 70B LoRa: Reference Implementation

## Setup

### Clone repo, create directories

```bash
BASE_DIR=/persistent_storage-daniel/daniel/mlperf_benchmarks/  # change to your base directory
cd $BASE_DIR

RESOURCES_DIR=$BASE_DIR/resources/
mkdir -p $RESOURCES_DIR/dataset $RESOURCES_DIR/model

git clone https://github.com/mlcommons/training.git

LLAMA_DIR=$BASE_DIR/training/llama2_70b_lora/
cd $LLAMA_DIR
```

### Add scripts

- The following two scripts were adapted from MLPerf Training v4.0 Results NVIDIA submission ([`download_dataset.py`](https://github.com/mlcommons/training_results_v4.0/blob/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo/scripts/download_dataset.py) and [`download_model.py`](https://github.com/mlcommons/training_results_v4.0/blob/main/NVIDIA/benchmarks/llama2_70b_lora/implementations/nemo/scripts/download_model.py)), as the instructions from [Download Data and Model](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/README.md#download-data-and-model) section was not possible (permission denied).

#### Dataset: `scripts/download_dataset.py`:

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

#### Model: `scripts/download_model.py`:

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

### (Optional) Custom config

- Copy `configs/default_config.yaml` into `configs/flex_config.yaml`:

```bash
cp configs/default_config.yaml configs/flex_config.yaml
```

- (Example) Change number of GPUs:

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

## Docker

- Some versions (`pytorch` and `flash-attn`) are inconsistent between the [_Setup_](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/README.md#setup) and the provided [`Dockerfile`](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/Dockerfile). We decide to use latest versions (from `Dockerfile`).
- The following commands were adapted from [`run_docker.sh`](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/run_docker.sh)

```bash
DOCKER_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3  # README uses 23.09, Dockerfile uses 24.01
docker pull $DOCKER_IMAGE

docker run \
  -it \
  --rm \
  --gpus all \
  --name mlperf-llama-reference \
  --volume $LLAMA_DIR:/root/workspace \
  --volume $RESOURCES_DIR/dataset:/root/workspace/dataset \
  --volume $RESOURCES_DIR/model:/root/workspace/model \
  --workdir /root/workspace \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  $DOCKER_IMAGE
```
-> takes ~ 5 minutes

## Dependencies

- The following commands were taken from the [`Dockerfile`](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/Dockerfile#L7-L8).

```bash
pip install -r requirements.txt
pip install flash-attn==2.4.1 --no-build-isolation  # README uses 2.1.0, Dockerfile uses 2.4.1
```
-> takes ~ 1.5 minutes

## Resources

### Dataset

- Size: 107 MB

```bash
python3 ./scripts/download_dataset.py --local_dir ./dataset/
```
-> takes ~ 10 seconds

### Model

- Size: 129 GB

```bash
python3 ./scripts/download_model.py --local_dir ./model/
```
-> takes ~ 30 minutes

## Launch training

- You can use the `flex_config.yaml` file here. All of the other parameters are the original ones (cf. [_Llama2-70B on 8 devices_](https://github.com/mlcommons/training/blob/master/llama2_70b_lora/README.md#llama2-70b-on-8-devices))

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
-> took X seconds
