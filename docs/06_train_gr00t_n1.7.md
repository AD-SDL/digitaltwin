# 6) Train GR00T N1.7 with uploaded dataset

This uses NVIDIA's official [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) workflow.

## Important format note

GR00T N1.7 does **not** train directly from IsaacLab HDF5 demo files.  
First convert your centrifuge dataset to GR00T-flavored LeRobot v2 format.

Expected dataset shape:

```text
<dataset_root>/
  meta/info.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/modality.json
  data/chunk-000/*.parquet
  videos/chunk-000/*.mp4
```

## A. Clone and install Isaac-GR00T

```bash
export WORKSPACE_ROOT="<path-to-workspace>"
cd "${WORKSPACE_ROOT}"
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
export GR00T_ROOT="$(pwd)"

sudo apt-get update && sudo apt-get install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.12
uv run huggingface-cli login
```

Also request access to gated backbone model: `nvidia/Cosmos-Reason2-2B`.

## B. Download your uploaded dataset locally

```bash
export HF_DATASET_ID="argonne-rpl/centrifuge-quest3-episodes"
export DATASET_DOWNLOAD_ROOT="<path-to-local-dataset-cache>"
mkdir -p "${DATASET_DOWNLOAD_ROOT}"
huggingface-cli download "${HF_DATASET_ID}" \
  --repo-type dataset \
  --local-dir "${DATASET_DOWNLOAD_ROOT}"
```

## C. Prepare embodiment/modality config

Create a modality config Python file for your robot (for `NEW_EMBODIMENT`), e.g.:

```text
<gr00t_root>/examples/openarm/centrifuge_modality_config.py
```

Use `examples/SO100/so100_config.py` as the starting template.

## D. Fine-tune GR00T N1.7

```bash
cd "${GR00T_ROOT}"
export NUM_GPUS=1
export CONVERTED_DATASET_DIR="<path-to-converted-groot-dataset>"
CUDA_VISIBLE_DEVICES=0 uv run python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "${CONVERTED_DATASET_DIR}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/openarm/centrifuge_modality_config.py \
  --num-gpus ${NUM_GPUS} \
  --output-dir "<path-to-checkpoint-output-dir>" \
  --max-steps 20000 \
  --save-steps 1000 \
  --global-batch-size 64
```

## E. Open-loop evaluation

```bash
cd "${GR00T_ROOT}"
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${CONVERTED_DATASET_DIR}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "<path-to-checkpoint-dir>/checkpoint-20000" \
  --traj-ids 0 1 2 \
  --execution-horizon 16 \
  --steps 400
```

If MSE/MAE does not improve across checkpoints, check modality mapping first (`meta/modality.json` vs modality config file).
