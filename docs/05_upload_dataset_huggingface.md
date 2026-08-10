# 5) Upload dataset to Hugging Face (`argonne-rpl`)

Use this after recording demos.

## A. Install/login HF CLI

```bash
python3 -m pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

## B. Create dataset repo

```bash
export HF_ORG="argonne-rpl"
export HF_DATASET_REPO="centrifuge-quest3-episodes"
export HF_DATASET_ID="${HF_ORG}/${HF_DATASET_REPO}"

huggingface-cli repo create "${HF_DATASET_ID}" --type dataset
```

## C. Upload raw HDF5

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
cd "${ISAACLAB_ROOT}"
huggingface-cli upload "${HF_DATASET_ID}" \
  ./datasets/centrifuge/quest3_centrifuge_dataset.hdf5 \
  quest3_centrifuge_dataset.hdf5 \
  --repo-type dataset
```

## D. (Recommended) Upload GR00T-ready converted dataset folder

GR00T N1.7 fine-tuning expects a GR00T-flavored LeRobot v2 dataset layout (`meta/`, `data/`, `videos/`, `meta/modality.json`), not only HDF5.

```bash
# Example: upload converted folder
huggingface-cli upload-large-folder "${HF_DATASET_ID}" \
  ./datasets/centrifuge/quest3_centrifuge_groot_v2 \
  --repo-type dataset
```

## E. Verify

```bash
huggingface-cli repo info "${HF_DATASET_ID}" --repo-type dataset
```
