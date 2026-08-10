# 1) Install IsaacLab `release-3.0.0-beta2` (with Isaac Teleop)

> Note: upstream uses branch `release/3.0.0-beta2`.  
> For reproducibility, use tag `v3.0.0-beta2.patch1`.

> Note: Isaac Teleop is introduced in the version `3.0.0`.
> We fix the branch to the beta release for now.

## Prerequisites

- Ubuntu 22.04+ with recent NVIDIA driver
- Isaac Sim 6.0.0 or 6.0.1 (binary install)
- `git`, `python3`, `pip`, `cmake`, `build-essential`

## Clone IsaacLab and checkout release

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v3.0.0-beta2.patch1
```

## Install IsaacLab using UV

> Note: Detailed instructions can be found in [here](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/pip_installation.html).

```bash
# Set up Python VENV
uv venv --python 3.12 --seed env_isaaclab
uv pip install --upgrade pip

# Install dependencies
sudo apt install -y cmake build-essential python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev
uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Installs core + submodules (including teleop by default)
./isaaclab.sh --install
```

## E. Verify install

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit
```

Expected result: Isaac Sim launches with a black viewport.
