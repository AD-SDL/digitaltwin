# 2) Install RPL tasks in IsaacLab

This repository keeps RPL laboratory tasks in IsaacLab under [tasks/](../tasks/).  
Each installable task package should contain its own `pyproject.toml` and `src/` tree.

## Package layout

Expected pattern:

```text
tasks/
  <task_name>/
    tasks/
      pyproject.toml
      src/
```

Example in this repository:

- [tasks/centrifuge/tasks/](../tasks/centrifuge/tasks/)

Current package naming for that example:

- distribution name: `rpl-centrifuge`
- Python import namespace: `rpl_centrifuge`

## A. Install one task package

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
export GUIDE_REPO_ROOT="<path-to-this-guide-repo>"
export TASK_PACKAGE_DIR="${GUIDE_REPO_ROOT}/tasks/centrifuge/tasks"

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p -m pip install -e \
  "${TASK_PACKAGE_DIR}"
```

After installation, manual imports should use:

```python
import rpl_centrifuge
```

If you previously installed an older copy of this task package under the former
OpenArm-specific naming, reinstall from the path above so the current
namespace and auto-register `.pth` shim are applied.

## B. Install all task packages in this repository

```bash
cd "${ISAACLAB_ROOT}"
find "${GUIDE_REPO_ROOT}/tasks" \
  -mindepth 2 -maxdepth 2 -name pyproject.toml | while read -r pyproject; do
    pkg_dir="$(dirname "${pyproject}")"
    echo "Installing ${pkg_dir}"
    ./isaaclab.sh -p -m pip install -e "${pkg_dir}"
  done
```

## C. Verify Gym task registration

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p - <<'PY'
import gymnasium as gym

keywords = ("centrifuge", "openarm", "gr1t2")
ids = sorted(
    spec.id
    for spec in gym.registry.values()
    if any(keyword in spec.id.lower() for keyword in keywords)
)

print("Discovered task IDs:")
for task_id in ids:
    print(" -", task_id)
PY
```

Expected custom IDs currently include:

- `Isaac-Centrifuge-Bimanual-OpenArm-IK-Rel-v0`
- `Isaac-Centrifuge-Bimanual-OpenArm-IK-Abs-v0`
- `Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0`

## D. Pick the task to run

```bash
export CENTRIFUGE_TASK_ID="Isaac-Centrifuge-GR1T2-WaistEnabled-Abs-v0"
echo "${CENTRIFUGE_TASK_ID}"
```

Replace the value as needed for your task.

## E. Smoke test

```bash
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task "${CENTRIFUGE_TASK_ID}" \
  --viz kit \
  --num_envs 1
```
