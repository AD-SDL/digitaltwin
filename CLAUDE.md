# CLAUDE.md

This file defines general repository usage guidance for AI assistants.

## Repository purpose

This repository is documentation-first. It provides a practical workflow for:

- IsaacLab installation,
- OpenArm task integration,
- XR teleoperation data collection,
- Dataset publication,
- GR00T N1.7 fine-tuning.

## Source of truth

1. Root [README.md](README.md) is the navigation entrypoint.
2. Detailed operational steps live under [docs/](docs/README.md).
3. Keep docs compact, command-first, and reproducible.

## AI editing guidelines

1. Prefer minimal, surgical edits that preserve document structure.
2. Keep command blocks copy-paste ready.
3. Clearly label placeholders (for example `<converted_dataset_dir>`).
4. State hard constraints explicitly (for example dataset format incompatibilities).
5. When upstream behavior changes, update docs in sequence order (`01` -> `06`) and update the root index links if needed.

## Path conventions

Avoid assuming machine-specific checkout paths.

- Use placeholders such as `<isaaclab_root>`, `<guide_repo_root>`, `<task_package_root>`, and `<gr00t_root>`.
- Prefer commands that work after `cd` into the relevant directory.
- If a path must be reused across multiple commands, define it as an environment variable first.

## Validation expectations

For documentation-only changes, verify:

1. All linked files exist,
2. File naming stays consistent with README navigation,
3. Commands are syntactically valid and aligned with upstream tooling.
