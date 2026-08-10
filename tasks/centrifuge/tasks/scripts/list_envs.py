# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Print all gym environments registered by the local centrifuge task package.

Run via Isaac Lab's Python:

    <isaaclab_root>/isaaclab.sh -p <task_package_root>/scripts/list_envs.py

Filters the gym registry for IDs matching the custom centrifuge task pack. This
includes both the OpenArm variants and the GR1T2 waist-enabled variant.
Importing the package below is what triggers its ``gym.register(...)`` calls.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app (headless — we only need the gym registry)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

# Import each local task package so its gym.register() calls run.
import rpl_centrifuge  # noqa: F401


def main():
    """Print all local centrifuge task environments registered in the gym registry."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Centrifuge Task Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    package_task_prefixes = (
        "Isaac-Centrifuge-Bimanual-OpenArm-",
        "Isaac-Centrifuge-GR1T2-",
    )
    for task_spec in gym.registry.values():
        if task_spec.id.startswith(package_task_prefixes):
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs.get("env_cfg_entry_point", ""),
                ]
            )
            index += 1

    if index == 0:
        print("No centrifuge task environments found in the gym registry.")
        print("Check that the task packages are installed (pip install -e tasks/).")
    else:
        print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
