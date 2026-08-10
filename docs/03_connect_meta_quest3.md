# 3) Connect Meta Quest 3 to IsaacLab centrifuge task

This flow uses [Isaac Teleop + CloudXR](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/how-to/cloudxr_teleoperation.html).

## (Optional) A. Install Isaac Teleop CloudXR runtime dependencies
> Note: The IsaacLab 3.0.0 or above should come with IsaacTeleop and does not require users to install it manually.
> However, for some reason the package is not available, execute the following.

```bash
export ISAACLAB_ROOT="<path-to-IsaacLab-root>"
cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p -m pip install "isaacteleop[retargeters,cloudxr]~=1.0.0" \
  --extra-index-url https://pypi.nvidia.com
```

## B. Run a Task with XR support
> Note: the IsaacLab launches CloudXR automatically.

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0 \
    --visualizer kit \
    --xr \
```

## C. Pair Quest 3
0. The VR headset should be under the same network of the IsaacSim workstation.
1. Put on Meta Quest 3.
2. Open CloudXR.js client in browser (Quest):  
   `https://nvidia.github.io/IsaacTeleop/client/release-1.3.x`
3. Enter host IP of your IsaacLab machine.
4. At the bottom of the page, accept the certificate issued from the workstation.
4. Connect.
5. In Isaac Sim XR panel, select **System OpenXR Runtime** and click **Start XR**.

## E. Connection check

- You should see the Isaac scene in headset.
- Head/hand/controller motion should update in simulation.
- If task does not respond, confirm your task config has Isaac Teleop support and relaunch with `--xr`.
