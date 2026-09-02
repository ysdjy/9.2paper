#!/usr/bin/env bash
# Build the isolated conda environment the official RMA² code runs in.
#
# Deliberately separate from this project's `env_isaaclab`: RMA² needs SAPIEN 2.2.2 and
# numpy < 1.24, which are incompatible with Isaac Sim. Nothing in src/probe_drawer/ imports
# any of it. Every version choice is justified in docs/RMA2_REPRODUCTION_REPORT.md §2-§3.
set -euxo pipefail

CONDA="${CONDA:-$HOME/anaconda3}"
RMA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../third_party/rma4rma" && pwd)"
PY="$CONDA/envs/rma2/bin/python"

# The official README's `git clone --recurse-submodules` produces nothing: .gitmodules
# declares both forks but no gitlink is committed (report §1.1). Clone them explicitly.
[ -d "$RMA/ManiSkill2" ] || git clone -b rma2 https://github.com/yichao-liang/ManiSkill2.git "$RMA/ManiSkill2"
[ -d "$RMA/stable-baselines3" ] || git clone -b rma2 https://github.com/yichao-liang/stable-baselines3.git "$RMA/stable-baselines3"

git -C "$RMA/ManiSkill2" apply --check "$RMA/../../patches/rma4rma/0001-maniskill2-fix-set_drive_target-typo.patch" 2>/dev/null \
    && git -C "$RMA/ManiSkill2" apply "$RMA/../../patches/rma4rma/0001-maniskill2-fix-set_drive_target-typo.patch"
for patch in 0002-rma4rma-fix-adapter-latent-width-for-peginsertion 0003-rma4rma-unhardcode-n_envs-in-adapter-loop; do
    git -C "$RMA" apply --check "$RMA/../../patches/rma4rma/$patch.patch" 2>/dev/null \
        && git -C "$RMA" apply "$RMA/../../patches/rma4rma/$patch.patch"
done

"$CONDA/bin/conda" create -n rma2 python=3.11 -y
"$PY" -m pip install --upgrade pip setuptools wheel
# cu128 is not optional on Blackwell (sm_120); older wheels have no kernels for it (§2.1).
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install "$RMA/ManiSkill2"
"$PY" -m pip install "$RMA/stable-baselines3"
"$PY" -m pip install -e "$RMA"
# SAPIEN 2.2.2 is a numpy-1 C extension; numpy 2 segfaults inside env.step (§3). pandas and
# matplotlib follow because pip's numpy-2 builds cannot import against numpy 1.23.
"$PY" -m pip install "numpy<1.24" "pandas==2.0.3" "matplotlib==3.7.5"

"$PY" - <<'PYCHECK'
import importlib.metadata as md
import sapien, stable_baselines3, torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print("sapien", sapien.__version__, "| ms2", md.version("mani_skill2"), "| sb3", stable_baselines3.__version__)
PYCHECK
