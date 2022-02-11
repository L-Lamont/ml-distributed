#!/bin/bash

msg() { echo "$*" >&2; }
errmsg() { msg "$(basename $0): $*"; }
die() { errmsg "!!! FATAL ERROR: $*"; exit 1; }

set -e

OPENMPI_version="openmpi/4.1.1-mlnx-gcc"
CMAKE_version="cmake/3.17.0"
GCC_version="gcc/9.2.0"
CUDA_version="cuda/nccl_2.8.4-1+cuda11.2"

export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_MXNET=1

export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_GLOO=1

export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_NCCL_HOME=/d/sw/cuda/nccl/nccl_2.8.4-1+cuda11.2_x86_64
export HOROVOD_NCCL_LINK=SHARED

ROOT="/p9/dug/teamhpc/liaml/ml/distributed"
LOG_DIR="$ROOT/logs"
MLPERF="/p9/dug/teamhpc/liaml/cosmoflow/venvs/mlperf-logging"
venvName=$1
newVenv="$ROOT/$venvName"

[[ -z "${venvName}" ]] && die "enter a venvName"
[[ -d "${newVenv}" ]] && die "venv already exists: $venvName"

msg "### Creating venv: $venvName"
python -m venv $newVenv
source $newVenv/bin/activate

[[ -z "${VIRTUAL_ENV}" ]] && die "venv activation unsuccessful: $venvName"
msg "### Using venv: $venvName"

module rm openmpi
module rm cmake
module rm gcc
module rm intel-composer
module rm cuda

module add ${OPENMPI_version}
module add ${CMAKE_version}
module add ${CUDA_version}
module add ${GCC_version}

module list &> "$LOG_DIR/modules.log"

printenv &> "$LOG_DIR/env.log"

msg "### Starting pip installs"
pip install --no-cache-dir --upgrade pip &> "$LOG_DIR/pip.log"

msg "### Installing torch"
pip install --no-cache-dir torch &> "$LOG_DIR/torch.log"
pip install --no-cache-dir torchvision &> "$LOG_DIR/torchvision.log"
pip install --no-cache-dir torchmetrics &> "$LOG_DIR/torchmetrics.log"

msg "### Installing tensorflow"
pip install --no-cache-dir tensorflow &> "$LOG_DIR/tensorflow.log" 

msg "### Installing horovod"
pip install --no-cache-dir horovod[pytorch,tensorflow] &> "$LOG_DIR/horovod.log"

msg "### Installing pytorch-lightning"
pip install --no-cache-dir pytorch-lightning==1.5.9 &> "$LOG_DIR/pytorch-lightning.log"
