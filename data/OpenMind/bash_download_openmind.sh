#!/bin/bash
#SBATCH --job-name=openmind_download
#SBATCH --nodes=8                        # number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # CPU cores per node (for --workers)
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/canz/slurm-%j-node%t.out
#SBATCH --error=/home/canz/slurm-%j-node%t.err

# ---------------------------------------------------------------------------
# OpenMind Dataset Download — Multi-Node SLURM Script
#
# Each node downloads a disjoint shard of the 800 OpenNeuro sub-datasets.
# Total files: ~114k NIfTI images. Runtime depends on network bandwidth.
#
# Usage:
#   sbatch bash_download_openmind.sh
#
# To change the number of nodes, update --nodes above AND NUM_NODES below.
# ---------------------------------------------------------------------------

PYTHON=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/maisi_conda_env_v2/bin/python
SCRIPT=/lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_MRBrain/data/OpenMind/0_download_openmind.py
OUTPUT_DIR=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/OpenMind

NUM_NODES=${SLURM_NNODES:-8}      # matches --nodes above
NODE_RANK=${SLURM_NODEID:-0}      # automatically set by SLURM per node
WORKERS=${SLURM_CPUS_PER_TASK:-32}

echo "============================================================"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)  (rank ${NODE_RANK} of ${NUM_NODES})"
echo "Workers   : ${WORKERS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================================"

# Run one download process per node, each assigned a shard of datasets
srun --ntasks=1 \
    $PYTHON $SCRIPT \
        --output-dir   $OUTPUT_DIR \
        --num-nodes    $NUM_NODES \
        --node-rank    $NODE_RANK \
        --workers      $WORKERS \
        --resume

echo "Node ${NODE_RANK} finished."
