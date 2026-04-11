#!/usr/bin/env bash
# Create a GCloud spot VM with L4 GPU for MergingPress benchmark.
# L4 = 24GB VRAM, enough for Qwen2.5-7B-Instruct in bf16 (~14GB).
#
# COST: ~$0.22/hr (spot) for g2-standard-8 + 1xL4
# ESTIMATED RUNTIME: ~2-4hr for RULER-4096 full sweep
#
# Usage:
#   bash gcloud_create_vm.sh          # create VM
#   bash gcloud_create_vm.sh delete   # tear down VM
#
# After creation:
#   gcloud compute ssh kvpress-bench --zone=us-central1-a
#   # Then run the setup inside the VM (see below)

set -euo pipefail

VM_NAME="kvpress-bench"
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-8"  # 8 vCPUs, 32GB RAM, 1x L4 GPU
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

if [[ "${1:-}" == "delete" ]]; then
  echo "Deleting VM $VM_NAME..."
  gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
  exit 0
fi

echo "Creating spot VM: $VM_NAME ($MACHINE_TYPE + 1x L4)"
echo "Zone: $ZONE"
echo ""

gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=nvidia-l4,count=1" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --metadata="install-nvidia-driver=True"

echo ""
echo "VM created. Connect with:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "Then run inside the VM:"
cat << 'SETUP'
  # ── Inside VM setup ──
  cd ~
  git clone https://github.com/jg-codes/kvpress.git
  cd kvpress
  git checkout feature/merging-press

  # Install
  pip install uv
  uv sync --extra eval

  # Verify GPU
  python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

  # Run benchmark (RULER-4096, full)
  cd evaluation
  bash run_gpu_benchmark.sh

  # Or quick test first (10% of data):
  bash run_gpu_benchmark.sh --fraction 0.1

  # For LongBench too:
  DATASET=longbench DATA_DIR=null bash run_gpu_benchmark.sh

  # Copy results back:
  # (from local machine)
  # gcloud compute scp --recurse kvpress-bench:~/kvpress/evaluation/results ./results --zone=us-central1-a
SETUP

echo ""
echo "When done, delete with: bash $0 delete"
