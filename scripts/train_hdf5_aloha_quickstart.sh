#!/bin/bash
# Quick start script for HDF5 ALOHA training

set -e  # Exit on error

cd /home/hlwang/Moto

echo "=========================================="
echo "HDF5 ALOHA Training Setup"
echo "=========================================="
echo ""

# Check if normalization stats exist
NORM_STATS_PATH="/home/hlwang/Moto/norm_stats/aloha_norm_stats.pt"

if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "❌ Normalization stats not found at: $NORM_STATS_PATH"
    echo "📊 Computing normalization statistics..."
    python moto_gpt/train/compute_hdf5_norm_stats.py
    echo "✅ Normalization stats computed!"
else
    echo "✅ Normalization stats found at: $NORM_STATS_PATH"
fi

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: actPredTrue_motionPredTrue_visionMaeLarge"
echo "  - Dataset: HDF5 ALOHA (/media/disk3/WHL/aloha)"
echo "  - Batch size: 32 per GPU"
echo "  - Sequence length: 1"
echo "  - Chunk size: 3"
echo ""

# Start training
python moto_gpt/train/train_moto_gpt_hdf5_aloha.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
