#!/usr/bin/env python3
"""
Standalone script to compute and save normalization statistics from HDF5 dataset
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.data.norm_utils import compute_norm_stats_from_hdf5, save_norm_stats


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics from HDF5 dataset")
    parser.add_argument("--hdf5_dir", type=str, required=True,
                        help="Path to HDF5 dataset directory (containing train/val subdirs)")
    parser.add_argument("--output", type=str, default="./norm_stats/norm_stats.pt",
                        help="Output path for normalization statistics")
    parser.add_argument("--qpos_key", type=str, default="observations/qpos",
                        help="Key for qpos in HDF5 files")
    parser.add_argument("--use_robot_base", action="store_true",
                        help="Include robot base action in statistics")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Number of episodes to use (default: all)")
    
    args = parser.parse_args()
    
    print(f"Computing normalization statistics from: {args.hdf5_dir}")
    print(f"qpos_key: {args.qpos_key}")
    print(f"use_robot_base: {args.use_robot_base}")
    print(f"num_episodes: {args.num_episodes if args.num_episodes else 'all'}")
    
    # Compute statistics
    norm_stats = compute_norm_stats_from_hdf5(
        hdf5_dir=args.hdf5_dir,
        qpos_key=args.qpos_key,
        use_robot_base=args.use_robot_base,
        num_episodes=args.num_episodes
    )
    
    # Save statistics
    save_norm_stats(norm_stats, args.output)
    
    print(f"\nNormalization statistics saved to: {args.output}")
    print("\nYou can now use this file in your training config:")
    print(f"  norm_stats_path: \"{args.output}\"")
    print(f"  compute_norm_stats: false")


if __name__ == "__main__":
    main()
