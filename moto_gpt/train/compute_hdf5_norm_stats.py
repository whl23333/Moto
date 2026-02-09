#!/usr/bin/env python3
"""
Compute normalization statistics for HDF5 ALOHA dataset before training
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import argparse
import os
from common.data.norm_utils import compute_norm_stats_from_hdf5, save_norm_stats


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics for HDF5 dataset")
    parser.add_argument("--config_path", type=str, 
                       default="/home/hlwang/Moto/moto_gpt/configs/data/hdf5_aloha.yaml",
                       help="Path to data config file")
    parser.add_argument("--num_episodes", type=int, default=None,
                       help="Number of episodes to use for computing stats (default: all)")
    args = parser.parse_args()
    
    # Load config
    import omegaconf
    config = omegaconf.OmegaConf.load(args.config_path)
    
    hdf5_dir = config.get('hdf5_dir')
    qpos_key = config.get('qpos_key', 'observations/qpos')
    use_robot_base = config.get('use_robot_base', False)
    norm_stats_path = config.get('norm_stats_path', './norm_stats/aloha_norm_stats.pt')
    
    print(f"Computing normalization statistics:")
    print(f"  HDF5 dir: {hdf5_dir}")
    print(f"  qpos_key: {qpos_key}")
    print(f"  use_robot_base: {use_robot_base}")
    print(f"  num_episodes: {args.num_episodes if args.num_episodes else 'all'}")
    
    # Compute statistics
    norm_stats = compute_norm_stats_from_hdf5(
        hdf5_dir=hdf5_dir,
        qpos_key=qpos_key,
        use_robot_base=use_robot_base,
        num_episodes=args.num_episodes
    )
    
    # Save statistics
    save_norm_stats(norm_stats, norm_stats_path)
    
    print(f"\n✓ Normalization statistics saved to: {norm_stats_path}")
    print("\nYou can now start training with:")
    print(f"  python moto_gpt/train/train_moto_gpt_hdf5_aloha.py")


if __name__ == "__main__":
    main()
