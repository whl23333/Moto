# huggingface download
from huggingface_hub import snapshot_download
import os
import argparse

def download_task_data(task_name: str):
    repo_id="haolunwang/aloha_multiview"
    local_dir="/media/disk3/WHL/aloha_cup_plate/train"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=[f"{task_name}/*.hdf5", f"{task_name}/*.json", f"{task_name}/*.txt"],
    )

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task to download data for.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    download_task_data(args.task_name)
