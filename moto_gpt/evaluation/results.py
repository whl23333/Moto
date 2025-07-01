import json
import argparse

def main(file_path):
    with open(file_path, "r") as f:
        data = json.loads(f.read())
    data = data["null"]["task_info"]
    output_file=f"{file_path.split('/')[-4]}.txt"
    with open(output_file, "w") as f:
        for key in sorted(data.keys()):
            success_rate = data[key]["success"] / data[key]["total"]
            f.write(f"{key}\t{success_rate:.4f}\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_path',
        type=str,
        default='/home/yyang-infobai/Moto/latent_motion_tokenizer/configs/train/data_calvin.yaml',
    )
    args= parser.parse_args()
    main(args.file_path)