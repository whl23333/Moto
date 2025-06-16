import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from moto_gpt.src.trainers.moto_gpt_trainer import MotoGPT_Trainer_RLBench
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset, load_instructions, traj_collate_fn
from common.data.datasets import RLBenchDataset_Moto
from torch.utils.data import random_split
import torch
import os


def main(cfg):
    # Prepare Moto-GPT
    moto_gpt_config_path = cfg['moto_gpt_config_path']
    print(f"initializing Moto-GPT from {moto_gpt_config_path} ...")
    moto_gpt_config = omegaconf.OmegaConf.load(moto_gpt_config_path)
    moto_gpt = hydra.utils.instantiate(moto_gpt_config)
    moto_gpt.config = moto_gpt_config

    # Prepare lang_tokenizer and rgb_processor
    lang_tokenizer = AutoTokenizer.from_pretrained(moto_gpt_config['model_lang']['pretrained_model_name_or_path'])
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Prepare Latent Motion Tokenizer
    if moto_gpt_config['latent_motion_pred']:
        latent_motion_tokenizer_path = cfg['latent_motion_tokenizer_path']
        assert latent_motion_tokenizer_path is not None
        print(f"loading Latent Motion Tokenizer from {latent_motion_tokenizer_path} ...")
        latent_motion_tokenizer = load_model(latent_motion_tokenizer_path)
    else:
        latent_motion_tokenizer=None

    vq_remap = cfg.get('vq_remap', None)
    unknown_index = cfg.get('unknown_index', 'closest')
    if vq_remap is not None:
        latent_motion_tokenizer.vector_quantizer.setup_remap(remap=vq_remap, unknown_index=unknown_index)

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    
    # extra_data_config = {
    #     'sequence_length': moto_gpt_config['sequence_length'],
    #     'chunk_size': moto_gpt_config['chunk_size'],
    #     'act_dim': moto_gpt_config['act_dim'],
    #     'do_extract_future_frames': moto_gpt_config['latent_motion_pred'],
    #     'do_extract_action': moto_gpt_config['act_pred']
    # }
    instruction = load_instructions(
        os.path.join("/group/ycyang/yyang-infobai/instructions/peract/", "instructions.pkl"),
        tasks=("place_cups", "close_jar", "insert_onto_square_peg", "light_bulb_in", "meat_off_grill", "open_drawer", "place_shape_in_shape_sorter", "place_wine_at_rack_location", "push_buttons", "put_groceries_in_cupboard", "put_item_in_drawer", "put_money_in_safe", "reach_and_drag", "slide_block_to_color_target", "stack_blocks", "stack_cups", "sweep_to_dustpan_of_size", "turn_tap"),
        variations=tuple(range(200))
    )
    
    if instruction is None:
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    image_rescale = "0.75,1.25"
    rlbench_dataset = RLBenchDataset_Moto(
        root="/group/ycyang/yyang-infobai/rlbench_train/Peract_packaged/train",
        instructions=instruction,
        taskvar=taskvar,
        cache_size=600,
        max_episodes_per_task=-1,
        num_iters=600000,
        cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        training=True,
        image_rescale=tuple(
            float(x) for x in image_rescale.split(",")
        ),
        return_low_lvl_trajectory=True,
        dense_interpolation=bool(1),
        interpolation_length=2,
    )
    train_ratio=0.8
    train_size = int(train_ratio * len(rlbench_dataset))
    eval_size = len(rlbench_dataset) - train_size
    train_dataset, eval_dataset = random_split(rlbench_dataset, [train_size, eval_size])

    g = torch.Generator()
    g.manual_seed(0)
    # dataloader_cls = partial(
    #     DataLoader, 
    #     pin_memory=True, # Accelerate data reading
    #     shuffle=True,
    #     persistent_workers=True,
    #     num_workers=cfg['dataloader_config']['workers_per_gpu'],
    #     batch_size=cfg['dataloader_config']['bs_per_gpu'],
    #     prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    # )
    # train_dataloader = dataloader_cls(train_dataset)
    # eval_dataloader = dataloader_cls(eval_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor'],
        collate_fn=traj_collate_fn,
        generator=g
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor'],
        collate_fn=traj_collate_fn,
        generator=g
    )
    # Prepare Trainer
    trainer = MotoGPT_Trainer_RLBench(
        moto_gpt=moto_gpt,
        moto_gpt_config=moto_gpt_config,
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        lang_tokenizer=lang_tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/home/yyang-infobai/Moto/moto_gpt/configs/train/data_rlbench.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    


