import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import decode_jpeg
import cv2
import os
from einops import rearrange
import random
import json
from PIL import Image
import random
import pandas as pd
from glob import glob
from torchvision import transforms
from pathlib import Path
from collections import defaultdict, Counter
import itertools
import pickle
import blosc
from scipy.interpolate import CubicSpline, interp1d
import torchvision.transforms.functional as transforms_f
import einops
def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)

def get_split_and_ratio(split, splits):
    assert split in ['train', 'val']
    assert 'train' in splits
    if 'val' in splits:
        start_ratio=0
        end_ratio=1
    else:
        if split == 'train':
            start_ratio=0
            end_ratio=0.95
        else:
            split = 'train' 
            start_ratio=0.95
            end_ratio=1
    return split, start_ratio, end_ratio

class DataPrefetcher():
    def __init__(self, loader, device, lang_tokenizer=None):
        self.device = device
        self.loader = loader
        self.lang_tokenizer = lang_tokenizer
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        if self.lang_tokenizer is not None:
            lang_inputs = self.lang_tokenizer(self.batch['lang'], return_tensors="pt", padding=True)
            lang_input_ids = lang_inputs.input_ids
            lang_attention_mask = lang_inputs.attention_mask
            self.batch["lang_input_ids"] = lang_input_ids
            self.batch["lang_attention_mask"] = lang_attention_mask

        with torch.cuda.stream(self.stream):
            for key in self.batch:
                if type(self.batch[key]) is torch.Tensor:
                    self.batch[key] = self.batch[key].to(self.device, non_blocking=True)
        
        # self.batch["lang"] = np.array(self.batch['lang'])

    def __len__(self):
        return len(self.loader.dataset)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if type(batch[key]) is torch.Tensor:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time


class LMDBDataset_Mix(Dataset):
    def __init__(self, datasets, sample_weights):
        super().__init__()
        self.datasets = datasets
        self.sample_weights = np.array(sample_weights)
        self.num_datasets = len(datasets)
        self.dataset_sizes = []
        for dataset in self.datasets:
            self.dataset_sizes.append(len(dataset))

    def __getitem__(self, idx):
        dataset_index = np.random.choice(self.num_datasets, p=self.sample_weights / self.sample_weights.sum())
        # idx is not used
        idx = np.random.randint(self.dataset_sizes[dataset_index])
        return self.datasets[dataset_index][idx]

    def __len__(self):
        return sum(self.dataset_sizes)


class LMDBDataset_for_MotoGPT(Dataset):
    def __init__(
        self, lmdb_dir, split, skip_frame, 
        sequence_length, #start_ratio, end_ratio, 
        chunk_size=3, act_dim=7, 
        do_extract_future_frames=True, do_extract_action=False,
        video_dir=None, rgb_shape=(224, 224), rgb_preprocessor=None, max_skip_frame=None):


        super().__init__()

        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.do_extract_future_frames = do_extract_future_frames
        self.do_extract_action = do_extract_action

        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, act_dim)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        self.lmdb_dir = lmdb_dir
        self.video_dir = video_dir
        self.rgb_preprocessor = rgb_preprocessor

        split, start_ratio, end_ratio = get_split_and_ratio(split, os.listdir(lmdb_dir))
        self.split = split
        env = lmdb.open(os.path.join(lmdb_dir, split), readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length * skip_frame - chunk_size
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(os.path.join(self.lmdb_dir, self.split), readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def extract_lang_goal(self, idx, cur_episode):
        feature_dict = loads(self.txn.get(f'feature_dict_{idx}'.encode()))
        lang = feature_dict['observation']['natural_language_instruction'].decode().lower().strip('.')
        return lang

    def get_video_path(self, cur_episode):
        # return os.path.join(self.video_dir, f'{self.split}_eps_{cur_episode:08d}.mp4')
        raise NotImplementedError

    def extract_frames(self, idx, cur_episode, delta_t, rgb_initial, rgb_future, latent_mask):
        start_local_step = loads(self.txn.get(f'local_step_{idx}'.encode()))
        video_path = self.get_video_path(cur_episode)
        video = cv2.VideoCapture(video_path)

        def _extract_frame(frame_idx):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            try:
                assert ret is True
            except Exception as e:
                # print(f"Failed to read video (path={video_path}, frame_idx={frame_idx})")
                raise e
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        rgb_initial[0] = _extract_frame(start_local_step)

        if self.do_extract_future_frames:
            for i in range(self.sequence_length):
                if loads(self.txn.get(f'cur_episode_{idx+(i+1)*delta_t}'.encode())) == cur_episode:
                    rgb_future[i] = _extract_frame(start_local_step+(i+1)*delta_t)
                    latent_mask[i] = 1
                else:
                    break

        video.release()

    def extract_actions(self, idx, cur_episode, delta_t, actions, mask):
        for i in range(self.sequence_length):
            for j in range(self.chunk_size):
                cur_idx = idx + i*delta_t + j
                if loads(self.txn.get(f'cur_episode_{cur_idx}'.encode())) == cur_episode:
                    mask[i, j] = 1
                    action = self.extract_action(cur_idx)
                    actions[i, j] = action

    def extract_action(self, idx):
        raise NotImplementedError

    
    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        while True:
            try:
                orig_idx = idx
                idx = idx + self.start_step
                cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))

                if self.max_skip_frame is None:
                    delta_t = self.skip_frame
                else:
                    delta_t = random.randint(self.skip_frame, self.max_skip_frame)

                # dummy features
                rgb_initial = self.dummy_rgb_initial.clone()
                rgb_future = self.dummy_rgb_future.clone()
                actions = self.dummy_actions.clone()
                mask = self.dummy_mask.clone()
                latent_mask = self.dummy_latent_mask.clone()

                # extract lang goal
                lang = self.extract_lang_goal(idx, cur_episode)

                # extract initial frame and future frames
                self.extract_frames(
                    idx=idx, cur_episode=cur_episode, delta_t=delta_t,
                    rgb_initial=rgb_initial, 
                    rgb_future=rgb_future, latent_mask=latent_mask
                )

                # extract actions
                if self.do_extract_action:
                    self.extract_actions(
                        idx=idx, cur_episode=cur_episode, delta_t=delta_t,
                        actions=actions, 
                        mask=mask
                    )

                if self.do_extract_future_frames and (not self.do_extract_action) and latent_mask.sum() == 0:
                    raise Exception("latent_mask should be larger than zero!")

                return {
                    "lang": lang,
                    "rgb_initial": rgb_initial,
                    "rgb_future": rgb_future,
                    "actions": actions,
                    "mask": mask,
                    "latent_mask": latent_mask,
                    "idx": orig_idx
                }
                    
            except Exception as e:
                # print(e)
                idx = random.randint(0, len(self))
            

    def __len__(self):
        return self.end_step - self.start_step

class LMDBDataset_for_MotoGPT_OXE(LMDBDataset_for_MotoGPT):
    def get_video_path(self, cur_episode):
        return os.path.join(self.video_dir, f'{self.split}_eps_{cur_episode:08d}.mp4')


class LMDBDataset_for_MotoGPT_Video(LMDBDataset_for_MotoGPT):
    def get_video_path(self, cur_episode):
        return os.path.join(self.video_dir, cur_episode)


class LMDBDataset_for_MotoGPT_RT1(LMDBDataset_for_MotoGPT_OXE):
    def __init__(self, world_vector_range=(-1.0, 1.0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_vector_range = world_vector_range

    def extract_action(self, idx):
        feature_dict = loads(self.txn.get(f'feature_dict_{idx}'.encode()))
        action = []
        for act_name, act_min, act_max in [
            ('world_vector', self.world_vector_range[0], self.world_vector_range[1]),
            ('rotation_delta', -np.pi / 2, np.pi / 2),
            ('gripper_closedness_action', -1.0, 1.0)
        ]:
            action.append(np.clip(feature_dict['action'][act_name], act_min, act_max))
        action = np.concatenate(action)
        action = torch.from_numpy(action)
        return action


class LMDBDataset_for_MotoGPT_CALVIN(LMDBDataset_for_MotoGPT):
    def extract_lang_goal(self, idx, cur_episode):
        lang = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        return lang

    def extract_frames(self, idx, cur_episode, delta_t, rgb_initial, rgb_future, latent_mask):
        rgb_initial[0] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx}'.encode())))

        if self.do_extract_future_frames:
            for i in range(self.sequence_length):
                if loads(self.txn.get(f'cur_episode_{idx+(i+1)*delta_t}'.encode())) == cur_episode:
                    rgb_future[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx+(i+1)*delta_t}'.encode())))
                    latent_mask[i] = 1
                else:
                    break

    def extract_actions(self, idx, cur_episode, delta_t, actions, mask):
        for i in range(self.sequence_length):
            for j in range(self.chunk_size):
                cur_idx = idx + i*delta_t + j
                if loads(self.txn.get(f'cur_episode_{cur_idx}'.encode())) == cur_episode:
                    mask[i, j] = 1
                    action = self.extract_action(cur_idx)
                    actions[i, j] = action

    def extract_action(self, idx):
        action = loads(self.txn.get(f'rel_action_{idx}'.encode()))
        action[-1] = (action[-1] + 1) / 2
        return action









class JsonDataset_for_MotoGPT_Video(Dataset):
    def __init__(
        self, split, skip_frame, 
        sequence_length, #start_ratio, end_ratio,
        video_dir=None, rgb_shape=(224, 224), 
        rgb_preprocessor=None, max_skip_frame=None, video_metadata_path=None, *args, **kwargs):

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame

        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        self.video_dir = video_dir
        self.rgb_preprocessor = rgb_preprocessor

        if video_metadata_path is None:
            video_metadata_path = os.path.join(video_dir, 'video_metadata.json')
        else:
            print(f"specified video_metadata_path: {video_metadata_path}")
        
        with open(video_metadata_path) as f:
            video_metadata = json.load(f)

        split, start_ratio, end_ratio = get_split_and_ratio(split, video_metadata.keys())
        self.split = split

        video_metadata = video_metadata[split]
        videos = video_metadata['videos']
        start_step = int(len(videos) * start_ratio) 
        end_step = int(len(videos) * end_ratio)
        self.videos = videos[start_step:end_step]
        self.num_videos = len(self.videos)
        total_frames = video_metadata['total_frames']
        self.dataset_len = int(total_frames*(end_ratio-start_ratio)) - skip_frame * self.num_videos

    def get_video_path(self, video_basename):
        return os.path.join(self.video_dir, video_basename)

    def extract_frames(self, video_basename, start_local_step, num_frames, delta_t, 
                       rgb_initial, rgb_future, latent_mask):
        video_path = self.get_video_path(video_basename)
        video = cv2.VideoCapture(video_path)

        def _extract_frame(frame_idx):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            try:
                assert ret is True
            except Exception as e:
                raise e
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        rgb_initial[0] = _extract_frame(start_local_step)

        for i in range(self.sequence_length):
            next_local_step = start_local_step+(i+1)*delta_t
            if next_local_step < num_frames:
                rgb_future[i] = _extract_frame(next_local_step)
                latent_mask[i] = 1
            else:
                break

        video.release()

    
    def obtain_item(self, idx, start_local_step=None, delta_t=None):
        video_basename, num_frames, *_ = self.videos[idx]

        assert self.skip_frame < num_frames
        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                max_skip_frame = min(num_frames-1, self.max_skip_frame)
                delta_t = random.randint(self.skip_frame, max_skip_frame)

        # dummy features
        rgb_initial = self.dummy_rgb_initial.clone()
        rgb_future = self.dummy_rgb_future.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        if start_local_step is None:
            start_local_step = random.randint(0, num_frames-1-delta_t)

        self.extract_frames(
            video_basename=video_basename, start_local_step=start_local_step, num_frames=num_frames,
            delta_t=delta_t,
            rgb_initial=rgb_initial, 
            rgb_future=rgb_future, latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        return {
            "rgb_initial": rgb_initial,
            "rgb_future": rgb_future,
            "latent_mask": latent_mask,
            "idx": idx,
            "delta_t": delta_t,
            "start_local_step": start_local_step
        }

    
    def __getitem__(self, idx):
        while True:
            try:
                video_idx = idx % self.num_videos
                return self.obtain_item(video_idx)
            except Exception as e:
                # print(e)
                idx = random.randint(0, len(self)-1)
            

    def __len__(self):
        return self.dataset_len








class NpzDataset_for_MotoGPT_Video(Dataset):
    def __init__(
        self, split, skip_frame, # split: train/val, skip_frame: 5
        sequence_length, # 1
        npz_dir=None, rgb_shape=(224, 224), # npz_dir: "/group/ycyang/yyang-infobai/task_ABC_D/", rgb_shape: [200, 200]
        rgb_preprocessor=None, max_skip_frame=None, npz_metadata_path=None, *args, **kwargs): # 'do_extract_future_frames': True, 'do_extract_action': False

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        if split == 'train':
            split = 'training'
        elif split == 'val':
            split = 'validation'
        else:
            raise NotImplementedError

        self.npz_dir = os.path.join(npz_dir, split)
        self.rgb_preprocessor = rgb_preprocessor

        if npz_metadata_path is None:
            npz_metadata_path = os.path.join(self.npz_dir, 'npz_metadata.json')
        else:
            print(f"specified npz_metadata_path: {npz_metadata_path}")
        
        with open(npz_metadata_path) as f:
            npz_metadata = json.load(f)

        self.npz_metadata = npz_metadata
        self.dataset_len = len(npz_metadata) - skip_frame

    def get_npz_path(self, npz_basename):
        return os.path.join(self.npz_dir, npz_basename)

    def extract_frames(self, npz_basename, delta_t, 
                       rgb_initial, rgb_future, latent_mask):
        
        def _extract_frame(npz_idx):
            npz_path = self.get_npz_path(f"episode_{str(npz_idx).zfill(7)}.npz")
            try:
                frame = Image.fromarray(np.load(npz_path)['rgb_static']).convert("RGB")
            except Exception as e:
                raise e

            frame = np.array(frame)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        start_npz_path = self.get_npz_path(npz_basename)
        start_npz_idx = int(npz_basename.split("_")[-1].split(".")[0])
        rgb_initial[0] = _extract_frame(start_npz_idx)

        for i in range(self.sequence_length):
            next_npz_idx = start_npz_idx+(i+1)*delta_t
            try:
                rgb_future[i] = _extract_frame(next_npz_idx)
                latent_mask[i] = 1
            except:
                break

    
    def obtain_item(self, idx, delta_t=None):
        npz_basename = self.npz_metadata[idx]
        npz_idx = int(npz_basename.split("_")[-1].split(".")[0])

        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                delta_t = random.randint(self.skip_frame, self.max_skip_frame)

        # dummy features
        rgb_initial = self.dummy_rgb_initial.clone()
        rgb_future = self.dummy_rgb_future.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        self.extract_frames(
            npz_basename=npz_basename,
            delta_t=delta_t,
            rgb_initial=rgb_initial, 
            rgb_future=rgb_future, 
            latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        return {
            "rgb_initial": rgb_initial,
            "rgb_future": rgb_future,
            "latent_mask": latent_mask,

            "idx": idx,
            "delta_t": delta_t,
        }

    
    def __getitem__(self, idx):
        while True:
            try:
                return self.obtain_item(idx)
            except Exception as e:
                idx = random.randint(0, len(self)-1)
            

    def __len__(self):
        return self.dataset_len

class NpzDataset_for_MotoGPT_Video_Multiview(Dataset):
    def __init__(
        self, split, skip_frame, # split: train/val, skip_frame: 5
        sequence_length, # 1
        npz_dir=None, rgb_shape_static=(224, 224), rgb_shape_gripper=(224, 224), # npz_dir: "/group/ycyang/yyang-infobai/task_ABC_D/", rgb_shape: [200, 200]
        rgb_preprocessor=None, max_skip_frame=None, npz_metadata_path=None, *args, **kwargs): # 'do_extract_future_frames': True, 'do_extract_action': False

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.dummy_rgb_initial_static = torch.zeros(1, 3, rgb_shape_static[0], rgb_shape_static[1], dtype=torch.uint8)
        self.dummy_rgb_future_static = torch.zeros(sequence_length, 3, rgb_shape_static[0], rgb_shape_static[1], dtype=torch.uint8)
        self.dummy_rgb_initial_gripper = torch.zeros(1, 3, rgb_shape_gripper[0], rgb_shape_gripper[1], dtype=torch.uint8)
        self.dummy_rgb_future_gripper = torch.zeros(sequence_length, 3, rgb_shape_gripper[0], rgb_shape_gripper[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        if split == 'train':
            split = 'training'
        elif split == 'val':
            split = 'validation'
        else:
            raise NotImplementedError

        self.npz_dir = os.path.join(npz_dir, split)
        self.rgb_preprocessor = rgb_preprocessor

        if npz_metadata_path is None:
            npz_metadata_path = os.path.join(self.npz_dir, 'npz_metadata.json')
        else:
            print(f"specified npz_metadata_path: {npz_metadata_path}")
        
        with open(npz_metadata_path) as f:
            npz_metadata = json.load(f)

        self.npz_metadata = npz_metadata
        self.dataset_len = len(npz_metadata) - skip_frame

    def get_npz_path(self, npz_basename):
        return os.path.join(self.npz_dir, npz_basename)

    def extract_frames(self, npz_basename, delta_t, 
                       rgb_initial_static, rgb_initial_gripper, rgb_future_static, rgb_future_gripper, latent_mask):
        
        def _extract_frame(npz_idx):
            npz_path = self.get_npz_path(f"episode_{str(npz_idx).zfill(7)}.npz")
            try:
                frame_static = Image.fromarray(np.load(npz_path)['rgb_static']).convert("RGB")
                frame_gripper = Image.fromarray(np.load(npz_path)['rgb_gripper']).convert("RGB")
            except Exception as e:
                raise e

            frame_static = np.array(frame_static)
            frame_gripper = np.array(frame_gripper)
            frame_static = torch.from_numpy(rearrange(frame_static, 'h w c -> c h w'))
            frame_gripper = torch.from_numpy(rearrange(frame_gripper, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame_static  = self.rgb_preprocessor(frame_static)
                frame_gripper  = self.rgb_preprocessor(frame_gripper)
            return frame_static, frame_gripper

        start_npz_path = self.get_npz_path(npz_basename)
        start_npz_idx = int(npz_basename.split("_")[-1].split(".")[0])
        rgb_initial_static[0], rgb_initial_gripper[0] = _extract_frame(start_npz_idx)

        for i in range(self.sequence_length):
            next_npz_idx = start_npz_idx+(i+1)*delta_t
            try:
                rgb_future_static[i], rgb_future_gripper[i] = _extract_frame(next_npz_idx)
                latent_mask[i] = 1
            except:
                break

    
    def obtain_item(self, idx, delta_t=None):
        npz_basename = self.npz_metadata[idx]
        npz_idx = int(npz_basename.split("_")[-1].split(".")[0])

        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                delta_t = random.randint(self.skip_frame, self.max_skip_frame)

        # dummy features
        rgb_initial_static = self.dummy_rgb_initial_static.clone()
        rgb_future_static = self.dummy_rgb_future_static.clone()
        rgb_initial_gripper = self.dummy_rgb_initial_gripper.clone()
        rgb_future_gripper = self.dummy_rgb_future_gripper.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        self.extract_frames(
            npz_basename=npz_basename,
            delta_t=delta_t,
            rgb_initial_static=rgb_initial_static,
            rgb_initial_gripper=rgb_initial_gripper,
            rgb_future_static=rgb_future_static,
            rgb_future_gripper=rgb_future_gripper,
            latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        return {
            "rgb_initial_static": rgb_initial_static,
            "rgb_future_static": rgb_future_static,
            "rgb_initial_gripper": rgb_initial_gripper,
            "rgb_future_gripper": rgb_future_gripper,
            "latent_mask": latent_mask,
            "idx": idx,
            "delta_t": delta_t,
        }

    
    def __getitem__(self, idx):
        while True:
            try:
                return self.obtain_item(idx)
            except Exception as e:
                idx = random.randint(0, len(self)-1)
            

    def __len__(self):
        return self.dataset_len

class MetaworldMultiView(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False, use_video = False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq
        self.use_video = use_video
        self.frame_skip = frameskip
        task_dirs = glob(f"{path}/**/metaworld_dataset/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.actions = []
        
        for task_dir in task_dirs:
            corner_num=len(glob(f"{task_dir}/*", recursive=True))
            video_num=len(glob(f"{task_dir}/*/*", recursive=True))
            sequence_num=video_num//corner_num
            for i in range(sequence_num):
                seq_dirs = glob(f"{task_dir}/*/{i:03}/")
                seqs = []
                for seq_dir in seq_dirs:
                    seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
                    seqs.append(seq)
                self.sequences.append(seqs)
                self.tasks.append(task_dir.split("/")[-2].replace("-", " "))
                action_dir = f"{task_dir}/corner/{i:03}/action.pkl"
                actions = pd.read_pickle(action_dir)
                action = actions[i]
                if len(action) != len(seqs[0]):
                    action.append(np.array([0.0, 0.0, 0.0, action[len(action)-1][3]]))
                self.actions.append(action)
                  
        # if randomcrop:
        #     self.transform = video_transforms.Compose([
        #         video_transforms.CenterCrop((160, 160)),
        #         video_transforms.RandomCrop((128, 128)),
        #         video_transforms.Resize(target_size),
        #         volume_transforms.ClipToTensor()
        #     ])
        # else:
        #     self.transform = video_transforms.Compose([
        #         video_transforms.CenterCrop((128, 128)),
        #         video_transforms.Resize(target_size),
        #         volume_transforms.ClipToTensor()
        #     ])
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx]
        action = self.actions[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None:
            start_idx = random.randint(0, len(seq[0])-1)
            for i in range(len(seq)):
                seq[i] = seq[i][start_idx:]
            action = action[start_idx:]
            N = len(seq[0])
            samples = []
            for i in range(self.sample_per_seq-1):
                samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
            samples.append(N-1)
        else:
            N = len(seq[0])
            start_idx = random.randint(0, len(seq[0])-self.frame_skip*self.sample_per_seq)
            samples = [i if i < len(seq[0]) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        for i in range(len(seq)):
            seq[i] = [seq[i][j] for j in samples]
        action_sample_seq = []
        if self.frame_skip is None:
            if N < self.sample_per_seq:
                for i in range(self.sample_per_seq-1):
                    if samples[i] < samples [i+1]:
                        action_sample_seq.append(action[samples[i]])
                    else:
                        action_sample_seq.append(np.array([0.0, 0.0, 0.0, action[samples[i]][3]]))
            else:
                for i in range(self.sample_per_seq-1):
                    a = np.add.reduce(action[samples[i]:samples[i+1]])
                    a[3] = action[samples[i+1]-1][3]
                    action_sample_seq.append(a)
        
        else:
            for i in range(self.sample_per_seq-1):
                if samples[i]!=samples[i+1] and samples[i]<N-1:
                    a = np.add.reduce(action[samples[i]:samples[i+1]])
                    a[3] = action[samples[i+1]-1][3]
                    action_sample_seq.append(a)                    
                else:
                    action_sample_seq.append(np.array([0.0, 0.0, 0.0, action[samples[i]][3]]))   
        return seq, action_sample_seq
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            samples, actions = self.get_samples(idx)
            actions = np.array(actions)
            actions = torch.tensor(actions)
            all_x = []
            all_x_cond = []
            for i in range(len(samples)):
                # images = self.transform([Image.open(s) for s in samples[i]])
                images = [Image.open(s) for s in samples[i]]
                images = [np.array(img) for img in images]
                images = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
                images = torch.stack(images, dim=0)
                images = rearrange(images, "f c h w -> c f h w")
                x_cond = images[:, 0] # first frame
                x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
                all_x.append(x)
                all_x_cond.append(x_cond)
            task = self.tasks[idx]
            if self.use_video:
                return torch.stack(all_x, dim=0), torch.stack(all_x_cond, dim=0), task, actions
            else:
                x = torch.stack(all_x, dim=0) # [corner (f c) h w]
                x_cond = torch.stack(all_x_cond, dim=0) # [corner c h w]
                x = rearrange(x, "v (f c) h w -> v f c h w", f=self.sample_per_seq-1, c=len(samples))
                x_target = x[:, -1] # [corner c h w], last frame
                return x_target, x_cond, task, actions
        except Exception as e:
            print(e)
            # return self.__getitem__(idx + 1 % self.__len__()) 


def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_w - resized_size[1], 0), max(
                raw_h - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == (trajectory.shape[1] - 1):  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled



class MetaworldDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=8, target_size=(128, 128), frameskip=1, randomcrop=False):
        print("Preparing dataset...")
        # self.sample_per_seq = sample_per_seq
        self.sample_per_seq = sample_per_seq
        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.actions = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4] # "assembly"
            seq_id= int(seq_dir.split("/")[-2]) # 0
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            action = pd.read_pickle(os.path.join(seq_dir, "action.pkl"))
            action = action[seq_id]
            if len(action) != len(seq):
                action.append(np.array([0.0, 0.0, 0.0, action[len(action)-1][3]]))
            
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
            self.actions.append(action)
    
        # if randomcrop:
        #     self.transform = video_transforms.Compose([
        #         video_transforms.CenterCrop((160, 160)),
        #         video_transforms.RandomCrop((128, 128)),
        #         video_transforms.Resize(target_size),
        #         volume_transforms.ClipToTensor()
        #     ])
        # else:
        #     self.transform = video_transforms.Compose([
        #         video_transforms.CenterCrop((128, 128)),
        #         video_transforms.Resize(target_size),
        #         volume_transforms.ClipToTensor()
        #     ])
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx] # 取某一个目录下的所有图片
        action = self.actions[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None: #随机顺序选取某一个任务下的8张图片，8张图片之间不一定紧挨着对方
            start_idx = random.randint(0, len(seq)-self.sample_per_seq)
            seq = seq[start_idx:]
            action = action[start_idx:]
            N = len(seq)
            samples = []
            for i in range(self.sample_per_seq-1):
                samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
            samples.append(N-1)
        else:
            N=len(seq)
            start_idx = random.randint(0, len(seq)-self.sample_per_seq*self.frame_skip)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        
        action_sample_seq = []
        if self.frame_skip is None:
            if N < self.sample_per_seq:
                for i in range(self.sample_per_seq-1):
                    if samples[i] < samples [i+1]:
                        action_sample_seq.append(action[samples[i]])
                    else:
                        action_sample_seq.append(np.array([0.0, 0.0, 0.0, action[samples[i]][3]]))
            else:
                for i in range(self.sample_per_seq-1):
                    a = np.add.reduce(action[samples[i]:samples[i+1]])
                    a[3] = action[samples[i+1]-1][3]
                    action_sample_seq.append(a)
        
        else:
            for i in range(self.sample_per_seq-1):
                if samples[i]!=samples[i+1] and samples[i]<N-1:
                    a = np.add.reduce(action[samples[i]:samples[i+1]])
                    a[3] = action[samples[i+1]-1][3]
                    action_sample_seq.append(a)                    
                else:
                    action_sample_seq.append(np.array([0.0, 0.0, 0.0, action[samples[i]][3]]))
        
        return [seq[i] for i in samples], action_sample_seq
    
    def __len__(self):
        return len(self.sequences) # 一共有多少种任务
    
    def __getitem__(self, idx):
        try:
            samples, actions = self.get_samples(idx)
            actions = np.array(actions)
            actions = torch.tensor(actions)
            action_arm = actions[:, :3]
            action_arm = action_arm.unsqueeze(0) # shape: [1 self.sample_per_seq-1 3]
            
            action_gripper = actions[:, 3] # shape: [self.sample_per_seq-1]
            action_gripper = action_gripper.unsqueeze(1) # shape: [self.sample_per_seq-1 1]
            action_gripper = action_gripper.unsqueeze(0) # shape: [1 self.sample_per_seq-1 1]
            images = [Image.open(s) for s in samples]
            images_np = [np.array(img) for img in images]
            images = [torch.from_numpy(img).permute(2, 0, 1) for img in images_np]
            images = torch.stack(images, dim=0) # [f c h w]
            rgb_initial = images[0].unsqueeze(0) # [1 c h w]
            rgb_future = images[-1].unsqueeze(0) # [1 c h w]
            
            # # images = self.transform([Image.open(s) for s in samples]) # [c f h w]
            # x_cond = images[:, 0] # first frame 选取8张图片中的第一张作为条件
            # x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
            task = self.tasks[idx]
            mask = torch.ones(1)
            return rgb_initial, rgb_future, task, action_arm, action_gripper, mask # [1 c h w], [1 c h w], str, [1 self.sampler_per_seq-1 3], [1 self.sample_per_seq-1 1], [1]
        except Exception as e:
            print(e)
            # return self.__getitem__(idx + 1 % self.__len__())    
            

class RLBenchDataset_Moto(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        cache_size=0,
        max_episodes_per_task=100, # -1
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False, # true
        interpolation_length=100,
        relative_action=False
    ):
        self._cache = {}
        self._cache_size = cache_size # 600
        self._cameras = cameras
        self._num_iters = num_iters # 600000
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action # false

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory: #true
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length # 2
            )

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                if instructions is not None:
                    self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per task and variation
        episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")] #
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            episodes_by_task[task] = sorted(
                eps, key=lambda t: int(str(t[2]).split('/')[-1][2:-4])
            )
            self._episodes += eps # [(task, var, filepath)], ep0, ep1, ...
            self._num_episodes += len(eps)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_task = episodes_by_task
        self.resize_transform = transforms.Resize((224, 224))

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)

        if args in self._cache:
            return self._cache[args]

        value = loader(args)

        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[args] = value

        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # randomly choose two adjacent frames
        start_id = random.randint(0, len(episode[0]) - 2)
        
        # chunk = random.randint(
        #     0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        # )

        # Get frame ids for this chunk
        frame_ids = [episode[0][start_id], episode[0][start_id + 1]]
        
        # frame_ids = episode[0][
        #     chunk * self._max_episode_length:
        #     (chunk + 1) * self._max_episode_length
        # ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ]) # (2, 4, 2, 3, 256, 256), (time_steps, cameras, [rgb, pcd], c, h, w)

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        if self._instructions:
            instr = random.choice(self._instructions[task][variation])
            instr = instr[None].repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids]) # (2, 8)

        # gripper history
        gripper_history = torch.stack([
            torch.cat([episode[4][max(0, i-2)] for i in frame_ids]), # (2, 8)
            torch.cat([episode[4][max(0, i-1)] for i in frame_ids]), # (2, 8)
            gripper # (2, 8)
        ], dim=1) # (2, 3, 8)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [
                    self._interpolate_traj(episode[5][i]) for i in frame_ids
                ] # [(interpolation_length, 8), ...]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, 8)
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]
        f, k, c, h, w = rgbs.shape
        rgbs = rgbs.view(f*k, c, h, w)
        pcds = pcds.view(f*k, c, h, w)
        rgbs = self.resize_transform(rgbs)
        pcds = self.resize_transform(pcds)
        rgbs = rgbs.view(f, k, c, 224, 224)
        pcds = pcds.view(f, k, c, 224, 224)
        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper, # e.g. tensor (n_frames, 8)
            "curr_gripper_history": gripper_history # e.g. tensor (n_frames, 3, 8)
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })
        ret_dict.update({
            "mask": torch.ones(f-1)
        })
        return ret_dict

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes
