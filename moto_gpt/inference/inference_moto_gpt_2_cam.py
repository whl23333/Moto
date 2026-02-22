#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotoGPT Inference Script for ALOHA Robot
Adapted from ACT inference code to work with MotoGPT model
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import sys
import os
import argparse
import time
import threading
import collections
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import omegaconf
import hydra
from transformers import AutoTokenizer

# ROS imports
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# Moto imports
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor

# Global variables for inference
inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def crop_image(image, crop_left=120, crop_right=40):
    """Crop image to match training preprocessing"""
    if len(image.shape) == 4:  # (b, c, h, w)
        width = image.shape[-1]
        crop_start = crop_left
        crop_end = width - crop_right
        return image[..., crop_start:crop_end]
    elif len(image.shape) == 3:  # (c, h, w)
        width = image.shape[-1]
        crop_start = crop_left
        crop_end = width - crop_right
        return image[..., crop_start:crop_end]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


def get_image(observation, camera_names):
    """Get and preprocess images from observation"""
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


class MotoGPTInference:
    def __init__(self, args):
        """Initialize MotoGPT inference"""
        # Load MotoGPT config
        moto_gpt_config_path = os.path.join(args.ckpt_dir, 'config.yaml')
        print(f"Loading MotoGPT config from {moto_gpt_config_path}")
        self.moto_gpt_config = omegaconf.OmegaConf.load(moto_gpt_config_path)
        
        # Initialize MotoGPT model
        print(f"Initializing MotoGPT model...")
        self.moto_gpt = hydra.utils.instantiate(self.moto_gpt_config)
        self.moto_gpt.config = self.moto_gpt_config
        
        # Load model checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        print(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.moto_gpt.load_state_dict(state_dict, strict=False)
        self.moto_gpt.cuda()
        self.moto_gpt.eval()
        print("MotoGPT model loaded successfully!")
        
        # Initialize language tokenizer
        lang_model_name = self.moto_gpt_config['model_lang']['pretrained_model_name_or_path']
        print(f"Loading language tokenizer: {lang_model_name}")
        self.lang_tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
        
        # Initialize RGB preprocessor
        rgb_preprocessor_config = args.rgb_preprocessor_config
        print(f"Initializing RGB preprocessor with config: {rgb_preprocessor_config}")
        self.rgb_preprocessor = get_rgb_preprocessor(**rgb_preprocessor_config)
        self.rgb_preprocessor = self.rgb_preprocessor.cuda()
        self.rgb_preprocessor.use_original_rgb = False  # We will normalize images to [0, 1] before feeding into preprocessor
        
        # Load latent motion tokenizer if needed
        if self.moto_gpt_config['latent_motion_pred'] and args.latent_motion_tokenizer_path:
            print(f"Loading latent motion tokenizer from {args.latent_motion_tokenizer_path}")
            self.latent_motion_tokenizer = load_model(args.latent_motion_tokenizer_path)
            self.latent_motion_tokenizer.cuda()
            self.latent_motion_tokenizer.eval()
        else:
            self.latent_motion_tokenizer = None
        
        # Load normalization stats
        stats_path = os.path.join(args.stats_dir, args.ckpt_stats_name)
        print(f"Loading normalization stats from {stats_path}")
        # import pickle
        # torch.load
        self.stats = torch.load(stats_path, map_location='cpu')
        print(f"Loaded normalization stats: {self.stats}")
        
        # Store config
        self.args = args
        self.camera_names = args.camera_names
        self.chunk_size = self.moto_gpt_config['chunk_size']
        self.sequence_length = self.moto_gpt_config['sequence_length']
        self.use_qpos_input = self.moto_gpt_config.get('use_qpos_input', False)
        
        print(f"Inference config:")
        print(f"  - Chunk size: {self.chunk_size}")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Use qpos input: {self.use_qpos_input}")
        print(f"  - Camera names: {self.camera_names}")
    
    def preprocess_qpos(self, qpos):
        """Normalize qpos using training stats"""
        qpos_mean = self.stats['qpos_mean']
        qpos_std = self.stats['qpos_std']
        qpos_mean = np.array(qpos_mean, dtype=np.float32)
        qpos_std = np.array(qpos_std, dtype=np.float32)
        breakpoint()
        qpos = (qpos - qpos_mean) / qpos_std
        return qpos
    
    def postprocess_action(self, action):
        """Denormalize action using training stats"""
        qpos_mean = self.stats['qpos_mean']
        qpos_std = self.stats['qpos_std']
        qpos_mean = np.array(qpos_mean, dtype=np.float32)
        qpos_std = np.array(qpos_std, dtype=np.float32)
        action = action * qpos_std + qpos_mean
        breakpoint()
        return action
    
    def tokenize_language(self, language_instruction):
        """Tokenize language instruction"""
        lang_inputs = self.lang_tokenizer(
            [language_instruction],
            return_tensors="pt",
            padding=True
        )
        lang_input_ids = lang_inputs.input_ids.cuda()
        lang_attention_mask = lang_inputs.attention_mask.cuda()
        return lang_input_ids, lang_attention_mask
    
    @torch.no_grad()
    def predict(self, obs, language_instruction=""):
        """
        Predict actions for current observation
        
        Args:
            obs: observation dict with 'images' and 'qpos'
            language_instruction: text instruction (optional)
        
        Returns:
            actions: numpy array of shape (chunk_size, act_dim)
        """
        # Process RGB images
        # 1. Get images and normalize to [0, 1]
        curr_image = get_image(obs, self.camera_names)  # (1, n_cameras, c, h, w)
        
        # 2. Crop image (match training preprocessing)
        if self.args.crop_image:
            curr_image_static = curr_image[:, 0] # (1, c, h, w)
            curr_image_static = crop_image(curr_image_static, 
                                           crop_left=self.args.crop_left, 
                                           crop_right=self.args.crop_right)
            curr_image_gripper = curr_image[:, 1] # (1, c, h, w)
            curr_image_gripper = crop_image(curr_image_gripper, 
                                           crop_left=80, 
                                           crop_right=80)  # Crop differently for gripper camera to keep more context
            # curr_image = crop_image(curr_image, 
            #                        crop_left=self.args.crop_left, 
            #                        crop_right=self.args.crop_right)
            curr_image = torch.stack([curr_image_static, curr_image_gripper], dim=1)  # (1, n_cameras, c, h, w)
        
        # 3. Apply RGB preprocessor (resize, normalize)
        rgb_initial = self.rgb_preprocessor(curr_image, train=False)  # (1, n_cameras, c, h, w)
        
        # Take only first camera view for model input
        rgb_initial = rgb_initial[:, :1]  # (1, 1, c, h, w)
        
        # Process language
        if language_instruction:
            lang_input_ids, lang_attention_mask = self.tokenize_language(language_instruction)
        else:
            # Use empty language token
            lang_input_ids, lang_attention_mask = self.tokenize_language("")
        
        # Process qpos
        qpos_tensor = None
        if self.use_qpos_input:
            qpos_numpy = np.array(obs['qpos'], dtype=np.float32)
            qpos_numpy = qpos_numpy[-7:]  # Take only last 7 dimensions as qpos
            qpos_normalized = self.preprocess_qpos(qpos_numpy)
            qpos_tensor = torch.from_numpy(qpos_normalized).float().cuda().unsqueeze(0)  # (1, qpos_dim)
        
        # Generate latent motion ids (dummy for inference without reconstruction)
        batch_size = 1
        if self.moto_gpt_config['latent_motion_pred']:
            # Use zero latent motion ids as placeholder
            per_latent_motion_len = self.moto_gpt_config['per_latent_motion_len']
            latent_motion_ids = torch.zeros(
                (batch_size, self.sequence_length, per_latent_motion_len),
                dtype=torch.long
            ).cuda()
        else:
            latent_motion_ids = None
        
        # Prepare attention masks
        attention_mask = torch.ones((batch_size, self.sequence_length), dtype=torch.long).cuda()
        latent_mask = torch.ones((batch_size, self.sequence_length), dtype=torch.long).cuda()
        
        # Forward pass through MotoGPT
        pred = self.moto_gpt(
            rgb=rgb_initial,
            language=lang_input_ids,
            attention_mask=attention_mask,
            latent_motion_ids=latent_motion_ids,
            latent_mask=latent_mask,
            train=False,
            lang_attention_mask=lang_attention_mask,
            qpos=qpos_tensor
        )
        
        # Extract predictions
        # arm_action_preds: (b, t, chunk_size, act_dim - 1)
        # gripper_action_preds: (b, t, chunk_size, 1)
        arm_actions = pred['arm_action_preds']
        gripper_actions = pred['gripper_action_preds']
        
        if arm_actions is None or gripper_actions is None:
            raise ValueError("Model did not predict actions. Check if act_pred=True in config.")
        
        # Flatten all timesteps: (b, t, chunk_size, act_dim) -> (b, t*chunk_size, act_dim)
        batch_size = arm_actions.shape[0]
        seq_len = arm_actions.shape[1]
        chunk_size = arm_actions.shape[2]
        
        # Reshape to flatten timesteps and chunks
        arm_actions = arm_actions.reshape(batch_size, seq_len * chunk_size, -1)  # (b, t*chunk_size, act_dim-1)
        gripper_actions = gripper_actions.reshape(batch_size, seq_len * chunk_size, -1)  # (b, t*chunk_size, 1)
        
        # Concatenate arm and gripper actions
        actions = torch.cat([arm_actions, gripper_actions], dim=-1)  # (b, t*chunk_size, act_dim)
        
        # Take first batch
        actions = actions[0]  # (t*chunk_size, act_dim)
        
        # Convert to numpy
        actions = actions.cpu().numpy()
        
        # Denormalize actions
        actions = self.postprocess_action(actions)
        
        return actions


class RosOperator:
    """ROS operator for ALOHA robot interface"""
    def __init__(self, args):
        self.args = args
        
        # Image buffers
        self.img_front = None
        self.img_left = None
        self.img_right = None
        
        # Joint state buffers
        self.puppet_arm_left = None
        self.puppet_arm_right = None
        
        # Robot base
        self.robot_base = None
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.puppet_arm_left_pub = None
        self.puppet_arm_right_pub = None
        self.robot_base_pub = None
        
        # Initialize ROS
        self.init_ros()
    
    def init_ros(self):
        """Initialize ROS node and subscribers/publishers"""
        rospy.init_node('moto_gpt_inference', anonymous=True)
        
        # Subscribers
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback)
        
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback)
        
        if self.args.use_robot_base:
            rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback)
        
        # Publishers
        self.puppet_arm_left_pub = rospy.Publisher(
            self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10
        )
        self.puppet_arm_right_pub = rospy.Publisher(
            self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10
        )
        
        if self.args.use_robot_base:
            self.robot_base_pub = rospy.Publisher(
                self.args.robot_base_cmd_topic, Twist, queue_size=10
            )
        
        print("ROS node initialized successfully!")
    
    def get_frame(self):
        """Get current observation from ROS topics"""
        if (self.img_front is None or self.img_right is None or 
            self.puppet_arm_left is None or self.puppet_arm_right is None):
            return None
        
        return (
            self.img_front, self.img_left, self.img_right,
            self.puppet_arm_left, self.puppet_arm_right,
            self.robot_base
        )
    
    def puppet_arm_publish(self, left, right):
        """Publish joint commands to puppet arms"""
        msg_left = JointState()
        msg_left.header = Header()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg_left.position = left.tolist() if hasattr(left, 'tolist') else left
        
        msg_right = JointState()
        msg_right.header = Header()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg_right.position = right.tolist() if hasattr(right, 'tolist') else right
        
        self.puppet_arm_left_pub.publish(msg_left)
        self.puppet_arm_right_pub.publish(msg_right)
    
    def publish_and_wait(self, left, right, position_threshold=0.02, timeout=2.0):
        """Publish joint commands and wait until arms reach target positions
        
        Args:
            left: Target position for left arm (7-dim array)
            right: Target position for right arm (7-dim array)
            position_threshold: Maximum allowed position error in radians (default: 0.02)
            timeout: Maximum time to wait in seconds (default: 2.0)
        
        Returns:
            bool: True if reached target within threshold, False if timeout
        """
        # Convert to numpy arrays
        target_left = np.array(left.tolist() if hasattr(left, 'tolist') else left, dtype=np.float32)
        target_right = np.array(right.tolist() if hasattr(right, 'tolist') else right, dtype=np.float32)
        
        # Publish target positions
        self.puppet_arm_publish(left, right)
        
        # Wait for arms to reach target
        rate = rospy.Rate(self.args.publish_rate)
        start_time = time.time()
        
        while not rospy.is_shutdown():
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Warning: Timeout after {elapsed:.2f}s, position not reached")
                return False
            
            # Get current positions
            if self.puppet_arm_left is None or self.puppet_arm_right is None:
                rate.sleep()
                continue
            
            current_left = np.array(self.puppet_arm_left.position, dtype=np.float32)
            current_right = np.array(self.puppet_arm_right.position, dtype=np.float32)
            
            # Calculate position errors
            left_error = np.abs(current_left - target_left)
            right_error = np.abs(current_right - target_right)
            max_left_error = np.max(left_error)
            max_right_error = np.max(right_error)
            
            # Check if all joints are within threshold
            if max_left_error < position_threshold and max_right_error < position_threshold:
                # print(f"Target reached in {elapsed:.3f}s (max error: L={max_left_error:.4f}, R={max_right_error:.4f})")
                return True
            
            rate.sleep()
        
        return False
    
    def puppet_arm_publish_continuous(self, left, right, arm_steps_length=None):
        """Smoothly move puppet arms to target positions"""
        if arm_steps_length is None:
            arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]  # Default step sizes
        
        rate = rospy.Rate(self.args.publish_rate)
        
        # Wait for current arm positions
        while (self.puppet_arm_left is None or self.puppet_arm_right is None) and not rospy.is_shutdown():
            rate.sleep()
        
        left_arm = list(self.puppet_arm_left.position)
        right_arm = list(self.puppet_arm_right.position)
        
        # Calculate movement direction
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        
        # Move smoothly to target
        flag = True
        while flag and not rospy.is_shutdown():
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            
            for i in range(len(left)):
                if left_diff[i] < arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * arm_steps_length[i]
                    flag = True
                    
                if right_diff[i] < arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * arm_steps_length[i]
                    flag = True
            
            self.puppet_arm_publish(np.array(left_arm), np.array(right_arm))
            rate.sleep()
    
    # Callback functions
    def img_front_callback(self, msg):
        self.img_front = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
    def img_left_callback(self, msg):
        self.img_left = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
    def img_right_callback(self, msg):
        self.img_right = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
    def puppet_arm_left_callback(self, msg):
        self.puppet_arm_left = msg
    
    def puppet_arm_right_callback(self, msg):
        self.puppet_arm_right = msg
    
    def robot_base_callback(self, msg):
        self.robot_base = msg


def main():
    parser = argparse.ArgumentParser()
    
    # Model paths
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                       help='Directory containing model checkpoint and config')
    parser.add_argument('--ckpt_name', type=str, default='pytorch_model.bin',
                       help='Checkpoint filename')
    parser.add_argument('--ckpt_stats_name', type=str, default='dataset_stats.pkl',
                       help='Normalization stats filename')
    parser.add_argument('--latent_motion_tokenizer_path', type=str, default=None,
                       help='Path to latent motion tokenizer checkpoint')
    
    # Camera settings
    parser.add_argument('--camera_names', type=str, nargs='+', 
                       default=['cam_high', 'cam_right_wrist'],
                       help='Camera names')
    parser.add_argument('--img_front_topic', type=str, 
                       default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', type=str,
                       default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str,
                       default='/camera_r/color/image_raw')
    
    # Robot interface
    parser.add_argument('--puppet_arm_left_cmd_topic', type=str,
                       default='/master/joint_left')
    parser.add_argument('--puppet_arm_right_cmd_topic', type=str,
                       default='/master/joint_right')
    parser.add_argument('--puppet_arm_left_topic', type=str,
                       default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', type=str,
                       default='/puppet/joint_right')
    parser.add_argument('--robot_base_topic', type=str,
                       default='/odom_raw')
    parser.add_argument('--robot_base_cmd_topic', type=str,
                       default='/cmd_vel')
    parser.add_argument('--use_robot_base', action='store_true',
                       help='Use robot base')
    
    # Image preprocessing
    parser.add_argument('--crop_image', action='store_true',
                       help='Crop image to match training')
    parser.add_argument('--crop_left', type=int, default=120)
    parser.add_argument('--crop_right', type=int, default=40)
    
    # RGB preprocessor config
    parser.add_argument('--rgb_preprocessor_type', type=str, default='MotoRGBPreprocessor',
                       help='RGB preprocessor type')
    parser.add_argument('--rgb_height', type=int, default=224)
    parser.add_argument('--rgb_width', type=int, default=224)
    
    # Inference settings
    parser.add_argument('--publish_rate', type=int, default=40,
                       help='Action publishing rate (Hz)')
    parser.add_argument('--language_instruction', type=str, default="",
                       help='Language instruction for the task')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum inference steps')
    parser.add_argument('--num_actions_to_execute', type=int, default=None,
                       help='Number of actions to execute from predicted action chunk (default: use all)')

    parser.add_argument('--stats_dir', type=str, default=None,
                       help='Directory containing normalization stats')
    
    args = parser.parse_args()
    
    # Build RGB preprocessor config
    args.rgb_preprocessor_config = {
        'model_vision_type': "mae",
        'vision_aug_config': {
            'do_random_resized_crop': False,
            'do_random_shift': False
        }
    }
    
    # Initialize ROS operator
    print("Initializing ROS operator...")
    ros_operator = RosOperator(args)
    
    # Initialize MotoGPT inference
    print("Initializing MotoGPT inference...")
    moto_gpt_inference = MotoGPTInference(args)
    
    # Wait for first observation
    print("Waiting for observations...")
    while ros_operator.get_frame() is None and not rospy.is_shutdown():
        time.sleep(0.1)
    print("Observations received!")
    
    # Reset arms to initial positions
    print("Resetting arms to initial position...")
    # Initial position with gripper open
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    # Initial position with gripper closed
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press Enter to close grippers and start inference...")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    print("Arms reset complete!")
    
    # Main inference loop
    print(f"Starting inference with language instruction: '{args.language_instruction}'")
    print("Press Ctrl+C to stop...")
    
    rate = rospy.Rate(args.publish_rate)
    step = 0
    action_buffer = None
    action_idx = 0
    
    while not rospy.is_shutdown() and step < args.max_steps:
        # Get current observation
        result = ros_operator.get_frame()
        if result is None:
            continue
        
        img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, robot_base = result
        
        # Build observation dict
        obs = collections.OrderedDict()
        obs['images'] = {
            args.camera_names[0]: img_front,
            args.camera_names[1]: img_right
        }
        # obs['qpos'] = np.concatenate([
        #     np.array(puppet_arm_left.position + [puppet_arm_left.velocity[0]]),
        #     np.array(puppet_arm_right.position + [puppet_arm_right.velocity[0]])
        # ])
        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0) # (14,)
        
        # Predict new actions when buffer is empty or exhausted
        if action_buffer is None or action_idx >= len(action_buffer):
            start_time = time.time()
            full_action_buffer = moto_gpt_inference.predict(obs, args.language_instruction)
            inference_time = time.time() - start_time
            
            # Limit to num_actions_to_execute if specified
            if args.num_actions_to_execute is not None:
                action_buffer = full_action_buffer[:args.num_actions_to_execute]
                print(f"Step {step}: Predicted {len(full_action_buffer)} actions, using first {len(action_buffer)} actions in {inference_time:.3f}s")
            else:
                action_buffer = full_action_buffer
                print(f"Step {step}: Predicted {len(action_buffer)} actions in {inference_time:.3f}s")
            
            action_idx = 0
        
        # Get current action from buffer
        action = action_buffer[action_idx]
        action_idx += 1
        
        # Model only predicts right arm (7-dim), left arm stays in current position
        # puppet_arm_left.position already contains 7 values: [joint0, joint1, ..., joint5, joint6(gripper)]
        left_action = np.array(puppet_arm_left.position)  # Keep left arm at current position (7-dim)
        right_action = action  # Model output is 7-dim right arm action
        
        # Publish actions
        ros_operator.publish_and_wait(left_action, right_action)
        
        step += 1
        rate.sleep()
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
