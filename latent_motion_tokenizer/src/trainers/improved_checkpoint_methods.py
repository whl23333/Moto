"""
Improved checkpoint saving and loading methods for LatentMotionTokenizer_Trainer

This file contains the improved save_checkpoint and load_checkpoint methods that should
replace the existing ones in your trainer classes to fix the issue of loss suddenly
increasing when resuming training.

Key improvements:
1. Save and restore optimizer state
2. Save and restore scheduler state  
3. Save and restore training progress (epoch, step)
4. Save and restore random number generator states
5. Save and restore accelerator states
6. Comprehensive error handling
"""

import torch
import numpy as np
import random
import os
import omegaconf

class ImprovedCheckpointMethods:
    """
    Mixin class containing improved checkpoint methods.
    These methods should be integrated into your existing trainer classes.
    """
    
    def save_checkpoint(self, save_dir, epoch=None, step=None, loss=None):
        """
        Save complete checkpoint including model, optimizer, scheduler and training states.
        
        Args:
            save_dir: Directory to save checkpoint
            epoch: Current epoch number
            step: Current step number
            loss: Current loss value
        """
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Get unwrapped model for saving
        unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
        model_state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
        
        # Save model weights and config
        torch.save(model_state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))
        
        # Prepare training state dictionary
        training_state = {
            'epoch': epoch if epoch is not None else getattr(self, 'current_epoch', 0),
            'step': step if step is not None else getattr(self, 'current_step', 0),
            'best_loss': getattr(self, 'best_loss', float('inf')),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        
        # Save scheduler state (handle custom LinearWarmup_CosineAnnealing scheduler)
        if hasattr(self.scheduler, 'scheduler_linear') and hasattr(self.scheduler, 'scheduler_cosine'):
            # Custom composite scheduler - save all components
            training_state['scheduler_state_dict'] = {
                'scheduler_type': 'LinearWarmup_CosineAnnealing',
                'nb_steps': self.scheduler.nb_steps,
                'switch': self.scheduler.switch,
                'scheduler_linear_state': self.scheduler.scheduler_linear.state_dict(),
                'scheduler_cosine_state': self.scheduler.scheduler_cosine.state_dict(),
            }
        else:
            # Standard scheduler
            training_state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add CUDA RNG states if CUDA is available
        if torch.cuda.is_available():
            training_state['cuda_rng_state'] = torch.cuda.get_rng_state()
            training_state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        
        # Save training state
        torch.save(training_state, os.path.join(save_dir, "training_state.bin"))
        
        # Save accelerator state for DDP and other distributed training states
        try:
            self.accelerator.save_state(save_dir)
        except Exception as e:
            self.print(f"Warning: Could not save accelerator state: {e}")
        
        # Update best loss if provided
        if loss is not None and hasattr(self, 'best_loss'):
            if loss < self.best_loss:
                self.best_loss = loss
                # Create a symlink or copy for best model
                best_model_dir = os.path.join(os.path.dirname(save_dir), 'best_model')
                if os.path.exists(best_model_dir):
                    import shutil
                    shutil.rmtree(best_model_dir)
                shutil.copytree(save_dir, best_model_dir)

        self.print(f"Complete checkpoint saved to {save_dir}!")
        if loss is not None:
            self.print(f"Current loss: {loss:.6f}, Best loss: {getattr(self, 'best_loss', float('inf')):.6f}")
    
    def load_checkpoint(self, resume_ckpt_path):
        """
        Load complete checkpoint including model, optimizer, scheduler and training states.
        
        Args:
            resume_ckpt_path: Path to checkpoint directory
        """
        self.print(f"Resuming training from {resume_ckpt_path}...")
        
        if not os.path.exists(resume_ckpt_path):
            self.print(f"Error: Checkpoint directory {resume_ckpt_path} does not exist!")
            return
        
        # Load model weights
        model_path = os.path.join(resume_ckpt_path, 'pytorch_model.bin')
        if os.path.exists(model_path):
            try:
                missing_keys, unexpected_keys = self.latent_motion_tokenizer.load_state_dict(
                    torch.load(model_path, map_location='cpu'), strict=False)
                missing_root_keys = set([k.split(".")[0] for k in missing_keys])
                self.print(f'✓ Loaded model weights from {model_path}')
                if missing_root_keys:
                    self.print(f'  Missing key groups: {missing_root_keys}')
                if unexpected_keys:
                    self.print(f'  Unexpected keys: {len(unexpected_keys)} keys')
            except Exception as e:
                self.print(f"Error loading model weights: {e}")
                return
        else:
            self.print(f"Warning: Model weights file not found at {model_path}")
        
        # Load training state
        training_state_path = os.path.join(resume_ckpt_path, 'training_state.bin')
        if os.path.exists(training_state_path):
            try:
                training_state = torch.load(training_state_path, map_location='cpu')
                
                # Restore training progress
                self.start_epoch = training_state.get('epoch', 0)
                self.start_step = training_state.get('step', 0)
                self.best_loss = training_state.get('best_loss', float('inf'))
                
                # Store current values for use in training loop
                self.current_epoch = self.start_epoch
                self.current_step = self.start_step
                
                # Restore optimizer state
                if 'optimizer_state_dict' in training_state:
                    try:
                        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                        self.print(f"✓ Restored optimizer state")
                    except Exception as e:
                        self.print(f"Warning: Could not restore optimizer state: {e}")
                        self.print(f"  This may cause loss to spike initially")
                
                # Restore scheduler state
                if 'scheduler_state_dict' in training_state:
                    try:
                        scheduler_state = training_state['scheduler_state_dict']
                        
                        # Check if it's our custom LinearWarmup_CosineAnnealing scheduler
                        if isinstance(scheduler_state, dict) and scheduler_state.get('scheduler_type') == 'LinearWarmup_CosineAnnealing':
                            # Restore custom composite scheduler
                            self.scheduler.nb_steps = scheduler_state.get('nb_steps', 0)
                            self.scheduler.switch = scheduler_state.get('switch', 0)
                            
                            if 'scheduler_linear_state' in scheduler_state:
                                self.scheduler.scheduler_linear.load_state_dict(scheduler_state['scheduler_linear_state'])
                            if 'scheduler_cosine_state' in scheduler_state:
                                self.scheduler.scheduler_cosine.load_state_dict(scheduler_state['scheduler_cosine_state'])
                            
                            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 'unknown'
                            phase = "warmup" if self.scheduler.nb_steps <= self.scheduler.switch else "cosine"
                            self.print(f"✓ Restored custom scheduler state (steps: {self.scheduler.nb_steps}, phase: {phase}, LR: {current_lr})")
                        else:
                            # Standard scheduler
                            self.scheduler.load_state_dict(scheduler_state)
                            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 'unknown'
                            self.print(f"✓ Restored scheduler state (LR: {current_lr})")
                            
                    except Exception as e:
                        self.print(f"Warning: Could not restore scheduler state: {e}")
                        self.print(f"  Learning rate schedule may be incorrect")
                
                # Restore random number generator states
                if 'torch_rng_state' in training_state:
                    torch.set_rng_state(training_state['torch_rng_state'])
                if 'numpy_rng_state' in training_state:
                    np.random.set_state(training_state['numpy_rng_state'])
                if 'python_rng_state' in training_state:
                    random.setstate(training_state['python_rng_state'])
                
                # Restore CUDA RNG states if available
                if torch.cuda.is_available():
                    if 'cuda_rng_state' in training_state:
                        torch.cuda.set_rng_state(training_state['cuda_rng_state'])
                    if 'cuda_rng_state_all' in training_state:
                        torch.cuda.set_rng_state_all(training_state['cuda_rng_state_all'])
                
                self.print(f"✓ Restored training state:")
                self.print(f"    Epoch: {self.start_epoch}")
                self.print(f"    Step: {self.start_step}")
                self.print(f"    Best loss: {self.best_loss:.6f}")
                
            except Exception as e:
                self.print(f"Error loading training state: {e}")
                self.print(f"  Training will start from scratch")
        else:
            self.print(f"Warning: Training state file not found at {training_state_path}")
            self.print(f"  Only model weights loaded, training state will start from scratch")
        
        # Load accelerator state if available
        accelerator_state_path = os.path.join(resume_ckpt_path, 'optimizer.bin')
        if os.path.exists(accelerator_state_path):
            try:
                self.accelerator.load_state(resume_ckpt_path)
                self.print(f"✓ Restored accelerator/DDP state")
            except Exception as e:
                self.print(f"Warning: Could not restore accelerator state: {e}")
        
        self.print(f"🎉 Checkpoint loading completed successfully!")
        self.print(f"    Ready to resume training from epoch {getattr(self, 'start_epoch', 0)}, step {getattr(self, 'start_step', 0)}")

    def update_training_loop_for_resume(self):
        """
        Helper method to modify training loop to support resuming.
        This should be called at the beginning of your train() method.
        """
        # Set starting points for training loop
        start_epoch = getattr(self, 'start_epoch', 0)
        start_step = getattr(self, 'start_step', 0)
        
        # If resuming mid-epoch, you may want to skip some data
        # This is a simplified approach - you might need more sophisticated logic
        # depending on your specific training loop structure
        
        return start_epoch, start_step


# Example of how to modify your train() method to use resuming:
def example_train_method_with_resume(self):
    """
    Example showing how to modify your train() method to support resuming.
    """
    # Get starting points from checkpoint if resuming
    start_epoch, start_step = self.update_training_loop_for_resume()
    
    # Initialize step counter
    step = start_step
    
    for epoch in range(start_epoch, self.num_epochs + 1):
        # Skip epoch 0 logic if resuming
        if epoch != start_epoch or start_step == 0:
            # Your normal epoch start logic here
            pass
        
        # Training loop
        batch_idx = 0
        for batch in self.train_dataloader:
            # If resuming mid-epoch, skip batches until we reach start_step
            if epoch == start_epoch and step < start_step:
                step += 1
                batch_idx += 1
                continue
            
            # Your normal training logic here
            with self.accelerator.accumulate(self.latent_motion_tokenizer):
                # ... training code ...
                # Example: loss = compute_loss(batch)
                pass
            
            # Save checkpoint with current state
            if step % self.save_steps == 0:
                save_dir = os.path.join(self.save_path, f'checkpoint_epoch_{epoch}_step_{step}')
                if self.is_main:
                    # Pass the actual loss from your training loop
                    # Example: self.save_checkpoint(save_dir, epoch=epoch, step=step, loss=loss.item())
                    self.save_checkpoint(save_dir, epoch=epoch, step=step, loss=None)
            
            step += 1
            batch_idx += 1


"""
Integration Instructions:

1. Replace your existing save_checkpoint method with the improved version above
2. Add the load_checkpoint method to your trainer class
3. Modify your __init__ method to call load_checkpoint at the end if resume_ckpt_path is provided
4. Update your train() method to use start_epoch and start_step for resuming
5. Update save_checkpoint calls to include epoch, step, and loss parameters

Example modifications needed in your existing code:

1. In __init__:
   # At the end, after all initialization
   if resume_ckpt_path is not None:
       self.load_checkpoint(resume_ckpt_path)

2. In train():
   start_epoch = getattr(self, 'start_epoch', 0)
   start_step = getattr(self, 'start_step', 0)
   step = start_step
   
   for epoch in range(start_epoch, self.num_epochs + 1):
       # ... rest of training loop ...

3. When saving checkpoints:
   self.save_checkpoint(save_dir, epoch=epoch, step=step, loss=current_loss)
"""