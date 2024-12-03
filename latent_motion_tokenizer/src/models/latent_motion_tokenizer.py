"""Latent Motion Tokenizer model."""
import torch
import torch.nn.functional as F
from torch import nn
import lpips
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict


class LatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            perceptual_loss_w=1.
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()


    @property
    def device(self):
        return next(self.parameters()).device


    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict


    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_motion_token_ids):
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_motion_tokens=latent_motion_tokens_up)
        return  {
            "recons_pixel_values": recons_pixel_values,
        }


    def forward(self, cond_pixel_values, target_pixel_values,
                return_recons_only=False, 
                return_motion_token_ids_only=False): 

        # Tokenization
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
        

        if return_motion_token_ids_only:
            return indices # (bs, motion_query_num)

        # Detokenization
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )
            
        if return_recons_only:
            return {
                "recons_pixel_values": recons_pixel_values,
                "indices": indices
            }


        # Compute loss
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        recons_loss = F.mse_loss(target_pixel_values, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss

        loss =  self.commit_loss_w * outputs["commit_loss"] + self.recon_loss_w * outputs["recons_loss"] + \
                self.perceptual_loss_w * outputs["perceptual_loss"]
        outputs["loss"] = loss

        active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).to(loss.device)
        outputs["active_code_num"] = active_code_num

        return outputs
