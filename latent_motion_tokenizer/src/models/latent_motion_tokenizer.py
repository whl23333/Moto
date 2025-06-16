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
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.hidden_state_decoder = hidden_state_decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss

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


    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_motion_tokens_up = self.vq_up_resampler(quant)
            flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_motion_tokens, None, None
        else:
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            return quant, indices, commit_loss


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
        
        # quant, indices, commit_loss = self.tokenize(cond_pixel_values, target_pixel_values)

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

        if self.hidden_state_decoder is not None:
            recons_hidden_states = self.hidden_state_decoder(
                cond_input = cond_hidden_states,
                latent_motion_tokens=latent_motion_tokens_up
            )

        # Compute loss
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        if self.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
        else:
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
        
        if self.hidden_state_decoder is not None:
            recon_hidden_loss = F.mse_loss(target_hidden_states, recons_hidden_states)
            outputs['recons_hidden_loss'] = recon_hidden_loss
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']

        outputs["loss"] = loss

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        outputs["active_code_num"] = active_code_num

        return outputs

class PairedLatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.hidden_state_decoder = hidden_state_decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss

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


    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_motion_tokens_up = self.vq_up_resampler(quant)
            flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_motion_tokens, None, None
        else:
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            return quant, indices, commit_loss


    def forward(self, 
                cond_pixel_values1,  # 第一组条件图像
                target_pixel_values1,  # 第一组目标图像（用于生成latent tokens）
                cond_pixel_values2,  # 第二组条件图像
                target_pixel_values2,  # 第二组目标图像（用于计算重建损失）
                return_recons_only=False,
                return_motion_token_ids_only=False,
                **kwargs):

        # ================== 第一阶段：用第一组数据生成latent tokens ==================
        with torch.no_grad():
            # 计算第一组数据的隐藏状态
            cond1_hidden = self.image_encoder(cond_pixel_values1).last_hidden_state  # (b, 197, 1024)
            target1_hidden = self.image_encoder(target_pixel_values1).last_hidden_state # (b, 197, 1024)

        # 生成运动token
        query_num = self.m_former.query_num # 8
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond1_hidden,
            target_hidden_states=target1_hidden
        ).last_hidden_state[:, :query_num] # (b, 8, 768)

        # 量化处理
        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down) # quant: [b 8 32] indices [b 8]

        if return_motion_token_ids_only:
            return indices

        # ================== 第二阶段：用第二组数据计算重建损失 ==================
        # 上采样量化结果
        latent_motion_tokens_up = self.vq_up_resampler(quant) # [b 8 768]
        
        # 使用第二组条件图像解码
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values2,  # 关键修改：使用第二组条件图像
            latent_motion_tokens=latent_motion_tokens_up,
            **kwargs
        )

        if return_recons_only:
            return {"recons_pixel_values": recons_pixel_values, "indices": indices}

        # ================== 计算所有损失项 ==================
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        # 重建像素损失（使用第二组目标图像）
        if self.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values2).mean()
        else:
            recons_loss = F.mse_loss(target_pixel_values2, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        # 感知损失（使用第二组目标图像）
        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values2, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss

        # 潜在空间重建损失（如果需要）
        if self.hidden_state_decoder is not None:
            # 计算第二组条件图像的隐藏状态
            with torch.no_grad():
                cond2_hidden = self.image_encoder(cond_pixel_values2).last_hidden_state
                target2_hidden = self.image_encoder(target_pixel_values2).last_hidden_state
            
            # 解码隐藏状态
            recons_hidden = self.hidden_state_decoder(
                cond_input=cond2_hidden,
                latent_motion_tokens=latent_motion_tokens_up
            )
            
            # 计算隐藏状态重建损失
            recon_hidden_loss = F.mse_loss(target2_hidden, recons_hidden)
            outputs['recons_hidden_loss'] = recon_hidden_loss

        # 总损失计算
        loss = (self.commit_loss_w * outputs["commit_loss"] + 
                self.recon_loss_w * outputs["recons_loss"] + 
                self.perceptual_loss_w * outputs["perceptual_loss"])
        
        if self.hidden_state_decoder is not None:
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']
        
        outputs["loss"] = loss
        outputs["active_code_num"] = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)

        return outputs
