from controlnet.share import *
from controlnet import config

import cv2
import einops

import numpy as np
import torch
import random
import os, sys
from PIL import Image
import datetime
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from controlnet.annotator.util import resize_image, HWC3
from controlnet.annotator.uniformer import UniformerDetector
from controlnet.cldm.model import create_model, load_state_dict
from controlnet.cldm.ddim_hacked import DDIMSampler
from controlnet.ldm.data.semantic import load_data
from controlnet.ldm.data.base import Txt2ImgIterableBaseDataset
from torch.utils.data import random_split, DataLoader, Dataset


def initialize_controlnet(cond='seg'):
    config_path = './controlnet/models/cldm_v15_canny.yaml'
    model = create_model(config_path).cpu()
    if cond == 'seg':
        m,u = model.load_state_dict(load_state_dict('./models/fgdm_control_sd15_seg.pth', location='cuda'), strict=False)
    elif cond == 'depth':
        m,u = model.load_state_dict(load_state_dict('./models/fgdm_control_sd15_depth.pth', location='cuda'), strict=False)
    elif cond == 'normal':
        m,u = model.load_state_dict(load_state_dict('./models/fgdm_control_sd15_normal.pth', location='cuda'), strict=False)
    elif cond == 'sketch':
        m,u = model.load_state_dict(load_state_dict('./models/fgdm_control_sd15_scribble.pth', location='cuda'), strict=False)
    else:
        raise NotImplementedError
    print("Missing keys: ", m)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def initialize_controlnet_depth():
    config_path = './controlnet/models/cldm_v15_canny.yaml'
    model = create_model(config_path).cpu()
    m,u = model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'), strict=False)
    print("Missing keys: ", m)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def initialize_controlnet_normal():
    config_path = './controlnet/models/cldm_v15_canny.yaml'
    model = create_model(config_path).cpu()
    m,u = model.load_state_dict(load_state_dict('./models/modified_control_sd15_normal.pth', location='cuda'), strict=False)
    print("Missing keys: ", m)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def initialize_controlnet_sketch():
    config_path = './controlnet/models/cldm_v15_canny.yaml'
    model = create_model(config_path).cpu()
    m,u = model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'), strict=False)
    print("Missing keys: ", m)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, num_repeats, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, class_map=None, spath='', index=0):
    with torch.no_grad():
        mask=None
        B, H, W, C = input_image.shape
        control = torch.from_numpy(input_image.copy()).float().cuda() / 255.0
        control = torch.cat([control for _ in range(num_repeats)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

      
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, 
                                                     shape, cond, verbose=False, eta=eta, #x_T=x_t,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     cond_mask=mask)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results
