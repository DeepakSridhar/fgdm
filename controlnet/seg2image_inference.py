from share import *
import config

import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random
import os, sys
from PIL import Image
import datetime
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.data.semantic import load_data
from ldm.data.base import Txt2ImgIterableBaseDataset
from torch.utils.data import random_split, DataLoader, Dataset

def val_dataloader(dataset, batch_size, num_workers=8, shuffle=False, use_worker_init_fn=False, worker_init_fn=None):
    if isinstance(dataset, Txt2ImgIterableBaseDataset) or use_worker_init_fn:
        init_fn = worker_init_fn
    else:
        init_fn = None
    return DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        worker_init_fn=init_fn,
                        shuffle=shuffle)

apply_uniformer = UniformerDetector()
config_path = './models/cldm_v15_canny.yaml'
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, class_map=None, spath='', index=0):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        if index == 0:
            file='sample1_-00001.png'
        else:
            ind=index-1
            file=f'sample1_{ind:06d}.png'
        file=f'{index:05d}.png'
        if spath.endswith('.png'):
            path = spath
        else:
            path=os.path.join(spath, file)
        if not os.path.exists(path):
            print(path)
            exit()
        detected_map = Image.open(path)
        detected_map = np.array(detected_map)#[:,:,:3]
        detected_map = HWC3(detected_map)
   
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)


        mask=None

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
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
    return [detected_map] + results



def get_input(batch, k):
    x = batch[k][0,:,:,:3]
    return ((x+1)*127.5).numpy().astype(np.uint8)

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config = OmegaConf.load(config_path)
    datac = config.get("data", dict())
    params = datac.get("params", dict())
    valid = params.get("validation", dict())
    val_params = valid.get("params", dict())

    spath = sys.argv[1] 
    prompt = sys.argv[2]
    seed = sys.argv[3] #48
    num_repeats = int(sys.argv[4])
    logdir = './models/'
    key1 = "map"
    key2 = "sample"
    
    logdir = os.path.join(os.path.dirname(spath), "seg_samples")
    os.makedirs(logdir)
    

    logdir1 = os.path.join(logdir, 'image')
    os.makedirs(logdir1, exist_ok=True)
    logdir2 = os.path.join(logdir, 'control')
    os.makedirs(logdir2, exist_ok=True)
    
    if os.path.isdir(spath):
        raise NotImplementedError
    imgname = spath.split('/')[-1]
    for jdx in range(num_repeats):
        seed = -1
        
        input_image = np.zeros((512, 512, 3), dtype=np.uint8)
        a_prompt='best quality, extremely detailed'
        n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        detect, result = process(input_image, prompt, a_prompt, n_prompt, 1, 512, eta=0.0, detect_resolution=512, ddim_steps=20, guess_mode=False,strength=1.0,scale=9.0,seed=seed, spath=spath, index=i)
        pil_detect = Image.fromarray(detect)
        pil_result = Image.fromarray(result)
        imgpath = os.path.join(logdir2, f"{imgname}")
        pil_result.save(imgpath)
        print(f"Result saved to {imgpath}")

