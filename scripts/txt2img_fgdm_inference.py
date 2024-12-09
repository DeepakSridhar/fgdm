import argparse, os, sys, glob
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import cv2
# import json

from ldm.util import instantiate_from_config
from pytorch_lightning import seed_everything
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.color_mapping import *

from torch import autocast
from contextlib import contextmanager, nullcontext


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--use_controlnet",
        action='store_true',
        help="use controlnet",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--cond",
        type=str,
        help="condition to generate",
        default="seg"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt"
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="use inv noise",
    )
    parser.add_argument(
        "--resize",
        action='store_true',
        help="use resize",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)
    cond = opt.cond

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    a_prompt='best quality, extremely detailed'
    n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    if len(opt.n_prompt) > 0:
        print("using n prompt")
        n_prompt = opt.n_prompt
    data = [opt.prompt]

    if opt.use_controlnet:
        import controlnet.initialize_cn as initialize_cn
        cn_model, cn_ddim_sampler = initialize_cn.initialize_controlnet(cond=cond)
    
    if opt.fixed_code:
        data_dict = torch.load('fixed.pt') #save noise code from existing mask
        uncond = data_dict['uncond'][-1]
        start_noise = data_dict['xt']
    else:
        start_noise = None

    sample_path = os.path.join(outpath, "sample1")
    os.makedirs(sample_path, exist_ok=True)
    # sample_path = os.path.join(outpath, "sample2")
    # os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    sample_path = os.path.join(outpath, "")

    all_samples=list()
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    if opt.fixed_code:
                        uc = torch.cat(opt.n_samples * [uncond])
                    else:
                        uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n, pr in enumerate(data):
                    prompt = pr
                    image_id = 1
                    org_prompt = prompt
                    
                    for niter in trange(opt.n_iter, desc="Sampling"):
                        c = model.get_learned_conditioning(opt.n_samples * [prompt])
                        shape = [opt.C, opt.H//8, opt.W//8]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        num=1,
                                                        x_T=start_noise,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        gen_samples = []
                        img_paths = []
                        count = 0
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            num = x_sample.shape[-1] // 3
                            for idx in range(num):
                                x_sample_idx = x_sample[:, :, idx*3:(idx+1)*3].astype(np.uint8)
                                nimg = Image.fromarray(x_sample_idx)
                                if opt.resize:
                                    nimg = nimg.resize((512, 512))
                                save_path = os.path.join(sample_path, "sample1", f"sample{idx+1}_{count:04}.png")
                                nimg.save(save_path)
                                resized_img = cv2.resize(x_sample_idx, (512, 512), interpolation=cv2.INTER_LINEAR)
                                gen_samples.append(resized_img)
                                img_paths.append(save_path)
                                # convert to ade mapping
                                # arr_class2 = decolorize(x_sample_idx[None, ...])
                                # label_indices = np.vectorize(label_mapping.get)(arr_class2)
                                # arr_class = colorize_ade(label_indices[None, ...]).squeeze(0).transpose([1, 2, 0])
                                # Image.fromarray(arr_class.astype(np.uint8)).save(os.path.join(sample_path, "sample2", f"sample_ade{idx+1}_{count:04}.png"))
                            count += 1
                            base_count += 1
                        spath = img_paths[0]
                        batch_size = 5
                        gen_samples = np.stack(gen_samples)[:batch_size]
                        if opt.use_controlnet:

            
                            prompt = org_prompt
                            
                            num_repeats = 1                         
                            
                            
                            
                            logdir = os.path.join(os.path.dirname(spath), f"{cond}_images")
                            os.makedirs(logdir, exist_ok=True)
                            
                            if os.path.isdir(spath):
                                raise NotImplementedError
                            
                            for jdx in range(num_repeats):
                                seed = -1                           
                                
                                a_prompt='best quality, extremely detailed'
                                n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
                                results = initialize_cn.process(cn_model, cn_ddim_sampler, gen_samples, prompt, a_prompt, n_prompt, batch_size, num_repeats, 512, eta=0.0, detect_resolution=512, ddim_steps=20, guess_mode=False,strength=1.0,scale=9.0,seed=seed, spath=spath, index=count)
                                
                                for ridx, result in enumerate(results):
                                    pil_result = Image.fromarray(result)
                                    imgname = img_paths[ridx].split('/')[-1]
                                    
                                    imgpath = os.path.join(logdir, f"{imgname}")
                                    pil_result.save(imgpath)

    print(f"Samples are available here: \n{outpath} \nEnjoy.")
