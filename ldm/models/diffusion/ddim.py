"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from ldm.models.diffusion.loss import  caculate_loss_att_fixed_cnt, caculate_loss_self_att, caculate_align_loss_self_att, caculate_align_loss_att_fixed, caculate_ground, caculate_align_ground
from copy import deepcopy

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               inference_loss=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    inference_loss=inference_loss,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      inference_loss=False, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        if 'return_conds' in kwargs:
            return_ids = kwargs["return_conds"]
        else:
            return_ids = False
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, i=i,
                                      inference_loss=inference_loss, **kwargs)
            if return_ids:
                img, pred_x0, caption2, crecon = outs
                kwargs["x2"] = caption2
            else:
                img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        if return_ids:
            return img, (intermediates, caption2)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, i=0, inference_loss=False, **kwargs):
        b, *_, device = *x.shape, x.device
        if 'return_conds' in kwargs:
            return_ids = kwargs["return_conds"]
        else:
            return_ids = False
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            if inference_loss:                    
                x = self.update_align_loss_self_cross(x, t, c, i, index, num=b,  return_ids=return_ids, **kwargs)
            e_t = self.model.apply_model(x, t, c, return_ids=return_ids, **kwargs)
            if return_ids:
                if isinstance(e_t, list) or isinstance(e_t, tuple):
                    if len(e_t) == 2:
                        e_t, x2w = e_t
                        cond_e_t = None
                        # print(x2w.shape, e_t.shape)
                        if x.shape[1] == 4:
                            e_t = x2w
                        # print(x2w.shape, e_t.shape)
                    else:
                        e_t = e_t[0]               
        elif 'composable_diffusion' in kwargs:
            num_prompts = kwargs['composable_diffusion']+1
            x_in = torch.cat([x] * num_prompts)
            t_in = torch.cat([t] * num_prompts)
            print(c.shape)
            c_in = torch.cat([unconditional_conditioning, c])
            noise_pred = self.model.apply_model(x_in, t_in, c_in, **kwargs)
            noise_pred_uncond, noise_pred_text = noise_pred[:1], noise_pred[1:]
            e_t = noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)).sum(dim=0, keepdims=True)
        elif 'augmented_conditoning' in kwargs:
            x_in = torch.cat([x] * 3)
            t_in = torch.cat([t] * 3)
            ac = kwargs['ac']
            c_in = torch.cat([unconditional_conditioning, c, ac])
            e_t_uncond, e_t, e_ac = self.model.apply_model(x_in, t_in, c_in, **kwargs).chunk(3)
            e_t = e_ac + unconditional_guidance_scale * (e_t - e_ac)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if return_ids and 'x2' in kwargs:
                kwargs['x2'] = torch.cat([kwargs['x2']] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            if return_ids:
                if inference_loss:                    
                    x_in = self.update_align_loss_self_cross(x_in, t_in, c_in, i, index, num=b,  return_ids=return_ids, **kwargs)
                model_output = self.model.apply_model(x_in, t_in, c_in, return_ids=return_ids, **kwargs)
                if isinstance(model_output, list) or isinstance(model_output, tuple):
                    if len(model_output) == 2:
                        model_output, x2w = model_output                        
                        if x.shape[1] == 4:
                            model_output = x2w
                        cond_e_t_uncond, cond_e_t = x2w.chunk(2)
                        cond_e_t = cond_e_t_uncond + unconditional_guidance_scale * (cond_e_t - cond_e_t_uncond)
                    else:
                        model_output = model_output[0]
                e_t_uncond, e_t = model_output.chunk(2)                                
            else:
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, **kwargs).chunk(2)   
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if return_ids:
            caption2 = kwargs["x2"]
            if e_t.shape[1] == 16 or cond_e_t is None or caption2.shape[1] == 16:
                return x_prev, pred_x0, x_prev, pred_x0
            if unconditional_guidance_scale != 1.:
                _, caption2 = caption2.chunk(2)
            # current prediction for x_0
            cond_pred_x0 = (caption2[:b] - sqrt_one_minus_at * cond_e_t) / a_t.sqrt()
            if quantize_denoised:
                cond_pred_x0, _, *_ = self.model.first_stage_model.quantize(cond_pred_x0)
            # direction pointing to x_t
            dir_condt = (1. - a_prev - sigma_t**2).sqrt() * cond_e_t
            cond_noise = sigma_t * noise_like(caption2[:b].shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                cond_noise = torch.nn.functional.dropout(cond_noise, p=noise_dropout)
            cond_x_prev = a_prev.sqrt() * cond_pred_x0 + dir_condt + cond_noise
            return x_prev, pred_x0, cond_x_prev, cond_pred_x0
        return x_prev, pred_x0

    def update_align_loss_self_cross(self, x_in, t_in, c_in,index1, index, num=2,type_loss='align_self_accross', return_ids=True, **kwargs):
       

        if index1 < 2:
            loss_scale = 4
            max_iter = 2
        elif index1 <5:
            loss_scale = 4
            max_iter = 6
        elif index1 < 10:
            loss_scale = 3
            max_iter = 3
        elif index1 < 20:
            loss_scale = 3
            max_iter =  2
        else:
            loss_scale = 1
            max_iter = 2
        x_in = deepcopy(x_in)
    
        loss_threshold = 0.1
        max_index = 10
        
        iteration = 0
        loss = torch.tensor(10000)
        
        print("optimize", index1)
        min_inside = 0
        # import pdb; pdb.set_trace()
        max_outside=1
        if (index1 < max_index):
            while (loss.item() > loss_threshold and iteration < max_iter and (index1 < max_index and (min_inside < 0.2 ) )) : #or max_outside>0.15
                x_in = x_in.requires_grad_(True)
                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model.apply_model(x_in, t_in, c_in, return_ids=return_ids, **kwargs)
                
                # self att losss
                loss1= caculate_align_loss_self_att(self_first, self_second, self_third, num)*loss_scale 
                # cross attention-loss
                loss2 = caculate_align_loss_att_fixed(att_second,att_first,att_third, num)
                # print('min, max', min_inside, max_outside)
                loss2 *= loss_scale
                # # self attention loss in gate-self attention
                # loss3 = caculate_align_ground(ground1, ground2, ground3, t = index1, num=num)
                # loss3 = 0
                
                loss = loss2 +  loss1  #+loss3 * loss_scale * 3

                print('loss', loss, loss1, loss2) #, loss3* loss_scale *3 
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x_in], retain_graph=True)[0]  
                # grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x_in], retain_graph=True)[0]
                # hh = torch.autograd.backward(loss.requires_grad_(True))
                # grad_cond = x_in.grad

            
                x_in = x_in - grad_cond
                x_in = x_in.detach()
                iteration += 1
                del loss1, loss2, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        if  (index1>=10):
       
            while ((index1%5==0 and index1<=35) and (iteration < max_iter)): # or (min_inside > 0.2 and max_outside< 0.1)  or max_outside>0.15
                x_in = x_in.requires_grad_(True)
                e_t, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3 = self.model.apply_model(x_in, t_in, c_in, return_ids=return_ids, **kwargs)
                
                # self att losss
                loss1= caculate_align_loss_self_att(self_first, self_second, self_third, num)*loss_scale 
                # cross attention-loss
                loss2 = caculate_align_loss_att_fixed(att_second,att_first,att_third, num)
                # print('min, max', min_inside, max_outside)
                loss2 *= loss_scale
                # self attention loss in gate-self attention
                # loss3 = caculate_align_ground(ground1, ground2, ground3, t = index1, num=num)
                
                
                loss = loss2 +  loss1  #+loss3 * loss_scale * 3

                print('loss', loss, loss1, loss2 ) #, loss3* loss_scale *3
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x_in])[0]  
                # grad_cond = torch.autograd.grad(loss.requires_grad_(True), [x], retain_graph=True)[0]
                

            
                x_in = x_in - grad_cond
                x_in = x_in.detach()
                iteration += 1
                del loss1, loss2, att_first, att_second, att_third, self_first, self_second, self_third, ground1, ground2, ground3

        return x_in

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec