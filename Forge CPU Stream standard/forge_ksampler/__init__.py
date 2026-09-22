"""Standalone Forge CPU stream sampler; no dependency on the reference Adapter."""
import inspect
import math
import torch
import comfy.samplers
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from .cpu_stream import CPUStreamNoise

# Only solvers whose upstream default is independent Gaussian noise are overridden.
# Brownian-tree solvers keep their native time-correlated noise implementation.
CPU_STREAM_SAMPLERS = frozenset({
    'er_sde', 'euler_ancestral', 'euler_ancestral_cfg_pp',
    'dpm_2_ancestral', 'dpmpp_2s_ancestral', 'dpmpp_2s_ancestral_cfg_pp',
    'ddpm', 'lcm', 'res_multistep', 'res_multistep_cfg_pp',
    'res_multistep_ancestral', 'res_multistep_ancestral_cfg_pp',
    'seeds_2', 'seeds_3', 'exp_heun_2_x0_sde', 'sa_solver', 'sa_solver_pece',
})


class LocalRunSampler(comfy.samplers.KSAMPLER):
    def __init__(self, base, stream, sampler_name):
        options = dict(base.extra_options)
        if sampler_name in CPU_STREAM_SAMPLERS:
            if 'noise_sampler' not in inspect.signature(base.sampler_function).parameters:
                raise RuntimeError(f'{sampler_name}: installed sampler API has changed (noise_sampler missing)')
            if 'noise_sampler' in options:
                raise RuntimeError('An existing noise override cannot be replaced')
            options['noise_sampler'] = stream.next_noise
        super().__init__(base.sampler_function, options, dict(base.inpaint_options))
        self.forge_off = sampler_name == 'er_sde'
        self.stream = stream

    def max_denoise(self, model_wrap, sigmas):
        if self.forge_off:
            return False
        return super().max_denoise(model_wrap, sigmas)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        self.stream.bind(noise, extra_args.get('seed'))
        try:
            return super().sample(model_wrap, sigmas, extra_args, callback, noise,
                                  latent_image, denoise_mask, disable_pbar)
        finally:
            self.stream.close()


def make_sigmas(model, steps, sampler_name, scheduler, beta_alpha, beta_beta, denoise):
    # Same threshold, step expansion, penultimate removal and suffix as KSampler.
    if denoise <= 0.0:
        return torch.FloatTensor([])
    full_steps = steps if denoise > 0.9999 else int(steps / denoise)
    discard = sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS
    schedule_steps = full_steps + int(discard)
    ms = model.get_model_object('model_sampling')
    if scheduler == 'beta':
        if not (math.isfinite(beta_alpha) and math.isfinite(beta_beta)
                and beta_alpha > 0 and beta_beta > 0):
            raise ValueError('Beta alpha and beta must be finite and greater than zero')
        sigmas = comfy.samplers.beta_scheduler(ms, schedule_steps, alpha=beta_alpha, beta=beta_beta)
    else:
        sigmas = comfy.samplers.calculate_sigmas(ms, scheduler, schedule_steps)
    if discard:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    if denoise <= 0.9999:
        sigmas = sigmas[-(steps + 1):]
    return sigmas


class ForgeKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff,
                             'control_after_generate': False}),
            'steps': ('INT', {'default': 20, 'min': 1, 'max': 10000}),
            'cfg': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1}),
            'sampler_name': (comfy.samplers.KSampler.SAMPLERS, {'default': 'er_sde'}),
            'scheduler': (comfy.samplers.KSampler.SCHEDULERS, {'default': 'beta'}),
            'beta_alpha': ('FLOAT', {'default': 0.6, 'min': 0.001, 'max': 50.0, 'step': 0.01}),
            'beta_beta': ('FLOAT', {'default': 0.6, 'min': 0.001, 'max': 50.0, 'step': 0.01}),
            'positive': ('CONDITIONING',),
            'negative': ('CONDITIONING',),
            'latent_image': ('LATENT',),
            'denoise': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
        }}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'sample'
    CATEGORY = 'sampling/Forge compatibility'

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               beta_alpha, beta_beta, positive, negative, latent_image, denoise=1.0):
        if not (0 <= seed <= 0xffffffffffffffff):
            raise ValueError('seed must be an unsigned 64-bit integer')
        if not math.isfinite(denoise) or not 0 <= denoise <= 1:
            raise ValueError('denoise must be between 0 and 1')
        if steps < 1:
            raise ValueError('steps must be positive')
        if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f'Unknown sampler: {sampler_name}')
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(f'Unknown scheduler: {scheduler}')
        if denoise == 0:
            return (latent_image.copy(),)
        sigmas = make_sigmas(model, steps, sampler_name, scheduler, beta_alpha, beta_beta, denoise)
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        stream = CPUStreamNoise(seed)
        base = comfy.samplers.sampler_object(sampler_name)
        sampler = LocalRunSampler(base, stream, sampler_name)
        try:
            # The reference workflow uses the first output, NOT denoised_output.
            result = SamplerCustomAdvanced().sample(stream, guider, sampler, sigmas, latent_image)
            return (result[0],)
        finally:
            stream.close()


NODE_CLASS_MAPPINGS = {'ForgeKSampler': ForgeKSampler} 
NODE_DISPLAY_NAME_MAPPINGS = {'ForgeKSampler': 'Forge CPU Stream KSampler'} 
WEB_DIRECTORY = './web'