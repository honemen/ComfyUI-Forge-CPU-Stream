import torch
import comfy.samplers
from comfy.k_diffusion import sampling as kd


class CPUStreamNoise:
    def __init__(self, seed):
        self.seed = int(seed)
        self.pending = None

    def generate_noise(self, latent):
        self.pending = None  # reset on every real execution
        x = latent["samples"]
        if "batch_index" in latent or "noise_mask" in latent:
            raise ValueError("Prototype excludes indexed batches and masks")
        if x.is_nested or x.ndim != 4 or x.shape[:2] != (1, 4):
            raise ValueError("Prototype requires [1, 4, H, W]")
        if x.dtype != torch.float32 or torch.count_nonzero(x).item() != 0:
            raise ValueError("Use a float32 EmptyLatentImage")

        shape = tuple(x.shape[1:])
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        initial = torch.randn(shape, generator=g, device="cpu",
                              dtype=torch.float32).unsqueeze(0)
        self.pending = (g, shape)
        return initial


class LocalRunSampler(comfy.samplers.KSAMPLER):
    def __init__(self, function, options, scale_mode):
        super().__init__(function, dict(options), {})
        self.scale_mode = scale_mode

    def max_denoise(self, model_wrap, sigmas):
        if self.scale_mode == "forge_off":
            return False
        if self.scale_mode == "forge_on":
            return True
        return super().max_denoise(model_wrap, sigmas)


class CPUStreamSampler(comfy.samplers.KSAMPLER):
    def __init__(self, base, stream, scale_mode):
        if base.sampler_function is not kd.sample_er_sde:
            raise ValueError("Use standard KSamplerSelect(er_sde)")
        if base.inpaint_options or "noise_sampler" in base.extra_options:
            raise ValueError("Use a sampler without an existing noise override")
        super().__init__(base.sampler_function, dict(base.extra_options), {})
        self.stream = stream
        self.scale_mode = scale_mode

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or self.stream.pending is None:
            raise ValueError("Connect both paired outputs to one Advanced sampler")
        g, shape = self.stream.pending
        self.stream.pending = None  # consume this handoff exactly once
        if extra_args.get("seed") != self.stream.seed:
            raise ValueError("Initial noise and sampler seeds differ")
        if tuple(noise.shape) != (1,) + shape or noise.dtype != torch.float32:
            raise ValueError("Initial noise shape/dtype changed")
        if len(sigmas) < 2 or float(sigmas[-1]) != 0.0:
            raise ValueError("Prototype requires a full schedule ending at zero")
        if not comfy.samplers.Sampler.max_denoise(self, model_wrap, sigmas):
            raise ValueError("Prototype requires sampling from maximum sigma")

        def next_noise(sigma, sigma_next):
            z = torch.randn(shape, generator=g, device="cpu",
                            dtype=torch.float32)
            return z.unsqueeze(0).to(device=noise.device)

        options = dict(self.extra_options)
        options["noise_sampler"] = next_noise
        local = LocalRunSampler(self.sampler_function, options, self.scale_mode)
        return local.sample(model_wrap, sigmas, extra_args, callback, noise,
                            latent_image, denoise_mask, disable_pbar)


class ForgeCPUStreamAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler": ("SAMPLER",),
            "seed": ("INT", {"default": 3257069847, "min": 0,
                             "max": 0xffffffffffffffff}),
            "initial_scale": (["comfy_auto", "forge_off", "forge_on"],),
        }}

    RETURN_TYPES = ("NOISE", "SAMPLER")
    RETURN_NAMES = ("noise", "sampler")
    FUNCTION = "build"
    CATEGORY = "sampling/forge_compat"

    def build(self, sampler, seed, initial_scale):
        stream = CPUStreamNoise(seed)
        return stream, CPUStreamSampler(sampler, stream, initial_scale)


NODE_CLASS_MAPPINGS = {"ForgeCPUStreamAdapter": ForgeCPUStreamAdapter}
NODE_DISPLAY_NAME_MAPPINGS = {"ForgeCPUStreamAdapter": "Forge CPU Stream Adapter"}