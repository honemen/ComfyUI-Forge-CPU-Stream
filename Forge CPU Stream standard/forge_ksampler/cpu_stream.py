"""Per-execution CPU RNG. The batch-one draw shape matches the supplied Adapter."""
import operator
import torch


class CPUStreamNoise:
    def __init__(self, seed):
        self.seed = int(seed)
        self.generator = None
        self.shapes = None
        self.indices = None
        self.nested = False
        self.device = None

    def _draw(self, shape):
        if self.indices is not None:
            wanted = set(self.indices)
            retained = {}
            for i in range(max(wanted) + 1):
                value = torch.randn((1,) + shape[1:], generator=self.generator,
                                    device='cpu', dtype=torch.float32)
                if i in wanted:
                    retained[i] = value
            return torch.cat([retained[i] for i in self.indices], dim=0)
        if shape[0] == 1:
            # Preserve the reference's exact randn(shape[1:]).unsqueeze(0) call.
            return torch.randn(shape[1:], generator=self.generator,
                               device='cpu', dtype=torch.float32).unsqueeze(0)
        return torch.randn(shape, generator=self.generator, device='cpu', dtype=torch.float32)

    def generate_noise(self, latent):
        self.close()
        x = latent['samples']
        self.nested = bool(x.is_nested)
        tensors = list(x.unbind()) if self.nested else [x]
        self.shapes = [tuple(t.shape) for t in tensors]
        if any(len(s) < 2 or any(d <= 0 for d in s) for s in self.shapes):
            raise ValueError('LATENT samples must have nonempty batch and channel dimensions')
        inds = latent.get('batch_index')
        if inds is not None:
            self.indices = [operator.index(i) for i in inds]
            if not self.indices or min(self.indices) < 0 or any(len(self.indices) != s[0] for s in self.shapes):
                raise ValueError('batch_index must contain one nonnegative integer per batch item')
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        noises = [self._draw(s).to(dtype=t.dtype) for s, t in zip(self.shapes, tensors)]
        if self.nested:
            from comfy.nested_tensor import NestedTensor
            return NestedTensor(noises)
        return noises[0]

    def bind(self, noise, seed):
        if self.generator is None or self.device is not None:
            raise RuntimeError('CPU noise stream must be initialized once per sampling execution')
        if seed != self.seed:
            raise RuntimeError('Initial-noise and sampler seed mismatch')
        self.device = noise.device
        self.bound_shape = tuple(noise.shape)

    def next_noise(self, sigma, sigma_next):
        if self.generator is None or self.device is None:
            raise RuntimeError('CPU noise stream is not active')
        values = [self._draw(s) for s in self.shapes]
        if self.nested:
            from comfy.utils import pack_latents
            value, _ = pack_latents(values)
        else:
            value = values[0]
        if tuple(value.shape) != self.bound_shape:
            raise RuntimeError('Sampler noise shape changed during execution')
        return value.to(device=self.device)

    def close(self):
        self.generator = None
        self.shapes = None
        self.indices = None
        self.device = None
