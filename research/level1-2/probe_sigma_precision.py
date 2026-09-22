"""Isolated CPU probe of the fixed sources' dtype-order difference, no model.

Run under Forge's or ComfyUI's Python. This does NOT prove user-run sigmas match
these defaults: compare actual sigmas.base and sigmas.sampler dumps first.
"""
import torch
import numpy as np
from scipy.stats import beta


def zsnr(sigmas):
    alphas = 1 / (sigmas * sigmas + 1)
    s = alphas.sqrt()
    first, last = s[0].clone(), s[-1].clone()
    s -= last
    s *= first / (first - last)
    alphas = s ** 2
    alphas[-1] = 4.8973451890853435e-08
    return ((1 - alphas) / alphas) ** 0.5


if __name__ == '__main__':
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    ac = torch.cumprod(1 - betas, dim=0)
    sigma64 = ((1 - ac) / ac) ** 0.5
    forge = zsnr(sigma64.float()).float()
    comfy = zsnr(sigma64).float()
    inds = np.rint(beta.ppf(1 - np.linspace(0, 1, 36, endpoint=False), .6, .6) * 999).astype(int)
    print('torch', torch.__version__, 'CPU; synthetic common SDXL schedule')
    print('base unequal:', int((forge != comfy).sum()), 'max_abs:', float((forge-comfy).abs().max()))
    f, c = forge[inds.tolist()], comfy[inds.tolist()]
    print('Beta-selected unequal:', int((f != c).sum()))
    for i, (t, a, b) in enumerate(zip(inds, f, c)):
        if a != b:
            print('step', i, 'training_index', int(t), 'forge', a.item(), 'comfy', b.item())
