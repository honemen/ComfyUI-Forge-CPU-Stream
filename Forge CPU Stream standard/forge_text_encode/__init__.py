"""SDXL text conditioning with Forge chunking and Original emphasis.
Derived from the validated ForgeTextEncodeMinimal chunk_and_original path.
"""
import re
import torch
from .forge_chunks import tokenize_forge


def original_emphasis(z, weights):
    mult = torch.asarray(weights).to(z)
    original_mean = z.mean()
    out = z * mult.reshape(mult.shape + (1,)).expand(z.shape)
    return out * (original_mean / out.mean())


class ForgeTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',),
            'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
        }}

    RETURN_TYPES = ('CONDITIONING',)
    RETURN_NAMES = ('conditioning',)
    FUNCTION = 'encode'
    CATEGORY = 'conditioning/Forge compatibility'

    def encode(self, clip, text):
        from comfy.sdxl_clip import SDXLClipModel
        if not isinstance(clip.cond_stage_model, SDXLClipModel):
            raise ValueError('Ordinary SDXL CLIP-L/G required')
        if re.search(r'<lora:|embedding:', text, re.I):
            raise ValueError('Use the LoRA node; remove Forge LoRA tags. Textual Inversion is not supported.')
        if re.search(r'\bAND\b', text):
            raise ValueError('AND prompt composition is not supported')
        tokens = {}
        for name in ['l', 'g']:
            tokens[name] = tokenize_forge(text, getattr(clip.tokenizer, 'clip_' + name))
        if len(tokens['l']) != len(tokens['g']):
            raise ValueError('Different L/G chunk counts are not supported')
        actual = {k: [[(t, 1.0) for t, w in row] for row in v] for k, v in tokens.items()}
        result = clip.encode_from_tokens(actual, return_dict=True)
        cond = result.pop('cond')
        if cond.shape[-1] != 2048 or cond.shape[1] != 77 * len(tokens['l']):
            raise ValueError('Unexpected SDXL conditioning shape')
        device = clip.patcher.load_device
        parts = []
        for i in range(len(tokens['l'])):
            z = cond[:, i * 77:(i + 1) * 77]
            l = original_emphasis(z[:, :, :768].to(device).contiguous(),
                                  [[w for t, w in tokens['l'][i]]])
            g = original_emphasis(z[:, :, 768:].to(device).contiguous(),
                                  [[w for t, w in tokens['g'][i]]])
            parts.append(torch.cat([l, g], dim=-1).to(cond.device))
        cond = torch.cat(parts, dim=1)
        return ([[cond, result]],)


NODE_CLASS_MAPPINGS = {'ForgeTextEncode': ForgeTextEncode}
NODE_DISPLAY_NAME_MAPPINGS = {'ForgeTextEncode': 'Forge CPU Stream Text Encode'}