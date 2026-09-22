"""Opt-in SDXL TextEncode: Forge chunking, optionally Forge Original emphasis.
No global monkeypatch. Remove this folder and restore standard CLIPTextEncode to roll back.
"""
import json
import os
import re
import torch
from .forge_chunks import tokenize_forge


def original_emphasis(z, weights):
    mult=torch.asarray(weights).to(z)
    original_mean=z.mean()
    out=z*mult.reshape(mult.shape+(1,)).expand(z.shape)
    return out*(original_mean/out.mean())


def trace_allowed(text):
    if not os.environ.get('FC_TRACE3_DIR'):return False
    target=os.environ.get('FC_TRACE3_TARGET_JSON')
    if target:
        from pathlib import Path
        return text==json.loads(Path(target).read_text(encoding='utf-8-sig'))
    return True


def trace_tokens(text, tokenizer, pairs, parsed):
    if not trace_allowed(text):return
    import fc_trace3 as t
    # Compatibility bridge to the supplied v4 Level 3 logger; no inference changes.
    with t.session('comfy','tokenize',tokenizer.embedding_key,[text],adapter='forge_chunks') as s:
        t.stage('prompt',text);t.parsed(parsed);t.token_chunks(pairs,tokenizer.tokenizer)
        key=(tokenizer.embedding_key,t._fingerprint(pairs))
        t._REGISTRY.setdefault(key,[]).append(dict(text=text,stages=dict(s.stage_data),session=s.directory.name))


class ForgeTextEncodeMinimal:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'clip':('CLIP',),'text':('STRING',{'multiline':True,'dynamicPrompts':True}),
                            'mode':(['chunk_only','chunk_and_original'],)}}
    RETURN_TYPES=('CONDITIONING','STRING')
    RETURN_NAMES=('conditioning','token_report')
    FUNCTION='encode'
    CATEGORY='conditioning/Forge compatibility'

    def encode(self,clip,text,mode):
        if mode not in ('chunk_only','chunk_and_original'):raise ValueError(mode)
        from comfy.sdxl_clip import SDXLClipModel
        if not isinstance(clip.cond_stage_model,SDXLClipModel):raise ValueError('Ordinary SDXL CLIP-L/G required')
        if re.search(r'<lora:|embedding:',text,re.I):
            raise ValueError('Use the LoRA node; remove Forge LoRA tags. Textual Inversion is outside this minimal adapter.')
        if re.search(r'\bAND\b',text):raise ValueError('AND prompt composition is not supported by this minimal adapter')
        tokens={};parsed_by_name={};report={'mode':mode,'encoders':{}}
        for name in ['l','g']:
            tok=getattr(clip.tokenizer,'clip_'+name)
            pairs,parsed=tokenize_forge(text,tok);tokens[name]=pairs;parsed_by_name[name]=parsed
            report['encoders'][name]={'ids':[[p[0] for p in row] for row in pairs],
                                      'weights':[[p[1] for p in row] for row in pairs]}
        if len(tokens['l'])!=len(tokens['g']):raise ValueError('Different L/G chunk counts need a separate SDXL assembly adapter')
        actual=tokens if mode=='chunk_only' else {k:[[(t,1.0) for t,w in row] for row in v] for k,v in tokens.items()}
        for name in ['l','g']:
            tok=getattr(clip.tokenizer,'clip_'+name)
            # For Original mode this records the actual unweighted extraction call.
            trace_tokens(text,tok,actual[name],parsed_by_name[name])
        result=clip.encode_from_tokens(actual,return_dict=True)
        cond=result.pop('cond')
        if mode=='chunk_and_original':
            if cond.shape[-1]!=2048 or cond.shape[1]!=77*len(tokens['l']):raise ValueError('Unexpected SDXL conditioning shape')
            # Match Forge's per-encoder, per-chunk reduction, on the CLIP execution device.
            device=clip.patcher.load_device
            parts=[]
            for i in range(len(tokens['l'])):
                z=cond[:,i*77:(i+1)*77]
                l=original_emphasis(z[:,:,:768].to(device).contiguous(),[[w for t,w in tokens['l'][i]]])
                g=original_emphasis(z[:,:,768:].to(device).contiguous(),[[w for t,w in tokens['g'][i]]])
                parts.append(torch.cat([l,g],dim=-1).to(cond.device))
            cond=torch.cat(parts,dim=1)
            if trace_allowed(text):
                import fc_trace3 as t
                with t.session('comfy','adapter_result','sdxl',[text],mode=mode):
                    t.put('context',cond);t.put('pooled',result.get('pooled_output'))
        return ([[cond,result]],json.dumps(report,ensure_ascii=False))

NODE_CLASS_MAPPINGS={'ForgeTextEncodeMinimal':ForgeTextEncodeMinimal}
NODE_DISPLAY_NAME_MAPPINGS={'ForgeTextEncodeMinimal':'Forge CPU Stream Text Encode (Minimal)'}