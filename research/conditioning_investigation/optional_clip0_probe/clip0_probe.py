"""Focused layer-0 observer. Records actual dense linear/LayerNorm operands.
Requires eager dense CLIP. No extra model evaluation or RNG draws.
Called only inside the existing fc_trace3.hooks scope. Effective weight hashes for
8 layer-0 modules; full effective q_proj weight only by default (~18 MiB per engine).
"""
import hashlib
import os
import torch
from torch.utils._python_dispatch import TorchDispatchMode

NAMES=('layer_norm1','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj',
       'self_attn.out_proj','layer_norm2','mlp.fc1','mlp.fc2')


def fingerprint(t):
    if t is None:return None
    if t.device.type=='meta':return {'unavailable':'meta parameter'}
    a=t.detach().to('cpu',copy=True).contiguous()
    return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'stride':list(t.stride()),
            'raw_sha256':hashlib.sha256(a.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            'float32_sha256':hashlib.sha256(a.float().numpy().tobytes()).hexdigest()}


class Operands(TorchDispatchMode):
    def __init__(self,module,name,trace):
        super().__init__();self.module=module;self.name=name;self.trace=trace;self.matches=0
    def record(self,op,w,b,**info):
        self.matches+=1
        self.trace.meta('clip0.'+self.name+'.effective',op=op,weight=fingerprint(w),bias=fingerprint(b),**info)
        selected=os.environ.get('FC_CLIP0_SAVE_WEIGHTS','self_attn.q_proj').split(',')
        if self.name in selected:
            self.trace.put('clip0.'+self.name+'.effective_weight',w)
            if b is not None:self.trace.put('clip0.'+self.name+'.effective_bias',b)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        name=str(func)
        if name=='aten.native_layer_norm.default':
            self.trace.put('clip0.'+self.name+'.kernel_input',args[0][:1])
            self.record(name,args[2],args[3],eps=args[4],input_dtype=str(args[0].dtype),input_shape=list(args[0].shape))
        elif name in ('aten.addmm.default','aten.mm.default') and hasattr(self.module,'in_features'):
            x,w=(args[1],args[2]) if name=='aten.addmm.default' else (args[0],args[1])
            # Exclude LoRA's up@down calculation; match the actual activation GEMM.
            if w.ndim==2 and tuple(w.shape)==(self.module.in_features,self.module.out_features) and x.shape[-1]==self.module.in_features:
                self.trace.put('clip0.'+self.name+'.kernel_input',x[:77].unsqueeze(0))
                self.record(name,w.t(),args[0] if name=='aten.addmm.default' else None,
                            input_dtype=str(x.dtype),input_shape=list(x.shape),input_stride=list(x.stride()),gemm_options=kwargs)
        return func(*args,**kwargs)


def attach(transformer,trace):
    handles=[];layer=transformer.text_model.encoder.layers[0]
    trace.meta('clip0.attention_config',implementation=getattr(getattr(layer.self_attn,'config',None),'_attn_implementation',None),
               cls=type(layer.self_attn).__module__+'.'+type(layer.self_attn).__name__)
    def observe(name,module):
        stack=[]
        def pre(m,args):
            trace.put('clip0.'+name+'.input',args[0][:1])
            trace.meta('clip0.'+name+'.stored',weight=fingerprint(getattr(m,'weight',None)),bias=fingerprint(getattr(m,'bias',None)),
                       cls=type(m).__module__+'.'+type(m).__name__,full_input_shape=list(args[0].shape))
            mode=Operands(m,name,trace);mode.__enter__();stack.append(mode)
        def post(m,args,out):
            if not stack:return
            mode=stack.pop();mode.__exit__(None,None,None)
            trace.meta('clip0.'+name+'.coverage',effective_operand_events=mode.matches)
            if isinstance(out,torch.Tensor):trace.put('clip0.'+name+'.output',out[:1])
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post,always_call=True))
    for name,module in layer.named_modules():
        if name in NAMES:observe(name,module)
    return handles
