"""Additive conditioning trace; uses v3's tensor event format without its lifecycle.
FC_TRACE3_DIR must be a NEW directory. No RNG calls, casts on live tensors, or model re-runs.
Ordinary SDXL, one text per encoder call, one process only. Disable by unsetting env.
"""
import contextlib
import contextvars
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import torch
from fc_trace import Trace, _json

_ROOT = os.environ.get('FC_TRACE3_DIR')
_ACTIVE = contextvars.ContextVar('fc3_active', default=None)
_INDEX = 0
_READY = False
_REGISTRY = {}
_LORA = []


def enabled():
    return bool(_ROOT)


def active():
    return _ACTIVE.get()


def _fingerprint(pairs):
    def conv(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().to('cpu', copy=True).contiguous()
            return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(conv(pairs), sort_keys=True).encode()).hexdigest()


class Session(Trace):
    def enabled(self, level=1):
        return self.directory is not None


def _versions():
    out = {}
    for p in ('torch', 'transformers', 'tokenizers', 'numpy', 'safetensors', 'xformers', 'sageattention'):
        try: out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: pass
    return out


@contextlib.contextmanager
def session(engine, kind, encoder, texts=None, **details):
    global _INDEX, _READY
    if not enabled():
        yield None
        return
    if not _READY:
        Path(_ROOT).mkdir(parents=True, exist_ok=False)
        _READY = True
    s = Session()
    s.root = _ROOT
    s.directory = Path(_ROOT) / ('session_%04d' % _INDEX)
    _INDEX += 1
    s.directory.mkdir(exist_ok=False)
    s.chunk = 0
    s.sections = None
    s.embed_row = 0
    s.engine = engine
    s.encoder = encoder
    s.stage_data = {}
    header = dict(session=s.directory.name, engine=engine, kind=kind, encoder=encoder, texts=texts, **details)
    (s.directory / 'session.json').write_text(json.dumps(_json(header), ensure_ascii=False, indent=2), encoding='utf-8')
    with (Path(_ROOT) / 'index.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(_json(header), ensure_ascii=False)+'\n')
    tok = _ACTIVE.set(s)
    s.meta('session', **header)
    s.meta('logger.sources', files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(inspect.getsourcefile(Trace)))})
    s.meta('runtime', versions=_versions(), matmul_precision=torch.get_float32_matmul_precision(),
           matmul_tf32=torch.backends.cuda.matmul.allow_tf32, deterministic=torch.are_deterministic_algorithms_enabled())
    s.meta('lora.load_history', events=_LORA)
    try:
        yield s
    except BaseException as e:
        s.meta('session.error', error=repr(e))
        raise
    else:
        s.meta('session.complete', ok=True)
    finally:
        _ACTIVE.reset(tok)


def meta(name, **values):
    if active(): active().meta(name, **values)


def put(name, value):
    if active(): active().put(name, value)


def stage(name, value):
    if active():
        active().stage_data[name] = _json(value)
        active().meta('stage.' + name, value=value)


def rows(name, value, start=None):
    s = active()
    if s is None or value is None: return
    if not isinstance(value, torch.Tensor): return
    start = (s.chunk if s.engine == 'forge' else 0) if start is None else start
    for i in range(value.shape[0]):
        n = start + i
        label = 'empty' if s.sections is not None and n >= s.sections else 'chunk_%03d' % n
        s.put(label + '.' + name, value[i:i+1])


def parsed(value, unescape=None):
    if not active(): return
    # JSON keeps original segmentation; canonical character/weight stream ignores segment boundaries only.
    meta('parsed.native', segments=value)
    value = [(unescape(t) if unescape else t, w) for t,w in value]
    stage('parsed_char_weights', [[c,w] for text,w in value for c in text])


def token_chunks(chunks, tokenizer=None, end=None, pad=None):
    if not active(): return
    ids, weights, strings = [], [], []
    for chunk in chunks:
        row_ids, row_w, row_s = [], [], []
        for pair in chunk:
            t,w = pair[:2]
            if isinstance(t, int):
                row_ids.append(t)
                row_s.append(tokenizer.convert_ids_to_tokens(t) if tokenizer is not None else None)
            else:
                row_ids.append({'embedding_sha256': _fingerprint([t])})
                row_s.append('<embedding>')
            row_w.append(w)
        ids.append(row_ids); weights.append(row_w); strings.append(row_s)
    meta('tokens.native_chunk_ids', chunks=ids)
    if end is not None and pad is not None and end != pad:
        ids = [row[:row.index(end)+1] + [pad]*(len(row)-row.index(end)-1) if end in row else row for row in ids]
    stage('chunk_ids', ids)
    stage('chunk_weights', weights)
    meta('token_strings', chunks=strings)


def _source(obj):
    try:
        p = inspect.getsourcefile(obj)
        if p: meta('source', path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
    except (TypeError, OSError): pass


def _tensor_summary(t):
    if t is None: return None
    x = t.detach().to('cpu', copy=True).contiguous()
    return dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device),
                sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def _weight_manifest(model):
    # Stored parameters, not guaranteed effective online/manual-cast weights.
    return {k.replace('.wrapped.', '.'): _tensor_summary(v) for k,v in model.named_parameters()}


@contextlib.contextmanager
def hooks(transformer):
    s = active()
    if s is None:
        yield
        return
    handles = []
    tm = transformer.text_model
    _source(type(transformer)); _source(type(tm)); _source(type(tm.encoder.layers[0]))
    params = next(transformer.parameters())
    meta('clip.config', transformer_class=type(transformer).__module__+'.'+type(transformer).__name__,
         parameter_dtype=str(params.dtype), parameter_device=str(params.device), layers=len(tm.encoder.layers),
         attention_classes=[type(getattr(l,'self_attn',None)).__module__+'.'+type(getattr(l,'self_attn',None)).__name__ for l in tm.encoder.layers],
         hf_config=getattr(getattr(transformer, 'config', None), 'to_dict', lambda: {})())
    if os.environ.get('FC_TRACE3_WEIGHTS') == '1':
        meta('weights.stored', parameters=_weight_manifest(transformer))
    def embedding_hook(module, args, out):
        start = s.chunk if s.engine == 'forge' else s.embed_row
        if args and isinstance(args[0], torch.Tensor): rows('clip_ids.runtime', args[0], start)
        rows('token_embedding', out, start)
        if s.engine == 'comfy': s.embed_row += out.shape[0]
    handles.append(tm.embeddings.token_embedding.register_forward_hook(embedding_hook))
    def enc_pre(module, args, kwargs):
        x = args[0] if args else kwargs.get('inputs_embeds', kwargs.get('x'))
        rows('encoder_input', x)
        # Causal/padding masks have different native shapes; preserve without claiming shape equivalence.
        for key in ('mask', 'attention_mask', 'causal_attention_mask'):
            if isinstance(kwargs.get(key), torch.Tensor): put('native.'+key, kwargs[key])
        meta('encoder.batch', shape=list(x.shape), dtype=str(x.dtype), device=str(x.device))
    handles.append(tm.encoder.register_forward_pre_hook(enc_pre, with_kwargs=True))
    def block_pre(module, args, kwargs):
        meta('attention.runtime', callables=[x for x in list(args)+list(kwargs.values()) if callable(x)])
    handles.append(tm.encoder.layers[0].register_forward_pre_hook(block_pre, with_kwargs=True))
    selected = os.environ.get('FC_TRACE3_LAYERS', 'all')
    indices = set(range(len(tm.encoder.layers))) if selected == 'all' else {int(x) for x in selected.split(',') if x.strip()}
    def layer_hook(i):
        def hook(module, args, out): rows('layer_%03d' % i, out[0] if isinstance(out, (tuple,list)) else out)
        return hook
    detail = {int(x) for x in os.environ.get('FC_TRACE3_DETAIL_LAYERS', '').split(',') if x.strip()}
    def module_pre(name):
        def hook(module, args):
            if args and isinstance(args[0], torch.Tensor): rows(name + '.input', args[0])
        return hook
    def module_post(name):
        def hook(module, args, out):
            if isinstance(out, torch.Tensor): rows(name + '.output', out)
        return hook
    for i, layer in enumerate(tm.encoder.layers):
        if i in indices: handles.append(layer.register_forward_hook(layer_hook(i)))
        if i in detail:
            for name, module in layer.named_modules():
                if name in ('layer_norm1','layer_norm2','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.out_proj','mlp.fc1','mlp.fc2'):
                    label = 'detail.layer_%03d.%s' % (i,name)
                    meta(label+'.module', cls=type(module).__module__+'.'+type(module).__name__)
                    handles.append(module.register_forward_pre_hook(module_pre(label)))
                    handles.append(module.register_forward_hook(module_post(label)))
    handles.append(tm.final_layer_norm.register_forward_hook(lambda m,a,o: rows('final_layer_norm', o)))
    projection = getattr(transformer, 'text_projection', None)
    if isinstance(projection, torch.nn.Module):
        handles.append(projection.register_forward_pre_hook(lambda m,a: rows('pool.pre_projection', a[0])))
        handles.append(projection.register_forward_hook(lambda m,a,o: rows('pool.projected', o)))
    try: yield
    finally:
        for h in handles: h.remove()


def forge_encoder(fn):
    @functools.wraps(fn)
    def wrapped(self, texts, *a, **kw):
        if not enabled(): return fn(self, texts, *a, **kw)
        if len(texts) != 1: raise ValueError('Level 3 supports one text per Forge CLIP call')
        with session('forge','encode',self.embedding_key,list(texts), emphasis=self.emphasis.name) as s:
            _source(fn)
            stage('prompt', texts[0]); s.sections = None
            meta('selection', clip_skip=self.clip_skip, minimal_clip_skip=self.minimal_clip_skip,
                 selected_hidden=-max(self.clip_skip,self.minimal_clip_skip), final_layer_norm=self.final_layer_norm,
                 projected=self.text_projection, pooled=self.return_pooled)
            with hooks(self.text_encoder.transformer): result = fn(self,texts,*a,**kw)
            z,pool = result if isinstance(result,tuple) else (result,None)
            put('context.encoder',z)
            if pool is not None: put('pooled.first',pool)
            return result
    return wrapped


def forge_chunk(fn):
    @functools.wraps(fn)
    def wrapped(self, tokens, weights, *a, **kw):
        if not active(): return fn(self,tokens,weights,*a,**kw)
        s=active()
        rows('weights',torch.as_tensor(weights))
        result=fn(self,tokens,weights,*a,**kw)
        rows('hidden.weighted',result)
        s.chunk += 1
        return result
    return wrapped


def comfy_tokenizer(fn):
    @functools.wraps(fn)
    def wrapped(self,text,*a,**kw):
        if not enabled(): return fn(self,text,*a,**kw)
        with session('comfy','tokenize',self.embedding_key,[text]) as s:
            _source(fn)
            stage('prompt',text)
            result=fn(self,text,*a,**kw)
            pairs=[[(p[0],p[1]) for p in row] for row in result]
            token_chunks(pairs,self.tokenizer)
            key=(self.embedding_key,_fingerprint(pairs))
            entry=dict(text=text, stages=dict(s.stage_data), session=s.directory.name)
            _REGISTRY.setdefault(key,[]).append(entry)
            return result
    return wrapped


def comfy_encoder(fn):
    @functools.wraps(fn)
    def wrapped(self,pairs,*a,**kw):
        if not enabled(): return fn(self,pairs,*a,**kw)
        encoder='clip_g' if self.special_tokens.get('pad') == 0 else 'clip_l'
        key=(encoder,_fingerprint([[(p[0],p[1]) for p in row] for row in pairs]))
        entries=_REGISTRY.get(key,[])
        texts=list(dict.fromkeys(e['text'] for e in entries))
        with session('comfy','encode',encoder,texts or None, tokenizer_matches=[e['session'] for e in entries]) as s:
            _source(fn)
            s.sections=len(pairs)
            if len(texts)==1:
                for k,v in entries[-1]['stages'].items(): stage(k,v)
            else:
                meta('alignment.warning', reason='tokenizer input missing or multiple texts map to same tokens; select sessions explicitly')
                token_chunks(pairs)
            meta('selection', layer=self.layer, layer_idx=self.layer_idx, final_layer_norm=self.layer_norm_hidden_state,
                 projected=self.return_projected_pooled, masks=self.enable_attention_masks)
            for i,row in enumerate(pairs): rows('weights',torch.tensor([[p[1] for p in row]]),i)
            with hooks(self.transformer): result=fn(self,pairs,*a,**kw)
            put('context.encoder',result[0])
            if result[1] is not None: put('pooled.first',result[1])
            return result
    return wrapped


def comfy_clip_input(tokens):
    s=active()
    if not s:return
    for i,row in enumerate(tokens):
        if all(isinstance(t,int) for t in row): rows('clip_ids',torch.tensor([row],dtype=torch.long),i)
        else: meta('clip_ids.embedding',row=i,fingerprint=_fingerprint(row))


def assembly(engine, context, pooled=None, vector=None, **details):
    if not enabled(): return
    with session(engine,'assembly','sdxl',None,**details):
        put('context',context)
        if pooled is not None: put('pooled',pooled)
        if vector is not None:
            put('y',vector);put('y.pooled',vector[:,:1280]);put('y.size',vector[:,1280:])


def lora(engine, strength_clip, loaded_keys, patches, **details):
    if not enabled(): return
    keys=sorted(str(k) for k in loaded_keys)
    def desc(v):
        if isinstance(v,(tuple,list)): return [desc(x) for x in v]
        if isinstance(v,torch.Tensor):return _tensor_summary(v)
        if isinstance(v,(int,float,str,bool)) or v is None:return v
        return dict(type=type(v).__module__+'.'+type(v).__name__,
                    weights=desc(v.weights) if hasattr(v, 'weights') else None)
    _LORA.append(dict(engine=engine,strength_clip=strength_clip,loaded_keys=keys,
                      patches={str(k):desc(v) for k,v in patches.items() if str(k) in keys},**details))
    with session(engine,'lora','clip',None):meta('lora.loaded',**_LORA[-1])
