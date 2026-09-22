"""Compare Level 3 encode sessions. numpy only. Root pairs auto-match unique exact texts.
python compare_conditioning.py FORGE_ROOT COMFY_ROOT --output comparison_level3.json
python compare_conditioning.py FORGE_ROOT/session_0001 COMFY_ROOT/session_0004
python compare_conditioning.py --list ROOT
"""
import argparse
import json
from pathlib import Path
from compare_traces_v3 import compare, read

STAGES=['prompt','parsed_char_weights','chunk_ids','chunk_weights']


def json_difference(a,b,path=()):
    if type(a) != type(b) and not (isinstance(a,(int,float)) and isinstance(b,(int,float))):
        return dict(path=path,forge=a,comfy=b)
    if isinstance(a,dict):
        for k in list(a)+[k for k in b if k not in a]:
            if k not in a or k not in b:return dict(path=path+(k,),forge=a.get(k),comfy=b.get(k))
            d=json_difference(a[k],b[k],path+(k,))
            if d:return d
    elif isinstance(a,list):
        for i,(x,y) in enumerate(zip(a,b)):
            d=json_difference(x,y,path+(i,))
            if d:return d
        if len(a)!=len(b):return dict(path=path,forge_length=len(a),comfy_length=len(b))
    elif a!=b:return dict(path=path,forge=a,comfy=b)
    return None


def _stages(events):
    return {e['name'][6:]:e['values']['value'] for e in events if e['kind']=='meta' and e['name'].startswith('stage.')}


def _tensor_stage(name,encoder):
    if name=='context.encoder':return 'context.encoder'
    if name=='pooled.first' and encoder=='clip_g':return name
    if not name.startswith('chunk_'):return None
    suffix=name.split('.',1)[1]
    if suffix in {'clip_ids','weights','token_embedding','encoder_input','hidden.selected','hidden.weighted','final_layer_norm'}:return suffix
    if suffix.startswith('layer_') or suffix.startswith('detail.'):return suffix
    if encoder=='clip_g' and suffix in {'pool.raw','pool.pre_projection','pool.projected'}:return suffix
    return None


def compare_pair(a,b):
    ah=json.loads((a/'session.json').read_text(encoding='utf-8'))
    bh=json.loads((b/'session.json').read_text(encoding='utf-8'))
    if ah['kind']!='encode' or bh['kind']!='encode' or ah['encoder']!=bh['encoder']:
        raise ValueError('Select matching encode sessions for the same CLIP encoder')
    full=compare(a,b)
    ae,at=read(a);be,bt=read(b)
    av,bv=_stages(ae),_stages(be)
    js=[]
    for k in STAGES:
        if k not in av or k not in bv:
            js.append(dict(stage=k,status='MISSING',forge_present=k in av,comfy_present=k in bv))
        else:
            d=json_difference(av[k],bv[k]);js.append(dict(stage=k,status='DIFF' if d else 'EXACT',first_difference=d))
    encoder=ah['encoder']
    tensors=[r for r in full['tensors'] if _tensor_stage(r['name'],encoder)]
    # Required observations, independent of whether either file happened to emit them.
    required=['chunk_000.'+n for n in ('clip_ids','weights','token_embedding','encoder_input','layer_000','hidden.selected','hidden.weighted')]+['context.encoder']
    if encoder=='clip_g':required+=['pooled.first','chunk_000.pool.raw','chunk_000.pool.projected']
    missing={lab:[n for n in required if n not in {e['name'] for e in table.values()}] for lab,table in [('forge',at),('comfy',bt)]}
    complete={lab:any(e['name']=='session.complete' for e in ev) for lab,ev in [('forge',ae),('comfy',be)]}
    order=['clip_ids','token_embedding','encoder_input']+[f'layer_{i:03d}' for i in range(256)]+['final_layer_norm','hidden.selected','pool.raw','pool.pre_projection','pool.projected','hidden.weighted','context.encoder','pooled.first']
    def rank(r):
        name=_tensor_stage(r['name'],encoder)
        return (order.index(name) if name in order else -1,r['name'],r['occurrence'])
    numerical=sorted([r for r in tensors if _tensor_stage(r['name'],encoder)!='weights' and not _tensor_stage(r['name'],encoder).startswith('detail.')],key=rank)
    groups={}
    for r in tensors:groups.setdefault(_tensor_stage(r['name'],encoder),[]).append(r)
    summaries=[dict(stage=k,status='EXACT' if all(r['status']=='EXACT' for r in rs) else 'NONEXACT',count=len(rs),
                    first_nonexact=next((r for r in rs if r['status']!='EXACT'),None)) for k,rs in groups.items()]
    diagnostic_only=[r for r in full['tensors'] if not _tensor_stage(r['name'],encoder)]
    return dict(forge=ah,comfy=bh,complete=complete,missing_required=missing,json_stages=js,
                first_json_difference=next((r for r in js if r['status']!='EXACT'),None),
                first_clip_path_difference=next((r for r in numerical if r['status']!='EXACT'),None),
                tensor_stages=summaries,tensors=tensors,diagnostic_only_tensors=diagnostic_only,
                metadata_forge=full['metadata_forge'],metadata_comfy=full['metadata_comfy'],
                interpretation='Stage order is a dependency guide, not a causal proof. Weights branch after CLIP; pooled branches before weighting. Missing data never means equality.')


def index(root):
    return [json.loads(x) for x in (root/'index.jsonl').read_text(encoding='utf-8').splitlines()]


def compare_assemblies(a_root,b_root,a_encode,b_encode):
    """Associate by exact pooled bytes within each app, never by call order.
    Comfy builds context and y at different times, so compare a composite manifest.
    """
    import tempfile
    allowed={'context','pooled','y','y.pooled','y.size'}
    manifests=[];coverage={}
    for label,root,enc in [('forge',a_root,a_encode),('comfy',b_root,b_encode)]:
        _,et=read(enc)
        pools=[v for v in et.values() if v['name']=='pooled.first']
        if len(pools)!=1:return {'status':'UNRESOLVED','reason':'missing or ambiguous pooled.first'}
        key=(pools[0]['raw_sha256'],pools[0]['dtype'],tuple(pools[0]['shape']))
        candidates=[];fields={}
        for h in index(root):
            if h['kind']!='assembly':continue
            folder=root/h['session'];_,table=read(folder)
            pool=next((v for v in table.values() if v['name']=='pooled'),None)
            if pool is None or (pool['raw_sha256'],pool['dtype'],tuple(pool['shape']))!=key:continue
            candidates.append(h)
            for event in table.values():
                if event['name'] in allowed:
                    event=dict(event,file=str((folder/event['file']).resolve()),step=-1,occurrence=0)
                    fields.setdefault(event['name'],[]).append(event)
        events=[];ambiguous=[]
        for name,items in fields.items():
            if len({(e['raw_sha256'],e['dtype'],tuple(e['shape'])) for e in items})!=1:ambiguous.append(name)
            else:events.append(items[0])
        coverage[label]={'candidates':candidates,'ambiguous_fields':ambiguous,'missing_fields':sorted(allowed-{e['name'] for e in events})}
        manifests.append(events)
    with tempfile.TemporaryDirectory(prefix='fc3_compare_') as tmp:
        roots=[]
        for name,events in zip(['forge','comfy'],manifests):
            p=Path(tmp)/name;p.mkdir();roots.append(p)
            (p/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
        result=compare(*roots)
    return {'coverage':coverage,'tensors':result['tensors'],
            'note':'Association uses exact pooled bytes within each engine. Zero-negative, pooled replacement or ambiguity must be resolved manually. This is pre-CFG-batch assembly; compare Level 2 after context repetition.'}


def compare_roots(a,b):
    aa=[x for x in index(a) if x['kind']=='encode'];bb=[x for x in index(b) if x['kind']=='encode']
    def key(x):return (x['encoder'],tuple(x['texts'])) if x.get('texts') and len(x['texts'])==1 else None
    pairs=[];unpaired=[];used=set()
    for x in aa:
        k=key(x);hits=[y for y in bb if key(y)==k] if k is not None else []
        if len(hits)==1 and sum(key(z)==k for z in aa)==1:
            y=hits[0];pair=compare_pair(a/x['session'],b/y['session'])
            if x['encoder']=='clip_g':pair['assembly']=compare_assemblies(a,b,a/x['session'],b/y['session'])
            pairs.append(pair);used.add(y['session'])
        else:unpaired.append(dict(engine='forge',session=x,reason='missing, ambiguous, or different prompt; select sessions explicitly'))
    unpaired.extend(dict(engine='comfy',session=y,reason='unmatched') for y in bb if y['session'] not in used)
    if not pairs and not unpaired:raise ValueError('No encode sessions. Restart apps with tracing enabled and execute text encoding.')
    return dict(pairs=pairs,unpaired=unpaired,warning='No automatic positive/negative assignment; matching uses exact effective text and encoder, never execution order.')


def main():
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='*',type=Path);p.add_argument('--list',type=Path);p.add_argument('--output',type=Path,default=Path('comparison_level3.json'));a=p.parse_args()
    if a.list:
        print(json.dumps(index(a.list),ensure_ascii=False,indent=2));return
    if len(a.paths)!=2:p.error('Provide Forge and Comfy roots, or two encode session directories')
    x,y=a.paths
    result=compare_roots(x,y) if (x/'index.jsonl').exists() else dict(pairs=[compare_pair(x,y)],unpaired=[])
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for pair in result['pairs']:
        print(json.dumps({k:pair[k] for k in ['forge','comfy','complete','missing_required','first_json_difference','first_clip_path_difference']},ensure_ascii=False,indent=2))
    print('Unpaired:',len(result['unpaired']));print('Full result:',a.output)
if __name__=='__main__':main()
