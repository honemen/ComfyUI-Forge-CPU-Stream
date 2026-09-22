"""Compare two selected NEGATIVE CLIP encode session folders. numpy only."""
import argparse,json
from pathlib import Path
from compare_traces_v3 import compare
NAMES=('layer_norm1','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.out_proj','layer_norm2','mlp.fc1','mlp.fc2')
def run(a,b):
 full=compare(a,b)
 rows=[r for r in full['tensors'] if r['name'].startswith('clip0.')]
 meta=[full['metadata_forge'],full['metadata_comfy']]
 weights=[];missing=[]
 for name in NAMES:
  es=[[e['values'] for e in m if e['name']=='clip0.'+name+'.effective'] for m in meta]
  if any(len(x)!=1 for x in es):
   missing.append({'module':name,'effective_event_counts':[len(x) for x in es]});continue
  x,y=es[0][0],es[1][0]
  r={'module':name,'forge':x,'comfy':y}
  for field in ['weight','bias']:
   f,c=x[field],y[field]
   r[field+'_status']=('BOTH_NONE' if f is None and c is None else 'MISSING' if f is None or c is None or 'raw_sha256' not in f or 'raw_sha256' not in c
     else 'EXACT' if f['dtype']==c['dtype'] and f['shape']==c['shape'] and f['raw_sha256']==c['raw_sha256'] else 'DIFF')
   r[field+'_float32_values_equal']=(f is not None and c is not None and f.get('shape')==c.get('shape') and f.get('float32_sha256') is not None and f.get('float32_sha256')==c.get('float32_sha256'))
  weights.append(r)
 order=[f'clip0.{name}.{point}' for name in NAMES for point in ['input','kernel_input','output']]
 activations=sorted([r for r in rows if r['name'] in order],key=lambda r:order.index(r['name']))
 absent={}
 # Required observations must exist on both sides; MISSING rows alone cannot detect absent-in-both.
 from compare_traces_v3 import read
 for label,folder in [('forge',a),('comfy',b)]:
  events,table=read(folder);absent[label]=[n for n in order if n not in {e['name'] for e in table.values()}]
 return {'effective_weights':weights,'unresolved_coverage':missing,'missing_activations':absent,
         'first_activation_difference':next((r for r in activations if r['status']!='EXACT'),None),
         'tensors':rows,'note':'Compare full input shapes/dtypes/eps too. Equal first-sample values with different batch shapes do not prove identical kernel execution. Stored parameters are not the effective operands.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('forge',type=Path);p.add_argument('comfy',type=Path);p.add_argument('--output',type=Path,default=Path('clip0_comparison.json'));a=p.parse_args();r=run(a.forge,a.comfy)
 a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k!='tensors' and k!='effective_weights'},ensure_ascii=False,indent=2));print(a.output)
