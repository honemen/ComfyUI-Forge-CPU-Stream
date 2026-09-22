"""Read header and SHA256 only; never load a model. Python standard library only."""
import argparse,hashlib,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('file',type=Path);p.add_argument('--output',type=Path,default=Path('safetensors_inventory.json'));a=p.parse_args()
with a.file.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0]
 if n>100_000_000:raise ValueError('Unexpectedly large header')
 header=json.loads(f.read(n))
with a.file.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
keys={k:{'dtype':v['dtype'],'shape':v['shape']} for k,v in header.items() if k!='__metadata__'}
result={'filename':a.file.name,'sha256':sha,'tensors':keys,'metadata':header.get('__metadata__',{})}
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('Tensor count:',len(keys));print('SHA256:',sha);print(a.output)
