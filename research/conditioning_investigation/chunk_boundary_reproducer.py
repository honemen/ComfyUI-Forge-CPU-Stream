"""Standard-library-only minimal reproduction of the observed boundary rule.
Comfy side is the single large-token-group path, not a general tokenizer clone.
"""
import importlib.util
from pathlib import Path
p = Path(__file__).parent / 'forge_chunks.py'
s=importlib.util.spec_from_file_location('forge_chunks',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def comfy_one_large_group(ids,pad):
    rows=[]
    for i in range(0,len(ids),75):
        part=ids[i:i+75];rows.append([49406]+part+[49407]+[pad]*(75-len(part)))
    return rows
for prefix in [[],[320]*74+[267]]:
    ids=prefix+[320]*73+[267,43746,1774]
    for pad,label in [(49407,'L'),(0,'G')]:
        forge=[[t for t,w in row] for row in m.build_chunks([(ids,'plain',1.)],pad=pad)]
        comfy=comfy_one_large_group(ids,pad);i=1 if prefix else 0
        assert forge[i][75]==49407 and comfy[i][75]==43746
        assert forge[i+1][1:3]==[43746,1774] and comfy[i+1][1]==1774
        print(label,'content tokens',len(ids),'chunk',i,'slot75',forge[i][75],comfy[i][75],
              'tail',forge[i][74:],comfy[i][74:])
print('Minimal actual prompt expression: "a " * 73 + ", steaming body"')
