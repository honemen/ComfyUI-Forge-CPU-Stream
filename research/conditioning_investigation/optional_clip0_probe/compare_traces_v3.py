"""Compare two pass directories. Requires numpy, not torch.

python compare_traces.py FORGE/pass_000 COMFY/pass_000 --output comparison.json
No tolerance is used to decide exact equality. NaN/Inf is never a PASS.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    events = [json.loads(x) for x in (path / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    tensors = {}
    for e in events:
        if e['kind'] == 'tensor':
            key = (e['step'], e['name'], e['occurrence'])
            if key in tensors:
                raise ValueError('Duplicate tensor key: ' + str(key))
            tensors[key] = e
    return events, tensors


def compare(a_dir, b_dir):
    ae, at = read(a_dir)
    be, bt = read(b_dir)
    rows = []
    # Dependency order comes from the observed Forge event order, not filenames.
    keys = list(at) + [k for k in bt if k not in at]
    for key in keys:
        a, b = at.get(key), bt.get(key)
        row = dict(step=key[0], name=key[1], occurrence=key[2])
        if a is None or b is None:
            row.update(status='MISSING', missing='forge' if a is None else 'comfy')
        elif a['shape'] != b['shape']:
            row.update(status='SHAPE', forge_shape=a['shape'], comfy_shape=b['shape'])
        else:
            av = np.load(a_dir / a['file'], allow_pickle=False)
            bv = np.load(b_dir / b['file'], allow_pickle=False)
            finite = bool(np.isfinite(av).all() and np.isfinite(bv).all())
            same_values = bool(np.array_equal(av, bv))
            same_dtype = a['dtype'] == b['dtype']
            same_bytes = same_dtype and a['raw_sha256'] == b['raw_sha256']
            row.update(status=('NONFINITE' if not finite else 'EXACT' if same_bytes else 'DIFF'),
                       equal_values=same_values, equal_dtype=same_dtype, equal_bytes=same_bytes,
                       forge_dtype=a['dtype'], comfy_dtype=b['dtype'],
                       forge_device=a['device'], comfy_device=b['device'],
                       forge_stride=a.get('stride'), comfy_stride=b.get('stride'))
            if finite and av.size:
                af, bf = av.astype(np.float64), bv.astype(np.float64)
                delta = af - bf
                reference_norm = float(np.linalg.norm(af.ravel()))
                delta_norm = float(np.linalg.norm(delta.ravel()))
                row.update(max_abs=float(np.max(np.abs(delta))),
                           rms=float(np.sqrt(np.mean(delta * delta))),
                           relative_l2=(delta_norm / reference_norm if reference_norm
                                        else 0.0 if delta_norm == 0 else None),
                           unequal_elements=int(np.count_nonzero(av != bv)))
                if not same_values:
                    index = tuple(int(x) for x in np.argwhere(av != bv)[0])
                    row.update(first_index=index, forge_value=float(af[index]), comfy_value=float(bf[index]))
        rows.append(row)
    differing = [r for r in rows if r['status'] != 'EXACT']
    required = {'eps0', 'sigmas.base', 'sigmas.sampler', 'sigmas.er',
                'x0', 'step.x_in', 'step.denoised', 'step.x_out', 'noise'}
    missing_core = {label: sorted(required - {e['name'] for e in table.values()})
                    for label, table in [('forge', at), ('comfy', bt)]}
    step_diffs = [r for r in differing if r['step'] >= 0]
    noise_rows = [r for r in rows if r['name'] == 'noise']
    # First observed mismatch is not automatically a causal attribution.
    return dict(first_observed_mismatch=differing[0] if differing else None,
                first_step_mismatch=step_diffs[0] if step_diffs else None,
                missing_required_core=missing_core,
                noise_events_compared=len(noise_rows),
                noise_events_nonexact=sum(r['status'] != 'EXACT' for r in noise_rows),
                tensor_count_forge=len(at), tensor_count_comfy=len(bt),
                nonexact_count=len(differing), tensors=rows,
                metadata_forge=[e for e in ae if e['kind'] == 'meta'],
                metadata_comfy=[e for e in be if e['kind'] == 'meta'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('forge', type=Path)
    p.add_argument('comfy', type=Path)
    p.add_argument('--output', type=Path, default=Path('comparison.json'))
    args = p.parse_args()
    result = compare(args.forge, args.comfy)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['tensors', 'metadata_forge', 'metadata_comfy']}, ensure_ascii=False, indent=2))
    print('Full result:', args.output)
