# Research material and instrumentation

This directory preserves a specific Forge Neo ↔ ComfyUI investigation as a handoff for review and extension. It is **not** a finished general purpose debugger or a validated universal measurement method. The investigation focused specifically on **ER SDE**. Other samplers have not been systematically investigated or validated; the selected measurement points may be inappropriate for them and equivalent observations cannot be promised. Main measured setup: **silvermoonmix v60 (V-Pred) + ER SDE**. The nodes are not intentionally checkpoint restricted, but other checkpoints have not been systematically tested. Noise/RNG alignment did **not** establish identical final images; conditioning/text encoding was examined afterward.

The intended path through the material is **Level 1–2 (sampler/noise and CFG/UNet) → Level 3 (text/conditioning) → conditioning investigation (focused follow ups)**. These stages are investigative observations, not proof that the first observed numerical mismatch is the cause of every downstream mismatch. Large raw trace logs are not included; summary JSON and source files are. Reproducing comparisons requires generating new traces on an explicitly checked source revision.

## `level1-2/`: sampling observations

| File | Purpose confirmed in source |
| --- | --- |
| `fc_trace.py` | Shared tensor/event logger; copies tensors to NumPy files and writes `events.jsonl`, metadata, runtime and source hashes to a fresh `pass_NNN` directory. `FC_TRACE_DIR` enables it; `FC_TRACE_LEVEL=1` records solver points, `2` adds CFG/UNet packets; `FC_TRACE_STEPS` selects Level 2 steps (default `0`, or `all`). Its stated scope is one process, single GPU, batch 1, ordinary txt2img. |
| `patches/forge_level1.patch`, `comfy_level1.patch` | Hook initial noise, sigma schedule, ER SDE solver state, per step denoised/latent, and noise draws in the respective application sources. |
| `patches/forge_level2.patch`, `comfy_level2.patch` | Add CFG outputs and labelled conditional/unconditional UNet input/output packet observations. Require the corresponding Level 1 logger imports. |
| `compare_traces.py` | Compares two `pass_NNN` directories by step/name/occurrence; reports exact bytes, value/shape/dtype differences and first **observed** mismatch to an output JSON. Requires NumPy. Missing observations are reported, not treated as matches. |
| `probe_sigma_precision.py` | Isolated synthetic CPU schedule calculation illustrating dtype order differences; needs PyTorch, NumPy, SciPy. Explicitly does not establish that actual run schedules differ in the same way. |
| `patches/baseline_normalized_sha256.json` | Normalized hashes of the Forge and ComfyUI baseline source files used to prepare these patches. |

To inspect a newly generated run, place `fc_trace.py` where each app's Python import resolves it, inspect patch targets and baseline hashes against **your exact source**, apply only matching research patches, set a fresh `FC_TRACE_DIR` for each app and run the same controlled ER SDE case. For example, `python level1-2/compare_traces.py FORGE/pass_000 COMFY/pass_000 --output comparison.json`. Both the patch hunks and the logger were written around this experiment; trace overhead and changed source versions can affect the investigation.

## `level3/`: text/conditioning observations

| File | Purpose confirmed in source |
| --- | --- |
| `fc_trace3.py` | Additional encode/chunk, CLIP, assembly and optional LoRA observation sessions (`session_NNNN`, `session.json`, `index.jsonl`, tensor events). Imports `Trace` from `fc_trace.py`. Enable with a **new** `FC_TRACE3_DIR`; `FC_TRACE3_LAYERS`, `FC_TRACE3_DETAIL_LAYERS`, and `FC_TRACE3_WEIGHTS=1` govern CLIP observation detail. Its stated scope is ordinary SDXL, one text per encoder call, one process. |
| `patches/forge_level3.patch`, `comfy_level3.patch` | Instrument Forge and ComfyUI parsing, tokens, CLIP stages and SDXL context/pooled/vector assembly respectively. These are not node installation steps. |
| `patches/forge_level3_lora.patch`, `comfy_level3_lora.patch` | Optional LoRA load metadata observation. |
| `compare_traces_v3.py`, `compare_conditioning.py` | NumPy based session tensor comparison and higher level paired encode/session comparison. The latter has `--list ROOT`, root pairing by unique exact text and encoder, or explicit session pair inputs; it reports missing stages and cannot infer positive/negative roles from execution order. |
| `inspect_safetensors.py` | Standard library only safetensors header inventory and file SHA256 without loading model tensors. |
| `observed_v3_summary.json`, `parser_source_probe.json` | Saved observation summary (including recorded noise comparison rows) and prompt parser example outputs. They are evidence from selected runs, not a universal result. |
| `verification.json`, `KIT_SHA256.json`, `patches/baseline_normalized_sha256.json` | Recorded verification scope and explicit untested cases; package file hashes; baseline source hashes. In particular, `verification.json` does **not** claim the user's GPU/checkpoint/LoRA/prompt Level 3 execution was tested in that verification pass. |

`fc_trace3.py` depends on `fc_trace.py` being importable. A sample comparison invocation is `python level3/compare_conditioning.py FORGE_ROOT COMFY_ROOT --output comparison_level3.json`. Trace folder creation requires a new path; do not reuse an existing trace root. Individual patch applicability must be checked against the actual application revision first.

## `conditioning_investigation/`: focused follow ups

| File | Purpose confirmed in source |
| --- | --- |
| `evidence.json`, `validation.json` | Selected L/G positive/negative token, embedding, layer and LoRA comparisons; chunk boundary replay, counterfactual CPU emphasis checks, and stated limits. `validation.json` explicitly says CPU reduction is not proof of GPU bit equality. |
| `forge_level3_hotfixed_for_reverse.patch` | Variant Forge Level 3 trace patch changing the SDXL assembly metadata keyword to `prompt_texts`; retained for the reverse/check investigation, not a replacement for all revisions. |
| `chunk_boundary_reproducer.py` | Minimal standard library replay of the chunk boundary rule versus a single large Comfy token group. | `chunk_boundary_reproducer.py` | Minimal standard library replay of the chunk boundary rule versus a single large Comfy token group. It uses the colocated `forge_chunks.py` to reproduce the Forge-side chunk construction. |

| `optional_clip0_probe/clip0_probe.py`, `clip0_only.patch` | Optional CLIP layer 0 operand observer and patch to `fc_trace3.py`; controls include `FC_CLIP0_PROBE`, `FC_TRACE3_TARGET_JSON`, and `FC_CLIP0_SAVE_WEIGHTS`. It requires eager dense CLIP and is focused on specific layer 0 operations. |
| `optional_clip0_probe/compare_clip0.py`, `compare_traces_v3.py` | Focused negative encode comparison and a local copy of the pass comparator. Requires NumPy. |
| `optional_clip0_probe/baseline.json`, `negative_prompt.json` | Expected normalized hash for the Level 3 logger before optional patching and a JSON string prompt selection. |
| `SHA256.json` | Hash manifest for the investigation files. |

The optional probe may emit sizable effective weights. The JSON evidence narrows the observed conditions but does not identify a universal cause or establish final image equivalence.

## Patch safety, attribution, and further work

All patches target source layouts from the time of the investigation. **There is no guarantee they apply unchanged to current ComfyUI or Forge Neo.** Check versions and baseline hashes, inspect `git apply --check` and review every hunk against the intended source before attempting an experiment; keep working trees recoverable. The patch set is research instrumentation, not an official verification suite. The author does not claim these tracing methods are exhaustive or academically validated. Independent verification, corrections, better measurement points, and investigations of other samplers are welcome.

The general and Minimal `forge_chunks.py` files credit Forge Classic/Neo [parsing source at commit `0c9273f`](https://github.com/Haoming02/sd-webui-forge-classic/blob/0c9273f69dbe491246fb11c3d4ac70123099d393/backend/text_processing/parsing.py). Preserve their source comments. The package includes patches against upstream Forge and ComfyUI source and **contains no LICENSE or NOTICE**. Before publication, verify the licenses of the exact referenced upstream revisions and derived portions; document provenance and applicable notices in an appropriate LICENSE/NOTICE set. The included baseline hashes are not license metadata. Do not infer a package license from this README.

If you know an existing node or implementation offering equivalent or improved behavior, more complete Forge Neo alignment, or bidirectional ComfyUI ↔ Forge Neo compatibility, please open an Issue with a link. Additional experiments, alternate environments, corrections and extensions can build on this handoff.
