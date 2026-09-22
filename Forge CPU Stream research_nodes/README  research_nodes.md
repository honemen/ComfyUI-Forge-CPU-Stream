# Forge CPU Stream — research nodes

These experimental ComfyUI nodes expose a narrower ER SDE setup and token reports for investigating observed Forge Neo versus ComfyUI differences. They are research prototypes, not a complete compatibility layer. For ordinary use, start with [`../Forge CPU Stream一般版/README.md`](../Forge%20CPU%20Stream一般版/README.md). The [research guide](../research/README.md) describes the separate instrumentation.

## Install and graph

Copy `forge_cpu_stream/` and `forge_conditioning_minimal/` into ComfyUI's `custom_nodes/`, keeping each `__init__.py` beside its accompanying `forge_chunks.py` where present. Restart ComfyUI. The research adapter uses ComfyUI's sampler API and PyTorch. Its text node does not require a global monkeypatch for normal encoding. The supplied JSON expects the checkpoint filename `silvermoonmix_v60VPred.safetensors`; the model is not included.

Open [`Forge CPU Stream　研究版.json`](Forge%20CPU%20Stream%E3%80%80研究版.json). It links `CheckpointLoaderSimple` → `ModelSamplingDiscrete(v_prediction, zsnr=true)` → `CFGGuider` and `BetaSamplingScheduler(36, 0.6, 0.6)`. Positive and negative **Forge CPU Stream Text Encode (Minimal)** outputs feed the guider. `KSamplerSelect(er_sde)` feeds **Forge CPU Stream Adapter**, whose **paired** `NOISE` and `SAMPLER` outputs feed the matching inputs of `SamplerCustomAdvanced`. `EmptyLatentImage(1024, 1024, batch 1)` feeds `latent_image`; the sampler's first `output` feeds `VAEDecode` → `PreviewImage`. The saved guider CFG is 4, text mode is `chunk_and_original`, and adapter `initial_scale` is `forge_off`. Replace the saved checkpoint and seed as needed; the screenshot is only a visual reference.

## Nodes and boundaries

| Node | Inputs / outputs | Behavior |
| --- | --- | --- |
| **Forge CPU Stream Adapter** | `SAMPLER`, unsigned 64 bit `seed`, `initial_scale` (`comfy_auto`, `forge_off`, `forge_on`) → `NOISE`, `SAMPLER` | Accepts only standard `KSamplerSelect(er_sde)`. CPU generator makes the initial draw and continues for each requested ER SDE noise draw; the two outputs share a one use handoff. Scale modes delegate to ComfyUI, force `max_denoise=False`, or force `True`, respectively. |
| **Forge CPU Stream Text Encode (Minimal)** | `CLIP`, multiline `text`, `mode` → `CONDITIONING`, `STRING` `token_report` | Reports L/G chunk token IDs and weights as JSON; supports the two modes below. |

The adapter is deliberately limited to unindexed, unmasked, non nested float32 zero `EmptyLatentImage` samples of shape `[1,4,H,W]`; it rejects denoise masks, a sampler with an existing noise override, a changed noise shape or dtype, a seed mismatch, an incomplete sigma schedule, and sampling that does not begin at maximum sigma. Use both paired outputs in one `SamplerCustomAdvanced` run. Unlike the general KSampler, it does not accept arbitrary samplers, partial denoise, or an image to image latent. These checks are runtime assertions, not a promise of compatibility across ComfyUI versions.

Text mode `chunk_only` passes the parsed per token weights to ComfyUI's encoder after Forge derived chunk construction. `chunk_and_original` sends unit weights into encoding, then applies Original emphasis separately to the L and G portions of each 77 token chunk. Both require ordinary SDXL CLIP L/G, equal L/G chunk counts, and 77 token tokenizers. Forge LoRA tags, textual inversion syntax, and `AND` composition are rejected; use ComfyUI's LoRA node. Neither mode establishes output equivalence with Forge Neo.

## Level 3 trace connection

The Minimal node optionally imports `fc_trace3` when `FC_TRACE3_DIR` is set. The supplied [`../research/level3/fc_trace3.py`](../research/level3/fc_trace3.py) must then be importable in the ComfyUI Python environment, alongside its `fc_trace` dependency; an environment variable alone does not install those modules. It records tokenization stages for L/G, and in `chunk_and_original` also an `adapter_result` context/pooled session. `FC_TRACE3_TARGET_JSON` can select text equal to the JSON string in that file. The separate Level 3 patches add additional application side observations; follow the research guide and check patch compatibility before use. With trace disabled, the Minimal node does not import `fc_trace3` along its trace paths.

## Evidence, limits, and contributions

The investigation focused specifically on **ER SDE**; other samplers were not systematically compared and equivalent behavior is not guaranteed. The main measured setup was **silvermoonmix v60 (V-Pred) + ER SDE**. The nodes are not checkpoint locked; other checkpoints have not been systematically tested. The CPU noise stream narrows a measured difference, but final image identity has not been established. The recorded conditioning comparisons include unresolved differences and do not prove a complete root cause. See the research JSONs for the actual observations and their coverage.

Independent verification, corrections, alternate instrumentation, and results on other environments are welcome. Please open an Issue if you know a related node or implementation with equivalent, more complete, or bidirectional ComfyUI ↔ Forge Neo compatibility. For licensing and parser attribution, see the general README and research guide; the supplied archive contains no LICENSE or NOTICE.
