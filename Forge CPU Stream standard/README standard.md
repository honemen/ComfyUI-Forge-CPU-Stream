# Forge CPU Stream — general nodes

Experimental ComfyUI nodes for aligning selected sampling and text encoding behavior with Forge Neo. They aim to reduce observed implementation differences; they do not establish pixel perfect reproduction or complete compatibility. For instrumentation and comparison, see [`../Forge CPU Stream研究版/README.md`](../Forge%20CPU%20Stream研究版/README.md) and [`../research/README.md`](../research/README.md).

## Install

Copy the **two folders** `forge_ksampler/` and `forge_text_encode/` into your ComfyUI `custom_nodes/` directory, keeping their Python files and `forge_ksampler/web/` files together. Restart ComfyUI. The modules use the installed ComfyUI and PyTorch; `forge_chunks.py` also uses the tokenizer supplied by the loaded CLIP. No research patches or Forge installation are needed for ordinary node use. The package contains no pinned ComfyUI revision or automated installer; an upstream sampler API change can raise an error.

## Nodes

| Node | Inputs | Output and role |
| --- | --- | --- |
| **Forge CPU Stream KSampler** | `model`, unsigned 64 bit `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `beta_alpha`, `beta_beta`, positive/negative `CONDITIONING`, `latent_image`, `denoise` | One `LATENT`. Builds a schedule (using the alpha/beta controls when `scheduler=beta`), CFG guider, and sampler. Draws initial noise with a CPU generator and, for the explicitly supported independent Gaussian noise samplers, continues the same generator for sampler noise. `denoise=0` returns a copy of the input latent. |
| **Forge CPU Stream Text Encode** | `clip`, multiline `text` | One `CONDITIONING`. Uses Forge derived prompt attention parsing and 77 token SDXL L/G chunk construction, then applies per encoder, per chunk Original emphasis to the returned conditioning. |

The sampler selector comes from the installed ComfyUI sampler list. The CPU stream override is implemented for `er_sde`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `dpm_2_ancestral`, `dpmpp_2s_ancestral`, `dpmpp_2s_ancestral_cfg_pp`, `ddpm`, `lcm`, `res_multistep`, `res_multistep_cfg_pp`, `res_multistep_ancestral`, `res_multistep_ancestral_cfg_pp`, `seeds_2`, `seeds_3`, `exp_heun_2_x0_sde`, `sa_solver`, and `sa_solver_pece`. For other selectable samplers the node still supplies CPU initial noise but does not replace the solver's noise sampler. Brownian tree solvers retain their native correlated noise. **This investigation focused specifically on ER SDE. Other samplers have not been systematically investigated or validated, and equivalent behavior is not guaranteed.** The `er_sde` path also forces `max_denoise=False` in this implementation.

The browser's **Random seed** button changes an unlinked seed manually; it does not randomize on every generation. The visible output label is presentation only. For text encoding, ordinary SDXL CLIP L/G with 77 token chunks is required. Forge `<lora:...>` tags, `embedding:` textual inversion, and `AND` prompt composition are rejected. Load LoRAs through the regular ComfyUI LoRA node.

## Example workflow

Open [`Forge CPU Stream　一般版.json`](Forge%20CPU%20Stream%E3%80%80一般版.json) in ComfyUI. Its graph is `CheckpointLoaderSimple` → `ModelSamplingDiscrete(v_prediction, zsnr=true)` → sampler, with the checkpoint's CLIP feeding two Text Encode nodes (positive and negative), and `EmptyLatentImage` feeding the sampler. The sampler's `LATENT` goes through `VAEDecode` to `PreviewImage`. The saved settings include ER SDE, beta alpha/beta 0.6, 36 steps, CFG 4, 1024 × 1024, batch 1, and `denoise=1`. Select a checkpoint file present on your system: the JSON refers to `silvermoonmix_v60VPred.safetensors`, which is **not included**. Use your own prompts, seed, and settings as appropriate. The supplied PNG is a screenshot, not an additional required node.

## Scope and limits

The main measured setup was **silvermoonmix v60 (V-Pred) + ER SDE**. The nodes are not intentionally restricted to that checkpoint, but other checkpoints have not been systematically tested. Tracking Forge Neo and ComfyUI noise/RNG behavior and implementing a continued CPU stream did not make final outputs fully identical. Conditioning and text encoding were investigated subsequently; the general text node incorporates the research node's `chunk_and_original` path, but this is not a claim of identical images. Behavior with other hardware, batch configurations, masks, and future ComfyUI versions is not established by this package. The sampler rejects an existing noise override and checks for the expected `noise_sampler` API on the override paths.

The project is also a handoff point for independent tests, corrections, other samplers, and improved implementations. If you know an existing ComfyUI node with equivalent or better features, more complete Forge Neo behavior, or bidirectional ComfyUI ↔ Forge Neo compatibility, please open an Issue with a reference. Related work is welcome.

## Attribution and licensing before publication

Both `forge_text_encode/forge_chunks.py` and the research version state that their parser derives from [Forge Classic/Neo parsing at commit `0c9273f`](https://github.com/Haoming02/sd-webui-forge-classic/blob/0c9273f69dbe491246fb11c3d4ac70123099d393/backend/text_processing/parsing.py). Retain that source credit. The supplied archive has **no LICENSE or NOTICE**. Check the applicable upstream licenses and the extent of derived code, including the research patches, before assigning a license and distributing the package; a compatible LICENSE plus a NOTICE naming source files and commits may be needed. No license for this package is asserted here.
