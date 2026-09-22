# ComfyUI Forge CPU Stream

Experimental ComfyUI nodes and research artifacts for investigating and
reducing selected implementation differences between ComfyUI and Forge Neo.

This project does not claim pixel-perfect reproduction or complete
compatibility.

## Contents

### Standard nodes

See `Forge CPU Stream standard/`.

For normal use:

- Forge CPU Stream KSampler
- Forge CPU Stream Text Encode
- Example workflow

### Research nodes

See `Forge CPU Stream research_nodes/`.

Experimental nodes used during the investigation:

- Forge CPU Stream Adapter
- Forge CPU Stream Text Encode (Minimal)
- Research workflow

### Research tools and artifacts

See `research/`.

Contains the Level 1–2, Level 3, and conditioning investigation tools and
artifacts used during the investigation.

## Validation scope

This investigation focused specifically on ER SDE.

Other samplers have not been systematically investigated or validated,
and equivalent behavior is not guaranteed.

The primary tested environment used silvermoonmix v60 (V-Pred) with
ER SDE. The nodes are not intentionally restricted to this checkpoint,
but other checkpoints have not been systematically tested.

## Further investigation

Additional testing, corrections, improved instrumentation, results from
other environments, and further compatibility work are welcome.

If you know of an existing ComfyUI node or implementation that provides
equivalent, more complete, or bidirectional compatibility between
ComfyUI and Forge Neo, please open an Issue. References to related
implementations that may have been missed are also welcome.

## License and attribution

See `LICENSE` and `NOTICE`.
