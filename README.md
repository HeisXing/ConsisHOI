# Consishoi — Consistent Human-Object Interaction Generation (SDXL extension)

<p align="center">
	<img src="assets/introduction.png" height=100>
</p>

<div align="center">

A codebase for training and inference of a Consistent HOI (Human-Object Interaction) conditioned Stable Diffusion XL variant.

</div>

---

**Project status**: Work-in-progress. Key files:

- `train.py` — training entrypoint.
- `inference.py` — single-image inference script.
- `pipeline_consishoi_sdxl.py` — custom SDXL pipeline integrating HOI and ID conditioning.
- `consishoi_unet_2d_condition.py` — custom UNet module.

**Requirements**

- Python 3.10+ (Conda recommended)
- See root `requirements.txt` for full dependency list. Common packages: `torch`, `diffusers` (local fork), `transformers`, `accelerate`, `xformers` (optional), `facexlib`, `insightface`, `opencv-python`, `Pillow`.

Quick setup:

```bash
conda create -n consishoi python=3.10 -y
conda activate consishoi
pip install -r requirements.txt
```

**Usage — Inference**

Generate one image with the unified pipeline:

```bash
python inference.py \
	--consishoi_model_path /path/to/consishoi_model_dir \
	--id_image /path/to/id_image.png \
	--prompt "a person is riding a motorcycle" \
	--out_image ./OUTPUTS_new/output.png \
	--steps 50 \
    --scale 5.0 \
    --device cuda
```

Example `interaction_cfg` :

```py
interaction_cfg = {
	'subject_phrases': ['person'],
	'object_phrases': ['motorcycle'],
	'action_phrases': ['a person is riding a motorcycle'],
	'subject_boxes': [[x0, y0, x1, y1]],
	'object_boxes': [[x0, y0, x1, y1]],
	'scheduled_sampling_beta': 1.0,
}
```

**Usage — Training**

Train with a single model path for base weights:

```bash
python train.py \
	--consishoi_model_path /path/to/consishoi_model_dir \
	--output_dir ./OUTPUTS/ \
	--train_batch_size 4 \
	--device cuda
```

- If GPU memory is tight, consider enabling CPU offload in the pipeline or using lower `dtype` where supported.


**Citations & References**

This project is based on the InteractDiffusion and PuLID methods for Human-Object Interaction and ID-conditioned generation, and also draws on related HOI/consistent-generation works.

If you use this code in published work, please cite the original InteractDiffusion and PuLID works and any other papers that motivated your particular model choices.

**License & Contact**

This project is licensed under the MIT License - see the LICENSE file for details.

---
