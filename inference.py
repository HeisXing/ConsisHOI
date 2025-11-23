import os
import argparse
import random
from pathlib import Path
import cv2
import torch
import numpy as np
from accelerate.utils import set_seed
from PIL import Image

# Import model components
import sys
sys.path.append("./diffusers_consishoi")
from diffusers_consishoi.models.attention import GatedSelfAttentionDense
from diffusers_consishoi.models.attention_processor import Attention
from diffusers.models import AutoencoderKL
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


# from pipeline_consisthoi_sdxl_plot import StableDiffusionXLConsistHOIPipeline
from pipeline_consishoi_sdxl import StableDiffusionXLConsishoiPipeline
from consishoi_unet_2d_condition import ConsishoiUNet2DConditionModel

# Default negative prompt
default_neg_prompt = (
    "watermark, text, logo, signature, extra limbs, multiple heads, multiple fingers, "
    "bad anatomy, malformed limbs, deformed hands, horse extra hooves, blurry background, "
    "flat background, washed out, overexposed, hazy, foggy, desaturated, low contrast, "
    "soft focus, unnatural light, muddy textures, low detail, flat shading, dull colors, muted palette"
)

# # get cross-attention map
# collected_ca = []
# def save_attn2(module, inp, out):
#     # out: Tensor[B, C, H, W] or [B, heads, HW, HW] depending on implementation
#     collected_ca.append(out.detach().cpu())

# Load fuser weights utility
from collections import OrderedDict
def load_fuser_weights(unet, load_path, strict=True):
    fuser_state = torch.load(load_path, map_location=unet.device)
    full_state = unet.state_dict()
    for k, v in fuser_state.items():
        if k in full_state:
            full_state[k] = v
    unet.load_state_dict(full_state, strict=strict)

# Build pipeline
def build_pipeline(model_path, device):
    """Build the pipeline using a single consishoi model path.
    Expected layout: model_path contains subfolders like 'vae', 'unet', 'text_encoder', 'tokenizer', 'scheduler', and optional files 'fuser.bin' and 'id_encoder.bin'.
    """
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    unet_config = ConsishoiUNet2DConditionModel.load_config(model_path, subfolder="unet")
    unet = ConsishoiUNet2DConditionModel.from_config(unet_config).to(device)
    unet_state = ConsishoiUNet2DConditionModel.from_pretrained(model_path, subfolder="unet").state_dict()
    unet.load_state_dict(unet_state, strict=False)

    # Attempt to load fuser weights and id encoder weights from the same model directory
    hoi_weights = os.path.join(model_path, "fuser.bin")
    if os.path.exists(hoi_weights):
        try:
            load_fuser_weights(unet, hoi_weights, strict=False)
        except Exception:
            print(f"Failed to load fuser weights from {hoi_weights}")

    id_model_path = os.path.join(model_path, "id_encoder.bin")
    if not os.path.exists(id_model_path):
        id_model_path = None

    pipe = StableDiffusionXLConsishoiPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer2,
        id_model_path=id_model_path,
    ).to(device)
    return pipe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a single target image with fixed inputs')
    parser.add_argument('--consishoi_model_path', type=str, default='path_to_consishoi_model', help='Path to Consishoi model directory; used to load VAE/UNet/text encoders and optional fuser/id weights')
    parser.add_argument('--id_image', type=str, default='path_to_id_image', help='ID source image path')
    parser.add_argument('--id_box', nargs=4, type=float, default=[0.33276571330916366, 0.16421240517626065, 0.4182138889906129, 0.2679607318161535], help='ID image normalized box: x0 y0 x1 y1')
    parser.add_argument('--subject_box', nargs=4, type=float, default=[0.17651762063451373, 0.13966979027219995, 0.5, 0.7409638554216867])
    parser.add_argument('--object_box', nargs=4, type=float, default=[0.1862831264266794, 0.3795180722891566, 0.4890138059838136, 0.8369031682284693])
    parser.add_argument('--prompt', type=str, default='a person is riding a motorcycle')
    parser.add_argument('--out_image', type=str, default='./OUTPUTS_new/output.png')
    parser.add_argument('--seed', type=int, default=1634840976)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--scale', type=float, default=5.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Set seed
    seed = args.seed or random.randint(0, 2**32-1)
    set_seed(seed)
    print(f"Using seed: {seed}")

    # Load fixed ID image
    id_img = cv2.cvtColor(cv2.imread(args.id_image), cv2.COLOR_BGR2RGB)

    # Build pipeline (single model path)
    pipe = build_pipeline(
        args.consishoi_model_path,
        args.device,
    )
    print("Pipeline ready")

    # Prepare configs
    id_cfg = {'id_images': [id_img], 'id_boxes': [args.id_box], 'id_scale': 0.8}
    interaction_cfg = {
        'subject_phrases': [args.prompt.split()[0]],
        'object_phrases': [args.prompt.split()[-1]],
        'action_phrases': [args.prompt],
        'subject_boxes': [args.subject_box],
        'object_boxes': [args.object_box],
        'scheduled_sampling_beta': 1.0
    }

    # Generate single image
    out = pipe(
        prompt=args.prompt,
        negative_prompt=default_neg_prompt,
        height=1024, width=1024,
        interaction_cfg=interaction_cfg,
        id_cfg=id_cfg,
        output_type='pil', num_inference_steps=args.steps,
        guidance_scale=args.scale
    ).images[0]

    # Save
    out.save(args.out_image)
    print(f"Saved target image: {args.out_image}")

    # import pdb; pdb.set_trace()  # For debugging, remove in production
