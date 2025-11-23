import argparse
import logging
import math
import os
from pydoc import text
import random
import shutil
from attr import dataclass
from click import Option
from h11 import Data
from matplotlib.pyplot import box
from regex import P
from sympy import Idx
import yaml
import shutil
import json
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from packaging import version
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, Sampler, BatchSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection
import torch.fft as fft
from torch.fft import fftn, fftshift, ifftn, ifftshift
from collections import defaultdict

# compatibility: use updated folder name
import sys
sys.path.append("./diffusers_consishoi")
# os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:3"

import diffusers_consishoi
from diffusers_consishoi import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers_consishoi.optimization import get_scheduler
from diffusers_consishoi.utils import check_min_version, is_wandb_available
from diffusers_consishoi.utils.import_utils import is_xformers_available
from diffusers_consishoi.models.attention import GatedSelfAttentionDense

import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from dataclasses import fields
from functools import partial
from datetime import datetime
from collections import OrderedDict

from consishoi_unet_2d_condition import ConsishoiUNet2DConditionModel
from HOI.ConsisHOI_v4.pipeline_consishoi_sdxl import IDEncoder_pipeline

sys.path.append("./id_encoder")
from id_encoder.encoder.utils import img2tensor, tensor2img
from id_encoder.eva_clip import create_model_and_transforms
from id_encoder.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from id_encoder.encoder.encoders import IDEncoder
from id_encoder.encoder.attention_processor import AttnProcessor2_0 as AttnProcessor
from id_encoder.encoder.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
from id_encoder.encoder.utils import resize_numpy_image_long

from torchvision.transforms.functional import normalize, resize
import cv2
from torchvision.transforms import InterpolationMode
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from facexlib.parsing import init_parsing_model
import insightface
from insightface.app import FaceAnalysis
from typing import Any, Dict, List
import gc
import pdb
from pathlib import Path

torch.autograd.set_detect_anomaly(True)

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__)

# Gloable Variance
GAMMA = 1.8
BRIGHTNESS = 1.1

def reduce_exposure_gamma(image, gamma=1.5):
    """Apply gamma correction to reduce image exposure.
    Args:
        image: PIL Image
        gamma: gamma value (>1 reduces exposure)
    Returns:
        PIL Image
    """
    img_array = np.array(image)
    
        # Normalize pixels to [0, 1]
    normalized = img_array / 255.0
        # Apply gamma correction: output = input ** gamma
    corrected = np.power(normalized, gamma)
        # Convert back to [0, 255] and cast to uint8
    corrected_array = (corrected * 255).astype(np.uint8)
    
        # Convert back to PIL Image
    return Image.fromarray(corrected_array)

def reduce_brightness(image, factor=0.7):
    """Linearly reduce image brightness.
    Args:
        image: PIL Image
        factor: brightness factor (0~1)
    Returns:
        PIL Image
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def resize_if_needed(image: np.ndarray, max_area: int = 512*512):
    """If image area is larger than max_area, scale it down proportionally."""
    h, w = image.shape[:2]
    area = h * w

    if area > max_area:
        scale_factor = (max_area / area) ** 0.5
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return image

@dataclass
class UnifiedSample:
    caption: str
    id_image: Optional[np.ndarray] = None
    id_boxes: Optional[List[List[int]]] = None
    subject_phrases: Optional[List[str]] = None
    object_phrases: Optional[List[str]] = None
    action_phrases: Optional[List[str]] = None
    subject_boxes: Optional[List[List[float]]] = None
    object_boxes: Optional[List[List[float]]] = None
    task_type: str = "UnKnown"
    target_image: Optional[torch.Tensor] = None

# define uncondationed simple
predefined_uncond_sample = UnifiedSample(
    caption="",
    id_image=np.zeros((512,512,3), dtype=np.uint8),
    id_boxes=[],
    subject_phrases=[],
    object_phrases=[],
    action_phrases=[],
    subject_boxes=[],
    object_boxes=[],
    task_type="Consishoi",
    target_image=torch.zeros((3, 512, 512), dtype=torch.float32),
)

class IDDataset(Dataset):
    def __init__(self, root_dir: str, transform: Any = None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.anno_path = os.path.join(root_dir, "annotations", "anno.json")

        # Read annotation JSON; new format: each annotation is a dict with 'boxes_id' and 'caption'
        with open(self.anno_path, 'r', encoding='utf-8') as f:
            self.anno_data = json.load(f)

        if transform is None:
            transform = transforms.Compose([
                transforms.ToPILImage(),       # NumPy -> PIL
                transforms.Lambda(lambda img: reduce_exposure_gamma(img, gamma=GAMMA)),
                transforms.Lambda(lambda img: reduce_brightness(img, factor=BRIGHTNESS)),
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Get filenames and annotation list
        self.filenames = self.anno_data.get("filenames", [])
        self.annotations = self.anno_data.get("annotation", [])

        assert len(self.filenames) == len(self.annotations), \
            f"Number of images ({len(self.filenames)}) does not match number of annotations ({len(self.annotations)})"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> UnifiedSample:
        # Load image path
        image_name = self.filenames[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2] # image size
        image = resize_if_needed(image)

        # Extract boxes_id and caption from annotation
        ann: Dict[str, Any] = self.annotations[index]
        id_boxes: List[List[float]] = ann.get("boxes_id", [])
        caption: str = ann.get("caption", "")

        # Apply transform
        target_image = self.transform(image)

        # Convert absolute pixel boxes to relative boxes
        id_boxes = convert_to_relative(id_boxes, width, height)

        # Return unified data structure
        return UnifiedSample(
            caption=caption,
            id_image=image,
            id_boxes= id_boxes,
            subject_phrases=[],
            object_phrases=[],
            action_phrases=[],
            subject_boxes=[],
            object_boxes=[],
            target_image=target_image,
            task_type="ID",
        )

class HOIDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.anno_path = os.path.join(root_dir, "annotations", "anno.json")
        
        if transform is None:
            transform = transforms.Compose([
                transforms.ToPILImage(),       # NumPy -> PIL
                transforms.Lambda(lambda img: reduce_exposure_gamma(img, gamma=GAMMA)),
                transforms.Lambda(lambda img: reduce_brightness(img, factor=BRIGHTNESS)),
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Read annotation JSON
        with open(self.anno_path, "r") as f:
            self.anno_data = json.load(f)

        self.filenames = self.anno_data["filenames"]
        self.annotations = self.anno_data["annotation"]
        self.objects = self.anno_data["objects"]
        self.verbs = self.anno_data["verbs"]

        assert len(self.filenames) == len(self.annotations), "Number of images does not match annotations"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> UnifiedSample:
        image_name = self.filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        anno = self.annotations[idx]

        # Load image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = resize_if_needed(image)

        # Extract HOI information
        subject_boxes = anno["boxes_h"]  # subject boxes
        object_boxes = anno["boxes_o"]
        object_labels = [self.objects[i] for i in anno["object"]]
        verb_labels = [self.verbs[i] for i in anno["verb"]]

        subject_boxes = convert_to_relative(subject_boxes, width, height)
        object_boxes = convert_to_relative(object_boxes, width, height)

        if self.transform:
            image = self.transform(image)

        # Normalize subject phrase to "person"
        subject_phrases = ["person"] * len(verb_labels)
        object_phrases = object_labels
        action_phrases = verb_labels

        # Extract caption
        caption = anno.get("caption", "")  # caption text

        return UnifiedSample(
            caption=caption,
            id_image=np.zeros((512,512,3), dtype=np.uint8),
            id_boxes=[],
            subject_phrases=subject_phrases,
            object_phrases=object_phrases,
            action_phrases=action_phrases,
            subject_boxes=subject_boxes,
            object_boxes=object_boxes,
            target_image=image,
            task_type="HOI"
        )

class ConsishoiDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.id_img_dir = os.path.join(root_dir, "id_images")
        self.anno_path = os.path.join(root_dir, "annotations", "anno.json")
        
        if transform is None:
            transform = transforms.Compose([
                transforms.ToPILImage(),       # NumPy -> PIL
                transforms.Lambda(lambda img: reduce_exposure_gamma(img, gamma=GAMMA)),
                transforms.Lambda(lambda img: reduce_brightness(img, factor=BRIGHTNESS)),
                transforms.ToTensor(),
            ])
        self.transform = transform

        # Read JSON annotations
        with open(self.anno_path, "r") as f:
            self.anno_data = json.load(f)

        self.filenames = self.anno_data["filenames"]
        self.annotations = self.anno_data["annotation"]
        self.objects = self.anno_data["objects"]
        self.verbs = self.anno_data["verbs"]

        assert len(self.filenames) == len(self.annotations), "filenames and annotations count mismatch"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> UnifiedSample:
        anno = self.annotations[idx]
        target_img_name = self.filenames[idx]
        target_img_path = os.path.join(self.image_dir, target_img_name)

        # Load Consishoi HOI image
        target_image = cv2.cvtColor(cv2.imread(target_img_path), cv2.COLOR_BGR2RGB)
        height, width = target_image.shape[:2]
        target_image = resize_if_needed(target_image)

        # Load ID image
        id_img_path = os.path.join(self.id_img_dir, anno["id_image_name"])
        id_image = cv2.cvtColor(cv2.imread(id_img_path), cv2.COLOR_BGR2RGB)
        id_image = resize_if_needed(id_image)

        # Load boxes and labels
        subject_boxes = anno["boxes_h"]
        object_boxes = anno["boxes_o"]
        id_boxes = anno["boxes_id"]
        object_labels = [self.objects[i] for i in anno["object"]]
        verb_labels = [self.verbs[i] for i in anno["verb"]]
        caption = anno.get("caption", "")

        subject_boxes = convert_to_relative(subject_boxes, width, height)
        object_boxes = convert_to_relative(object_boxes, width, height)
        id_boxes = convert_to_relative(id_boxes, width, height)

        if self.transform:
            target_image = self.transform(target_image)
            # id_image = self.transform(id_image)

        # Normalize subject phrase to "person"
        subject_phrases = ["person"] * len(verb_labels)
        object_phrases = object_labels
        action_phrases = verb_labels

        return UnifiedSample(
            caption=caption,
            id_image=id_image,
            id_boxes=id_boxes,
            subject_phrases=subject_phrases,
            object_phrases=object_phrases,
            action_phrases=action_phrases,
            subject_boxes=subject_boxes,
            object_boxes=object_boxes,
            target_image=target_image,
            task_type="Consishoi"
        )

class HOIFromConsistDataset(Dataset):
    """Treat a ConsishoiDataset as pure HOI data, discarding id_image and id_boxes."""
    def __init__(self, consist_dataset: ConsishoiDataset):
        self.consist_ds = consist_dataset

    def __len__(self):
        return len(self.consist_ds)

    def __getitem__(self, idx: int) -> UnifiedSample:
        base: UnifiedSample = self.consist_ds[idx]
        # Reuse subject_phrases, object_phrases, action_phrases, boxes, caption, target_image
        return UnifiedSample(
            caption=base.caption,
            id_image=np.zeros((512, 512, 3), dtype=np.uint8),
            id_boxes=[],
            subject_phrases=base.subject_phrases,
            object_phrases=base.object_phrases,
            action_phrases=base.action_phrases,
            subject_boxes=base.subject_boxes,
            object_boxes=base.object_boxes,
            target_image=base.target_image,
            task_type="HOI",
        )

class MixedDataset(Dataset):
    def __init__(self, id_dataset, hoi_dataset, consishoi_dataset, ratios=[1, 2, 3], unconditional_prob=0.1, unconditional_sample = None):
        self.id_dataset = id_dataset
        # self.hoi_dataset = hoi_dataset
        self.hoi_dataset = HOIFromConsistDataset(consishoi_dataset)
        self.consishoi_dataset = consishoi_dataset

        self.datasets = [self.id_dataset, self.hoi_dataset, self.consishoi_dataset]
        self.ratios = ratios
        self.total_ratio = sum(ratios)

        # self.max_len = max(len(id_dataset), len(hoi_dataset), len(consishoi_dataset))
        self.max_len = len(id_dataset) + len(consishoi_dataset)

        # unconditional sample
        assert 0.0 <= unconditional_prob <= 1.0, "unconditional_prob must be between 0 and 1"
        self.unconditional_prob = unconditional_prob
        self.unconditional_sample = unconditional_sample

        if self.unconditional_prob >= 0:
            assert self.unconditional_sample is not None, (
                "unconditional_sample mask be given"
            )

        # Record starting offsets for the three index segments for quick sampler lookup
        self.len_id   = len(self.id_dataset)
        self.len_hoi  = len(self.hoi_dataset)
        self.len_cons = len(self.consishoi_dataset)

    def __len__(self):
        # return self.max_len * round(self.total_ratio)
        return self.max_len

    def set_ratios(self, ratios: List[float]):
        """
        Can be called during training to update dataset sampling ratios per epoch or step.
        Example: new_ratios = [r_id, r_hoi, r_consist]
        """
        assert len(ratios) == 3, "new_ratios must be a list of three floats [r_id, r_hoi, r_consist]"
        assert all(r >= 0 for r in ratios), "ratios must be non-negative"
        self.ratios = ratios[:]
        self.total_ratio = sum(self.ratios)

    def __getitem__(self, index) -> UnifiedSample:
        # # target unconditional sample
        # if self.unconditional_prob > 0:
        #     if random.random() < self.unconditional_prob:
        #         # return the prebuilt unified 'unconditional' sample
        #         return self.unconditional_sample

        # r = random.uniform(0, self.total_ratio)
        # cum = 0.0
        # if r < (cum := cum + self.ratios[0]):
        #     dataset = self.id_dataset
        # elif r < (cum := cum + self.ratios[1]):
        #     dataset = self.hoi_dataset
        #     # dataset = self.consishoi_dataset
        # else:
        #     dataset = self.consishoi_dataset

        # # Randomly sample an item
        # sample_idx = random.randint(0, len(dataset) - 1)
        # sample = dataset[sample_idx]
        # return sample
        if isinstance(index, tuple):
            ds_id, local_idx = index
            if ds_id == -1:
                return self.unconditional_sample
            elif ds_id == 0:
                return self.id_dataset[local_idx]
            elif ds_id == 1:
                return self.hoi_dataset[local_idx]
            elif ds_id == 2:
                return self.consishoi_dataset[local_idx]
            else:
                raise ValueError(f"Unknown dataset id {ds_id}")
        else:
            raise RuntimeError("Mixed dataset index must include ds_id; use a sampler that yields (ds_id, local_idx)")

class ResolutionMixedBatchSampler(BatchSampler):
    """
    Yield batches where all indices in a batch refer to images of the same resolution.
    Index format: (ds_id, local_idx)
    ds_id mapping: 0 -> id_dataset, 1 -> hoi_dataset, 2 -> consist_dataset
    Supports unconditional batches: yield [(-1, -1), ...]
    """
    def __init__(self, mixed_dataset, batch_size, drop_last=False):
        self.ds = mixed_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Group lookup
        self.groups = []
        for sub_ds in [self.ds.id_dataset, self.ds.consishoi_dataset]:
            names = getattr(sub_ds, 'filenames')
            img_dir = getattr(sub_ds, 'image_dir')
            g = {}
            for i, name in enumerate(names):
                img_path = os.path.join(img_dir, name)
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                h, w = img.shape[:2]
                g.setdefault((h,w), []).append(i)
            self.groups.append(g)

        self.groups = [self.groups[0], self.groups[1], self.groups[1]]
        cum=0; self.cum_ratios=[]
        for r in self.ds.ratios:
            cum+=r; self.cum_ratios.append(cum)
        self.total_ratio=cum
        import math
        self.total_batches = sum(math.ceil(len(idxs)/batch_size) for grp in [self.groups[0], self.groups[1]] for idxs in grp.values())

    def __iter__(self):
        for _ in range(self.total_batches):
            if self.ds.unconditional_prob>0 and random.random()<self.ds.unconditional_prob:
                yield [(-1,-1)]*self.batch_size
                continue
            x=random.random()*self.total_ratio
            if x<self.cum_ratios[0]: ds_id=0
            elif x<self.cum_ratios[1]: ds_id=1
            else: ds_id=2
            grp=self.groups[ds_id]
            res=random.choice(list(grp.keys()))
            idxs=grp[res]
            if len(idxs)>=self.batch_size:
                choose = random.sample(idxs, self.batch_size)
            else:
                choose = idxs.copy()
            batch = [(ds_id, i) for i in choose]
            yield batch

    def __len__(self):
        return self.total_batches

# Convert absolute coordinates to relative coordinates
def convert_to_relative(boxes, width, height):
    relative_boxes = []
    for box in boxes:
        if len(box) == 4:  # assume box format is [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            rel_x1 = x1 / width
            rel_y1 = y1 / height
            rel_x2 = x2 / width
            rel_y2 = y2 / height
            relative_boxes.append([rel_x1, rel_y1, rel_x2, rel_y2])
        else:
            relative_boxes.append(box)  # keep anomalous format
    return relative_boxes

def unified_collate_fn(batch, device=None):
    # batch: List[UnifiedSample]
    collated_batch = {}

    for field in fields(batch[0]):  # Iterate over each field of UnifiedSample
        key = field.name
        values = [getattr(sample, key) for sample in batch]

        # Find the first non-None element to infer how to handle this field
        first_non_none = next((v for v in values if v is not None), None)

        if first_non_none is None:
            # If the entire batch field is None, keep it None
            collated_batch[key] = None

        elif isinstance(first_non_none, torch.Tensor):
            # It's a Tensor type; pad None entries with zero tensors
            for i, v in enumerate(values):
                if v is None:
                    values[i] = torch.zeros_like(first_non_none)
            
            stacked = torch.stack(values)
            if device is not None:
                stacked = stacked.to(device)

            collated_batch[key] = stacked
        elif isinstance(first_non_none, list):
            # If it's a list, e.g., boxes or phrases
            for i, v in enumerate(values):
                if v is None:
                    values[i] = []  # Pad with empty list
            collated_batch[key] = values
        elif isinstance(first_non_none, (int, float, str)):
            # Simple types: collect into a list
            collated_batch[key] = values
        else:
            # Other types: keep as a list
            collated_batch[key] = values

    # 2) If task_type present, sort: HOI -> Consishoi -> ID
    #    Get original indices and sort by priority
    if collated_batch.get("task_type") is not None:
        # priority mapping: HOI < Consishoi < ID
        priority_map = {"HOI": 0, "Consishoi": 1, "ID": 2}
        original_types: List[str] = collated_batch["task_type"]  # e.g. ["ID","HOI","ConsistHOI", ...]
        # Generate sorted index list
        indices_sorted = sorted(
            range(len(original_types)),
            key=lambda i: priority_map.get(original_types[i], 3)
        )

        # 3) Reorder each field in the new dict `reordered_batch` accordingly
        reordered_batch: Dict[str, Any] = {}
        for key, value in collated_batch.items():
            if isinstance(value, torch.Tensor):
                # Tensor first dim is batch_size; reorder using indices_sorted
                reordered_batch[key] = value[indices_sorted]
            elif isinstance(value, list):
                # List: reorder each position
                reordered_batch[key] = [value[i] for i in indices_sorted]
            else:
                # None or other types: no reordering necessary
                reordered_batch[key] = value

        return reordered_batch

    return collated_batch

def spectral_decompose(features: torch.Tensor, low_freq_radius: float = 0.1):
    """
    Args:
        features: Tensor of shape [B, C, H, W]
        low_freq_radius: float, radius for low-frequency mask in normalized frequency space (0 < r < 0.5)

    Returns:
        low_feat, high_feat: Tensors of shape [B, C, H, W] corresponding to low- and high-frequency components.
    """
    B, C, H, W = features.shape
    # 2D FFT and shift
    z_f_shift = torch.fft.fftshift(torch.fft.fft2(features, norm='ortho'), dim=(-2, -1))

    # Build frequency grid
    fy = torch.linspace(-0.5, 0.5, H, device=features.device)
    fx = torch.linspace(-0.5, 0.5, W, device=features.device)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')
    radius = torch.sqrt(grid_x**2 + grid_y**2)[None, None, :, :]

    mask_low  = (radius <= low_freq_radius).float()
    mask_high = 1.0 - mask_low

    # Inverse transform
    low  = torch.fft.ifft2(torch.fft.ifftshift(z_f_shift * mask_low,  (-2,-1)), norm='ortho').real
    high = torch.fft.ifft2(torch.fft.ifftshift(z_f_shift * mask_high, (-2,-1)), norm='ortho').real

    return low, high  # [B,C,H,W]

def make_box_mask_list(boxes_list, H: int, W: int, device):
    """
    boxes_list: List of length B, each element is List of [x0,y0,x1,y1]
    returns: Tensor [B, H, W], where pixels inside any box are 1.
    Coordinates normalized in [0,1].
    """
    B = len(boxes_list)
    mask = torch.zeros(B, H, W, device=device)
    for b, boxes in enumerate(boxes_list):
        for box in boxes:
            x0, y0, x1, y1 = box
            x0i = max(0, min(W, math.floor(x0 * W)))
            x1i = max(0, min(W, math.ceil(x1 * W)))
            y0i = max(0, min(H, math.floor(y0 * H)))
            y1i = max(0, min(H, math.ceil(y1 * H)))
            if x1i > x0i and y1i > y0i:
                mask[b, y0i:y1i, x0i:x1i] = 1.0
    return mask  # [B,H,W]

def pad_boxes_list_to_tensor(boxes_list: list, device: torch.device) -> torch.Tensor:
    """
    Pad irregular List[List[List[4]]] into a [B, N_max, 4] tensor; fill empty slots with 0.
    """
    B = len(boxes_list)
    N_max = max(len(boxes) for boxes in boxes_list)  # maximum number of boxes
    padded = torch.zeros((B, N_max, 4), dtype=torch.float32, device=device)

    for b, boxes in enumerate(boxes_list):
        if len(boxes) > 0:
            padded[b, :len(boxes)] = torch.tensor(boxes, dtype=torch.float32, device=device)

    return padded  # [B, N_max, 4]

def get_between_box(bbox1, bbox2):
    """ Between Set Operation
    Operation of Box A between Box B from Prof. Jiang idea
    """
    all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]],dim=-1)
    all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]],dim=-1)
    all_x, _ = all_x.sort()
    all_y, _ = all_y.sort()
    return torch.stack([all_x[:,:,1], all_y[:,:,1], all_x[:,:,2], all_y[:,:,2]],2)

def compute_frequency_losses(
    model_pred, target, timesteps,
    id_boxes_list, subject_boxes_list, object_boxes_list,
    noise_scheduler,
    low_freq_radius=0.02, high_freq_radius=0.7
):
    """
    Compute frequency-based losses.
    - HOI low-frequency loss (when subject/object/action boxes exist)
    - ID high-frequency loss (when id_boxes exist)

    Args:
        model_pred: [B, C, H, W], predicted noise
        target:     [B, C, H, W], ground-truth noise
        timesteps:  [B], timesteps
        *_boxes_list: List[List[List[4]]]
        noise_scheduler: diffusion noise scheduler
        low_freq_radius: low-frequency radius for decomposition
        high_freq_radius: high-frequency radius for decomposition
    Returns:
        loss_hoi_lf: tensor or 0
        loss_id_hf: tensor or 0
    """
    # import pdb; pdb.set_trace()
    B, C, H, W = model_pred.shape
    T = noise_scheduler.config.num_train_timesteps

    # 1) Compute time weight w_t = α_t / (1 - α_t)
    alpha_bar = noise_scheduler.alphas_cumprod.to(model_pred.device)[timesteps]  # [B]
    decay = (1.0-timesteps.to(torch.float32)/T).view(B)
    w_base = alpha_bar / (1 - alpha_bar)
    w_t = (decay * w_base).view(B, 1, 1, 1)

    # 2) Frequency decomposition
    _, high_pred = spectral_decompose(model_pred, low_freq_radius)
    _, high_tgt   = spectral_decompose(target, low_freq_radius)
    low_pred, _ = spectral_decompose(model_pred, high_freq_radius)
    low_tgt, _ = spectral_decompose(target, high_freq_radius)

    # HOI low-frequency loss, applied when subject/object/action boxes all exist
    loss_hoi_lf = torch.tensor(0.0, device=model_pred.device)
    loss_id_hf  = torch.tensor(0.0, device=model_pred.device)
    
    subject_boxes = pad_boxes_list_to_tensor(subject_boxes_list, device=model_pred.device)  # [B, N, 4]
    object_boxes  = pad_boxes_list_to_tensor(object_boxes_list,  device=model_pred.device)  # [B, N, 4]
    action_boxes = get_between_box(subject_boxes, object_boxes)
    
    if any(len(lst) > 0 for lst in (subject_boxes, object_boxes, action_boxes)):
        # hoi_boxes = [
        #     subject_boxes[i] + object_boxes[i] + action_boxes[i]
        #     for i in range(B)
        # ]
        hoi_boxes = torch.cat([subject_boxes, object_boxes, action_boxes], dim=1)
        hoi_mask = make_box_mask_list(hoi_boxes, H, W, model_pred.device).unsqueeze(1)  # [B,1,H,W]
        loss_hoi_lf = F.mse_loss(
            w_t * (low_pred * hoi_mask),
            w_t * (low_tgt  * hoi_mask),
            reduction="mean"
        )

    # ID high-frequency loss, applied when id_boxes is non-empty
    if any(len(lst) > 0 for lst in id_boxes_list):
        id_mask = make_box_mask_list(id_boxes_list, H, W, model_pred.device).unsqueeze(1)
        loss_id_hf = F.mse_loss(
            w_t * (high_pred * id_mask),
            w_t * (high_tgt  * id_mask),
            reduction="mean"
        )

    return loss_hoi_lf, loss_id_hf


# time ids
def compute_time_ids(original_size, crops_coords_top_left, target_size):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    return add_time_ids # tensor(1, 6)

# encode prompt
def encode_prompt(
    prompt: str,
    device: Optional[torch.device],
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = [prompt] if isinstance(prompt, str) else prompt

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    prompt_embeds_list = []

    if prompt is None:
        print("Warning: prompt is None, using empty prompt.")
        prompt = ""
    prompts = [prompt, prompt]

    for prompt_now, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt_now,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)

        encoder_outputs = text_encoder(input_ids, output_hidden_states=True)

        pooled_prompt_embeds = encoder_outputs[0].to(dtype=text_encoder.dtype, device=device)  # take last_hidden_state (batch, seq_len, hidden_dim)
        prompt_embeds = encoder_outputs.hidden_states[-2]  # second-to-last hidden state

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        prompt_embeds_list.append(prompt_embeds)

    # Finally, concatenate the two encodings
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)

    return prompt_embeds, pooled_prompt_embeds

def encode_prompt_gligen(
    prompt,
    device: Optional[torch.device],
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt == []:
        prompt = ""

    tokenizers = [tokenizer_one, tokenizer_two] if tokenizer_one is not None else [tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two] if text_encoder_one is not None else [text_encoder_two]

    embeds = []
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)

        outputs = encoder(input_ids, output_hidden_states=False)

        if isinstance(encoder, CLIPTextModel):
            embeds.append(outputs.pooler_output)
        elif isinstance(encoder, CLIPTextModelWithProjection):
            embeds.append(outputs.text_embeds)

    return torch.cat(embeds, dim=-1)

# prepare feature encoder inputs
def prepare_gligen_inputs(
    batch,
    device: torch.device,
    dtype: torch.dtype,
    cross_attention_dim: int,
    batch_size: int,
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    cross_attention_kwargs: Optional[dict] = None,
):
    max_objs = 30
    batch_size = len(batch["caption"])

    batch_gligen = {
        "interaction_cfg": {
            "subject_boxes": None,
            "object_boxes": None,
            "subject_positive_embeddings": None,
            "object_positive_embeddings": None,
            "action_positive_embeddings": None,
        },
        "id_cfg": {
            "id_image": None,
            "id_boxes": None,
            "id_embedding": None,
            "id_masks": None,
        },
        "masks": None,
        "task_type": [],
    }

    for index in range(batch_size):
        consishoi_inputs = {
            "task_type": batch["task_type"][index],
            "subject_phrases": batch["subject_phrases"][index],
            "object_phrases": batch["object_phrases"][index],
            "action_phrases": batch["action_phrases"][index],
            "subject_boxes": batch["subject_boxes"][index],
            "object_boxes": batch["object_boxes"][index],
            "id_image": batch["id_image"][index],
            "id_boxes": batch["id_boxes"][index],
        }

        # prepare HOI Inputs
        interactdiffusion_subject_phrases = consishoi_inputs["subject_phrases"]
        interactdiffusion_subject_boxes = consishoi_inputs["subject_boxes"]
        interactdiffusion_object_phrases = consishoi_inputs["object_phrases"]
        interactdiffusion_object_boxes = consishoi_inputs["object_boxes"]
        interactdiffusion_action_phrases = consishoi_inputs["action_phrases"]
        
        if len(interactdiffusion_action_phrases) > max_objs:
            interactdiffusion_subject_phrases = interactdiffusion_subject_phrases[:max_objs]
            interactdiffusion_subject_boxes = interactdiffusion_subject_boxes[:max_objs]
            interactdiffusion_object_phrases = interactdiffusion_object_phrases[:max_objs]
            interactdiffusion_object_boxes = interactdiffusion_object_boxes[:max_objs]
            interactdiffusion_action_phrases = interactdiffusion_action_phrases[:max_objs]
        # prepare batched input to the InteractDiffusionInteractionProjection (boxes, phrases, mask)
        # obtain its text features for phrases
        # Encode conditional text embeddings (subject, object, action) interactdiffusion_embeds.shape torch.Size([3*n_objs], 2048)
        (
            interactdiffusion_embeds
        ) = encode_prompt_gligen(
            prompt=interactdiffusion_subject_phrases+interactdiffusion_object_phrases+interactdiffusion_action_phrases,
            device=device,
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
        )
        
        n_objs = min(len(interactdiffusion_subject_boxes), max_objs)
        # For each entity, described in phrases, is denoted with a bounding box,
        # we represent the location information as (xmin,ymin,xmax,ymax)
        # boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
        # boxes[:n_objs] = torch.tensor(gligen_boxes)
        # Initialize subject and object boxes (4 values per box: x0, y0, x1, y1)
        subject_boxes = torch.zeros(max_objs, 4, device=device, dtype=dtype)
        object_boxes = torch.zeros(max_objs, 4, device=device, dtype=dtype)
        if n_objs > 0:
            subject_boxes[:n_objs] = torch.tensor(interactdiffusion_subject_boxes[:n_objs])
            object_boxes[:n_objs] = torch.tensor(interactdiffusion_object_boxes[:n_objs])
        
        # Initialize conditional text embeddings
        text_embeddings = torch.zeros(
            max_objs*3, cross_attention_dim, device=device, dtype=dtype
        )
        text_embeddings[:n_objs*3] = interactdiffusion_embeds
        
        subject_text_embeddings = torch.zeros(max_objs, cross_attention_dim, device=device, dtype=dtype)
        subject_text_embeddings[:n_objs] = text_embeddings[:n_objs*1]
        object_text_embeddings = torch.zeros(max_objs, cross_attention_dim, device=device, dtype=dtype)
        object_text_embeddings[:n_objs] = text_embeddings[n_objs*1:n_objs*2]
        action_text_embeddings = torch.zeros(max_objs, cross_attention_dim, device=device, dtype=dtype)
        action_text_embeddings[:n_objs] = text_embeddings[n_objs*2:n_objs*3]
        # Generate a mask for each object that is entity described by phrases
        # Generate mask for number of subjects (mark valid objects)
        hoi_masks = torch.zeros(max_objs, device=device, dtype=dtype)
        hoi_masks[:n_objs] = 1

        repeat_batch = 1
        # boxes = boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        # text_embeddings = text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        # Expand boxes and text embeddings for batch dimension (unsqueeze adds a new dim at position 0)
        subject_boxes = subject_boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        object_boxes = object_boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        subject_text_embeddings = subject_text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        object_text_embeddings = object_text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        action_text_embeddings = action_text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        hoi_masks = hoi_masks.unsqueeze(0).expand(repeat_batch, -1).clone()
        
        '''
        subject_boxes.shape torch.Size([1, 30, 4])
        subject_text_embeddings.shape torch.Size([1, 30, 2048])
        hoi_masks.shape torch.Size([1, 30])
        '''
        # Save HOI features into batch_gligen
        if batch_gligen["interaction_cfg"]["subject_boxes"] is None:
            batch_gligen["interaction_cfg"]["subject_boxes"] = subject_boxes
            batch_gligen["interaction_cfg"]["object_boxes"] = object_boxes
            batch_gligen["interaction_cfg"]["subject_positive_embeddings"] = subject_text_embeddings
            batch_gligen["interaction_cfg"]["object_positive_embeddings"] = object_text_embeddings
            batch_gligen["interaction_cfg"]["action_positive_embeddings"] = action_text_embeddings
        else:
            batch_gligen["interaction_cfg"]["subject_boxes"] = torch.cat((batch_gligen["interaction_cfg"]["subject_boxes"], subject_boxes), dim=0)
            batch_gligen["interaction_cfg"]["object_boxes"] = torch.cat((batch_gligen["interaction_cfg"]["object_boxes"], object_boxes), dim=0)
            batch_gligen["interaction_cfg"]["subject_positive_embeddings"] = torch.cat((batch_gligen["interaction_cfg"]["subject_positive_embeddings"], subject_text_embeddings), dim=0)
            batch_gligen["interaction_cfg"]["object_positive_embeddings"] = torch.cat((batch_gligen["interaction_cfg"]["object_positive_embeddings"], object_text_embeddings), dim=0)
            batch_gligen["interaction_cfg"]["action_positive_embeddings"] = torch.cat((batch_gligen["interaction_cfg"]["action_positive_embeddings"], action_text_embeddings), dim=0)

        if batch_gligen["masks"] is None:
            batch_gligen["masks"] = hoi_masks
        else:
            batch_gligen["masks"] = torch.cat((batch_gligen["masks"], hoi_masks), dim=0)

        # prepare ID Inputs
        id_image = consishoi_inputs["id_image"]
        id_boxes = consishoi_inputs["id_boxes"]

        n_id_objs = min(len(id_boxes), max_objs)
        id_masks = torch.zeros(max_objs, device=device, dtype=dtype)
        id_masks[n_objs: n_objs+n_id_objs] = 1

        subject_id_boxes = torch.zeros(max_objs, 4, device=device, dtype=dtype)
        if n_id_objs > 0:
            subject_id_boxes[:n_id_objs] = torch.tensor(id_boxes[:n_id_objs])

        subject_id_boxes = subject_id_boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        id_masks = id_masks.unsqueeze(0).expand(repeat_batch, -1).clone()

        # Save ID features into batch_gligen
        if batch_gligen["id_cfg"]["id_boxes"] is None:
            # batch_gligen["id_cfg"]["id_image"] = id_image
            batch_gligen["id_cfg"]["id_boxes"] = subject_id_boxes
        else:
            # batch_gligen["id_cfg"]["id_image"] = torch.cat((batch_gligen["id_cfg"]["id_image"], id_image), dim=0)
            batch_gligen["id_cfg"]["id_boxes"] = torch.cat((batch_gligen["id_cfg"]["id_boxes"], subject_id_boxes), dim=0)

        if batch_gligen["id_cfg"]["id_masks"] is None:
            batch_gligen["id_cfg"]["id_masks"] = id_masks
        else:
            batch_gligen["id_cfg"]["id_masks"] = torch.cat((batch_gligen["id_cfg"]["id_masks"], id_masks), dim=0)

    batch_gligen["id_cfg"]["id_embedding"] = cross_attention_kwargs.get("id_embedding", None)
    batch_gligen["task_type"] = batch["task_type"]

    return batch_gligen

# save weights
def save_id_weights(id_encoder, save_path):
    # logger.info("Saving id_encoder")
    ckpt: dict[str, torch.Tensor] = {}

    # 1) Flatten id_adapter
    for k, v in id_encoder.id_adapter.state_dict().items():
        ckpt[f"id_adapter.{k}"] = v

    # 2) Flatten id_adapter_attn_layers
    #    Note: keys from a ModuleList state_dict appear as '0.<param>', '1.<param>', ...
    for k, v in id_encoder.id_adapter_attn_layers.state_dict().items():
        ckpt[f"id_adapter_attn_layers.{k}"] = v

    # 3) Write to disk
    torch.save(ckpt, save_path + "id_encoder.bin")

def save_hoi_weights(unet, save_path):
    # logger.info("Saving hoi fuser")
    """
    Extract weights of all GatedSelfAttentionDense (fuser) modules from UNet and save to file.
    """
    fuser_state = OrderedDict()
    for name, module in unet.named_modules():
        # Find all fuser submodules
        if isinstance(module, GatedSelfAttentionDense):
            # module.state_dict() returns all parameters of the submodule
            for k, v in module.state_dict().items():
                # When saving, prepend parent module path to keys
                fuser_state[f"{name}.{k}"] = v

    if hasattr(unet, "position_net"):
        position_net_state = unet.position_net.state_dict()
        for k, v in position_net_state.items():
            fuser_state[f"position_net.{k}"] = v
    else:
        logger.warning("UNet does not have a position_net, skipping its weights.")

    torch.save(fuser_state, save_path + "fuser.bin")

def load_fuser_weights(unet, load_path, strict=False):
    """
    Load previously saved fuser weights back into UNet. strict=False allows missing or extra parameters.
    """
    # Load the file first
    fuser_state = torch.load(load_path, map_location=unet.device)

    # Extract UNet's full state_dict for merging
    full_state = unet.state_dict()

    # Update full_state with keys from fuser_state
    for k, v in fuser_state.items():
        if k in full_state:
            full_state[k] = v
        else:
            print(f"[Warning] key {k} not found in UNet state_dict, skipping.")

    # Reload into model
    unet.load_state_dict(full_state, strict=strict)
    print(f"Loaded fuser weights from {load_path}")

def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir='logs')
    # Initialize Accelerator object for distributed training setup
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='no',
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers_consishoi.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers_consishoi.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed >= 0:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            # Create output directory (if not exists)
            # Base directory path
            base_dir = Path(args.output_dir)
            
            # Add date suffix
            now = datetime.now()
            date_suffix = f"_{now.month}_{now.day}"
            base_dir = base_dir.parent / (base_dir.name + date_suffix)
            
            # Check if directory exists; add counter suffix if it does
            original_dir = base_dir
            counter = 1
            while base_dir.exists():
                base_dir = original_dir.parent / f"{original_dir.name}_{counter}"
                counter += 1
            
            # Update args.output_dir to the final path
            # import pdb; pdb.set_trace() # debug model weight path
            args.output_dir = str(base_dir)

            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.output_dir + '/ckpt', exist_ok=True)
            os.makedirs(args.output_dir + '/weight', exist_ok=True)

    # Dataset and DataLoaders creation:
    id_dataset = IDDataset(root_dir=args.id_dataset_path)
    hoi_dataset = HOIDataset(root_dir=args.hoi_dataset_path)
    consist_dataset = ConsishoiDataset(root_dir=args.consishoi_dataset_path)
    mixed_dataset = MixedDataset(id_dataset, hoi_dataset, consist_dataset, ratios=args.mixed_ratios_start, unconditional_prob = args.unconditional_prob,unconditional_sample = predefined_uncond_sample)

    collate_fn = partial(unified_collate_fn, device=accelerator.device)
    batch_sampler = ResolutionMixedBatchSampler(
        mixed_dataset,
        batch_size=args.batch_size,
    )
    # train_dataloader = DataLoader(
    #     mixed_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=collate_fn,
    # )
    train_dataloader = DataLoader(
        mixed_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )

    # Load Models
    logger.info(f"Loading models and tokenizers from:{args.consishoi_model_path}")
    pretrained_model_name_or_path = args.consishoi_model_path
    
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", device_map="auto"
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", device_map="auto"
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        device_map="auto",
    )

    unet_config = ConsishoiUNet2DConditionModel.load_config(pretrained_model_name_or_path, subfolder="unet")
    unet = ConsishoiUNet2DConditionModel.from_config(unet_config)
    unet_state_dict = ConsishoiUNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", map_location=accelerator.device).state_dict()
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)

    logger.info(f"Loaded models and tokenizers from:{args.consishoi_model_path}")

    # Initialize ID encoder using the same consishoi model path
    id_model_path_to_use = args.consishoi_model_path
    id_encoder = IDEncoder_pipeline(unet, accelerator.device, model_path=id_model_path_to_use)


    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Collect trainable parameters
    trainable_params = []
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, IDAttnProcessor):
            for sub_name, param in proc.named_parameters(recurse=True):
                if sub_name.startswith("id_to_") or sub_name in ("alpha", "beta"):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

    # for param in id_encoder.id_adapter.parameters():
    #     param.requires_grad = True
    #     trainable_params.append(param)

    # position_net trainable parameters
    ################
    for name, param in unet.position_net.named_parameters():
        if name in("null_id_feature", "alpha_id"):
            param.requires_grad = True
            trainable_params.append(param)

    for param in unet.position_net.conv1d.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    for param in unet.position_net.linear_id.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    #################

    for module in unet.modules():
        if isinstance(module, GatedSelfAttentionDense):
            for name, param in module.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)

    if args.use_xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer (changed to joint optimizer)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    max_train_steps = args.epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=1,
    )

    # Prepare everything with our `accelerator`.
    unet, id_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, id_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if overrode_max_train_steps:
        max_train_steps = args.epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Consishoi", config={"learning_rate": args.lr, "batch_size": args.batch_size},)

    # region Train!
    total_batch_size = args.batch_size * accelerator.num_processes

    # Save training configuration to a JSON file
    train_config = {
        "num_examples": len(mixed_dataset),
        "num_epochs": args.epochs,
        "batch_size_per_device": args.batch_size,
        "total_train_batch_size": total_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": max_train_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "output_dir": args.output_dir,
        "lr_scheduler": args.lr_scheduler if hasattr(args, "lr_scheduler") else None,
        "seed": args.seed if hasattr(args, "seed") else None,
        "consishoi_model_path": args.consishoi_model_path if hasattr(args, "consishoi_model_path") else None,
        "lr": args.lr,
        "frequency_loss": args.frequency_loss,
        "lambda_lf": args.lambda_lf,
        "lambda_hf": args.lambda_hf,
        "mixed_ratios_start": args.mixed_ratios_start,
        "mixed_ratios_end": args.mixed_ratios_end,
        "unconditional_prob": args.unconditional_prob,
        "args": str(args)
    }

    config_path = os.path.join(args.output_dir, "train_config.json")
    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=4)

    logger.info("***** Running Consishoi training *****")
    logger.info(f"  Num examples = {len(mixed_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir+'/ckpt')
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, 'ckpt', path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # region Train!
    # import pdb; pdb.set_trace() # debug training

    best_loss = 1000.0
    log_file = "bad_batches.log"
    for epoch in range(first_epoch, args.epochs):
        unet.train()
        train_loss = 0.0

        # Compute interpolation factor t in [0,1] for current epoch
        t = epoch / (args.epochs - 1)
        # Linear interpolation: cur_ratios = start * (1 - t) + end * t
        cur_ratios = [
            args.mixed_ratios_start[i] * (1.0 - t) + args.mixed_ratios_end[i] * t
            for i in range(3)
        ]
        mixed_dataset.set_ratios(cur_ratios)

        for step, batch in enumerate(train_dataloader):
            # Validate the batch
            captions = batch.get("caption", None)

            invalid = (
                captions is None or
                len(captions) == 0 or
                not isinstance(captions, list) or
                not all(isinstance(c, str) for c in captions)
            )

            if invalid:
                print(f"⚠️ Skipping invalid batch at step {step}")

                # Save to log file
                with open(log_file, "a") as f:
                    json.dump({
                        "step": step,
                        "caption": captions,
                        "batch_keys": list(batch.keys()),
                        "id_boxes": batch.get("id_boxes", "no id_boxes"),
                        "subject_phrases": batch.get("subject_phrases", "no subject_phrases"),
                        "object_phrases": batch.get("object_phrases", "no object_phrases"),
                        "action_phrases": batch.get("action_phrases", "no action_phrases"),
                        "subject_boxes": batch.get("subject_boxes", "no subject_boxes"),
                        "object_boxes": batch.get("object_boxes", "no object_boxes"),
                        "task_type": batch.get("task_type", "no task_type"),
                    }, f)
                    f.write("\n")

                continue
            # import pdb; pdb.set_trace() # debug training
            with accelerator.accumulate(unet, id_encoder):
                if args.offload:
                    vae.to(accelerator.device, dtype=weight_dtype)
                    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
                    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
                # Convert images to latent space
                #################################### Inspect latent sizes at different resolutions
                latents = vae.encode(batch["target_image"].to(dtype=weight_dtype)).latent_dist.sample().detach()

                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # time ids
                height = batch["target_image"].shape[2]
                width = batch["target_image"].shape[3]
                original_size = (height, width)
                target_size = (height, width)
                crops_coords_top_left = (0, 0)

                add_time_ids = compute_time_ids(original_size, crops_coords_top_left, target_size).repeat(bsz, 1).to(accelerator.device, dtype=weight_dtype)
                unet_added_conditions = {"time_ids": add_time_ids}

                # encode prompt
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    prompt=batch["caption"],
                    device=accelerator.device,
                    tokenizer_one=tokenizer_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two
                )

                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                prompt_embeds = prompt_embeds.to(accelerator.device)

                # prepare feature encoder inputs
                _, id_embedding = id_encoder.get_id_embeddings(batch["id_image"]).chunk(2, dim=0)
                id_scale = 0.8

                cross_attention_kwargs = {}

                cross_attention_kwargs.update({
                    "id_embedding": id_embedding,
                    "id_scale": id_scale,
                })

                batch_gligen = prepare_gligen_inputs(
                    batch,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    cross_attention_dim=unet.config.cross_attention_dim,
                    batch_size=args.batch_size,
                    tokenizer_one=tokenizer_one,
                    tokenizer_two=tokenizer_two,
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

                cross_attention_kwargs.update({
                    "gligen": batch_gligen,
                })


                if args.offload:
                    del batch["id_image"]
                    del batch["target_image"]
                    del batch["caption"]
                    vae.to("cpu")
                    text_encoder_one.to("cpu")
                    text_encoder_two.to("cpu")
                    torch.cuda.empty_cache()

                # Predict the noise residual
                # region Predict
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    cross_attention_kwargs=cross_attention_kwargs,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred, target, reduction="mean")
                train_loss = loss.item()

                # frequency loss
                if batch["task_type"] == "Consishoi" and args.frequency_loss:
                    loss_hoi_lf, loss_id_hf = compute_frequency_losses(
                        model_pred, target, timesteps,
                        batch["id_boxes"], batch["subject_boxes"], batch["object_boxes"],
                        noise_scheduler,
                        args.low_freq_radius_loss, args.high_freq_radius_loss,
                    )

                    loss = loss + args.lambda_lf * loss_hoi_lf + args.lambda_hf * loss_id_hf


                accelerator.backward(loss)

                # if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, 'ckpt', f"checkpoint-{initial_global_step + global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            train_loss = 0.0
        
        # Save weights
        weight_name = (
            f"learned-steps-{global_step}-"
        )
        save_path = os.path.join(args.output_dir, 'weight', weight_name)

        save_id_weights(id_encoder, save_path)
        save_hoi_weights(unet, save_path)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    # print(f'Best loss:{best_loss} @@@ Step:{best_step}')
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--consishoi_model_path', type=str, required=True, help='Path to pre-trained Consishoi model.')
    
    parser.add_argument('--id_dataset_path', type=str, required=True, help='Path to ID dataset root directory.')
    parser.add_argument('--hoi_dataset_path', type=str, required=True, help='Path to HOI dataset root directory.')
    parser.add_argument('--consishoi_dataset_path', type=str, required=True, help='Path to Consishoi dataset root directory.')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offload', type=bool, default=False)
    parser.add_argument('--use_xformers', action='store_true')
    parser.add_argument('--allow_tf32', action='store_true')

    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--mixed_ratios', type=List, default=[2, 1, 3])
    parser.add_argument('--mixed_ratios_start', type=List, default=[2, 1, 8])
    parser.add_argument('--mixed_ratios_end', type=List, default=[2, 1, 8])
    parser.add_argument('--mixed_ratios_mode', type=str, default='ratios_214')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--unconditional_prob', type=float, default=0.1)
    parser.add_argument('--frequency_loss', type=bool, default=True, help="Whether to use frequency loss")
    parser.add_argument("--lambda_lf", type=float, default=0.1, help="Weight for the loss function of HOI")
    parser.add_argument("--lambda_hf", type=float, default=0.2, help="Weight for the loss function of ID")
    parser.add_argument('--low_freq_radius_loss', type=float, default=0.02)
    parser.add_argument('--high_freq_radius_loss', type=float, default=0.7)

    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--checkpointing_steps', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default='./OUTPUTS/train')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--ablation', type=int, default=0)

    args = parser.parse_args()

    if args.mixed_ratios_mode == 'ratios_114':
        args.mixed_ratios_start = [1, 1, 4]
        args.mixed_ratios_end = [1, 1, 4]

    main(args)