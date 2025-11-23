"""Consishoi UNet module: removed Chinese comments for clarity."""
from typing import Optional, Tuple, Union
from diffusers.models.embeddings import get_fourier_embeds_from_boundingbox
import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel


class AbsolutePositionalEmbedding(nn.Module):
    """Absolute positional embedding for sequence positions."""
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class ConsishoiInteractionProjection(nn.Module):
    """Projection and embedding of bounding boxes and interactions."""
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4

        # id_image MLP dim
        self.clip_in_dim = 768
        self.clip_out_dim = 768

        # absolute positional embeddings for instances and roles
        self.interaction_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=32)
        self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linear_action = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_action_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def get_between_box(self, bbox1, bbox2):
        """Compute the box between two boxes (sorted coordinates)."""
        all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]], dim=-1)
        all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]], dim=-1)
        all_x, _ = all_x.sort()
        all_y, _ = all_y.sort()
        return torch.stack([all_x[:, :, 1], all_y[:, :, 1], all_x[:, :, 2], all_y[:, :, 2]], 2)

    def forward(
        self,
        subject_boxes, object_boxes,
        masks,
        subject_positive_embeddings, object_positive_embeddings, action_positive_embeddings,
        id_images,
    ):
        masks = masks.unsqueeze(-1)

        action_boxes = self.get_between_box(subject_boxes, object_boxes)
        subject_xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, subject_boxes)
        object_xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, object_boxes)
        action_xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, action_boxes)

        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        action_null = self.null_action_feature.view(1, 1, -1)

        subject_positive_embeddings = subject_positive_embeddings * masks + (1 - masks) * positive_null
        object_positive_embeddings = object_positive_embeddings * masks + (1 - masks) * positive_null

        subject_xyxy_embedding = subject_xyxy_embedding * masks + (1 - masks) * xyxy_null
        object_xyxy_embedding = object_xyxy_embedding * masks + (1 - masks) * xyxy_null
        action_xyxy_embedding = action_xyxy_embedding * masks + (1 - masks) * xyxy_null

        action_positive_embeddings = action_positive_embeddings * masks + (1 - masks) * action_null

        objs_subject = self.linears(torch.cat([subject_positive_embeddings, subject_xyxy_embedding], dim=-1))
        objs_object = self.linears(torch.cat([object_positive_embeddings, object_xyxy_embedding], dim=-1))
        objs_action = self.linear_action(torch.cat([action_positive_embeddings, action_xyxy_embedding], dim=-1))

        objs_subject = objs_subject + self.interaction_embedding(objs_subject)
        objs_object = objs_object + self.interaction_embedding(objs_object)
        objs_action = objs_action + self.interaction_embedding(objs_action)

        objs_subject = objs_subject + self.position_embedding.emb(torch.tensor(0).to(objs_subject.device))
        objs_object = objs_object + self.position_embedding.emb(torch.tensor(1).to(objs_object.device))
        objs_action = objs_action + self.position_embedding.emb(torch.tensor(2).to(objs_action.device))

        objs = torch.cat([objs_subject, objs_action, objs_object], dim=1)

        return objs


class ConsishoiUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        super(ConsishoiUNet2DConditionModel, self).__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads
        )

        # load position_net
        positive_len = 768
        if isinstance(self.config.cross_attention_dim, int):
            positive_len = self.config.cross_attention_dim
        elif isinstance(self.config.cross_attention_dim, tuple) or isinstance(self.config.cross_attention_dim, list):
            positive_len = self.config.cross_attention_dim[0]

        # 创建一个position_net对HOI进行建模
        self.position_net = ConsishoiInteractionProjection(
            in_dim=positive_len, out_dim=self.config.cross_attention_dim
        )
