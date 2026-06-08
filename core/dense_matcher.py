import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.utils import gauss_blur


class RowTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn0 = nn.MultiheadAttention(d_model, nhead)
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn0 = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead)

        self.norm0_self = nn.LayerNorm(d_model)
        self.norm1_self = nn.LayerNorm(d_model)
        self.norm0_cross = nn.LayerNorm(d_model)
        self.norm1_cross = nn.LayerNorm(d_model)
        self.norm0_ffn = nn.LayerNorm(d_model)
        self.norm1_ffn = nn.LayerNorm(d_model)

        self.ffn0 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def _attend(self, query, key, value, attn, norm):
        residual = query
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        updated, _ = attn(query, key, value)
        updated = updated.transpose(0, 1)
        return norm(residual + updated)

    def _feedforward(self, tokens, ffn, norm):
        return norm(tokens + ffn(tokens))

    def forward(self, feat0, feat1):
        feat0 = self._attend(feat0, feat0, feat0, self.self_attn0, self.norm0_self)
        feat1 = self._attend(feat1, feat1, feat1, self.self_attn1, self.norm1_self)

        cross_feat0 = self._attend(feat0, feat1, feat1, self.cross_attn0, self.norm0_cross)
        cross_feat1 = self._attend(feat1, feat0, feat0, self.cross_attn1, self.norm1_cross)

        feat0 = self._feedforward(cross_feat0, self.ffn0, self.norm0_ffn)
        feat1 = self._feedforward(cross_feat1, self.ffn1, self.norm1_ffn)
        return feat0, feat1


class DenseMatcher(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            RowTransformerLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

    def forward(self, feat0, feat1, image0=None, image1=None):
        batch, channels, height, width = feat0.shape
        feat0 = feat0.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        feat1 = feat1.permute(0, 2, 3, 1).reshape(batch * height, width, channels)

        for layer in self.layers:
            feat0, feat1 = layer(feat0, feat1)

        flow_init = self.compute_flow_init(feat0, feat1, batch, height, width)

        feat0 = feat0.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        feat1 = feat1.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return feat0, feat1, flow_init

    def compute_flow_init(self, feat0, feat1, batch, height, width):
        feat0 = F.normalize(feat0, dim=-1)
        feat1 = F.normalize(feat1, dim=-1)

        scores = torch.matmul(feat0, feat1.transpose(1, 2)) * math.sqrt(feat0.shape[-1])

        x_left = torch.arange(width, device=feat0.device)
        x_right = torch.arange(width, device=feat0.device)
        valid = x_right.view(1, 1, width) <= x_left.view(1, width, 1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        x_right = torch.sum(probs * x_right.view(1, 1, width).to(probs.dtype), dim=-1)
        x_left = x_left.view(1, width).to(probs.dtype)

        flow_x = (x_right - x_left).reshape(batch, height, width).unsqueeze(1)
        flow_y = torch.zeros_like(flow_x)
        return torch.cat([flow_x, flow_y], dim=1)


class MatchProjector(nn.Module):
    def __init__(self, confidence_thresh=0.2, max_vertical_offset=2.0, blur_kernel=7, blur_std=2.0):
        super().__init__()
        self.confidence_thresh = confidence_thresh
        self.max_vertical_offset = max_vertical_offset
        self.blur_kernel = max(1, int(blur_kernel))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.blur_std = blur_std

    def forward(self, keypoints0, keypoints1, confidence, batch_indexes, image_shape, feature_shape):
        keypoints0, keypoints1, confidence, batch_indexes = self._flatten_matches(
            keypoints0,
            keypoints1,
            confidence,
            batch_indexes,
        )

        batch_size = image_shape[0]
        feature_height, feature_width = feature_shape
        flow_init = keypoints0.new_zeros(batch_size, 2, feature_height, feature_width)
        if keypoints0.numel() == 0:
            return flow_init

        image_height, image_width = image_shape[1:]
        valid = torch.isfinite(keypoints0).all(dim=-1) & torch.isfinite(keypoints1).all(dim=-1)
        valid &= torch.isfinite(confidence)
        valid &= confidence >= self.confidence_thresh
        valid &= batch_indexes >= 0
        valid &= batch_indexes < batch_size
        valid &= keypoints1[:, 0] <= keypoints0[:, 0] + 1e-3
        if self.max_vertical_offset >= 0:
            valid &= (keypoints1[:, 1] - keypoints0[:, 1]).abs() <= self.max_vertical_offset

        if not valid.any():
            return flow_init

        keypoints0 = keypoints0[valid]
        keypoints1 = keypoints1[valid]
        confidence = confidence[valid]
        batch_indexes = batch_indexes[valid]

        scale_x = (feature_width - 1) / max(image_width - 1, 1)
        scale_y = (feature_height - 1) / max(image_height - 1, 1)
        keypoints0 = keypoints0.clone()
        keypoints1 = keypoints1.clone()
        keypoints0[:, 0] *= scale_x
        keypoints1[:, 0] *= scale_x
        keypoints0[:, 1] *= scale_y
        keypoints1[:, 1] *= scale_y

        src_x = torch.round(keypoints0[:, 0]).long()
        src_y = torch.round(keypoints0[:, 1]).long()
        valid = (src_x >= 0) & (src_x < feature_width) & (src_y >= 0) & (src_y < feature_height)
        if not valid.any():
            return flow_init

        keypoints0 = keypoints0[valid]
        keypoints1 = keypoints1[valid]
        confidence = confidence[valid]
        batch_indexes = batch_indexes[valid]
        src_x = src_x[valid]
        src_y = src_y[valid]

        flow_x = keypoints1[:, 0] - keypoints0[:, 0]
        flow_y = keypoints1[:, 1] - keypoints0[:, 1]
        linear_index = batch_indexes * (feature_height * feature_width) + src_y * feature_width + src_x

        weight = confidence.new_zeros(batch_size * feature_height * feature_width)
        flow_x_acc = confidence.new_zeros(batch_size * feature_height * feature_width)
        flow_y_acc = confidence.new_zeros(batch_size * feature_height * feature_width)

        weight.index_add_(0, linear_index, confidence)
        flow_x_acc.index_add_(0, linear_index, confidence * flow_x)
        flow_y_acc.index_add_(0, linear_index, confidence * flow_y)

        weight = weight.view(batch_size, 1, feature_height, feature_width)
        flow_init[:, 0] = flow_x_acc.view(batch_size, feature_height, feature_width)
        flow_init[:, 1] = flow_y_acc.view(batch_size, feature_height, feature_width)
        flow_init = flow_init / weight.clamp(min=1e-6)

        valid_map = (weight > 0).to(flow_init.dtype)
        if self.blur_kernel > 1:
            smooth_flow = gauss_blur(flow_init * valid_map, N=self.blur_kernel, std=self.blur_std)
            smooth_weight = gauss_blur(valid_map, N=self.blur_kernel, std=self.blur_std)
            filled_flow = smooth_flow / smooth_weight.clamp(min=1e-6)
            flow_init = torch.where(valid_map.bool(), flow_init, filled_flow)

        flow_init[:, 0] = torch.minimum(flow_init[:, 0], torch.zeros_like(flow_init[:, 0]))
        flow_init[:, 1] = 0.0
        return flow_init

    def _flatten_matches(self, keypoints0, keypoints1, confidence, batch_indexes):
        if keypoints0.ndim == 3:
            batch_size, num_matches, _ = keypoints0.shape
            if batch_indexes is None:
                batch_indexes = torch.arange(batch_size, device=keypoints0.device).view(batch_size, 1).expand(batch_size, num_matches)
            keypoints0 = keypoints0.reshape(-1, 2)
            keypoints1 = keypoints1.reshape(-1, 2)
        elif keypoints0.ndim == 2:
            if batch_indexes is None:
                batch_indexes = keypoints0.new_zeros(keypoints0.shape[0], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported match tensor shape: {tuple(keypoints0.shape)}")

        if confidence is None:
            confidence = keypoints0.new_ones(keypoints0.shape[0])
        else:
            confidence = confidence.reshape(-1).to(keypoints0.device)

        batch_indexes = batch_indexes.reshape(-1).long().to(keypoints0.device)
        return keypoints0, keypoints1, confidence, batch_indexes


class LoFTRDenseMatcher(nn.Module):
    def __init__(self, pretrained='outdoor', trainable=False, max_side=1024, **projector_kwargs):
        super().__init__()
        try:
            from kornia.feature import LoFTR
        except ImportError as exc:
            raise ImportError(
                "LoFTR frontend requires kornia. Install a compatible kornia build before using --dense_frontend_type loftr."
            ) from exc

        self.matcher = LoFTR(pretrained=pretrained)
        self.projector = MatchProjector(**projector_kwargs)
        self.trainable = trainable
        self.max_side = max_side
        if not trainable:
            for parameter in self.matcher.parameters():
                parameter.requires_grad_(False)
            self.matcher.eval()

    def forward(self, feat0, feat1, image0=None, image1=None):
        if image0 is None or image1 is None:
            raise ValueError("LoFTR frontend requires the input image pair")

        self.matcher.train(self.training and self.trainable)
        matcher_image0 = self._prepare_image(image0)
        matcher_image1 = self._prepare_image(image1)
        matcher_image0, scale = self._resize_for_match(matcher_image0)
        matcher_image1, _ = self._resize_for_match(matcher_image1, scale=scale)
        matcher_inputs = {
            'image0': matcher_image0,
            'image1': matcher_image1,
        }
        context = contextlib.nullcontext() if self.trainable else torch.no_grad()
        with context:
            correspondences = self.matcher(matcher_inputs)

        if scale != 1.0:
            correspondences = dict(correspondences)
            correspondences['keypoints0'] = correspondences['keypoints0'] / scale
            correspondences['keypoints1'] = correspondences['keypoints1'] / scale

        flow_init = self.projector(
            correspondences['keypoints0'],
            correspondences['keypoints1'],
            correspondences.get('confidence'),
            correspondences.get('batch_indexes'),
            (image0.shape[0], image0.shape[-2], image0.shape[-1]),
            feat0.shape[-2:],
        )
        return feat0, feat1, flow_init

    def _prepare_image(self, image):
        image = image.float() / 255.0
        if image.shape[1] == 1:
            return image

        weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        return (image[:, :3] * weights).sum(dim=1, keepdim=True)

    def _resize_for_match(self, image, scale=None):
        _, _, height, width = image.shape
        if scale is None:
            scale = min(1.0, float(self.max_side) / float(max(height, width)))
        if scale >= 1.0:
            return image, 1.0

        target_height = max(1, int(round(height * scale)))
        target_width = max(1, int(round(width * scale)))
        if target_height == height and target_width == width:
            return image, 1.0

        resized = F.interpolate(image, size=(target_height, target_width), mode='bilinear', align_corners=False)
        return resized, scale


class RoMaDenseMatcher(nn.Module):
    def __init__(self, pretrained='outdoor', trainable=False, **projector_kwargs):
        super().__init__()
        try:
            from romatch import roma_outdoor
        except ImportError as exc:
            raise ImportError(
                "RoMa frontend requires romatch. Install a compatible romatch build before using --dense_frontend_type roma."
            ) from exc

        if pretrained != 'outdoor':
            raise ValueError("The current RoMa adapter supports the public roma_outdoor preset only")

        self.matcher = roma_outdoor(device='cpu')
        self.projector = MatchProjector(**projector_kwargs)
        self.trainable = trainable
        if hasattr(self.matcher, 'parameters') and not trainable:
            for parameter in self.matcher.parameters():
                parameter.requires_grad_(False)
        if hasattr(self.matcher, 'eval') and not trainable:
            self.matcher.eval()

    def forward(self, feat0, feat1, image0=None, image1=None):
        if image0 is None or image1 is None:
            raise ValueError("RoMa frontend requires the input image pair")

        if hasattr(self.matcher, 'train'):
            self.matcher.train(self.training and self.trainable)

        matcher_image0 = self._prepare_image(image0)
        matcher_image1 = self._prepare_image(image1)
        context = contextlib.nullcontext() if self.trainable else torch.no_grad()
        with context:
            try:
                warp, certainty = self.matcher.match(matcher_image0, matcher_image1, device=str(image0.device))
            except Exception as exc:
                raise RuntimeError(
                    "The installed romatch package did not accept tensor inputs. Use a tensor-capable romatch build or switch to --dense_frontend_type loftr."
                ) from exc

            matches, match_confidence = self.matcher.sample(warp, certainty)

        keypoints0, keypoints1, batch_indexes = self._matches_to_pixel_coordinates(matches, image0.shape)
        if match_confidence is None:
            match_confidence = certainty

        flow_init = self.projector(
            keypoints0,
            keypoints1,
            match_confidence,
            batch_indexes,
            (image0.shape[0], image0.shape[-2], image0.shape[-1]),
            feat0.shape[-2:],
        )
        return feat0, feat1, flow_init

    def _prepare_image(self, image):
        image = image.float() / 255.0
        if image.shape[1] == 3:
            return image
        if image.shape[1] == 1:
            return image.repeat(1, 3, 1, 1)
        return image[:, :3]

    def _matches_to_pixel_coordinates(self, matches, image_shape):
        if matches.shape[-1] != 4:
            raise ValueError(f"Unsupported RoMa match shape: {tuple(matches.shape)}")

        if matches.ndim == 2:
            batch_indexes = matches.new_zeros(matches.shape[0], dtype=torch.long)
        elif matches.ndim == 3:
            batch_size, num_matches, _ = matches.shape
            batch_indexes = torch.arange(matches.shape[0], device=matches.device).view(batch_size, 1).expand(batch_size, num_matches)
        else:
            raise ValueError(f"Unsupported RoMa match shape: {tuple(matches.shape)}")

        height, width = image_shape[-2:]
        keypoints0 = self._normalized_to_pixels(matches[..., :2], width, height)
        keypoints1 = self._normalized_to_pixels(matches[..., 2:], width, height)
        return keypoints0, keypoints1, batch_indexes

    def _normalized_to_pixels(self, points, width, height):
        points = points.clone()
        points[..., 0] = (points[..., 0] + 1.0) * 0.5 * max(width - 1, 1)
        points[..., 1] = (points[..., 1] + 1.0) * 0.5 * max(height - 1, 1)
        return points


def build_dense_frontend(args):
    frontend_type = getattr(args, 'dense_frontend_type', 'row_transformer')
    if frontend_type == 'row_transformer':
        return DenseMatcher(
            d_model=256,
            nhead=args.dense_matcher_heads,
            dim_feedforward=args.dense_matcher_ffn_dim,
            num_layers=args.dense_matcher_layers,
        )

    projector_kwargs = dict(
        confidence_thresh=getattr(args, 'dense_frontend_confidence_thresh', 0.2),
        max_vertical_offset=getattr(args, 'dense_frontend_max_vertical_offset', 2.0),
        blur_kernel=getattr(args, 'dense_frontend_blur_kernel', 7),
        blur_std=getattr(args, 'dense_frontend_blur_std', 2.0),
    )

    if frontend_type == 'loftr':
        return LoFTRDenseMatcher(
            pretrained=getattr(args, 'dense_frontend_pretrained', 'outdoor'),
            trainable=getattr(args, 'dense_frontend_trainable', False),
            **projector_kwargs,
        )

    if frontend_type == 'roma':
        return RoMaDenseMatcher(
            pretrained=getattr(args, 'dense_frontend_pretrained', 'outdoor'),
            trainable=getattr(args, 'dense_frontend_trainable', False),
            **projector_kwargs,
        )

    raise ValueError(f"Unsupported dense frontend type: {frontend_type}")
