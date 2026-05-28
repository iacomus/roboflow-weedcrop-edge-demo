#!/usr/bin/env python
"""Local RF-DETR Small -> CoreML export from a trained checkpoint (macOS).

Going deep on the deformable-attention conversion. Four patches, each closing
a real gap between the open-source model and a CoreML-convertible graph:

  1) coremltools int/bool op override -> stock `_cast` does int(np.array([x])),
     which numpy>=1.25 rejects for 1-D arrays. Take element 0 instead.
  2) F.interpolate bicubic -> bilinear at trace time (coremltools supports
     neither bicubic variant; bilinear is fine for the smooth DINOv2 pos-embed).
  3) torch.meshgrid inputs flattened to 1-D (coremltools rejects rank>1 inputs).
  4) MSDeformAttn.forward reimplemented rank-<=4. RF-DETR is single-scale
     (n_levels=1), so the standard (N,Lq,heads,levels,points,2) rank-6 sampling
     tensor has a redundant level dim that blows CoreML's rank-5 ceiling. This
     version drops it and reuses the SAME `_bilinear_grid_sample` + conventions,
     so it's numerically identical (asserted below before converting).

Run (from training/, venv active):  python export_coreml.py
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from rfdetr import RFDETRSmall
from rfdetr.models.ops.functions.ms_deform_attn_func import _bilinear_grid_sample

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "colab", "checkpoint_best_ema.pth")
OUT = os.path.join(HERE, "rfdetr_small.mlpackage")


def _patch_coreml_ops():
    def _safe_cast(context, node, pydtype, mil_dtype):
        x = _get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const():
            v = x.val
            if isinstance(v, np.ndarray):
                v = v.reshape(-1)[0].item()
            res = mb.const(val=pydtype(v), name=node.name)
        elif len(x.shape) > 0:
            xs = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=xs, dtype=mil_dtype, name=node.name)
        else:
            res = mb.cast(x=x, dtype=mil_dtype, name=node.name)
        context.add(res, node.name)

    @register_torch_op(torch_alias=["int"], override=True)
    def _int(context, node):
        _safe_cast(context, node, int, "int32")

    @register_torch_op(torch_alias=["bool"], override=True)
    def _bool(context, node):
        _safe_cast(context, node, bool, "bool")


def _forward_rank_safe(self, query, reference_points, input_flatten,
                       input_spatial_shapes, input_level_start_index,
                       input_padding_mask=None, input_spatial_shapes_hw=None):
    """Single-scale (n_levels==1) deformable attention with all tensors rank<=4.
    Mirrors ms_deform_attn_core_pytorch exactly, minus the redundant level dim."""
    assert self.n_levels == 1, "rank-safe forward is specialized for single-scale"
    B, Lq, _ = query.shape
    B, S, _ = input_flatten.shape
    M, P = self.n_heads, self.n_points
    Dh = self.d_model // M

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    sampling_offsets = self.sampling_offsets(query).view(B, Lq, M, P, 2)
    attention_weights = self.attention_weights(query).view(B, Lq, M, P)
    attention_weights = F.softmax(attention_weights, -1)

    ref = reference_points[:, :, 0]  # (B, Lq, 2 or 4)
    if ref.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)[0]  # (2,)
        sampling_locations = ref[:, :, None, None, :] + sampling_offsets / offset_normalizer
    elif ref.shape[-1] == 4:
        sampling_locations = (ref[:, :, None, None, :2]
                              + sampling_offsets / P * ref[:, :, None, None, 2:] * 0.5)
    else:
        raise ValueError("reference_points last dim must be 2 or 4")

    if input_spatial_shapes_hw is not None:
        H, W = input_spatial_shapes_hw[0]
    else:
        H = int(input_spatial_shapes[0, 0])
        W = int(input_spatial_shapes[0, 1])

    value = value.transpose(1, 2).contiguous().view(B, M, Dh, S).view(B * M, Dh, H, W)
    grid = (2 * sampling_locations - 1).transpose(1, 2).reshape(B * M, Lq, P, 2)
    sampled = _bilinear_grid_sample(value, grid, padding_mode="zeros", align_corners=False)
    aw = attention_weights.transpose(1, 2).reshape(B * M, 1, Lq, P)
    out = (sampled * aw).sum(-1).view(B, M * Dh, Lq).transpose(1, 2).contiguous()
    return self.output_proj(out)


def main():
    _patch_coreml_ops()
    print(f"[export] loading {CKPT}")
    m = RFDETRSmall(pretrain_weights=CKPT)
    net = copy.deepcopy(m.model.model).eval().to("cpu")
    R = int(getattr(m.model, "resolution", 512))
    print(f"[export] input resolution: {R}")

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            # ImageNet normalization baked in so the .mlpackage accepts a plain
            # RGB image (CoreML ImageType feeds [0,1] via scale=1/255) and iOS
            # Vision can drive it without manual MLMultiArray construction.
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            o = self.net(x)
            return o["pred_boxes"], o["pred_logits"]

    dummy = torch.randn(1, 3, R, R)

    # 1) numerically validate the rank-safe attention BEFORE converting.
    deform_cls = None
    for _, mod in net.named_modules():
        if "DeformAttn" in mod.__class__.__name__:
            deform_cls = type(mod)
            break
    assert deform_cls is not None, "no deformable-attention module found"
    with torch.no_grad():
        b0, l0 = Wrap(net).eval()(dummy)
    _orig_forward = deform_cls.forward
    deform_cls.forward = _forward_rank_safe
    with torch.no_grad():
        b1, l1 = Wrap(net).eval()(dummy)
    db = (b0 - b1).abs().max().item()
    dl = (l0 - l1).abs().max().item()
    print(f"[export] rank-safe attention max|d|: boxes={db:.2e} logits={dl:.2e}")
    assert db < 1e-3 and dl < 1e-3, "rank-safe attention diverged from original!"

    # 2) trace with interpolate + meshgrid patches, then convert.
    _oi, _om = F.interpolate, torch.meshgrid

    def _no_aa(*a, **k):
        if k.get("mode") == "bicubic":
            k["mode"] = "bilinear"
        if k.get("mode") == "bilinear":
            k["antialias"] = False
        return _oi(*a, **k)

    def _flat_meshgrid(*ts, **k):
        return _om(*(t.reshape(-1) for t in ts), **k)

    F.interpolate, torch.meshgrid = _no_aa, _flat_meshgrid
    try:
        traced = torch.jit.trace(Wrap(net).eval(), dummy, strict=False)
    finally:
        F.interpolate, torch.meshgrid = _oi, _om

    print("[export] trace OK -> converting to CoreML mlprogram ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, R, R), scale=1 / 255.0,
                             bias=[0.0, 0.0, 0.0], color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="boxes"), ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    mlmodel.short_description = ("RF-DETR Small weed/crop detector. Input: 512x512 RGB image. "
                                 "boxes=cxcywh normalized [0,1]; logits pre-sigmoid; class channel 1=crop, 2=weed.")
    mlmodel.input_description["image"] = "512x512 RGB image (ImageNet normalization handled inside the model)"
    mlmodel.save(OUT)
    print(f"[export] saved {OUT}")

    print("[export] validating: load mlpackage + predict on an image ...")
    loaded = ct.models.MLModel(OUT)
    spec = loaded.get_spec()
    print("  inputs :", [(i.name, i.type.WhichOneof("Type")) for i in spec.description.input])
    print("  outputs:", [o.name for o in spec.description.output])
    from PIL import Image
    img = Image.fromarray((np.random.rand(R, R, 3) * 255).astype(np.uint8))
    pred = loaded.predict({"image": img})
    for k, v in pred.items():
        print(f"  out {k}: shape={getattr(v, 'shape', type(v).__name__)}")
    print("[export] CoreML validation OK")


if __name__ == "__main__":
    main()
