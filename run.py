"""Track 1 inference entry point: deepdrivepl, testing-phase submission #19263 (`test-e19g-s64k`).

The organizers call   main(model_dir, input_dir, output_dir, device).

Every output comes from its own input image and the water type / depth level at the end of its
own file name (`<scene>_<A>_<B>.png`, read with rsplit; the scene part is never used).
Outputs are written at the scoring size (longest side 1024, aspect kept), so the organizers'
`resize_for_submission.py` step is an identity on them, as it was for our upload.

model_dir (weights/) holds:
  dinov3-vitl16/            DINOv3 ViT-L/16 LVD-1689M (Meta, DINOv3 License), Hugging Face format
  e23-s64k-all.pt           mask decoder + FiLM, 64k steps
  e19g-ref-b64-s48k-all.pt  restoration refiner, U-Net base 64, 48k steps
  generator_affine.csv      per-(A, B) sRGB affine table
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.eurs_t1 import (  # noqa: E402
    Decoder,
    UNet,
    load_eval_rgb,
    load_table,
    parse_cell,
    refine,
    segment,
    unsharp,
)

BACKBONE = "dinov3-vitl16"
DECODER = "e23-s64k-all.pt"
REFINER = "e19g-ref-b64-s48k-all.pt"
TABLE = "generator_affine.csv"
UNSHARP = 0.25


def main(model_dir, input_dir, output_dir, device):
    from transformers import AutoModel

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dev = torch.device(device)
    backbone = AutoModel.from_pretrained(os.path.join(model_dir, BACKBONE)).eval().to(dev)
    st = torch.load(os.path.join(model_dir, DECODER), map_location=dev, weights_only=True)
    dec = Decoder(dim=int(backbone.config.hidden_size), cond=st["cond"]).to(dev)
    dec.load_state_dict(st["decoder"])
    dec.eval()
    ck = torch.load(os.path.join(model_dir, REFINER), map_location=dev, weights_only=True)
    net = UNet(base=ck["base"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    table = load_table(os.path.join(model_dir, TABLE))

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        a, b = parse_cell(stem)
        img = load_eval_rgb(os.path.join(input_dir, name))
        restored = unsharp(refine(net, img, table, a, b, dev), amount=UNSHARP)
        Image.fromarray(restored, mode="RGB").save(os.path.join(output_dir, "restored", stem + ".png"))
        mask = segment(backbone, dec, img, a, b, dev)
        Image.fromarray(mask, mode="L").save(os.path.join(output_dir, "seg_masks", stem + ".png"))
