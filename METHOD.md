# Method description

Team: deepdrivepl   Track(s): both (this repository: Track 1)   Reported testing-phase submission(s): Track 1 #19263 (`test-e19g-s64k`, composite 0.655)

## Approach (one page at most)

Two independent branches per image. Restoration and segmentation share nothing except the input frame and the water type / depth level `(A, B)`. `(A, B)` is read from the end of the image's own file name (`stem.rsplit("_", 2)`; the scene part is never used). Our leaderboard rows are flagged Meta for this. No other image is used: no scene pooling, no cross-image fusion, no test-time adaptation.

**Restoration**
1. Per-cell sRGB affine (`weights/generator_affine.csv`). For each `(A, B)` cell and channel, `gt ≈ a·degraded + b` is fitted by least squares on the EURS training split (all 20 scenes). The fit recovers the RSUIGM degradation per cell.
2. Residual U-Net refiner (4 levels, base 64, GroupNorm/GELU).
   - Input, 8 channels: the affine output, the degraded frame, and `A/10`, `B/100` as constant planes. It predicts a residual on the affine image and runs in fp16.
   - Training: L1 + 0.5·(1 − SSIM) against the clean ground truth, AdamW lr 3e-4, batch 8, 384² crops, 48,000 steps, all 20 training scenes.
   - The output is rounded to multiples of 4 (to stay under the 1 GB upload cap).
3. Fixed unsharp mask: amount 0.25, Gaussian sigma 1 (OpenCV), rounded to multiples of 4 again.

**Segmentation**
- Backbone: DINOv3 ViT-L/16 (LVD-1689M), frozen, run in fp16. Input is the frame rotated to landscape (portraits by 90°) and resized to 1024×768.
- Decoder (3.8 M trainable parameters):
  - hidden states of blocks 6/12/18/24 go through LayerNorm and a 1×1 projection to 256 ch each, and are fused at 1/16;
  - FiLM on the fused map from the `(A, B)` cell (one-hot A + B/100 → MLP → per-channel scale/shift);
  - two ×2 upsample-conv stages to 1/4, a 1×1 head to 5 logits, bilinear back to the image size.
- Training:
  - cross-entropy (ignore 255), AdamW lr 1e-3, OneCycle, batch 4, 64,000 steps, random area crop 0.5–1 + horizontal flip;
  - half of every batch is EURS train (20 scenes × 200 variants);
  - the other half is a public underwater image re-degraded on the fly with the forward model `I = J·t + B·(1 − t)` fitted on EURS train, at a random `(A, B)` cell (see Training data).
- Inference: horizontal-flip TTA (mean of logits), argmax.

## Training data

- EURS training split: all 20 scenes for the final weights.
- Model selection used two sources:
  - a 5-fold split by scene of the 20 training scenes;
  - the development-phase leaderboard (public Dev split, only through its scores).
- The released testing-phase images were not used for training, tuning, model selection or pseudo-labelling.

External data and pretrained models (declared "Other" on every submission):
- **DINOv3 ViT-L/16 LVD-1689M** (Meta, https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), DINOv3 License. Its licence text is in the weights archive at `weights/dinov3-vitl16/LICENSE.md`.
- **Public underwater segmentation sets**, 33,548 images in total:
  - AquaOV255
  - UIIS10K
  - MAS3K
  - Coralscapes
  - SUIM
- How they were prepared:
  - Labels were remapped to the five EURS classes. In UIIS10K/SUIM, "fish" can contain sharks, so it is set to ignore; external class-3 images are drawn 6× more often.
  - Unlabelled pixels are filled with background/coral pseudo-labels from an EURS-only model where its confidence is ≥ 0.9.
  - All sets were de-duplicated by perceptual hash against EURS train and Dev.

## Anything non-standard at inference

- Reads `(A, B)` from the end of the file name (both branches).
- Horizontal-flip TTA on the masks.
- Restored output is rounded to multiples of 4 and sharpened with a fixed unsharp mask.
- All outputs are produced directly at the scoring size: full-resolution input → PIL bilinear resize to longest side 1024, exactly as for our leaderboard zip. The organizers' `resize_for_submission.py` is therefore an identity on them.

## Reproduction notes

- **Image.** Own `Dockerfile`: `pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime`, the PyTorch/CUDA version of our leaderboard runs, plus pinned `transformers==5.17.0` and the versions of our environment. `numpy` is 2.4.6 because the image has Python 3.11.
  - It runs offline (`HF_HUB_OFFLINE=1`) and loads the backbone from `weights/dinov3-vitl16/`.
- **Weights.** One zip behind the link in `weights/weights_link.yaml`, 1.3 GB, unpacked into `weights/`:
  - `dinov3-vitl16/`
  - `e23-s64k-all.pt`
  - `e19g-ref-b64-s48k-all.pt`
  - `generator_affine.csv`
- **Resources.** Fits an 8 GB GPU (also runs on an RTX 2070 8 GB). On an RTX A4000: 1.6 s per image including start-up and full-resolution decoding, about 35 minutes for the 1200 test images.
- **Checked against our leaderboard zip.** Test: our image, `--network none`, read-only root, 12 testing-phase images (two per scene, all three aspect ratios).
  - Restored images: identical on 99.93% of pixels; the rest differ by one or two quantisation steps (4 or 8 grey levels) because of fp16 GPU arithmetic.
  - Masks: identical on 99.99% of pixels.
  - The leaderboard zip was produced on a different GPU. Two runs on the same GPU give byte-identical files.
