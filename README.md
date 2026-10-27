# deepdrivepl — EURS @ MaCVi WACV 2027, Track 1 reproduction

Testing-phase submission **#19263** (`test-e19g-s64k`, composite 0.655), team `deepdrivepl`.
Made from the organizers' [reproduction template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template).

```bash
python repro_check.py . --download-weights                     # fetch + unpack the weights zip into weights/
docker build -t deepdrivepl-t1 .                                # our image: PyTorch 2.9.1 / CUDA 12.6
docker run --rm --gpus all --network none -v "$PWD/weights":/pkg/weights:ro -v /path/to/test:/in:ro -v /tmp/out:/out \
    deepdrivepl-t1 python /pkg/repro_check.py /pkg --run --input /in --output /out
```

- `run.py` — `main(model_dir, input_dir, output_dir, device)`
- `src/eurs_t1.py` — affine table + U-Net refiner + unsharp (restoration); DINOv3 ViT-L + FiLM decoder (masks)
- `weights/weights_link.yaml` — public link + SHA-256 of the weights zip (DINOv3 licence text inside)
- `METHOD.md` — method, data, external resources, reproduction check
