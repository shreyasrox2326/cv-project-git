# Final Reproduction Bundle

This bundle contains only the final files that need to be added on top of the upstream project in order to reproduce our setup. It also includes the final local HTML viewer shell inside `interpretability_report/`.

## Bundle Contents

- `modified/train.py`
  - Replacement for `./proto_non_param/train.py`
- `modified/backbone.py`
  - Replacement for `./proto_non_param/modeling/backbone.py`
- `build_cub_bbox_crops.py`
  - Final bounding-box cropping script used to create `cub200_cropped/`
- `check_cub_copy_integrity.py`
  - Dataset verification script for the cropped split structure
- `generate_interpretability_report.py`
  - Final interpretability export script used to generate prototype atlas / explorer data
- `package_report.py`
  - Helper script to zip the generated interpretability report
- `interpretability_report/`
  - The HTML viewer shell: `index.html`, `atlas.html`, and `assets/`

## Expected Working Directory

All commands below assume you are in a project root directory of your choice. Let that directory be `.`.

## 1. Bootstrap This Repository Into the Current Directory

If you want this repository's contents to become the contents of your current working directory rather than a nested folder, run:

```bash
gh repo clone shreyasrox2326/cv-project-git temp
mv temp/* temp/.git .
rm -rf temp
```

After that, continue the rest of the setup from the same project root `.`.

## 2. Clone the Upstream Repository

```bash
git clone https://github.com/zijizhu/proto-non-param.git proto_non_param
cd proto_non_param
git checkout c41f3443b3dee7670d5272a261a591a859481f1f
cd ..
```

At this point, your directory should look like:

```text
./
  dinov2/
  proto_non_param/
```

## 3. Clone and Install DINOv2

The upstream repository depends on a separate local clone of DINOv2.

```bash
git clone https://github.com/facebookresearch/dinov2.git dinov2
cd dinov2
git checkout e1277af2ba9496fbadf7aec6eba56e8d882d1e35
pip install --no-deps -e .
cd ..
```

Now the structure should be:

```text
./
  dinov2/
  proto_non_param/
```

## 4. Create and Install the Python Environment

Install the upstream requirements from `proto_non_param`.

```bash
cd proto_non_param
pip install -r requirements.txt
cd ..
```

## 5. Apply the Two Modified Source Files

Replace the upstream files with the two modified files from this bundle.

```bash
cp modified/train.py ./proto_non_param/train.py
cp modified/backbone.py ./proto_non_param/modeling/backbone.py
```

On Windows PowerShell, the equivalent is:

```powershell
Copy-Item ./modified/train.py ./proto_non_param/train.py -Force
Copy-Item ./modified/backbone.py ./proto_non_param/modeling/backbone.py -Force
```

## 6. Download and Extract the Dataset

From the project root, run:

```bash
mkdir -p dataset-root
cd dataset-root
wget -O CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
wget -O segmentations.tgz "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
tar -xzf CUB_200_2011.tgz
tar -xzf segmentations.tgz
cd ..
```

After extraction, the workspace should look like:

```text
./
  dinov2/
  dataset-root/
    CUB_200_2011/
      attributes/
      images/
      parts/
      README
      bounding_boxes.txt
      classes.txt
      image_class_labels.txt
      images.txt
      train_test_split.txt
    segmentations/
  proto_non_param/
```

The original CUB metadata remains in `CUB_200_2011/`. The cropped images used by training will be generated next.

## 7. Build the Cropped Dataset

Run:

```bash
python build_cub_bbox_crops.py --dataset-root ./dataset-root --clear
```

This creates:

```text
./dataset-root/cub200_cropped/test_cropped/
./dataset-root/cub200_cropped/train_cropped_augmented/
```

To verify the cropped split layout:

```bash
python check_cub_copy_integrity.py --dataset-root ./dataset-root
```

## 8. Train the Model

Run training from inside `proto_non_param`:

```bash
cd proto_non_param
python train.py --data-root ../dataset-root --log-dir ../log-dir-paper-run --epochs 6 --fine-tuning-start-epoch 1 --num-splits 1 --num-prototypes 5 --gamma 0.99 --backbone-lr 1e-4 --classifier-lr 1e-6
cd ..
```

This is the exact training setup used for our final reproduction run.

## 9. Evaluate the Checkpoint

Run evaluation from inside `proto_non_param`:

```bash
cd proto_non_param
python evaluate.py --ckpt-path ../log-dir-paper-run/ckpt.pth
cd ..
```

## 10. Generate Prototype / Interpretability Assets

Run the final interpretability export from the project root:

```bash
PYTHONPATH=./proto_non_param python generate_interpretability_report.py --ckpt-path ./log-dir-paper-run/ckpt.pth --dataset-root ./dataset-root --report-root ./interpretability_report --split test --topk 5 --vote-percentile 75 --min-vote-items 5 --batch-size 64 --num-workers 8 --prediction-examples 12 --prediction-visual-prototypes 20 --clear
```

This generates:

- prototype atlas assets
- prediction explorer assets
- prototype activation examples
- Gaussian part-vote labels and vote breakdown metadata

## 11. Package the Report for Download

```bash
python package_report.py --report-root ./interpretability_report --zip-path ./interpretability_report.zip
```

This produces:

```text
./interpretability_report.zip
```

The generated `data/`, `examples/`, and `prototypes/` folders are written directly into `./interpretability_report/`, alongside `index.html`, `atlas.html`, and `assets/`.

## Why These Two `proto_non_param` Files Were Replaced

- `train.py`
  - fixes best-checkpoint tracking so the best epoch and best accuracy are updated correctly
- `backbone.py`
  - skips the expensive default DINOv2 weight initialization before loading the expanded-state checkpoint

## Final Directory Layout After Full Setup

```text
./
  dinov2/
  dataset-root/
    CUB_200_2011/
    cub200_cropped/
    segmentations/
  build_cub_bbox_crops.py
  check_cub_copy_integrity.py
  generate_interpretability_report.py
  interpretability_report/
  log-dir-paper-run/
  package_report.py
  proto_non_param/
```

## Attribution

Please cite the original paper and upstream repository:

- Zhu et al., Interpretable Image Classification via Non-parametric Part Prototype Learning, CVPR 2025
- Upstream repository: https://github.com/zijizhu/proto-non-param
