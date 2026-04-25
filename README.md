# Final Reproduction Bundle

This bundle contains only the final files that need to be added on top of the upstream project in order to reproduce our setup. It also includes the final local HTML viewer shell inside `interpretability_report/`.

## Bundle Contents

- `requirements.txt`
  - Replacement for `./proto_non_param/requirements.txt`
- `modified/train.py`
  - Replacement for `./proto_non_param/train.py`
- `modified/backbone.py`
  - Replacement for `./proto_non_param/modeling/backbone.py`
- `modified/evaluate.py`
  - Replacement for `./proto_non_param/evaluate.py`
- `modified/eval/utils.py`
  - Replacement for `./proto_non_param/eval/utils.py`
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

All commands below assume `.` is the root of this repository.

## 1. Bootstrap This Repository Into the Current Directory

```bash
git clone https://github.com/shreyasrox2326/cv-project-git.git temp
mv temp/* temp/.git .
rm -rf temp
```

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
  build_cub_bbox_crops.py
  check_cub_copy_integrity.py
  generate_interpretability_report.py
  interpretability_report/
  modified/
  package_report.py
  proto_non_param/
  requirements.txt
```

## 3. Clone and Install DINOv2

The upstream repository depends on a separate local clone of DINOv2.

```bash
git clone https://github.com/facebookresearch/dinov2.git dinov2
cd dinov2
git checkout e1277af2ba9496fbadf7aec6eba56e8d882d1e35
cd ..
```

Now the structure should be:

```text
./
  build_cub_bbox_crops.py
  check_cub_copy_integrity.py
  dinov2/
  generate_interpretability_report.py
  interpretability_report/
  modified/
  package_report.py
  proto_non_param/
  requirements.txt
```

## 4. Create the Python Environment

Create and activate your Python environment first. Do not install the upstream `requirements.txt` yet, because this bundle replaces it with the corrected working version in the next step.

After activation, install the local DINOv2 clone from the project root:

```bash
cd dinov2
pip install --no-deps -e .
cd ..
```

## 5. Apply the Modified Project Files

Replace the upstream files with the modified files from this bundle.

```bash
cp requirements.txt ./proto_non_param/requirements.txt
cp modified/train.py ./proto_non_param/train.py
cp modified/backbone.py ./proto_non_param/modeling/backbone.py
cp modified/evaluate.py ./proto_non_param/evaluate.py
cp modified/eval/utils.py ./proto_non_param/eval/utils.py
```

On Windows PowerShell, the equivalent is:

```powershell
Copy-Item ./requirements.txt ./proto_non_param/requirements.txt -Force
Copy-Item ./modified/train.py ./proto_non_param/train.py -Force
Copy-Item ./modified/backbone.py ./proto_non_param/modeling/backbone.py -Force
Copy-Item ./modified/evaluate.py ./proto_non_param/evaluate.py -Force
Copy-Item ./modified/eval/utils.py ./proto_non_param/eval/utils.py -Force
```

## 6. Install the Project Requirements

Now install the patched `requirements.txt` from inside `proto_non_param`:

```bash
cd proto_non_param
pip install -r requirements.txt
cd ..
```

## 7. Download and Extract the Dataset

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
  build_cub_bbox_crops.py
  check_cub_copy_integrity.py
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
  generate_interpretability_report.py
  interpretability_report/
  modified/
  package_report.py
  proto_non_param/
  requirements.txt
```

The original CUB metadata remains in `CUB_200_2011/`. The cropped images used by training will be generated next.

## 8. Build the Cropped Dataset

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

## 9. Train the Model

Run training from inside `proto_non_param`:

```bash
cd proto_non_param
python train.py --data-root ../dataset-root --log-dir ../log-dir-paper-run --epochs 6 --fine-tuning-start-epoch 1 --num-splits 1 --num-prototypes 5 --gamma 0.99 --backbone-lr 1e-4 --classifier-lr 1e-6
cd ..
```

This is the exact training setup used for our final reproduction run.

## 10. Evaluate the Checkpoint

Run evaluation from inside `proto_non_param`:

```bash
cd proto_non_param
python evaluate.py --ckpt-path ../log-dir-paper-run/ckpt.pth
cd ..
```

## 11. Generate Prototype / Interpretability Assets

Run the final interpretability export from the project root:

```bash
PYTHONPATH=./proto_non_param python generate_interpretability_report.py --ckpt-path ./log-dir-paper-run/ckpt.pth --dataset-root ./dataset-root --report-root ./interpretability_report --split test --topk 5 --vote-percentile 75 --min-vote-items 5 --batch-size 64 --num-workers 8 --prediction-examples 12 --prediction-visual-prototypes 20 --clear
```

This generates:

- prototype atlas assets
- prediction explorer assets
- prototype activation examples
- Gaussian part-vote labels and vote breakdown metadata

## 12. Package the Report for Sharing and Portability

```bash
python package_report.py --report-root ./interpretability_report --zip-path ./interpretability_report.zip
```

This produces:

```text
./interpretability_report.zip
```

The generated `data/`, `examples/`, and `prototypes/` folders are written directly into `./interpretability_report/`, alongside `index.html`, `atlas.html`, and `assets/`.

## Why These `proto_non_param` Files Were Replaced

- `requirements.txt`
  - fixes the `albumentations` version pin to the working form `albumentations==1.4.14`
  - uses the pip-installable OpenCV package name `opencv-python`
- `train.py`
  - fixes best-checkpoint tracking so the best epoch and best accuracy are updated correctly
- `backbone.py`
  - skips the expensive default DINOv2 weight initialization before loading the expanded-state checkpoint
- `evaluate.py`
  - exports the actual dataset root into the evaluation helpers before importing them
- `eval/utils.py`
  - replaces the hardcoded `datasets/CUB_200_2011` metadata path with an environment-configurable path

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
