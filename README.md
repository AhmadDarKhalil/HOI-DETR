<h1 align="center">
  Improving and Evaluating Hand-Object Interaction Detection
</h1>

<p align="center">
  <a href="https://ahmaddarkhalil.github.io/HOI-DETR/"><img src="https://img.shields.io/badge/Project_Page-green?logo=github&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.17384"><img src="https://img.shields.io/badge/arXiv-2606.17384-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/spaces/ahmaddarkhalil/hoi-detr-demo"><img src="https://img.shields.io/badge/🤗_Demo-Hugging_Face-yellow" alt="Demo"></a>
</p>

<p align="center">
  <a href="https://ahmaddarkhalil.github.io/">Ahmad Darkhalil</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://dimadamen.github.io/">Dima Damen</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://cs.nyu.edu/~fouhey/">David Fouhey</a><sup>2</sup>
</p>

<p align="center">
  <b><sup>1</sup>University of Bristol</b>&nbsp;&nbsp;&nbsp;<b><sup>2</sup>New York University</b>
</p>

HOI-DETR is a transformer-based framework for detecting hands, hand-held objects, and their interactions in images and video. Built on the [Co-DETR](https://github.com/Sense-X/Co-DETR) architecture, it adds a lightweight interaction module that jointly predicts all visible hands, 1st objects (objects in direct contact with a hand), and 2nd objects (objects acted upon through a tool), along with their pairwise interaction links — all in a single forward pass.

![HOI-DETR method overview](assets/method.png)

---

## Highlights

- **+20 mAP₅₀** improvement over Hands23 on both Hands23 and FineBio benchmarks
- **Interaction module** predicting hand → 1st object and 1st object → 2nd object relations
- **Refined Hands23 annotations** correcting duplicate bounding boxes across 26k images
- **Strong zero-shot generalisation** to unseen datasets and domains
---

## Installation

This codebase is adopted from [Co-DETR](https://github.com/Sense-X/Co-DETR), built on MMDetection V2.25.3 and MMCV V1.5.0. The source code of MMDetection is included in this repo. We have tested on two systems:

| System | Python | PyTorch | CUDA | GPU |
|--------|--------|---------|------|-----|
| x86 (standard) | 3.7 | 1.11.0+cu113 | 12.2 (driver) | RTX 4090 |
| aarch64 HPC | 3.10 | 2.4.1 (cu120) | 12.0 | GH200 Hopper |

Follow the **x86 setup** below if you are on a standard x86 machine. If you are on an ARM64 HPC cluster or your GPU does not support PyTorch 1.11 (SM 9.0 / Hopper and newer), see [INSTALL_HOPPER.md](INSTALL_HOPPER.md).
### x86 — Standard NVIDIA GPU

<details>
<summary>Tested on Ubuntu 24.04, RTX 4090, CUDA Driver 12.2, Python 3.7</summary>

**1. Create conda environment**

```bash
conda create -n codetr python=3.7 -y
conda activate codetr
```

**2. Install PyTorch**

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

> CUDA 11.3 wheels work with CUDA 12.x drivers due to NVIDIA's backward compatibility guarantee.

**3. Install mmcv**

```bash
pip install mmcv-full==1.5.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
```

**4. Clone and install Co-DETR**

```bash
git clone https://github.com/AhmadDarKhalil/HOI-DETR.git
cd Co-DETR
pip install -e .
```

**5. Install remaining dependencies**

```bash
pip install timm==0.6.13 fairscale==0.4.6 scipy==1.7.3 yapf==0.40.1 \
  opencv-python numpy==1.21.6 pycocotools
```

**6. Verify**

```bash
python -c "import torch, mmcv; print(torch.__version__, mmcv.__version__)"
# Expected: 1.11.0+cu113  1.5.0
```

</details>

For the ARM64 / Hopper setup, see [INSTALL_HOPPER.md](INSTALL_HOPPER.md).

---

## Demo

### 1. Download checkpoint

```bash
mkdir -p checkpoints
wget -O checkpoints/epoch_5.pth \
  https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/main/epoch_5.pth
```

### 2. Run

Edit the paths and settings at the top of `demo/demo.py`:

```python
MODEL_CONFIG = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py'
CHECKPOINT   = 'checkpoints/epoch_5.pth'
DEVICE       = 'cuda:0'
INPUT_DIR    = 'demo/example_images'   # directory of input images
OUTPUT_DIR   = None                    # None -> demo/results/<input_dir_name>/
SCORE_THR    = 0.3
VERBOSE_LABELS = False
```

Then run:

```bash
export PYTHONPATH=".:$PYTHONPATH"
python demo/demo.py
```

Results are saved to `demo/results/<input_dir_name>/` by default, preserving original filenames.

---

## Datasets

Evaluation uses the refined version of [Hands23](https://github.com/ddshan/hands23_data). Download the images and splits from the Hands23 repo, then add our [corrected annotation file](https://www.dropbox.com/scl/fo/5k2lc9krgtf5mzr7m6og2/AOaOSXRRrI9GyJwfO76QiIA?rlkey=n0y7wrybvey6ybvnfll7aqksl&st=yy2nb7zk&dl=0).

Expected directory structure:

```
hands23_data/
├── annotations/
│   └── val_h_first_second_full_corrected_w_area.json   ← provided in this repo
├── allMergedSplit/
│   ├── TEST.txt
│   ├── TRAIN.txt
│   └── VAL.txt
├── allMergedBlur/
│   └── *.jpg
└── allMergedTxt/
    └── *.jpg.txt
```

Images and splits are unchanged from Hands23 and can be downloaded directly from their repository. The annotation file `val_h_first_second_full_corrected_w_area.json` is provided under `annotations/` in this repo.

---

## Evaluation

### 1. Set dataset path

Open the config file:

```
projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py
```

Set `data_root` to your local Hands23 path and confirm the annotation and image paths match your directory structure:

```python
data_root = '/path/to/hands23_data/'

data = dict(
    ...
    val=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/val_h_first_second_full_corrected_w_area.json',
        img_prefix=data_root + 'allMergedBlur/'),
    test=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/val_h_first_second_full_corrected_w_area.json',
        img_prefix=data_root + 'allMergedBlur/')
)
```

### 2. Run evaluation

```bash
bash -c 'export PYTHONPATH=".:$PYTHONPATH" && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  bash ./tools/dist_test.sh \
    projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py \
    checkpoints/epoch_5.pth \
    4 \
    --eval bbox'
```

---

## Results

**Hands23 val (refined annotations)**

| Method | Overall AP₅₀ | Hand AP₅₀ | 1st obj AP₅₀ | 2nd obj AP₅₀ | F1 inter |
|--------|-------------|-----------|-------------|-------------|----------|
| [Hands23](https://github.com/EvaCheng-cty/hands23_detector) | 63.6 | 85.2 | 59.4 | 46.2 | 90.7 |
| HOI-DETR (ours) | **86.1** | **93.1** | **86.5** | **78.7** | **95.5** |

**HD-EPIC-HOI (zero-shot video, 1st obj)**

| Method | Frame-AP | Video-AP | LTC |
|--------|----------|----------|-----|
| [Hands23](https://github.com/EvaCheng-cty/hands23_detector) | 46.9 | 26.8 | 31.4 |
| [HOIST](https://github.com/SupreethN/HOISTFormer) | 30.4 | 16.1 | 27.2 |
| HOI-DETR (ours) | **72.6** | **60.2** | **61.0** |

**Zero-shot cross-dataset (1st obj AP₅₀)**

| Method | [HOIST](https://github.com/SupreethN/HOISTFormer) | [FineBio](https://github.com/aistairc/FineBio) |
|--------|-------|-------------------|
| [Hands23](https://github.com/EvaCheng-cty/hands23_detector) | 43.1 | 26.0 |
| [HOIST](https://github.com/SupreethN/HOISTFormer) | 70.7 | — |
| HOI-DETR (ours) | **76.6** | **55.8** |

Evaluation datasets: [Hands23](https://github.com/ddshan/hands23_data) · [HOIST](https://supreethn.github.io/research/hoistformer/index.html) · [FineBio](https://github.com/aistairc/FineBio) · [HD-EPIC](https://hd-epic.github.io/)

---

## Citation

If you use HOI-DETR in your research, please cite:

```bibtex
@article{darkhalil2026improving,
  title={Improving and Evaluating Hand-Object Interaction Detection},
  author={Darkhalil, Ahmad and Damen, Dima and Fouhey, David},
  journal={arXiv preprint arXiv:2606.17384},
  year={2026}
}
```
---

## Acknowledgements

This work was supported by EPSRC Program Grant Visual AI (EP/T028572/1). A. Darkhalil was supported by the EPSRC Doctoral Training Program (DTP). We acknowledge the usage of GPU node hours granted as part of the AIRR Innovator project "5D Hand-Object Interaction Modelling from In-the-wild Videos" (Mar 2026 – Sep 2026), the AIRR Gateway project "HOI Foundational Model from Egocentric Data" (Dec 2025 – Mar 2026), and the Sovereign AI Unit call project "Gen Model in Ego-sensed World" (Aug – Nov 2025). D. Fouhey was supported by the National Science Foundation under Grant No. 2006619 and 2437330.

We thank **Sidhartha Reddy Potu** for his contributions during the early stages of this project. This work builds on [Co-DETR](https://github.com/Sense-X/Co-DETR) and [MMDetection](https://github.com/open-mmlab/mmdetection), and we gratefully acknowledge their authors for open-sourcing the codebase and architecture. We also thank the authors of [Hands23](https://github.com/ddshan/hands23_data), [HOIST](https://supreethn.github.io/research/hoistformer/index.html), [FineBio](https://github.com/aistairc/FineBio), and [HD-EPIC](https://hd-epic.github.io/) for making their datasets publicly available.
