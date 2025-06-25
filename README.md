# SimWorld
This project contains data sets and code for SimWorld: A Unified Benchmark for Simulator-Conditioned Scene Generation via World Model and is designed to support researchers in related fields. This paper presents a simulator-conditioned scene generation engine based on world models and providing a benchmark for world model evaluation.

## Project content
### 1. Dataset
The data set provided in this project contains the following:
- **Name**: SimWorld
- **Description**: This dataset includes 4k real mine data and 11.6k simulation images (including groundtruth, segmentation mask, 2D detection label, natural language description).
- **File format**: The groundtruth is jpg image, the segmentation mask is png image, the detection label is yolo format txt text, and the natural language description is txt text.
- **Structure**: The dataset contains the following main files/folders:
```
dataset/
│
├── AutoMine/                    # real data
│   ├── target/                  # groundtruth
│   ├── source/                  # segmentation mask
│   ├── bounding_boxes/          # 2D detection label
│   └── prompts/                 # natural language description
├── PMScenes/                    # synthetic data
│   ├── target/                  # groundtruth
│   ├── source/                  # segmentation mask
│   ├── bounding_boxes/          # 2D detection label
│   └── prompts/                 # natural language description
```
You can download it from the following Google Drive link: https://drive.google.com/drive/folders/1JdyGvFU4KkpqHht8VVe_fsMOn-UnMj3S?usp=drive_link


### 2. Train
This repository contains two multi-card distributed training scripts, applicable to SimWorld XL and SimWorld fine-tuning, respectively.

**Environment Dependencies** 
It is recommended to install the dependency packages in ```requirements_train.txt``` under a Python 3.12.2 environment.

**Installation**
```bash
conda create -n diffusion python=3.12.2
conda activate diffusion

pip install -r requirements_train.txt
``` 
**Data Preparation**
Please prepare the training data in the format of a ```dataset.json``` file. Each line should represent a single sample and contain the following fields:
```json
{
  "source": "path/to/conditional_image.png",
  "target": "path/to/target_image.png",
  "prompts": "text description",
  "labels": [
    ["class", "x", "y", "w", "h"], ["..."]
  ]
}
```
- source: The conditional control image (e.g., segmentation map, depth map, etc.).
- target: The target image.
- prompts: The text prompt.
- labels: Object detection bounding boxes (in YOLO format, float type, with normalized center coordinates).

**Train**
Before training, please download the corresponding versions of the Stable Diffusion (SD) and ControlNet base models. After preparing these models, you can run the training script.

**Training Script 1**: train_SimWorld.py
```bash
python train_SimWorld.py \
  --sd_path "/path/to/stable-diffusion-v1-5" \
  --cn_path "/path/to/control_v11p_sd15_seg" \
  --dataset_json "/path/to/dataset.json" \
  --pretrained_controlnet "/path/to/pretrained.model" \
  --cuda_devices "0,1,2,3" \
```


**Training Script 2**: train_SimWorldXL.py
```bash
python train_SimWorldXL.py \
  --sd_path "/path/to/stable-diffusion-xl-base-1.0" \
  --cn_path "/path/to/controlnet-depth-sdxl-1.0" \
  --vae_path "/path/to/sdxl-vae-fp16-fix" \
  --dataset_json "/path/to/dataset.json" \
  --pretrained_controlnet "/path/to/pretrained_epoch.model" \
  --cuda_devices "0,1,2,3" \
```
### 3.Inference
This script performs inference on a given conditional input image and a text prompt, using the pre-trained SimWorld models.

**Installation**
It is recommended to install the dependency packages in requirements_inference.txt under a Python 3.12.2 environment.
```bash
pip install -r requirements_train.txt
```

**Run Inference**
1. Prepare the following:
   - **Stable Diffusion model directory** (`--sd_path`), containing tokenizer, text_encoder, unet, scheduler, vae subfolders.
   - **ControlNet model directory** (`--cn_path`).
   - **ControlNet trained weights** checkpoint (`--controlnet_ckpt`).
   - **Conditional input image** (e.g., segmentation mask).
   - **Text prompt file** (plain text).
2. Run the inference script with the appropriate arguments:
```bash
python inference_SimWorld.py \
  --sd_path /path/to/stable-diffusion-v1-5 \
  --cn_path /path/to/controlnet-depth-sdxl-1.0 \
  --controlnet_ckpt /path/to/pretrained_epoch.model \
  --source_image /path/to/conditional_image.png \
  --target_prompt /path/to/prompt.txt \
  --cuda_device 0 \
  --time_steps 50 \
  --output_image output.png
```
or
```bash
python inference_SimWorldXL.py \
  --sd_path /path/to/stable-diffusion-xl-base-1.0 \
  --controlnet_path /path/to/controlnet-depth-sdxl-1.0 \
  --vae_path /path/to/sdxl-vae-fp16-fix \
  --controlnet_ckpt /path/to/pretrained_epoch.model \
  --input_image /path/to/conditional_image.png \
  --prompt_file /path/to/prompt.txt \
  --cuda_device 3 \
  --output_image ./outputs.png
```
### 4.Cite
If this project is helpful for your work, please cite the following paper:
```bibtex
@article{li2025simworld,
  title={Simworld: A unified benchmark for simulator-conditioned scene generation via world model},
  author={Li, Xinqing and Song, Ruiqi and Xie, Qingyu and Wu, Ye and Zeng, Nanxin and Ai, Yunfeng},
  journal={arXiv preprint arXiv:2503.13952},
  year={2025}
}
```