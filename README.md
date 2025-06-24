# SimWorld
This project contains data sets and code for SimWorld: A Unified Benchmark for Simulator-Conditioned Scene Generation via World Model and is designed to support researchers in related fields. This paper presentsa simulator-conditioned scene generation engine based on world models and providing a benchmark for world model evaluation.

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


### 2. Code
This repository contains two multi-card distributed training scripts, applicable to SimWorld XL and SimWorld fine-tuning, respectively.

**Environment Dependencies** 
It is recommended to install the dependency packages in ```requirements.txt``` under a Python 3.12.2 environment.
**Installation**
```bash
conda create -n diffusion python=3.12.2
conda activate diffusion

pip install -r requirements.txt
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

**Training Parameters** 
The parameters for both scripts are similar, with some being optional.
|Parameter Name  | Description   | Required   |
|---------|---------|---------|
| --sd_path   | Path to the main Stable Diffusion model.   | Yes   |
| --cn_path   | Path to the ControlNet model.   | Yes   |
| --vae_path   | Path to the VAE model   | Yes (for XL)   |
| --dataset_json   | Path to the training data JSON file.   | Yes   |
| --pretrained_controlnet   | Pre-trained weights for ControlNet (optional).   | No   |
| --save_directory   | Directory to save the models.   | No   |
| --log_directory   | Directory for logs and TensorBoard.   | No   |
| --cuda_devices   | GPU IDs to use (e.g., "0,1,2").   | No (default: 0)   |
| --port   | Port number for distributed communication.   | No   |

You need to first download the corresponding versions of the SD (Stable Diffusion) and ControlNet base models. And then run the Training Script.
**Training Script 1**: train_SimWorldXL.py
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
**cite**
If this project is helpful for your work, please cite the following paper:
```bibtex
@article{li2025simworld,
  title={Simworld: A unified benchmark for simulator-conditioned scene generation via world model},
  author={Li, Xinqing and Song, Ruiqi and Xie, Qingyu and Wu, Ye and Zeng, Nanxin and Ai, Yunfeng},
  journal={arXiv preprint arXiv:2503.13952},
  year={2025}
}
```