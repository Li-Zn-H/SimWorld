# SimWorld
This project contains data sets and code for SimWorld: A Unified Benchmark for Simulator-Conditioned Scene Generation via World Model and is designed to support researchers in related fields. This paper presentsa simulator-conditioned scene generation engine based on world models and providing a benchmark for world model evaluation.

## Project content
### 1. Dataset
The data set provided in this project contains the following:
- **Name**: SimWorld
- **Description**: This dataset includes 4k real mine data and 11.6k simulation images (including groundtruth, segmentation mask, 2D detection label, natural language description).
- **File format**: The groundtruth is jpg image, the segmentation mask is png image, the detection label is yolo format txt text, and the natural language description is txt text.
- **Structure**: The dataset contains the following main files/folders:
project/
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
You can download it from the following Google Drive link: https://drive.google.com/drive/folders/1JdyGvFU4KkpqHht8VVe_fsMOn-UnMj3S?usp=drive_link
