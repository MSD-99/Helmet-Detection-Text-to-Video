# Helmet Detection and Text-to-Video Integration

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://docs.ultralytics.com/)
[![HuggingFace Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An experimental pipeline that combines a fine-tuned **YOLOv11** helmet detector with a **text-to-video diffusion model** and applies the detector frame by frame to generated clips.

---

## 📌 Project Overview

This project consists of two tightly integrated components:
1. **Two-Class Safety Object Detection**: A customized **YOLOv11** model trained to classify heads as `With helmet` or `Without helmet`.
2. **Generative Scenario Synthesis & Video Inference**: Four short synthetic rider clips generated from prompts, followed by automated frame-by-frame inference.

---

## 🎯 Part 1: Helmet & Vehicle Object Detection (YOLOv11)

### 1. Dataset & Annotation Pipeline
- **Base Dataset**: [Kaggle Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection).
- **Split**: 764 total annotated images (611 Training / 153 Validation).
- **Classes**: `With helmet` and `Without helmet`.

#### Sample Annotated Ground-Truth:
![Sample Dataset Annotation](results/dataset_annotation_sample.png)

---

### 2. Model Training & Performance
- **Architecture**: **YOLOv11** (Ultralytics implementation)
- **Input Resolution**: $640 \times 640$
- **Optimizer**: AdamW with Cosine Annealing Learning Rate Schedule
- **Epochs**: 100

#### Quantitative Metrics:
| Metric | Score |
| :--- | :---: |
| **mAP@0.5** | **0.83** |
| **mAP@0.5:0.95** | **0.52** |
| **Precision** | **0.80** |
| **Recall** | **0.80** |

These metrics are the values recorded in the notebook for its 153-image validation split. The split is produced by a seeded image-level shuffle; results should therefore be read as an experiment on this split rather than a cross-dataset benchmark.

#### Training Loss & Metric Curves:
![Training Curves](results/training_curves.png)

#### Class Performance & Confusion Matrix:
| Confusion Matrix (Normalized) | Precision-Recall (PR) Curve |
| :---: | :---: |
| ![Confusion Matrix](results/confusion_matrix_normalized.png) | ![PR Curve](results/box_pr_curve.png) |

#### Validation Batch Predictions:
![Validation Predictions](results/val_batch_predictions.jpg)

---

## 🎬 Part 2: Generative Text-to-Video & Automated Detection Pipeline

### 1. Synthetic Video Generation
Using Hugging Face's `damo-vilab/text-to-video-ms-1.7b` diffusion model, synthetic video clips are generated directly from natural language prompts:
- *"A motorcyclist wearing a helmet riding a motorcycle on a highway"*
- *"A person on a motorcycle with safety helmet driving through the city"*
- *"Motorcyclist with protective helmet riding a sports bike on a road"*

#### Generated Video Frames:
| Scene 1 (Highway) | Scene 2 (City) |
| :---: | :---: |
| ![Video Frame 1](results/text_to_video_sample_1.png) | ![Video Frame 2](results/text_to_video_sample_2.png) |

| Scene 3 (Sports Bike) | Scene 4 (Traffic) |
| :---: | :---: |
| ![Video Frame 3](results/text_to_video_sample_3.png) | ![Video Frame 4](results/text_to_video_sample_4.png) |

---

### 2. Side-by-Side Video Inference Comparison
The detector is applied independently to each generated frame. The recorded clips are qualitative stress tests: one of four clips produced no detections, while the other three produced detections on 75–100% of frames. No end-to-end frames-per-second benchmark was recorded.

#### Scenario 1: Highway Cruiser
![Video Comparison 1](results/video_detection_comparison_1.png)

#### Scenario 2: City Commute
![Video Comparison 2](results/video_detection_comparison_2.png)

#### Scenario 3: Sports Bike
![Video Comparison 3](results/video_detection_comparison_3.png)

#### Scenario 4: Urban Traffic
![Video Comparison 4](results/video_detection_comparison_4.png)

---

## 📁 Repository Structure

```
Helmet_Detection_Text_to_Video/
├── helmet_detection_yolo.ipynb  # Comprehensive Notebook (Data Prep, Training, GenAI & Video Inference)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation & benchmark report
├── LICENSE                      # MIT Open-Source License
├── .gitignore                   # Standard DL/PyTorch ignore rules
└── results/                     # Quantitative metric plots & qualitative comparison grids
    ├── dataset_annotation_sample.png
    ├── training_curves.png
    ├── confusion_matrix_normalized.png
    ├── box_pr_curve.png
    ├── val_batch_predictions.jpg
    ├── text_to_video_sample_1.png ... 4.png
    └── video_detection_comparison_1.png ... 5.png
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Running the Complete Pipeline
Open the notebook in Jupyter Notebook or Google Colab:
```bash
jupyter notebook helmet_detection_yolo.ipynb
```
Follow the step-by-step cells to:
1. Load dataset & inspect annotations.
2. Fine-tune / evaluate the YOLOv11 model.
3. Synthesize videos via Hugging Face diffusion models.
4. Run automated video inference and export comparison grids.

The dataset, trained weights, and generated MP4 files are not stored in this repository. Update the notebook paths before running it locally. Generating the clips requires substantial GPU memory and downloads the diffusion model from Hugging Face.
