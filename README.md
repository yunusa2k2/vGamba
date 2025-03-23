# vGamba: Attentive State Space Bottleneck for efficient Long-range Dependencies in Visual Recognition

## Overview
vGamba is a novel model designed to capture long-range dependencies in complex images while improving efficiency compared to existing Vision Transformers (ViTs) and State-Space Models (SSMs). It integrates attention mechanisms with a state-space model using a gated fusion.

## Features
- Efficient long-range dependency modeling
- Optimized for classification, segmentation, and detection tasks
- Scalable with high accuracy-to-computation ratio
- Suitable for real-world applications in AI vision systems

## Experiments
We evaluated vGamba on multiple datasets:
- **ImageNet-1K** for classification
- **ADE20K** for segmentation
- **COCO** for detection
- **AID** dataset for additional validation

### 1. Classification
#### Settings
The ImageNet-1K dataset consists of 1.28M training images and 50K validation images across 1,000 categories. Our training followed ConvNeXt settings, utilizing augmentations like:
- Random cropping & flipping
- Label smoothing
- Mixup and random erasing

We trained vGamba models for 250 epochs using:
- AdamW optimizer with momentum 0.9
- Batch size of 64
- Weight decay of 0.05
- Learning rate starting at 1 × 10−3, scheduled with cosine decay and EMA
- Center cropping to 224² images for testing

Experiments were conducted on **8 NVIDIA Titan XP 12GB GPUs**.

#### Results
Below is the model performance comparison on **ImageNet-1K**:

| Method | Input Size | FLOPs (G) | Params (M) | Throughput (img/s) | Top-1 Accuracy (%) |
|--------|------------|------------|------------|----------------|-----------------|
| ResNet-50 [13] | 224² | 3.9 | 25 | 1226.1 | 76.2 |
| ResNet-101 [13] | 224² | 7.6 | 45 | 753.6 | 77.4 |
| ResNet-152 [13] | 224² | 11.3 | 60 | 526.4 | 78.3 |
| BoT50 [31] | 224² | - | 20.8 | - | 78.3 |
| ViT-B/16 [7] | 384² | 55.4 | 86 | 85.9 | 77.9 |
| ViT-L/16 [7] | 384² | 190.7 | 307 | 27.3 | 76.5 |
| ViM-S [45] | 224² | 5.3 | 26 | 811 | 80.3 |
| ViM-B [45] | 224² | - | 98 | - | 81.9 |
| VMamba-T [19] | 224² | 4.9 | 30 | 1686 | 82.6 |
| vGamba-B | 224² | 3.77 | 18.94 | 1125 | 81.1 |
| vGamba-L | 224² | 6.32 | 31.89 | 746.2 | 82.8 |

vGamba outperforms CNNs, Transformers, and existing SSMs while maintaining computational efficiency. Specifically, **vGamba-L** achieves **82.8% Top-1 accuracy** with only **6.32G FLOPs**, making it significantly more efficient than ViT-B/16 (77.9% with 55.4G FLOPs).

Compared to state-of-the-art SSMs like **VMamba-T (82.6%)**, vGamba demonstrates an optimal balance of accuracy, efficiency, and scalability, making it a **promising architecture for real-world AI applications**.

## Citation
If you use vGamba in your research, please cite our work:
```bibtex
@article{vGamba2025,
  author    = {Your Name},
  title     = {vGamba: Attentive State Space Bottleneck for efficient Long-range Dependencies in Visual Recognition},
  journal   = {Conference Name},
  year      = {2025}
}
```
