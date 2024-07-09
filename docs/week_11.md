<h1>Week Eleven: 2 July - 9 July</h1>

## First paper outline

### 1. Introduction:
- What are granules?
- Significance of identifying/classifying them
- Semantic segmentation approaches
- Introduce DKIST

### 2. Related work:
- Survey semantic segmentation on solar granules or other similar research
    - Datasets
    - Architectures
    - Loss functions

### 3. Methodologies:
- Architectures:
    - UNet
    - UNet++
    - AttUNet
    - SegNet
- Frameworks:
    - BT-UNet (pre-training)
- Loss functions:
    - Focal
    - mIoU
    - Lovasz Softmax

- Dataset preprocessing/augmentation approaches

### 4. Results and Discussion

From section 3:
- About 16 to 20 experiments on combinations, depending on acceptable results

- For each architecture, starting with the simplest one (lowest #params) until no improvement is observed. That would be the final model of that category.

- Comparative results

## Labeling DKIST

Going to start labeling DKIST

- Frontiers' classes + 1
    - Intergranular lane
    - Complex shaped granules
    - Uniform shaped granules
    - Granules with a dot
    - Granules with a lane
    - **Bright points**
- Most diverse frames
- Frames with no prepheral blur

## Email to Kevin

