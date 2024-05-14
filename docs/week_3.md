<h1>Week Three: 7 May - 14 May</h1>

# 1. New data
- https://www.kaggle.com/datasets/peterwarren/voronoi-artificial-grains-gen/data

- https://datasetninja.com/annotated-quantitative-phase-microscopy-cell-dataset

- https://datasetninja.com/malaria-segmentation

- https://datasetninja.com/ccagt

- paper: https://www.sciencedirect.com/science/article/pii/S104458032100108X data: https://data.mendeley.com/datasets/t4wvpy29fz/2

# 2. Experiments
## 2.1. ImaX/Sunrise
### 2.1.1. SegNet
#### 2.1.1.1. Focal loss function
<img src="resources/week_3/plots/segnet_imax_focal.svg">
<img src="resources/week_3/maps/2.svg">

#### 2.1.1.2. Cross entropy loss function
<img src="resources/week_3/plots/segnet_imax_ce.svg">
<img src="resources/week_3/maps/segnet_imax_ce.svg">

### 2.1.2. UNet
#### 2.1.2.1. Focal loss function
<img src="resources/week_3/plots/unet_imax_focal.svg">
<img src="resources/week_3/maps/unet_imax_focal.svg">

#### 2.1.2.2. mIoU loss function
<img src="resources/week_3/plots/unet_imax_iou.svg">
<img src="resources/week_3/maps/unet_imax_iou.svg">

## 2.2. Steel grains
### SegNet - Cross entropy loss
<img src="resources/week_3/plots/segnet_metal.svg">
<img src="resources/week_3/maps/segnet_steel.svg">

## 2.3. MitoLab
### SegNet - Cross entropy loss
<img src="resources/week_3/plots/segnet_mito.svg">
