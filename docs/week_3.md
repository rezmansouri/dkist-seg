<h1>Week Three: 7 May - 14 May</h1>

# 1. New data
## 1.1. <a href="https://www.kaggle.com/datasets/peterwarren/voronoi-artificial-grains-gen/data" target="_blank">ExON Steel Grains</a>
- A Kaggle dataset (not benchmarked)
- 480 Images
- 400 x 400
- 2 classes: Foreground and background
- Ground truth available

<img src="resources/week_3/steel.svg">

## 1.2. <a href="https://datasetninja.com/malaria-segmentation" target="_blank">Malaria</a>
- 3572 Images
- 1382 x 1030
- 9 classes
- Ground truth available

<img src="resources/week_3/malaria Medium.jpeg">

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
<img src="resources/week_3/maps/segnet_mito.svg">
