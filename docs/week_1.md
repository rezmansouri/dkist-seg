<h1>Week One: 26 April - 3 May</h1>

## 1. Labeling software for semantic segmentation

### 1.1. Label Studio
Advantages:
- Free
- Browser based (able to install via `pip`)

Disadvantages:
- Interpolates output masks (it is not bitwise): preprocessing required

### 1.2. LabKit (plugin from imageJ / fiji)
Advantages:
- Free
- Available for windows, macOS, linux
- Very good UI for drawing
- Has traditional ML for segmentation, as a starting point

Disadvantages:
- Need to work image by image and export annotations class by class

### 1.3. QuPath
*Software for Bioimage Analysis*

Advantages:
- Free
- Available for windows, macOS, linux
- Good UI

Disadvantages:
- Exporting the annotations is challenging (scripting is needed)

### 1.4. LabelBox

## 2. Datasets from other disciplines

### Medical
1. Synapse multi-organ (CT) -> swin/trans-unet
2. ACDC (MRI) -> swin/trans-unet
3. unet: Web page of the em segmentation challenge, http://brainiac2.mit.edu/ isbi_challenge/

## 3. Multiple Level Tracking (MLT-4) toolkit
