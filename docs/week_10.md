<h1>Week Ten: 25 June - 2 July</h1>

## Pre-labeling DKIST with Frontiers' best model
### 1. Concatenating all ImaX/Sunrise rasters
<img src="resources/week_10/imax_concatenated.png">

### 2. Resizing DKIST instance from 4096 x 4096 to 768 x 768
### 3. Getting sharpness of 1
### 4. Applying blurring to DKIST instance to match the sharpness in 3
### 5. Histogram matching (adjustment of brightness/contrast)
<img src="resources/week_10/org_transformed_imax.png">

### 6. Getting the predicted mask for DKIST instance
<img src="resources/week_10/pred.jpg">

### 7. Resizing mask from 6 from 768 x 768 to 4096 x 4096