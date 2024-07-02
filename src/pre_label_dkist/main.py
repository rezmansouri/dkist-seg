import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from model import UNet
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage import exposure, filters


UNET_PATH = 'C:/Users/Reza/Desktop/dmlab/dkist/SegGranules_Unet_model/model_params/unet_epoch_12_0.52334_IoU_non_Dropout.pt'
IMAX_PATH = 'C:/Users/Reza/Desktop/dmlab/dkist/SegGranules_Unet_model/data/Masks_S_v5/Train'
DKIST_PATH = 'F:/dkist_batch_1/vbi/vbi/gband_destretched'
OUT_PATH = 'C:/Users/Reza/Desktop/dmlab/dkist/dkist-seg/src/pre_label_dkist/output'
IMG_PATH = 'C:/Users/Reza/Desktop/dmlab/dkist/dkist-seg/src/pre_label_dkist/images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
colors = [
    [142, 1, 82],
    [231, 150, 196],
    [247, 247, 247],
    [155, 206, 99],
    [39, 100, 25]
]


def _768_mask_np_to_4096_mask_png(mask_np):
    mask_png = np.zeros((768, 768, 3), dtype=np.uint8)
    for i in range(5):
        mask_png[mask_np == i, :] = colors[i]
    mask_png = Image.fromarray(mask_png)
    mask_png = mask_png.resize((4096, 4096))
    return mask_png


def predict(model, x):
    dx = []
    for i in range(6):
        for j in range(6):
            dx.append(x[i*128:(i+1)*128, j*128:(j+1)*128])
    dx = np.array(dx)
    dx = np.expand_dims(dx, axis=1)
    xx = torch.Tensor(dx)
    model.eval()

    with torch.no_grad():
        pred_mask = model(xx.to(device))
    pred_mask_class = torch.argmax(pred_mask, axis=1)
    pred_mask_class_np = pred_mask_class.cpu().detach().numpy()
    yhat = np.zeros((768, 768), dtype=np.int32)

    for i in range(6):
        for j in range(6):
            yhat[i*128:(i+1)*128, j*128:(j+1)*128] = pred_mask_class_np[i*6+j]

    return yhat


def save_image(dkist, path):
    x = dkist
    x = x / x.max() * 255

    img = Image.fromarray(x.astype(np.uint8))
    img.save(path)


def transform(dkist, imax):
    x, reference = dkist, imax
    x = x / x.max() * 255

    img = Image.fromarray(x.astype(np.uint8))
    img = img.resize((768, 768))
    x = np.array(img, dtype=np.float32)

    gy, gx = np.gradient(reference)
    gy, gx = float(np.std(gy)), float(np.std(gx))

    reference_sharpness = np.var(filters.laplace(reference))
    target_sharpness = np.var(filters.laplace(x))
    sigma = np.sqrt(target_sharpness / reference_sharpness)

    matched = filters.gaussian(x, sigma=sigma)
    matched = exposure.match_histograms(matched, reference)

    matched /= np.max(matched)

    return matched


def main():
    model = UNet(1, 5, 1, False, False)
    state = torch.load(UNET_PATH, map_location=device)
    model.load_state_dict(state)

    refs = np.zeros((768, 7*768), np.float32)
    for i, imax in enumerate(os.listdir(IMAX_PATH)):
        ref = np.load(os.path.join(IMAX_PATH, imax))
        ref = ref['smap'].astype(np.float32)
        refs[:, i*768:(i+1)*768] = ref
    reference = refs

    for dkist in tqdm(os.listdir(DKIST_PATH)):
        hdul = fits.open(os.path.join(DKIST_PATH, dkist))
        x = hdul[0].data.astype(np.float32)
        save_image(x, os.path.join(IMG_PATH, dkist[:-5]) + '.png')
        x = transform(x, reference)
        _768_mask_np = predict(model, x)
        _4096_mask_png = _768_mask_np_to_4096_mask_png(_768_mask_np)
        _4096_mask_png.save(os.path.join(OUT_PATH, dkist[:-5]) + '.png')
    


if __name__ == '__main__':
    main()