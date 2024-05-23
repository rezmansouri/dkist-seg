import os
import sys
import json
import utils
import torch
import train
import shutil
import losses
import models
import numpy as np
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 1 if torch.cuda.is_available() else 0


def main():

    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as file:
        cfg = json.load(file)

    TRAIN_DIR = cfg['train_dir']
    VAL_DIR = cfg['val_dir']
    TRAIN_SIZE = cfg['train_size']
    BATCH_SIZE = cfg['batch_size']
    VAL_SIZE = cfg['val_size']
    LOSS_STR = cfg['loss']
    N_EPOCHS = cfg['n_epochs']
    MODEL_STR = cfg['model']
    OUTPUT_PATH = cfg['output_path']

    if not os.path.exists(OUTPUT_PATH):
        raise ValueError('output path does not exist')

    print('Device:', device)

    train_files = []
    train_masks = []
    for f in os.listdir(TRAIN_DIR):
        file = np.load(os.path.join(TRAIN_DIR, f))
        train_files.append(file)
        train_masks.append(file['cmask_map'])
    train_dataset = utils.segDataset(train_files, TRAIN_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    val_files = []
    for f in os.listdir(VAL_DIR):
        file = np.load(os.path.join(VAL_DIR, f))
        val_files.append(file)
    val_dataset = utils.segDataset_val(val_files, VAL_SIZE)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    if MODEL_STR == 'unet':
        model = models.UNet()
    else:
        raise ValueError('wrong model in config')

    loss_weights = utils.get_weights(train_masks).to(device)

    if LOSS_STR == 'focal':
        criterion = losses.FocalLoss(alpha=loss_weights).to(device)
    elif LOSS_STR == 'lovazs':
        criterion = losses.Lovazs()
    else:
        raise ValueError('wrong loss in config')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    experiment_dir_name = MODEL_STR + '_' + LOSS_STR + \
        '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    OUTPUT_PATH = os.path.join(
        OUTPUT_PATH, experiment_dir_name)

    os.mkdir(OUTPUT_PATH)
    shutil.copy2(cfg_path, OUTPUT_PATH)

    train.run(model=model, train_loader=train_loader, val_loader=val_loader, n_epochs=N_EPOCHS, criterion=criterion,
              optimizer=optimizer, device=device, output_path=OUTPUT_PATH, save_model=False, model_summary=True)


if __name__ == '__main__':
    main()
