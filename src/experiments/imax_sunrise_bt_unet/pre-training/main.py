import os
import torch
import train
import model
import utils
import numpy as np
import barlow_twins
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 1 if torch.cuda.is_available() else 0


def main():

    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    TRAIN_SIZE = 27_000
    BATCH_SIZE = 64
    VAL_SIZE = 3_000
    N_EPOCHS = 20
    OUTPUT_PATH = './results'

    if not os.path.exists(OUTPUT_PATH):
        raise ValueError('output path does not exist')

    print('Device:', device)

    train_files = []
    for f in os.listdir(TRAIN_DIR):
        file = dict(np.load(os.path.join(TRAIN_DIR, f)))
        train_files.append(file)
    train_dataset = utils.segDataset(train_files, TRAIN_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=barlow_twins.collate)

    val_files = []
    for f in os.listdir(VAL_DIR):
        file = dict(np.load(os.path.join(VAL_DIR, f)))
        val_files.append(file)
    val_dataset = utils.segDataset_val(val_files, VAL_SIZE)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=barlow_twins.collate)

    encoder = model.UNetEncoder(img_ch=1, scale=4).to(device)

    projector = model.Projector(4, 1024, 256, 128).to(device)

    criterion = barlow_twins.bt_loss

    params = list(encoder.parameters()) + list(projector.parameters())

    optimizer = torch.optim.Adam(params, lr=1e-6)#barlow_twins.LARS(params, lr=0)

    experiment_dir_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    OUTPUT_PATH = os.path.join(
        OUTPUT_PATH, experiment_dir_name)

    os.mkdir(OUTPUT_PATH)

    train.run(net=encoder, projector=projector, train_loader=train_loader, val_loader=val_loader, n_epochs=N_EPOCHS, criterion=criterion,
              optimizer=optimizer, device=device, output_path=OUTPUT_PATH, save_model=True, model_summary=False)


if __name__ == '__main__':
    main()
