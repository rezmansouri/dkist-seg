import os
import torch
import numpy as np
from segnet import SegNet
from torch.utils.data import DataLoader
from sunrise import segDataset, segDataset_val, FocalLoss, acc, eval_metrics_sem


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    train_data = segDataset(
        '/home/dkist/data/imax/Train/', l=27_000)
    validation_data = segDataset_val(
        '/home/dkist/data/imax/Validate/', l=3_000)
    n_epochs = 200
    batch_size = 32
    model = SegNet(in_chn=1, out_chn=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)
    weights = torch.Tensor([1.0, 10.0, 10.0, 10.0, 1.0]).to(device)
    loss_fn = FocalLoss(gamma=10, alpha=weights)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False)

    stats = []
    lr_strikes = 0
    min_loss = torch.tensor(float('inf'))

    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = []
        train_acc = []
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc(y, yhat).numpy())
        lr_strikes += 1

        model.eval()
        val_loss_list = []
        val_acc_list = []
        val_overall_pa_list = []
        val_per_class_pa_list = []
        val_jaccard_index_list = []
        val_dice_index_list = []
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                pred_mask = model(x)
            val_loss = loss_fn(pred_mask, y.to(device))
            pred_mask_class = torch.argmax(pred_mask, axis=1)
        val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, _, _, _ = eval_metrics_sem(
            y.to(device), pred_mask_class.to(device), 5, device)
        val_overall_pa_list.append(val_overall_pa.cpu().detach().numpy())
        val_per_class_pa_list.append(val_per_class_pa.cpu().detach().numpy())
        val_jaccard_index_list.append(val_jaccard_index.cpu().detach().numpy())
        val_dice_index_list.append(val_dice_index.cpu().detach().numpy())
        val_loss_list.append(val_loss.cpu().detach().numpy())
        val_acc_list.append(acc(y, pred_mask).numpy())

        stats.append([epoch, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss_list),  np.mean(val_acc_list),
                      np.mean(val_overall_pa_list), np.mean(
                          val_per_class_pa_list),
                      np.mean(val_jaccard_index_list), np.mean(val_dice_index_list)])

        print(
            '\t'.join([f'{s:0.4f}' if i > 0 else f'{s}' for i, s in enumerate(stats[-1])]))

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        if is_best:
            lr_strikes = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model.state_dict(
            ), '/home/dkist/out/experiments/segnet_sunrise/model/segnet_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list)))

        if lr_strikes > 5:
            lr_scheduler.step()
            print(
                f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            lr_strikes = 0

    np.save('/home/dkist/out/experiments/segnet_sunrise/stats/stats.npy',
            np.array(stats, dtype=np.float32))


if __name__ == '__main__':
    main()
