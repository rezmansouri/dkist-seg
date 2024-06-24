import os
import sys
import torch
import numpy as np


def run(net, projector, train_loader, val_loader, n_epochs, criterion, optimizer, device, output_path, save_model=False, model_summary=False):

    # Ajust learing rate
    # Decays the learning rate of each parameter group by gamma every step_size epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)
    min_loss = torch.tensor(float('inf'))

    save_losses = []
    # Histograms
    save_h_train_losses = []
    save_h_val_losses = []
    scheduler_counter = 0

    for epoch in range(1, n_epochs+1):
        # training
        net.train()
        projector.train()
        loss_list = []
        for batch_i, (x1, x2) in enumerate(train_loader):

            z1 = net(x1.to(device))
            z2 = net(x2.to(device))
            z1 = projector(z1)
            z2 = projector(z2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    n_epochs,
                    batch_i,
                    len(train_loader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1

        # testing
        net.eval()
        projector.eval()
        val_loss_list = []

        for batch_i, (x1, x2) in enumerate(val_loader):
            with torch.no_grad():
                z1 = net(x1.to(device))
                z2 = net(x2.to(device))
                z1 = projector(z1)
                z2 = projector(z2)
            val_loss = criterion(z1, z2)
            val_loss_list.append(val_loss.cpu().detach().numpy())

        print('\n epoch {} - loss : {:.5f} - val loss : {:.5f}'.format(epoch, np.mean(loss_list), np.mean(val_loss_list)))

        save_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        compare_loss=np.mean(val_loss_list)
        is_best=compare_loss < min_loss
        print('\n', min_loss, compare_loss)
        if is_best and save_model:
            print("Best_model")
            scheduler_counter=0
            min_loss=min(compare_loss, min_loss)
            torch.save(net.state_dict(
            ), os.path.join(output_path, 'encoder_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))))
            torch.save(projector.state_dict(
            ), os.path.join(output_path, 'projector_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))))

        if scheduler_counter > 5:
            lr_scheduler.step()
            print(
                f"\nlowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter=0

        if epoch == n_epochs:
            print("Final Model")
            torch.save(net.state_dict(
            ), os.path.join(output_path, 'encoder_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))))
            torch.save(projector.state_dict(
            ), os.path.join(output_path, 'projector_epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))))

    if save_model:
        with open(os.path.join(output_path, 'stats.npy'), 'wb') as f:
            np.save(f, save_losses)
            np.save(f, save_h_train_losses)
            np.save(f, save_h_val_losses)
