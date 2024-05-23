import os
import cv2
import PIL
import time
import torch
import random
import numpy as np
from glob import glob
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from scipy.ndimage import rotate
from scipy.special import softmax
from torch.autograd import Variable
import torchvision.transforms as Ttorch
from scipy.ndimage import gaussian_filter

EPS = 1e-10


def get_param(degree, size):
    """
    Generate random angle for rotation and define the extension box for define their
    center
    """
    angle = float(torch.empty(1).uniform_(
        float(degree[0]), float(degree[1])).item())
    extent = int(np.ceil(np.abs(size*np.cos(np.deg2rad(angle))) +
                 np.abs(size*np.sin(np.deg2rad(angle))))/2)
    return angle, extent


def subimage(image, center, theta, width, height):
    """
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    """
    shape = (image.shape[1], image.shape[0]
             )  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    x = int(center[0] - width/2)
    y = int(center[1] - height/2)

    image = image[y:y+height, x:x+width]
    return image


def warp(x, flo):

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    flo = flo.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=False)
    mask = torch.ones(x.size())  # .cuda()
    mask = F.grid_sample(mask, vgrid, align_corners=False)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output*mask


class RandomRotation_crop(torch.nn.Module):
    def __init__(self, degrees, size):
        super().__init__()
        self.degree = [float(d) for d in degrees]
        self.size = int(size)

    def forward(self, img, pmap):
        """Rotate the image by a random angle.
           If the image is torch Tensor, it is expected
           to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

          Args:
              degrees (sequence or number): Range of degrees to select from.
              If degrees is a number instead of sequence like (min, max), the range of degrees
              will be (-degrees, +degrees).
              size (single value): size of the squared croped box

        Transformation that selects a randomly rotated region in the image within a specific
        range of degrees and a fixed squared size.
        """
        angle, extent = get_param(self.degree, self.size)

        if isinstance(img, Tensor):
            d_1 = img.size(dim=1)
            d_2 = img.size(dim=2)
        else:
            raise TypeError("Img should be a Tensor")

        ext_1 = [float(extent), float(d_1-extent)]
        ext_2 = [float(extent), float(d_2-extent)]

        end = time.time()
        print('2 -> ', end-start)
        start = end

        cut_pmap = softmax(pmap[int(ext_1[0]): int(
            ext_1[1]), int(ext_2[0]): int(ext_2[1])])
        end = time.time()
        print('3 -> ', end-start)
        start = end

        ind = np.array(list(np.ndindex(cut_pmap.shape)))
        end = time.time()
        print('4 -> ', end-start)
        start = end

        pos = np.random.choice(
            np.arange(len(cut_pmap.flatten())), 1, p=cut_pmap.flatten())
        end = time.time()
        print('5 -> ', end-start)
        start = end

        c = (int(ind[pos[0], 1])+int(ext_1[0]),
             int(ind[pos[0], 0])+int(ext_2[0]))

        img_raw = img.cpu().detach().numpy()

        cr_image_0 = subimage(img_raw[0], c, angle, self.size, self.size)
        cr_image_1 = subimage(img_raw[1], c, angle, self.size, self.size)

        end = time.time()
        print('6 -> ', end-start)
        start = end

        return torch.Tensor(np.array([cr_image_0, cr_image_1]), device='cpu')


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return Ttorch.functional.rotate(x, angle)


class SRS_crop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = int(size)

    def forward(self, img, pmap, ind):
        counter = np.arange(len(pmap))
        pos = np.random.choice(counter, 1, p=pmap)
        c = (int(ind[pos[0], 0])+int(self.size/2),
             int(ind[pos[0], 1])+int(self.size/2))
        img_raw = img.cpu().detach().numpy()

        x = int(c[0] - self.size/2)
        y = int(c[1] - self.size/2)

        cr_image_0 = img_raw[0, y:y+self.size, x:x+self.size]  # raw image
        cr_image_1 = img_raw[1, y:y+self.size, x:x+self.size]  # raw mask

        return torch.Tensor(np.array([cr_image_0, cr_image_1]), device='cpu'), c


class Secuential_trasn(torch.nn.Module):
    """Generates a secuential transformation"""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, pmap, ind):
        t_list = [img]
        for t in range(len(self.transforms)):
            if t == 1:
                rotation, c = self.transforms[t](t_list[-1], pmap, ind)
                t_list.append(rotation)
            else:
                t_list.append(self.transforms[t](t_list[-1]))

        return t_list[-1], c


class segDataset(torch.utils.data.Dataset):
    def __init__(self, root, l=1000, s=128):
        super(segDataset, self).__init__()
        start = time.time()
        self.root = root
        self.size = s
        self.l = l
        self.classes = {'Intergranular lane': 0,
                        'Uniform-shape granules': 1,
                        'Granules with dots': 2,
                        'Granules with a lane': 3,
                        'Complex-shape granules': 4}

        self.bin_classes = ['Intergranular lane', 'Uniform-shape granules', 'Granules with dots', 'Granules with a lane',
                            'Complex-shape granules']

        self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                                SRS_crop(self.size),
                                                RotationTransform(
                                                    angles=[0, 90, 180, 270]),
                                                Ttorch.RandomPerspective(
                                                    0.3, p=0.5, interpolation=Ttorch.InterpolationMode.NEAREST),
                                                Ttorch.RandomHorizontalFlip(
                                                    p=0.5),
                                                Ttorch.RandomVerticalFlip(
                                                    p=0.5)
                                                 ])

        self.file_list = sorted(glob(os.path.join(self.root, '*.npz')))

        print("Reading images...")
        self.smap = []
        self.mask_smap = []
        self.weight_maps = []
        self.index_list = []
        for f in self.file_list:
            file = np.load(f)
            psmap = file['smap'].astype(np.float32)
            pmsmap = file['cmask_map'].astype(np.float32)
            psmap = psmap/psmap.max()

            pad_value = int(
                ((np.sqrt(2*(psmap.shape[0]**2))-psmap.shape[0]))/2)

            # Padding for rotation
            pad_psmap = np.pad(
                psmap, ((pad_value, pad_value), (pad_value, pad_value)), mode='reflect')
            pad_pmsmap = np.pad(
                pmsmap, ((pad_value, pad_value), (pad_value, pad_value)), mode='reflect')

            self.rot_angle = np.arange(0, 90, 5)
            for a in self.rot_angle:
                # Continumm image
                vis1 = PIL.Image.fromarray(pad_psmap)
                p_map1 = rotate(vis1, a)
                x01 = int(abs(p_map1.shape[0]/2) - (psmap.shape[0]/2))
                x02 = int(abs(p_map1.shape[0]/2) + (psmap.shape[0]/2))
                p_map1 = p_map1[x01:x02, x01:x02]
                # Mask image
                vis2 = PIL.Image.fromarray(pad_pmsmap)
                p_map2 = np.asarray(vis2.rotate(a))
                x11 = int(abs(p_map2.shape[0]/2) - (pmsmap.shape[0]/2))
                x12 = int(abs(p_map2.shape[0]/2) + (pmsmap.shape[0]/2))
                p_map_mask = p_map2[x11:x12, x11:x12]
                # Deformation parameter
                nx, ny = p_map1.shape
                flo = 100 * np.random.randn(2, nx, ny)
                flo[0, :, :] = gaussian_filter(flo[0, :, :], sigma=10)
                flo[1, :, :] = gaussian_filter(flo[1, :, :], sigma=10)
                flo = torch.tensor(flo.astype('float32'))[None, :, :, :]

                # Deforming Continum image
                tmp1 = p_map1[None, None, :, :]
                tmp1 = torch.tensor(tmp1.astype('float32'))
                p_map_res = warp(tmp1, flo)

                # Deforming mask image
                u_lables = np.unique(p_map_mask).astype(int)
                tmp2 = np.array([(p_map_mask == c).astype(int)
                                for c in u_lables])
                tmp3 = []
                for im in tmp2:
                    t_im = im[None, None, :, :]
                    t_im = torch.tensor(t_im.astype('float32'))
                    t_res = warp(t_im, flo)
                    tmp3.append(torch.round(t_res))

                # edited myself
                p_map_mask_res = []
                for i in range(5):
                    p_map_mask_res.append(np.array(tmp3[i]*u_lables[i]))
                p_map_mask_res = np.array(p_map_mask_res).sum(axis=0)

                # p_map_mask_res = np.array(
                #     [tmp3[i]*u_lables[i] for i in range(5)]).sum(axis=0)

                p_map_res = p_map_res.cpu().detach().numpy()
                # p_map_mask_res = p_map_mask_res.cpu().detach().numpy()

                self.smap.append(p_map_res[0, 0, :, :])
                self.mask_smap.append(p_map_mask_res[0, 0, :, :])

                weight_maps = np.zeros_like(p_map_mask_res[0, 0, int(
                    self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]).astype(np.float32)
                weight_maps[p_map_mask_res[0, 0, int(
                    self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 0.0] = 1
                weight_maps[p_map_mask_res[0, 0, int(
                    self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 4.0] = 1
                weight_maps[p_map_mask_res[0, 0, int(
                    self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 1.0] = 10
                weight_maps[p_map_mask_res[0, 0, int(
                    self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 2.0] = 10
                weight_maps[p_map_mask_res[0, 0, int(
                    self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)] == 3.0] = 10

                wm_blurred = gaussian_filter(weight_maps, sigma=14)

                self.weight_maps.append(softmax(wm_blurred.flatten()))
                self.index_list.append(
                    np.array(list(np.ndindex(weight_maps.shape))))

        print("Done!")

    def __getitem__(self, idx):

        ind = np.random.randint(low=0, high=len(self.smap))
        smap = self.smap[ind]
        mask_smap = self.mask_smap[ind]

        # Full probability maps calculation
        weight_map = self.weight_maps[ind]
        index_l = self.index_list[ind]
        img_t, c = self.transform_serie(
            np.array([smap, mask_smap]).transpose(), weight_map, index_l)

        self.image = img_t[0].unsqueeze(0)
        self.mask = img_t[1].type(torch.int64)
        # return self.image, self.mask, ind, c  #for test central points
        return self.image, self.mask

    def __len__(self):
        return self.l


class segDataset_val(torch.utils.data.Dataset):
    def __init__(self, root, l=1000, s=128):
        super(segDataset_val, self).__init__()
        self.root = root
        self.size = s
        self.l = l
        self.classes = {'Intergranular lane': 0,
                        'Normal-shape granules': 1,
                        'Granules with dots': 2,
                        'Granules with lanes': 3,
                        'Complex-shape granules': 4}

        self.bin_classes = ['Intergranular lane', 'Normal-shape granules', 'Granules with dots', 'Granules with lanes',
                            'Complex-shape granules']

        self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                                SRS_crop(self.size),
                                                Ttorch.RandomHorizontalFlip(
                                                    p=0.5),
                                                Ttorch.RandomVerticalFlip(
                                                    p=0.5)
                                                 ])

        self.file_list = sorted(glob(os.path.join(self.root, '*.npz')))

        print("Reading images...")
        self.smap = []
        self.mask_smap = []
        self.weight_maps = []
        self.index_list = []
        for f in self.file_list:
            file = np.load(f)
            psmap = file['smap'].astype(np.float32)
            pmsmap = file['cmask_map'].astype(np.float32)
            psmap = psmap/psmap.max()

            self.smap.append(psmap)
            self.mask_smap.append(pmsmap)

            weight_maps = np.zeros_like(pmsmap[int(
                self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]).astype(np.float32)
            weight_maps[pmsmap[int(self.size/2):-int(self.size/2),
                               int(self.size/2):-int(self.size/2)] == 0] = 1
            weight_maps[pmsmap[int(self.size/2):-int(self.size/2),
                               int(self.size/2):-int(self.size/2)] == 4] = 1
            weight_maps[pmsmap[int(self.size/2):-int(self.size/2),
                               int(self.size/2):-int(self.size/2)] == 1] = 10
            weight_maps[pmsmap[int(self.size/2):-int(self.size/2),
                               int(self.size/2):-int(self.size/2)] == 2] = 10
            weight_maps[pmsmap[int(self.size/2):-int(self.size/2),
                               int(self.size/2):-int(self.size/2)] == 3] = 10

            wm_blurred = gaussian_filter(weight_maps, sigma=14)

            self.weight_maps.append(softmax(wm_blurred.flatten()))
            self.index_list.append(
                np.array(list(np.ndindex(weight_maps.shape))))

        print("Done!")

    def __getitem__(self, idx):

        ind = np.random.randint(low=0, high=len(self.smap))
        smap = self.smap[ind]
        mask_smap = self.mask_smap[ind]

        # Full probability maps calculation
        weight_map = self.weight_maps[ind]
        index_l = self.index_list[ind]
        img_t, c = self.transform_serie(
            np.array([smap, mask_smap]).transpose(), weight_map, index_l)

        self.image = img_t[0].unsqueeze(0)
        self.mask = img_t[1].type(torch.int64)
        # return self.image, self.mask, ind, c  #for test central points
        return self.image, self.mask

    def __len__(self):
        return self.l


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class mIoULoss(nn.Module):
    def __init__(self, weight = None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.w = weight

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        # inputs = F.softmax(inputs,dim=1)

        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        # target_oneHot = target
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        if self.w != None:
            loss = torch.sum(loss * self.w, 1)/torch.sum(self.w)

        ## Return average loss over classes and batch
        return 1-loss.mean()


def acc(label, predicted):
    seg_acc = (label.cpu() == torch.argmax(predicted, axis=1).cpu()
               ).sum() / torch.numel(label.cpu())
    return seg_acc


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def Per_class_OPA(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    return per_class_acc


def Per_class_jaccard(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    return jaccard


def Per_class_dice(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    return dice


def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def eval_metrics_sem(true, pred, num_classes, device):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes)).to(device)
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    pc_opa = Per_class_OPA(hist)
    pc_j = Per_class_jaccard(hist)
    pc_d = Per_class_dice(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice, pc_opa, pc_j, pc_d
