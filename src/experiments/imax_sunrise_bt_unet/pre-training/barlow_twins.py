import torch
import random
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def bt_loss(z_a, z_b, lambd):
    batch_size, n_features = z_a.shape

    z_a_norm = nn.BatchNorm1d(n_features)(z_a)
    z_b_norm = nn.BatchNorm1d(n_features)(z_b)

    c = z_a_norm.T @ z_b_norm / batch_size

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    loss = on_diag + lambd * off_diag
    return loss


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    

transform = Transform()


def collate(batch):
    return transform(batch)
