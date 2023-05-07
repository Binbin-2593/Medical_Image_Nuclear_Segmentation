import numpy as np
import torch
import SimpleITK as sitk
#import surface_distance as surfdist
import cv2
from sklearn.metrics import f1_score
import torch.nn.functional as F
import os

from torch import device


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    #return (2. * intersection + smooth) / \
        #(output.sum() + target.sum() + smooth)
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

