import argparse
import time

import cv2
import numpy as np
import torch
import torch.optim as optim

import copy
from tqdm import tqdm
import PIL

import archs
import losses
from utils import str2bool

import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from torch.optim import lr_scheduler
import os
from glob import glob
from sklearn.model_selection import train_test_split
from dataset import Dataset,Dataset_P
from metrics import iou_score

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__



def parse_args():
    parser = argparse.ArgumentParser()  # 创建对象，下面是向此对象中添加参数

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='inputs channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')  # 类别数，希望获得的每个像素的概率数，对于一个类和背景，使用n_classes=1
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2022_256',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config
config = vars(parse_args())

class SegModel:

    def __init__(
            self,
            path_to_pretrained_model: str = None
            ):
        """
        Allows for training, evaluation, and prediction of ResNet Models

        params
        ---------------
        path_to_pretrained_model - string - relative path to
            pretrained model - default None
        map_location - string - device to put model on - default cpu
        num_classes - int - number of classes to put on the deheaded ResNet
        """
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self._setup_resnet()
        
        if path_to_pretrained_model:
            self.model.load_state_dict(torch.load(
                path_to_pretrained_model, map_location=self.device
            ))
        """
        if path_to_pretrained_model:
            self.model = torch.load(
                path_to_pretrained_model, map_location=self.device
            )
        """
        (self.train_transform,self.val_transform) = self._setup_transform()

    def train(
            self,
            config, train_loader, val_loader
            ):
        """
        Impliments transfer learning based on new data

        params
        ---------------
        train_loader - torch DataLoader - Configured DataLoader
            for training, helpful when images are flowing from folder
        val_loader - torch DataLoader - Configured DataLoader
            for validation, helpful when images are flowing from folder
        num_epochs - int - number of epochs to use during training
        criterion - Loss function to assess model during training
            and evaluation - default None and CrossEntropyLoss
        optimizer - Optimizer algorithm - default Adam
        batch_size - int - Number of images to use per optimizer
            update - default 16
        early_stop_min_increase - float - Minimum increase in
            accuracy score to indicate model improvement per epoch
            - default 0.003
        early_stop_patience - int - Number of epochs to allow less
            than or equal to minimum improvement - default 10
        lr - float - Learning rate for optimization algorithm - default 0.0001

        returns
        ---------------
        model - trained torch model
        loss - list - history of loss across epochs
        acc - list - history of accuracy across epochs
        """
        # TODO: change data loader params to data params then in this function,
        # Transform those to data loaders so that the batch size can be dynamic
        start = time.time()
        model = self.model
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_iou = 0
        trigger = 0
        train_loss_over_time = []
        val_loss_over_time = []
        train_iou_over_time = []
        val_iou_over_time = []
        phases = ['train', 'val']

        # 网络的优化器
        params = filter(lambda p: p.requires_grad, model.parameters())
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                params, lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                                  nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        # 对优化器的学习率调整，https://zhuanlan.zhihu.com/p/344294796
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(e) for e in config['milestones'].split(',')],
                                                 gamma=config['gamma'])
        elif config['scheduler'] == 'ConstantLR':
            scheduler = None
        else:
            raise NotImplementedError

        criterion = losses.__dict__[config['loss']]().cuda()

        for epoch in tqdm(range(config['epochs'])):
            print(f"Epoch number: {epoch + 1} / {config['epochs']}")

            for phase in phases:
                if phase == 'train':
                    data_loader = train_loader
                    model.train()
                else:
                    data_loader = val_loader
                    model.eval()

                running_loss = 0
                running_iou = 0

                for inputs, labels in data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if config['deep_supervision']:
                            outputs = model(inputs)
                            loss = 0
                            for output in outputs:
                                loss += criterion(output, labels)
                            loss /= len(outputs)
                            iou = iou_score(outputs[-1], labels)
                        else:
                            output = model(inputs)  # 在网络中跑一遍
                            loss = criterion(output, labels)  # 输出和标签计算loss
                            iou = iou_score(output, labels)  # 计算IOU

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_iou += iou.item() * inputs.size(0)

                if phase == 'train':
                    epoch_loss = running_loss / len(train_loader.dataset)
                    train_loss_over_time.append(epoch_loss)
                    epoch_iou = (
                        running_iou.double() /
                        len(train_loader.dataset)
                        )
                    train_iou_over_time.append(epoch_iou)

                else:
                    epoch_loss = running_loss / len(val_loader.dataset)
                    val_loss_over_time.append(epoch_loss)
                    epoch_iou = (
                        running_iou.double() /
                        len(val_loader.dataset)
                        )
                    val_iou_over_time.append(epoch_iou)

                    if config['scheduler'] == 'CosineAnnealingLR':
                        scheduler.step()
                    elif config['scheduler'] == 'ReduceLROnPlateau':
                        scheduler.step(epoch_loss)

                print(f"{phase} loss: {epoch_loss:.3f}, acc: {epoch_iou:.3f}")

                trigger += 1
                if phase == 'val' and epoch_iou > best_iou:
                    best_iou = epoch_iou
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, 'trained_model_unet++.pth')

                    # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

            print('-' * 60)
            total_time = (time.time() - start) / 60
            print(
                f"Training completed. Time taken: {total_time:.3f} "
                f"min\nBest accuracy: {best_iou:.3f}"
                )
            model.load_state_dict(best_model_wts)
            self.model = model
            loss = {'train': train_loss_over_time, 'val': val_loss_over_time}
            iou = {'train': train_iou_over_time, 'val': val_iou_over_time}

            return model, loss, iou

    #一组数据的测试
    def evaluate(
            self,
            test_loader,
            config
            ):
        """
        Feeds set of images through model and evaluates relevant metrics
        as well as batch predicts. Prints loss and accuracy

        params
        ---------------
        test_loader - torch DataLoader - Configured
            DataLoader for evaluation, helpful when images flow from directory
        model - trained torch model - Model to use during
            evaluation - default None which retrieves model from attributes
        criterion - Loss function to assess model - Default
            None which equates to CrossEntropyLoss.

        returns
        ---------------
        preds - list - List of predictions to
            use for evaluation of non-included metrics
        labels_list - list - List of labels to
            use for evaluation of non-included metrics
        """
        criterion = losses.__dict__[config['loss']]().cuda()

        model = self.model
        model.eval()
        test_loss = 0
        test_iou = 0


        for c in range(config['num_classes']):
            os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

        for inputs, labels, meta in tqdm(test_loader, total=len(test_loader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)
                loss = criterion(output, labels)
                iou = iou_score(output, labels)

                # 把网络output转化成”0、1“数据（二分类）
                output = torch.sigmoid(output).cpu().numpy()

                # 把训练完的数据画图,存到output
                for i in range(len(output)):
                    for c in range(config['num_classes']):
                        cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                    (output[i, c] * 255).astype('uint8'))

            test_loss += loss.item() * inputs.size(0)
            test_iou += iou.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        test_iou = test_iou / len(test_loader.dataset)

        print(f"Test loss: {test_loss:.4f}\nTest acc: {test_iou:.4f}")

    #单张图片的测试结果，返回数组形式的预测图，需要cv2.imwrite()保存图片
    def predict_proba(
            self,
            img: PIL.Image.Image,
            show: bool = False
            ):
        """
        Feeds single image through network and returns
        top k predicted labels and probabilities

        params
        ---------------
        img - PIL Image - Single image to feed through model
        k - int - Number of top predictions to return
        index_to_class_labels - dict - Dictionary
            to map indices to class labels
        show - bool - Whether or not to
            display the image before prediction - default False

        returns
        ---------------
        formatted_predictions - list - List of top k
            formatted predictions formatted to include a tuple of
            1. predicted label, 2. predicted probability as str
        """
        if show:
            img.show()

        self.model.eval()
        test_dataset = Dataset_P(img=img,num_classes=config['num_classes'],transform=self.val_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=config['num_workers'],drop_last=False)

        for img in test_loader:

            img = img.to(self.device)
            output = self.model(img)
            output = torch.sigmoid(torch.squeeze(torch.squeeze(output,0),0)).detach().cpu().numpy()
            output = PIL.Image.fromarray(np.uint8(output*255))
        return output
    # change predict proba to actually be in line with sklearn's
    # API and create another function that formats the raw probabilities.
    # Then go back in to the website, and change the code accordingly.

    #接入网络模型代码
    def _setup_resnet(self):
        """
        Hidden function used in init if no pretrained model is specified.
        Helpful for implimenting transfer learning.
        It freezes all layers and then adds two final layers: one fully
        connected layer with RELU activation and dropout,
        and another as a final layer with number of class predictions
        as number of nodes. Also sends model to necessary device.

        params
        ---------------
        num_classes - int - Number of classes to predict

        returns
        ---------------
        model - torch model - torch model set up for transfer learning
        """
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
        model.to(self.device)

        return model

    # 设置数据
    def _setup_transform(self):
        """
        Sets up transformations needed for train data, val data, and test data.

        params
        ---------------
        None

        returns
        ---------------
        train_transform - torch transformer - transformer
            to use during training
        val_transform - torch transformer - transformer
            to use during validation
        test_transform - torch transformer - transformer
            to use during testing and inference
        """

        train_transform = Compose([  # 训练集
            albu.RandomRotate90(),
            transforms.Flip(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightness(),
                transforms.RandomContrast(),
            ], p=1),  # 按照归一化的概率选择执行哪一个
            albu.Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])

        val_transform = Compose([
            albu.Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])


        return (train_transform, val_transform)

    def _get_loader(self):
        img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=0)

        train_dataset = Dataset(  # 训练集
            img_ids=train_img_ids,
            img_dir=os.path.join('inputs', config['dataset'], 'images'),
            mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=self.train_transform)
        val_dataset = Dataset(  # 验证集
            img_ids=val_img_ids,
            img_dir=os.path.join('inputs', config['dataset'], 'images'),
            mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=self.val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)  # 不能整除的batch是否就不要了
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False)
        return (train_loader, val_loader)
