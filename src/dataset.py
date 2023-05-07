import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids: object, img_dir: object, mask_dir: object, img_ext: object, mask_ext: object, num_classes: object, transform: object = None) -> object:
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.#目录
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)#数据集，图片个数

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))#读取一张图

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])


        #数组沿深度方向进行拼接。
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


class Dataset_P(torch.utils.data.Dataset):
    def __init__(self, img: object,
                 num_classes: object, transform: object = None) -> object:

        self.img = img
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self,idx=0):
        img =self.img

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img = torch.Tensor(img)
        if torch.cuda.is_available():
            img = img.cuda()


        return img