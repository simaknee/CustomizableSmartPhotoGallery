from torch.utils.data import Dataset
import numpy as np
import os
import random
import cv2
import albumentations as A


class MiniImagenetDataset(Dataset):
    def __init__(self, dataset_dir, k, n, is_train=True):
        self.dataset_dir = dataset_dir
        self.kshot = k
        self.nway = n
        self.is_train = is_train

        self.classes = os.listdir(self.dataset_dir)
        self.num_classes = len(self.classes)
        self.transform = A.Compose([
            A.ShiftScaleRotate(border_mode=cv2.BORDER_REPLICATE),
            A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
            A.ToTensorV2()
        ])

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        # pick two images from same class
        if index % 2 == 1:
            label = 1
            class_index = random.randint(0, self.num_classes-1)
            selected_class = self.classes[class_index]
            class_path = os.path.join(self.dataset_dir, selected_class)
            image_paths = os.listdir(class_path)
            image1 = os.path.join(class_path, random.choice(image_paths))
            image2 = os.path.join(class_path, random.choice(image_paths))

        # pick two images from different class
        else:
            label = 0
            class_index1 = random.randint(0, self.num_classes-1)
            class_index2 = random.randint(0, self.num_classes-1)
            while class_index1 == class_index2:
                class_index2 = random.randint(0, self.num_classes-1)
            selected_class1 = self.classes[class_index1]
            selected_class2 = self.classes[class_index2]
            class_path1 = os.path.join(self.dataset_dir, selected_class1)
            class_path2 = os.path.join(self.dataset_dir, selected_class2)
            image_paths1 = os.listdir(class_path1)
            image_paths2 = os.listdir(class_path2)
            image1 = os.path.join(class_path1, random.choice(image_paths1))
            image2 = os.path.join(class_path2, random.choice(image_paths2))

        # read image
        print(image1)
        print(image2)
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)

        # apply transform and make tensor
        img1 = self.transform(image=image1)['image']
        img2 = self.transform(image=image2)['image']

        return img1, img2, label
