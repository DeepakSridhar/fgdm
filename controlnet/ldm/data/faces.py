import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        if hasattr(self, "keys"):
            if "caption" == self.keys:
                example["caption"] = "A human face with eyes, mouth, nose, hair, cheeks, chin, ears, and forehead."
        return example



class CustomTrain(CustomBase):
    def __init__(self, root, size, training_images_list_file, keys=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CustomTest(CustomBase):
    def __init__(self, root, size, test_images_list_file, keys=None):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, root1, root2, size1, size2, train_list1, train_list2, keys=None, crop_size=None, coord=False):
        d1 = CustomTrain(root=root1, size=size1, training_images_list_file=train_list1, keys=keys)
        d2 = CustomTrain(root=root2, size=size2, training_images_list_file=train_list2, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if keys is not None:
            self.keys = keys
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "keys"):
            if "caption" == self.keys:
                ex["caption"] = "A human face with eyes, mouth, nose, hair, cheeks, chin, ears, and forehead."
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, root1, root2, size1, size2, train_list1, train_list2, keys=None, crop_size=None, coord=False):
        d1 = CustomTest(root=root1, size=size1, test_images_list_file=train_list1, keys=keys)
        d2 = CustomTest(root=root2, size=size2, test_images_list_file=train_list2, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex