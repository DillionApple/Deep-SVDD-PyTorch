import os

from PIL import Image
from torch.utils.data.dataset import Dataset
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import global_contrast_normalization

import torchvision.transforms as transforms


class Mobile_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = tuple([1])

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        self.train_set = Mobile(
            os.path.join(self.root, "training"),
            os.path.join(self.root, "test"),
            transform,
            True
        )

        self.test_set = Mobile(
            os.path.join(self.root, "training"),
            os.path.join(self.root, "test"),
            transform,
            False
        )


class Mobile(Dataset):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __load_images_from_path(self, root):
        imgs = []
        labels = []
        for parent_dir, dirs, files in os.walk(root):
            for filename in files:
                if (filename.startswith("neg")):
                    labels.append(1)
                else:
                    labels.append(0)
                filepath = os.path.join(parent_dir, filename)
                img = Image.open(filepath).resize((256, 256)).convert("L")
                imgs.append(img)
        return imgs, labels

    def __init__(self, train_root, test_root, transform, train):
        self.transform = transform

        self.train_data, self.train_labels = self.__load_images_from_path(train_root)

        self.test_data, self.test_labels = self.__load_images_from_path(test_root)

        self.train = train

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index  # only line changed

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
