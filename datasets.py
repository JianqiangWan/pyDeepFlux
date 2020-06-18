import cv2
import numpy as np
import os.path as osp
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms as T

class FluxSkeletonDataset(Dataset):

    def __init__(self, dataset='sklarge'):
        
        self.dataset = dataset

        if self.dataset == 'sklarge':
            self.data_root_dir = 'data/SK-LARGE/'
        elif self.dataset == 'sympascal':
            self.data_root_dir = 'data/SymPASCAL-by-KZ/'

        file_dir = self.data_root_dir + 'aug_data/train_pair.lst'

        with open(file_dir, 'r') as f:
            self.image_names = f.read().splitlines()

        self.dataset_length = len(self.image_names)

        self.normalize = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
    
    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        image_path, label_path = self.image_names[index].split()
        image_name = image_path.split('/')[-1]

        image = cv2.imread(self.data_root_dir + image_path, 1)
        vis_image = image.copy()

        image = image.astype(np.float32)
        # normalize input image
        image = self.normalize(image)

        skeleton = cv2.imread(self.data_root_dir + label_path, 0)
        skeleton = (skeleton > 0).astype(np.uint8)

        # compute flux and dilmask
        # https://github.com/YukangWang/DeepFlux/blob/master/examples/DeepFlux/pylayerUtils.py
        kernel = np.ones((15,15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1-skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
        direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

        direction[0] = direction[0]*(dilmask > 0)
        direction[1] = direction[1]*(dilmask > 0)

        flux = -1*np.stack((direction[0], direction[1]))

        dilmask = (dilmask>0).astype(np.float32)
        dilmask = dilmask[np.newaxis, ...]

        skeleton = (skeleton > 0).astype(np.float32)
        skeleton = skeleton[np.newaxis]

        return image, vis_image, skeleton, dilmask, flux, self.dataset_length, image_name