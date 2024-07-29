# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
# import h5py
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
import visdom
import sklearn.model_selection
from sklearn import preprocessing
import torchvision.transforms as transforms
from scipy import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os
import imageio
import scipy.io as sio
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve



DATASETS_CONFIG = {
        'PaviaC': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                     'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                     'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Botswana': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                     'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            'img': 'Botswana.mat',
            'gt': 'Botswana_gt.mat',
            },
        'houston2013': {
            'urls': [#'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                      #'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'Houston2013.mat',
            'gt': 'Houston2013_gt.mat',
            },
        'houston2018': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
            # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
            ],
            'img': 'Houston2013.mat',
            'gt': 'Houston2013_gt.mat',
        },
        'Indian1': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'Indian2': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'ShangHai': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'Dioni': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'Loukia': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'HangZhou': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            }

    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt1 = open_file(folder + 'Pavia_gt.mat')['pavia_gt']
        print(np.count_nonzero(gt1))
        w, h = gt1.shape
        gt_ts = np.zeros_like(gt1)
        gt_ts[(gt1 == 2)[:w, :h]] = 1
        gt_ts[(gt1 == 5)[:w, :h]] = 2
        gt_ts[(gt1 == 7)[:w, :h]] = 3
        gt_ts[(gt1 == 4)[:w, :h]] = 4
        gt_ts[(gt1 == 6)[:w, :h]] = 5
        gt_ts[(gt1 == 3)[:w, :h]] = 6
        gt_ts[(gt1 == 9)[:w, :h]] = 7

        label_values = ['Undefined', 'Trees', "Bare Soil", 'Bitumen', 'Self-Blocking Bricks', "Asphalt","Meadows","Shadows"]
        # label_values = ["Undefined", "Water", "Trees", "Asphalt",
        #                 "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
        #                 "Meadows", "Bare Soil"]

        gt=np.copy(gt_ts)
        ignored_labels = [0]
        rest_band = 102

    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']
        # H,W,B=img1.shape
        # img=np.zeros((H,W,B+5))
        # img[:,:,0:B]=np.copy(img1)
        # img[:, :,B] = np.copy(img1[:,:,B-1])
        # img[:,:,B+1:B+3]=np.copy(img[:,:,B-1:B])
        # img[:, :, B + 3:B + 5] = np.copy(img[:, :, B - 1:B])

        rgb_bands = (55, 41, 12)

        gt1 = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']
        w,h=gt1.shape

        gt_ts=np.zeros_like(gt1)
        gt_ts[(gt1 == 4)[:w, :h]] = 1
        gt_ts[(gt1 == 6)[:w, :h]] = 2
        gt_ts[(gt1 == 7)[:w, :h]] = 3
        gt_ts[(gt1 == 8)[:w, :h]] = 4
        gt_ts[(gt1 == 1)[:w, :h]] = 5
        gt_ts[(gt1 == 2)[:w, :h]] = 6
        gt_ts[(gt1 == 9)[:w, :h]] = 7

        label_values = ['Undefined', 'Trees', "Bare Soil", 'Bitumen', 'Self-Blocking Bricks','Asphalt',
                        "Meadows","Shadows"]

        # label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
        #                 'Painted metal sheets', 'Bare Soil', 'Bitumen',
        #                 'Self-Blocking Bricks', 'Shadows']

        gt = np.copy(gt_ts)
        rest_band=103
        ignored_labels = [0]

    elif dataset_name == 'houston2013':
        # Load the image
        imgsd = open_file(folder + 'Houston.mat')['Houston']
        img_sampling=np.zeros_like(imgsd)
        # H,W,B=img1.shape
        # img=np.zeros((H,W,B+3))
        # img[:,:,0:B]=np.copy(img1)
        # img[:, :,B] = np.copy(img1[:,:,B-1])
        # img[:,:,B+1:B+3]=np.copy(img[:,:,B-1:B])

        # --------sampling to 48 bands--------
        bands = 0
        for i in range(48):
            img_sampling[:, :, i] = np.copy(imgsd[:, :, bands])
            bands += 3

        print(bands)
        img = np.copy(img_sampling)



        rgb_bands = (59, 40, 13)

        gt =open_file(folder + 'houston_tf13.npy')
        H,W=gt.shape

        label_values = ['Undefined','Healthy grass','Stressed grass',
                        'Trees','Water','Residential','Commercial','Road']

        ignored_labels = [0]
        rest_band = 108

    elif dataset_name == 'houston2018':
        # Load the image
        img = open_file(folder + '2018_IEEE_GRSS_DF_Contest_Samples_TR.tif')[:,:,:-2]
        # H,W,B=img1.shape
        # img=np.zeros((H,W,B+1))
        # img[:,:,0:B]=np.copy(img1)


        rgb_bands = (19, 40, 13)

        gt = open_file(folder + 'houston_tf18.npy')

        label_values = ['Undefined','Healthy grass','Stressed grass','trees','Water',
                        'Residential buildings','Non-residential buildings','Roads']

        ignored_labels = [0]
        rest_band = 48


    elif dataset_name == 'Indian1':
        # Load the image
        img = open_file(folder + 'Indiana_1.mat')
        img = img['DataCube1']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indiana_1_gt.mat')['gt1']
        print(np.max(gt))
        label_values = ["Undefined", "Alfalfa", "Corn-Cleantill", "Corn-Cleantill-EW",
                        "Orchard","Soybean-CleanTill", "Soybeans-CleanTill-EW",
                        "Wheat"]

        ignored_labels = [0]
        rest_band=192

    elif dataset_name == 'Indian2':
        # Load the image
        img = open_file(folder + 'Indiana_2.mat')
        img = img['DataCube2']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indiana_2_gt.mat')['gt2']
        label_values = ["Undefined", "Alfalfa", "Corn-Cleantill", "Corn-Cleantill-EW",
                        "Orchard","Soybean-CleanTill", "Soybeans-CleanTill-EW",
                        "Wheat"]

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'Dioni':
        # Load the image
        img = open_file(folder + 'Dioni.tif')

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Dioni_gt_out68.mat')['map']

        label_values = ['Undefined', 'Dense Urban Fabric', "Mineral Extraction Sites", 'Non Irrigated Arable Land',
                        'Fruit Trees', "Olive Groves",'Coniferous Forest','Dense Sderophyllous Vegetation',
                        'Sparse Sderophyllous Vegetation','Sparcely Vegetated Areas','Rocks and Sand',
                        'Water','Coastal Water']

        rest_band=176
        ignored_labels = [0]
    elif dataset_name == 'Loukia':
        # Load the image
        img = open_file(folder + 'Loukia.tif')

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Loukia_gt_out68.mat')['map']

        label_values = ['Undefined', 'Dense Urban Fabric', "Mineral Extraction Sites", 'Non Irrigated Arable Land',
                        'Fruit Trees', "Olive Groves",'Coniferous Forest','Dense Sderophyllous Vegetation',
                        'Sparse Sderophyllous Vegetation','Sparcely Vegetated Areas','Rocks and Sand',
                        'Water','Coastal Water']

        rest_band = 176
        ignored_labels = [0]

    elif dataset_name == 'HangZhou':
        # Load the image
        img = open_file(folder + '2.mat')
        img = img['DataCube2']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + '2_gt.mat')['gt2']
        label_values = ["Undefined", "Water", "Land/Building", "Plant"]

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'ShangHai':
        # Load the image
        img = open_file(folder + '1.mat')
        img = img['DataCube1']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + '1_gt.mat')['gt1']
        label_values = ["Undefined", "Water", "Land/Building", "Plant"]

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    #data = preprocessing.scale(data)
    data  = preprocessing.minmax_scale(data)
    img = data.reshape(img.shape)
    return img, gt, label_values, ignored_labels, rgb_bands, palette,rest_band


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data,data_name, rest_band,gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = data_name
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.flip_augmentation = hyperparams['flip_augmentation']
        # self.radiation_augmentation = hyperparams['radiation_augmentation']
        # self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']

        self.spectral_fusion=hyperparams['spectral_fusion']
        self.rest_band=rest_band
        supervision = 'full'
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # if self.flip_augmentation and self.patch_size > 1:
        #     # Perform data augmentation (only on 2D patches)
        #     data, label = self.flip(data, label)
        # if self.radiation_augmentation and np.random.random() < 0.1:
        #         data = self.radiation_noise(data)
        # if self.mixture_augmentation and np.random.random() < 0.2:
        #         data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # data=(np.copy(data1[0:144,:,:])).reshape((4,96,96))
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN





            # viz = visdom.Visdom(env=self.DATASET + ' ' + self.MODEL)
            # display_predictions(np.transpose(data[0:3,:,:],(1,2,0)),viz)
            # display_predictions(np.transpose(data1[0:3, :, :], (1, 2, 0)), viz)
            # quit()


        return data , label


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
        print("Sampling {} with train size = {}".format(mode, train_size))
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features

            train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    elif ext == '.npy':
        img = np.load(dataset)
        H,W=img.shape
        return img
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def principal_component_extraction(spectral_original, variance_required):
    ## Variable List
    ## spectral_original: The original non reduced image
    ## variance_required: The required variance  ratio from 0 to 1

    ## Output list
    ## spectral_pc_final: The dimensional reduces image

    # 2d reshape
    spectral_2d = spectral_original.reshape(
        (spectral_original.shape[0] * spectral_original.shape[1], spectral_original.shape[2]))
    # Feature scaling preprocessing step
    spectral_2d = preprocessing.scale(spectral_2d)

    if (spectral_2d.shape[1] < 100):
        pca = PCA(n_components=spectral_2d.shape[1])
    else:
        pca = PCA(n_components=100)
    spectral_pc = pca.fit_transform(spectral_2d)
    explained_variance = pca.explained_variance_ratio_

    if (np.sum(explained_variance) < variance_required):
        raise ValueError("The required variance was too high. Values should be between 0 and 1.")

    # Select the number of principal components that gives the variance required
    explained_variance_sum = np.zeros(explained_variance.shape)
    sum_ev = 0
    component_number = 0
    for i in range(explained_variance.shape[0]):
        sum_ev += explained_variance[i]
        if (sum_ev > variance_required and component_number == 0):
            component_number = i + 1
        explained_variance_sum[i] = sum_ev

    # Removed the unnecessary components and reshape in original 3d form
    spectral_pc = spectral_pc[:, :component_number]
    spectral_pc_final = spectral_pc.reshape((spectral_original.shape[0], spectral_original.shape[1], component_number))

    return spectral_pc_final

def HSI_dataloder(args):
    imageSD, gtSD, LABEL_VALUES, IGNORED_LABELS, RGB_BANDSSD, palette, rest_bandSD = get_dataset(args.source_HSI,
                                                                                                          args.folder)
    imageTD, gtTD, _, _, RGB_BANDSTD, _, rest_bandTD = get_dataset(args.target_HSI,
                                                                            args.folder)
    # imgSD_PCA = principal_component_extraction(imageSD, 0.998)
    # imgTD_PCA = principal_component_extraction(imageTD, 0.998)
    H_SD, W_SD, Channel_SD = imageSD.shape
    H_TD, W_TD, Channel_TD = imageTD.shape

    Inchannel_pca = min(Channel_SD, Channel_TD)
    imgSD = np.copy(imageSD[:, :, 0:Inchannel_pca])
    imgTD = np.copy(imageTD[:, :, 0:Inchannel_pca])



    # pseudo_tgt = np.load('pseudo_labe_houston13_18.npy')

    # src_img, src_labels =get_dataset(args.source, path=args.db_path)
    # tgt_img, tgt_labels = get_dataset(args.target, path=args.db_path)
    train_gt,rest_src = sample_gt(gtSD, args.training_sample, mode='random')
    img_src_con=np.copy(imgSD)
    train_gt_src_con=np.copy(train_gt)





    train_gt_src_con=np.copy(train_gt)

    img_src_con=np.copy(imgSD)

    sparse_ground_truth = sparseness_operator(rest_src, 5)


    sparse_gt,_=sample_gt(gtSD, 0.5, mode='random')
    n_classes = int(np.max(gtSD) + 1)




# Ensure the number of per class is at least 100.
    for i in range(n_classes-1):
        count_class = np.copy(train_gt_src_con)
        sparse_class=np.copy(sparse_ground_truth)

        count_class[(train_gt != i + 1)] = 0
        sparse_class[(sparse_ground_truth != i + 1)[:H_SD, :W_SD]] = 0
        class_num=np.count_nonzero(count_class)
        if class_num<=50:
            train_gt_src_con+=sparse_class
        add_class = np.copy(train_gt_src_con)
        add_class[(train_gt_src_con != i + 1)] = 0

        print(LABEL_VALUES[i + 1],':', class_num,np.count_nonzero(add_class))

    train_gt, val_gt = sample_gt(train_gt_src_con, 0.95, mode='random')
    pseudo_tar, _ = sample_gt(gtTD, np.count_nonzero(train_gt_src_con), mode='random')
    pseudo_tgt, _ = sample_gt(pseudo_tar, 0.95, mode='random')



    for i in range(n_classes):
        count_src = np.copy(train_gt)
        count_tar = np.copy(pseudo_tgt)
        count_src[(train_gt != i + 1)] = 0
        count_tar[(pseudo_tgt != i + 1)] = 0
        classnum_src=np.count_nonzero(count_src)
        classnum_tar = np.count_nonzero(count_tar)

        print(i + 1, classnum_src, classnum_tar)

    print(np.count_nonzero(train_gt), np.count_nonzero(pseudo_tgt))
    # quit()
    test_gt = np.copy(gtTD)

    cfg = vars(args)
    cfg['ignored_labels'] = IGNORED_LABELS
    cfg['center_pixel'] = True
    cfg['HSI_class']=n_classes
    cfg['inchanl']=Inchannel_pca
    cfg['label_values']=LABEL_VALUES
    src_trainset = HyperX(img_src_con, cfg['source_HSI'], rest_bandSD, train_gt, **cfg)
    src_valset = HyperX(img_src_con, cfg['source_HSI'], rest_bandSD, val_gt, **cfg)
    tgt_trainset = HyperX(imgTD, cfg['target_HSI'], rest_bandTD, pseudo_tgt, **cfg)
    tgt_testset = HyperX(imgTD, cfg['target_HSI'], rest_bandTD, test_gt, **cfg)






    #Visualization of high-dimensional data
    #vis_SNE(src_trainset.data,src_trainset.label,tgt_trainset.data,tgt_trainset.label)


    return src_trainset,imgSD,tgt_trainset,tgt_testset

def vis_SNE(dataset_src,label_src,dataset_tar,label_tar):


    #img_src=np.copy(dataset_src)
    img_src=np.load('./Feature/HangZhousrc_features.npy')
    gt_src=np.copy(label_src)
    #img_tar = np.copy(dataset_tar)
    img_tar = np.load('Feature/HangZhoutar_features.npy')
    gt_tar = np.copy(label_tar)
    class_show=1
    H,W,C=img_src.shape
    print(np.count_nonzero(gt_src))
    # data_sne_src=np.zeros((np.count_nonzero(gt_src!=0),C))
    # data_sne_tar = np.zeros((np.count_nonzero(gt_tar!=0), C))
    data_sne_src=img_src[(gt_src==class_show),:]
    data_sne_tar = img_tar[(gt_tar == class_show), :]

    # for i in range(C):
    #     clip_data_src=np.copy(img_src[:,:,i])
    #     clip_data_tar = np.copy(img_tar[:, :, i])
    #     # clip_data[(gt==0)]=0
    #     data_sne_src[:,i]=np.copy(clip_data_src[(gt_src!=0)])
    #     data_sne_tar[:, i] = np.copy(clip_data_tar[(gt_tar!=0)])
    # dataset.data=np.copy(data_sne)
    labels_src=np.copy(gt_src[(gt_src==class_show)])
    labels_tar = np.copy(gt_tar[(gt_tar ==class_show)])

    # vis_data = dataset
    # digits = vis_data


    X_tsne_src = TSNE(n_components=2, random_state=100).fit_transform(data_sne_src)
    X_tsne_tar = TSNE(n_components=2, random_state=100).fit_transform(data_sne_tar)

    preb_src = KMeans(n_clusters=1,random_state=50).fit(X_tsne_src).cluster_centers_
    preb_tar = KMeans(n_clusters=1, random_state=50).fit(X_tsne_tar).cluster_centers_


    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    plt.figure()
    plt.scatter(X_tsne_src[:, 0], X_tsne_src[:, 1], cmap='plasma', label="src_class1",marker='.',s=1.8)
    plt.scatter(preb_src[:, 0], preb_src[:, 1], cmap='b', label="center_src", marker='o', s=25)

    plt.scatter(X_tsne_tar[:, 0], X_tsne_tar[:, 1], cmap='rainbow', label="tar_class1", marker='.',s=1.8)
    plt.scatter(preb_tar[:, 0], preb_tar[:, 1], c='r', label="center_tar", marker='o', s=25)
    plt.legend()
    plt.savefig('images/featuresSH_1.png', dpi=120)
    plt.show()
    quit()

def feature_SNE(data_src,label_src,data_tar,label_tar):

    img_src=np.copy(data_src.cpu().detach().numpy())
    labels_src=np.copy(label_src.cpu().detach().numpy())


    img_tar = np.copy(data_tar.cpu().detach().numpy())
    labels_tar = np.copy(label_tar.cpu().detach().numpy())

    X_tsne_src = TSNE(n_components=2, random_state=33).fit_transform(img_src)
    X_tsne_tar = TSNE(n_components=2, random_state=33).fit_transform(img_tar)


    plt.scatter(X_tsne_src[:, 0], X_tsne_src[:, 1], c=labels_src, label="t-SNE",marker='.',s=1.2)
    plt.scatter(X_tsne_tar[:, 0], X_tsne_tar[:, 1], c=labels_tar+8, label="t-SNE", marker='.',s=1.2)
    plt.legend()
    plt.savefig('images/features.png', dpi=120)
    plt.show()
    quit()

def sparseness_operator(ground_truth,number_of_samples):
    sparse_ground_truth =  np.reshape(np.copy(ground_truth) , -1)
    # HOW MANY OF EACH LABEL DO WE WANT
    number_of_classes = np.amax(ground_truth)
    for i in range(number_of_classes):
        index = np.where(sparse_ground_truth == i+1)[0]
        bing=index.shape[0]
        if(index.shape[0] < number_of_samples):
            index = np.random.choice(index,index.shape[0],replace = False)
        else:
            index = np.random.choice(index,index.shape[0] - number_of_samples,replace = False)
        index = np.sort(index)
        sparse_ground_truth[index] = 0
    sparse_ground_truth = sparse_ground_truth.reshape((ground_truth.shape))
    # sio.savemat('Houston_train_3.mat',mdict={'houston':sparse_ground_truth})

    return sparse_ground_truth

