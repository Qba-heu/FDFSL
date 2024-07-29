

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset

import numpy as np
import os
import math
import argparse

import pickle

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from timm.models.layers import trunc_normal_, DropPath

import time,datetime
import utils
import models


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200) # PaviaU=103；salinas=204 ; Indian=200
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 2000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-l1","--lambda1", type = float, default = 1)
parser.add_argument("-l2","--lambda2", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-da","--tar_dataset",type=str, default='PaviaU')
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

# args = parser.parse_args(args=[])
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
N_DIMENSION = args.n_dim
Target_data=args.tar_dataset
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
Lambda1 = args.lambda1
Lambda2 = args.lambda2
# Lambda3 = args.lambda3


if Target_data=='PaviaU':
    TAR_INPUT_DIMENSION = 103
    TEST_CLASS_NUM = CLASS_NUM = 9
elif Target_data == 'IndianPines':
    TAR_INPUT_DIMENSION = 200
    TEST_CLASS_NUM = CLASS_NUM = 16
elif Target_data== 'Salinas':
    TAR_INPUT_DIMENSION = 204
    TEST_CLASS_NUM = CLASS_NUM = 16
elif Target_data== 'PaviaC':
    TAR_INPUT_DIMENSION = 102
    TEST_CLASS_NUM = CLASS_NUM = 9
elif Target_data== 'HHK':
    TAR_INPUT_DIMENSION = 285
    TEST_CLASS_NUM = CLASS_NUM = 18
else:
    raise ValueError('Dataset out of scoop')

# Hyper Parameters in target domain data set
 # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data'] # (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train


print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
if Target_data=='PaviaU':
    test_data = 'datasets/PaviaU/PaviaU.mat'
    test_label = 'datasets/PaviaU/PaviaU_gt.mat'
elif Target_data == 'IndianPines':
    test_data = 'datasets/IndianPines/Indian_pines_corrected.mat'
    test_label = 'datasets/IndianPines/Indian_pines_gt.mat'
elif Target_data== 'Salinas':
    test_data = 'datasets/Salinas/Salinas_corrected.mat'
    test_label = 'datasets/Salinas/Salinas_gt.mat'
elif Target_data== 'PaviaC':
    test_data = 'datasets/PaviaC/Pavia.mat'
    test_label = 'datasets/PaviaC/Pavia_gt.mat'
elif Target_data== 'HHK':
    test_data = 'datasets/YRE/data_hsi.mat'
    test_label = 'datasets/YRE/mask_test.mat'
else:
    raise ValueError('Dataset out of scoop')



Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.emb_size = 100

        self.patch_size = 9

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)


        # self.final_feat_dim = 128
        # self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM, bias=False)

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1, 100,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            _, t, c, w, h = x.size()
            s1 = t * c * w * h
            x = self.conv(x)
            x = x.view(x.shape[0],-1)
            s2 = x.size()[1]
        return s1, s2

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)
        x = self.block1(x) #(1,8,100,9,9)
        x = self.maxpool1(x) #(1,8,25,5,5)
        x = self.block2(x) #(1,16,25,5,5)
        x = self.maxpool2(x) #(1,16,7,3,3)
        x = self.conv(x) #(1,32,5,1,1)
        x = x.view(x.shape[0],-1) #(1,160)


        # y = self.classifier(x)
        return x

class ECAAttention(nn.Module):

    def __init__(self,channel=3, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
        self.out_channel = channel
        self.pos_encode=self.Position(9)


    def Position(self,L):
        pos = np.zeros((L, L))
        xc = L // 2
        yc = L // 2
        for h in range(L):
            for k in range(L):
                pos[h, k] = np.max([np.abs(h - xc), np.abs(k - yc)]) + 1

        cc = np.expand_dims(pos, axis=0)
        cc = np.tile(cc, (self.out_channel, 1, 1))

        pos_weight = 1 / cc
        pos_weight=pos_weight.astype(np.float32)
        return pos_weight

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # b,c,h,w=x.shape
        # cc = np.expand_dims(self.pos_encode, axis=0)
        # batch_pos = np.tile(cc, (b, 1,1, 1))
        # batch_pos=torch.from_numpy(batch_pos).cuda()
        # up_pos=1/batch_pos
        x=x#*up_pos
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class gnconv(nn.Module):
    def __init__(self, dim, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)


        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 3, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class hor_Block(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.out_channel=dim
        self.gnconv = gnconv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.scaler_pos = 1.0

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        # self.pos_embed = torch.nn.Parameter(
        #     torch.normal(0, 1e-1, size=[1, self.out_channel, 9,9], dtype=torch.float,
        #                  device='cuda'),
        #     requires_grad=True)
        self.pos_encode= torch.from_numpy(self.Position(9)).cuda()
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 9, 9))
        # self.pos_embed = nn.Parameter(torch.from_numpy(self.Position(9)))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.num=1

    def Position(self,L):
        pos = np.zeros((L, L))
        xc = L // 2
        yc = L // 2
        for h in range(L):
            for k in range(L):
                pos[h, k] = np.max([np.abs(h - xc), np.abs(k - yc)]) + 1

        cc = np.expand_dims(pos, axis=0)
        cc = np.tile(cc, (self.out_channel, 1, 1))

        pos_weight = 1 / cc
        pos_weight=pos_weight.astype(np.float32)
        return pos_weight

    def forward(self, x):
        B, C, H, W  = x.shape
        # self.num+=1
        # print(x.shape)
        x = x + self.scaler_pos *self.pos_embed*self.pos_encode

        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

        self.CAM = hor_Block(out_dimension)



    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        x_attn = self.CAM(x)


        return x+x_attn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)

        self.final_feat_dim = self._get_final_flattened_size()  # 64+32
        # self.final_feat_dim = 160  # 64+32
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.fea_emd = nn.Linear(in_features=N_DIMENSION, out_features=self.final_feat_dim)
        self.fc_norm = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier2 = nn.Linear(in_features=N_DIMENSION, out_features=CLASS_NUM)
        self.fc_norm2 = nn.BatchNorm1d(CLASS_NUM)

        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

        self.class_num = CLASS_NUM

        self.domain_num = 2

        self.sms = torch.nn.Parameter(
            torch.normal(0, 1e-1, size=[self.domain_num, self.final_feat_dim, self.class_num], dtype=torch.float, device='cuda'),
            requires_grad=True)
        self.sm_biases = torch.nn.Parameter(
            torch.normal(0, 1e-1, size=[self.domain_num, self.class_num], dtype=torch.float, device='cuda'),
            requires_grad=True)

        self.embs = torch.nn.Parameter(
            torch.normal(mean=0., std=1e-4, size=[3, self.domain_num - 1], dtype=torch.float, device='cuda'),
            requires_grad=True)
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda'),
                                        requires_grad=True)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.ones((2, N_DIMENSION,
                             9, 9))
            x = self.feature_encoder(x)

            _, c = x.size()
        return c

    def forward(self, x, domain='source'):  # x

        batch_size = x.shape[0]


        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
            domains = 2*(torch.ones(batch_size,dtype=torch.int64))

        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
            domains = torch.ones(batch_size,dtype=torch.int64)
        elif domain == 'test':
            x = self.target_mapping(x)  # (45, 100,9,9)
            domains = torch.zeros(batch_size, dtype=torch.int64)

        x_agg = (torch.flatten(x,2)).mean(dim=2)
        # x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        feature_shellow = self.fc_norm(self.fea_emd(x_agg))
        feature_final = self.feature_encoder(x)  # (45, 64)
        feature=(feature_shellow+feature_final)/2
        # feature = feature_final

        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        # 8th Layer: FC and return unscaled activations
        logits_common = torch.matmul(feature, w_c) + b_c

        domains=domains.cuda()

        domain_label=torch.nn.functional.one_hot(domains,self.domain_num+1)
        domain_label=domain_label.float()


        domain_label=domain_label.cuda()
        #
        c_wts = torch.matmul(domain_label, self.embs)
        # B x K
        # batch_size = uids.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float, device='cuda') * self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts)
        w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        # print(w_d.shape,)
        logits_specialized = torch.einsum("brl,br->bl", w_d, feature) + b_d

        diag_tensor = torch.stack([torch.eye(self.domain_num) for _ in range(self.class_num)], dim=0).cuda()
        cps = torch.stack(
            [torch.matmul(self.sms[:, :, _], torch.transpose(self.sms[:, :, _], 0, 1)) for _ in range(self.class_num)], dim=0)
        #
        orth_loss = torch.mean((cps - diag_tensor) ** 2)
        dilit_loss = F.kl_div(torch.softmax(feature_final,dim=0).log(),torch.softmax(feature_shellow,dim=0),reduction='batchmean')
            # torch.dist(feature_final,feature_shellow)
        # output = self.classifier(logits_common)
        # orth_loss = 0
        if domain == 'target':
            return logits_specialized,feature,orth_loss,dilit_loss
            # return (logits_specialized+feature_shellow)/2, feature, orth_loss, dilit_loss
        elif domain == 'source':
            return logits_common, feature, orth_loss,dilit_loss
            # return (logits_common+feature_shellow)/2, feature, orth_loss, dilit_loss
        elif domain == 'test':
            return feature, feature,orth_loss



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits



# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None


seeds = [1233,1329,1338,1240, 1230, 1223, 1226, 1236, 1337, 1224]


for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    domain_classifier = models.DomainClassifier()
    random_layer = models.RandomLayer([args.feature_dim, args.class_num], 1024)


    feature_encoder.apply(weights_init)
    domain_classifier.apply(weights_init)

    feature_encoder.cuda()
    domain_classifier.cuda()
    random_layer.cuda()  # Random layer

    feature_encoder.train()
    domain_classifier.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(5000):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()

        # source domain few-shot + domain adaptation
        # if episode % 2 == 0:
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_outputs,support_features, ori_loss1,dilit_loss1 = feature_encoder(supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
            query_outputs, query_features, ori_loss2,dilit_loss2 = feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])
            target_outputs,target_features, ori_loss3,dilit_loss3 = feature_encoder(target_data.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_outputs.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_outputs

            # fsl_loss
            logits = euclidean_metric(query_outputs, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            '''domain adaptation'''
            # calculate domain adaptation loss
            features = torch.cat([support_features, query_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_loss = utils.CORAL(features, target_features)
            ori_loss = ori_loss1 + ori_loss2 + ori_loss3
            dilit_loss = dilit_loss1+dilit_loss2+dilit_loss3



            # total_loss = fsl_loss + domain_loss
            loss = f_loss + Lambda1*(domain_loss+ori_loss)+Lambda2*dilit_loss  # 0.01

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_outputs,support_features,ori_loss1,dilit_loss1= feature_encoder(supports.cuda(),  domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_outputs,query_features, ori_loss2,dilit_loss2 = feature_encoder(querys.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])
            source_outputs,source_features, ori_loss3,dilit_loss3= feature_encoder(source_data.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_outputs.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_outputs

            # fsl_loss
            logits = euclidean_metric(query_outputs, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            '''domain adaptation'''
            features = torch.cat([support_features, query_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_loss=utils.CORAL(features,source_features)
            ori_loss=ori_loss1+ori_loss2+ori_loss3
            dilit_loss = dilit_loss1+dilit_loss2+dilit_loss3

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + Lambda1*(domain_loss+ori_loss)+Lambda2*dilit_loss
            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  domain loss: {:6.4f}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                domain_loss.item(),
                                                                                                                f_loss.item(),
                                                                                                                total_hit / total_num,
                                                                                                                loss.item()))

        if (episode + 1) % 500 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            embed_fea = np.empty((0,160), dtype=np.int64)


            train_datas, train_labels = train_loader.__iter__().next()
            train_features, _,_ = feature_encoder(Variable(train_datas).cuda(), domain='test')  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features,_, _ = feature_encoder(Variable(test_datas).cuda(), domain='test')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)
                embed_fea = np.append(embed_fea,test_features.cpu().detach().numpy(),axis=0)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str("checkpoints/DFSL_feature_encoder_" + "IP_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                best_predict_all = predict
                best_emd = embed_fea

                best_G, best_RandPerm, best_Row, best_Column, best_nTrain,best_labels = G, RandPerm, Row, Column, nTrain,labels

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

        # if test_accuracy > best_acc_all:
        #
        #
        #     quit()

# utils.color_SNE(best_emd,best_labels)
AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

with open('Results/out_'+Target_data+'1-5.log', 'a') as f:
    f.write("\n")
    f.write("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start)+"\n")
    f.write("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end)+"\n")
    f.write("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd)+"\n")
    f.write("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd)+"\n")
    f.write("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd)+"\n")
    f.write("accuracy for each class: "+"\n")
    f.write("accuracy for each class: " + "\n")
    f.write("accuracy for each class: " + "\n")
    for i in range(CLASS_NUM):
        f.write("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i])+"\n")

    curret_tiem = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.write("\n")
    f.write(curret_tiem)
    f.write("\n")

f.close()

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]
# class_acc = best_G[4:-4, 4:-4]
np.save('classmap_'+Target_data+'.npy',best_G[4:-4, 4:-4])
np.save('fea_tsne_UP'+Target_data+'.npy',best_emd)
utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/IP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))